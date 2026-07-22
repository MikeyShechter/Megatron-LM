# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os

import pytest
import torch

from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.transformer.moe.moe_utils import (
    _RectangularIndicatorSTE,
    _TanhSTE,
    _TriangleSTE,
    _load_balance_margin,
    compute_routing_scores_for_aux_loss,
    direct_load_balancing_loss_func,
    topk_routing_with_score_function,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _routing_map_from_topk(logits, topk):
    topk_indices = torch.topk(logits, k=topk, dim=-1).indices
    routing_map = torch.zeros_like(logits, dtype=torch.bool)
    routing_map.scatter_(1, topk_indices, True)
    return routing_map, topk_indices


def _topk_plus_one_indices(logits, topk):
    if topk >= logits.size(-1):
        return None
    topk_plus_one = torch.topk(logits, k=topk + 1, dim=-1).indices
    return topk_plus_one[:, topk : topk + 1]


def _router_activation(logits, score_function):
    if score_function == "sigmoid":
        return torch.sigmoid(logits.float())
    if score_function == "sqrtsoftplus":
        return torch.nn.functional.softplus(logits.float()).sqrt()
    raise ValueError(f"Unsupported test score function: {score_function}")


def _reference_direct_load_balance_from_load(load_frac, num_experts, load_balancing_type):
    expected_frac = load_frac.new_tensor(1.0 / num_experts)
    num_experts_tensor = load_frac.new_tensor(float(num_experts))
    if load_balancing_type == "fsq":
        return num_experts_tensor * torch.square(load_frac).sum(dim=-1)
    if load_balancing_type in ("centered_fsq", "centered_fsq_and_var"):
        return 1.0 + num_experts_tensor * torch.square(load_frac - expected_frac).sum(dim=-1)
    if load_balancing_type == "maxvio":
        return 1.0 + (torch.amax(load_frac, dim=-1) - expected_frac) / expected_frac
    if load_balancing_type == "maxviosq":
        raw_penalty = (torch.amax(load_frac, dim=-1) - expected_frac) / expected_frac
        max_penalty = max(num_experts - 1.0, 1.0)
        return 1.0 + torch.square(raw_penalty) / max_penalty
    if load_balancing_type == "totalvio":
        raw_penalty = torch.abs(load_frac - expected_frac).sum(dim=-1) / expected_frac
        return 1.0 + 0.5 * raw_penalty
    raise ValueError(f"Unsupported direct load balancing type: {load_balancing_type}")


def _reference_centered_fsq_loss(
    logits,
    routing_map,
    tokens_per_expert,
    total_num_tokens,
    topk,
    num_experts,
    moe_aux_loss_coeff,
    load_balance_ste_width,
    reduce_group=None,
    load_balancing_type="centered_fsq",
    load_balance_ste_type="rect",
    load_balance_tanh_ste_slope=1.0,
    load_balance_ste_rect_poistion="topk",
):
    total_num_tokens_tensor = tokens_per_expert.new_tensor(float(total_num_tokens))
    denom = torch.clamp(total_num_tokens_tensor * float(topk), min=1.0)
    hard_load_frac = tokens_per_expert.float() / denom
    load_frac = hard_load_frac

    if load_balance_ste_type == "tanh" or load_balance_ste_width > 0.0:
        ste_rect_poistion = (
            load_balance_ste_rect_poistion
            if load_balance_ste_type in ("rect", "triangle")
            else "topk"
        )
        margin, valid_tokens = _load_balance_margin(logits, routing_map, ste_rect_poistion)
        if load_balance_ste_type == "tanh":
            soft_mask = _TanhSTE.apply(margin, load_balance_tanh_ste_slope)
        elif load_balance_ste_type == "triangle":
            soft_mask = _TriangleSTE.apply(margin, load_balance_ste_width)
        else:
            soft_mask = _RectangularIndicatorSTE.apply(margin, load_balance_ste_width)
        soft_mask = soft_mask * valid_tokens.unsqueeze(-1).to(dtype=soft_mask.dtype)
        ste_tokens_per_expert = soft_mask.sum(dim=0)
        if reduce_group is not None:
            ste_tokens_per_expert = reduce_from_tensor_model_parallel_region(
                ste_tokens_per_expert, reduce_group
            )
        ste_load_frac = ste_tokens_per_expert.float() / denom
        load_frac = hard_load_frac + ste_load_frac - ste_load_frac.detach()

    loss = _reference_direct_load_balance_from_load(load_frac, num_experts, load_balancing_type)
    if load_balancing_type == "centered_fsq_and_var" and load_balance_ste_width > 0.0:
        margin, valid_tokens = _load_balance_margin(
            logits, routing_map, load_balance_ste_rect_poistion
        )
        half_width = load_balance_ste_width * 0.5
        expected_indicator = torch.clamp(
            (margin + half_width) / load_balance_ste_width, 0.0, 1.0
        )
        expected_indicator = expected_indicator * valid_tokens.unsqueeze(-1).to(
            dtype=expected_indicator.dtype
        )
        variance_sum = (expected_indicator * (1.0 - expected_indicator)).sum()
        if reduce_group is not None:
            variance_sum = reduce_from_tensor_model_parallel_region(variance_sum, reduce_group)
        loss = loss + load_frac.new_tensor(float(num_experts)) * variance_sum / torch.square(denom)

    return loss * moe_aux_loss_coeff


def _loss_and_grad(loss_fn, logits):
    logits = logits.clone().detach().requires_grad_(True)
    loss = loss_fn(logits)
    grad = torch.autograd.grad(loss, logits)[0]
    return loss.detach(), grad.detach()


def _assert_close_to_reference(
    optimized_value,
    reference_value,
    optimized_grad,
    reference_grad,
    atol=1e-8,
    rtol=1e-6,
):
    torch.testing.assert_close(optimized_value, reference_value, atol=atol, rtol=rtol)
    torch.testing.assert_close(optimized_grad, reference_grad, atol=atol, rtol=rtol)


def _centered_fsq_loss_and_grad(load_balance_ste_width):
    logits = torch.tensor([[2.7, 2.5, 2.4, 0.1]], dtype=torch.float32, requires_grad=True)
    routing_map = torch.tensor([[True, True, False, False]])
    tokens_per_expert = routing_map.sum(dim=0)
    loss = direct_load_balancing_loss_func(
        load_balancing_type="centered_fsq",
        logits=logits,
        routing_map=routing_map,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=1,
        topk=2,
        num_experts=4,
        moe_aux_loss_coeff=1.0,
        load_balance_ste_width=load_balance_ste_width,
    )
    if loss.requires_grad:
        grad = torch.autograd.grad(loss, logits)[0][0]
    else:
        grad = torch.zeros_like(logits[0])
    return loss.detach(), grad.detach()


def _direct_loss_and_grad(load_balancing_type):
    logits = torch.tensor(
        [[2.7, 2.5, 2.4, 0.1], [2.7, 2.5, 2.4, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    routing_map = torch.tensor([[True, True, False, False], [True, True, False, False]])
    tokens_per_expert = routing_map.sum(dim=0)
    loss = direct_load_balancing_loss_func(
        load_balancing_type=load_balancing_type,
        logits=logits,
        routing_map=routing_map,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=2,
        topk=2,
        num_experts=4,
        moe_aux_loss_coeff=1.0,
        load_balance_ste_width=0.5,
    )
    grad = torch.autograd.grad(loss, logits)[0]
    return loss.detach(), grad.detach()


def test_centered_fsq_forward_value_matches_hard_load_with_and_without_ste():
    loss_no_ste, grad_no_ste = _centered_fsq_loss_and_grad(load_balance_ste_width=0.0)
    loss_ste, _ = _centered_fsq_loss_and_grad(load_balance_ste_width=0.5)

    assert torch.allclose(loss_no_ste, torch.tensor(2.0))
    assert torch.allclose(loss_ste, loss_no_ste)
    assert torch.allclose(grad_no_ste, torch.zeros_like(grad_no_ste))


def test_centered_fsq_ste_updates_overloaded_and_near_boundary_underloaded_experts():
    _, grad = _centered_fsq_loss_and_grad(load_balance_ste_width=0.5)

    assert grad[0] > 1e-4
    assert grad[1].abs() < 1e-6
    assert grad[2] < -1e-4
    assert grad[3].abs() < 1e-7


def test_triangle_ste_uses_piecewise_second_order_gradient():
    bandwidth = 2.0
    margin = torch.tensor(
        [-2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5],
        dtype=torch.float32,
        requires_grad=True,
    )

    soft_mask = _TriangleSTE.apply(margin, bandwidth)
    grad = torch.autograd.grad(soft_mask.sum(), margin)[0]

    expected_forward = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    expected_grad = torch.tensor([0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0])
    torch.testing.assert_close(soft_mask, expected_forward)
    torch.testing.assert_close(grad, expected_grad)


def test_centered_fsq_forward_value_for_arbitrary_load():
    logits = torch.zeros((10, 4), dtype=torch.float32)
    routing_map = torch.zeros((10, 4), dtype=torch.bool)
    tokens_per_expert = torch.tensor([4, 3, 1, 2])

    loss = direct_load_balancing_loss_func(
        load_balancing_type="centered_fsq",
        logits=logits,
        routing_map=routing_map,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=5,
        topk=2,
        num_experts=4,
        moe_aux_loss_coeff=1.0,
    )

    assert torch.allclose(loss, torch.tensor(1.2))


def test_noisy_centered_fsq_uses_expected_load_plus_variance():
    loss, grad = _direct_loss_and_grad("noisy_centered_fsq")

    assert torch.allclose(loss, torch.tensor(1.725))
    assert grad.abs().sum() > 0


def test_noisy_centered_fsq_gradient_differs_from_hard_ste_plus_variance():
    _, noisy_grad = _direct_loss_and_grad("noisy_centered_fsq")
    _, hard_ste_grad = _direct_loss_and_grad("centered_fsq_and_var")

    assert not torch.allclose(noisy_grad, hard_ste_grad)


def test_centered_fsq_and_var_adds_variance_forward_and_gradient():
    centered_loss, centered_grad = _direct_loss_and_grad("centered_fsq")
    with_var_loss, with_var_grad = _direct_loss_and_grad("centered_fsq_and_var")

    assert with_var_loss.item() > centered_loss.item()
    assert not torch.allclose(with_var_grad, centered_grad)


def test_centered_fsq_transformer_config_accepts_ste_width():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        num_moe_experts=4,
        moe_router_load_balancing_type="centered_fsq",
        moe_aux_loss_coeff=0.01,
        moe_load_balance_ste_width=0.5,
    )

    assert config.moe_router_load_balancing_type == "centered_fsq"
    assert config.moe_load_balance_ste_width == 0.5


def test_noisy_centered_fsq_transformer_config_accepts_ste_width():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        num_moe_experts=4,
        moe_router_load_balancing_type="noisy_centered_fsq",
        moe_aux_loss_coeff=0.01,
        moe_load_balance_ste_width=0.5,
    )

    assert config.moe_router_load_balancing_type == "noisy_centered_fsq"
    assert config.moe_load_balance_ste_width == 0.5


def test_centered_fsq_optimized_path_matches_reference_global_counts():
    torch.manual_seed(1234)
    num_tokens = 37
    other_rank_tokens = 29
    num_experts = 16
    topk = 4
    load_balance_ste_width = 0.75
    moe_aux_loss_coeff = 0.37

    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    routing_map, _ = _routing_map_from_topk(logits, topk)

    other_logits = torch.randn(other_rank_tokens, num_experts, dtype=torch.float32)
    other_routing_map, _ = _routing_map_from_topk(other_logits, topk)
    global_tokens_per_expert = routing_map.sum(dim=0) + other_routing_map.sum(dim=0)
    total_num_tokens = num_tokens + other_rank_tokens

    def reference_loss(local_logits):
        return _reference_centered_fsq_loss(
            local_logits,
            routing_map,
            global_tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            moe_aux_loss_coeff,
            load_balance_ste_width,
        )

    def optimized_loss(local_logits):
        return direct_load_balancing_loss_func(
            load_balancing_type="centered_fsq",
            logits=local_logits,
            routing_map=routing_map,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            load_balance_ste_width=load_balance_ste_width,
        )

    reference_value, reference_grad = _loss_and_grad(reference_loss, logits)
    optimized_value, optimized_grad = _loss_and_grad(optimized_loss, logits)

    _assert_close_to_reference(optimized_value, reference_value, optimized_grad, reference_grad)


def test_centered_fsq_topk_threshold_reuse_closely_matches_reference_global_counts():
    torch.manual_seed(1234)
    num_tokens = 37
    other_rank_tokens = 29
    num_experts = 16
    topk = 4
    load_balance_ste_width = 0.75
    moe_aux_loss_coeff = 0.37

    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    routing_map, topk_indices = _routing_map_from_topk(logits, topk)
    topk_plus_one_indices = _topk_plus_one_indices(logits, topk)

    other_logits = torch.randn(other_rank_tokens, num_experts, dtype=torch.float32)
    other_routing_map, _ = _routing_map_from_topk(other_logits, topk)
    global_tokens_per_expert = routing_map.sum(dim=0) + other_routing_map.sum(dim=0)
    total_num_tokens = num_tokens + other_rank_tokens

    def reference_loss(local_logits):
        return _reference_centered_fsq_loss(
            local_logits,
            routing_map,
            global_tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            moe_aux_loss_coeff,
            load_balance_ste_width,
        )

    def optimized_loss(local_logits):
        return direct_load_balancing_loss_func(
            load_balancing_type="centered_fsq",
            logits=local_logits,
            routing_map=routing_map,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            load_balance_ste_width=load_balance_ste_width,
            load_balance_topk_indices=topk_indices,
            load_balance_topk_plus_one_indices=topk_plus_one_indices,
        )

    reference_value, reference_grad = _loss_and_grad(reference_loss, logits)
    optimized_value, optimized_grad = _loss_and_grad(optimized_loss, logits)

    _assert_close_to_reference(optimized_value, reference_value, optimized_grad, reference_grad)


@pytest.mark.parametrize(
    "load_balancing_type",
    ("centered_fsq", "centered_fsq_and_var"),
)
@pytest.mark.parametrize(
    "load_balance_ste_type,load_balance_ste_rect_poistion,load_balance_tanh_ste_slope",
    (
        ("rect", "topk", 1.0),
        ("rect", "topk_plus_one", 1.0),
        ("rect", "midpoint", 1.0),
        ("triangle", "topk", 1.0),
        ("triangle", "topk_plus_one", 1.0),
        ("triangle", "midpoint", 1.0),
        ("tanh", "topk", 1.7),
    ),
)
def test_centered_fsq_surrogate_matches_reference_for_ste_variants(
    load_balancing_type,
    load_balance_ste_type,
    load_balance_ste_rect_poistion,
    load_balance_tanh_ste_slope,
):
    torch.manual_seed(1234)
    num_tokens = 37
    other_rank_tokens = 29
    num_experts = 16
    topk = 4
    load_balance_ste_width = 0.75
    moe_aux_loss_coeff = 0.37

    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    routing_map, _ = _routing_map_from_topk(logits, topk)

    other_logits = torch.randn(other_rank_tokens, num_experts, dtype=torch.float32)
    other_routing_map, _ = _routing_map_from_topk(other_logits, topk)
    global_tokens_per_expert = routing_map.sum(dim=0) + other_routing_map.sum(dim=0)
    total_num_tokens = num_tokens + other_rank_tokens

    def reference_loss(local_logits):
        return _reference_centered_fsq_loss(
            local_logits,
            routing_map,
            global_tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            moe_aux_loss_coeff,
            load_balance_ste_width,
            load_balancing_type=load_balancing_type,
            load_balance_ste_type=load_balance_ste_type,
            load_balance_tanh_ste_slope=load_balance_tanh_ste_slope,
            load_balance_ste_rect_poistion=load_balance_ste_rect_poistion,
        )

    def optimized_loss(local_logits):
        return direct_load_balancing_loss_func(
            load_balancing_type=load_balancing_type,
            logits=local_logits,
            routing_map=routing_map,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            load_balance_ste_width=load_balance_ste_width,
            load_balance_ste_type=load_balance_ste_type,
            load_balance_tanh_ste_slope=load_balance_tanh_ste_slope,
            load_balance_ste_rect_poistion=load_balance_ste_rect_poistion,
        )

    reference_value, reference_grad = _loss_and_grad(reference_loss, logits)
    optimized_value, optimized_grad = _loss_and_grad(optimized_loss, logits)

    _assert_close_to_reference(optimized_value, reference_value, optimized_grad, reference_grad)


@pytest.mark.parametrize(
    "load_balancing_type",
    ("centered_fsq", "centered_fsq_and_var"),
)
@pytest.mark.parametrize(
    "load_balance_ste_type,load_balance_ste_rect_poistion,load_balance_tanh_ste_slope",
    (
        ("rect", "topk", 1.0),
        ("rect", "topk_plus_one", 1.0),
        ("rect", "midpoint", 1.0),
        ("triangle", "topk", 1.0),
        ("triangle", "topk_plus_one", 1.0),
        ("triangle", "midpoint", 1.0),
        ("tanh", "topk", 1.7),
    ),
)
def test_centered_fsq_topk_reuse_matches_reference_for_ste_variants(
    load_balancing_type,
    load_balance_ste_type,
    load_balance_ste_rect_poistion,
    load_balance_tanh_ste_slope,
):
    torch.manual_seed(1234)
    num_tokens = 37
    other_rank_tokens = 29
    num_experts = 16
    topk = 4
    load_balance_ste_width = 0.75
    moe_aux_loss_coeff = 0.37

    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    routing_map, topk_indices = _routing_map_from_topk(logits, topk)
    topk_plus_one_indices = _topk_plus_one_indices(logits, topk)

    other_logits = torch.randn(other_rank_tokens, num_experts, dtype=torch.float32)
    other_routing_map, _ = _routing_map_from_topk(other_logits, topk)
    global_tokens_per_expert = routing_map.sum(dim=0) + other_routing_map.sum(dim=0)
    total_num_tokens = num_tokens + other_rank_tokens

    def reference_loss(local_logits):
        return _reference_centered_fsq_loss(
            local_logits,
            routing_map,
            global_tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            moe_aux_loss_coeff,
            load_balance_ste_width,
            load_balancing_type=load_balancing_type,
            load_balance_ste_type=load_balance_ste_type,
            load_balance_tanh_ste_slope=load_balance_tanh_ste_slope,
            load_balance_ste_rect_poistion=load_balance_ste_rect_poistion,
        )

    def optimized_loss(local_logits):
        return direct_load_balancing_loss_func(
            load_balancing_type=load_balancing_type,
            logits=local_logits,
            routing_map=routing_map,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            load_balance_ste_width=load_balance_ste_width,
            load_balance_ste_type=load_balance_ste_type,
            load_balance_tanh_ste_slope=load_balance_tanh_ste_slope,
            load_balance_ste_rect_poistion=load_balance_ste_rect_poistion,
            load_balance_topk_indices=topk_indices,
            load_balance_topk_plus_one_indices=topk_plus_one_indices,
        )

    reference_value, reference_grad = _loss_and_grad(reference_loss, logits)
    optimized_value, optimized_grad = _loss_and_grad(optimized_loss, logits)

    _assert_close_to_reference(optimized_value, reference_value, optimized_grad, reference_grad)


def test_topk_routing_returns_topk_plus_one_indices():
    logits = torch.tensor(
        [
            [9.0, 7.0, 6.5, 6.4, 1.0],
            [0.1, 0.4, 0.3, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_topk_plus_one = torch.topk(logits, k=3, dim=-1).indices[:, 2:3]

    routing_probs, routing_map, topk_indices, topk_plus_one_indices = (
        topk_routing_with_score_function(
            logits,
            topk=2,
            return_top_indices=True,
            return_topk_plus_one_indices=True,
        )
    )

    assert routing_probs.shape == logits.shape
    assert routing_map.sum(dim=-1).tolist() == [2, 2]
    assert torch.equal(topk_indices, torch.topk(logits, k=2, dim=-1).indices)
    assert torch.equal(topk_plus_one_indices, expected_topk_plus_one)


def test_topk_routing_returns_sorted_topk_plus_one_indices_under_no_grad():
    logits = torch.tensor(
        [
            [9.0, 7.0, 6.5, 6.4, 1.0],
            [0.1, 0.4, 0.3, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_topk_plus_one = torch.topk(logits, k=3, dim=-1).indices[:, 2:3]

    with torch.no_grad():
        _, _, _, topk_plus_one_indices = topk_routing_with_score_function(
            logits,
            topk=2,
            return_top_indices=True,
            return_topk_plus_one_indices=True,
        )

    assert torch.equal(topk_plus_one_indices, expected_topk_plus_one)


@pytest.mark.parametrize("score_function", ["sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("topk", [1, 2])
def test_sigmoid_like_topk_routing_weights_are_normalized(score_function, topk):
    logits = torch.tensor(
        [
            [2.0, -1.0, 0.5, -2.0],
            [-0.7, 1.4, 0.3, 0.0],
        ],
        dtype=torch.float32,
    )
    activated_scores = _router_activation(logits, score_function)
    top_indices = torch.topk(activated_scores, k=topk, dim=-1).indices
    top_scores = torch.gather(activated_scores, dim=-1, index=top_indices)
    top_probs = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
    expected_probs = torch.zeros_like(logits).scatter(1, top_indices, top_probs)

    routing_probs, routing_map = topk_routing_with_score_function(
        logits,
        topk=topk,
        score_function=score_function,
    )

    torch.testing.assert_close(routing_probs, expected_probs)
    torch.testing.assert_close(routing_probs.sum(dim=-1), torch.ones(logits.size(0)))
    assert torch.equal(routing_map, expected_probs.bool())


@pytest.mark.parametrize("score_function", ["sigmoid", "sqrtsoftplus"])
def test_aux_routing_scores_use_normalized_sigmoid_like_activation_for_pe(score_function):
    logits = torch.tensor(
        [
            [2.0, -1.0, 0.5, -2.0],
            [-0.7, 1.4, 0.3, 0.0],
        ],
        dtype=torch.float32,
    )
    activated_scores = _router_activation(logits, score_function)
    expected_scores = activated_scores / (activated_scores.sum(dim=-1, keepdim=True) + 1e-20)
    expected_top_indices = torch.topk(expected_scores, k=2, dim=-1).indices
    expected_routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(
        1, expected_top_indices, True
    )
    expected_pe = expected_scores.sum(dim=0)

    routing_map, scores = compute_routing_scores_for_aux_loss(
        logits,
        topk=2,
        score_function=score_function,
    )

    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(scores.sum(dim=0), expected_pe)
    assert torch.equal(routing_map, expected_routing_map)


def test_aux_routing_scores_return_topk_plus_one_indices():
    logits = torch.tensor(
        [
            [9.0, 7.0, 6.5, 6.4, 1.0],
            [0.1, 0.4, 0.3, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_topk_plus_one = torch.topk(torch.softmax(logits, dim=-1), k=3, dim=-1).indices[
        :, 2:3
    ]

    routing_map, scores, topk_indices, topk_plus_one_indices = (
        compute_routing_scores_for_aux_loss(
            logits,
            topk=2,
            score_function="softmax",
            return_top_indices=True,
            return_topk_plus_one_indices=True,
        )
    )

    assert scores.shape == logits.shape
    assert routing_map.sum(dim=-1).tolist() == [2, 2]
    assert torch.equal(topk_indices, torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1).indices)
    assert torch.equal(topk_plus_one_indices, expected_topk_plus_one)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for timing loop")
def test_centered_fsq_optimized_path_timing_loop():
    device = torch.device("cuda")
    torch.manual_seed(1234)
    num_tokens = 8192
    other_rank_tokens = 8192
    num_experts = 64
    topk = 8
    load_balance_ste_width = 0.75
    moe_aux_loss_coeff = 0.01
    warmup_iters = 5
    timed_iters = 20

    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    routing_map, topk_indices = _routing_map_from_topk(logits.detach(), topk)
    topk_plus_one_indices = _topk_plus_one_indices(logits.detach(), topk)
    other_logits = torch.randn(
        other_rank_tokens, num_experts, device=device, dtype=torch.float32
    )
    other_routing_map, _ = _routing_map_from_topk(other_logits, topk)
    global_tokens_per_expert = routing_map.sum(dim=0) + other_routing_map.sum(dim=0)
    total_num_tokens = num_tokens + other_rank_tokens

    def reference_loss(local_logits):
        return _reference_centered_fsq_loss(
            local_logits,
            routing_map,
            global_tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            moe_aux_loss_coeff,
            load_balance_ste_width,
        )

    def optimized_loss(local_logits):
        return direct_load_balancing_loss_func(
            load_balancing_type="centered_fsq",
            logits=local_logits,
            routing_map=routing_map,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            load_balance_ste_width=load_balance_ste_width,
            load_balance_topk_indices=topk_indices,
            load_balance_topk_plus_one_indices=topk_plus_one_indices,
        )

    reference_value, reference_grad = _loss_and_grad(reference_loss, logits)
    optimized_value, optimized_grad = _loss_and_grad(optimized_loss, logits)
    _assert_close_to_reference(optimized_value, reference_value, optimized_grad, reference_grad)

    def time_loss(loss_fn):
        local_logits = logits.detach().clone().requires_grad_(True)
        for _ in range(warmup_iters):
            local_logits.grad = None
            loss_fn(local_logits).backward()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timed_iters):
            local_logits.grad = None
            loss_fn(local_logits).backward()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / timed_iters

    reference_ms = time_loss(reference_loss)
    optimized_ms = time_loss(optimized_loss)
    speedup = reference_ms / optimized_ms
    print(
        "centered_fsq timing: "
        f"reference={reference_ms:.4f} ms, "
        f"optimized={optimized_ms:.4f} ms, "
        f"speedup={speedup:.3f}x"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for timing loop")
def test_centered_fsq_optimized_path_distributed_timing_loop():
    if os.environ.get("RUN_CENTERED_FSQ_DISTRIBUTED_TIMING") != "1":
        pytest.skip("set RUN_CENTERED_FSQ_DISTRIBUTED_TIMING=1 to run distributed timing")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        pytest.skip("distributed timing requires torch.distributed.run with WORLD_SIZE > 1")

    if not torch.distributed.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl")

    rank = torch.distributed.get_rank()
    group = torch.distributed.group.WORLD
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1234 + rank)

    num_tokens = 8192
    num_experts = 64
    topk = 8
    load_balance_ste_width = 0.75
    moe_aux_loss_coeff = 0.01
    warmup_iters = 5
    timed_iters = 20

    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    routing_map, topk_indices = _routing_map_from_topk(logits.detach(), topk)
    topk_plus_one_indices = _topk_plus_one_indices(logits.detach(), topk)
    global_tokens_per_expert = routing_map.sum(dim=0)
    torch.distributed.all_reduce(global_tokens_per_expert, group=group)
    total_num_tokens = num_tokens * world_size

    def reference_loss(local_logits):
        return _reference_centered_fsq_loss(
            local_logits,
            routing_map,
            global_tokens_per_expert,
            total_num_tokens,
            topk,
            num_experts,
            moe_aux_loss_coeff,
            load_balance_ste_width,
            reduce_group=group,
        )

    def optimized_loss(local_logits):
        return direct_load_balancing_loss_func(
            load_balancing_type="centered_fsq",
            logits=local_logits,
            routing_map=routing_map,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            load_balance_ste_width=load_balance_ste_width,
            reduce_group=group,
            load_balance_topk_indices=topk_indices,
            load_balance_topk_plus_one_indices=topk_plus_one_indices,
        )

    reference_value, reference_grad = _loss_and_grad(reference_loss, logits)
    optimized_value, optimized_grad = _loss_and_grad(optimized_loss, logits)
    _assert_close_to_reference(optimized_value, reference_value, optimized_grad, reference_grad)

    def time_loss(loss_fn):
        local_logits = logits.detach().clone().requires_grad_(True)
        for _ in range(warmup_iters):
            local_logits.grad = None
            loss_fn(local_logits).backward()
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timed_iters):
            local_logits.grad = None
            loss_fn(local_logits).backward()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = torch.tensor(start.elapsed_time(end) / timed_iters, device=device)
        torch.distributed.all_reduce(elapsed_ms, op=torch.distributed.ReduceOp.MAX, group=group)
        return elapsed_ms.item()

    reference_ms = time_loss(reference_loss)
    optimized_ms = time_loss(optimized_loss)
    speedup = reference_ms / optimized_ms
    if rank == 0:
        print(
            "centered_fsq distributed timing: "
            f"reference={reference_ms:.4f} ms, "
            f"optimized={optimized_ms:.4f} ms, "
            f"speedup={speedup:.3f}x, "
            f"world_size={world_size}"
        )
