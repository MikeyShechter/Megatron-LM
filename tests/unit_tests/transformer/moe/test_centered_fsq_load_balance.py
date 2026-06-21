# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.transformer.moe.moe_utils import direct_load_balancing_loss_func
from megatron.core.transformer.transformer_config import TransformerConfig


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


def test_centered_fsq_and_var_uses_hard_forward_with_ste_variance_gradient():
    centered_loss, centered_grad = _direct_loss_and_grad("centered_fsq")
    with_var_loss, with_var_grad = _direct_loss_and_grad("centered_fsq_and_var")

    assert torch.allclose(with_var_loss, centered_loss)
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
