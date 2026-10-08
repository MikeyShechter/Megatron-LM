# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Global quantile estimators (Kimi K3 histogram, EQB exact), quantile-correction margins, and the
post-activation / mass-normalized STE path of the direct load-balancing losses.

All tests run on CPU. `test_global_kth_smallest_matches_gathered_kthvalue_distributed` needs
`torch.distributed.run` with more than one process (gloo on CPU, NCCL on GPU), e.g.
`python -m torch.distributed.run --nproc-per-node 4 -m pytest -k distributed <this file>`.
"""

import os

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import (
    _bf16_from_order_key,
    _bf16_order_key,
    _fixed_boundary_load_surrogate,
    _global_expert_kth_smallest,
    _load_balance_margin,
    _load_balance_ste_load_surrogate,
    _quantile_correction_delta_bias,
    _quantile_correction_load_surrogate,
    _quantile_correction_window,
    compute_routing_scores_for_aux_loss,
    direct_load_balancing_loss_func,
    qb_dual_update,
    qb_global_beta,
    switch_load_balancing_loss_func,
)


def _routing(logits, topk):
    top = torch.topk(logits, k=topk + 1, dim=-1).indices
    routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, top[:, :topk], True)
    return routing_map, top[:, :topk], top[:, topk : topk + 1]


def _imbalanced_logits(num_tokens, num_experts, spread, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, num_experts, generator=generator) + spread * torch.randn(
        num_experts, generator=generator
    )


def _bf16_exact(values):
    return values.to(torch.bfloat16).float()


def test_bf16_order_key_is_monotone_and_invertible():
    values = torch.cat(
        [
            torch.randn(4000) * 10,
            torch.tensor([0.0, -0.0, 1e-30, -1e-30, 3.0e38, -3.0e38, 1.0, -1.0]),
        ]
    )
    rounded = _bf16_exact(values)
    keys = _bf16_order_key(values)
    assert keys.min() >= 0 and keys.max() <= 0xFFFF
    assert torch.equal(_bf16_from_order_key(keys), rounded)
    order = torch.argsort(keys, stable=True)
    assert torch.all(rounded[order][1:] >= rounded[order][:-1])


@pytest.mark.parametrize("with_mask", [False, True])
def test_exact_estimator_matches_kthvalue_of_bf16_values(with_mask):
    num_tokens, num_experts, topk = 600, 16, 2
    values = _imbalanced_logits(num_tokens, num_experts, spread=0.5)
    valid = torch.rand(num_tokens) > 0.2 if with_mask else None
    kth = lambda n: n - n * topk // num_experts
    result = _global_expert_kth_smallest(values, valid, kth, None, "exact")

    rounded = _bf16_exact(values if valid is None else values[valid])
    k = int(kth(torch.tensor(rounded.size(0))))
    expected = torch.kthvalue(rounded, k, dim=0).values
    assert torch.equal(result, expected)


def test_histogram_estimator_is_within_one_bin():
    num_tokens, num_experts, topk, bins = 2000, 32, 2, 1000
    values = _imbalanced_logits(num_tokens, num_experts, spread=0.5)
    kth = lambda n: n - n * topk // num_experts
    result = _global_expert_kth_smallest(values, None, kth, None, "histogram", num_bins=bins)
    expected = torch.kthvalue(values, int(kth(torch.tensor(num_tokens))), dim=0).values
    bin_width = (values.max(dim=0).values - values.min(dim=0).values) / bins
    assert torch.all((result - expected).abs() <= bin_width + 1e-6)


def test_histogram_estimator_respects_given_value_range():
    values = torch.rand(500, 8)  # in (0, 1)
    kth = lambda n: n // 2
    result = _global_expert_kth_smallest(
        values, None, kth, None, "histogram", num_bins=1000, value_range=(-1.0, 2.0)
    )
    expected = torch.kthvalue(values, 250, dim=0).values
    assert torch.all((result - expected).abs() <= 3.0 / 1000 + 1e-6)


def test_qb_global_beta_matches_single_rank_dual_update():
    # One rank holds the whole batch, so the global quantile is the legacy per-rank quantile.
    num_tokens, num_experts, topk = 512, 16, 2
    scores = _bf16_exact(_imbalanced_logits(num_tokens, num_experts, spread=0.5))
    beta = _bf16_exact(0.1 * torch.randn(num_experts))
    indices_legacy, beta_legacy = qb_dual_update(scores, topk, beta)
    indices, margins = qb_dual_update(scores, topk, beta, return_margins=True)
    assert torch.equal(indices, indices_legacy)
    exact = qb_global_beta(margins, topk, num_experts, None, "exact")
    # The exact estimator returns the bf16 rounding of the legacy (fp32) quantile.
    assert torch.equal(exact, _bf16_exact(beta_legacy))
    histogram = qb_global_beta(margins, topk, num_experts, None, "histogram", num_bins=1000)
    bin_width = (margins.max(dim=0).values - margins.min(dim=0).values) / 1000
    assert torch.all((histogram - beta_legacy).abs() <= bin_width + 1e-6)


@pytest.mark.parametrize("spread", [0.1, 0.3])
def test_exact_margin_gives_every_imbalanced_expert_a_window_of_its_load_error(spread):
    num_tokens, num_experts, topk = 4096, 128, 2
    logits = _imbalanced_logits(num_tokens, num_experts, spread=spread)
    routing_map, _, _ = _routing(logits, topk)
    load = routing_map.sum(dim=0)
    fair_share = num_tokens * topk // num_experts
    underloaded = load < fair_share

    windows = {}
    for position in ("topk_plus_one", "exact_margin"):
        margin, valid = _load_balance_margin(logits, routing_map, position)
        delta_bias = _quantile_correction_delta_bias(margin, valid, topk, num_experts, None)
        window = _quantile_correction_window(margin, delta_bias.unsqueeze(0), False)
        windows[position] = (delta_bias, window.sum(dim=0))

    delta_bias, window_count = windows["exact_margin"]
    assert torch.equal(window_count, (load - fair_share).abs())
    assert torch.all(delta_bias[load != fair_share] != 0)
    # Against the runner-up, every token's runner-up sits at margin 0: some underloaded experts
    # get no correction at all.
    delta_bias_runner_up, _ = windows["topk_plus_one"]
    assert bool((delta_bias_runner_up[underloaded] == 0).any())


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_identity_kernel_on_normalized_scores_is_the_aux_loss(score_function):
    num_tokens, num_experts, topk, coeff = 256, 32, 2, 0.7
    logits = _imbalanced_logits(num_tokens, num_experts, spread=0.5).requires_grad_(True)
    routing_map, scores = compute_routing_scores_for_aux_loss(logits, topk, score_function)
    tokens_per_expert = routing_map.sum(dim=0)
    aux = switch_load_balancing_loss_func(
        scores, tokens_per_expert, num_tokens, topk, num_experts, coeff
    )
    (aux_grad,) = torch.autograd.grad(aux, logits)

    _, scores = compute_routing_scores_for_aux_loss(logits, topk, score_function)
    ste = direct_load_balancing_loss_func(
        "centered_fsq",
        logits,
        routing_map,
        tokens_per_expert,
        num_tokens,
        topk,
        num_experts,
        coeff,
        load_balance_ste_type="full",
        load_balance_ste_post_scores=scores,
    )
    (ste_grad,) = torch.autograd.grad(ste, logits)
    torch.testing.assert_close(ste_grad, aux_grad, rtol=1e-4, atol=1e-9)


def _surrogate_grad(kind, logits, scores, routing_map, top, top_plus_one, topk, num_experts,
                    detach, mass, post):
    num_tokens = logits.size(0)
    load = routing_map.sum(dim=0).float()
    common = dict(
        topk_indices=top,
        topk_plus_one_indices=top_plus_one,
        detach_threshold=detach,
        post_scores=scores if post else None,
        mass_normalized=mass,
        mass_tokens=num_tokens,
    )
    if kind == "rect":
        surrogate, _, _ = _load_balance_ste_load_surrogate(
            logits, routing_map, load, "rect", 1.0, 1.0, "exact_margin", **common
        )
    elif kind == "fixed":
        surrogate, _, _ = _fixed_boundary_load_surrogate(
            logits, routing_map, load, 0.05, num_experts, None, "exact_margin", **common
        )
    else:
        surrogate, _, _, _ = _quantile_correction_load_surrogate(
            logits, routing_map, load, topk, num_experts, None, exact_margin=True, **common
        )
    return surrogate


@pytest.mark.parametrize("kind", ["rect", "fixed", "qc"])
def test_mass_normalization_gives_every_expert_mass_T(kind):
    num_tokens, num_experts, topk = 1024, 32, 2
    logits = _imbalanced_logits(num_tokens, num_experts, spread=0.3).requires_grad_(True)
    routing_map, top, top_plus_one = _routing(logits.detach(), topk)
    surrogate = _surrogate_grad(
        kind, logits, None, routing_map, top, top_plus_one, topk, num_experts,
        detach=True, mass=True, post=False,
    )
    (grad,) = torch.autograd.grad(surrogate.sum(), logits)
    mass = grad.sum(dim=0)
    nonempty = mass != 0
    assert bool(nonempty.any())
    torch.testing.assert_close(mass[nonempty], torch.full_like(mass[nonempty], num_tokens))


@pytest.mark.parametrize("kind", ["rect", "fixed", "qc"])
def test_post_detached_gradient_has_the_sign_of_the_load_error(kind):
    num_tokens, num_experts, topk = 1024, 32, 2
    logits = _imbalanced_logits(num_tokens, num_experts, spread=0.3).requires_grad_(True)
    routing_map, top, top_plus_one = _routing(logits.detach(), topk)
    scores = torch.softmax(logits.detach(), dim=-1).requires_grad_(True)
    tokens_per_expert = routing_map.sum(dim=0)
    load_balancing_type = {"rect": "centered_fsq", "fixed": "fixed_number_boundary_ste",
                           "qc": "quantile_correction_ste"}[kind]
    loss = direct_load_balancing_loss_func(
        load_balancing_type,
        logits,
        routing_map,
        tokens_per_expert,
        num_tokens,
        topk,
        num_experts,
        1.0,
        load_balance_ste_width=1.0 if kind == "rect" else 0.0,
        load_balance_ste_type="rect",
        load_balance_ste_rect_poistion="exact_margin",
        load_balance_topk_indices=top,
        load_balance_topk_plus_one_indices=top_plus_one,
        load_balance_ste_boundary_fraction=0.05 if kind == "fixed" else 0.0,
        load_balance_ste_detach_threshold=True,
        quantile_correction_exact_margin=True,
        load_balance_ste_post_scores=scores,
        load_balance_ste_mass_normalized=True,
    )
    logits_grad, scores_grad = torch.autograd.grad(loss, [logits, scores], allow_unused=True)
    assert logits_grad is None or bool((logits_grad == 0).all())
    load_error = tokens_per_expert.float() - num_tokens * topk / num_experts
    pushed = scores_grad != 0
    assert bool(pushed.any())
    expected_sign = torch.sign(load_error).unsqueeze(0).expand_as(scores_grad)
    assert torch.equal(torch.sign(scores_grad[pushed]), expected_sign[pushed])


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="needs torch.distributed.run with more than one process",
)
@pytest.mark.parametrize("method", ["exact", "histogram"])
def test_global_kth_smallest_matches_gathered_kthvalue_distributed(method):
    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend=backend)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    num_tokens, num_experts, topk = 300, 16, 2
    shards = [_imbalanced_logits(num_tokens, num_experts, 0.5, seed=r) for r in range(world_size)]
    local = shards[rank].to(device)
    kth = lambda n: n - n * topk // num_experts
    result = _global_expert_kth_smallest(
        local, None, kth, torch.distributed.group.WORLD, method, num_bins=1000
    ).cpu()

    pooled = torch.cat(shards, dim=0)
    k = int(kth(torch.tensor(pooled.size(0))))
    if method == "exact":
        assert torch.equal(result, torch.kthvalue(_bf16_exact(pooled), k, dim=0).values)
    else:
        expected = torch.kthvalue(pooled, k, dim=0).values
        bin_width = (pooled.max(dim=0).values - pooled.min(dim=0).values) / 1000
        assert torch.all((result - expected).abs() <= bin_width + 1e-6)
