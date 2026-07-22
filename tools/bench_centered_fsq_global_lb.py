# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
import os

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.transformer.moe.moe_utils import (
    _RectangularIndicatorSTE,
    _TanhSTE,
    _TriangleSTE,
    _load_balance_margin,
    direct_load_balancing_loss_func,
)


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


def _reference_centered_fsq_loss(
    logits,
    routing_map,
    tokens_per_expert,
    total_num_tokens,
    topk,
    num_experts,
    moe_aux_loss_coeff,
    load_balance_ste_width,
    reduce_group,
    load_balancing_type,
    load_balance_ste_type,
    load_balance_tanh_ste_slope,
    load_balance_ste_rect_poistion,
):
    total_num_tokens_tensor = tokens_per_expert.new_tensor(float(total_num_tokens))
    denom = torch.clamp(total_num_tokens_tensor * float(topk), min=1.0)
    hard_load_frac = tokens_per_expert.float() / denom

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
    ste_tokens_per_expert = reduce_from_tensor_model_parallel_region(
        ste_tokens_per_expert, reduce_group
    )
    ste_load_frac = ste_tokens_per_expert.float() / denom
    load_frac = hard_load_frac + ste_load_frac - ste_load_frac.detach()

    expected_frac = load_frac.new_tensor(1.0 / num_experts)
    if load_balancing_type in ("centered_fsq", "centered_fsq_and_var"):
        loss = 1.0 + load_frac.new_tensor(float(num_experts)) * torch.square(
            load_frac - expected_frac
        ).sum(dim=-1)
    elif load_balancing_type == "fsq":
        loss = load_frac.new_tensor(float(num_experts)) * torch.square(load_frac).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported benchmark load balancing type: {load_balancing_type}")

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
        variance_sum = reduce_from_tensor_model_parallel_region(variance_sum, reduce_group)
        loss = loss + load_frac.new_tensor(float(num_experts)) * variance_sum / torch.square(denom)

    return loss * moe_aux_loss_coeff


def _loss_and_grad(loss_fn, logits):
    logits = logits.clone().detach().requires_grad_(True)
    loss = loss_fn(logits)
    grad = torch.autograd.grad(loss, logits)[0]
    return loss.detach(), grad.detach()


def _time_loss(loss_fn, logits, warmup_iters, timed_iters, group):
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

    elapsed_ms = torch.tensor(start.elapsed_time(end) / timed_iters, device=logits.device)
    torch.distributed.all_reduce(elapsed_ms, op=torch.distributed.ReduceOp.MAX, group=group)
    return elapsed_ms.item()


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--width", type=float, default=0.75)
    parser.add_argument("--coeff", type=float, default=0.01)
    parser.add_argument(
        "--load-balancing-type",
        choices=("centered_fsq", "centered_fsq_and_var", "fsq"),
        default="centered_fsq",
    )
    parser.add_argument("--ste-type", choices=("rect", "tanh", "triangle"), default="rect")
    parser.add_argument(
        "--rect-position",
        choices=("topk", "topk_plus_one", "midpoint"),
        default="topk",
    )
    parser.add_argument("--tanh-slope", type=float, default=1.0)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--timed-iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    if "RANK" not in os.environ:
        raise RuntimeError("Run with torch.distributed.run")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.group.WORLD
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())

    torch.manual_seed(1234 + rank)
    logits = torch.randn(args.num_tokens, args.num_experts, device=device, dtype=torch.float32)
    routing_map, topk_indices = _routing_map_from_topk(logits.detach(), args.topk)
    topk_plus_one_indices = _topk_plus_one_indices(logits.detach(), args.topk)
    global_tokens_per_expert = routing_map.sum(dim=0)
    torch.distributed.all_reduce(global_tokens_per_expert, group=group)
    total_num_tokens = args.num_tokens * world_size

    def reference_loss(local_logits):
        return _reference_centered_fsq_loss(
            local_logits,
            routing_map,
            global_tokens_per_expert,
            total_num_tokens,
            args.topk,
            args.num_experts,
            args.coeff,
            args.width,
            group,
            args.load_balancing_type,
            args.ste_type,
            args.tanh_slope,
            args.rect_position,
        )

    def optimized_loss(local_logits):
        return direct_load_balancing_loss_func(
            load_balancing_type=args.load_balancing_type,
            logits=local_logits,
            routing_map=routing_map,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=args.topk,
            num_experts=args.num_experts,
            moe_aux_loss_coeff=args.coeff,
            load_balance_ste_width=args.width,
            load_balance_ste_type=args.ste_type,
            load_balance_tanh_ste_slope=args.tanh_slope,
            load_balance_ste_rect_poistion=args.rect_position,
            reduce_group=group,
            load_balance_topk_indices=topk_indices,
            load_balance_topk_plus_one_indices=topk_plus_one_indices,
        )

    reference_value, reference_grad = _loss_and_grad(reference_loss, logits)
    optimized_value, optimized_grad = _loss_and_grad(optimized_loss, logits)
    value_close = torch.allclose(
        optimized_value, reference_value, atol=args.atol, rtol=args.rtol
    )
    grad_close = torch.allclose(optimized_grad, reference_grad, atol=args.atol, rtol=args.rtol)
    max_grad_abs_diff = (optimized_grad - reference_grad).abs().max()
    value_close_all = torch.tensor(int(value_close), device=device)
    grad_close_all = torch.tensor(int(grad_close), device=device)
    torch.distributed.all_reduce(value_close_all, op=torch.distributed.ReduceOp.MIN, group=group)
    torch.distributed.all_reduce(grad_close_all, op=torch.distributed.ReduceOp.MIN, group=group)
    torch.distributed.all_reduce(max_grad_abs_diff, op=torch.distributed.ReduceOp.MAX, group=group)

    if value_close_all.item() != 1 or grad_close_all.item() != 1:
        torch.distributed.barrier(group=group)
        print(
            "exactness failed: "
            f"rank={rank}, value_close={value_close}, grad_close={grad_close}, "
            f"local_max_grad_abs_diff={(optimized_grad - reference_grad).abs().max().item()}, "
            f"global_max_grad_abs_diff={max_grad_abs_diff.item()}",
            flush=True,
        )
        torch.distributed.barrier(group=group)
        raise SystemExit(1)

    reference_ms = _time_loss(reference_loss, logits, args.warmup_iters, args.timed_iters, group)
    optimized_ms = _time_loss(optimized_loss, logits, args.warmup_iters, args.timed_iters, group)
    speedup = reference_ms / optimized_ms

    if rank == 0:
        print(
            "centered_fsq global-LB timing: "
            f"reference={reference_ms:.4f} ms, "
            f"optimized={optimized_ms:.4f} ms, "
            f"speedup={speedup:.3f}x, "
            f"value_close={value_close}, "
            f"grad_close={grad_close}, "
            f"max_grad_abs_diff={max_grad_abs_diff.item()}, "
            f"load_balancing_type={args.load_balancing_type}, "
            f"ste_type={args.ste_type}, "
            f"rect_position={args.rect_position}, "
            f"world_size={world_size}"
        )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
