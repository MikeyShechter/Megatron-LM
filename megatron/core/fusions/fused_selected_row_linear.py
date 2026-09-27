# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


_BLOCK_H = 256


@triton.jit
def _selected_row_linear_forward_kernel(
    input_ptr,
    weight_ptr,
    indices_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    route_idx = tl.program_id(0)
    token_idx = route_idx // TOPK
    expert_idx = tl.load(indices_ptr + route_idx)

    accumulator = 0.0
    for block_start in tl.static_range(0, HIDDEN_SIZE, BLOCK_H):
        hidden_offsets = block_start + tl.arange(0, BLOCK_H)
        hidden_mask = hidden_offsets < HIDDEN_SIZE
        input_values = tl.load(
            input_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            weight_ptr + expert_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(input_values * weight_values, axis=0)

    tl.store(output_ptr + route_idx, accumulator)


@triton.jit
def _selected_row_linear_input_grad_kernel(
    grad_output_ptr,
    weight_ptr,
    indices_ptr,
    grad_input_ptr,
    HIDDEN_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    hidden_block_idx = tl.program_id(1)
    hidden_offsets = hidden_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < HIDDEN_SIZE

    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for topk_idx in tl.static_range(0, TOPK):
        route_idx = token_idx * TOPK + topk_idx
        expert_idx = tl.load(indices_ptr + route_idx)
        grad_output = tl.load(grad_output_ptr + route_idx).to(tl.float32)
        weight_values = tl.load(
            weight_ptr + expert_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += grad_output * weight_values

    tl.store(
        grad_input_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        accumulator,
        mask=hidden_mask,
    )


@triton.jit
def _selected_row_linear_weight_grad_kernel(
    grad_output_ptr,
    input_ptr,
    route_order_ptr,
    expert_offsets_ptr,
    grad_weight_ptr,
    HIDDEN_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    expert_idx = tl.program_id(0)
    hidden_block_idx = tl.program_id(1)
    hidden_offsets = hidden_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < HIDDEN_SIZE

    route_position = tl.load(expert_offsets_ptr + expert_idx)
    route_end = tl.load(expert_offsets_ptr + expert_idx + 1)
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    while route_position < route_end:
        route_idx = tl.load(route_order_ptr + route_position)
        token_idx = route_idx // TOPK
        grad_output = tl.load(grad_output_ptr + route_idx).to(tl.float32)
        input_values = tl.load(
            input_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += grad_output * input_values
        route_position += 1

    tl.store(
        grad_weight_ptr + expert_idx * HIDDEN_SIZE + hidden_offsets,
        accumulator,
        mask=hidden_mask,
    )


class _SelectedRowLinear(torch.autograd.Function):
    """Evaluate only the expert rows selected independently for each token."""

    @staticmethod
    def forward(
        ctx,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        indices: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        num_tokens, hidden_size = input_2d.shape
        topk = indices.shape[1]
        output = torch.empty((num_tokens, topk), device=input_2d.device, dtype=output_dtype)
        _selected_row_linear_forward_kernel[(num_tokens * topk,)](
            input_2d,
            weight,
            indices,
            output,
            HIDDEN_SIZE=hidden_size,
            TOPK=topk,
            BLOCK_H=_BLOCK_H,
            num_warps=4,
        )
        ctx.save_for_backward(input_2d, weight, indices)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_2d, weight, indices = ctx.saved_tensors
        num_tokens, hidden_size = input_2d.shape
        num_experts = weight.shape[0]
        topk = indices.shape[1]
        hidden_grid = triton.cdiv(hidden_size, _BLOCK_H)
        grad_output = grad_output.contiguous()

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.empty_like(input_2d)
            _selected_row_linear_input_grad_kernel[(num_tokens, hidden_grid)](
                grad_output,
                weight,
                indices,
                grad_input,
                HIDDEN_SIZE=hidden_size,
                TOPK=topk,
                BLOCK_H=_BLOCK_H,
                num_warps=4,
            )

        grad_weight = None
        if ctx.needs_input_grad[1]:
            sorted_experts, route_order = torch.sort(indices.reshape(-1), stable=True)
            expert_counts = torch.bincount(sorted_experts, minlength=num_experts)
            expert_offsets = torch.cat(
                (expert_counts.new_zeros(1), expert_counts.cumsum(dim=0))
            )
            grad_weight_fp32 = torch.empty(
                weight.shape, device=weight.device, dtype=torch.float32
            )
            _selected_row_linear_weight_grad_kernel[(num_experts, hidden_grid)](
                grad_output,
                input_2d,
                route_order,
                expert_offsets,
                grad_weight_fp32,
                HIDDEN_SIZE=hidden_size,
                TOPK=topk,
                BLOCK_H=_BLOCK_H,
                num_warps=4,
            )
            grad_weight = grad_weight_fp32.to(dtype=weight.dtype)

        return grad_input, grad_weight, None, None


def _selected_row_linear_reference(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    compute_dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
    input_compute = input_2d.to(dtype=compute_dtype)
    weight_compute = weight.to(dtype=compute_dtype)
    output = torch.stack(
        [
            (input_compute * weight_compute.index_select(0, expert_indices)).sum(dim=-1)
            for expert_indices in indices.unbind(dim=1)
        ],
        dim=1,
    )
    return output.to(dtype=output_dtype)


def selected_row_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute ``input[t] @ weight[indices[t, k]]`` without dense expert logits.

    The CUDA path fuses row selection and dot-product reduction. Other devices and dtypes
    use the same selected-row operation expressed with PyTorch primitives.
    """
    input_2d = input.reshape(-1, input.shape[-1])
    indices = indices.reshape(input_2d.shape[0], -1).contiguous()
    if (
        HAVE_TRITON
        and input_2d.is_cuda
        and output_dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return _SelectedRowLinear.apply(input_2d, weight, indices, output_dtype)
    return _selected_row_linear_reference(input_2d, weight, indices, output_dtype)
