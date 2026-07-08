# Centered-FSQ Load-Balance Optimization Log

## Scope

This change optimizes the non-fused direct load-balancing STE path, with focus on
`centered_fsq` and `centered_fsq_and_var`.
It keeps global load balancing: hard routed-load fractions still use the globally reduced
`tokens_per_expert` and global token count.

## Changes

- Replaced the dense STE soft-mask load path with a `_LoadBalanceLoadSTE`
  surrogate for both rect and tanh STE. It preserves the old
  `hard + ste_load_frac - ste_load_frac.detach()`
  forward algebra using the already available routed-load counts, while the backward
  pass returns the same STE gradient through the margin.
- Removed the STE load all-reduce from the direct-LB STE path. The global hard-load
  counts are still reduced; only the cancelled STE forward value is no longer reduced.
- Reused routing `topk_indices` to build the top-k STE threshold when they are
  available. This avoids the dense masked-fill/min threshold construction in the
  hot non-fused path.
- Computes the top-k-plus-one index in the same non-fused top-k operation when
  `topk_plus_one` or `midpoint` needs the best unselected expert. The loss then
  gathers that threshold from the exact margin tensor. Dense unselected max remains
  as a fallback when ordinary k+1 is not equivalent, such as group-limited routing
  or routing replay.
- Optimized all rect positions (`topk`, `topk_plus_one`, `midpoint`) and tanh.
- Optimized `centered_fsq_and_var` by reusing the already computed margin when the
  load term and variance term use the same margin position. The variance term still
  needs its real dense margin math and scalar all-reduce because it contributes to
  the actual forward value, not just a cancelled STE forward value.
- The resulting gradients differ from the previous implementation only by
  floating-point reduction/order noise in the measured distributed cases.
- Skipped STE rectangle diagnostics during training forwards. Those diagnostics are
  only emitted by validation metric prefixes.

## Exactness Tests

Added tests in `tests/unit_tests/transformer/moe/test_centered_fsq_load_balance.py`:

- `test_centered_fsq_optimized_path_matches_reference_global_counts`
  compares the optimized implementation against a reference copy of the previous
  dense STE implementation.
- The parity test uses simulated global-LB counts from local plus other-rank routing.
- It checks close equality for the loss and gradients with `atol=1e-8`, `rtol=1e-6`.
- `test_centered_fsq_topk_threshold_reuse_closely_matches_reference_global_counts`
  checks the faster top-k threshold reuse path with tolerance.
- `test_centered_fsq_surrogate_matches_reference_for_ste_variants` checks rect
  `topk`, `topk_plus_one`, `midpoint`, and tanh for `centered_fsq` and
  `centered_fsq_and_var`.
- `test_centered_fsq_topk_reuse_matches_reference_for_ste_variants` checks the
  same variants with `topk_indices` passed into the optimized path.
- `test_topk_routing_returns_topk_plus_one_indices`,
  `test_topk_routing_returns_sorted_topk_plus_one_indices_under_no_grad`, and
  `test_aux_routing_scores_return_topk_plus_one_indices` check the routing helpers
  that produce the k+1 index.
- `test_centered_fsq_optimized_path_timing_loop` also checks parity before timing
  reference vs optimized CUDA forward+backward loops.
- `tools/bench_centered_fsq_global_lb.py` runs the same exactness check and a timing
  loop under `torch.distributed.run`, including the old reference path's extra STE
  all-reduce.

## Timing Results

Environment:

- Host: 8x NVIDIA RTX A6000, driver 555.42.02, CUDA 12.5.
- Conda env: `megatron-submit`.
- PyTorch changed from `2.12.1+cu130` to `2.6.0+cu124` because CUDA 13 wheels could
  not initialize with the host driver.

Focused pytest:

```text
centered_fsq timing: reference=2.0970 ms, optimized=2.5554 ms, speedup=0.821x
```

This single-process timing is noisy and does not include the global-LB STE all-reduce
that was removed, so it is not the target performance case.

Distributed global-LB benchmark:

```text
centered_fsq global-LB timing: reference=12.8007 ms, optimized=9.6760 ms, speedup=1.323x, value_close=True, grad_close=True, max_grad_abs_diff=5.898376231883162e-10, load_balancing_type=centered_fsq, ste_type=rect, rect_position=topk, world_size=8
centered_fsq global-LB timing: reference=12.5156 ms, optimized=9.3164 ms, speedup=1.343x, value_close=True, grad_close=True, max_grad_abs_diff=0.0, load_balancing_type=centered_fsq, ste_type=rect, rect_position=topk_plus_one, world_size=8
centered_fsq global-LB timing: reference=12.3101 ms, optimized=9.3754 ms, speedup=1.313x, value_close=True, grad_close=True, max_grad_abs_diff=2.7939706104262996e-11, load_balancing_type=centered_fsq, ste_type=rect, rect_position=midpoint, world_size=8
centered_fsq global-LB timing: reference=10.1829 ms, optimized=9.2344 ms, speedup=1.103x, value_close=True, grad_close=True, max_grad_abs_diff=6.003784136510149e-10, load_balancing_type=centered_fsq, ste_type=tanh, rect_position=topk, world_size=8
centered_fsq global-LB timing: reference=21.0241 ms, optimized=18.1206 ms, speedup=1.160x, value_close=True, grad_close=True, max_grad_abs_diff=5.894737475919953e-10, load_balancing_type=centered_fsq_and_var, ste_type=rect, rect_position=topk, world_size=8
```

Opt-in distributed pytest timing:

```text
centered_fsq distributed timing: reference=10.4621 ms, optimized=9.2755 ms, speedup=1.128x, world_size=8
```

Run focused/full tests:

```bash
python -m pytest -q tests/unit_tests/transformer/moe/test_centered_fsq_load_balance.py -k "optimized_path or topk_threshold"
python -m pytest -q tests/unit_tests/transformer/moe/test_centered_fsq_load_balance.py
```

Run distributed benchmark:

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m torch.distributed.run --nproc-per-node 8 tools/bench_centered_fsq_global_lb.py
```

Additional benchmark options:

```bash
python -m torch.distributed.run --nproc-per-node 8 tools/bench_centered_fsq_global_lb.py \
  --ste-type tanh --tanh-slope 1.7
python -m torch.distributed.run --nproc-per-node 8 tools/bench_centered_fsq_global_lb.py \
  --rect-position midpoint
python -m torch.distributed.run --nproc-per-node 8 tools/bench_centered_fsq_global_lb.py \
  --load-balancing-type centered_fsq_and_var --rect-position topk
```

Run opt-in distributed pytest timing:

```bash
RUN_CENTERED_FSQ_DISTRIBUTED_TIMING=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/transformer/moe/test_centered_fsq_load_balance.py::test_centered_fsq_optimized_path_distributed_timing_loop
```

## Current Verification

- `python -m py_compile megatron/core/transformer/moe/moe_utils.py megatron/core/transformer/moe/router.py tests/unit_tests/transformer/moe/test_centered_fsq_load_balance.py tools/bench_centered_fsq_global_lb.py`: passed.
- `python -m pytest -q tests/unit_tests/transformer/moe/test_centered_fsq_load_balance.py`: 26 passed, 1 skipped. The skip is the opt-in distributed timing test.
- `python -m torch.distributed.run --nproc-per-node 8 tools/bench_centered_fsq_global_lb.py`: passed parity and timing for rect/topk.
- `python -m torch.distributed.run --nproc-per-node 8 tools/bench_centered_fsq_global_lb.py --ste-type tanh --tanh-slope 1.7`: passed parity and timing for tanh.
- `python -m torch.distributed.run --nproc-per-node 8 tools/bench_centered_fsq_global_lb.py --rect-position midpoint`: passed parity and timing for rect/midpoint.
- `python -m torch.distributed.run --nproc-per-node 8 tools/bench_centered_fsq_global_lb.py --rect-position topk_plus_one`: passed parity and timing for rect/topk-plus-one.
- `python -m torch.distributed.run --nproc-per-node 8 tools/bench_centered_fsq_global_lb.py --load-balancing-type centered_fsq_and_var --rect-position topk`: passed parity and timing for `centered_fsq_and_var`.
- `RUN_CENTERED_FSQ_DISTRIBUTED_TIMING=1 python -m torch.distributed.run --nproc-per-node 8 -m pytest -q tests/unit_tests/transformer/moe/test_centered_fsq_load_balance.py::test_centered_fsq_optimized_path_distributed_timing_loop`: passed on all ranks.
