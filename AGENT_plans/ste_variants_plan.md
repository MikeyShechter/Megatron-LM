# Plan: finite-difference STE variants in every2

## Setup

**Shared by every run.** E=2048, k=2, MoE every 2nd layer, Muon, 6000 iters, global LB, seed 1000 (results.md).
Unless a run says otherwise: plain router (no split router-weighter), softmax,
`moe_load_balance_ste_detach_threshold: false`.

**How runs are scored.**
- **vf:** final val loss.
- **MaxVio:** val MaxVioGlobal, mean of the last 5 evals.
- **Ties:** with no seeds, |Δvf| < 0.003 counts as a tie (results.md #8).
- **Matched MaxVio:** two runs are compared at matched MaxVio when their achieved MaxVio is within ±0.15.

**Controller fields** (from phase 1 on):
- `moe_aux_loss_coeff`: λ0
- `moe_balance_target_vio`: target T
- `moe_balance_update_mode`: additive | multiplicative
- `moe_balance_update_rate`: r
- `moe_balance_update_start_iter: 300`

λ need not converge: it can drift while MaxVio stays on target, then rise again. Runs are summarized by
λ̄ = median of `train/moe_aux_loss_coeff` over steps 1000–4250:

```
q.py sql "SELECT run_id, median(value) FROM train WHERE key='train/moe_aux_loss_coeff' AND step BETWEEN 1000 AND 4250 GROUP BY run_id"
```

The controller reads the per-step train MaxVio. Its sampling floor is ≥0.15, so achieved val MaxVio lands somewhat
below T. Always report the achieved value.

**Variant names:** V0 is the anchor, centered_fsq + topk + rect. The seven new variants:
- V1: centered_fsq + exact_margin + rect
- V2: centered_fsq + topk + higher_order_rect
- V3: centered_fsq + exact_margin + higher_order_rect
- V4: exact_jump_ste + exact_margin + rect
- V5: coordinate_perturbation_ste + exact_margin + rect
- V6: exact_jump_ste + exact_margin + higher_order_rect
- V7: coordinate_perturbation_ste + exact_margin + higher_order_rect

### Configs and submitting (details: run-experiments skill)

Every sweep is one config file, a copy of `configs/2048_experts_fd_base.yaml` with a new `job_details.name` and only
the keys listed for that phase changed. The base is the plain-router every2 setup set to run V0 (w=1, λ=0.01) with
the controller off and its start iter at 300. Phases 0, 0b and 1 are already written:
`configs/2048_experts_fd_smoke.yaml`, `configs/2048_experts_fd_early_lambda.yaml` and
`configs/2048_experts_fd_phase1.yaml`. The reduced phase 3 is `configs/2048_experts_fd_variants_fixed_lambda.yaml`.

Submit from the repo root with the megatron-submit env:

```bash
python submit_multiple.py configs/<file>.yaml -s run_megatron.sh        # with `conda activate megatron-submit`
```

## Phase 0: smoke tests

None of this code has run yet: the variants, the controller mode, the start-iter flag, and the window logs.

Unit tests (container):
- `tests/unit_tests/transformer/moe/test_centered_fsq_load_balance.py`
- `tests/unit_tests/transformer/moe/test_moe_router_metrics.py`
- `tests/unit_tests/test_arguments.py`

Smoke sweep `configs/2048_experts_fd_smoke.yaml` (8 runs of 300 iterations): V0–V7 as a LIST block, w=1,
`exit_interval: [300]` (keeps the full LR schedule; evals at 150 and 300). Each run uses the multiplicative controller
with λ0=0.01, r=1e-2, T=1.5 and start iter 150.

The runs end without `done`; monitor-runs shows them as `exited`. Afterwards, sync or delete the set, because an
unsynced older set blocks sync-runs.

Checks:
- **If a variant crashes or produces NaN** → fix it before phase 3.
- **If `train/moe_aux_loss_coeff` is not flat until step 150** (and from then on changing by a factor e^±r per step)
  → controller bug, so fix it before phase 1. Check with `q.py curve train/moe_aux_loss_coeff --runs <id> --raw
  --steps 149,150,151,152`.
- **If `ste/all_layers/window_frac_*` is missing, or not increasing from half_r to 4r** → logging bug.

### Phase 0 results (`26-10-01-experts2048_every2_fd_smoke`, deleted unsynced on 2026-10-02)

All three checks pass:
- No variant crashed or logged a non-finite value.
- λ stays at 0.01 through step 149. From step 150 it changes by e^+0.01 on every step, because MaxVio never came near
  T, and reaches 0.045 at step 300.
- The window keys are logged and increase from half_r to 4r.

The unit tests have not been run yet.

No variant balanced, though:
- **MaxVio at step 300 is 30–39 for all 8** (val loss 4.29–4.34). DeepSeek, QB, aux and split rect w=1 λ=0.01 are at
  2.2–3.5 by step 300.
- **By phase 1's definition, the w=1 window is empty** in the rect variants. `window_experts_empty_frac` is 0.89–0.91
  at steps 150 and 300, so ~90% of experts get no STE gradient. `window_frac_1r` × 2048 is ≈0.5–1.2.
- **The early imbalance is the same for every method.** Per-call train MaxVio (`train/router_balance/max_vio_dispatch_mean`,
  ≈5–6 when balanced) starts at 129 in every run with this seed and peaks at steps 5–10 in all of them.
  - At step 50, DeepSeek (sxwv1bbi) and split rect w=1 λ=0.01 (1wufeet2) are *more* imbalanced than the smoke runs
    (≈160 vs ≈90), yet they reach ≈10 by step 150.
  - The smoke runs stall at 35–43 from step 150 on, while λ rose 4.5×.
- **The LB gradient vanishes by step ~30.**
  - Grad-norm at step 5 is 37–57 (DeepSeek: 5.3). By step 30 it is down to DeepSeek's level (≈1), while per-call MaxVio
    is still ≈150.
  - With loads that imbalanced, the LB gradient can only be that small if almost no pairs are in the window. So the
    window empties around steps 10–30. This is an inference, since `ste/*` is logged only at evals.
  - The split rect run keeps grad-norm at 29–43 through step 30, and it recovers.
- **V6 anomaly** (exact_jump + higher_order_rect): from step 150 on, one of the 6 MoE layers has nearly every expert in
  the window for every token.
  - Evidence: `window_frac_1r` ≈ 1/6 and `window_expert_tokens_rel_p10` ≈ (E/K)/6.
  - Over 90% of those margins lie between w/2 and w, the kernel's negative lobe (weight −1/(6w), against 7/(6w) inside
    w/2).
  - It started at λ=0.01, before the controller turned on.
  - Hypothesis: exact_jump's uncentered n_i weights plus the kernel's sign flip trap the margins there.
  - V4 (same estimator, rect kernel) doesn't show it. It matters for phase 3.

## Phase 0b: early-λ test (8 runs of 300 iterations, 1 sweep)

**Why.** A rect STE gives no gradient to an expert that is outside the window for every token, whatever λ is. Raising
λ later only scales the pairs still inside. In the smoke runs, 4.5× more λ over steps 150–300 left the rect variants'
empty fraction unchanged.
- If the window empties in the first ~30 steps, the λ during those steps decides whether V0 can balance at all.
- No controller can act in time: even multiplicative r=1e-2 moves λ by only 1.35× in 30 steps. So in practice λ0
  decides it.
- Phase 1 as written would spend its low-λ runs on the smoke's failure: fixed λ=0.001 and 0.01 at w=1, and the four
  λ0=0.001 controller runs, which hold 0.001 until step 300. Its init-robustness test would then compare early λ, not
  the controllers.
- Raising λ has its own risks:
  - plain-router STE losses blow up above a narrow λ window (results.md #6);
  - a higher λ can make the window sparser (results.md #12).

**Sweep** `configs/2048_experts_fd_early_lambda.yaml` (`26-10-02-experts2048_every2_rect_early_lambda`):
- V0 at fixed λ, controller off;
- `moe_aux_loss_coeff: [0.03, 0.1, 0.3, 1.0]` × `load_balance_ste_width: [1.0, 3.0]`;
- `exit_interval: [300]`, and `eval_interval: [25]`, so that the window is logged at steps 25, 50, …, 300.

λ=0.01 at w=1 is covered by smoke V0, which has evals at 150 and 300 only. λ=1 probes the blow-up edge. As in phase 0,
the runs end as `exited`; afterwards, sync or delete the set.

What to read, per (λ, w):
- `window_experts_empty_frac` and `window_frac_1r` × 2048 from step 25 on: does the window stay populated, and when
  does it empty?
- Per-call train MaxVio at steps 50, 150 and 300, against DeepSeek (166, 11, 6.2) and the smoke runs (≈90, ≈45, ≈39).
- Grad-norm:
  - how long the LB gradient lasts (steps 5–50);
  - blow-up, i.e. far above DeepSeek's 0.2–0.5 after step 50.
- `router_logits/top1_pre_activation` − `top2_pre_activation`: the logit scale.

How it feeds phase 1 (to review together; not applied, because phase 1 went ahead as written, see below):
- **If some λ keeps the window populated and MaxVio near the references by step 300 without blowing up** → use it as
  phase 1's low init, and drop the fixed λ values below it.
- **If the window empties at every stable λ at w=1 but holds at w=3** → w=1 is too narrow for the plain router. Move
  phase 1's w=1 runs to w=3.
- **If no stable λ holds the window at either width** → the rect STE cannot pull the plain router out of the init
  transient. Decide together before phase 1.

### Phase 0b results (`26-10-02-experts2048_every2_rect_early_lambda`, deleted unsynced on 2026-10-02)

Only 3 of the 8 runs ran: λ=0.03 at w=1, and λ=0.3 and λ=1 at w=3.
- The other 5 crashed at startup with `ValueError: mmap length is greater than file size`.
- Cause: the new `eval_interval` needs a new validation index in the shared `data_cache_path`. All 8 jobs started
  together, and one job was still writing the index while the others read it.
- They were not rerun.
- Any sweep that changes the number of validation samples (`eval_interval`, `eval_iters`, `train_iters`,
  `global_batch_size`) can hit this when its jobs start together.

| λ, w | empty frac, step 25 → 300 | `window_frac_1r` × E: step 25 / peak / step 300 | val MaxVio @300 | val loss @300 | max grad-norm, steps 50–300 | top-1 logit @300 |
|---|---|---|---|---|---|---|
| smoke V0: 0.01, 1 | – → 0.91 | – / – / 0.99 | 35.4 | 4.315 | 0.51 | 7.1 |
| 0.03, 1 | 0.81 → 0.84 | 1.4 / 3.0 (step 50) / 1.7 | 30.2 | 4.403 | 0.69 | 4.7 |
| 0.3, 3 | 0.75 → 0.93 | 6.3 / 9.9 (step 75) / 0.21 | 46.7 | 7.461 | 553 | 65 |
| 1, 3 | 0.75 → 0.93 | 6.2 / 6.8 (step 75) / 0.20 | 48.8 | 7.561 | 8659 | 73 |

DeepSeek (sxwv1bbi): val MaxVio 3.4 at step 300, grad-norm 0.2–0.5 after step 50.
- **λ=0.03 at w=1 behaves like the smoke's 0.01.**
  - 81% of experts have no in-window token at step 25, and 84% at step 300.
  - The LB gradient is gone by step 50, where grad-norm is 0.46, DeepSeek's level.
  - MaxVio stalls at ≈30.
  - Val loss is 4.40 against 4.32: grad-norm is 75–83 at steps 1–5, so clipping shrinks the LM's update early.
- **λ=0.3 and λ=1 at w=3 blow up** (results.md #6).
  - Grad-norm is 600–2000 at step 1 and 10–1000× DeepSeek's for the whole run.
  - The LM stops learning at ≈7.5 from step 50 on, and MaxVio is worse (47–49).
  - The logits explode (top-1 65–73, top1 − top2 20–26), so the window empties anyway. Occupancy peaks at step 75
    and falls to 0.2 per token by step 300.
  - `1r/half_r` ≈ 6 at step 75: pairs pile up at the window edge, consistent with the LB pushing them out.
- **Most experts may never have been in the window.**
  - The empty fraction is 0.75–0.81 at step 25, across a 30× range of λ and both widths.
  - Routing is heavily imbalanced right at initialization: per-call MaxVio is 129 at step 1 in every run with this
    seed.
  - Hypothesis: the initial logits share a large token-independent part, so the same few experts sit near the
    threshold for every token.
  - Not tested: the first window log is at step 25. An eval every 5 steps up to step 50 would show it.
- **The 5 missing runs were expected to interpolate**, so phase 1 went ahead without them.
  - λ=0.3 and λ=1 at w=1 should blow up even harder, since the per-pair gradient scales with 1/w.
  - λ=0.1 should fall between the stall and the blow-up.

## Phase 1: controller test and plain rect anchor (16 runs, 1 sweep)

**Submitted on 2026-10-02 as written** (`26-10-02-experts2048_every2_rect_controller_test`), without the rest of
phase 0b, to see what plain rect V0 does in full-length runs. Phases 0 and 0b suggest two outcomes:
- the low-λ runs at w=1 will stall at high MaxVio;
- λ=0.1 may blow up (see the rule on fixed λ=0.1 below).

Sweep `configs/2048_experts_fd_phase1.yaml` (`experts2048_every2_rect_controller_test`). All runs are V0, and the
controller runs use target 1.5 and start at iteration 300. One `LIST_GROUP` entry per row:

| group | w | λ0 / λ | controller | runs |
|---|---|---|---|---|
| fixed λ | 1, 3 | 0.001, 0.01, 0.1 | none | 6 |
| additive | 1 | 0.001, 0.1 | additive, r ∈ {1e-5, 1e-4} | 4 |
| multiplicative | 1 | 0.001, 0.1 | multiplicative, r ∈ {2e-3, 1e-2} | 4 |
| sigmoid | 1 | 0.01, 0.1 | none, `moe_router_score_function: ["sigmoid"]` | 2 |

These rules are the starting point for reviewing the results together; nothing in later phases is fixed until then.
- **If fixed λ=0.1 at w=1 has collapsed or blown up before step 300** (grad-norm far above DeepSeek's) → the four
  λ0=0.1 controller runs are identical to it until step 300 and say nothing about the controller.
  - Judge the controllers on the λ0=0.001 runs only.
  - Repeat the high-init pair for the winner with λ0=0.03.
- **If the w=1 window is empty but w=3 balances** → w=1 is too narrow, and the controller results are uninformative.
  Repeat the 8 controller runs at w=3. "Empty" means `window_frac_1r` × 2048 < 0.3 at step 600, or
  `window_experts_empty_frac` > 0.01.
- **What a good controller looks like** (judged on outcomes; λ itself may never settle):
  - MaxVio sits near T from step ~1000 on;
  - both inits and both rates end within ±0.2 MaxVio and ±0.003 vf of each other;
  - vf is no more than 0.003 above the fixed-λ run with the nearest MaxVio.
- **If λ keeps drifting while MaxVio stays on target** → not a failure in itself. Compare the λ trajectories of the
  two inits: if they meet, init does not matter.
- **If one controller type meets these and the other doesn't** → use it in phases 2–3, preferring multiplicative at
  equal results (one rate for every scale of λ).
- **If neither does** → decide together between fixed λ ∈ {λ*/3, λ*, 3λ*} around the best fixed λ (3× the runs in
  phases 2–3) and a different controller setting.
- **If a sigmoid run beats the softmax fixed-λ run at matched MaxVio by more than 0.003** → add sigmoid V0 at both
  phase-2 working widths, target 1.5 (2 runs).
- **If any fixed-λ softmax run is within 0.003 of the bar** (vf 3.208–3.211 at MaxVio ≤ 2.9) → plain rect already
  matches DeepSeek/QB. The main question for the variants then becomes target 0.5.

### Phase 1 results (`26-10-02-experts2048_every2_rect_controller_test`, synced)

vf @ MaxVio (mean of the last 5 evals). The bar is 3.208–3.211 at MaxVio 1.4–2.9.

**Fixed λ, softmax.** The w=3 runs at λ=0.0001 and 0.0003 come from the follow-up sweep
`26-10-02-experts2048_every2_rect_w3_low_lambda`, which extends the grid below λ*=0.001.

| w \ λ | 0.0001 | 0.0003 | 0.001 | 0.01 | 0.1 |
|---|---|---|---|---|---|
| 1 | – | – | 3.327 @ 36.8 | 3.353 @ 35.3 | 3.898 @ 10.8 |
| 3 | 3.293 @ 44 | 3.289 @ 16.6 | **3.273 @ 4.28** | 3.360 @ 32.5 | 3.893 @ 11.3 |

Sigmoid at w=1: λ=0.01 gives 3.381 @ 36.7, and λ=0.1 gives 3.465 @ 7.5.

**Controllers** (w=1, T=1.5):

| controller | λ0=0.001 | λ0=0.1 |
|---|---|---|
| additive 1e-5 | 3.339 @ 17.6 (λ̄ 0.024) | 3.858 @ 19.9 (λ̄ 0.12) |
| additive 1e-4 | 3.362 @ 4.88 (λ̄ 0.23, final λ 0.57) | 6.226 @ 13.1 |
| multiplicative 2e-3 | 4.580 @ 18.0 (final λ 90) | 7.617 @ 314 (final λ 8950) |
| multiplicative 1e-2 | crashed at step 4631 (λ ≈ 6e15, Inf gradient) | crashed at step 4095 (λ ≈ 3e15) |

How the rules came out:
- **λ=0.1 at w=1 blew up early.**
  - Grad-norm reached 401 before step 300, against 25 at λ=0.001, and stayed 5–10× DeepSeek's for the whole run.
  - Every λ0=0.1 run that didn't diverge ends at vf 3.86–3.90.
  - So the λ0=0.1 controller runs say nothing about the controller.
- **The w=1 window is empty** (`window_experts_empty_frac` 0.40–0.94 at step 600), **and w=3 balances only at
  λ=0.001.**
  - So w=1 is too narrow.
  - The rule says to repeat the controller runs at w=3. But the lowest MaxVio at w=3 is 4.3, above T=1.5, and MaxVio
    isn't monotone in λ there. A controller aiming for 1.5 would push λ past 0.001 into the bad region.
- **No controller qualifies:** no run holds MaxVio near 1.5. "If neither does" applies.
  - **Multiplicative** runs away in all 4 runs. MaxVio never drops below T, and a higher λ makes balance worse, so λ
    grows exponentially.
  - **Additive** is bounded by its rate. Its best run (r=1e-4 from 0.001) ends at MaxVio 4.9.
- **A high λ late hurts much less than early.** Additive r=1e-4 from λ0=0.001 ends at λ=0.57 with vf 3.362, while
  λ=0.1 from step 0 gives 3.86–3.90 (single seed).
- **Sigmoid doesn't help.** At matched MaxVio (36.7 vs 36.8), sigmoid λ=0.01 is 0.054 behind softmax λ=0.001. So it
  is not added to phase 2.
- **No fixed-λ run is near the bar.** The best (w=3, λ=0.001) is 0.063 behind and less balanced.

**Windows** (`window_frac_1r` × 2048 = experts per token within ±r of the threshold; r = w/2):
- **w=3, λ=0.001 saturates by step ~1500.**
  - At the end, 1687 of 2048 experts are in the window, 99% are within 2r, and `1r/half_r` = 1.2. That is
    "saturated" by phase 2's classification.
  - Since rect's derivative is 1/w, a saturated rect at w=3 behaves roughly like the full STE at λ/3.
- **w=3, λ ≤ 0.0003 also saturate** (1734–1786 experts per token), but they are too weak to balance.
- **w=3, λ=0.01 stays at ≈320–343 experts per token**, with empty frac 0.80 and `p10` ≈ (E/K)/6. Apparently one MoE
  layer has all its experts in the window and the other five are empty, the same signature as V6 in phase 0.
- **w=1, λ=0.01 stays at ≈1.1 experts per token for all 6000 steps.**
- **At step 150, w=3 holds only 1.3–3.9 experts per token.** It would be 4.7–17 at 2r and 83–172 at 4r, so only
  about 4× the width (w≈12) would populate the window early.

**Low-load tail** (`vio/ZeroLoadFracGlobal`, `vio/LowLoadFracGlobal`, `vio/LoadP10Global`, added 2026-10-02;
`vio/MinVioGlobal` added 2026-10-03). Only runs submitted after those dates log them.
- **Dense or saturated estimators keep almost every expert alive.** In the aux, full-STE and saturated w=3 rect runs
  (λ ≤ 0.0003), ≤0.1% of experts are unused at the end, even at MaxVio 44. There, high MaxVio comes from a few
  overloaded experts.
  - The two low-λ w=3 rect runs had ≈26% unused experts at step 600, and they recovered later.
- **Failing boundary-local runs are mostly dead experts.** In the phase 3 variant runs (below), every run that
  doesn't balance has 88–94% of its experts completely unused, already by step 600.
  - `window_experts_empty_frac` tracks the dead fraction closely, e.g. 0.89 empty against 0.90 dead.
  - So an "empty window" there means dead experts, not merely experts far from the boundary.

## Phase 2: anchor width scan (8 runs, 1 sweep)

**Skipped (2026-10-03).** The controller was dropped, and phase 3 runs at fixed λ ∈ {0.001, 0.01} × w ∈ {1, 3}
instead (see phase 3). As written, the scan also had a gap:
- w=1 is "not working" (empty window);
- w=3 is "saturated";
- so no tested width is both working and local.

Sweep `..._rect_width_scan`: V0 with the chosen controller.
- λ0 = the median λ̄ of the chosen controller's phase-1 runs.
- `load_balance_ste_width: [0.3, 1.0, 3.0, 10.0]` × `moe_balance_target_vio: [1.5, 0.5]`.

Classify each width from its T=1.5 run; the T=0.5 runs only decide the extreme target (last two rules).

Decisions, per width:
- **Not working:** MaxVio stays above T + 0.5 while λ keeps rising, or the window is empty as defined in phase 1. →
  Too few tokens near the boundary, or LB too weak at this w.
- **Local:** `window_frac_1r / window_frac_half_r` is between 1.5 and 2.5. → The 1/w per-token gradient and the token
  count cancel; λ and w are separable here.
- **Saturated:** that ratio is < 1.3 (expected at w=10). → Near-`full` STE; keep it only as the practical
  reference.
- **Steep tail:** that ratio is > 3. → Occupancy is very sensitive to w; prefer a neighboring width.

Then pick the phase-3 widths:
- w_lo = the smallest working width; w_hi = the largest working width that is still local.
- **If only one width qualifies** → w_hi = the next larger working width.
- **If V0 misses T=0.5 but stays stable** → keep T=0.5 for phase 3; it measures which method gets closest.
- **If V0 blows up at T=0.5 at every width** (vf > 3.5, or grad-norm far above DeepSeek after step 300) → set the
  extreme target to the lowest MaxVio V0 held stably, rounded up to 0.1.

## Phase 3: the 7 variants (28 runs, 1 sweep)

**Replaced (2026-10-03) by a reduced fixed-λ version, run as written below in "Phase 3 (reduced) results".** The
original design follows for reference.

Sweep `..._fd_variants`, two LIST blocks (they multiply: 7 × 4 = 28):

```yaml
LIST_VARIANT:   # same 3 keys in every entry
  - moe_router_load_balancing_type: ["centered_fsq"]
    moe_ste_rect_poistion: ["exact_margin"]
    load_balance_ste_type: ["rect"]                 # V1
  # ... V2–V7 the same way
LIST_CELL:      # (w, T, λ0), with λ0 = V0's λ̄ at that w and T
  - load_balance_ste_width: [<w_lo>]
    moe_balance_target_vio: [1.5]
    moe_aux_loss_coeff: [<λ̄>]
  # ... the other 3 (w, T) cells
```

Compare each variant with V0 at the same (w, T). A cell counts if |Δvf| > 0.003 at matched MaxVio, or if one method
holds a MaxVio the other cannot. At T=0.5, a method dominates if it reaches lower MaxVio with no worse vf.

Effect of each axis (pairs that differ in one axis only):

| axis | pairs |
|---|---|
| margin | V1 vs V0, V3 vs V2 |
| kernel | V2 vs V0, V3 vs V1, V6 vs V4, V7 vs V5 |
| exact jump | V4 vs V1, V6 vs V3 |
| coordinate vs margin form | V5 vs V4, V7 vs V6 |

Decisions:
- **If an axis has the same sign in ≥3 of the 4 cells and none against** → real effect. **If it flips with
  width** → report the width dependence, not a winner.
- **If higher_order_rect helps only at w_hi** → consistent with removing the w² bias. **If it hurts at w_lo** → its
  39% extra variance dominates at small w.
- **If V4/V6 run at λ̄ ≥3× below V1/V3** → as predicted from the uncentered n_i weights (stronger per unit λ).
  Compare only at matched MaxVio.
- **If V4 is worse than V1 at w_hi but not at w_lo, and V5 does not show this** → finite-width artifact from experts
  that sit in the window without their partner. Prefer the coordinate form.
- **If a variant holds T=0.5 and V0 cannot** (or the reverse) → that is the headline result at the extreme target.
- **If a variant beats V0 by more than 0.003 at its better width for both targets** → carry it to phase 4.
- **If none beats V0** → report the null result, and carry V0 plus the best-ranked variant to phase 4.

### Phase 3 (reduced) results (`26-10-03-experts2048_every2_fd_variants_fixed_lambda`, synced)

**Design.** The controller was dropped. The sweep runs V1, V2, V4 and V5 at fixed λ ∈ {0.001, 0.01} × w ∈ {1, 3}:
16 runs, config `configs/2048_experts_fd_variants_fixed_lambda.yaml`.
- V0 in the same 4 cells is phase 1's fixed-λ runs.
- Each axis is read from one pair:
  - margin position: V1 vs V0;
  - kernel: V2 vs V0;
  - exact jump: V4 vs V1;
  - coordinate vs margin form: V5 vs V4.
- V4 and V5 use `exact_margin` (V5 has to), so they are one change from V1, not from V0.

vf @ MaxVio (mean of the last 5 evals), with the share of completely unused experts (`vio/ZeroLoadFracGlobal`, mean
of the last 5 evals) in brackets. V0 predates that metric.

| | w=1, λ=0.001 | w=1, λ=0.01 | w=3, λ=0.001 | w=3, λ=0.01 |
|---|---|---|---|---|
| V0 (phase 1) | 3.327 @ 36.8 | 3.353 @ 35.3 | 3.273 @ 4.28 | 3.360 @ 32.5 |
| V1 exact_margin | 3.316 @ 17.4 (29%) | 3.365 @ 36.1 (90%) | **3.266 @ 4.80 (0.2%)** | **3.319 @ 5.14 (16%)** |
| V2 higher_order_rect | 3.388 @ 73 (94%) | 3.395 @ 61 (92%) | 3.392 @ 74 (94%) | 3.374 @ 44 (88%) |
| V4 exact_jump | 3.378 @ 49 (94%) | 3.371 @ 37 (92%) | 3.384 @ 56 (94%) | 3.372 @ 32 (92%) |
| V5 coordinate | 3.352 @ 28 (91%) | 3.331 @ 19 (89%) | 3.359 @ 179 (94%) | 6.535 @ 175 (diverged) |

- **Margin position (V1 vs V0): the only change that helps.**
  - V1 is better in 3 of the 4 cells, by 0.007–0.041.
  - The biggest gain is at w=3, λ=0.01: V1 balances (MaxVio 5.1) where V0 fails (32.5). So with `exact_margin`,
    w=3 works at both λ values.
  - V1 is worse at w=1, λ=0.01, by 0.012 at the same MaxVio. One cell is against it, so by the rule above it is not a
    clean effect.
- **Kernel (V2 vs V0): worse in all 4 cells** (vf +0.014 to +0.119, MaxVio 44–74), so it is a real negative effect.
  - `1r/half_r` is 13–29 at the end, a steep tail: margins pile up toward the window edge, where the kernel is
    negative.
  - That is the same trap as V6 in phase 0.
- **Exact jump (V4 vs V1): worse in all 4 cells.**
  - Hypothesis from the code: exact_jump weights each pair by the expert's own token count
    (`jump_weight = global_counts − own assignment`).
  - That is 0 for a dead expert, so it can never pull dead experts back.
- **Coordinate vs margin form (V5 vs V4): flips with width.**
  - V5 is better at w=1 in both cells.
  - At w=3, V5 diverges at λ=0.01 (median grad-norm 10 after step 1000) and reaches MaxVio 179 at λ=0.001.
  - By the rule above, report the width dependence, not a winner.
- **Best rect run so far: V1 at w=3, λ=0.001 (3.266 @ 4.8).**
  - Its vf is level with the full STE at λ=0.01 (3.265), but it is less balanced (4.8 against 1.7).
  - It is still ≈0.055 behind the bar.
  - Its window is saturated (1649 of 2048 experts within ±r at the end, `1r/half_r` 1.24).

## Phase 4: follow-ups (≈10 runs)

**Sigmoid:** the 1–2 best variants with sigmoid, at their better width, T=1.5.
- **If sigmoid wins by more than 0.003** → also run it at T=0.5, plus V0 with sigmoid.

**Learnable biases:** sweep `..._fd_bias`.
- Grid: `moe_learnable_bias_type: [per_token_bias, expert_bias]` × `moe_router_score_function: [softmax, sigmoid]` ×
  a 2-entry LIST (best variant, V0), at its better width, T=0.5 (8 runs).
- Keep `tie_learnable_bias_lr_to_aux_loss_coeff: [true]`, and don't set `moe_learnable_bias_lr_mult`. Without the tie
  the controller does nothing: with the scores detached, the LB gradient reaches only the bias, whose optimizer update
  ignores the gradient's scale, so λ acts only through the tied bias LR.
- Softmax biases are additive in logit space, so the phase-3 width applies.
- Sigmoid biases put the margin in score space, so the width does not transfer. **If `window_frac_1r` × 2048 at
  step 600 is outside the range V0 had at its chosen width** → rerun with w scaled to match.
- **If expert_bias again fails to balance** (MaxVio > 100, as in all 6 earlier every2 runs) → drop it.
- **If per_token_bias improves vf at T=0.5 by more than 0.003** → also run it at T=1.5.

## Idea for later: full-STE warm start, then rect + controller (not scheduled)

Start each run with the `full` STE (identity derivative for every token-expert score) at a standard λ. After N
iterations, switch to the rect STE under study and turn on the MaxVio controller at the same iteration.

**Why.**
- **Rect can't reach most experts from the start.** In phases 0 and 0b, 75–81% of experts had no in-window token by
  step 25, at every λ (0.01–1) and both widths. A rect STE gives these experts no gradient, so no λ can bring them back:
  - low λ stalls at MaxVio 30–35;
  - high λ blows up both the network and the logits, which empties the window anyway.
- **The full STE reaches every expert from step 0**, like the methods that do recover from the same init: the aux
  loss's gradient reaches every expert through the softmax, and DeepSeek's bias moves every expert's bias each step.
  Running it through the init transient should leave a balanced router whose experts sit near the boundary, where the
  rect window can act.
- **The comparison is about rect, so rect runs from the switch on.** Starting the controller at the switch means λ̄
  measures only the rect regime. That is what the controller's start-iter delay was meant to do.

**Open before it can run:**
- **New code:** no flag switches the STE type at an iteration. `--load-balance-ste-schedule` only interpolates the
  width over all of `train_iters`, and a wide rect is not the full STE: rect weights in-window pairs by 1/w, while
  full weights every pair by 1.
- **The full-phase λ:** somewhere in 0.001–0.01 (see the full-STE results below).
- **λ at the switch:** for the same λ, full and rect have different strength (all pairs at weight 1 against in-window
  pairs at 1/w). The carried-over λ only sets how long the multiplicative controller needs to reach the rect λ.
- **The switch iteration N.**
- **Whether the window stays populated after the switch:** in phase 0b the rect LB pushed pairs out of the window
  (also results.md #12).

**Full STE alone** (centered_fsq, fixed λ, full length; `26-10-02-experts2048_every2_full_ste` and
`26-10-02-experts2048_every2_full_ste_low_lambda`):

| λ | 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|
| vf @ MaxVio | 3.2547 @ 3.44 | 3.2651 @ 1.72 | 3.442 @ 2.54 | 5.737 @ 189 (blew up) |

- **Lower λ buys vf with balance.** At matched balance, the best λ lies between 0.001 and 0.01.
- **λ=0.01 is the only run of these sweeps inside the bar's MaxVio range.** It is 0.055 behind the bar, and 0.045
  behind plain aux (5w5e2lak, 3.2206 @ 1.68).
- **The full STE balances from the start**, unlike rect at any λ. That is the property the warm start relies on.

## Side experiment: does the split router-weighter win by capacity? (`26-10-02-experts2048_every2_aux_extra_router_compute`)

**Setup.** Single router with aux loss at λ=1 (as 5w5e2lak), plus a residual bottleneck MLP of width m before the
router projection (`moe_router_extra_computation`). The MLP costs 2·m·H parameters and 2·m·H multiply-adds per token.
- m=1 matches the weighter's compute, since the weighter evaluates only the K=2 selected rows.
- m=1024 matches the weighter's parameters, E·H.

| run | vf @ MaxVio |
|---|---|
| plain aux (5w5e2lak) | 3.2206 @ 1.68 |
| m=1 (compute match) | 3.2193 @ 1.81 |
| m=1024 (parameter match) | 3.2155 @ 2.51 |
| split aux, no router bias (f4stiqse) | 3.2123 @ 1.22 |
| split aux, router bias (0nl5u4q1) | 3.2079 @ 1.41 |

- **Compute alone does nothing.**
- **The extra parameters close ≈0.005 of the gap.** The split router keeps 0.003 without router bias (noise level)
  and 0.008 with it, at better balance. So capacity explains part of the split router's advantage, not all of it.
- Unlike DeepSeek (`26-09-30-…extra_router_compute_deepseek`), aux stays balanced at m=1024.

## Rationale

- The controller turns λ into a measurement: each run is compared at matched balance, which is the question being
  asked (vf at a given MaxVio, and which MaxVio a method can hold at all). Testing it against fixed λ first checks
  that it is robust to init and rate and does not cost loss. The 300-step delay skips the init transient, during
  which MaxVio is high regardless of λ.
- Width does not transfer as a number, because occupancy depends on the logit scale the router learns. So widths are
  picked from the anchor's window logs and controller behaviour. Every variant gets two widths because the corrections
  pull in opposite directions: exact margin and jump matter at small w, the higher-order kernel at larger w.
- V1–V7 form a factorial design over margin, kernel and estimator, so each effect is read from pairs that differ in one
  axis.
- Biases change which parameter the LB gradient trains, and for sigmoid the margin space too, so they come after the
  main comparison.
