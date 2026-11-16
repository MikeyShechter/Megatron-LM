# QC + QB routing: scaling or smoothing the routing bias (for a later run)

Flag: `quantile_correction_qb_routing` (added 2026-11-10). Each step routes on QC's margin scores plus the previous
step's δb, unscaled and without EMA. This note covers whether that bias should be scaled or smoothed.

## Why the full previous-step δb can overshoot

δb_{t−1} removes all of batch t−1's load error, but only part of that error is still there in batch t:

1. **The router moves between steps.** Under plain QC a large load error keeps 60–70% of its size into the next step
   mid-run (`train/load_dynamics/all_layers/carry_over` 0.59–0.71 on 1s2fnw4q, steps 600–4200). A full correction
   also fixes the other 30–40%, which is already gone, so it overshoots by that much.
2. **Batch noise.** Batch t−1's own sampling and document-level noise is in δb_{t−1} but not in batch t, so routing
   with it adds fresh error.

QB with EMA 0 shows this pattern (7jgya8mw vs QC 1s2fnw4q):
- `carry_over` −0.16…0 vs 0.6–0.7;
- `persist_corr` −0.06…−0.22 vs ≈ 0.6;
- `sig_flip_frac` 0.17 vs 0.07–0.09.

The expert's error flips sign from step to step.

## Rule of thumb

Route with c·δb_{t−1}. Let ρ be the step-to-step correlation of the router's own load error. Relative to plain QC,
the dispatch error variance is then 1 − 2cρ + c²:
- c = 1 (current) helps only when ρ > 0.5;
- c = ρ is the best choice, leaving 1 − ρ².

QC's ρ, read as `carry_over` per 150-step window (1s2fnw4q), and the resulting RMS load error relative to plain QC:

| window | ρ | c = 1 | c = ρ |
|---|---|---|---|
| steps 1–150 | 0.11 | ×1.34 | ×0.99 |
| 150–300 | 0.42 | ×1.07 | ×0.91 |
| 600–4200 | 0.59–0.71 | ×0.76–0.91 | ×0.70–0.81 |
| decay, 4200–6000 | 0.20 | ×1.27 | ×0.98 |

So the unscaled bias likely hurts early and in the decay, and helps mid-run.

The estimate is crude:
- `carry_over` covers only the clearly imbalanced experts and is a ratio of sums;
- the combination may change the router's dynamics, because the experts now see balanced loads;
- MaxVio is a max, not an RMS.

Use it as a guide to what to look at, not as a prediction.

## Scale or smooth

- **Scale c < 1:** cuts both the overshoot and the injected noise, with no lag. c ≈ 0.6 matches mid-run ρ.
- **Smooth (EMA of δb):** averages the batch noise away but adds lag, which hurts most where the router moves fast
  (early training).

Neither is implemented. Either is a one-line change in `_update_router_qc_routing_bias`
(`megatron/core/distributed/finalize_model_grads.py`).

## What to look at in the first (unscaled) run

- `train/vio/MaxVioBatch` per step vs 1s2fnw4q. Worse than plain QC in steps ~1–300 or after the decay starts
  (4286) = overshoot.
- `train/load_dynamics/all_layers/carry_over` below 0, or `sig_flip_frac` well above QC's 0.07–0.09 = overshoot
  (QB's signature).
- If either shows up, try c ≈ 0.6.
