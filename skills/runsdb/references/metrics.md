# Metric glossary (megatron-moe runs)

Compiled on 2026-10-01 from the code in this repo. The `file:line` references are as of that date and will drift.
- T = megatron/training/training.py
- U = megatron/core/transformer/moe/moe_utils.py
- R = megatron/core/transformer/moe/router.py
- P = pretrain_gpt.py

Older runs may have used older definitions or names; `q.py keys <regex>` shows when each key was logged.

**Notation**
- E = #experts, K = top-k.
- f_e = expert e's share of a layer's attempted top-k assignments (before capacity drops, Σf = 1).
- P_e = mean over tokens of the full E-way router score.
- MaxVio = E·max_e f_e − 1. It is 0 when perfectly balanced and at most E/K − 1 when the router collapses; 1023 means collapse for E=2048, K=2.

**Pooling.** Per-layer counters are summed over every micro-batch of a train step (or over the whole eval pass), then
all-reduced over ranks (SUM; MAX for `*_worst`). "Mean over layers" means over MoE layers only (U:2852-2884, U:2955-3332).

**When things are logged.** The wandb step is the iteration. Eval runs every `eval_interval` (150) on `eval_iters`
(20) batches. The code also runs one more eval after training at the final step, and one on the test split
(T:1266-1309, T:3918-3957). In the mirrored data, a 6000-iter run has 40 val points, at steps 150 through 6000.

## Megatron standard (T:2584-3001)
- `lm loss`: per-iteration token-weighted cross-entropy over all micro-batches and DP ranks. Logged every step.
- `lm loss validation`: Megatron's own validation loss. It becomes `lm loss validation-regular` when task eval adds
  validation sets. Gotcha: the end-of-run test eval also writes `lm loss validation` at the final step, so prefer
  `val/regular/lm_loss`.
- `grad-norm`: global L2 norm of the gradients after unscaling, before clipping. Logged every step.
- `learning-rate` and `train/learning_rate`: the LR of the main param group.
- `decoupled_lr` and `train/decoupled_lr`: the LR of the embedding and output layers. Logged only with `--decoupled-lr`.
- `iteration-time`: wall seconds per iteration, averaged over each `log_interval` (10), with eval time excluded.
  Logged only with `--log-timers-to-tensorboard`.
- `loss-scale`, `batch-size` (global batch), `samples vs steps` (cumulative samples): bookkeeping values.

## train/* (MoE runs; every iteration; T:2793-2869)
- `train/lm_loss`: same value as `lm loss`.
- `train/aux_loss`: the coefficient-free LB loss, recomputed from the step's pooled loads. It is not the
  per-micro-batch loss actually optimized. The formula depends on the LB type:
  - `aux_loss`: E·Σ f_e·P_e
  - centered_fsq family (incl. quantile_correction_ste): 1 + E·Σ(f_e − 1/E)²
  - `fsq`: E·Σf²
  - `maxvio`: 1 + MaxVio

  It is the mean over layers (U:3228-3246, U:495-522).
- `train/router_entropy`: per-token entropy of the E-way scores ÷ log E (0..1); mean over tokens, then over layers.
- `train/router_values/f_p_l1`: Σ_e |f_e − P_e|, hard load vs soft mass (0..2); mean over layers.
- `train/router_values/avg_1_2_coef_diff`: top-1 minus top-2 combine weight per token (split router: the
  weighter's weights), averaged.
- `train/router_balance/max_vio_dispatch_mean|worst`: MaxVio of each individual router call (one rank-local
  micro-batch × one layer); mean / max over calls. Measures local, per-batch balance, unlike `vio/*`.
- `train/router_balance/max_vio_sequence_mean|worst`: MaxVio of each sequence's own expert loads; mean / max.
- `train/lm_ste/extras_better_frac` (added 2026-10-03): split router with `--moe-router-lm-loss-extra-experts` m > 0
  only. The fraction of tokens whose LM STE pushes their m extra experts (the top-m unselected experts by router
  score) up on net: the summed STE gradient on the extras' router scores is negative. "Better" follows
  `--moe-router-lm-loss-extra-experts-signal` (h_e = expert output, y = MoE output, g = grad of the LM loss wrt y):
  - `replacement`: the mean over (selected i, extra j) pairs of ⟨g, h_j − h_i⟩ is negative.
  - `weighted_replacement`: the mean over pairs of ⟨g, y_ij − y⟩ is negative, where y_ij is the weighter mixture with
    j in place of i.
  - `addition`: Σ_j w_j·⟨g, h_j − y⟩ < 0. Since 2026-10-03 ~19:30, w_j = q_j/(Z + q_j), the weight j would get
    when added at full weight (q = weighter weights, Z = their total over the selected experts). The first addition
    runs (26-10-03-…extra_experts_aux, runs 002 and 005) used q_j/Z and diverged.

  Tokens without a valid extra (padding, capacity drops) are excluded. Pooled over layers and micro-batches. It is
  recorded in backward, so there is no val/test version (U:2847-2888, U:3128-3148, R:1215-1413).
- `train/lm_ste/extras_better_frac_rank{r}` (added 2026-10-03), r = 1..m: the same per runner-up rank r (extras are
  sorted by router score): the fraction of tokens where the STE gradient on the rank-r extra's own score is
  negative. Replacement: ⟨g, h_j − mean_i h_i⟩ < 0. Weighted replacement: Σ_i ⟨g, y_ij − y⟩ < 0. Addition:
  ⟨g, h_j − y⟩ < 0. Shows whether ranks beyond the first carry signal, i.e. whether a larger m is worth its compute.
- `train/lm_ste/selected_better_frac_rank{r}` (added 2026-10-04), r = 1..K: the same for the selected experts, by
  router-score rank: the fraction of tokens where the LM STE gradient on the rank-r selected expert's score is
  negative (the STE pushes it up). With `addition` and K=2 the two selected experts always get opposite signs
  (h_1 − y and h_2 − y point in opposite directions), so rank 1 and rank 2 sum to about 1 and rank 1 is the
  fraction of tokens where the router's first choice beats its second to first order (U:2847-2888).
- `train/ste_bandwidth`: the scheduled STE width at this step (0 for STE type `full`). It is logged even when no
  STE loss is active.
- `train/moe_aux_loss_coeff`: the current coefficient. It changes only via the MaxVio controller or metagrad. The
  controller reads each step's train MaxVioGlobal and moves the coefficient up when it is above
  `--moe-balance-target-vio`, down otherwise:
  - `--moe-balance-update-rate r` sets the step size.
  - `--moe-balance-update-mode additive` (default) adds or subtracts r, clamped at 0.
  - `--moe-balance-update-mode multiplicative` (added 2026-10-01) multiplies by e^±r.
  - `--moe-balance-update-start-iter X` (added 2026-10-01) holds the initial value until iteration X.

  With `--moe-router-enable-expert-bias` the controller moves `moe_router_bias_update_rate` instead.
- `train/moe_router_bias_update_rate`: the DeepSeek expert-bias update rate; with
  `--moe-router-bias-update-type id` it is ID Balancing's integral gain K_i, and with `proportional` (added
  2026-10-06) the u of Loss-Free's b += u·e, e = n̄ − n_e in token counts of the global batch. Logged only with
  `--moe-router-enable-expert-bias`.
- `train/id_balancing/gate_frac` (added 2026-10-04): logged only with `--moe-router-bias-update-type id`. Per layer,
  the fraction of experts whose derivative gate was open in this step's bias update: g_e = 1[e_prev·(e − e_prev) > 0],
  with e = (n̄ − n_e)/n̄ from the step's global token counts, i.e. the expert's load error grew without crossing
  zero. Mean over layers. 0 at the first step. The paper (arXiv 2609.39137, Fig. 4a) sees about 0.38 early and 0.25
  later (U:3333-3343, U:3677-3713, R:2105).
- `train/expert_bias/{abs_delta,std,range}` (added 2026-10-04): the balancing bias of the loss-free methods, i.e.
  `expert_bias` with `--moe-router-enable-expert-bias` (`sign` or `id` update) or `qb_beta` with
  `quantile_balancing`. Per layer, at this step's global-batch update, then mean over layers:
  - `abs_delta`: mean over experts of |Δb_e − mean_e Δb_e|. The common shift is removed because it does not change
    routing (ID and QB biases are zero-mean anyway; DeepSeek `sign` is not). With `sign` this is ≈ the update rate by
    construction. The paper's bias-drift plot (Fig. 5a) is |Δb| between checkpoints 1k steps apart instead.
  - `std`: standard deviation of the bias over experts after the update.
  - `range`: max − min of the bias over experts. Compare to the score scale: a range above 1 with sigmoid scores
    can override the router entirely.

  Bias units are those of the selection scores: sigmoid/sqrtsoftplus scores, or logits for softmax
  (U:3318-3331, R:2105). For `quantile_balancing` the bias is on the logits for every score function, unless
  `moe_router_quantile_balancing_on_scores` (added 2026-10-07) puts it on the activated scores. With `--moe-router-expert-bias-post-softmax` (added 2026-10-06) the softmax bias is added
  to the softmax probabilities, so its units are probabilities, and `vio/QBShift*` are then in probability units
  too.
- `train/vio/*` (added 2026-10-04): per-step load violations of the step's global batch (attempted assignments
  summed over micro-batches and ranks), the per-step metrics of the ID Balancing paper (Eq. 13). Unlike `vio/*`
  they are logged every iteration, so early spikes are visible; take max over steps from `raw()`, since the
  `train` table averages over 50-step windows.
  - `train/vio/MaxVioBatch`: per-layer MaxVio, mean over layers. The same value the MaxVio controller reads.
    `train/vio/MaxVioBatchWorstLayer`: max over layers.
  - `train/vio/MinVioBatch`: per layer 1 − E·min_e f_e, mean over layers. `train/vio/MinVioBatchWorstLayer`: max over
    layers.
  - `train/vio/Top10PctShare`: share of assignments that go to the busiest ⌈0.1·E⌉ experts (205 for E=2048); mean
    over layers. ≈ 0.10 when balanced (paper Fig. 4c).
  - The paper's other summaries (worst MaxVio over steps, mean over the last 1k steps, cumulative excess of MaxVio
    above 1) come from `train/vio/MaxVioBatch` in `raw()`, which has every step.

  (U:3304-3316)
- `train/load_dynamics/*` (added 2026-10-14): how each expert's load moves from one training step to the next,
  from the same per-step global loads as `train/vio/*`. Per step and expert: the load error r_e = n_e/q − 1 (> 0
  overloaded; q = the mean load, T·K/E = 512 here) and z_e = (n_e − q)/√q, the same error in units of sampling
  noise. A fresh batch moves n_e by about √q ≈ 23 tokens (4.4%) even at perfect balance, so step-to-step counts
  must be read against their noise level. "Clearly" over- or underloaded means |z_e| > 3, a load error above 13%
  at q = 512.
  - Accumulated over each eval interval (150 steps) and logged at the eval steps, per layer
    (`train/load_dynamics/Layer N/<stat>`, N as in `vio/MaxVio/Layer N`) and as the mean over layers
    (`train/load_dynamics/all_layers/<stat>`). A ratio whose denominator is empty in a window (e.g. no clearly
    overloaded expert) is not logged for that window.
  - `flip_frac`: fraction of experts whose side (over or under) changed since the previous step. ≈ 0.5 under
    noise, because experts near balance flip about every other step; on its own it mostly counts the experts
    inside the noise band.
  - `dwell_steps`: mean number of consecutive steps an expert stays on one side, within the window. ≈ 2 under
    noise, ≈ 1 for strict alternation (overshoot), long (up to 150) when experts stay on one side.
  - `sig_flip_frac`: fraction of expert-steps that swing from clearly overloaded to clearly underloaded or back.
    The side has hysteresis at |z| = 3: it changes only when the expert crosses the opposite threshold. ≈ 0
    under noise, so any clear value is overshoot that noise can't explain. Swings smaller than ±13% are missed.
  - `worsen_frac`: fraction of experts whose error grew in the same direction without crossing zero: ID
    Balancing's derivative gate g_e = 1[e_e(t−1)·Δe_e(t) > 0] (arXiv 2609.39137, Eq. 11). ≈ 0.24 under noise.
  - `worsen_over_frac` / `worsen_under_frac`: the same, among the experts that were clearly overloaded /
    underloaded at the previous step: the share that got even more overloaded / underloaded at the next step.
    ≈ 0 under noise or with a strong restoring push; ≈ 0.45 when the same experts stay imbalanced and nothing
    pushes them back; above that, the imbalance feeds itself. Only changes faster than the sampling noise
    show up here; slow runaways are in `drift_*`.
  - `carry_over`: Σ r_e(t)·r_e(t−1) / Σ r_e(t−1)² over the clearly imbalanced experts: the share of their load
    error still there one step later. 1 − carry_over is corrected per step; ≈ 1 no correction, > 1 growing, < 0
    the expert lands on the other side (overshoot). ≈ 0 under noise.
  - `persist_corr`: lag-1 autocorrelation of each expert's r around its own window mean, pooled over experts.
    < 0 oscillation, ≈ 0 noise or static imbalance, > 0 the error drifts slowly within the window (a runaway, a
    slow oscillation or a slow correction; `drift_*` tells them apart).
  - `persistent_share`: Σ_e T·m_e² / Σ_e Σ_t r_e², with m_e the expert's mean r over the window's T steps: the
    share of the squared load error that is a fixed per-expert offset. ≈ 1 the same experts stay over- or
    underloaded; 1/T ≈ 0.007 under noise.
  - `fluct_noise_ratio`: RMS of r around each expert's window mean, times √q. ≈ 1 the step-to-step movement is
    sampling noise; well above 1, real dynamics (oscillation, drift or spikes).
  - `imbalanced_frac`: fraction of experts whose mean r over the window is clearly imbalanced (|m_e| > 3/√q).
    ≈ 0 under noise.
  - `drift_worsen_over_frac` / `drift_worsen_under_frac`: among the experts that were clearly overloaded /
    underloaded on average over the first half of the window, the share whose average imbalance is larger, in
    the same direction, over the second half. Averaging over 75 steps removes the sampling noise, so these catch
    slow runaways that the per-step `worsen_*` miss: ≈ 0.5 static imbalance, → 1 a runaway (an overload feeding
    itself, or an expert dying), → 0 the method is correcting. Not logged when no expert qualifies (e.g. under
    noise or fast oscillation).
  - `drift_ratio`: Σ m1·m2 / Σ m1² over the same experts (m1, m2: the mean r over the first and the second half
    of the window): the share of the first half's imbalance still there in the second half. ≈ 1 static, < 1
    correcting, > 1 growing, < 0 reversed.
  - Values on synthetic multinomial loads (E=2048, q=512, one 150-step window, all_layers; – = not logged):

    | regime | flip_frac | dwell_steps | sig_flip_frac | worsen_frac | worsen_over_frac | carry_over | persist_corr | persistent_share | fluct_noise_ratio | imbalanced_frac | drift_worsen_over_frac | drift_ratio |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|
    | noise only (balanced) | 0.50 | 2.0 | 0.000 | 0.24 | 0.00 | −0.01 | −0.01 | 0.007 | 1.0 | 0.00 | – | – |
    | static imbalance | 0.06 | 14 | 0.000 | 0.46 | 0.45 | 0.98 | −0.01 | 0.98 | 1.0 | 0.65 | 0.48 | 1.00 |
    | overshoot (half the experts ±10% every step) | 0.97 | 1.0 | 0.13 | 0.01 | 0.00 | −0.61 | −0.84 | 0.009 | 2.5 | 0.00 | – | – |
    | slow runaway | 0.11 | 8.6 | 0.000 | 0.46 | 0.47 | 1.00 | 0.95 | 0.71 | 4.9 | 0.65 | 1.00 | 3.20 |
    | slow oscillation (period 60 steps) | 0.15 | 6.6 | 0.020 | 0.42 | 0.42 | 0.96 | 0.96 | 0.06 | 5.0 | 0.03 | 0.06 | 0.94 |
    | correcting (imbalance decays over 50 steps) | 0.26 | 3.7 | 0.000 | 0.35 | 0.34 | 0.92 | 0.75 | 0.53 | 2.1 | 0.16 | 0.00 | 0.22 |

  (U:4199-4329, U:4378-4386, T:2873)
- `train/token_dropping/*`: logged only with a capacity factor. `train/metagrad_*`: logged only with `--metagrad-params`.
- `train/quantile_correction/*`: present in a few old runs only; the definition is not in the current code.

## vio/* (regular validation set only, at each eval; U:3214-3226)
- `vio/MaxVioGlobal`: per-layer MaxVio from loads pooled over the whole eval pass; mean over layers.
- `vio/MaxVioGlobalWorstLayer`: max over layers. `vio/TotalVioGlobal`: per layer Σ|f_e − 1/E|/(1/E); mean over layers.
- `vio/MinVioGlobal` (added 2026-10-03): per layer (n̄ − min_e n_e)/n̄ = 1 − min_e f_e·E, from the same pooled loads;
  mean over layers. 0 = the least-loaded expert gets its fair share; 1 = some expert gets nothing.
  `vio/MinVioGlobalWorstLayer`: max over layers.
- `vio/MaxVio/Layer N`: per layer, N = 0-based global layer index. Dense layers are absent (with moe_layer_freq=2,
  only every other layer appears).
- Low-load tail (added 2026-10-02), from the same pooled eval loads. Each expert's load is taken relative to its fair
  share (f_e·E, 1 = balanced); the per-layer values are averaged over layers:
  - `vio/ZeroLoadFracGlobal`: fraction of experts with no assignment in the whole eval pass (unused experts).
  - `vio/LowLoadFracGlobal`: fraction of experts below 10% of their fair share (almost unused).
  - `vio/LoadP10Global`: 10th percentile of f_e·E over experts.
  - `vio/ZeroLoadFrac/Layer N` (added 2026-10-04): the unused-expert fraction per layer, N as in `vio/MaxVio/Layer N`.
- Distance to balance in logit units (added 2026-10-05, every run, regular validation only):
  `vio/QBShiftAbsP50Global`, `vio/QBShiftAbsP90Global`, `vio/QBShiftAbsMaxGlobal`, and
  `vio/QBShiftAbsP50/Layer N`.
  - Per expert: the Quantile-Balancing shift δb_e that would give it exactly its fair share N·K/E, holding the
    token thresholds fixed (margins vs the (K+1)-th score; the same δb_e quantile correction uses). It is the
    signed mean over the eval calls, then its absolute value; P50 / P90 / max over experts, mean over layers.
  - Units are those of the selection scores (logits without selection biases). Compare it with the rect half-width
    r = w/2: when most experts need to move by more than r, a fixed window cannot reach the tokens that must flip.
  - Computed from each rank's local micro-batch and averaged over the load-balancing group
    (`quantile_correction_mean_local` style); quantile-correction runs reuse their own δb_e.
- Per-batch distance to balance (added 2026-10-14, every run, regular validation only):
  `vio/QBShiftBatch/Layer N/<stat>` and `vio/QBShiftBatch/all_layers/<stat>` (mean over layers), with <stat> one of
  `abs_mean`, `abs_p10`, `abs_p25`, `abs_p50`, `abs_p75`, `abs_p90`, `abs_p99`, `abs_max`, `under_abs_mean`,
  `over_abs_mean`, `over_frac`.
  - The same δb_e as `vio/QBShift*`, but for each eval batch on its own, from exact margins and the exact global
    quantile (EQB) for every method. Quantile-correction runs with `quantile_correction_exact_margin` and
    `moe_quantile_estimator: exact` reuse their own δb_e, the shift their STE window uses.
  - Each batch is summarized over the experts of a layer, then the summaries are averaged over the eval batches
    (20). `vio/QBShift*` instead averages each expert's δb_e over the batches first, which cancels the
    batch-to-batch noise; the ratio of the two shows how much of the shift persists across batches.
  - `abs_*`: mean, quantiles and max of |δb_e| over experts. `under_abs_mean` / `over_abs_mean`: mean |δb_e| over
    the underloaded (δb_e > 0) / overloaded (δb_e < 0) experts. `over_frac`: fraction of overloaded experts.
  - Units are those of the selection scores (logits without selection biases); compare with
    `router_logits/std_pre_activation/Layer N`.

  (U:1514-1540, U:4041-4060, R:1735-1763)
- The train-time MaxVio is logged per step as `train/vio/MaxVioBatch` since 2026-10-04 (see train/*); before
  that it only drove the coefficient controller.

## val/* (at every eval and once after training; T:4211-4511)
- `val/regular/lm_loss`: LM loss on the held-out regular validation set.
  - The 17 early runs without task eval log it as `val/lm_loss`; runsdb fills `val/regular/lm_loss` from it.
  - Some older runs also have `val/regular/{aux_loss,router_entropy,...}`.
- `val/aux_loss`, `val/router_entropy`, `val/router_values/*`, `val/router_balance/*`: the same formulas as
  `train/*`, pooled over the regular eval pass.
- Split-router eval modes (only with `--moe-router-use-separate-weighter`; they replay the same eval batches):
  - regular: the router selects and the weighter weights. Logged as `val/regular/lm_loss` and `vio/*`.
  - `val/weighter/{lm_loss,MaxVioGlobal,MaxVioGlobalWorstLayer}`: the weighter both selects and weights, i.e. a
    plain single-router MoE built from the weighter. Gated by `--eval-split-router-with-weighter`.
  - `val/router_both/*`: the router selects and also weights (the router logits go through
    `moe_weighter_activation`). Gated by `--eval-split-router-with-router-weights`.
  - `val/router_ranks_<a>_<b>/lm_loss` (added 2026-10-04): each token goes to the experts at router-score ranks
    a, b instead of its top-k, and the weighter weights them as usual. One key per rank set of
    `--eval-router-selection-ranks "a,b c,d"` (regular validation only; T:4346-4356, T:4520-4532, R:2248-2261). Used by `tools/eval_routing_ranks.py`, whose runs go to the separate wandb project
    megatron-moe-analysis.
  - `val/router/*` (2026-08-09 to 08-20 only): a copy of the regular-mode numbers.
- Before 2026-06-08, every eval set (test included) wrote router stats under plain `val`/`vio`/`ste`, so later
  sets overwrote earlier ones at the same step. Before 2026-06-07, the test loss also went to `val/lm_loss`.
- `val/router_logits/*`, `val/router_bias/*`: names used on 2026-07-18 only. They are now the unprefixed
  `router_logits/*` and `router_bias/*`.

## ste/* (regular validation only; R:1318-1353, U:596-647, U:3248-3275)
- Margin = selection score (incl. any selection bias) minus the top-k threshold, where the threshold comes from
  `--moe-ste-rect-poistion`: `topk` (lowest selected), `topk_plus_one`, `midpoint` or `exact_margin`.
- Radius r = width/2 for `rect`, and width (the full support) for `higher_order_rect`.
- `ste/all_layers/in_rect_frac`: fraction of selected (token, expert) pairs with |margin| < r. It is 1 for STE
  `full` and 0 for tanh/triangle or width 0.
- `ste/all_layers/all_experts_in_rect_frac`: the fraction over all (token, expert) pairs, all E experts, with
  |margin| < r. Added 2026-08-02.
- `ste/all_layers/avg_over_rect|max_over_rect`: per (layer, expert), the fraction of tokens selecting it with
  margin ≥ r (deep inside the selection, so no STE gradient); mean / max over layers × experts.

**Window shape and coverage (added 2026-10-01)**

These are logged only for STE `rect` or `higher_order_rect` with width > 0; they are absent otherwise, while the
keys above log 0. Margins of exactly 0 are left out, because they belong to the expert that sets the threshold under
`topk` / `topk_plus_one`. For `exact_margin`, `window_frac_1r` equals `all_experts_in_rect_frac`; for `topk` it is
`all_experts_in_rect_frac` − 1/E.

- `ste/all_layers/window_frac_half_r|1r|2r|4r`: the fraction of all (token, expert) pairs with 0 < |margin| < m·r,
  for m = 0.5, 1, 2, 4.
  - What it is for: checking, within one run, whether occupancy grows linearly with the window. The rect gradient per
    token is ∝ 1/w, so w only cancels out of the total LB gradient while the margin density is flat across the window.
  - Reading `window_frac_1r / window_frac_half_r`:
    - ≈ 2: the density is flat at this scale (λ and w separate);
    - ≫ 2: the density rises away from the boundary, and occupancy grows faster than w;
    - ≈ 1: the window is saturated.
  - `window_frac_2r / window_frac_1r` predicts what doubling w would do on the same logits.
  - Unit conversion: `window_frac_1r` × E = experts near the boundary per token, and × (tokens per step) / E =
    in-window tokens per expert per step (×256 in the every2 setup).
- `ste/all_layers/window_expert_tokens_rel_p10`: per-expert coverage.
  - Definition: per (layer, expert), the number of in-window pairs at radius r over the eval pass, divided by that
    expert's balanced share of assignments (assignments / E). It reports the 10th percentile over experts, averaged
    over layers. The mean over experts is `window_frac_1r` × E / K.
  - What it is for: a low value means a tail of experts with few tokens near the boundary, which a fixed-width STE
    can barely move.
- `ste/all_layers/window_experts_empty_frac`: the fraction of (layer, expert) pairs with no in-window pair at
  radius r over the whole eval pass. These experts get no STE gradient at all.
- Per layer (added 2026-10-04): `ste/window_frac_{half_r,1r,2r,4r}/Layer N` and
  `ste/window_experts_empty_frac/Layer N`, the same quantities for MoE layer N (0-based global layer index, as in
  `vio/MaxVio/Layer N`). The rect STE balances layer by layer: a layer's window stays empty until its logits
  compress to the window scale, then fills within ~100 steps (c7ckylsu: layers 2–10 at step ~810, layer 0 never), so
  the all-layer averages can hide one stuck layer.
- Which side of the boundary the window spends its gradient on (added 2026-10-05; rect and higher_order_rect only).
  The load error sign of each expert comes from its pooled eval load n_e vs its fair share q = assignments / E.
  - `ste/all_layers/window_useful_frac` (and `ste/window_useful_frac/Layer N`): the fraction of in-window pairs on
    the useful side, i.e. selected pairs of overloaded experts and unselected pairs of underloaded experts: the
    pairs whose flip would reduce the load error. The rest of the window pushes pairs that change no load away from
    the boundary. Quantile correction is 1 by construction.
  - `ste/all_layers/window_need_coverage_p10|median`: per expert, useful in-window pairs / |n_e − q| (the flips
    that are needed); 10th percentile and median over experts, mean over layers. Values ≫ 1 mean the window holds
    far more pairs than balance needs; 0 means the STE gives that expert nothing to fix it with.
  - `ste/all_layers/window_unreachable_frac` (and `ste/window_unreachable_frac/Layer N`): the fraction of experts
    off balance by at least 10% of q with no useful in-window pair at all. These experts can only recover through
    other experts' gradients (the dead-expert phase).
- How to use them: check after the first evals of a sweep, e.g.
  `q.py curve ste/all_layers/window_frac_1r --sweep X --steps 150,300,600`.
  - An empty window (`window_frac_1r` × E ≲ 0.3), or `window_experts_empty_frac` above ~0.01, early in training
    means w is too small for the router's logit scale.
  - Comparing these keys across widths and variants tells you whether two runs had comparable windows.

## Logits, biases, weighter, quantile correction (regular validation only)
- `router_logits/top1|top2|avg_pre_activation`: per-token largest and second-largest logit, and the mean over all
  logits, before the activation. They include learnable `*_weight` biases but never the DeepSeek expert bias; with
  a split router they are the selection head's logits. `weighter_logits/*`: the same for the weighter's logits.
- `router_logits/std_pre_activation` (added 2026-10-05, and `router_logits/std_pre_activation/Layer N`): the
  standard deviation of a token's router logits over all experts, averaged over tokens (and layers for the
  all-layer key). The direct measure of logit compression; compare with the STE half-width r.
- `router_bias/expert/*`: the learnable per-expert bias (top1/top2/mean over experts).
- `router_bias/token/*`: the per-token bias from the learnable projection. `router_bias/both/*`: the token +
  expert sum.
- `weighter/*` (split router only):
  - `effective_k` = exp(entropy of the combine weights over the selected experts), between 1 and K.
  - `top1_weight`
  - `selected_weight_entropy_normalized`
  - `weighted_vs_assignment_l1` = Σ_e |W_e − f_e|, combine-weight mass vs load
  - `per_expert_mean_selected_weight_min|max`
- `quantile_correction/delta_b_e/{mean,max,min}`: the per-expert shift that would give every expert exactly N·K/E
  tokens (from topk_plus_one margins, or exact margins with `quantile_correction_exact_margin`); mean / max / min
  over (layer, expert). Logged only with `quantile_correction_ste`.
- `quantile_correction/affected_token_expert_fraction`: the fraction of (token, expert) pairs whose selection would
  flip under that shift. This is the fraction of pairs inside the quantile-correction windows. Each expert's window
  holds exactly |n_e − q| tokens, so it equals Σ_e |n_e − q| / (N·E) = K·TotalVio / E² on the same batch (about
  1–2× the eval-pooled `vio/TotalVioGlobal` value, since each eval batch is noisier than the pooled loads).
- `quantile_correction/ste_height_max` (added 2026-10-05): the largest per-token STE height 1/|δb_e| over experts
  with a non-empty window, i.e. 1 / (the narrowest window). With `quantile_correction_unnormalized` the height is
  1 everywhere, and this key is still 1 / (the narrowest window).
- `quantile_correction/mirror_token_expert_fraction` (added 2026-10-14): the fraction of (token, expert) pairs in the
  mirror bands, the flip bands' mirror images across the current boundary.
  - The mirror band holds the tokens within |δb_e| on the other side of the boundary: an underloaded expert's
    selected tokens with margin below δb_e, an overloaded expert's unselected tokens with margin above −|δb_e|.
  - It has the same normalization as `affected_token_expert_fraction` (the flip bands), so the symmetric window's
    in-window fraction is the sum of the two.
  - Logged for every QC run. The band gets gradient only with `quantile_correction_symmetric_window`; otherwise it
    shows how many tokens that window would add.
  - Like the other `quantile_correction/*` keys, it is eval only and pooled over layers.
- `quantile_correction/mirror_to_flip_ratio` (added 2026-10-14): Σ mirror-band pairs / Σ flip-band pairs, over all
  layers and experts of the eval batches.
  - With the symmetric window and mass normalization, the flip band's share of an expert's push is
    |flip_e| / (|flip_e| + |mirror_e|), so about 1 / (1 + ratio) when experts are alike.
  - A ratio near 1 means the margin density is about the same on both sides of the boundary.
  - Logged for every QC run, like the fraction above.
- Config `quantile_correction_unnormalized` (added 2026-10-05): the quantile-correction STE uses height 1 inside each
  expert's window instead of 1/|δb_e|, so the window width does not set the gradient scale. Each expert's total STE
  weight is then |n_e − q| (the number of flips needed) instead of about the token density at its boundary.
- Config `quantile_correction_extended_window` (added 2026-10-06): each expert's quantile-correction STE window
  extends from the tokens the QB shift flips to every token on the far side of the QB-corrected boundary. For an
  overloaded expert, that is all tokens unselected after the correction (about N − q tokens). For an underloaded
  expert, it is all tokens selected after it (q tokens). Independent of `quantile_correction_unnormalized`: the
  normalized height is 1/width, where the width runs from the corrected boundary to the farthest token in the window
  over the load-balancing batch, so it is much smaller than 1/|δb_e|. `quantile_correction/ste_height_max` and
  `affected_token_expert_fraction` still describe the flip window, not the extended one.
- Config `quantile_correction_symmetric_window` (added 2026-10-14): each expert's quantile-correction STE window is
  centered on the current boundary, i.e. the rect |m| ≤ |δb_e|.
  - It holds the flip band (the tokens between margin 0 and −δb_e) and its mirror band: the tokens within |δb_e| on
    the other side, whose assignment already agrees with the correction (an underloaded expert's selected tokens, an
    overloaded expert's unselected ones).
  - Every token in the window is pushed the same way.
  - The heights follow the window: 1/width uses 2|δb_e|; mass normalization gives T, and count normalization q,
    spread over the tokens of both bands.
  - Margins, post and detach work as for the flip window. No effect with `quantile_correction_extended_window`.
  - `quantile_correction/ste_height_max` and `affected_token_expert_fraction` still describe the flip window. The
    mirror band's size is `quantile_correction/mirror_token_expert_fraction` and `mirror_to_flip_ratio`.
- Config `quantile_correction_count_normalized` (added 2026-10-06): scales each expert's STE height by
  q / (tokens in its window), q = its fair share of the load-balancing batch, so every expert's window carries
  a total weight of q. With height 1, each expert's total push is then 2λ(f_e − 1/E): it depends only on the load
  error, not on the batch size or the window size (the q factor cancels the loss's 1/(T·K)). Combines with
  `quantile_correction_unnormalized` (height 1 vs 1/width) and `quantile_correction_extended_window`.
- Config `quantile_correction_exact_margin` (added 2026-10-07): quantile-correction margins against the expert each
  pair would swap with, i.e. the runner-up for selected pairs and the token's weakest selected expert for unselected
  pairs (the `exact_margin` position), instead of against the runner-up for every pair. Each expert's window then
  holds exactly the |n_e − q| tokens that a bias on that expert alone would flip. Against the runner-up, each token's
  runner-up sits at margin exactly 0. An underloaded expert short by fewer tokens than it is runner-up for (about
  q/K) then gets δb_e = 0 and an empty window. In a toy at E=2048 ratios this hit 23–71% of underloaded experts, more
  near balance and with more tokens. It likely also explains the very large `quantile_correction/ste_height_max`
  values with `quantile_correction_mean_local`.
- Config `moe_quantile_estimator` (added 2026-10-07): how per-expert quantiles of the global batch are computed for
  `quantile_balancing` (`qb_beta`), quantile correction (δb_e) and the `fixed_number_boundary_ste` radius.
  - `legacy` (default): the previous estimate of each method. QB averages per-rank quantiles (Su/Dial); QC gathers
    every margin, or averages per-rank quantiles with `quantile_correction_mean_local`; fixed-N gathers every
    margin (about 4 GB per layer per step at E=2048).
  - `histogram`: Kimi K3 (tech report App. D). Per-expert histograms with `moe_quantile_histogram_bins` uniform
    bins (default 1000) are all-reduced, and the quantile is interpolated linearly inside the selected bin; the
    error is at most one bin width. For QB on sigmoid scores the bins span [min(qb_beta) − 1, max(qb_beta) + 1]
    as in Kimi K3; otherwise each expert's global min/max (one extra small all-reduce).
  - `exact`: Exact Quantile Balancing (arXiv 2609.28053). Two all-reduced 256-bin histograms over the bfloat16 bit
    pattern give the exact quantile of the bfloat16-rounded values.

  Both new estimators are partition invariant and all-reduce only counts. QB pools its margins over the step's
  microbatches and the data-parallel group at finalize (the bias applies from the next step); QC and fixed-N compute
  theirs in the forward pass over the load-balancing group.
- Config `moe_router_quantile_balancing_on_scores` (added 2026-10-07): quantile balancing on the activated scores
  instead of the logits. Top-k is taken on `score_function(logits) − qb_beta`, so `qb_beta` and
  `train/expert_bias/*` are in score units. With sigmoid this is Kimi K3's recipe. The combine weights stay the
  unbiased scores. Plain router only.
- Config `moe_load_balance_ste_post_activation` (added 2026-10-07): the STE load-balancing gradient enters at the
  normalized router scores π (the aux-loss scores: softmax probabilities, or sigmoid/sqrtsoftplus scores divided by
  their sum) instead of the logits. Which tokens get gradient still comes from the logit margins. With a margin
  kernel, the gradient goes to π_te − π_ref against the same threshold expert (π_te alone with
  `moe_load_balance_ste_detach_threshold`). With the full STE and `centered_fsq`, the gradient is exactly the aux
  loss's at top-k 2. Applies to `centered_fsq` (rect, triangle, tanh, higher_order_rect, full),
  `quantile_correction_ste` and `fixed_number_boundary_ste`. With a selection bias (DeepSeek or a selection-only
  learnable bias) added to the logits under softmax selection, the gradient enters at the softmax of the biased
  logits, so it also reaches the bias (changed 2026-10-10; before, the flag had no effect with any selection bias).
  With a bias added to scores (sigmoid, sqrtsoftplus, post-softmax expert bias) it still has no effect.
- Config `moe_lm_loss_ste_post_activation` (added 2026-10-14; split router): the normalized-relative LM STE
  (`moe_lm_loss_ste_normalized_relative`) enters at the normalized router scores π, the aux-loss scores, instead of
  the selection scores. π is softmax(z) for selection activation `none`/`softmax` and s/Σs for sigmoid/sqrtsoftplus;
  the selection scores are the logits, or s. The per-expert signal p_e·∇y·(h_e − y) is unchanged. Through the
  normalization it also reaches the unselected experts' logits.
  - Bias handling is that of `moe_load_balance_ste_post_activation`: a logit-space selection bias gives softmax(z + b);
    a score-space bias makes the flag a no-op.
  - Other LM STEs ignore it.
  - The runs before 2026-10-14 all had it off (LM pre at the raw logits).
- Config `moe_load_balance_ste_mass_normalized` (added 2026-10-07): every expert's STE kernel is scaled to a total
  mass of T, the tokens in the load-balancing batch (the full STE's mass). Each expert's total push is then the same
  for the same load error, whatever the kernel; the kernel only spreads it over tokens. Windowed kernels get height
  T / (tokens in the window). For quantile correction this replaces the height options (1/|δb_e|, unnormalized,
  count-normalized). Same load-balancing types as the post-activation option.

## Downstream tasks (validation sets added after "regular"; run at every eval; `--skip-task-eval` turns them off)
- `--task-eval-tasks dclm-core-22` means 20 lm-eval-harness tasks:
  - ARC-e/c, BoolQ, CSQA, COPA, HellaSwag, OBQA, PIQA, Winogrande, WSC273
  - LAMBADA, CoQA, SQuADv2, AGIEval-LSAT-AR
  - 6 BigBench tasks

  The prompts are zero-shot with the gold answer, at most 1000 examples per task
  (megatron/training/datasets/task_loss_dataset.py, tools/prepare_task_loss_datasets.py).
- `tasks/<task>`: token-weighted cross-entropy on the gold answer tokens. It is a loss, not an accuracy:
  lower is better.
- `tasks/average`: the unweighted mean of `tasks/<task>` over the tasks evaluated.
- `task_entropy/<task>`: router entropy on that task's batches (padding included), not LM entropy.

## test/* (once, at the end of training, on the test split)
- `test/lm_loss` (which also overwrites `lm loss validation` at that step), `test/router_entropy`,
  `test/aux_loss`, `test/router_values/*`, `test/router_balance/*`. There are no vio, ste, weighter or quantile keys.
