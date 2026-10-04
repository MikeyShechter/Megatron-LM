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
- `train/moe_router_bias_update_rate`: the DeepSeek expert-bias update rate. Logged only with
  `--moe-router-enable-expert-bias`.
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
- The train-time MaxVioGlobal is not logged; it only drives the coefficient controller.

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
- How to use them: check after the first evals of a sweep, e.g.
  `q.py curve ste/all_layers/window_frac_1r --sweep X --steps 150,300,600`.
  - An empty window (`window_frac_1r` × E ≲ 0.3), or `window_experts_empty_frac` above ~0.01, early in training
    means w is too small for the router's logit scale.
  - Comparing these keys across widths and variants tells you whether two runs had comparable windows.

## Logits, biases, weighter, quantile correction (regular validation only)
- `router_logits/top1|top2|avg_pre_activation`: per-token largest and second-largest logit, and the mean over all
  logits, before the activation. They include learnable `*_weight` biases but never the DeepSeek expert bias; with
  a split router they are the selection head's logits. `weighter_logits/*`: the same for the weighter's logits.
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
  tokens (from topk_plus_one margins); mean / max / min over (layer, expert). Logged only with
  `quantile_correction_ste`.
- `quantile_correction/affected_token_expert_fraction`: the fraction of (token, expert) pairs whose selection would
  flip under that shift.

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
