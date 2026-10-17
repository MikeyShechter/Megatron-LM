# Paper plan: settle the "necessary" items, then the "interesting" ones (2026-10-07)

## Goal and ground rules

- **The paper:** a load-balancing (LB) method in the every2 setting.
  - Setup: E=2048, K=2, MoE in every 2nd layer (6 MoE layers of 12), hidden size 768, Muon, 6000 iterations.
  - Batch: 524,288 tokens per step, so q = 512 assignments per expert per step.
  - Global LB, seed 1000, deterministic mode.
  - At fewer experts the methods looked alike, which is why we work here.
- **Order:** the "necessary" items (a fair comparison with baselines) come before conclusions about the "interesting"
  ones (split router-weighter, STE kernel, per-expert controller).
- **Every run is a full 6000-step run**, about 6 h × 16 GPUs ≈ 96 GPU-h. A half-length proxy was considered and
  rejected (Appendix C).
- **No seeds for now.** The seed also changes the data order, so when seeds are added, every config in a comparison
  gets the same seed set (Part 10).
- **Configs:** change one thing at a time.
  - A standard-router config is a copy of `configs/2048_experts_fd_base.yaml`.
  - A split-router config copies the split block of `configs/2048_experts_split_count_normalized_qc.yaml`: softmax
    weighter, router bias, `moe_lm_loss_ste_normalized_relative`.
  - Submission details are in the run-experiments skill.
- **Vocabulary:**
  - r_e = n_e/q − 1 is expert e's relative load error.
  - The "standard router" is the plain router without the split.
  - "Pre" and "post" mean the STE gradient enters at the logits z or at the normalized affinities π (softmax
    probabilities, or s/Σs for sigmoid and sqrtsoftplus).

## Part 0 — How runs are judged

- **Loss:** the final val loss, `val/regular/lm_loss` at step 6000. Test loss is a check.
- **Balance**, two numbers per run:
  - end of run: `vio/MaxVioGlobal`, mean of the last 3 evals;
  - whole run: `vio/MaxVioGlobal`, mean over all 40 evals, as reported in the ID Balancing paper.
  - Extras:
    - `val/router_balance/max_vio_dispatch_mean`: local balance, i.e. one rank's micro-batch with about 32 tokens
      per expert;
    - dead experts, `vio/ZeroLoadFracGlobal` (newer runs only).
- **Comparison:** each method is a curve in (loss, balance) over its λ or rate sweep. A method wins if either holds:
  - better loss at similar balance;
  - better balance (end of run or whole run) at similar loss.

  Winning on both is a plus, not a requirement. Until seeds exist, loss gaps under 0.003 count as similar.
- **Tooling (small):** add `maxvio_last3` and `maxvio_mean` to `SUMMARY_METRICS` in `tools/runsdb/common.py`, then run
  `tools/runsdb/build.py`. Until then, use the SQL in Appendix D.

## Part 1 — Code (no runs)

**Status: implemented on 2026-10-07; all options default off. Tested on CPU only (no GPU run yet).**

New config keys:

| key | what it does |
|---|---|
| `moe_quantile_estimator` | `legacy` (default) / `histogram` (Kimi) / `exact` (EQB); used by QB, QC δb, fixed-N radius |
| `moe_quantile_histogram_bins` | default 1000 |
| `moe_router_quantile_balancing_on_scores` | QB bias on the activated scores; Kimi's sigmoid recipe |
| `quantile_correction_exact_margin` | 1.2 |
| `moe_load_balance_ste_post_activation` | 1.3; with a selection bias added to the logits (softmax), the gradient enters at softmax(z + b) and reaches the bias (fixed 2026-10-10); no effect with a score-space bias |
| `moe_load_balance_ste_mass_normalized` | 1.4 |

The existing `moe_load_balance_ste_detach_threshold` is the detach switch.

Tests:
- `tests/unit_tests/transformer/moe/test_lb_quantile_estimators.py`: 17 CPU tests, plus a multi-rank test run with
  `torch.distributed.run`.
- With all new keys at their defaults, losses and gradients are bit-identical to the previous code for 87 LB
  configurations.
- A router-level CPU harness confirmed that identity + post equals the aux loss through the whole router.

### 1.1 Global per-expert quantiles

Write one shared routine:
1. scatter-add a per-expert histogram of a per-(token, expert) value;
2. one all-reduce of the E×B integer counts;
3. a per-expert cumulative search for the target rank.

Use it in two modes.

- **Kimi** (Kimi K3 tech report, §2.3.3 and App. D):
  - **Router:** sigmoid scores s = σ(z). The bias b is added to the sigmoid scores for Top-K selection only. Gates
    are the unbiased s, normalized over the selected experts.
  - **What is histogrammed:** the required bias r_te = τ_t − s_te, where τ_t is the token's (K+1)-th largest
    *biased* score (s + b).
  - **Bins:** B = 1000 uniform bins over [b_min − 1, b_max + 1], recomputed every step. Counts are scatter-added
    over all micro-batches, then one all-reduce per step.
  - **Readout:** take the first bin whose cumulative count reaches ⌈q⌉ and interpolate linearly inside it:
    b̂_e = b_min − 1 + (bin + clip((q − c)/h, 0, 1))·w.
  - **Update:** mean-center the biases and apply them at the next step. An EMA of the quantiles across steps is an
    optional refinement.
  - **Softmax:** the [b_min − 1, b_max + 1] range relies on sigmoid scores. For logit margins, use per-expert
    min/max of r, all-reduced with MIN/MAX.
  - No public code was found; App. D is detailed enough to implement from.
- **EQB** (Neitemeier et al. 2026, arXiv 2609.28053):
  - Run the same histogram twice over BF16 bit patterns: 256 high-byte bins, then 256 low-byte bins inside the
    selected one. This gives the exact global k-th value.
  - Cost: 2 × E × 256 int32 all-reduced per layer, about 4 MiB at E=2048.
- **Uses:**
  - the QB baseline, in both modes (Part 2);
  - QC's δb, replacing `quantile_correction_mean_local` and the all-gather path. QC uses whichever mode wins the QB
    comparison in Part 2;
  - the fixed-N radius, replacing an all-gather of about 4 GB per layer per step, so that fixed-N can run at E=2048
    at all.

### Implementation notes (things that are easy to get wrong)

- **Size of the work.** Each MoE layer has a 32k × 2048 margin matrix per rank (~67M entries) every step. All new
  code is dense ops on it plus small all-reduces; no raw margins are ever gathered.
- **Deterministic mode** (`torch.use_deterministic_algorithms(True)`, torch 2.11):
  - On CUDA, `torch.histc` and float `cumsum` raise errors.
  - `torch.bincount` without weights is allowed, and `scatter_add_` switches to a slower deterministic kernel.
  - So: build histograms with `bincount` over integer ids (expert·B + bin), and run `cumsum` on int64 counts.
- **Memory:** the int64 bin ids for 67M entries take ~0.5 GB, so process tokens in chunks. Post mode additionally
  keeps the full softmax for backward (~0.27 GB per layer in fp32).
- **Kimi's bin range** [b_min − 1, b_max + 1] only holds for sigmoid scores, s ∈ (0, 1).
  - Logit margins (softmax) have no such bound, so we have to choose a range ourselves: the global per-expert
    min/max, one small extra all-reduce.
  - With 1000 bins over ~20 logits, a bin is ~0.02 logits wide, coarser relative to the token density at the
    boundary than in Kimi's sigmoid setting. "Kimi on softmax" is our adaptation.
- **Reading the comparison:** the sigmoid pair differs in two things, since Kimi puts the bias on the sigmoid scores
  and EQB puts it on the logits. So the aggregation choice for QC is read from the softmax pair, where both use
  logits. The sigmoid pair supplies the faithful baselines.
- **Ties.** QB's margins have the same tie structure as QC's: each expert's biased-runner-up tokens sit exactly at
  m = β_e, about q/K tokens.
  - EQB returns exactly β_e when the target rank lands in that block, so the bias doesn't move.
  - Kimi's interpolation lands partway into the bin.
  - This may matter as much as resolution.
- **Off-by-one and sign:**
  - the repo subtracts its β (scores − β), while Kimi adds b;
  - it's the (q+1)-th largest margin vs the q-th smallest required bias;
  - interpolation must run in the matching direction.

  Unit-test both modes against an exact `kthvalue`.
- **BF16 exactness (EQB):** the quantile is exact for BF16-rounded margins. The window and selection tests must use
  the same rounded values, otherwise counts are off by the tokens that round onto the threshold.
- **Process group and target:**
  - histogram all-reduces use the same group as the global loads (`tp_dp_cp_group` with `moe_use_global_lb`, all
    16 ranks);
  - q is the global target (512).

  A mismatch gives a silently wrong quantile.
- **Timing:**
  - QB accumulates over the step's micro-batches, all-reduces at finalize, and applies the bias at the next step.
    Accumulators are reset every step and untouched during eval.
  - For QB-EQB, keep the BF16 margins (~134 MB per layer) until finalize, so both passes for all layers take 2
    batched all-reduces per step.
  - QC needs δb in the same forward: per MoE layer, 2 all-reduces in the forward for EQB, or 1–2 for Kimi (range,
    then counts).
- **Loss scaling:** the unified STE path must attach its loss through `attach_and_log_load_balancing_loss` /
  `MoEAuxLossAutoScaler`, like aux. The identity + post test should go through the router, not just the loss
  function, so that a micro-batch or DP-size factor can't slip in.
- **Speed:** the concern is a large slowdown from communication overhead (e.g. 2×), not a few percent. Compare the
  average `iteration-time` with the existing QB runs (median 3.18–3.21 s) and the other baselines (3.10–3.37 s).
  That is enough; no extra timers.
- **Adopting Kimi for QC** (moot: EQB won Part 2): its interpolated quantile puts approximately |n − q| tokens in each window, off by up to
  one bin. With mass normalization the total push stays exact; only the membership is approximate. If Kimi wins the
  QB comparison, log window count / |n − q| before adopting it for QC.

### 1.2 QC margin

Use `exact_margin` instead of the hard-coded `topk_plus_one` in `_quantile_correction_load_surrogate` (moe_utils.py).
Make it a flag so old runs stay reproducible. Appendix F has the reason and the toy numbers.

### 1.3 One STE path for every kernel

The surrogate load is n_e ≈ Σ_t k_te·u_te.
- **k_te** is a stop-gradient kernel computed from logit-space margins (identity, rect, QC, fixed-N, extended). It
  decides which tokens carry expert e's signal. Supports stay in logit space even when the gradient enters at π:
  - rect widths keep their logit units;
  - QC's window is the set of tokens a *logit* bias would flip, the space where the bias baselines worked best for
    softmax.
  - With π-space margins the window would differ even with an exact quantile, because π-margins ≈ z-margins × a
    token-dependent probability, which reorders tokens.
- **u_te** is the variable the gradient flows into: z (pre) or π (post).
- **Detach switch**, kept for the Part 4c experiment and fixed afterwards (4c: detached is the default; the runs tied,
  so this may be revisited):
  - detached: u_te;
  - attached: u_te − u_ref, where ref is the margin's competitor (the runner-up for selected pairs; the token's K-th
    expert for unselected pairs, with exact_margin).
- **Per-expert signal:** keep the centered load error, ∝ r_e. Appendix A explains why centering matters.

### 1.4 Mass normalization

- **Height:** each expert's window gets height T/|W_e|.
  - |W_e| is the number of affected tokens, all-reduced.
  - T is the number of tokens in the LB batch; it's only a unit.
  - The identity kernel then has height 1, and identity + post equals the aux loss at the same λ.
- **Relation to existing code:** this is `quantile_correction_count_normalized` with q replaced by T, extended to
  every kernel.
- **Effect:** every kernel gives expert e the same total push, ∝ λ·r_e. The kernel only spreads it over tokens.
  The remaining dependence on balance is the explicit r_e, which Part 9 replaces.

### 1.5 Unit tests

- Identity + post gives the same gradient as `switch_load_balancing_loss_func` at K=2: softmax, and sigmoid with
  normalized scores.
- Identity + post with the uncentered f_e gives the same gradient as with the centered r_e. For windowed kernels the
  two differ; this documents Appendix A.
- With exact_margin and the exact quantile, every QC window holds |n_e − q| tokens, and δb ≠ 0 for every imbalanced
  expert.
- With detach on, every in-window gradient has the sign of its expert's load error (overloaded: down only;
  underloaded: up only).
- The Kimi histogram lands within one bin of the exact value. EQB equals an all-gather `kthvalue` (multi-rank test).
- With the new flags off, gradients are bit-identical to today.

## Part 2 — Baselines (parallel track) — done 2026-10-08

**Results:**

| run | run id | val loss | MaxVio end / whole run | avg iteration time |
|---|---|---|---|---|
| QB-Kimi, softmax | tbj4grul | 3.2106 | 1.26 / 1.61 | 3.25 s |
| QB-EQB, softmax | jrvtxfq7 | 3.2114 | **0.99 / 1.50** | 3.34 s |
| QB-Kimi, sigmoid (as in K3) | icv32040 | 3.2173 | 1.76 / 2.12 | 3.35 s |
| QB-EQB, sigmoid (as in the EQB paper) | c7mgfh3a | 3.2174 | **1.32 / 1.90** | 3.27 s |
| QB rank-averaged, softmax | x3e8nqq3 | 3.2105 | 1.44 / 1.82 | 3.21 s |
| QB rank-averaged, sigmoid | 59r9075q | 3.2180 | 1.86 / 2.27 | 3.18 s |
| ID, softmax, rate 0.6 | 3dzy5txz | 3.2660 | 367 / 344 (collapsed) | 3.33 s |

- **EQB wins** both pairs: the loss gaps are ≤ 0.001, and EQB balances better on both numbers. QC's δb uses
  `moe_quantile_estimator: exact`.
- **Global vs rank-averaged:** both global estimators balance better than the rank-averaged QB at the same loss. The
  paper's QB baseline is EQB softmax, with Kimi softmax reported next to it.
- **Speed:** 1–5% slower than the rank-averaged QB. Neither estimator is consistently slower (EQB is faster with
  sigmoid, slower with softmax), so the difference is node variation. No drastic slowdown.
- **ID, rate 0.6:** MaxVio climbs from 30 at step 150 to 130–470 for the rest of the run. This is the same failure as
  sigmoid at 10× its best rate (3t7ba3a9: rate 0.06, MaxVio 389, 3.2672). Softmax's best rate stays 0.06 (6c549raw:
  3.2117, 1.00 / 1.89), now bracketed by 0.006 (3.2206) and 0.6.

**Setup:**

| run | why |
|---|---|
| QB-Kimi, sigmoid, bias on the sigmoid scores | faithful Kimi K3 |
| QB-Kimi, softmax, bias on the logits | softmax is QB's best activation here |
| QB-EQB, sigmoid, bias on the logits | faithful EQB |
| QB-EQB, softmax, bias on the logits | |
| ID, softmax, rate 0.6 | the best rate so far (0.06) was at the sweep edge |

- **5 runs.**
  - The ID run was submitted on 2026-10-07: sweep `26-10-07-experts2048_every2_id_balancing_softmax_high_rate`,
    job 2213091, config `configs/2048_experts_id_balancing_softmax_high_rate.yaml`.
  - The 4 QB runs were submitted on 2026-10-07: sweep `26-10-07-experts2048_every2_qb_estimators`, config
    `configs/2048_experts_qb_estimators.yaml`. Run dirs: 000 Kimi sigmoid (job 2218934), 001 EQB sigmoid (2218935),
    002 EQB softmax (2218936), 003 Kimi softmax (2218937).
- **The 4 QB runs decide two things:**
  - report loss and MaxVio (end of run and whole run) for Kimi vs EQB;
  - check from the average iteration time that neither is drastically slower (e.g. 2× from communication
    overhead).
  - No smoke run before these runs (user's choice, 2026-10-07).
  - The winner, read from the softmax pair (see the implementation notes), also computes QC's δb.
- The existing rank-averaged QB runs (Su/Dial: x3e8nqq3, 59r9075q, le6xhgsa) stay as the ablation.
- **Not run:** sqrtsoftplus, which was worse for every method here, and LEI (Appendix A).
- **Rule:** report each baseline at its best activation and rate. Where our version differs from the paper's, report
  both and use the better one as the baseline. Example: DeepSeek softmax with the bias on the logits got 3.2102,
  the paper's post-softmax version 3.2166.

## Part 3 — Activation: softmax (decided, no runs)

- **Evidence (Appendix B):**
  - every baseline gets its best loss with softmax, by 0.005–0.02, and equal or better balance on at least one of
    the two balance numbers;
  - with the split router, the softmax weighter is better for aux and tied for QC;
  - sigmoid did better only for pre-activation STEs on the standard router (QC, rect), which Part 4 changes;
  - the full STE was only ever run with softmax.
- **If** post-QC on the standard router (Part 4b) is unstable at every λ with softmax, **then** test sigmoid there
  (normalized affinities, 2 runs) before dropping softmax. Not triggered: post-QC works with softmax (4b).
- One sigmoid run of the final method at the end, so the paper can say it was checked.

## Part 4 — QC fundamentals on the standard router

Why the standard router first:
- every baseline lives there;
- it is the harder case: today's softmax QC collapses there, and the full STE works there but not in the split;
- it isolates the LB signal from the split's extra parts.

The two routers have reacted very differently to the same LB signal before (Appendix E), so the winner is confirmed
in the split afterwards (Part 5).

### 4a. Margin check — done 2026-10-08

As soon as 1.2 exists, 2 runs: the two best existing QC configs with only the margin changed to exact_margin.
- **Split:** pj2yuna0 (softmax weighter, router bias, λ=0.1): 3.2186, end 0.48 / whole run 1.07.
- **Standard router:** kcw5t9fv (sigmoid, λ=0.01): 3.2383, end 0.94 / whole run 2.31. It is the only working
  standard-router QC config without a bias.
- Submitted on 2026-10-07. The specs equal the references' except for exact_margin and the data cache path.
  - Split: sweep `26-10-07-experts2048_every2_split_qc_exact_margin`, job 2218939.
  - Standard router: sweep `26-10-07-experts2048_every2_qc_exact_margin`, job 2218938.
  - **Code since the references:**
    - Standard router: today's code computes the same numbers as kcw5t9fv's code (code read, 2026-10-07). The QC
      backward was rewritten after 2026-10-04 but gives the same 1/|δb| on the same window. The other changes are off
      by default or only log.
    - Split: since 2026-09-27 (e8653d901), a fused kernel computes the weighter logits for the selected experts
      only. The math is the same but the rounding differs, so the run drifts from pj2yuna0 for reasons other than
      exact_margin, by an amount never measured.
    - **If** the split result is within ~0.003 of pj2yuna0, or its balance is close, **then** rerun pj2yuna0 on
      today's code. That run also measures how much rounding alone moves a run.

**Result (2026-10-08): keep exact_margin.**

| | run id | val loss | MaxVio end / whole run |
|---|---|---|---|
| standard router, topk_plus_one | kcw5t9fv | 3.2383 | 0.94 / 2.31 |
| standard router, exact_margin | 86qm8xye | 3.2339 | 1.00 / 5.52 |
| split, topk_plus_one | pj2yuna0 | 3.2186 | 0.48 / 1.07 |
| split, exact_margin | mayjuzg6 | 3.2167 | 0.45 / 1.18 |

- **Standard router:** the loss is 0.0044 better and the end balance is the same.
  - The worse whole-run number comes from the first eval alone: MaxVio 162 at step 150 vs 35. From step 300 on the
    curves are alike.
  - The early loss lagged (4.22 vs 4.04 at step 600) and caught up by step 3000.
- **Split:** within noise on loss and balance. The pj2yuna0 rerun above would only tell how much of the 0.0019 is
  rounding drift; the decision doesn't need it, so it was not run.
- **Heights:** `quantile_correction/ste_height_max` still reaches 5e4–2.5e5 in both exact_margin runs. The 1/|δb|
  height itself causes them: a nearly balanced expert has a tiny |δb|. Mass normalization (4b) replaces this height.

Decision:
- **If** exact_margin is not worse (loss within 0.003, similar balance), **then** keep it; it is the correct
  per-expert flip set.
- **If** it is clearly worse, **then** the runner-up handoff through the attached reference was doing useful work.
  Look at which experts lose their signal before going on.

### 4b. Pre vs post — done 2026-10-08: post

**Results:**

| gradient enters at | λ | run id | val loss | MaxVio end / whole run | local MaxVio, end |
|---|---|---|---|---|---|
| **post** | **0.1** | 8cvnm4ny | **3.2126** | **0.66 / 1.13** | 3.78 |
| post | 1 | ftiqvhnn | 3.2226 | 0.54 / 1.17 | 2.83 |
| post | 10 | gl0sb4qr | 3.2664 | 0.55 / 1.13 | 2.07 |
| pre | 1e-4 | chsya0yh | 3.2135 | 4.47 / 5.20 | 7.77 |
| pre | 1e-3 | tec3dm4o | 3.2197 | 0.57 / 0.91 | 2.88 |

Baselines on this router: EQB QB 3.2114, 0.99 / 1.50; Kimi QB 3.2106, 1.26 / 1.61; ID (0.06) 3.2117, 1.00 / 1.89;
DeepSeek (0.1) 3.2102, 2.82 / 4.35; aux λ=0.1 / 1 / 10: 3.2322 (11.8 / 14.9), 3.2206 (1.61 / 2.24), 3.2322
(1.10 / 1.87).

- **Post wins.** Neither pre run beats post λ=0.1:
  - pre λ=1e-4 reaches about the same loss with 7× worse balance;
  - pre λ=1e-3 balances a little better but loses 0.007 in loss.
  - At the tightest balance (end ~0.55), pre 1e-3 is 0.003 better than post λ=1, right at the threshold.
- **Against the baselines:** post λ=0.1 is within 0.001–0.0024 of every baseline's loss and balances clearly better
  (end 0.66 vs 0.99–2.82, whole run 1.13 vs 1.50–4.35, local 3.8 vs 5.2–6.3). That is a win by the Part 0 rule,
  until seeds exist. The DeepSeek gap (0.0024) is the closest to the threshold.
- **Against aux** (the identity member, same normalization): better loss by about 0.008 at every λ, with better
  balance. The flip-set support is what helps.
- Above λ=1, balance saturates (end ~0.55) and only the loss gets worse.
- No early spike like 4a's. Dead experts 0.04–0.07% (QB, ID: 0.1%). Iteration time 3.27–3.40 s.
- **Edge run, done 2026-10-08:** post λ=0.01 (5el7u2rw, sweep `26-10-08-experts2048_every2_qc_pre_post_low_lambda`,
  config `configs/2048_experts_qc_pre_post_low_lambda.yaml`): 3.2165, MaxVio 49.2 / 35.6, local 51.8. It never
  balances, and the loss is worse too. Pre needs no extension: both of its runs already lose to post's best.
- **Normalization check: passes.** Post-QC's best λ is 0.1, one 10× step from aux's best (1). Every later kernel starts
  at λ ∈ {0.1, 1, 10}.

**Setup:**

5 runs. Standard router, softmax, corrected QC: exact margin, EQB quantile, mass normalization, attached.

| | λ |
|---|---|
| post | 0.1, 1, 10 |
| pre | 1e-4, 1e-3 (the split count-normalized runs' best λ of 0.1–1, converted to mass T by ×K/E) |

Submitted on 2026-10-08: sweep `26-10-08-experts2048_every2_qc_pre_post`, config `configs/2048_experts_qc_pre_post.yaml`.
- Run dirs: 000 post λ=1 (job 2225487), 001 post λ=0.1 (2225488), 002 pre λ=1e-4 (2225489), 003 post λ=10
  (2225490), 004 pre λ=1e-3 (2225491).
- These are the first GPU runs of mass normalization, the post path, and the exact quantile inside QC's forward.

Decision:
- **If** post is similar or better, **then** use post for every STE kernel. Aux is then the identity member
  (Appendix A), and its runs are reused.
- **If** post is clearly worse, **then** compute the windows from π margins once (3 runs). **If** that still loses,
  **then** use pre everywhere and present aux separately.
- **If** either best λ is at an edge, **then** add one run 10× beyond it.
- **Normalization check:** **if** post-QC's best λ is within 10× of aux's (λ=1 on this router), **then** the
  normalization works, and every later kernel starts at λ ∈ {0.1, 1, 10}.

### 4c. Detach — done 2026-10-08: detach, chosen on theory because the runs tied

**Result:**

| post-QC, λ=0.1 | run id | val loss | MaxVio end / whole run | local MaxVio, end | dead expert slots at 6000 |
|---|---|---|---|---|---|
| attached | 8cvnm4ny | 3.2126 | 0.66 / 1.13 | 3.78 | 9 of 12,288 |
| detached | vf44gjaf | 3.2129 | 0.65 / 1.08 | 3.60 | 4 of 12,288 |

- **A tie:** the loss and MaxVio curves overlap throughout. The dead-expert difference is too small to count.
- **Decision (user, 2026-10-08): detach is the default**, chosen on theory, not on data:
  - **Attached is the exact derivative only with two-sided windows.** It is the chain rule through the competitor's
    logit, which is right when both sides of every boundary are in windows of equal height: e's cross term (−g_e on
    j) then pairs with j's own term (+g_j) into the exchange value g_j − g_e. QC's windows are one-sided per expert
    (overloaded: only selected tokens; underloaded: only unselected ones). So the competitor's own term is usually
    missing, and attached can push it the wrong way: an overloaded runner-up up, an underloaded K-th expert down.
  - **Detach matches QC's window:** the window is the flip set when only e's bias moves; detach is the gradient of
    that move. The handoff still happens through the argmax, and through the softmax Jacobian.
  - **Mass normalization's guarantee and Part 9 need detach:** with detach, each expert's probabilities get a total
    push ∝ λ·r_e, in the direction of its own error. Attached adds pushes driven by neighbours' errors.
  - **Detach keeps aux in the family:** identity + post (= aux) has no competitor, so it is already detached. Every
    kernel is then aux's gradient restricted to a support and mass-normalized.
- **Because the runs tied, this may be revisited later**, for example with seeds, or with a kernel whose windows are
  two-sided. The paper reports attached as an ablation with the same result.
- **Constraint (Part 5 ablation, 2026-10-09): detach requires mass normalization.** With the old 1/|δb| height it
  collapsed in the split: MaxVio 513, a third of the expert slots dead, STE height up to 3.8e6. Detach removes the
  competitor's counter-push, and 1/|δb| is unbounded for nearly balanced experts. Mass normalization bounds each
  expert's push. A variant without mass normalization must not use detach.

**Setup:**

1–2 runs: the 4b winner at its best λ with detach on, plus the neighbouring λ if the result is close.
- Ran post λ=0.1 with detach: sweep `26-10-08-experts2048_every2_qc_detach`, config `configs/2048_experts_qc_detach.yaml`.
  The spec differs from 8cvnm4ny only in the detach flag. No λ=0.01 detach run was needed (λ=0.01 lost).
- **If** detach is not worse, **then** it becomes the default for every margin kernel. It enforces the up/down rule
  by construction. Don't re-test it per variant.
- This replaces the 8-run detach proposal of 2026-10-07. Six of those eight were QC height and window variants that
  the normalization retires.

### 4d. Isolated re-checks: height, activation with post, detach on a second kernel — planned 2026-10-14

**Why (user, 2026-10-14):** three earlier decisions rest on comparisons that change more than one thing. Each run below
differs from an existing run in one key, as in the margin table (4a). Where the height changes, λ also changes,
because the height sets λ's scale.

- **Height (the Normalization tab).** The comparisons behind it are confounded:
  - The split rows are pre, rank-averaged δb and attached, with mixed margins.
  - The standard-router rows compare mass (softmax, exact margin, EQB, λ=1e-3) against 1/|δb| (sigmoid, old or exact
    margin, rank-averaged, λ=0.01). With softmax, 1/|δb| was only ever run with the old margin.
  - The rect rows compare mass (post, detached) against 1/w (pre, attached).
  - **The only isolated pair** is on the split router (pre, exact margin, rank-averaged, attached): 1/|δb| at λ=0.1
    (mayjuzg6: 3.2167, 0.45 / 1.18) against mass at λ=1e-3 (d2qw4ory: 3.2207, 0.55 / 1.21). That is 1/|δb| ahead by
    0.004, with one λ each. Mass normalization was chosen on theory (one λ scale for every kernel, as the identity's)
    and because detach needs it (4c).
- **Activation.** On the standard router every non-softmax STE run is pre (old recipe), and the activations can behave
  differently with post. The identity kernel (aux) already has sigmoid runs. The QC sigmoid run below is also Part 3's
  planned "one sigmoid run of the final method".
- **Detach.** Only two attached/detached pairs exist: QC, standard, post, mass (a tie), and the split's old recipe
  (detach collapsed). 4c chose detach on theory and named "a kernel whose windows are two-sided" as the case to
  revisit. Rect and symmetric QC (7b) are such kernels.

**Runs** (standard router; post, exact margin, EQB as in the reference runs). Config: `configs/2048_experts_holes.yaml`,
8 runs, revised after the smoke run (below the table).

| hole | run | compared to | λ |
|---|---|---|---|
| activation | QC, sigmoid router | vf44gjaf (softmax): router score function | 0.1 |
| detach | rect w=1, mass, attached | 7sxgd69b (detached): detach | 0.1 |
| height, rect, attached | rect w=1, 1/w height, attached | the attached rect run above (new): height | 0.1, 1 |
| height, rect, detached | rect w=1, 1/w height, detached | 7sxgd69b (mass, detached): height | 0.1, 1 |
| height, QC | QC, 1/\|δb\| height, attached | 8cvnm4ny (mass, attached, λ=0.1): height | 0.1, 1 |

- **Attached or detached for the height pairs.**
  - QC stays attached. 4c's constraint says a variant without mass normalization must not use detach: 1/|δb| +
    detach collapsed in the split, because 1/|δb| is unbounded. 8cvnm4ny, the attached mass reference, ties vf44gjaf.
  - Rect is compared both ways (user, 2026-10-16). 1/w is a constant height, so the constraint's reason doesn't apply.
- **λ for the 1/w and 1/|δb| heights: 0.1 and 1 (user, 2026-10-16).** My first estimate matched each height to mass
  normalization by the total push per expert later in training: λ ≈ 6.6 for 1/w, 25–50 for 1/|δb|. The smoke showed
  that early training sets the scale instead:
  - The router logits start nearly equal, so the margins and δb are tiny and the 1/|δb| heights are huge.
  - Nearly every token is inside a w=1 window, so 1/w pushes about as hard as mass normalization.
  - At λ ≥ 10 both blew up within 150 steps.
- **Symmetric QC's detach check** is in 7b: attached and detached at λ 0.1 and 1, alongside its λ search.
- **Submitted 2026-10-16:** sweep `26-10-16-experts2048_every2_holes`, jobs 2349452–2349459.

  | run dir | run | λ |
  |---|---|---|
  | 000 | QC, sigmoid router | 0.1 |
  | 001 | rect w=1, mass, attached | 0.1 |
  | 002 | rect w=1, 1/w, attached | 0.1 |
  | 003 | rect w=1, 1/w, attached | 1 |
  | 006 | rect w=1, 1/w, detached | 0.1 |
  | 007 | rect w=1, 1/w, detached | 1 |
  | 004 | QC, 1/\|δb\|, attached | 0.1 |
  | 005 | QC, 1/\|δb\|, attached | 1 |
- **Split router height pair, submitted 2026-10-16:** the best split QC run, gt9tbg17 (post, 1/|δb|, exact margin,
  rank-averaged δb, attached, λ=10: 3.2086, 0.70 / 1.63), had no mass-normalized run that differs only in the height.
  Its partner is sweep `26-10-16-experts2048_every2_split_qc_post_mass` (config
  `configs/2048_experts_split_qc_post_mass.yaml`, jobs 2356896–2356897): the same spec with mass normalization, run
  dir 000 at λ=0.1 and 001 at λ=1.
- **Smoke run (2026-10-14, 300 iterations; deleted 2026-10-16).** What changed because of it:
  - λ for the 1/w and 1/|δb| heights went from {1, 10} and {10, 100} to {0.1, 1}: at λ ≥ 10 both blew up within 150
    steps.
  - The rect height pair is also run detached (1/w against 7sxgd69b), because attached rect broke even with mass
    normalization at λ=0.1: the router logits ran away by step 100. The attached runs stay, as full runs for the
    result.
  - Nothing changed for symmetric QC and QC with a sigmoid router: both trained normally.

**Decision rules (Part 0 rule throughout):**
- **Height:** **if** 1/|δb| at its best λ beats mass (both attached), **then** the default height is reopened, as
  1/|δb| + attached against mass + detach (vf44gjaf). Otherwise mass stays, now on data. The rect pairs (attached and
  detached) only inform the paper's normalization claim.
- **Activation:** **if** QC sigmoid beats softmax, **then** Part 3 is reopened for the method. Otherwise softmax stays,
  and the paper reports sigmoid with post as checked.
- **Detach:** **if** rect attached beats detached, **then** that supports 4c's theory (attached is the exact derivative
  with two-sided windows), and symmetric QC's check decides it for the method. A tie leaves detach as is.

**What made rect work (user's question, 2026-10-14).** The corrected recipe changed rect in three ways at once:
pre → post, 1/w → mass and attached → detached. The exact margin alone changed nothing (4 margin pairs). No run
separates them, so it is not known which one did it.
- Old rect: 12 runs on the standard softmax router (pre, 1/w, attached; w ∈ {1, 3}; λ 1e-4 … 0.1; both margins). None
  balanced; the best ended at MaxVio 3.98 (w=3, λ=1e-3, topk margin). At λ=0.1 the loss broke (3.89) and MaxVio stayed
  near 10.
- The new runs change one thing at a time between the old recipe and the corrected one:

  | pre / post | height | attached / detached | runs |
  |---|---|---|---|
  | pre | 1/w | attached | old: never balanced |
  | post | 1/w | attached | 4d, λ 0.1 and 1 |
  | post | mass | attached | 4d, λ=0.1 (broke in the smoke run) |
  | post | 1/w | detached | 4d, λ 0.1 and 1 |
  | post | mass | detached | 7sxgd69b: 3.2164, 1.18 / 2.66 |

  λ is re-matched where pre changes to post, because pre and post differ about 100× in λ's scale (QC: 1e-3 against
  0.1). The smoke run already points at detach: with mass normalization and post, attached rect broke at λ=0.1.
- **fixed-N** was never run with the old recipe in this setting: its only older runs are at E=256 and E=64 (August).
  So nothing shows that the corrected recipe "made it work"; it was only ever run with it.
- **QC** needed neither post nor detach. With softmax, the old QC (pre, 1/|δb|, topk+1 margin, rank-averaged δb,
  attached) never balanced at any λ from 1e-5 to 1 (best end 22). Pre + mass + exact margin + EQB, still attached,
  balanced at λ=1e-3 (tec3dm4o: 3.2197, 0.57 / 0.91). In the split, the isolated height pair balances with both
  heights, so the margin, whose old version gave many underloaded experts δb_e = 0 (Appendix F), is what mattered
  there. On the standard softmax router there is no pre + exact margin + 1/|δb| run, so the height's share there is
  open.

## Part 5 — Confirm in the split router — done 2026-10-09 (with ablation)

**Results:**

| run | run id | val loss | MaxVio end / whole run | local MaxVio, end | dead slots |
|---|---|---|---|---|---|
| corrected QC, post λ=0.1 | iwfdgrhg | **3.2118** | 0.79 / 1.87 | 3.06 | 4 |
| corrected QC, post λ=1 | fbxn16wa | 3.2292 | 1.03 / 2.14 | 2.39 | 2 |
| corrected QC, post λ=10 | 8dgrbz7h | 3.2529 | 0.79 / 2.18 | 2.14 | 1 |
| corrected QC, pre λ=1e-3 | ocuc2x7t | 3.2256 | 0.66 / 1.62 | 2.37 | 8 |
| full STE (pre), λ=1e-4 | uddvx5eh | 3.4270 | 325 / 304 (collapsed) | 326 | 3979 |
| full STE (pre), λ=1e-3 | x44pf1tk | 3.2122 | 1.38 / 8.12 | 4.60 | 18 |
| ref: aux λ=1 (identity, post) | 0nl5u4q1 | 3.2079 | 1.21 / 2.60 | 4.08 | – |
| ref: aux λ=0.1 / 10 | z9rg2k7y / naqhljbr | 3.2174 / 3.2133 | 1.36 / 3.98 · 1.30 / 2.97 | – | – |
| ref: old QC, exact margin | mayjuzg6 | 3.2167 | **0.45 / 1.18** | 2.89 | 17 |
| ref: old QC | pj2yuna0 | 3.2186 | 0.48 / 1.07 | 2.92 | – |

Dead slots are out of 6 layers × 2048 experts.

- **Post vs pre: the split agrees with 4b**, so both routers keep the same defaults.
  - QC: post λ=0.1 beats pre λ=1e-3 by 0.014 in loss, at somewhat looser balance.
  - Identity kernel: aux (post) beats the full STE (pre) on loss and on both balance numbers.
  - The full STE works in the split for the first time at λ=1e-3, though with a MaxVio spike of 65 at step 150. It
    collapses at λ=1e-4 and at λ ≥ 0.01.
- **Against aux:** 0.004 worse than aux λ=1 in loss, with better balance, so neither wins. Better than aux λ=0.1 and
  λ=10 on both numbers.
- **Against the old QC (mayjuzg6): better loss (−0.005) but clearly worse balance.**
  - In the stable phase (steps 2400–4200), MaxVio is 1.8–3.0 against 1.0–1.1; at step 6000 it is 0.71 against 0.40.
  - Raising λ to 1 or 10 costs loss without better balance.
  - On the standard router the corrected recipe beat the old QC on both numbers; in the split it doesn't.
  - Pre with the new recipe also balances worse than the old QC, so post alone is not the cause. The suspects are EQB,
    mass normalization (push ∝ r_e instead of ∝ r_e × boundary density) and detach.

**Follow-up, done 2026-10-09:**
- **Edge run:** corrected QC, post λ=0.01 (duv7kpus; sweep `26-10-09-experts2048_every2_split_corrected_qc_low_lambda`,
  config `configs/2048_experts_split_corrected_qc_low_lambda.yaml`): 3.2165, 1.06 / 7.37. Worse than λ=0.1 on both, so
  0.1 is the split's best λ too, and the edge is closed.
- **Ablation from mayjuzg6, one ingredient of the corrected recipe per run.** Sweep
  `26-10-09-experts2048_every2_split_qc_ablation`, config `configs/2048_experts_split_qc_ablation.yaml`. Each spec
  differs from mayjuzg6's only in its ingredient (and λ where the scale changes):

  | ingredient added to mayjuzg6 | λ | run id | val loss | MaxVio end / whole run | MaxVio, steps 2400–4200 | dead slots |
  |---|---|---|---|---|---|---|
  | none (mayjuzg6) | 0.1 | mayjuzg6 | 3.2167 | 0.45 / 1.18 | 1.10 | 17 |
  | EQB δb (instead of rank-averaged) | 0.1 | 0e94cmz9 | 3.2158 | 0.49 / 1.14 | 1.06 | 11 |
  | mass normalization (instead of the 1/\|δb\| height) | 1e-3 | d2qw4ory | 3.2207 | 0.55 / 1.21 | 1.19 | 13 |
  | post | 10 (×100) | gt9tbg17 | 3.2086 | 0.70 / 1.63 | 1.68 | 10 |
  | detach | 0.1 | 0b4o9g03 | 3.4528 | 513 / 224 (collapsed) | 187 | 4343 |
  | all four = corrected QC | 0.1 | iwfdgrhg | 3.2118 | 0.79 / 1.87 | 2.10 | 4 |

  - **EQB:** neutral.
  - **Mass normalization:** a small balance and loss cost, at a single λ.
  - **Post:** most of the balance change, with the largest loss gain. It moves along the trade-off, as on the standard
    router.
  - **Detach:** collapses with the unbounded 1/|δb| height (see the constraint in 4c).
  - **Old QC + post (gt9tbg17)** is the best split QC run: about the same loss as split aux λ=1 with much better balance,
    and better than the corrected QC on both numbers. It is not a coherent configuration, though: its height exists
    only for QC, is unbounded, and is unsafe with detach. It goes in the paper as an ablation.
- **Decision (user, 2026-10-10): the configuration stays as it is on both routers.** Softmax, post, exact margin, EQB
  quantile, mass normalization, detach. No combination search across routers; a choice changes only for a principled
  reason.
- **For Parts 7 and 8:** in the split, the corrected QC is not a clean win over aux λ=1 (its loss is 0.004 worse, its
  balance better). The standard router is the clear win.

**Setup:**

6 runs:
- **Corrected QC** with the Part 4 choices: λ ∈ {0.1, 1, 10} (3 runs).
- **The variable that lost in 4b**, at its best converted λ (1 run), to see whether the split agrees.
- **Identity-pre (full STE) calibration:** λ ∈ {1e-4, 1e-3} (2 runs). The earlier split full-STE sweep (0.01–1) sat
  10–1000× above the scale where aux works (aux λ=1 corresponds to a full-STE λ of about 5e-4–1e-3). So the split's
  pre/post comparison for the identity kernel isn't calibrated yet.

**If** the split prefers a different choice than 4b, **then** keep per-architecture defaults and say so in the paper.

Submitted on 2026-10-08 (split block of `configs/2048_experts_split_count_normalized_qc.yaml`, as pj2yuna0):
- Sweep `26-10-08-experts2048_every2_split_corrected_qc`, config `configs/2048_experts_split_corrected_qc.yaml`.
  Corrected QC with detach. Run dirs: 000 pre λ=1e-3 (job 2239332), 001 post λ=10 (2239333), 002 post λ=1
  (2239334), 003 post λ=0.1 (2239335).
  - The spec differs from the 4c standard-router run (vf44gjaf) only in the split keys, plus λ and pre/post.
- Sweep `26-10-08-experts2048_every2_split_full_ste_calibration`, config
  `configs/2048_experts_split_full_ste_calibration.yaml`: 000 λ=1e-4 (job 2239337), 001 λ=1e-3 (2239338).
  - Same LB and split keys as the 26-10-03 split full-STE run with router bias (zpwbqfdp); only λ differs.
  - That 26-10-03 sweep never balanced: MaxVio 76–370; best loss 3.2604 at λ=0.01 with router bias.
- **Split references:**
  - split aux λ=1: 3.2079, MaxVio 1.21 / 2.60;
  - split QC, old margin (pj2yuna0): 3.2186, 0.48 / 1.07;
  - split QC, exact margin (mayjuzg6): 3.2167, 0.45 / 1.18.

## Part 6 — λ (a rule, no runs of its own)

- Sweep in 10× steps, and extend whenever the best value is at an edge.
- With mass normalization and post, start every kernel and controller at λ ∈ {0.1, 1, 10}.
- **Adjusted after Parts 4 and 5 (2026-10-10):**
  - The best λ was 0.1 for QC on both routers and 1 for the identity kernel (aux) on both.
  - λ=10 cost 0.03–0.05 in loss for QC without better balance, and λ=0.01 failed to balance.
  - So a new kernel starts at λ=0.1. It gets λ=1 when its support is closer to the identity's (all tokens) than to
    QC's, or as the neighbour at its best variant. λ=10 and 0.01 run only if the edge rule asks for them.
- The bias-rate baselines are bracketed once Part 2 is done.

## Part 7 — Which tokens carry the signal (interesting #2) — done 2026-10-10: QC

Shared across kernels: post, mass normalization, exact margin, detach as decided in 4c, softmax, global quantiles. The
kernels differ only in their support.

**Moved to the standard router (user, 2026-10-10).** The reasons are the same as for Part 4: every baseline lives there,
it isolates the LB signal, and it's where the method is a clean win. It also avoids the split → standard back-and-forth:
the LB method is decided on the standard router (Parts 4 and 7), and the split is studied once, in Part 8. The λs
follow the Part 6 adjustment; they hold on both routers.

| kernel | tokens per expert | standard-router runs |
|---|---|---|
| identity (= aux) | all tokens | exist: λ=0.1 / 1 / 10 → 3.2322 / **3.2206** / 3.2322 |
| QC | the exact flip set, \|n_e − q\| tokens | exist (4b/4c): best λ=0.1 → 3.2126 / 3.2129 (detach), 0.65 / 1.08 |
| fixed-N | the N tokens nearest the boundary, N ∈ {128, 512, 2048} per step | 3 at λ=0.1; then λ=1 at the best N |
| extended QC | everything past the corrected boundary | 2: λ=0.1 and 1, since its support is nearly every token for overloaded experts. Retests its collapse with detach on. |
| rect | \|m\| < w/2, w ∈ {1, 3} logits | 2 at λ=0.1; then λ=1 at the best w, if rect works |

- **First batch, submitted 2026-10-10:** sweep `26-10-10-experts2048_every2_kernels`, config
  `configs/2048_experts_kernels.yaml`.
  - Each spec differs from vf44gjaf (standard-router QC, λ=0.1, detach) only in the kernel's keys.
  - fixed-N's fraction is N / 524,288 tokens per step: 2^-12, 2^-10, 2^-8 for N = 128, 512, 2048.

  | run dir | kernel | λ | job |
  |---|---|---|---|
  | 001 | fixed-N, N=128 | 0.1 | 2258716 |
  | 000 | fixed-N, N=512 | 0.1 | 2258715 |
  | 005 | fixed-N, N=2048 | 0.1 | 2258720 |
  | 006 | extended QC | 0.1 | 2258721 |
  | 003 | extended QC | 1 | 2258718 |
  | 002 | rect, w=1 | 0.1 | 2258717 |
  | 004 | rect, w=3 | 0.1 | 2258719 |

  - These are the first GPU runs of fixed-N with the global (EQB) radius, and of rect with mass normalization.

**First-batch results (2026-10-10):**

| kernel | λ | run id | val loss | MaxVio end / whole run | local MaxVio, end | dead slots |
|---|---|---|---|---|---|---|
| **QC** | 0.1 | vf44gjaf | **3.2129** | **0.65 / 1.08** | 3.60 | 4 |
| fixed-N, N=128 | 0.1 | p8r58fi2 | 3.2227 | **0.53 / 0.94** | 3.14 | 14 |
| fixed-N, N=512 | 0.1 | jyq81eo8 | 3.2151 | 0.72 / 1.01 | 3.55 | 15 |
| fixed-N, N=2048 | 0.1 | 9gj6hpnj | 3.2131 | 0.76 / 1.18 | 4.15 | 6 |
| rect, w=1 | 0.1 | 7sxgd69b | 3.2164 | 1.18 / 2.66 | 4.20 | 28 |
| rect, w=3 | 0.1 | jjlj7by2 | 3.2231 | 1.65 / 3.05 | 5.73 | 11 |
| extended QC | 0.1 | nnu3wcz6 | 6.7889 | 12.3 / 18.0 | 12.5 | 799 |
| extended QC | 1 | 7rrddj51 | 6.6833 | 4.97 / 7.98 | 5.36 | 1986 |
| identity = aux | 1 | 5w5e2lak | 3.2206 | 1.61 / 2.24 | 5.17 | – |

- **QC is still the best.**
- **fixed-N** trades loss against balance through N: N=128 balances best of all runs but loses 0.010; N=2048 matches
  QC's loss with slightly looser balance; N=512 is about tied with QC.
- **rect** is worse: w=1 loses 0.0035 with looser balance and the most dead experts of the working kernels (fits
  empty windows); w=3 is worse still.
- **Extended QC broke at both λ. Do not run the extended variant again (user, 2026-10-10).** It fails by design, not
  from a bug: the window matches its definition, and the old extended-QC runs collapsed the same way.
  - **Mechanism:** an underloaded expert's window is the q tokens it should own after the correction, and it already
    owns a share 1 + r_e of them. An overloaded expert's window is the T − q tokens it shouldn't own. So almost none
    of the push moves a token across the boundary: the load error never shrinks, and the push repeats every step.
  - **Why mass normalization doesn't prevent it:** it equalizes the push on the probabilities π. A push on π_te
    moves the logits in proportion to π_te. The underloaded experts' push lands on their own high-π tokens; the
    overloaded experts' push lands on tokens where their π is near zero. So upward pushes win, the logits spread, and
    the effect feeds itself.
  - **Data:** top-1 router logit 26→75 (λ=0.1) and 29→222 (λ=1), against 2–3 for QC and aux; logit std 3→14 and
    4→33, against 0.3–0.5; grad norm in steps 100–1000 of 1.4 and 119, against 0.18. The LM loss stopped improving
    by iteration ~50.
  - **General point:** mass normalization equalizes strength across kernels in logit terms only when the window sits
    at the boundary, where over- and underloaded experts' window tokens have similar π. That holds for QC, fixed-N
    and rect.

**Second batch, done 2026-10-10:** sweep `26-10-10-experts2048_every2_kernels_followup`, config
`configs/2048_experts_kernels_followup.yaml`. Each spec differs from its first-batch run in one key.

| run dir | kernel | λ | why | job |
|---|---|---|---|---|
| 000 | fixed-N, N=2048 | 1 | neighbouring λ at the best fixed-N by loss | 2261256 |
| 002 | rect, w=1 | 1 | neighbouring λ at the best rect | 2261258 |
| 001 | rect, w=0.1 | 0.1 | narrow fixed-width window, about the size of QC's flip set (user) | 2261257 |

**What the second batch decides; no further Part 7 runs are planned:**
- **Each kernel's best λ:** with the λ=1 runs, every kernel is reported at two λ.
- **Narrowness or exactness:** rect w=0.1 against QC. If they match, narrowness is what matters. If QC wins, the
  per-expert exact flip set is what matters: QC adapts its window to each expert's error and can't have empty windows.
- **Stopping rule:** a kernel that beats QC by the Part 0 rule gets at most one edge run (λ or N). Otherwise the stage
  ends with QC as the support.
- Gaps under 0.003 (QC, fixed-N 512/2048) stay ties until seeds (Part 10). QC still has an advantage there: it has no
  N or w to tune.
- The split version of the best kernel, if it isn't QC, is part of Part 8.

**If** rect still loses dead experts (empty windows), **then** report it as the reason for per-expert supports: QC and
fixed-N can't have empty windows.

**Second-batch results and conclusion (2026-10-10): QC is the support.** Every kernel at its best λ, with the λ=1
neighbours:

| kernel | λ | run id | val loss | MaxVio end / whole run | local MaxVio, end | dead slots |
|---|---|---|---|---|---|---|
| **QC** | 0.1 | vf44gjaf | 3.2129 | 0.65 / 1.08 | 3.60 | **4** |
| QC (attached, 4b) | 1 | ftiqvhnn | 3.2226 | 0.54 / 1.17 | 2.83 | 9 |
| rect, w=0.1 | 0.1 | t6wlfzf9 | **3.2115** | 0.66 / 1.00 | 3.86 | 115 |
| rect, w=1 | 0.1 | 7sxgd69b | 3.2164 | 1.18 / 2.66 | 4.20 | 28 |
| rect, w=1 | 1 | raybwh1v | 7.4227, diverged | 102 / 168 | 102 | 3643 |
| fixed-N, N=2048 | 0.1 | 9gj6hpnj | 3.2131 | 0.76 / 1.18 | 4.15 | 6 |
| fixed-N, N=2048 | 1 | sqsnzijp | 3.2159 | 0.71 / 1.28 | 3.19 | 8 |
| identity = aux | 1 | 5w5e2lak | 3.2206 | 1.61 / 2.24 | 5.17 | – |

Dead slots are out of 6 layers × 2048 experts.

- **λ:** λ=1 is worse than 0.1 for every kernel, so 0.1 is each kernel's best (λ=0.01 failed for QC on both routers).
- **The push has to land at the boundary:** extended QC broke (see above).
- **Among boundary supports, narrowness gives the loss and balance.** rect w=0.1, a window about the size of QC's
  flip set, ties QC on both (−0.0014 loss, within noise). Wider supports lose: rect w=1 and w=3, and fixed-N N=128
  on loss.
- **Exactness prevents empty windows.** rect w=0.1 ends with 115 dead expert slots against QC's 4: a fixed-width
  window can be empty, and that expert is never pushed. QC sizes each expert's window to its error |n_e − q|, so it
  can't be empty, and it has no width or N to tune.
- **No kernel beats QC by the Part 0 rule,** so by the stopping rule the stage ends with QC; no edge runs.
- **rect w=1 at λ=1 diverged** the same way as extended QC: the router logits ran away (top-1 4 → 85, std 1 → 21)
  and the grad norm reached 177–640. At λ=0.1 the same kernel is healthy. No confirmed mechanism: its windows grew
  (in-window fraction 0.009 → 0.10) rather than shrank.
- Gaps under 0.003 stay ties until seeds (Part 10). For the final table, QC and the best alternative (rect w=0.1)
  are candidates for the seed runs.

### 7b. Symmetric QC — to test (user, 2026-10-14)

**The kernel.** δb_e is computed as in QC, but the rect is centered on the current boundary (margin 0) and runs from
−|δb_e| to +|δb_e|. Besides QC's flip band, it holds the mirror band: the tokens within |δb_e| on the other side,
whose assignment already agrees with the correction. These are an underloaded expert's selected tokens near the
boundary, and an overloaded expert's unselected ones.

```
underloaded (δb_e > 0)   QC:        [−δb_e ════] 0
                         symmetric: [−δb_e ════] 0 [════ +δb_e]
overloaded  (δb_e < 0)   QC:                     0 [════ +|δb_e|]
                         symmetric: [−|δb_e| ══] 0 [════ +|δb_e|]
```

- Every token in the window is pushed the same way: down for an overloaded expert, up for an underloaded one. The
  mirror band's push moves tokens away from the boundary, so they don't flip the wrong way.
- QC is a one-sided finite difference of the load with step δb_e. The symmetric version is the centered difference
  with the same step. It is also the rect at the exact margin with a per-expert width 2|δb_e| instead of a global w.
- **The question:** does the one-sided support matter? The comparison is QC against symmetric QC at the same
  per-expert step. Rect w=0.1 (a narrow, symmetric, fixed-width window) tied QC but ended with 115 dead slots; the
  symmetric QC window can't be empty.
- **Heights follow the window.** 1/width uses 2|δb_e|. Mass normalization divides T by the tokens in both halves, and
  count normalization divides q by them. With mass normalization, the flip band's share of the push drops to
  |flip| / (|flip| + |mirror|), about half in the CPU toy (41 vs 37 tokens per expert).

**Code — done 2026-10-14 (CPU-tested, no GPU run yet).**
- Flag `quantile_correction_symmetric_window`. It works with every other QC option: pre or post, detach or not,
  exact or runner-up margin, any estimator, and every height variant. It has no effect with
  `quantile_correction_extended_window`.
- `quantile_correction/ste_height_max` and `affected_token_expert_fraction` still describe the flip band, as for the
  extended window.
- New eval logs for the mirror band (user, 2026-10-14): `quantile_correction/mirror_token_expert_fraction`, with the
  same normalization as the flip band's `affected_token_expert_fraction`, and
  `quantile_correction/mirror_to_flip_ratio`.
  - With mass normalization the flip band's share of the push is about 1/(1 + ratio).
  - They are logged for every QC run. So the next QC runs show the mirror band's size even without the symmetric
    window.
- CPU check, all pass:
  `apptainer exec <sif> python /e/data1/mmlaion/shechter1/pending_changes/test_qc_symmetric_window.py`.
  It checks that the window is the flip band plus its mirror, the heights (1/(2|δb|), T and q per expert over both
  halves), that it has no effect with the extended window, and that it runs pre/post, attached/detached and with
  runner-up margins. The 17 existing QC CPU tests still pass (`pytest --noconftest
  tests/unit_tests/transformer/moe/test_lb_quantile_estimators.py -k 'not distributed'`).

**Runs.** Standard router, vf44gjaf's spec (the fixed configuration: softmax, post, exact margin, EQB, mass
normalization, detach) plus `quantile_correction_symmetric_window: [true]`:
- smoke run first: symmetric QC at λ=0.1 trained normally (4d, deleted);
- then λ ∈ {0.1, 1} × {detached, attached}: `configs/2048_experts_symmetric_qc.yaml`, 4 runs (user, 2026-10-15: the
  λ search and the detach check in parallel, one run more than doing them in turn). λ=1 is there because the flip
  band now gets only part of each expert's push. Symmetric QC's windows are two-sided, the case 4c named for
  revisiting detach.
- **Submitted 2026-10-16:** sweep `26-10-16-experts2048_every2_symmetric_qc`, jobs 2349460–2349463. Run dirs: 000
  detached λ=0.1, 001 detached λ=1, 002 attached λ=0.1, 003 attached λ=1.

Decision (Part 7's stopping rule):
- **If** symmetric QC beats QC by the Part 0 rule, **then** it becomes the support, and gets at most one edge run.
- **Otherwise** QC stays: the one-sided flip band is enough, and it is the simpler kernel.

## Part 8 — Split router-weighter and the learnable bias (interesting #1) — first runs done 2026-10-10

- **Split vs standard at a fixed method:** for identity this exists (3.2079 vs 3.2206), and for QC too (split
  iwfdgrhg 3.2118, 0.79 / 1.87 vs standard vf44gjaf 3.2129, 0.65 / 1.08). If Part 7's best kernel isn't QC, run it in
  the split (1 run).
- **Submitted 2026-10-10, assuming QC is the method** (user). The 3 Part 7 follow-up runs only matter if a kernel beats
  QC. All Part 8 runs are on the standard router with softmax, and "QC" is the fixed configuration (post, mass
  normalization, exact margin, EQB, detach).

  | load balancing | λ / rate | added | sweep (`26-10-10-experts2048_every2_…`) | job | compare with |
  |---|---|---|---|---|---|
  | EQB QB | – | extra compute 1024 | `qb_extra_router_compute` | 2261701 | EQB 3.2114 |
  | ID | 0.06 | extra compute 1024 | `id_extra_router_compute` | 2261702 | ID 3.2117 |
  | QC | 0.1 | extra compute 1024 | `qc_extra_router_compute` | 2261703 | QC 3.2129 |
  | QC | 0.1 | per-token bias | `learnable_bias` 002 | 2261706 | QC 3.2129 |
  | QC | 0.1 | per-expert bias | `learnable_bias` 003 | 2261707 | QC 3.2129 |
  | identity (full STE + post) | 1 | per-token bias | `learnable_bias` 000 | 2261704 | aux 3.2206 |
  | identity (full STE + post) | 1 | per-expert bias | `learnable_bias` 001 | 2261705 | aux 3.2206 |

  - Configs: `configs/2048_experts_{qb,id,qc}_extra_router_compute.yaml` and `configs/2048_experts_learnable_bias.yaml`.
    Each spec differs from its reference run (jrvtxfq7, 6c549raw, vf44gjaf) only in the added keys.
  - **After these:** the best bias combination gets at most 2 runs at the neighbouring λ.

**Results (2026-10-10).** Parameter-matched standard router (extra compute 1024). Each cell is val loss, then MaxVio
end / whole run:

| method | standard | + extra compute | split |
|---|---|---|---|
| QC 0.1 | 3.2129, **0.65 / 1.08** (vf44gjaf) | 3.2137, 0.74 / 1.40 (l1xnczwd) | 3.2118, 0.79 / 1.87 (iwfdgrhg) |
| EQB QB | 3.2114, 0.99 / 1.50 (jrvtxfq7) | 3.2105, 1.57 / 2.33 (l8rwsnwy) | gradient-free LB fails in the split |
| ID 0.06 | 3.2117, 1.00 / 1.89 (6c549raw) | 3.2136, 1.33 / 2.18 (aeanhai2) | same |
| DeepSeek 0.1 | 3.2102, 2.82 / 4.35 (sxwv1bbi) | 3.2082, 2.45 / 5.57 (o5er2imf) | same |
| aux 1 | 3.2206, 1.61 / 2.24 (5w5e2lak) | 3.2155, 2.29 / 3.02 (65qnx4s3) | **3.2079**, 1.21 / 2.60 (0nl5u4q1) |

- **Extra parameters change loss by only ±0.002** and slightly worsen balance for most methods.
- **The split's gain for aux (−0.013) is not explained by its parameters.** The split doesn't help QC.
- **QC on the plain standard router is still the best-balanced method at similar loss,** including against the
  parameter-matched baselines.

Learnable selection bias (standard router):

| | λ | run id | val loss | MaxVio end / whole run | dead slots |
|---|---|---|---|---|---|
| QC + per-token | 0.1 | vov8qldl | 3.4961 | 699 / 731 (collapsed) | 11958 |
| QC + per-expert | 0.1 | y5yjhvhx | 3.5590 | 1022 / 920 (collapsed) | 11865 |
| identity + per-token | 1 | op6r9fks | 3.2267 | 1.84 / 4.02 | 14 |
| identity + per-expert | 1 | erotwxhf | 3.2183 | 2.26 / 17.2 | 35 |
| ref: aux, no bias | 1 | 5w5e2lak | 3.2206 | 1.61 / 2.24 | – |

- **With identity, the bias doesn't help:** per-token is worse on both numbers; per-expert gains 0.002 in loss but
  balances much worse.
- **QC + bias never balanced at λ=0.1.** The load-balancing loss stayed at 6,000–13,000 (balanced runs: ~17–20). The
  likely cause is the LR tie: with a bias the router gets no LB gradient, so λ only sets the bias LR (lr × λ). At
  λ=0.1 the bias moved 0.002 per step, 10× slower than in the identity runs, and never caught up with the early
  routing collapse that the LM loss causes. The identity runs recovered by iteration ~600.
- **Follow-up, submitted 2026-10-11 (the neighbouring λ):** QC + bias at λ=1, the same bias LR as the identity runs.
  Sweep `26-10-11-experts2048_every2_learnable_bias_qc_lambda1`, config
  `configs/2048_experts_learnable_bias_qc_lambda1.yaml`: 000 per-token (job 2270737), 001 per-expert (2270738).
  Specs differ from vov8qldl / y5yjhvhx only in λ.
  - **If** neither beats QC without a bias, **then** the bias is dropped from the method.

**Split + learnable bias: not run, because it would duplicate parameters the split router already has**
(2026-10-11):
- No earlier run combined them.
- **The code is correct for what the user expected.** In the split, both the LM STE (normalized relative) and the LB
  STE use `_margin_input_for_ste` = router logits + bias, with neither detached. So the bias gets exactly the router's
  signal, and since the post fix the LB gradient enters at softmax(z + b).
- **CPU check:** with the split's router bias on (`moe_router_enable_bias`, as in pj2yuna0 and iwfdgrhg), the
  learnable expert bias's gradient equals the router bias's gradient exactly, with the LM STE alone and with QC on top.
- **So it adds no new mechanism.**
  - `expert_bias` is a second copy of the router's per-expert bias with the same gradient; it only changes the
    effective LR.
  - `per_token_bias` is a second copy of the router matrix: both read the detached input, so z + W_b h = (W + W_b) h.
  - On the standard router a selection bias does add something: it separates selection from the combine weights.
    In the split, the weighter already does that.
- **Parameter-matched standard routers:** `moe_router_extra_computation: 1024`, the weighter's parameter count.
  - aux and DeepSeek already exist;
  - add EQB QB (it won Part 2) and ID (2 runs) and the best kernel (1 run).
- **What counts as a win:** better balance at similar loss, or better loss at similar balance, against the best
  parameter-matched standard config. Today split aux (3.2079, end 1.21 / whole run 2.60) against DeepSeek with the MLP
  (3.2082, end 2.45 / whole run 5.57) is a balance win at equal loss.
- **Learnable selection bias**, per-token (`per_token_bias`) and per-expert (`expert_bias`): decide with or without it
  for the final standard-router method.
  - **Both kernels are tested, QC and identity** (user, 2026-10-10). In the split, decoupling selection from the LB loss
    helped aux a lot (−0.013) but not QC. A selection bias is the standard-router version of that decoupling. Testing
    only QC + bias would leave open whether aux + bias beats QC.
  - The identity kernel with a bias must be `centered_fsq` + full STE + post: it equals aux on softmax(z + b). The
    plain `aux_loss` type uses the unbiased scores and would never train the bias.
  - Run the with-bias version at the best λ and one neighbour, since the bias LR is tied to λ.
  - **Every learnable-bias run sets `tie_learnable_bias_lr_to_aux_loss_coeff: true`** (user, 2026-10-10). This makes
    the bias LR = lr × λ. Don't also set `moe_learnable_bias_lr_mult`: the tie sets it, and arguments.py refuses both
    together.
    - Reason: Muon and Adam normalize the gradient's scale, so without the tie λ would not change how fast the bias
      moves.
    - With the router logits detached from the LB loss (the default), the tie is the only way λ acts at all.
  - The earlier evidence is mixed: sigmoid QC with the per-token bias was the best standard-router STE (eblyy2pd,
    3.218, end 0.59), while softmax with the per-token bias failed to balance at λ=0.1.
  - Under the post path the gradient enters at softmax(z + b), the selection probabilities. Code fixed and checked on CPU
    (2026-10-10): with the identity kernel this equals the aux loss on softmax(z + b), the bias gets the LB gradient, and
    the router gets none unless `moe_learnable_bias_pass_grad_through_scores`. Before the fix, post was silently off with
    any selection bias.
  - The learnable `expert_bias` failed 6 of 6 times before, but all 6 used the old recipe. The old recipe also failed
    for plain softmax QC on this router, so the per-expert bias is retested with the fixed configuration (user,
    2026-10-10). The post fix covers it too, and its CPU check passed (identity + post = aux on softmax(z + b)).
- **Gradient-free LB in the split** (all 11 runs ended at 3.54–3.58): an appendix note only.

### 8b. Router activation in the split: where the LB and LM STEs enter — planned 2026-10-14

**Today (code read, 2026-10-14).** Every split run selects with `moe_router_selection_activation: none`, i.e. top-K
on the router's raw logits (including the router's linear bias). The two STEs then enter at different points:

| signal | enters at | |
|---|---|---|
| LB, aux (= identity + post) | softmax of the logits (`_compute_split_scores_for_aux_loss`) | post |
| LB, STE kernels with `moe_load_balance_ste_post_activation` (corrected QC) | the same softmax | post |
| LB, STE kernels without it (full STE, pre QC) | the raw logits (`_margin_input_for_ste`) | pre |
| LM, normalized relative (`moe_lm_loss_ste_normalized_relative`) | the raw logits, selected experts only | pre |

- So split aux (0nl5u4q1) and the corrected split QC (iwfdgrhg) are **unmatched** (LB post, LM pre). The split full
  STE (x44pf1tk) and the pre QC runs are matched pre.
- The router input is detached in the split (`moe_router_pass_grad_to_input` off), so both signals train only the
  router. The standard router has no LM STE (its LM gradient reaches the router through the combine weights), so
  this section is split-only.

**Question.** How the router activation, and where each STE enters, changes the identity kernel (aux) and the best QC
(the fixed configuration).
- There is no theoretical reason to match or not match the LM and LB entry points. So the default becomes
  **matched** (both at the same point), with a few unmatched runs as the check (user, 2026-10-14).
- One effect to keep in mind when reading the λs: an activation's Jacobian scales a signal by about p (≈ a few/E for
  softmax). Muon normalizes the size of the router update, but not the ratio of the two signals summed before it. So
  moving one signal between pre and post changes the LM:LB balance, and λ absorbs it (aux needs a λ about 1000× the
  full STE's). Matched, the ratio is as in the all-pre case, so the best λ can move.

**Identity kernel: aux only where the scores are normalized (user, 2026-10-14).**
- Where the LB STE enters at the raw logits (pre, the `none` activation: blocks C and E), the identity run is
  `moe_router_load_balancing_type: centered_fsq` + `load_balance_ste_type: full`, never `aux_loss`.
- The reason is the uncentered constant. Aux's gradient on a token's score is λ(1 + r_e)/T; the centered full STE's
  is λ·r_e/T (r_e = the expert's relative load error, K=2).
  - On normalized scores (softmax, or s/Σs) the constant λ/T is the same for every expert of the token, and the
    normalization's Jacobian removes it, so aux = `centered_fsq` + `full` + `moe_load_balance_ste_post_activation`
    exactly (2/K times it at other K; Appendix A).
  - On unnormalized scores nothing removes it. Every logit of every token is pushed down by λ/T, a bias. It does not
    change top-K, but it does change the router's weights and bias, and under Muon it changes the update's direction.
- So blocks A, B and D may use aux, and A and B do, for comparability with 0nl5u4q1. Blocks C and E use the full
  STE (x44pf1tk's LB keys).
- The LM STE has no such constant. Its per-token signal sums to zero over the selected experts, because the weighter
  probabilities sum to 1; the CPU check confirms the sum is 0 both pre and post.

**Code — done 2026-10-14 (CPU-tested, no GPU run yet).**
- New flag `moe_lm_loss_ste_post_activation` (`transformer_config.py`; implemented in
  `TopKRouter._apply_normalized_relative_lm_ste`). The normalized-relative LM STE enters at π, the split router's
  normalized scores, instead of at the selection scores. π is the aux loss's scores:
  - softmax(z) for selection activation `none` or `softmax`;
  - s/Σs for `sigmoid` and `sqrtsoftplus`.

  The signal p_e·∇y·(h_e − y) is unchanged; through the normalization it also reaches the unselected experts'
  logits.
- The bias handling mirrors `moe_load_balance_ste_post_activation`:
  - with a selection bias added to the logits (DeepSeek, selection-only learnable bias), it enters at softmax(z + b);
  - with a score-space bias (split QB, post-softmax expert bias) the flag does nothing.
- Only the normalized-relative LM STE reads the flag. The rect LM STE and the extra-experts STEs ignore it.
- Sigmoid already exists as `moe_router_selection_activation: sigmoid`.
  - Aux and the LB post target become s/Σs.
  - The margins, the QC windows and the LM STE's pre point move to the sigmoid scores s (score space).
  - Part 1.3 keeps supports in logit space, so the sigmoid QC windows are a different support, not only a different
    activation.
  - The top logits of the D/C runs are 2.4–6.8, where σ′ is about 0.08–0.001, so the sigmoid compresses the top margins.
- Softmax needs no new flag: top-K on the softmax equals top-K on the logits, and LB post already uses softmax(z).
  So the config keeps `moe_router_selection_activation: none` for blocks A, C, D and E. "softmax" in the table below
  is the post target, not a config value. This keeps A one flag away from D.
- CPU check, all pass: `apptainer exec <sif> python /e/data1/mmlaion/shechter1/pending_changes/test_lm_post.py`
  (a router object built without `__init__`):
  - forward unchanged and the weighter gradient unchanged;
  - pre: the gradient at the logits is c_e = p_e(G_e − Σ_sel p_i G_i) on the selected experts;
  - post: the softmax VJP of c, and it equals the `softmax` activation;
  - sigmoid: σ′·c pre, and through s/Σs post;
  - logit bias: post = the softmax VJP at z + b, and the bias gets it too;
  - score bias: post has no effect.

**Runs.** Split router, softmax weighter, router bias, the split keys of iwfdgrhg and 0nl5u4q1. λ per Part 6, plus
the neighbour on the lower side, because moving LM to post weakens it relative to LB.

| block | activation | LB enters | LM enters | identity | QC (fixed configuration) | new runs |
|---|---|---|---|---|---|---|
| A, matched post | softmax | post | post | aux, λ ∈ {0.1, 1} | λ ∈ {0.01, 0.1} | 4 |
| B, matched post | sigmoid | post (s/Σs) | post (s/Σs) | aux, λ=1 | λ=0.1 | 2 |
| C, matched pre | none | pre (logits) | pre (logits) | exists: full STE, λ=1e-3 (x44pf1tk: 3.2122, 1.38 / 8.12) | exists: λ=1e-3 (ocuc2x7t: 3.2256, 0.66 / 1.62) | 0 |
| D, unmatched | softmax | post | pre | exists: aux, λ=1 (0nl5u4q1: 3.2079, 1.21 / 2.60) | exists: λ=0.1 (iwfdgrhg: 3.2118, 0.79 / 1.87) | 0 |
| E, unmatched | softmax | pre | post | full STE (not aux), λ=1e-3 | QC pre, λ=1e-3 | 2 |

- 8 runs, after a smoke run of the new flag (`exit_interval: 300`).
- Order: A first (softmax is the decided activation), then E (the other unmatched direction), then B.

**How to run it.** Both configs copy `2048_experts_split_corrected_qc.yaml` and change only the name and the LB
block. Each group is one entry of `LIST_BLOCK`; entries set every LB key, so no defaults leak between them.

| block | `moe_router_selection_activation` | `moe_lm_loss_ste_post_activation` | identity run's LB keys | QC run's LB keys |
|---|---|---|---|---|
| A | none | true | `aux_loss`, λ ∈ {0.1, 1} | corrected QC + `moe_load_balance_ste_post_activation`, λ ∈ {0.01, 0.1} |
| B | sigmoid | true | `aux_loss`, λ=1 | corrected QC + post, λ=0.1 |
| C (exists) | none | false | `centered_fsq` + `full`, λ=1e-3 | corrected QC, no post, λ=1e-3 |
| D (exists) | none | false | `aux_loss`, λ=1 | corrected QC + post, λ=0.1 |
| E | none | true | `centered_fsq` + `full`, λ=1e-3 | corrected QC, no post, λ=1e-3 |

- Corrected QC = `quantile_correction_ste`, `quantile_correction_exact_margin`, `moe_quantile_estimator: exact`,
  `moe_load_balance_ste_mass_normalized`, `moe_load_balance_ste_detach_threshold`.
- Smoke (the 3 new code paths, 300 iterations): `configs/2048_experts_split_router_activation_smoke.yaml`. It holds
  A's QC at λ=0.1, B's QC and E's full STE.
- Full (8 runs): `configs/2048_experts_split_router_activation.yaml`.
- Submit (only on the user's request): `submit_multiple.py configs/<file>.yaml -s run_megatron.sh -d -y`, then `-o -y`
  (run-experiments skill).
- After the smoke run, check three things: the 3 runs start; `lm loss` and MaxVio are close to the D/C runs at
  steps 150 and 300; and B's `router_logits/top1_pre_activation` stays well below 17, where the fp32 sigmoid
  saturates to 1.
- Runs are identified by label. The varied keys are the activation, the two post flags, the LB type, the STE type
  and λ.

Decision:
- **If** A is similar to or better than D for both kernels (Part 0 rule), **then** matched post is the split default,
  and D and E go in the paper as the unmatched ablation.
- **If** one unmatched block (D or E) beats A on both kernels, **then** use it, and say that there is no a priori
  reason for it.
- **If** the kernels disagree, keep matched (the default) and report both.
- **Sigmoid (B):** keep softmax unless B wins by the Part 0 rule. B also counts as Part 3's sigmoid check for the
  split.
- C against A is the matched pre/post question of Part 4b in the split; C only serves as the reference here.

## Part 9 — Per-expert strength as a controller (interesting #3), to discuss later

- After Part 1 the gradient is λ·a_e·k_te, with a_e = r_e (proportional to the error).
- The optimizer accumulates the pushes in the router weights, so this is already integral control of the logit
  level, and Muon's momentum adds lag. ID-Balancing's "I + gated D" therefore maps to a_e = r_e + K_D·g_e·Δr_e.
- Candidates, on the identity kernel in the split at its best λ (1, split aux):
  - sign(r_e);
  - tanh(r_e/c), with c ∈ {0.5, 2};
  - gated derivative, with K_D ∈ {1, 10};
  - a leaky integral.
- Port the best one to the best kernel and to the other router. About 10 runs.

## Part 10 — Later

- **Seeds:** run the same seed set (for example 1001 and 1002) for every config in the final comparison.
- **Final table:** Part 0's metrics plus the downstream tasks.
  - `train/vio/MaxVioBatch` exists only for runs from 2026-10-04 on.
  - So the table uses MaxVioGlobal (end of run and whole run) plus the local dispatch MaxVio.

## Run budget (full runs, about 96 GPU-h each)

| | runs | GPU-h |
|---|---|---|
| Necessary: Parts 2, 4, 5, plus the sigmoid check | ~20 | ~1,900 |
| Interesting: Parts 7–9 | ~29 | ~2,800 |
| Part 8b: router activation in the split | 8 | ~770 |
| Part 4d: isolated re-checks (height, activation, detach) | 8 | ~770 |
| Part 7b: symmetric QC (λ {0.1, 1} × attached / detached) | 4 | ~380 |
| Seeds | later | |

**Status (2026-10-11):** Parts 1, 2, 4, 5 and 7 are done. The method is QC with the fixed configuration on both
routers: softmax, post, exact margin, EQB quantile, mass normalization, detach, λ=0.1. Part 8's first 7 runs are done:
extra parameters don't help, and the bias doesn't help identity. QC + bias at λ=1 (2 runs) is running; split + bias
is not run because it duplicates the split router's own parameters. Part 9 comes last.

## Appendix A — Aux vs identity STE: what is the same and what isn't

- **Aux** (Megatron): L = λ·E/(K·T²)·Σ_e n_e·Σ_t p_te, with n_e detached. So ∂L/∂p_te = (λ/T)(1 + r_e), the
  uncentered signal.
- **Identity STE** with the centered loss λ[1 + E·Σ(f_e − 1/E)²] (f_e = n_e/(T·K)) and ∂n_e/∂p_te := 1:
  ∂L/∂p_te = (2/K)(λ/T)·r_e, the centered signal.
- **The difference** is the constant λ/T on every (t, e). At K=2, 2/K = 1.
  - For π-normalized variables (Σ_e π_te = 1), a constant added to every expert of a token vanishes in the
    Jacobian.
  - So **identity + post is exactly the aux loss**, and the existing aux runs are the identity member of the family.
- **Where the centering does matter** (it is right that Σf² and Σ(f − 1/E)² differ under an STE):
  - **Windowed kernels** (rect, QC, fixed-N): the constant lands only on in-window pairs. Uncentered, every expert's
    window gets pushed down, underloaded experts included. That is exactly why `exact_jump_ste` (weighted by n_e)
    failed in all 4 cells. So always use r_e.
  - **Pre (u = z):** the constant becomes a common push on all of a token's logits. Softmax ignores it in the forward
    pass, but it still enters the router-weight gradient as a component shared by every row, and it enters the hidden
    state. Our full STE uses centered_fsq, so it is centered.
- **Aux vs the full STE (identity-pre).** The difference is the Jacobian, which also rescales the logit gradient by
  roughly p_te ≈ a few/E. That is why the full STE needs a λ 100–1000× below aux.
  - Standard router: the sweeps overlap at matched balance. Full STE at λ=0.01 got 3.2651 (end 1.61, whole run 2.77);
    aux at λ=1 got 3.2206 (end 1.61, whole run 2.24). So this comparison stands as "pre vs post".
  - Split: not calibrated yet (Part 5).
- **LEI** (EQB paper) is not "pre without global LB". It injects η·ρ_e at the *unnormalized* sigmoid score s, so the
  logit gradient carries σ'(z), and it differs in three more ways:
  - it uses the local (micro-batch) load error;
  - it bounds the error vector with tanh;
  - it is paired with EQB, which handles global balance.

  It is not planned.

## Appendix B — Activation evidence (standard router unless noted)

Loss is the final val loss; balance is `vio/MaxVioGlobal`, end of run (mean of last 3 evals) / whole run.

| method | softmax | sigmoid | sqrtsoftplus |
|---|---|---|---|
| aux, λ=1 | **3.2206**, 1.61 / 2.24 | 3.2401, 2.03 / 8.50 | 3.2453, 2.34 / 6.73 |
| DeepSeek, best rate | **3.2102** (0.1), 2.82 / 4.35 | 3.2174 (1e-3), 1.38 / 18.3 | 3.2212 (1e-2), 2.09 / 6.24 |
| QB, rank-averaged | **3.2105**, 1.44 / 1.82 | 3.2180, 1.86 / 2.27 | 3.2205, 1.73 / 2.16 |
| QB, EQB (global) | **3.2114**, 0.99 / 1.50 | 3.2174, 1.32 / 1.90 | — |
| QB, Kimi (global) | **3.2106**, 1.26 / 1.61 | 3.2173, 1.76 / 2.12 (bias on the sigmoid scores) | — |
| ID, best rate | **3.2117** (0.06), 1.00 / 1.89 | 3.2169 (6e-3), 1.18 / 1.83 | — |
| split aux, λ=1, weighter | **3.2079**, 1.21 / 2.60 | 3.2219, 1.94 / 3.39 | — |
| split QC, λ=0.1, weighter | 3.2186, 0.48 / 1.07 | 3.2189, 0.48 / 1.10 | — |

## Appendix C — Why there is no half-length proxy

I checked runs from 110 healthy every2 runs. Val loss at step 3000 (pre-decay) orders run pairs the same way as the
final value in:
- 94.5% of pairs when their final gap is > 0.003;
- 96% when > 0.005;
- 98% when > 0.01.

The best λ at step 3000 matched the final best in 37 of 45 sweep groups. The user preferred full runs, because the
differences that matter are around 0.003–0.005.

## Appendix D — Queries

End-of-run and whole-run MaxVioGlobal:

```sql
WITH mv AS (SELECT run_id, step, value FROM eval WHERE key='vio/MaxVioGlobal'),
r AS (SELECT run_id, max(step) AS last FROM mv GROUP BY run_id)
SELECT mv.run_id, avg(CASE WHEN mv.step > r.last-450 THEN value END) AS end3, avg(value) AS whole_run
FROM mv JOIN r USING (run_id) GROUP BY mv.run_id
```

## Appendix E — The two routers react differently to the same LB signal

| method | standard router | split router |
|---|---|---|
| full STE | 3.2547 @ 3.4 | fails at every λ tried |
| softmax QC (pre) | collapses at λ ≥ 0.01 | works at 0.01–10 |
| rect, w=1 | empty windows, fails | works |

- **Likely mechanism:** in the standard router the same logits also gate the expert outputs, so LB and LM gradients
  meet on the same parameters. With K=2 softmax, the LM gradient only constrains the gap between the two chosen
  logits, and the LB gradient also flows into the hidden state.
- In the split router, the LB gradient trains only the selection head.

## Appendix F — QC margin bug

- **The bug:** QC measures every margin against the token's runner-up (`topk_plus_one`), so each token's runner-up
  sits at a margin of exactly 0.
  - Each expert is the runner-up for about T/E = q/K tokens.
  - An underloaded expert that is short by fewer than its tied tokens gets δb = 0, which means an empty window and
    no direct push.
  - Today such experts are pushed only indirectly, as the attached reference of overloaded experts' windows.
- **Toy evidence** (E=2048, K=2, random logits with expert offsets): the share of underloaded experts with δb = 0 is
  - 23–67% at q = 32 (larger near balance);
  - 50% → 65% → 71% as T grows 8×. More tokens make it worse, because the tied mass grows like q while sampling
    noise shrinks.
  - With exact_margin it is 0%, and every window holds exactly |n − q| tokens.
  - It does not explain the 1e5–1e7 heights (`quantile_correction/ste_height_max`) that were blamed on
    `mean_local`, as first guessed here. With exact_margin they still reach 5e4–2.5e5 (4a): the 1/|δb| height is large
    for any nearly balanced expert.
- **Why exact_margin is the correct one:** if only expert e's bias changes, token t selects e iff z_te + b exceeds
  the K-th largest logit among the *other* experts. That threshold is the runner-up if e is selected, and the token's
  K-th expert if not. Those are exactly the exact_margin references.
- **Which tokens get pushed:**
  - Overloaded e: its n_e − q selected tokens with the smallest lead over the runner-up. These are unchanged from
    today.
  - Underloaded e: its q − n_e unselected tokens closest to the token's K-th (weakest selected) expert. Today they
    are measured against the runner-up, which is the wrong threshold, and the runner-up itself is tied at 0.
- **Caveat:** this is a per-expert view. When all biases move at once (QB), flips interact.
- **exact_margin vs detach:** they are not alternatives.
  - exact_margin decides *which* tokens are in the window. It is a correctness fix.
  - detach decides *where* the gradient goes: the expert alone, or also its competitor in the opposite direction.
    It is a design choice, tested in 4c.
  - Detach with `topk_plus_one` is the worst combination: it removes the only push that slightly underloaded experts
    get today.

## Rationale

- Points 3, 4 and 6 are one decision: the definition of the LB gradient.
  - With supports from logit margins, mass normalization and a gradient that enters at π, the aux loss is exactly the
    identity member.
  - λ means the same thing for every kernel.
  - The only balance dependence left is the explicit r_e, which Part 9 replaces with a controller.
- Point 7 and QC's correctness share one piece of code: the global per-expert quantile.
- Settling activation, margin, pre/post and detach once, on the harder router, prevents a factorial blow-up. Every
  later comparison then changes only the thing being studied.
