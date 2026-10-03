# Rect STE: results so far (2026-10-01)

Scope: the `experts2048_every2` sweeps (E=2048, k=2, MoE every 2nd layer, Muon, 6000 iters, global LB,
512 assignments per expert per step, seed 1000, deterministic mode), plus rect-STE results from earlier settings.
Commands are `tools/runsdb/q.py ...`. vf = final val loss; MaxVio = val MaxVioGlobal (mean of the last 5 evals).

**n_win** = experts inside the STE window per token = `ste/all_layers/all_experts_in_rect_frac` × E. With position
`topk` the K-th expert always sits at margin 0, so n_win ≥ 1, and n_win ≈ 1 means the window is empty. The unit that
transfers across E is per expert: (n_win − 1) × T/E in-window tokens per expert per step, where T/E = 256 in every2
and 2048 at E=256 (T = 524288 tokens per step in both).

## every2: what we learned

1. **The bar is vf ≈ 3.208–3.211 at MaxVio 1.4–2.9.** It is set by DeepSeek softmax bur=0.1 (sxwv1bbi), QB softmax
   (x3e8nqq3) and split aux λ=1 with router bias (0nl5u4q1). Plain aux at its best λ is ≈0.01 behind (5w5e2lak).
   `runs --sweep experts2048_every2 --sort val_loss_final`
2. **No plain-router rect STE run exists in every2.** The only rect runs use the split router-weighter with router bias
   (`26-09-29-experts2048_every2_split_router_weighter_rect_with_bias`, 12 runs).
3. **There, w=0.1 fails for every λ (0.01, 0.1, 1) because the window is empty.** n_win is 1.03–1.17 through step 600,
   i.e. ≈10–40 in-window tokens per expert per step, and ≤ 2 at the end. MaxVio is already 45–58 at step 150 and ends at
   32–43, with vf 3.36–3.38 (895cn4ji, xshkzl8b, cl6tu4e9, bp0x6kdw, lpmz3cuk, dvjjl7e9). These six runs say nothing
   about λ. `curve ste/all_layers/all_experts_in_rect_frac --sweep rect_with_bias`,
   `curve vio/MaxVioGlobal --sweep rect_with_bias`.
   E=256 split also lost at w=0.1 against w=1 in all 24 pairs, but there the window was not empty on average
   (n_win 1.1–2.0 ≈ 160–2000 tokens per expert per step; `runs --sweep 26-08-07-experts256_split_router_weighter`).
   So the per-expert average alone does not decide it; per-expert coverage is not logged.
4. **At w=1 it works, and λ trades balance for loss monotonically.** With the softmax weighter, λ=0.01/0.1/1 gives
   MaxVio 2.0/1.06/0.69 and vf 3.2173/3.231/3.2459 (1wufeet2, dmgwweid, fgcb31fc). n_win falls from 16–160 at step 150
   to 2.5–4.3 later, i.e. ≈400–850 in-window tokens per expert per step. The best rect run is 0.007–0.009 behind the
   bar. Split quantile_correction shows the same cost of over-balancing: MaxVio floors at ≈0.5 for λ ≥ 0.1 while vf
   keeps rising (pj2yuna0 → buw6ue4o → fyfa9o83).
5. **A second failure mode is logit compression.** At λ=0.01 with the sigmoid weighter (igig8wdr), n_win ≈ 250 for the
   whole run and MaxVio climbs from 6 to 33–67. The same λ is fine with the softmax weighter, so λ did not transfer
   across weighter activations.
6. **Plain-router STE losses blow up above a narrow λ window.** For plain quantile_correction_ste:
   - softmax collapses at λ ≥ 0.01 (4h3onim0, znrn29va, hxbhuzzs);
   - sigmoid goes from MaxVio 5.4 at λ=0.001 (20qrebkv) to 1.0 at λ=0.01 (kcw5t9fv) to collapse at λ=0.1 (818l9pyq);
   - sqrtsoftplus at λ ≥ 0.1 ends at MaxVio ≈ 10 but vf 5.8–6.4 (avzl9mpp, o4afetxr).

   The failed runs sit at grad-norm 10–8000 for long stretches, against 0.05–0.35 for DeepSeek
   (`curve grad-norm --runs sxwv1bbi,kcw5t9fv,818l9pyq,znrn29va,avzl9mpp,o4afetxr`). With clip_grad=1 the LB gradient
   then sets the update direction of the whole network, and in the plain router it also flows into the residual stream.
   Even the healthy plain STE run (kcw5t9fv) is 0.028 behind the bar.
7. **The global grad-norm detects LB domination, but not the LB:LM ratio at the router.** At w=1 the step-50 grad-norm
   scales with λ (≈29/380/4000 for λ=0.01/0.1/1) and is back at the DeepSeek level by steps 300–600. Even so, λ still
   moves the final MaxVio 3× (`curve grad-norm --sweep rect_with_bias`). Muon normalizes the router's update, so the
   router-level LB/LM gradient ratio is what matters, and it is not logged.
8. **The noise floor has not been measured directly.** Every run uses seed 1000 in deterministic mode. The `26-09-30` runs
   with xc=1 are near-replicates of the `26-09-15` DeepSeek runs: they add a rank-1 residual router MLP and shift the
   init RNG. Their differences:
   - balanced pairs: 0.001–0.002 (sxwv1bbi/5xpxw33q, 9nmkq7oh/xkj4l6a2);
   - imbalanced pairs: 0.01–0.02 (hmrlk21e/ks79y359, us1gsihn/7h9qk7e1);
   - sigmoid bur=0.001 flipped from balanced to collapsed (v6ymgkl0 vs vbipfmkf).

   Working assumption: a difference < 0.003 is noise for balanced runs, and a single run near a stability edge is not
   reliable.
9. **Logit scale in plain softmax runs.** This is an inference for choosing w, since no plain rect run exists.
   top1 − top2 ≈ 0.4–0.8 and top1 − mean ≈ 1.7–3.2 (5w5e2lak, 8mqwfwau, x3e8nqq3;
   `curve router_logits/top1_pre_activation --runs 5w5e2lak,8mqwfwau,x3e8nqq3`, same for top2/avg). The K/K+1 gap is
   probably smaller than the top1/top2 gap. So w ≈ 1 should usually include the (K+1)-th expert, w ≲ 0.3 risks an empty
   window, and w=10 covers most experts, which is close to the `full` STE.

## Earlier settings with rect STE

10. **At E=64 k=8, width never mattered.** At fixed λ, moving w from 0.01 to 10 changed vf by ≤ 0.0035, and λ alone set the
    operating point (λ=0.001/0.01/0.1 → MaxVio ≈ 3/0.4/0.1; `26-06-09-rect_activations`, `26-06-08-dclm`). Switching the
    position between topk, topk_plus_one and midpoint changed vf by ≤ 0.003 (`26-06-14-rect_location_sqrtsoftplus`). At
    matched MaxVio, the rect, triangle, expert-bias and aux variants differed by ≈0.002 (`26-07-20-sweep_target_maxvio`,
    `26-07-23-sweep_target_maxvio_triangle`), which is about noise level. That is why methods looked alike there.
11. **The MaxVio-target controller worked at E=64.** Achieved MaxVio matched the target for targets 0.8–3.2 with every
    method, and down to 0.1–0.2 with rect (same sweeps, rate 1e-5). It is additive (λ ± rate per step;
    `_maybe_update_moe_balance_control_from_vio` in training.py) and acts on any LB type, including the new ones.
12. **At E=256 k=2 (plain), width matters non-monotonically, and occupancy is endogenous.**
    - sqrtsoftplus: w=0.1 and w=10 both reach 3.302–3.304, but w=1 only 3.348 (`26-08-02-experts256_ste_extra` vs
      `26-07-31-experts256_ste`; the configs differ only in width).
    - In `26-08-20-experts256_rect_ste_activations`, softmax is best at w=10, where n_win ≈ 240 of 256 (close to `full`).
      The best runs overall are sigmoid w=0.1 (qab0blqm, cpzs4dio: 3.286–3.288), level with the best baseline (DeepSeek
      sigmoid 3.2908, mzsft26j). cpzs4dio has n_win = 1.21, which looks sparse per token but is ≈430 in-window tokens
      per expert per step, about the same as every2 split w=1 (point 4).
    - At w=1, n_win depends on λ: 49–60 at λ ≤ 0.01 but 2.5 at λ=0.1 (t0u3hsji, drnjvnes, tf446bd7). The router reshapes
      its logit scale in response to the STE.
13. **Small boundary fractions (fixed_number_boundary_ste) were not small windows.** Each expert gets gradient from its
    M = fraction × 524288 closest tokens per step (weight 1). With position topk, every token's K-th expert sits at margin
    exactly 0. At E=256 each expert is the K-th choice for ≈2000 tokens per step, so for M < ~2000 (fraction < ~0.004) the
    radius is 0. The window is then all of the expert's K-th-choice tokens, which carry gradient because these runs
    detach the threshold. Only rarely chosen experts get a radius > 0 that reaches their nearest tokens, so every expert
    is always covered. Consistent with this, median MaxVio drops from 2.6 at fraction 0.001 to 0.45 at 0.005, where M
    passes ~2000. The best runs are similar across fractions (3.286–3.292;
    `runs --where "moe_router_load_balancing_type='fixed_number_boundary_ste' AND num_experts=256"`).
14. **detach_threshold with position topk is unstable.** 8 of the 18 detached runs collapsed (all at w ≤ 1), against 0 of 18
    attached (`26-08-20-experts256_rect_ste_activations`). The likely cause: the pinned K-th expert gets an STE gradient
    every step once the threshold is detached. coordinate_perturbation_ste also detaches internally, but it uses
    exact_margin, where no expert is pinned at 0.
15. **E=2048 with Adam and MoE in every layer** (`26-09-06-experts2048_centered_fsq`): the best plain rect without a bias is
    vf 3.393 (p76qsbq6), against DeepSeek 3.313 (36xemegh) and QB 3.324 (eziezjz9). Softmax n_win is 1.0–1.3 at w=0.1 and
    ≈2 at w=1, and λ=0.1 collapses or degrades (k0eou9pm, zqbrtnpw). Both the optimizer and the MoE frequency differ
    from every2, so this does not transfer directly.

## Confounded comparisons

- **λ in the split rect sweep** changes both the early LB transient (point 7) and the steady-state balancing strength.
- **λ interacts with weighter activation (point 5) and score function (point 6).** The safe λ moves by ≥ 10× between
  activations.
- **Score function in plain runs without a bias:** the STE margin is on the raw logits for every score function, so score
  function effects on the STE are indirect, through how the LM shapes the logit scale.
- **n_win across positions and kernels:**
  - `topk` adds the pinned K-th expert (+1 to n_win); `exact_margin` does not.
  - `higher_order_rect` logs occupancy over its full support ±w; rect logs it over ±w/2.
- **`26-09-06` vs every2:** the optimizer and the MoE frequency both changed.
- **Single seed everywhere:** every difference between configs also includes a different noise realization.

## Open questions

- What is the smallest plain-router w whose window is non-empty during the first ~600 steps? Point 9 suggests ≈1.
- Does a fixed-width window fail because some experts have no in-window tokens at all (they can never be pulled back),
  rather than because the average is low? The fraction runs (point 13) always cover every expert, and E=256 split w=0.1
  failed with a non-empty average (point 3). Neither per-expert coverage nor occupancy at other radii is logged.
- **Learnable biases in every2** (all with quantile_correction_ste, bias LR tied to λ):
  - expert_bias failed to balance in all 6 runs (MaxVio 187–995);
  - per_token_bias with sigmoid gave the best low-MaxVio plain-router results (eblyy2pd 3.218 @ 0.63; lciuk82a 3.248 @ 0.5)
    but failed at λ ≤ 0.01 and with softmax at λ=0.1.

  Does the per-token bias help the new estimators too? `runs --sweep experts2048_every2_quantile_correction_with_bias`
- Is vf vs MaxVio monotone in every2 (split rect, point 4), or U-shaped (plain aux: for λ=0.1/1/10, MaxVio is 12/1.7/1.1
  and vf 3.2322/3.2206/3.2322; m7c3vpen, 5w5e2lak, 8mqwfwau)?
- Why is w=1 worse than both w=0.1 and w=10 at E=256 plain (point 12)?
- **Does λ transfer between the new estimators?** In `moe_utils.py`, each in-window (token, expert) pair gets the backward
  weight λ·2E/D²·(1/w) times a variant-specific factor:
  - centered_fsq: c_i − D/E (centered);
  - coordinate_perturbation_ste: n_i − n_j (centered, a difference of two counts);
  - exact_jump_ste: n_i (not centered; ≈ D/E = 512 here).

  In exact_jump the large common part cancels only between the K-th and (K+1)-th experts. Any other expert inside the
  window keeps it. Prediction: λ transfers within ~2× among centered_fsq (topk or exact_margin), higher_order_rect and
  coordinate_perturbation, but not to exact_jump, whose finite-width bias also grows with n_win.
- Does the size of the early LB transient (point 7) affect the final result?
