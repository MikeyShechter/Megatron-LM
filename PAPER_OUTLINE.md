# Paper Outline

A map of the paper: what each section is for, what it holds now, and what is
still open. It is the map for stage 1 and the checklist for stage 1.5 (see
`PAPER.md`).

Claude drafted it on 2026-11-18 from the tex files. The user corrects it. The
user also edits the paper directly, so this file can fall behind. Check it
against the tex before relying on it. Update it when the structure changes.

Marks:

- (note): taken from a `\mikey{}` note or an outline item in the tex.
- (confirm): Claude's reading. The user has not confirmed it. Do not treat it
  as decided. The "For" lines are also Claude's reading.

## Story

From the introduction outline (the user's items, shortened):

1. MoE is important.
2. Load balancing (LB) is important. Loss-free methods are motivated by not
   interfering with the LM loss.
3. The aux loss is a naive STE of the LB objective. We propose an STE designed
   for LB: Quantile Correction (QC).
4. Treating the aux loss as the objective itself is incorrect. Optimizing the
   aux loss with an STE gives bad balance.
5. We get good performance.
6. Variants? (MaxVio objective, separate router weighter.) The question mark is
   the user's.

## Front matter (`main.tex`)

- Title: "Direct Load Balancing". Abstract: placeholder.
- Template: NeurIPS 2026. The target is ICML, and arXiv comes first.
- Order: Introduction, Related Work, Method, Experiments, Discussion. Appendix:
  Hyperparameter Sweep Results, NeurIPS checklist.
- Open (confirm): the name. The title and the Method opening say "Direct Load
  Balancing (DLB)". The Method now builds up to "Quantile Correction (QC)". A
  note asks about "Quantile rect-STE (QRSTE)".

## 1. Introduction (`sections/introduction.tex`)

- For: the motivation, the idea, the claims.
- Now: the six-item outline above. No prose yet.

## 2. Related Work (`sections/related-work.tex`)

- For: place the work among MoE load-balancing methods and STE work.
- Now: reading notes in lists, not prose. The groups:
  - STE papers, with Daniel's notes (Bengio et al. is the identity STE, BNN is
    the rect STE, Quest).
  - The four Shazeer papers: sparsely-gated MoE, GShard, Switch, ST-MoE.
  - LB alternatives to the aux loss: DeepSeek loss-free, global LB ("Demons in
    the Detail"), BASE layers.
  - Big-model papers: Qwen2.5, OLMoE, DeepSeekMoE.
  - Perfect balance: expert choice routing, hash layers.
  - STE in MoE training: "Bridging Discrete and Backpropagation", sparse
    backpropagation for MoE, GRIN, unbiased gradient estimation with balanced
    assignments.
  - Orthogonal-expert methods, and a list of other papers to cite.
  - Unsure: ReMoE. Reading notes on MoE sparsity for reasoning tasks and on
    Skywork-MoE.
- Open:
  - (note, in the Method) Earlier work that used the identity STE still needs
    organizing.
  - (confirm) QB is used in the Method but has no entry in these notes. LongCat,
    ZAYA1 and the prior art from the literature searches are not in the notes
    either.

## 3. Method (`sections/method.tex`)

- For: derive QC as an STE of the squared load objective. The aux loss appears
  as the identity-STE case.
- Now, active text:
  - Opening paragraph. It still describes the old structure (DLB). It points to
    labels that exist only in commented-out text.
  - Preliminaries and notation:
    - MoE layer: router, logits, probabilities (softmax or normalized sigmoid),
      top-K, gate, output.
    - Auxiliary LB loss: f_e, P_e, the loss.
    - QB bias computation.
  - Squared load balance objective:
    - The objective and its gradient. The missing piece is the gradient of f_e.
    - Identity STE. It gives the aux loss gradient.
    - STE through indicators: routing margin, indicator form, Dirac delta.
    - Rectangular-window STE, with its pros and cons.
    - Quantile Correction (QC).
- Now, commented out: the old DLB derivation, the DLB variants (MaxVio,
  uncentered, total violation, squared MaxVio), the specialization subsection.
- Open (note):
  - An appendix on optimizing the aux loss with a better STE: results and an
    explanation.
  - How to describe the weakness of the identity STE: "estimation error",
    "noise", or only the MoE intuition.
  - "STE through indicators": keep it as a paragraph? Rename it? Say that the
    identity STE through the indicator differs from the one through top-K.
  - Rect STE: where to discuss the probability spread. Show empirically that the
    spread changes during training.
  - The name of QC.
  - QC and LM interference: cite the batch results. Maybe add runs with a 4x and
    a 16x bias.
- Open (confirm): choices that the runs fix and the text does not state yet.
  - Where the LB gradient enters (post or full post).
  - Detaching the threshold.
  - Mass normalization.
  - Attached or detached.
  - The half-token fix.
  - The number of bias iterations.
  - The final loss with its coefficient.

## 4. Experiments (`sections/experiments.tex`)

- For: the setting, the baselines, the results.
- Now: old text and old tables (modded-nanogpt, E=64, K=2 and 16). They will be
  replaced. Only Megatron results go in the paper.
- Needed for stage 1.5 (confirm):
  - Setting: code base, model, MoE configuration (E, K, which layers, router
    activation), data, steps and tokens, batch size, optimizer, hardware,
    parameter counts.
  - Metrics: validation loss, MaxVio (end of run and whole run), local
    (dispatch) MaxVio. How the reported run of each method is chosen.
  - Baselines: aux loss, DeepSeek loss-free, QB, LongCat, ZAYA1, ID balancing.
  - Settings with runs (checked in runsdb on 2026-11-18): E=2048 K=2, which has
    almost all the runs, and E=64 K=8. Planned: E=768 K=3, and a large run with
    about 1B active parameters.
- Result groups on the results pages (confirm which go in):
  - Baselines, and the best run per method.
  - QC design choices: margin, normalization, activation, pre or post, detach,
    kernel (identity, rect, QC), half-token fix, attach, bias iterations.
  - Local against pooled balance.
  - Single-batch experiment: the LM cost of balancing with QC and with aux.
  - Bias and variance of the STE kernels (Monte Carlo).
  - Dead experts. Training MaxVio curves.
  - Other groups: split router-weighter, QB estimators, QB with the LM STE,
    balanced init, QC with QB routing, squared QC.
- Tables: the user adds them, or asks Claude to build them from the logs.

## 5. Discussion (`sections/discussion.tex`)

- Now: a three-item outline.
  1. Token dropping is not covered. Using an STE with such a practical metric
     would be interesting.
  2. Even the large runs are not that large.
  3. There may be better STE estimators.

## Appendix

- Hyperparameter Sweep Results (`sections/hyperparameter_sweep_results.tex`,
  `tables/`): old modded-nanogpt sweeps.
- Checklist (`sections/checklist.tex`): the NeurIPS template.
- Planned (note): the appendix on optimizing the aux loss with an STE.
