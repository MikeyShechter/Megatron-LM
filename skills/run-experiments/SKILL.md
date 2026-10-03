---
name: run-experiments
description: How to set up and submit training experiments in this research fork. Covers writing a sweep config YAML as a copy of a clean base config (configs/2048_experts_fd_base.yaml for the current finite-difference STE plan) with only the experiment's values changed, previewing the grid, and submitting with submit_multiple.py + run_megatron.sh from the megatron-submit conda env (SLURM, 4 nodes x 4 GPUs, ~6 h per run). Also covers how runs are named and laid out, rerunning failed or time-limited runs, and the gotchas that silently break a sweep. Use it whenever the user wants to run, launch, submit, queue, rerun or prepare an experiment, sweep, ablation or config, even if submit_multiple.py isn't mentioned, and prefer it over the generic run-on-slurm skill. Submitting needs the user's explicit go-ahead.
---

# run-experiments: config → submit_multiple.py → SLURM

A sweep goes through these steps:
1. A YAML config lists values for Megatron arguments.
2. `submit_multiple.py` expands the config into a grid, writes one run dir with a `spec.yaml` per grid point, and
   `sbatch`es `run_megatron.sh <spec.yaml>` for each.
3. Each job trains in the container and logs to wandb offline.
4. Afterwards:
   - follow progress with the monitor-runs skill;
   - when everything has finished, upload with the sync-runs skill;
   - analyse with the runsdb skill.

Only submit when the user explicitly asks. Preparing configs and dry runs is fine without asking.

## 1. Write the config

**Copy a clean base config** to a new file under `configs/`, e.g. `configs/2048_experts_fd_<sweep>.yaml`, and change
only `job_details.name` and the sweep's experiment keys. Leave data, model, optimizer, schedule, seed,
`deterministic_mode` and the eval cadence alone, so that results stay comparable with earlier sweeps. Keep the header
comment saying what the sweep is and what changed from the base.

- **`configs/2048_experts_fd_base.yaml`** is the base for the finite-difference STE plan (`plan.md`). It is the
  plain-router "every2" setting (E=2048, k=2, MoE every 2nd layer, Muon, 6000 iters) set to run the anchor V0:
  centered_fsq + rect at topk, w=1, λ=0.01, MaxVio controller off, controller start iter 300.
- Examples built from it:
  - `configs/2048_experts_fd_smoke.yaml`: a LIST over the 8 variants, plus `exit_interval`;
  - `configs/2048_experts_fd_phase1.yaml`: a LIST over 4 groups with different controller settings.
- **`configs/2048_experts.yaml` and `configs/2048_experts_split_router_weighter.yaml` are not clean.** They are
  edited for each sweep, so they still hold the last sweep's experiment values. 2048_experts.yaml still has the
  DeepSeek setup: expert bias on, three bias update rates, extra router compute, two score functions. Every leftover
  list multiplies the grid, and leftover switches change the method. In particular, `moe_router_enable_expert_bias`
  makes the MaxVio controller move the bias update rate instead of λ. Starting from one of these, reset every key
  that isn't part of the new experiment, and compare against a reference run with
  `tools/runsdb/q.py config <run_id> --grep <key>`.

**Set `job_details.name`** to `experts2048_every2_<what this sweep tests>`. The sweep dir is
`<output_dir>/<YY-MM-DD>-<name>`, and that dir name is also the sweep name in runsdb. Reusing a name on the same day
raises `FileExistsError`. Leave `output_dir` as it is.

**Grid rules** (submit_multiple.py):
- Every value is a list, and the grid is the Cartesian product of all lists. Never write a bare string such as
  `key: softmax`: string values are passed to `eval()`. Write `key: ["softmax"]`.
- Keys that must vary together go in a `LIST_<anything>:` block: a list of sub-grids whose union forms one axis.
  Several LIST blocks multiply like normal keys. Every sub-grid in a block must list the same keys in the same order,
  otherwise the grid summary's assertion fails. Fill keys a group doesn't use with neutral values, e.g.
  `moe_balance_update_rate: [0.0]` turns the controller off.

  ```yaml
  LIST_ESTIMATOR:
    - moe_router_load_balancing_type: ["centered_fsq"]
      moe_ste_rect_poistion: ["exact_margin"]
    - moe_router_load_balancing_type: ["coordinate_perturbation_ste"]
      moe_ste_rect_poistion: ["exact_margin"]
  load_balance_ste_width: [1.0, 3.0]        # 2 x 2 = 4 runs
  ```
- Keys are Megatron argument names in snake_case: `--moe-aux-loss-coeff` becomes `moe_aux_loss_coeff`. Booleans:
  `true` adds the flag and `false` omits it, so a default-true flag cannot be turned off this way.
- Check that every new key exists. Many flags are generated from config dataclasses, with names in either quote
  style, so `arguments.py` alone is not enough:
  `grep -rnE -- "--<key-with-dashes>[\"']|\b<key>\b" megatron/training megatron/core/transformer/transformer_config.py megatron/core/model_parallel_config.py`.
  A misspelled key kills every job at startup with an argparse error. Note that `moe_ste_rect_poistion` is spelled
  that way.

**Run naming.** Keys that differ between runs form the run label, using the first 3 letters of each word
(`moe_aux_loss_coeff=0.1` becomes `moe_aux_los_coe=0.1`). The label appears in the wandb run name
`NNN_<sweep>+<label>` and in runsdb's `label` column. SLURM job names are `<sweep>_<NNN>`. With LIST blocks the run
numbers don't follow the file order, so identify runs by their label.

## 2. Before submitting

- **New code paths have never run** (new flags, new LB types). Run the relevant unit tests in the container (testing
  skill; needs a GPU node), and submit a short smoke sweep first.
  - Add `exit_interval: [300]` to the config. That keeps the full LR schedule and stops after the evals at 150 and 300,
    followed by the usual final validation and test eval. `configs/2048_experts_fd_smoke.yaml` is an example.
  - Smoke runs end without a `done` marker; monitor-runs shows them as `exited`.
  - Afterwards, sync or delete the smoke set. An unsynced set older than the newest synced one blocks sync-runs.
- **The job uses the code as it is when the job starts, not when it was submitted.** `run_megatron.sh` mounts the repo
  at `/workspace`. Editing code, or switching branches, while jobs are queued changes what they run.
- **Cost:** each run uses 4 nodes × 4 GPUs (account reformo, partition booster) for ~6 h in every2 (5.6–6.1 h). The
  limit is 8 h (`#SBATCH --time=480`). The dry run prints the number of runs.
- **Duplicates:** check whether runsdb already has the run:
  `tools/runsdb/q.py runs --where "num_experts=2048 AND moe_router_load_balancing_type='...'"`.

## 3. Submit (only on the user's explicit request)

Run from the repo root, because relative paths such as `configs/dclm_baseline_gpt_neox_paths.txt` resolve there:

```bash
cd /e/project1/laionize/shechter1/repos/Megatron-LM
P=/e/project1/laionize/shechter1/miniforge3/envs/megatron-submit/bin/python
$P submit_multiple.py configs/<file>.yaml -s run_megatron.sh -d -y   # preview: "Ready to run N configurations: a x b ..."
$P submit_multiple.py configs/<file>.yaml -s run_megatron.sh -o -y   # submit for real
```

The user's own way is `conda activate megatron-submit` followed by
`python submit_multiple.py configs/<file>.yaml -s run_megatron.sh`, which waits for Enter after printing the grid.
Claude's shell can't press Enter, hence `-y`.

**The `-d -y` / `-o -y` pairing:** a dry run still writes `<sweep dir>/NNN/spec.yaml`, so the real submission on the
same day needs `-o`, which deletes the sweep dir first. Use `-o` only on a dir you just dry-ran. On a sweep with
queued, running or finished runs it deletes their checkpoints and logs.

| flag | meaning |
|---|---|
| `-s run_megatron.sh` | sbatch script (always this one) |
| `-y` | don't wait for Enter |
| `-d` | dry run: no sbatch, but spec dirs are written |
| `-o` | delete the existing sweep dir first |
| `-r` | rerun (section 5) |
| `-c N` | N runs per SLURM job (keep 1) |
| `-n N` | skip the first N configurations |

## 4. What a sweep produces

`/e/data1/mmlaion/shechter1/checkpoints/megatron/<YY-MM-DD>-<name>/` contains `job.yaml` (a copy of the config) and
one dir per run, `NNN/`, holding:
- `spec.yaml`: this run's flat arguments.
- `<jobid>.out`: the Megatron log. A rerun adds another one.
- `iter_XXXXXXX/` and `latest_checkpointed_iteration.txt`: the newest checkpoint (saved every 150 iterations, only the
  latest kept), plus the pre-decay one at the WSD decay start (4286 for 6000 iters).
- `wandb/`: offline wandb files.
- `done`: written by training at the very end. Without it, the run did not finish.

## 5. Rerun failed or killed runs

```bash
$P submit_multiple.py <sweep dir> -s run_megatron.sh -d -r -y   # lists the runs without `done`
$P submit_multiple.py <sweep dir> -s run_megatron.sh -r -y      # resubmits them
```

This resubmits every run dir that lacks `done`. Megatron resumes from that dir's latest checkpoint, because save and
load are both the run dir. Check the queue first, so that runs that are still queued or running aren't submitted
twice. A run killed by the 8 h limit resumes the same way. Only rerun when the user asks.

## 6. After submitting

- **Check the first log within a few minutes** (monitor-runs skill). Argparse errors and bad paths fail at startup,
  before any wandb file exists.
- **Then sync:** when all runs of a sweep are done, upload them with the sync-runs skill, which also refreshes runsdb.
- **Tags:** wandb tags are not set by the config; nothing in the submission path sets them.

## 7. Flags of the current research (load balancing)

- `moe_router_load_balancing_type`: centered_fsq | exact_jump_ste | coordinate_perturbation_ste |
  quantile_correction_ste | aux_loss | none (DeepSeek, with `moe_router_enable_expert_bias`) | ...
  coordinate_perturbation_ste requires `moe_ste_rect_poistion: ["exact_margin"]`.
- `load_balance_ste_type`: rect | higher_order_rect | triangle | tanh | full.
- `load_balance_ste_width`: the STE width w, in logit units for a plain router without biases.
- `moe_ste_rect_poistion`: topk | topk_plus_one | midpoint | exact_margin.
- `moe_aux_loss_coeff`: λ.
- MaxVio controller:
  - `moe_balance_target_vio`: target MaxVio.
  - `moe_balance_update_rate`: step size; 0 = off.
  - `moe_balance_update_mode`: additive | multiplicative.
  - `moe_balance_update_start_iter`: hold λ fixed until this iteration.
- Learnable biases:
  - `moe_learnable_bias_type`: none | expert_bias | per_token_bias.
  - `tie_learnable_bias_lr_to_aux_loss_coeff`: makes the bias LR follow λ.
  - `moe_learnable_bias_lr_mult`: can't be combined with the tie.
- `moe_router_score_function`: softmax | sigmoid | sqrtsoftplus.
- `moe_router_use_separate_weighter`: split router-weighter.

The experiment plan and its decision rules are in `plan.md`. Metric definitions are in
`skills/runsdb/references/metrics.md`.
