---
name: runsdb
description: Look up past training runs of this repo from a fast local mirror of the wandb project megatron-moe (~2100 MoE load-balancing runs since 2026-06), covering sweeps, configs, tags, val/train loss curves, MaxVio and router balance, STE rect fractions, iteration time, and downstream task scores. Use this whenever the user asks about older or recent runs, sweeps, experiments, results, comparisons between methods or hyperparameters, metric values or curves, or what has already been tried, even if wandb isn't mentioned. Prefer it over the wandb API, which is much slower and costs far more tokens. Also use it to refresh the mirror after new runs, from wandb or straight from run dirs.
---

# runsdb: local mirror of the megatron-moe wandb runs

All data is local parquet queried with DuckDB, so each query takes about a second and prints a small table.
- Data: `/e/data1/mmlaion/shechter1/runsdb/megatron-moe/`
- Code: `tools/runsdb/`. The scripts' shebang points at the megatron-submit env, so call them directly.

## Start here

1. `tools/runsdb/q.py status` shows how many runs are mirrored and when the last sync was. If the user asks
   about runs newer than that, refresh first (see "Refreshing" below).
2. Get oriented with `q.py sweeps --since 2026-09` or `q.py sweeps --grep quantile`. It prints one line per sweep:
   date, name, #runs, fixed setup, the config keys the sweep varied, and tags. `sweeps.md` in the data dir
   has the same for every sweep. Grep it instead of reading it whole.

## Commands (`tools/runsdb/q.py ...`; output is capped at `-n` rows, default 40)

| goal | command |
|---|---|
| runs of a sweep with their varied settings and key results | `q.py runs --sweep 26-09-30-experts2048 --sort val_loss_final` |
| pick other columns | `q.py runs --sweep X --cols moe_aux_loss_coeff,val_loss_predecay,maxvio_last5` |
| filter on config | `q.py runs --where "num_experts=2048 AND moe_router_load_balancing_type='aux_loss'" --since 2026-09-01` |
| curve at a few steps | `q.py curve vio/MaxVioGlobal --sweep X` (9 evenly spaced points by default) |
| the complete val loss curve | `q.py curve val/regular/lm_loss --runs id1,id2 --all` (all 40 eval points) |
| value at chosen steps | `q.py curve val/regular/lm_loss --sweep X --steps 1500,3000,4200,6000` |
| full resolution (spikes, exact steps) | `q.py curve grad-norm --runs id --raw --steps 2990,3000,3010` · `q.py sql "SELECT _step, \"grad-norm\" FROM raw('id') ORDER BY 2 DESC LIMIT 5"` |
| which metrics exist, since when | `q.py keys router_balance` (add `--full` for one row per key) |
| config of a run / where runs differ | `q.py config id` · `q.py config id1 id2 id3` · `q.py config id --grep lr` |
| anything else | `q.py sql "SELECT ... FROM runs JOIN eval USING (run_id) ..."` |

Selection flags work with `runs` and `curve` and are ANDed: `--sweep RE`, `--tag RE`, `--name RE`, `--runs id,id`,
`--since YYYY-MM-DD`, `--where "SQL on runs columns"`. `curve` prints one column per run when there are 8
runs or fewer, otherwise one row per run. `--plot out.png` needs matplotlib, which megatron-submit doesn't
have. Ask the user before installing it.

## Data model (SQL table names)

- `runs`: one row per run.
  - Identity: `run_id` (wandb id), `sweep` (run name without the `NNN_` prefix and the `+label`; the sweep dir name, which
    starts with YY-MM-DD), `idx` (NNN), `label` (the sweep's varied params, abbreviated as in the run name),
    `name`, `tags` (comma-joined; the user's experiment labels), `started` (time of the first logged step;
    `--since` uses it; wandb's createdAt is the upload time for offline-synced runs), `state`, `runtime_h`, `steps` (last
    logged step), `rows`, `decay_start` (first WSD decay iteration), `source` (wandb|local), `local_dir` +
    `local_exists` (whether the run/checkpoint dir is still on disk).
  - Config columns: Megatron arg names, for the keys that differ between runs plus the setup keys.
  - Summary metrics: `<m>_final` (last value), `<m>_last5` (mean of the last 5 values), and
    `<m>_predecay` (last value at or before `decay_start`), for each `<m>`:
    - `val_loss`: val/regular/lm_loss
    - `maxvio`: vio/MaxVioGlobal
    - `maxvio_dispatch`: val/router_balance/max_vio_dispatch_mean
    - `rect_frac`: ste/all_layers/all_experts_in_rect_frac
    - `tasks_avg`: tasks/average

    Also `test_loss` (test/lm_loss) and `iter_time_med` (median iteration-time after step 100, seconds).
- `eval` (run_id, step, key, value): every logged point of the eval-cadence keys (val/\*, vio/\*, ste/\*,
  tasks/\*, task_entropy/\*, router_logits/\*, test/\*, ...). Values are exact.
- `train` (same columns): keys logged every step or every 10 steps (`lm loss`, `grad-norm`, train/\*, `iteration-time`,
  ...), averaged over 50-step windows. `step` is the window's last step.
- `raw('<run_id>')`: the full history of one run, one column per key. Quote keys:
  `SELECT _step, "vio/MaxVioGlobal" FROM raw('7h9qk7e1') WHERE "vio/MaxVioGlobal" IS NOT NULL`.
- `sweeps`, `keys`: what `q.py sweeps` and `q.py keys` print.
- `meta/<run_id>.json`: the full config (~800 keys), summary, and per-key value counts.

## Conventions and pitfalls

- The usual recent setup is 6000 iters with eval every 150 (40 eval points). WSD decay (`lr_wsd_decay_iters`=1714)
  starts at 4286, so the last pre-decay eval is at 4200. `test/*` is logged once at the end. Check
  `steps`/`decay_start` instead of assuming; crashed or short runs exist, so compare `steps` before
  comparing `_final` values.
- Older runs lack newer metrics. `-`/NaN means "not logged", not zero. `q.py keys <regex>` shows the first and
  last date each metric was logged. 69 runs have no history at all (`rows = 0`).
- Renamed key: the 17 earliest runs log `val/lm_loss`. The derived tables copy it into `val/regular/lm_loss`
  (raw keeps the original). Add more renames to `ALIASES` in `tools/runsdb/common.py`.
- `lm loss` (with a space) is Megatron's per-step train loss. Prefer `val/regular/lm_loss` over `lm loss validation`:
  the latter is missing in ~90 runs, and the end-of-run test eval overwrites it at the final step.
- `tasks/*` are answer-token cross-entropy losses (lower is better), not accuracies. `task_entropy/*` is router
  entropy, not LM entropy.
- For what a metric means and how it is computed (with file:line references), read `references/metrics.md`.
- Keep output small: narrow the selection, use `--cols`, and use `--steps` when comparing many runs.
  `--all` is meant for a few runs.

## Refreshing (login node only; compute nodes are offline)

- To upload finished runs to wandb and then refresh in one go, use the sync-runs skill (`tools/runsdb/wandb_sync.py`).

- After the user syncs runs to wandb: `tools/runsdb/sync.py`. It lists all runs in ~25 s, downloads only new or changed runs
  (~0.2 s each with 8 workers), and rebuilds the derived tables.
- Straight from run dirs, before or without `wandb sync`: `tools/runsdb/ingest_local.py <sweep_dir> [...]`.
  It reads `<run>/wandb/wandb/offline-run-*/run-*.wandb`. The next `sync.py` replaces these runs with wandb's copy.
- `tools/runsdb/build.py` rebuilds only the derived tables (~15 s). Run it after editing `SUMMARY_METRICS`,
  `ALIASES` or `TRAIN_WINDOW` in `common.py`. No download is needed, because raw/ has every key at every step.
- Not mirrored on purpose: system metrics (GPU stats), `output.log`, model artifacts. If ever needed, use the
  wandb API on that one run (`wandb.Api().run("mikeyshechter/megatron-moe/<id>")`).
