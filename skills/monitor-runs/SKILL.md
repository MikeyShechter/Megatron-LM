---
name: monitor-runs
description: Check on the user's training runs. Reports what is queued or running in SLURM (their `sq`), how far each run is and roughly when it will finish, whether it will hit the 8 h time limit, and whether runs that left the queue finished or crashed (and why), for the experiment sets not yet synced to wandb. One read-only command, tools/runsdb/monitor_runs.py. Use it whenever the user asks about the status of runs, jobs or sweeps ("how are my runs doing", "are they done", "when will it finish", "did anything crash", "what's in the queue", "can I sync yet"), even if they don't say "monitor" or "sq".
---

# monitor-runs: queue, progress, ETA, and how runs ended

Run this on the login node. It is read-only: it never submits, cancels, reruns or syncs.

```bash
tools/runsdb/monitor_runs.py            # queued/running jobs + all run dirs of the unsynced sets
tools/runsdb/monitor_runs.py --set RE   # any set whose name matches RE, synced or not
tools/runsdb/monitor_runs.py -v         # plus error lines and log path of failed runs
```

## What it prints

First a header with the time and the job counts per SLURM state. Then, for each relevant experiment set, one line
per run dir:
- **RUNNING:** `it N/M (p%)`, wall seconds per iteration over the last 300 iterations (evals and checkpoint saves
  included), and the ETA (remaining iterations × that rate + ~5 min for the final validation/test eval). It also
  shows the last val loss and the SLURM job id. Flags, in capitals:
  - `WILL HIT THE 8h00m TIME LIMIT` (elapsed + remaining > limit);
  - `NO NEW ITERATION FOR …` (20+ min without a log line: hung, or a node problem);
  - `NaN ITERATIONS IN LOG`.
- **PENDING:** the SLURM reason (Priority, Resources, …) and the expected start time, if the scheduler has one.
- **done:** the `done` marker exists (training writes it at the very end), plus the final val and test loss.
- **exited:** stopped early on purpose. The log says `exiting program at iteration N` and has no errors. This is the
  normal end of smoke runs with `exit_interval`; there is no `done` marker.
- **FAILED:** not in the queue and no `done`. Shows the last iteration reached, whether a checkpoint exists
  (`resumable with -r`), and the first error line of the newest log.
- **no log:** never started, or only a dry run created the dir.

A set whose runs all ended normally (done or exited) gets the line `all runs ended normally: ready to sync`. The
sync-runs skill can then upload it; a smoke set can be deleted instead, if the user prefers.

## How it decides what to look at (and how to do it by hand)

- **Queue:** `squeue -u $USER -o "%i|%j|%T|%M|%l|%R|%S"`. The user's `sq` is an alias for `squeue` with a wide
  format, and aliases don't exist in Claude's shell. submit_multiple.py names jobs `<set>_<NNN>`, so a job maps to the
  run dir `/e/data1/mmlaion/shechter1/checkpoints/megatron/<set>/<NNN>/`, whose log is `<jobid>.out`.
- **Relevant sets:**
  - sets with queued or running jobs;
  - sets that `tools/runsdb/wandb_sync.py check` classifies as not yet synced;
  - sets from the last 3 days with no wandb file at all. A run that died at startup, e.g. from an argparse error,
    never creates one, so the sync check can't see it.
- **Progress:** from the log lines
  `[2026-09-30 18:50:31.706] iteration   10/ 6000 | ... | elapsed time per iteration (ms): 4823.5 | ...`. The wall
  rate comes from the timestamps; the per-iteration `elapsed time` excludes evals and is too optimistic.
- **Errors:** the first lines matching:
  - `CANCELLED ... DUE TO TIME LIMIT`;
  - `ERROR:megatron...`, e.g. the rerun state machine stopping on NaN gradients, exit code 16;
  - `error: unrecognized arguments` (a bad config key);
  - `…Error: ` / `…Exception: `;
  - `out of memory`;
  - `Bus error`;
  - NCCL timeouts.

  The torch-elastic summary at the end of a crash log (`ChildFailedError`, `srun: error: … exit code 1`) is generic,
  so the real cause is earlier in the log.

## What to tell the user

- **Running:** report progress per set, not all lines. Give the earliest and latest ETA, and anything flagged in
  capitals.
- **Failed:** give the cause and whether it can resume. A run killed by the time limit, or a transient crash such as
  a node or NCCL failure, resumes from its checkpoint with
  `submit_multiple.py <set dir> -s run_megatron.sh -r` (see the run-experiments skill). Only resubmit when the user
  asks. An OOM, an argparse error or NaN gradients will fail again until something changes.
- **All done:** suggest syncing (sync-runs skill).
- **Early metrics** (val loss, MaxVio, STE window) of runs still in progress:
  `tools/runsdb/ingest_local.py <set dir>`, then the usual runsdb queries. The ingest reads the offline wandb files
  of running jobs, and the next sync replaces those rows.
