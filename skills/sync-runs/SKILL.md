---
name: sync-runs
description: Upload finished experiment runs from the checkpoint dir (/e/data1/mmlaion/shechter1/checkpoints/megatron) to wandb with `wandb sync` in a detached background process, then refresh the local runsdb mirror. Use this whenever the user asks to sync or upload runs, experiments or sweeps to wandb, says new experiments finished and should be synced, or wants wandb and the local db updated with new runs. It first checks which experiment sets are already synced. If older sets are unsynced while newer ones are synced, it starts nothing and reports them.
---

# sync-runs: offline runs → wandb → runsdb

Training runs log to wandb offline into `<set>/<run>/wandb/wandb/offline-run-*/`, because compute nodes have no
internet. This skill uploads them from the login node and then pulls them into the local mirror (see the runsdb
skill). Everything goes through `tools/runsdb/wandb_sync.py`.

## 1. Check (foreground, ~3 s)

`tools/runsdb/wandb_sync.py check` reports the experiment sets that matter and exits with:
- 0: there are runs to sync and nothing is blocking. Go to step 2.
- 1: nothing to sync. Tell the user, and mention any sets still waiting for SLURM jobs.
- 2: blocked. Go to step 3.

## 2. Sync in the background

1. Run `tools/runsdb/wandb_sync.py start`. It checks again, launches a detached worker in its own session, and
   returns at once. The worker keeps running if the SSH connection or this session dies. It runs `wandb sync` on
   each unsynced offline-run dir (4 in parallel; uploading takes minutes), then `tools/runsdb/sync.py` to update
   the local db.
2. Right after, run `tools/runsdb/wandb_sync.py wait` with the Bash tool's `run_in_background: true`. It exits
   when the worker finishes, and that notifies you. Tell the user it started (which sets, how many runs) and
   don't block on it.
3. When notified, relay the summary: runs uploaded or not, and what runsdb downloaded. If something failed, read
   that run's section of the log, which starts with a `=== wandb sync <dir>` header.

If the session was interrupted, `tools/runsdb/wandb_sync.py status` shows progress or the final summary. The
worker runs on the login node where it was started. Status and logs are in
`/e/data1/mmlaion/shechter1/runsdb/megatron-moe/wandb_sync/`.

## 3. Blocked: ask, don't act

An unsynced set that is older than the newest synced set is usually an abandoned or crashed experiment, and the
user wants to decide about it. Start nothing. For each blocked set, report what the check printed: unsynced vs.
total offline runs, how many run dirs have `done`, and the last activity. Then ask whether to delete or sync it.
- **Sync it:** run `tools/runsdb/wandb_sync.py start --include <set> [<set> ...]`, then continue with step 2.2.
  `--include` approves only the sets you name.
- **Delete it:** run `rm -rf /e/data1/mmlaion/shechter1/checkpoints/megatron/<set>` for exactly the sets the user
  named, then run `check` again. This also removes the checkpoints and cannot be undone, so only do it when the
  user asks. If a set is partly synced, first ask whether they mean the whole set or only its unsynced offline-run
  dirs (`check --list` prints those).

## How the check decides

- **Synced offline run:** it has a `run-<id>.wandb.synced` marker (written by `wandb sync`), or its id is in runsdb.
- **Synced set:** all its offline runs are synced, or runsdb has at least as many runs of that sweep as the set
  has offline runs. The second case covers some older sets that were uploaded under new run ids, so their markers
  and ids don't match. It also prevents duplicate uploads.
- **Active runs are skipped.** These are runs whose SLURM job (named `<set>_<NNN>`) is still queued or running.
  Syncing a run that is still being written would mark it synced, and the rest of it would never be uploaded.
- **Ordering:** sets are ordered by last activity (newest `.wandb` write), not by name. So a set that was still
  running when a newer one got synced is not flagged later.
