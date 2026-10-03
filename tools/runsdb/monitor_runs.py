#!/e/project1/laionize/shechter1/miniforge3/envs/megatron-submit/bin/python
"""Status of queued, running and recently finished training runs. Read-only: never submits, cancels or syncs.

  monitor_runs.py            the user's queued/running jobs, plus every run dir of the experiment sets that are
                             not yet synced to wandb (and recent sets without any wandb file: startup failures)
  monitor_runs.py --set RE   every experiment set whose name matches RE, synced or not
  monitor_runs.py -v         also print the error lines and the log path of failed runs

Running job: progress, wall seconds per iteration over the last RATE_WINDOW iterations (evals and checkpoint saves
included), estimated finish time, whether that exceeds the SLURM time limit, last val loss.
Run dir no longer in the queue: done (it has the `done` marker that training writes at the very end) or failed
(last iteration reached, first error lines of its log, and whether a checkpoint allows a rerun with -r).
"""

import argparse
import getpass
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timedelta

import common as C
import wandb_sync

ITER_RE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[.\d]*\] iteration\s+(\d+)/\s*(\d+) \|")
VAL_RE = re.compile(r"validation loss at iteration (\d+) on validation set \| lm loss value: ([0-9.Ee+-]+)")
TEST_RE = re.compile(r"validation loss at iteration \d+ on test set \| lm loss value: ([0-9.Ee+-]+)")
NAN_RE = re.compile(r"number of nan iterations:\s+(\d+)")
EXIT_RE = re.compile(r"exiting program (?:at iteration \d+|after [\d.]+ minutes)")  # exit_interval / exit_duration
ERROR_RE = re.compile(
    r"DUE TO TIME LIMIT|CANCELLED AT|ERROR:|error: unrecognized arguments|error: argument |out of memory|OutOfMemory"
    r"|Bus error|Segmentation fault|NCCL.*(?:timeout|Timeout|Error)|\b\w*(?:Error|Exception): "
)
IGNORE_RE = re.compile(r"ChildFailedError|To enable traceback|error_file:|Warning")
RATE_WINDOW = 300  # iterations used for the wall-clock rate
FINAL_S = 300  # final validation + test eval and the last checkpoint save, roughly
STALL_S = 1200  # a running job with no new iteration line for this long gets flagged
RECENT_DAYS = 3  # sets without any wandb file are checked if modified this recently


def seconds(s):
    """squeue durations: [D-]HH:MM:SS or MM:SS. None for UNLIMITED, N/A, INVALID."""
    if not s or not s[0].isdigit():
        return None
    days, _, rest = s.rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m"


def queue():
    """{(set, run): job} for the user's SLURM jobs. submit_multiple.py names jobs <set>_<NNN>."""
    out = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%T|%M|%l|%R|%S"],
                         capture_output=True, text=True, check=True).stdout
    jobs = {}
    for line in out.splitlines():
        job_id, name, state, elapsed, limit, reason, start = line.split("|", 6)
        set_name, _, run = name.rpartition("_")
        jobs[(set_name, run)] = dict(id=job_id, name=name, state=state, elapsed=seconds(elapsed),
                                     limit=seconds(limit), reason=reason, start=start)
    return jobs


def read_log(path):
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    iters = [(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2)), int(m.group(3)))
             for m in map(ITER_RE.search, lines) if m]
    errors = []
    for i, line in enumerate(lines, 1):
        if ERROR_RE.search(line) and not IGNORE_RE.search(line):
            msg = re.sub(r"^\[rank\d+\]:\s*", "", line.strip())[:200]
            if msg not in (e for _, e in errors):
                errors.append((i, msg))
            if len(errors) == 3:
                break
    vals, test = VAL_RE.findall(text), TEST_RE.findall(text)
    return dict(iters=iters, val=vals[-1] if vals else None, test=test[-1] if test else None,
                nan=max(map(int, NAN_RE.findall(text)), default=0), errors=errors,
                exited=bool(EXIT_RE.search(text)), tail=[line.strip() for line in lines[-3:]])


def rate(iters):
    """Wall seconds per iteration over the last RATE_WINDOW iterations, skipping the slow first ones of the job."""
    if len(iters) < 2:
        return None
    t1, n1, _ = iters[-1]
    start = max(n1 - RATE_WINDOW, iters[0][1] + 20)
    for t0, n0, _ in iters:
        if start <= n0 < n1:
            return (t1 - t0).total_seconds() / (n1 - n0)
    return None


def describe_running(run, job, now):
    log_path = run / f"{job['id']}.out"
    if not log_path.exists():
        return f"{job['state']:<8} no log yet"
    log = read_log(log_path)
    if not log["iters"]:
        return f"{job['state']:<8} starting, no iteration logged yet"
    t1, n1, total = log["iters"][-1]
    parts = [f"{job['state']:<8} it {n1}/{total} ({100 * n1 // total}%)"]
    spi = rate(log["iters"])
    if spi:
        remaining = (total - n1) * spi + FINAL_S
        parts.append(f"{spi:.2f} s/it, ETA {now + timedelta(seconds=remaining):%a %H:%M} (in {fmt(remaining)})")
        if job["limit"] and job["elapsed"] is not None and job["elapsed"] + remaining > job["limit"]:
            parts.append(f"WILL HIT THE {fmt(job['limit'])} TIME LIMIT")
    if (now - t1).total_seconds() > STALL_S:
        parts.append(f"NO NEW ITERATION FOR {fmt((now - t1).total_seconds())}")
    if log["val"]:
        parts.append(f"val {float(log['val'][1]):.4f} @ {log['val'][0]}")
    if log["nan"]:
        parts.append("NaN ITERATIONS IN LOG")
    return ", ".join(parts)


def describe_finished(run, verbose):
    """Returns (status, text) for a run dir that is not in the queue."""
    logs = sorted(run.glob("*.out"), key=lambda p: p.stat().st_mtime)
    if (run / "done").exists():
        if not logs:
            return "done", "done"
        log = read_log(logs[-1])
        val = f"val {float(log['val'][1]):.4f}" if log["val"] else "val -"
        test = f"test {float(log['test']):.4f}" if log["test"] else "test -"
        return "done", f"done     {val}, {test}" + (", NaN ITERATIONS IN LOG" if log["nan"] else "")
    if not logs:
        return "no log", "no log (never started, or a dry run)"
    log = read_log(logs[-1])
    last = f"{log['iters'][-1][1]}/{log['iters'][-1][2]}" if log["iters"] else "0"
    if log["exited"] and not log["errors"]:
        val = f"val {float(log['val'][1]):.4f}" if log["val"] else "val -"
        test = f"test {float(log['test']):.4f}" if log["test"] else "test -"
        return "exited", f"exited   early on purpose at it {last} (exit_interval), {val}, {test}"
    ckpt = run / "latest_checkpointed_iteration.txt"
    resume = f"checkpoint {ckpt.read_text().strip()}, resumable with -r" if ckpt.exists() else "no checkpoint"
    first = log["errors"][0][1] if log["errors"] else (log["tail"][-1] if log["tail"] else "")
    text = f"FAILED   last it {last}, {resume}: {first[:150]}"
    if verbose:
        text += f"\n        log: {logs[-1]}"
        for i, msg in log["errors"]:
            text += f"\n        line {i}: {msg}"
        if not log["errors"]:
            text += "".join(f"\n        {line[:200]}" for line in log["tail"])
    return "failed", text


def sets_to_check(jobs, pattern):
    if pattern:
        return sorted(d.name for d in C.CHECKPOINT_ROOT.iterdir() if d.is_dir() and re.search(pattern, d.name))
    names = {set_name for set_name, _ in jobs if (C.CHECKPOINT_ROOT / set_name).is_dir()}
    cutoff = time.time() - RECENT_DAYS * 86400
    for s in wandb_sync.scan():
        recent_empty = s["status"] == "empty" and (C.CHECKPOINT_ROOT / s["name"]).stat().st_mtime > cutoff
        if s["status"] in ("active", "unsynced") or recent_empty:
            names.add(s["name"])
    return sorted(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", help="regex on the experiment set name; also matches synced sets")
    ap.add_argument("-v", "--verbose", action="store_true", help="error lines and log path of failed runs")
    a = ap.parse_args()

    now = datetime.now()
    jobs = queue()
    states = Counter(j["state"] for j in jobs.values())
    print(f"{now:%Y-%m-%d %H:%M}  squeue: {len(jobs)} jobs" + "".join(f", {n} {k}" for k, n in states.items()))
    other = sorted(j["name"] for (s, _), j in jobs.items() if not (C.CHECKPOINT_ROOT / s).is_dir())
    if other:
        print("jobs without an experiment set dir: " + ", ".join(other))

    names = sets_to_check(jobs, a.set)
    if not names:
        print("no unsynced or recent experiment sets: nothing to check")
        return
    for name in names:
        runs = sorted(p for p in (C.CHECKPOINT_ROOT / name).iterdir() if p.is_dir() and p.name.isdigit())
        lines, counts = [], Counter()
        for run in runs:
            job = jobs.get((name, run.name))
            if job and job["state"] == "PENDING":
                status, text = "pending", (f"PENDING  {job['reason']}, expected start "
                                           f"{job['start'] if job['start'][:1].isdigit() else 'unknown'}")
            elif job:
                status, text = "running", describe_running(run, job, now)
            else:
                status, text = describe_finished(run, a.verbose)
            counts[status] += 1
            lines.append(f"  {run.name} {text}" + (f"  [job {job['id']}]" if job else ""))
        print(f"\n{name}: {len(runs)} runs (" + ", ".join(f"{n} {k}" for k, n in counts.items()) + ")")
        print("\n".join(lines))
        if counts and set(counts) <= {"done", "exited"}:
            print("  all runs ended normally: ready to sync (sync-runs skill), or delete it if it was a smoke test")


if __name__ == "__main__":
    main()
