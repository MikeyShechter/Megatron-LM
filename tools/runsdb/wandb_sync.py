#!/e/project1/laionize/shechter1/miniforge3/envs/megatron-submit/bin/python
"""Upload finished experiment sets from the checkpoint root to wandb, then refresh runsdb.

  wandb_sync.py check [--include SET ...] [--list]  classify sets; exit 0 = runs to sync, 1 = nothing, 2 = blocked
  wandb_sync.py start [--include SET ...]           check, then sync in a detached process that survives logout
  wandb_sync.py wait                                block until that process ends, then print its summary
  wandb_sync.py status                              progress or summary of the current/last sync

An experiment set is a directory in CHECKPOINT_ROOT.
- An offline run (<run>/wandb/wandb/offline-run-*/) is synced if it has a .synced marker or its run id is in
  runsdb.
- A set is synced when all its offline runs are, or when runsdb holds at least as many runs of that sweep as
  there are offline runs. Some older sets were uploaded under new run ids.
- Runs whose SLURM job is still queued or running are left alone.

Sets are ordered by last activity (newest .wandb write). An unsynced set that is older than the newest synced
set blocks everything: nothing starts until it is deleted or approved with --include. Otherwise:
1. all unsynced finished runs are uploaded (`wandb sync <offline-run dir>`, 4 at a time);
2. then tools/runsdb/sync.py pulls them into runsdb.
"""

import argparse
import getpass
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

import common as C

SCRIPT = Path(__file__).resolve()
STATUS = C.SYNC_DIR / "status.json"
WANDB = Path(sys.executable).with_name("wandb")
PARALLEL = 4


def active_runs():
    """{set name: {run dir names}} of the user's queued or running SLURM jobs (job name = <set>_<NNN>)."""
    out = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
                         capture_output=True, text=True, check=True).stdout
    active = {}
    for name in out.split():
        base, _, run = name.rpartition("_")
        active.setdefault(base, set()).add(run)
    return active


def scan(include=()):
    active = active_runs()
    db_ids = {p.stem for p in C.META.glob("*.json")}
    db_counts = pd.read_parquet(C.ROOT / "runs.parquet", columns=["sweep"])["sweep"].value_counts().to_dict()
    sets = []
    for d in sorted(p for p in C.CHECKPOINT_ROOT.iterdir() if p.is_dir()):
        files = sorted(d.glob("*/wandb/wandb/*run-*/run-*.wandb"))  # <set>/<run>/wandb/wandb/<offline-run>/run-<id>.wandb
        run_dirs = {f.parents[3] for f in files}
        act = active.get(d.name, set())
        covered = db_counts.get(d.name, 0) >= len(files)
        pending = [] if covered else [
            str(f.parent) for f in files
            if not Path(f"{f}.synced").exists() and f.stem[4:] not in db_ids and f.parents[3].name not in act
        ]
        sets.append({
            "name": d.name,
            "status": "active" if act else "empty" if not files else "unsynced" if pending else "synced",
            "pending": pending,
            "active": sorted(act),
            "offline_runs": len(files),
            "runs": len(run_dirs),
            "done": sum((r / "done").exists() for r in run_dirs),
            "last": max(f.stat().st_mtime for f in files) if files else None,
        })
    newest = max((s["last"] for s in sets if s["status"] == "synced"), default=None)
    for s in sets:
        s["blocked"] = (s["status"] == "unsynced" and newest is not None and s["last"] < newest
                        and s["name"] not in include)
    return sets


def when(t):
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else "-"


def report(sets, paths=False):
    counts = Counter(s["status"] for s in sets)
    print(f"{len(sets)} experiment sets in {C.CHECKPOINT_ROOT}: "
          + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    synced = [s for s in sets if s["status"] == "synced"]
    if synced:
        newest = max(synced, key=lambda s: s["last"])
        print(f"newest synced set: {newest['name']} (last activity {when(newest['last'])})")
    groups = (
        ("to sync", [s for s in sets if s["pending"] and not s["blocked"]]),
        ("waiting for SLURM jobs (their runs are skipped)", [s for s in sets if s["active"] and not s["pending"]]),
        ("BLOCKED, unsynced sets older than the newest synced set", [s for s in sets if s["blocked"]]),
    )
    for title, group in groups:
        if group:
            print(title + ":")
        for s in group:
            print(f"  {s['name']}: {len(s['pending'])} of {s['offline_runs']} offline runs unsynced, "
                  f"{s['done']}/{s['runs']} run dirs done, last activity {when(s['last'])}"
                  + (f", queued/running: {','.join(s['active'])}" if s["active"] else ""))
            for p in s["pending"] if paths else []:
                print("      " + p)


def verdict(sets):
    if any(s["blocked"] for s in sets):
        print("nothing will be started: delete the blocked sets or approve them with --include <set>")
        return 2
    if not any(s["pending"] for s in sets):
        print("nothing to sync")
        return 1
    return 0


def cmd_check(a):
    sets = scan(a.include)
    report(sets, a.list)
    return verdict(sets)


def cmd_start(a):
    sets = scan(a.include)
    report(sets)
    code = verdict(sets)
    if code:
        return code
    todo = [s for s in sets if s["pending"]]
    C.SYNC_DIR.mkdir(parents=True, exist_ok=True)
    log = C.SYNC_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.log"
    C.write_json(STATUS, {"state": "starting", "host": socket.gethostname(), "started": time.time(),
                          "log": str(log), "sets": [s["name"] for s in todo],
                          "plan": [p for s in todo for p in s["pending"]], "results": {}})
    with open(log, "w") as f:  # own session: immune to the terminal/SSH session ending
        subprocess.Popen([sys.executable, str(SCRIPT), "_worker"], stdin=subprocess.DEVNULL, stdout=f,
                         stderr=subprocess.STDOUT, cwd=C.SYNC_DIR, start_new_session=True)
    print(f"started in the background: {sum(len(s['pending']) for s in todo)} offline runs; log: {log}")
    return 0


def cmd_worker(a):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    st = C.read_json(STATUS)
    st.update(state="running", pid=os.getpid())
    C.write_json(STATUS, st)
    lock = threading.Lock()

    def upload(path):
        t0 = time.time()
        r = subprocess.run([str(WANDB), "sync", path], capture_output=True, text=True, cwd=C.SYNC_DIR)
        ok = any(Path(path).glob("*.wandb.synced"))
        with lock:
            print(f"=== wandb sync {path}: {'ok' if ok else 'NOT SYNCED'} (exit {r.returncode}, "
                  f"{time.time() - t0:.0f}s)\n{r.stdout}{r.stderr}", flush=True)
            st["results"][path] = {"ok": ok, "exit": r.returncode}
            C.write_json(STATUS, st)

    try:
        with ThreadPoolExecutor(PARALLEL) as ex:
            list(ex.map(upload, st["plan"]))
        print("=== tools/runsdb/sync.py", flush=True)
        r = subprocess.run([sys.executable, str(SCRIPT.with_name("sync.py"))], capture_output=True, text=True)
        print(r.stdout + r.stderr, flush=True)
        st["runsdb"] = {"exit": r.returncode, "lines": [
            line for line in r.stdout.splitlines() if line.startswith(("listed", "built")) or "failed" in line]}
        st["state"] = "done" if r.returncode == 0 and all(x["ok"] for x in st["results"].values()) else "failed"
    except Exception as e:
        st.update(state="failed", error=repr(e))
        raise
    finally:
        st["finished"] = time.time()
        C.write_json(STATUS, st)
    return 0


def summary(st):
    res = st.get("results", {})
    minutes = ((st.get("finished") or time.time()) - st["started"]) / 60
    print(f"wandb sync {st['state']}: {sum(r['ok'] for r in res.values())}/{len(st['plan'])} offline runs uploaded "
          f"in {minutes:.1f} min ({', '.join(st['sets'])})")
    bad = [p for p, r in res.items() if not r["ok"]]
    if bad:
        print("not uploaded: " + " ".join(bad))
    if st.get("error"):
        print("error: " + st["error"])
    if "runsdb" in st:
        print("runsdb: " + " | ".join(st["runsdb"]["lines"])
              + (f" (sync.py exit {st['runsdb']['exit']})" if st["runsdb"]["exit"] else ""))
    print(f"log: {st['log']}")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def cmd_wait(a):
    while True:
        st = C.read_json(STATUS)
        if st["state"] in ("done", "failed"):
            summary(st)
            return 0 if st["state"] == "done" else 1
        if "pid" in st and st["host"] == socket.gethostname() and not alive(st["pid"]):
            time.sleep(5)
            if C.read_json(STATUS)["state"] not in ("done", "failed"):
                print(f"the sync process ({st['pid']}) ended without finishing; see {st['log']}")
                return 1
            continue
        time.sleep(10)


def cmd_status(a):
    if not STATUS.exists():
        print("no sync has been started yet")
        return 1
    st = C.read_json(STATUS)
    if st["state"] in ("done", "failed"):
        summary(st)
    else:
        res = st.get("results", {})
        print(f"{st['state']} on {st['host']} since {when(st['started'])}: {len(res)}/{len(st['plan'])} uploads "
              f"finished ({sum(r['ok'] for r in res.values())} ok); log: {st['log']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "start"):
        p = sub.add_parser(name)
        p.add_argument("--include", nargs="+", default=[], metavar="SET", help="approve these older sets for syncing")
        if name == "check":
            p.add_argument("--list", action="store_true", help="also print the unsynced offline-run dirs")
    for name in ("wait", "status", "_worker"):
        sub.add_parser(name)
    a = ap.parse_args()
    if hasattr(a, "include"):
        a.include = {Path(s).name for s in a.include}
    sys.exit(globals()[f"cmd_{a.cmd.lstrip('_')}"](a))


if __name__ == "__main__":
    main()
