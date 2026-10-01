#!/e/project1/laionize/shechter1/miniforge3/envs/megatron-submit/bin/python
"""Mirror the megatron-moe wandb project into runsdb. Run on the login node (needs internet).

Lists all runs (~20 s; rewrites every meta/<run_id>.json), downloads the full history of new or
changed runs as parquet (~2 s/run, in parallel), then rebuilds the derived tables.

  tools/runsdb/sync.py                              # incremental
  tools/runsdb/sync.py --runs a1b2c3d4,e5f6g7h8 --force
"""

import argparse
import glob
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wandb

import build
import common as C


def as_dict(x):
    return json.loads(x) if isinstance(x, str) else (x or {})


def run_meta(run):
    a = run._attrs
    keys = as_dict(a.get("historyKeys")).get("keys") or {}
    return {
        "run_id": run.id,
        "name": a.get("displayName"),
        "state": a.get("state"),
        "created_at": a.get("createdAt"),
        "heartbeat_at": a.get("heartbeatAt"),
        "tags": a.get("tags") or [],
        "notes": a.get("notes") or "",
        "history_rows": a.get("historyLineCount") or 0,
        "history_keys": {
            k: sum(t.get("count", 0) for t in v.get("typeCounts", []))
            for k, v in keys.items()
            if not k.startswith("system/")
        },
        "config": C.clean_config(as_dict(a.get("config"))),
        "summary": as_dict(a.get("summaryMetrics")),
        "source": "wandb",
    }


def fetch_history(run):
    """Full history -> raw/<run_id>.parquet, from wandb's parquet export (or a paginated scan if the
    export is not ready yet, e.g. for a just-synced run)."""
    tmp = C.TMP / run.id
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        try:
            run.download_history_exports(tmp, require_complete_history=True)
            parts = sorted(glob.glob(f"{tmp}/**/*.parquet", recursive=True))
            table = pa.concat_tables([pq.read_table(p) for p in parts], promote_options="default")
            source = "export"
        except Exception:
            df = pd.DataFrame(list(run.scan_history(page_size=1000)))
            table = pa.Table.from_pandas(df.apply(pd.to_numeric, errors="coerce"), preserve_index=False)
            source = "scan"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return source, C.write_raw(run.id, table)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="comma-separated run ids to consider (default: all)")
    ap.add_argument("--force", action="store_true", help="re-download history even if unchanged")
    ap.add_argument("--limit", type=int, help="download at most N runs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-build", action="store_true", help="don't rebuild the derived tables")
    args = ap.parse_args()
    C.ensure_dirs()

    t0 = time.time()
    runs = list(wandb.Api(timeout=120).runs(f"{C.ENTITY}/{C.PROJECT}", per_page=200, lazy=False))
    manifest = C.load_manifest()
    todo = []
    for run in runs:
        meta = run_meta(run)
        C.write_json(C.META / f"{run.id}.json", meta)
        sig = f"{meta['heartbeat_at']}|{meta['history_rows']}|{meta['state']}"
        if args.runs and run.id not in args.runs.split(","):
            continue
        done = manifest["runs"].get(run.id, {}).get("sig") == sig and (C.RAW / f"{run.id}.parquet").exists()
        if meta["history_rows"] and (args.force or not done):
            todo.append((run, sig))
    todo = todo[: args.limit]
    print(f"listed {len(runs)} runs in {time.time() - t0:.0f}s; downloading history of {len(todo)}", flush=True)

    t1 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(fetch_history, run): (run, sig) for run, sig in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            run, sig = futures[fut]
            entry = manifest["runs"].setdefault(run.id, {})
            try:
                source, rows = fut.result()
                entry.update(sig=sig, source=source, rows=rows, fetched=C.now(), error=None)
            except Exception as e:  # keeps the old sig, so the run is retried on the next sync
                entry["error"] = repr(e)[:300]
            if i % 100 == 0 or i == len(todo):
                C.write_json(C.MANIFEST, manifest)
                print(f"  {i}/{len(todo)} runs, {time.time() - t1:.0f}s", flush=True)
    manifest["last_sync"] = C.now()
    C.write_json(C.MANIFEST, manifest)
    failed = [r for r, e in manifest["runs"].items() if e.get("error")]
    if failed:
        print(f"{len(failed)} runs failed (see 'error' in manifest.json): {failed[:10]}")
    if not args.no_build:
        build.main()


if __name__ == "__main__":
    main()
