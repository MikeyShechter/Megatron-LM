#!/e/project1/laionize/shechter1/miniforge3/envs/megatron-submit/bin/python
"""Ingest runs straight from local run dirs (offline wandb files) into runsdb. No network needed.

  tools/runsdb/ingest_local.py /e/data1/mmlaion/shechter1/checkpoints/megatron/<sweep_dir> [more dirs]

Accepts sweep dirs (containing NNN/ run dirs) and run dirs. Runs already mirrored from wandb are skipped
unless --force. After a run is synced to wandb, the next sync.py replaces the local copy with wandb's.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import pandas as pd
import pyarrow as pa
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

import build
import common as C


def records(path):
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    while True:
        try:
            data = ds.scan_data()
        except Exception:  # truncated tail of a file that is still being written
            return
        if data is None:
            return
        rec = pb.Record()
        rec.ParseFromString(data)
        yield rec


def item_key(item):
    return item.key or "/".join(item.nested_key)


def ingest_run(run_id, files):
    """files: the run's .wandb files in chronological order (a resumed run has several; later wins)."""
    rows, config, summary, info = {}, {}, {}, {}
    for path in files:
        for rec in records(path):
            kind = rec.WhichOneof("record_type")
            if kind == "history":
                row = {}
                for it in rec.history.item:
                    v = json.loads(it.value_json)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        row[item_key(it)] = v
                row["_step"] = rec.history.step.num
                rows[row["_step"]] = row
            elif kind == "config":
                config.update({item_key(it): json.loads(it.value_json) for it in rec.config.update})
            elif kind == "summary":
                summary.update({item_key(it): json.loads(it.value_json) for it in rec.summary.update})
                for it in rec.summary.remove:
                    summary.pop(item_key(it), None)
            elif kind == "run":  # carries the initial config; later "config" records update it
                config.update({item_key(it): json.loads(it.value_json) for it in rec.run.config.update})
                info.update(name=rec.run.display_name, tags=list(rec.run.tags),
                            created_at=rec.run.start_time.ToDatetime().strftime("%Y-%m-%dT%H:%M:%SZ"))
            elif kind == "exit":
                info["exit_code"] = rec.exit.exit_code
    df = pd.DataFrame([rows[s] for s in sorted(rows)])
    n = C.write_raw(run_id, pa.Table.from_pandas(df, preserve_index=False))
    exit_code = info.get("exit_code")
    C.write_json(C.META / f"{run_id}.json", {
        "run_id": run_id,
        "name": info.get("name"),
        "state": "unknown" if exit_code is None else ("finished" if exit_code == 0 else "crashed"),
        "created_at": info.get("created_at"),
        "heartbeat_at": None,
        "tags": info.get("tags", []),
        "notes": "",
        "history_rows": n,
        "history_keys": {c: int(df[c].notna().sum()) for c in df.columns},
        "config": C.clean_config(config),
        "summary": summary,
        "source": "local",
        "local_dir": files[0].split("/wandb/")[0],
    })
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="sweep dirs or run dirs")
    ap.add_argument("--force", action="store_true", help="also re-ingest runs already mirrored from wandb")
    ap.add_argument("--no-build", action="store_true", help="don't rebuild the derived tables")
    args = ap.parse_args()
    C.ensure_dirs()

    by_run = defaultdict(list)
    for d in args.dirs:
        for f in glob.glob(f"{d}/wandb/wandb/*run-*/run-*.wandb") + glob.glob(f"{d}/*/wandb/wandb/*run-*/run-*.wandb"):
            by_run[re.search(r"run-(\w+)\.wandb$", f).group(1)].append(f)
    manifest = C.load_manifest()
    for run_id, files in sorted(by_run.items()):
        if manifest["runs"].get(run_id, {}).get("source") in ("export", "scan") and not args.force:
            print(f"{run_id}: already mirrored from wandb, skipped")
            continue
        files.sort(key=lambda f: re.search(r"run-(\d{8}_\d{6})", f).group(1))
        n = ingest_run(run_id, files)
        manifest["runs"][run_id] = {"sig": f"local|{n}", "source": "local", "rows": n, "fetched": C.now(),
                                    "error": None}
        print(f"{run_id}: {n} history rows from {len(files)} file(s)")
    C.write_json(C.MANIFEST, manifest)
    if not args.no_build:
        build.main()


if __name__ == "__main__":
    main()
