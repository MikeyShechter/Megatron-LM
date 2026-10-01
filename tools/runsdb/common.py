"""Shared paths and helpers for runsdb, the local mirror of the megatron-moe wandb project."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ENTITY = "mikeyshechter"
PROJECT = "megatron-moe"
ROOT = Path(os.environ.get("RUNSDB_ROOT", "/e/data1/mmlaion/shechter1/runsdb")) / PROJECT
RAW = ROOT / "raw"  # <run_id>.parquet: full history, one column per key, sorted by _step
META = ROOT / "meta"  # <run_id>.json: name, state, dates, tags, config, summary, per-key value counts
TMP = ROOT / "tmp"
MANIFEST = ROOT / "manifest.json"
SYNC_DIR = ROOT / "wandb_sync"  # status.json and logs of wandb_sync.py
CHECKPOINT_ROOT = Path("/e/data1/mmlaion/shechter1/checkpoints/megatron")  # one dir per experiment set

# Older names of a key. A run that lacks the canonical key gets it filled from the first alias it has.
ALIASES = {"val/regular/lm_loss": ["val/lm_loss"]}

# Per-run summary columns in runs.parquet: <name>_final, <name>_last5 (mean of the last 5 values) and
# <name>_predecay (last value at or before the first WSD decay iteration).
SUMMARY_METRICS = {
    "val_loss": "val/regular/lm_loss",
    "maxvio": "vio/MaxVioGlobal",
    "maxvio_dispatch": "val/router_balance/max_vio_dispatch_mean",
    "rect_frac": "ste/all_layers/all_experts_in_rect_frac",
    "tasks_avg": "tasks/average",
}
METRIC_COLS = [f"{n}_{s}" for n in SUMMARY_METRICS for s in ("final", "last5", "predecay")] + [
    "test_loss",
    "iter_time_med",
]
ID_COLS = ["run_id", "sweep", "idx", "label", "name", "tags", "started", "state", "runtime_h", "steps",
           "rows", "decay_start", "source", "local_dir", "local_exists"]
# Settings shown as a sweep's fixed setup in sweeps.md and as "fixed:" in q.py runs.
SETUP = {"num_experts": "E", "moe_router_topk": "k", "moe_router_load_balancing_type": "lb",
         "moe_router_score_function": "score", "optimizer": "opt"}
# Config keys that are unique per run (paths, names); never used as runs.parquet columns.
RUN_SPECIFIC = {"save", "load", "output", "output_dir", "run_name", "config_label", "slurm_job_name",
                "wandb_exp_name", "tensorboard_dir", "wandb_save_dir"}

TRAIN_WINDOW = 50  # per-step keys are averaged over windows of this many steps in train.parquet
ITER_TIME_FROM = 100  # iter_time_med ignores the first steps


def ensure_dirs():
    for d in (RAW, META, TMP):
        d.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, path)


def load_manifest():
    return read_json(MANIFEST) if MANIFEST.exists() else {"runs": {}}


def clean_config(cfg):
    """Drop wandb-internal keys and unwrap {"value": v} entries."""
    out = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and "value" in v and set(v) <= {"value", "desc"}:
            v = v["value"]
        out[k] = v
    return out


def parse_name(name):
    """'006_<sweep>+<label>' -> ('<sweep>', 6, '<label>')."""
    m = re.match(r"^(\d+)_(.*)$", name or "")
    if not m:
        return name, None, ""
    sweep, _, label = m.group(2).partition("+")
    return sweep, int(m.group(1)), label


def decay_start(cfg):
    """First iteration of the WSD decay phase, or None."""
    if cfg.get("lr_decay_style") != "WSD" or not cfg.get("lr_wsd_decay_iters"):
        return None
    total = cfg.get("lr_decay_iters") or cfg.get("train_iters")
    return total - cfg["lr_wsd_decay_iters"] if total else None


def write_raw(run_id, table):
    """Write a run's full history to raw/<run_id>.parquet (integer _step, sorted). Returns #rows."""
    i = table.schema.get_field_index("_step")
    table = table.set_column(i, "_step", table["_step"].cast(pa.int64())).sort_by("_step")
    tmp = RAW / f"{run_id}.parquet.tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, RAW / f"{run_id}.parquet")
    return table.num_rows
