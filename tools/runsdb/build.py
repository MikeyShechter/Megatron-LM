#!/e/project1/laionize/shechter1/miniforge3/envs/megatron-submit/bin/python
"""Rebuild the derived tables from raw/ and meta/ (sync.py and ingest_local.py call this at the end).

  runs.parquet     one row per run: identity, config keys that differ between runs, summary metrics
  eval.parquet     (run_id, step, key, value) for eval-cadence keys, every logged point
  train.parquet    (run_id, step, key, value) for per-step keys, mean over TRAIN_WINDOW-step windows
                   (step = last step of the window)
  sweeps.parquet + sweeps.md   one row per sweep: date, #runs, fixed setup, what varies, tags
  keys.parquet + keys.md       one row per key: eval/train, #runs, points per run, first/last seen
"""

import json
import os
import re
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import common as C

KINDS = {}


def key_kinds(metas):
    """'train' for keys logged on most steps (median over runs of #values/#rows > 5%), else 'eval'."""
    shares = defaultdict(list)
    for m in metas:
        if m["history_rows"]:
            for k, n in m["history_keys"].items():
                shares[k].append(n / m["history_rows"])
    return {k: "train" if statistics.median(v) > 0.05 else "eval" for k, v in shares.items()}


def _init_worker(kinds):
    global KINDS
    KINDS = kinds


def to_long(df, cols):
    if not cols:
        return pd.DataFrame(columns=["_step", "key", "value"])
    return df[["_step"] + cols].melt(id_vars="_step", var_name="key").dropna(subset=["value"])


def process_run(job):
    run_id, decay = job
    df = pd.read_parquet(C.RAW / f"{run_id}.parquet")
    for canon, aliases in C.ALIASES.items():
        if canon not in df or df[canon].isna().all():
            for alias in aliases:
                if alias in df:
                    df[canon] = df[alias]
                    break
    cols = [c for c in df.columns if c not in ("_step", "_timestamp")]
    for c in cols:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    step = df["_step"]
    ecols = [c for c in cols if KINDS.get(c, "eval") == "eval"]
    tcols = [c for c in cols if KINDS.get(c) == "train"]
    ev = to_long(df.loc[df[ecols].notna().any(axis=1)], ecols)
    window_end = ((step - 1) // C.TRAIN_WINDOW + 1) * C.TRAIN_WINDOW
    tr = to_long(df[tcols].groupby(window_end.rename("_step")).mean().reset_index(), tcols)

    out = {"steps": int(step.max()) if len(step) else None}
    if "_timestamp" in df and df["_timestamp"].notna().any():
        out["started"] = float(df["_timestamp"].min())
    for name, key in C.SUMMARY_METRICS.items():
        if key not in df:
            continue
        s = df.loc[df[key].notna(), ["_step", key]]
        if len(s):
            out[f"{name}_final"] = s[key].iloc[-1]
            out[f"{name}_last5"] = s[key].iloc[-5:].mean()
            pre = s.loc[s["_step"] <= decay, key] if decay else []
            if len(pre):
                out[f"{name}_predecay"] = pre.iloc[-1]
    if "test/lm_loss" in df and df["test/lm_loss"].notna().any():
        out["test_loss"] = df["test/lm_loss"].dropna().iloc[-1]
    if "iteration-time" in df:
        out["iter_time_med"] = df.loc[step > C.ITER_TIME_FROM, "iteration-time"].median()
    return run_id, ev.assign(run_id=run_id), tr.assign(run_id=run_id), out


def write_parquet(df, name):
    tmp = C.ROOT / f"{name}.parquet.tmp"
    df.to_parquet(tmp, index=False, compression="zstd", row_group_size=200_000)
    os.replace(tmp, C.ROOT / f"{name}.parquet")


def write_long(name, frames):
    df = pd.concat([f for f in frames if len(f)], ignore_index=True).rename(columns={"_step": "step"})
    df = df.astype({"step": "int32", "run_id": "category", "key": "category", "value": "float64"})
    write_parquet(df.sort_values(["key", "run_id", "step"])[["run_id", "step", "key", "value"]], name)


def config_keys(metas):
    """The SETUP keys plus config keys with 2..100 distinct values across runs (missing counts as a
    value). These tell runs apart; keys with more values are run-specific (paths, names, job ids)."""
    values, present = defaultdict(set), defaultdict(int)
    for m in metas.values():
        for k, v in m["config"].items():
            values[k].add(json.dumps(v, sort_keys=True, default=str))
            present[k] += 1
    out = []
    for k, vals in values.items():
        n = len(vals) + (present[k] < len(metas) and "null" not in vals)
        if k in C.SETUP or (2 <= n <= 100 and k not in C.RUN_SPECIFIC | set(C.ID_COLS) | set(C.METRIC_COLS)):
            out.append(k)
    return sorted(out)


def typed(values):
    """Column of config values with a proper dtype; lists/dicts become JSON strings."""
    col = pd.Series(values, dtype=object)
    vals = [v for v in values if v is not None]
    if vals and all(isinstance(v, bool) for v in vals):
        return col.astype("boolean")
    if vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return col.astype("Int64")
    if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return col.astype("float64")
    return col.map(lambda v: v if v is None or isinstance(v, str) else json.dumps(v))


def runs_table(metas, finals):
    ids = list(metas)
    rows = []
    for rid in ids:
        m, f = metas[rid], finals.get(rid, {})
        cfg = m["config"]
        sweep, idx, label = C.parse_name(m["name"])
        local_dir = m.get("local_dir") or cfg.get("output_dir") or cfg.get("save")
        rows.append({
            "run_id": rid, "sweep": sweep, "idx": idx, "label": label, "name": m["name"],
            "tags": ",".join(m["tags"]), "started": f.get("started"), "created_at": m["created_at"],
            "state": m["state"],
            "runtime_h": (m["summary"].get("_runtime") or np.nan) / 3600,
            "steps": f.get("steps", m["summary"].get("_step")), "rows": m["history_rows"],
            "decay_start": C.decay_start(cfg), "source": m["source"], "local_dir": local_dir,
            "local_exists": bool(local_dir) and os.path.isdir(local_dir),
            **{k: f.get(k) for k in C.METRIC_COLS},
        })
    df = pd.DataFrame(rows)
    # wandb's createdAt is the upload time for offline-synced runs, so prefer the first logged timestamp
    created = pd.to_datetime(df["created_at"], utc=True).dt.tz_localize(None)
    df["started"] = pd.to_datetime(df["started"], unit="s").fillna(created)
    for c in ("idx", "steps", "decay_start"):
        df[c] = df[c].astype("Int64")
    df[C.METRIC_COLS] = df[C.METRIC_COLS].astype("float64")
    keys = config_keys(metas)
    cfg = pd.DataFrame({k: typed([metas[r]["config"].get(k) for r in ids]) for k in keys})
    df = pd.concat([df[C.ID_COLS], cfg, df[C.METRIC_COLS]], axis=1)
    return df.sort_values(["sweep", "idx"]).reset_index(drop=True)


def value_list(col):
    v = sorted(col.astype(str).unique())
    return "[" + ",".join(v[:8]) + (f",...+{len(v) - 8}" if len(v) > 8 else "") + "]"


def sweeps_table(runs):
    cfg_cols = [c for c in runs.columns if c not in C.ID_COLS and c not in C.METRIC_COLS]
    rows = []
    for sweep, g in runs.groupby("sweep", sort=False):
        nvals = {c: g[c].astype(str).nunique() for c in cfg_cols}
        rows.append({
            "sweep": sweep, "date": g["started"].min(), "runs": len(g), "with_history": int((g["rows"] > 0).sum()),
            "setup": " ".join(f"{s}={g[k].iloc[0]}" for k, s in C.SETUP.items() if k in g and nvals[k] == 1),
            "varies": " ".join(f"{c}={value_list(g[c])}" for c in cfg_cols if nvals[c] > 1),
            "tags": ",".join(sorted({t for ts in g["tags"] for t in ts.split(",") if t})),
        })
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


def keys_table(metas, kinds, run_dates):
    counts, dates = defaultdict(list), defaultdict(list)
    for m in metas.values():
        d = run_dates.get(m["run_id"])
        for k, n in m["history_keys"].items():
            if k not in ("_step", "_timestamp"):
                counts[k].append(n)
                if isinstance(d, str):
                    dates[k].append(d)
    return pd.DataFrame([
        {"key": k, "pattern": re.sub(r"\d+", "#", k), "kind": kinds.get(k, "eval"), "runs": len(n),
         "pts": float(np.median(n)), "first": min(dates[k], default=None), "last": max(dates[k], default=None)}
        for k, n in counts.items()
    ]).sort_values(["runs", "key"], ascending=[False, True]).reset_index(drop=True)


def write_markdown(sweeps, keys):
    lines = [f"# Sweeps in {C.PROJECT} (generated by tools/runsdb/build.py, newest first)", "",
             "date **sweep** (#runs) fixed setup | varies: config_key=[values] | tags", ""]
    for r in sweeps.itertuples():
        date = f"{r.date:%Y-%m-%d}" if pd.notna(r.date) else "?"
        lines.append(f"- {date} **{r.sweep}** ({r.runs}) {r.setup} | varies: {r.varies or '-'}"
                     + (f" | tags: {r.tags}" if r.tags else ""))
    (C.ROOT / "sweeps.md").write_text("\n".join(lines) + "\n")

    g = keys.groupby("pattern").agg(n_keys=("key", "size"), runs=("runs", "max"), kind=("kind", "first"),
                                    pts=("pts", "median"), first=("first", "min"), last=("last", "max"))
    g = g.sort_values("runs", ascending=False).reset_index()
    lines = [f"# Metric keys in {C.PROJECT} (generated by tools/runsdb/build.py)", "",
             "`#` stands for digits (layer index etc.). kind=eval: every logged point is in eval.parquet; "
             f"kind=train: logged every step (or every 10), {C.TRAIN_WINDOW}-step means in train.parquet. "
             "pts = median logged points per run; runs = max over the keys of a pattern.", "",
             "| pattern | keys | runs | kind | pts | first | last |", "|---|---|---|---|---|---|---|"]
    lines += [f"| `{r.pattern}` | {r.n_keys} | {r.runs} | {r.kind} | {r.pts:g} | {r.first} | {r.last} |"
              for r in g.itertuples()]
    (C.ROOT / "keys.md").write_text("\n".join(lines) + "\n")


def main():
    t0 = time.time()
    metas = {}
    for path in C.META.glob("*.json"):
        m = C.read_json(path)
        metas[m["run_id"]] = m
    kinds = key_kinds(metas.values())
    jobs = [(r, C.decay_start(m["config"])) for r, m in metas.items() if (C.RAW / f"{r}.parquet").exists()]
    with ProcessPoolExecutor(min(16, os.cpu_count() or 4), initializer=_init_worker, initargs=(kinds,)) as ex:
        results = list(ex.map(process_run, jobs, chunksize=16))
    write_long("eval", [r[1] for r in results])
    write_long("train", [r[2] for r in results])
    runs = runs_table(metas, {r[0]: r[3] for r in results})
    write_parquet(runs, "runs")
    sweeps = sweeps_table(runs)
    write_parquet(sweeps, "sweeps")
    keys = keys_table(metas, kinds, dict(zip(runs["run_id"], runs["started"].dt.strftime("%Y-%m-%d"))))
    write_parquet(keys, "keys")
    write_markdown(sweeps, keys)
    print(f"built tables for {len(metas)} runs ({len(jobs)} with history) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
