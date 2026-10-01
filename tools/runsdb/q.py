#!/e/project1/laionize/shechter1/miniforge3/envs/megatron-submit/bin/python
"""Query runsdb, the local mirror of the megatron-moe wandb project. Output is compact and capped (-n).

  q.py status                                 data location, last sync, counts
  q.py sweeps [--since DATE] [--grep RE]      one line per sweep: fixed setup, what varies, tags
  q.py runs   SEL [--cols a,b] [--sort [-]COL] [--fixed]
  q.py curve  KEY SEL [--steps 1500,4200 | --every N | --all] [--raw] [--plot out.png]
  q.py keys   [RE] [--full]                   which metrics exist, in how many runs, eval or train
  q.py config RUN [RUN ...] [--grep RE]       one run's config, or the keys where several runs differ
  q.py sql    "SELECT ..."                    DuckDB SQL

SEL (AND-ed): --sweep RE  --tag RE  --name RE  --runs id,id  --since YYYY-MM-DD  --where "SQL on runs"
SQL tables: runs, sweeps, keys, eval and train (run_id, step, key, value), raw('<run_id>') = full history.
"""

import argparse
import json
import re

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import common as C

DEFAULT_METRICS = ["val_loss_final", "maxvio_final", "maxvio_dispatch_final", "rect_frac_final", "iter_time_med"]
MAX_COLS = 8  # more differing config keys than this -> runs shows the label; more runs -> curve shows rows


def connect():
    con = duckdb.connect()
    for t in ("runs", "eval", "train", "sweeps", "keys"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{C.ROOT}/{t}.parquet')")
    con.execute(f"CREATE MACRO raw(id) AS TABLE SELECT * FROM read_parquet('{C.RAW}/' || id || '.parquet')")
    return con


def show(df, n):
    if df.empty:
        print("(no rows)")
        return
    print(df.head(n).to_string(index=False, float_format=lambda x: f"{x:.5g}", na_rep="-", max_colwidth=70))
    if len(df) > n:
        print(f"... {len(df) - n} more rows (raise -n)")


def select(con, a):
    where, params = [], []
    for col, val in (("sweep", a.sweep), ("tags", a.tag), ("name", a.name)):
        if val:
            where.append(f"regexp_matches({col}, ?)")
            params.append(val)
    if a.runs:
        where.append("list_contains(?, run_id)")
        params.append(a.runs.split(","))
    if a.since:
        where.append("started >= CAST(? AS TIMESTAMP)")
        params.append(a.since)
    if a.where:
        where.append(f"({a.where})")
    sql = "SELECT * FROM runs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY sweep, idx"
    return con.execute(sql, params).df()


def config_cols(df):
    return [c for c in df.columns if c not in C.ID_COLS and c not in C.METRIC_COLS]


def varying(df):
    return [c for c in config_cols(df) if df[c].astype(str).nunique() > 1]


def labels(runs):
    multi = runs["sweep"].nunique() > 1
    return {r.run_id: (f"{r.sweep} " if multi else "") + f"#{r.idx} {r.label}".strip() for r in runs.itertuples()}


def cmd_status(con, a):
    m = C.load_manifest()
    n, hist, sweeps, newest = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE rows > 0), count(DISTINCT sweep), max(started) FROM runs").fetchone()
    print(f"data: {C.ROOT}")
    print(f"last wandb sync: {m.get('last_sync')}; {n} runs ({hist} with history) in {sweeps} sweeps; "
          f"newest run started {newest}")
    failed = [r for r, e in m["runs"].items() if e.get("error")]
    local = [r for r, e in m["runs"].items() if e.get("source") == "local"]
    if failed:
        print(f"failed downloads (sync.py retries them): {len(failed)} {failed[:10]}")
    if local:
        print(f"runs ingested from local run dirs, not yet from wandb: {len(local)}")


def cmd_sweeps(con, a):
    df = con.execute("SELECT * FROM sweeps ORDER BY date DESC").df()
    if a.since:
        df = df[df["date"] >= pd.Timestamp(a.since)]
    if a.grep:
        df = df[df.astype(str).apply(" ".join, axis=1).map(lambda s: re.search(a.grep, s) is not None)]
    for r in df.head(a.n).itertuples():
        print(f"{r.date:%Y-%m-%d} {r.sweep} ({r.runs}) {r.setup} | varies: {r.varies or '-'}"
              + (f" | tags: {r.tags}" if r.tags else ""))
    if len(df) > a.n:
        print(f"... {len(df) - a.n} more sweeps (raise -n)")


def cmd_runs(con, a):
    df = select(con, a)
    if df.empty:
        return print("no runs match")
    var = varying(df)
    lead = ["run_id"] + (["sweep"] if df["sweep"].nunique() > 1 else []) + ["idx"]
    if a.cols:
        cols = lead + a.cols.split(",")
    else:
        cols = lead + (var if len(var) <= MAX_COLS else ["label"])
        cols += [m for m in DEFAULT_METRICS if df[m].notna().any()]
    if a.sort:
        df = df.sort_values(a.sort.lstrip("-"), ascending=not a.sort.startswith("-"))
    print(f"{len(df)} runs" + (f"; {len(var)} config keys differ, so the label is shown (pick keys with --cols)"
                               if len(var) > MAX_COLS and not a.cols else ""))
    fixed = [c for c in config_cols(df) if c not in var and (a.fixed or c in C.SETUP) and df[c].notna().all()]
    if fixed:
        print("fixed: " + ", ".join(f"{c}={df[c].iloc[0]}" for c in fixed))
    show(df[list(dict.fromkeys(cols))], a.n)


def last_at(s, t, tol):
    s = s.dropna()
    s = s[(s.index <= t) & (s.index > t - tol)]
    return s.iloc[-1] if len(s) else np.nan


def plot(wide, lab, a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for rid in wide.columns:
        s = wide[rid].dropna()
        ax.plot(s.index, s.values, label=f"{rid} {lab[rid]}"[:80])
    ax.set_xlabel("step")
    ax.set_ylabel(a.key)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.savefig(a.plot, dpi=120, bbox_inches="tight")
    print(f"plot saved: {a.plot}")


def cmd_curve(con, a):
    runs = select(con, a)
    if runs.empty:
        return print("no runs match")
    ids = runs["run_id"].tolist()
    if a.raw:
        frames = []
        for rid in ids:
            path = C.RAW / f"{rid}.parquet"
            if path.exists() and a.key in pq.read_schema(path).names:
                d = pq.read_table(path, columns=["_step", a.key]).to_pandas().dropna()
                frames.append(pd.DataFrame({"run_id": rid, "step": d["_step"], "value": d[a.key]}))
        data = pd.concat(frames) if frames else pd.DataFrame()
        source = "raw (full resolution)"
    else:
        for table in ("eval", "train"):
            data = con.execute(f"SELECT run_id, step, value FROM {table} WHERE key = ? AND list_contains(?, run_id)",
                               [a.key, ids]).df()
            if len(data):
                break
        source = "eval (every logged point)" if table == "eval" else f"train ({C.TRAIN_WINDOW}-step means)"
    if data.empty:
        return print(f"'{a.key}' is not logged in these runs (find keys with: q.py keys <regex>)")
    wide = data.pivot_table(index="step", columns="run_id", values="value", aggfunc="last")
    wide = wide[[r for r in ids if r in wide.columns]]
    wide.columns.name = None
    lab = labels(runs)
    if a.plot:
        plot(wide, lab, a)
    if a.steps:
        want = [int(s) for s in a.steps.split(",")]
        wide = pd.DataFrame({r: [last_at(wide[r], t, a.tol) for t in want] for r in wide.columns},
                            index=pd.Index(want, name="step"))
    elif a.every:
        wide = wide[wide.index % a.every == 0]
    elif not a.all:
        wide = wide.iloc[np.unique(np.linspace(0, len(wide) - 1, min(9, len(wide))).round().astype(int))]
    print(f"{a.key} from {source}; {wide.shape[1]} runs")
    if wide.shape[1] <= MAX_COLS:
        for rid in wide.columns:
            print(f"  {rid}: {lab[rid]}")
        show(wide.reset_index(), a.n)
    else:
        t = wide.T
        t.columns = [str(int(c)) for c in t.columns]
        t.insert(0, "label", [lab[r] for r in t.index])
        show(t.reset_index(names="run_id"), a.n)


def cmd_keys(con, a):
    df = con.execute("SELECT * FROM keys").df()
    if a.regex:
        df = df[df["key"].map(lambda k: re.search(a.regex, k) is not None)]
    if not a.full:
        df = df.groupby("pattern").agg(keys=("key", "size"), runs=("runs", "max"), kind=("kind", "first"),
                                       pts=("pts", "median"), first=("first", "min"), last=("last", "max"))
        df = df.reset_index()
    show(df.sort_values("runs", ascending=False), a.n)


def cmd_config(con, a):
    cfgs = {r: C.read_json(C.META / f"{r}.json")["config"] for r in a.run_ids}
    keys = sorted(set().union(*cfgs.values()))
    if a.grep:
        keys = [k for k in keys if re.search(a.grep, k)]
    else:  # only keys that tell runs apart; paths, names and job ids are left out
        informative = set(config_cols(con.execute("SELECT * FROM runs LIMIT 0").df()))
        keys = [k for k in keys if k in informative]
    if len(cfgs) > 1:
        dump = lambda v: json.dumps(v, sort_keys=True, default=str)
        keys = [k for k in keys if len({dump(c.get(k)) for c in cfgs.values()}) > 1]
    df = pd.DataFrame({r: [str(c.get(k)) for k in keys] for r, c in cfgs.items()}, index=pd.Index(keys, name="key"))
    show(df.reset_index(), a.n)


def cmd_sql(con, a):
    show(con.execute(a.query).df(), a.n)


def main():
    sel = argparse.ArgumentParser(add_help=False)
    sel.add_argument("--sweep", help="regex on the sweep name")
    sel.add_argument("--tag", help="regex on the comma-joined tags")
    sel.add_argument("--name", help="regex on the full run name")
    sel.add_argument("--runs", help="comma-separated run ids")
    sel.add_argument("--since", help="started on or after YYYY-MM-DD")
    sel.add_argument("--where", help='SQL condition on runs columns, e.g. "num_experts=2048 AND moe_router_topk=2"')
    cap = argparse.ArgumentParser(add_help=False)
    cap.add_argument("-n", type=int, default=40, help="max rows to print")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", parents=[cap])
    p = sub.add_parser("sweeps", parents=[cap])
    p.add_argument("--since")
    p.add_argument("--grep", help="regex over the whole sweep line")
    p = sub.add_parser("runs", parents=[sel, cap])
    p.add_argument("--cols", help="comma-separated runs columns to show instead of the defaults")
    p.add_argument("--sort", help="column to sort by; prefix with - for descending")
    p.add_argument("--fixed", action="store_true", help="print all config keys that are constant in the selection")
    p = sub.add_parser("curve", parents=[sel, cap])
    p.add_argument("key", help="exact metric key, e.g. val/regular/lm_loss")
    p.add_argument("--steps", help="comma-separated steps (value at or up to --tol before each)")
    p.add_argument("--tol", type=int, default=150)
    p.add_argument("--every", type=int, help="only steps divisible by N")
    p.add_argument("--all", action="store_true", help="every point (eval keys: every eval)")
    p.add_argument("--raw", action="store_true", help="full resolution from raw/ instead of eval/train")
    p.add_argument("--plot", help="also save a PNG of the full curves to this path")
    p = sub.add_parser("keys", parents=[cap])
    p.add_argument("regex", nargs="?")
    p.add_argument("--full", action="store_true", help="one row per key instead of per pattern")
    p = sub.add_parser("config", parents=[cap])
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--grep", help="regex on config key names (searches all keys, incl. paths)")
    p.set_defaults(n=200)
    p = sub.add_parser("sql", parents=[cap])
    p.add_argument("query")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](connect(), a)


if __name__ == "__main__":
    main()
