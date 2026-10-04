#!/e/project1/laionize/shechter1/miniforge3/envs/megatron-submit/bin/python
"""Evaluate a trained split-router run with tokens routed to lower router ranks, and read its LM STE signal.

  eval_routing_ranks.py submit <run dir> [--ranks "1,3 3,4"] [--iters 20]   write a spec and sbatch it
  eval_routing_ranks.py report <analysis dir>                                print the results

The job loads the run's latest checkpoint and trains --iters more iterations with every learning rate at 0, so
the weights don't change; these iterations log the run's LM STE metrics (train/lm_ste/*) on the trained model.
The final validation then evaluates eval_iters batches with the usual top-k routing and replays the same batches
with each rank set routed instead (--eval-router-selection-ranks). The test eval uses the top-k.
Nothing is written to the run dir: the outputs go to ANALYSIS_ROOT/<set>_<run>, and wandb logs offline to the
project megatron-moe-analysis there, which the sync tool doesn't scan.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = Path("/e/data1/mmlaion/shechter1/analysis/eval_routing_ranks")


def cmd_submit(a):
    run_dir = Path(a.run_dir).resolve()
    spec = yaml.safe_load((run_dir / "spec.yaml").read_text())
    iteration = int((run_dir / "latest_checkpointed_iteration.txt").read_text())
    out = ANALYSIS_ROOT / f"{run_dir.parent.name}_{run_dir.name}"
    out.mkdir(parents=True)
    spec.pop("output_dir")  # it sets both save and load to the run dir
    spec.update(
        load=str(run_dir),
        no_load_optim=True,
        train_iters=iteration + a.iters,
        lr=0.0,
        min_lr=0.0,
        decoupled_lr=0.0,
        decoupled_min_lr=0.0,
        override_opt_param_scheduler=True,
        eval_router_selection_ranks=a.ranks,
        eval_split_router_with_weighter=False,
        eval_split_router_with_router_weights=False,
        wandb_project="megatron-moe-analysis",
        wandb_save_dir=str(out / "wandb"),
        run_name=f"eval_routing_ranks_{out.name}",
    )
    (out / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=True))
    subprocess.run(
        ["sbatch", f"--output={out}/%j.out", f"--error={out}/%j.out",
         f"--job-name=eval_routing_ranks_{run_dir.name}", str(REPO / "run_megatron.sh"),
         str(out / "spec.yaml")],
        check=True,
    )
    print(f"analysis dir: {out}")


def eval_values(text, which):
    """{key: value} from the last '... on <which> set | key value: x | ...' line of a Megatron log."""
    lines = [line for line in text.splitlines() if f"on {which} set |" in line]
    if not lines:
        return {}
    return {k.strip(): float(v) for k, v in re.findall(r"\|\s*([^|]+?) value: ([0-9.eE+-]+)", lines[-1])}


def lm_ste_means(out):
    """Mean of each train/lm_ste/* key over the logged steps of the job's offline wandb run."""
    sys.path.insert(0, str(REPO / "tools" / "runsdb"))
    from ingest_local import item_key, records

    sums, counts = defaultdict(float), defaultdict(int)
    for path in sorted(out.glob("wandb/wandb/offline-run-*/run-*.wandb")):
        for rec in records(str(path)):
            if rec.WhichOneof("record_type") != "history":
                continue
            for it in rec.history.item:
                key = item_key(it)
                if key.startswith("train/lm_ste/"):
                    sums[key] += json.loads(it.value_json)
                    counts[key] += 1
    return {key: (sums[key] / counts[key], counts[key]) for key in sorted(sums)}


def cmd_report(a):
    out = Path(a.analysis_dir).resolve()
    log = max(out.glob("*.out"), key=lambda p: p.stat().st_mtime)
    text = log.read_text(errors="replace")
    val, test = eval_values(text, "validation"), eval_values(text, "test")
    if not val:
        print(f"no final validation in {log} yet")
        return 1
    top_k = val["lm loss"]
    print(f"{out.name}  (log {log.name})\n\nvalidation loss on the same batches for every routing:")
    print(f"  {'top-k':<12} {top_k:.4f}")
    for key, value in val.items():
        match = re.fullmatch(r"val/router_ranks_([0-9_]+)/lm_loss", key)
        if match:
            name = "ranks " + match.group(1).replace("_", ",")
            print(f"  {name:<12} {value:.4f}  ({value - top_k:+.4f})")
    if "lm loss" in test:
        print(f"test loss (top-k): {test['lm loss']:.4f}")
    means = lm_ste_means(out)
    if means:
        print("\nLM STE signal on the trained model (mean over the zero-LR training iterations):")
        for key, (value, n) in means.items():
            print(f"  {key:<45} {value:.4f}  ({n} steps)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit")
    p.add_argument("run_dir")
    p.add_argument("--ranks", default="1,3 3,4", help='rank sets for the replayed evals, e.g. "1,3 3,4"')
    p.add_argument("--iters", type=int, default=20, help="zero-LR training iterations for the STE metrics")
    p = sub.add_parser("report")
    p.add_argument("analysis_dir")
    a = ap.parse_args()
    return {"submit": cmd_submit, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
