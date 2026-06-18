#!/usr/bin/env python3
"""Summarize Megatron elapsed-time-per-iteration logs.

Example:
    python scripts/stable_elapsed_time.py /path/to/slurm.out
"""

import argparse
import re
from pathlib import Path


LOG_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)\s*/\s*\d+.*?"
    r"elapsed time per iteration \(ms\):\s+(?P<elapsed_ms>[0-9.]+)"
)


def mean(values):
    return sum(values) / len(values)


def variance(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


def trim_values(values, trim_fraction):
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim fraction must be >= 0.0 and < 0.5")

    sorted_values = sorted(values)
    trim_count = int(len(sorted_values) * trim_fraction)
    if trim_count == 0:
        return sorted_values
    return sorted_values[trim_count:-trim_count]


def parse_records(path):
    records = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_RE.search(line)
            if not match:
                continue

            iteration = int(match.group("iteration"))
            elapsed_ms = float(match.group("elapsed_ms"))
            records.append((iteration, elapsed_ms))

    return records


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print a stable elapsed-time-per-iteration summary from Megatron stdout logs."
        )
    )
    parser.add_argument("log_file", type=Path)
    parser.add_argument(
        "--min-iteration",
        type=int,
        default=3000,
        help="Only use timing entries printed at this iteration or later. Default: 3000.",
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.1,
        help="Fraction to remove from each tail before computing statistics. Default: 0.1.",
    )
    args = parser.parse_args()

    records = parse_records(args.log_file)
    selected_values = [
        elapsed_ms for iteration, elapsed_ms in records if iteration >= args.min_iteration
    ]

    if not selected_values:
        raise SystemExit(f"No timing entries found at iteration >= {args.min_iteration}.")

    trimmed_values = trim_values(selected_values, args.trim_fraction)

    print(f"mean_ms: {mean(trimmed_values):.3f}")
    print(f"variance_ms2: {variance(trimmed_values):.6f}")


if __name__ == "__main__":
    main()
