#!/usr/bin/env python3
"""Summarize raw runtime samples emitted by the ROS profiling hooks."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from eval_common import repo_root_from_script


def percentile(values, pct):
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--samples", type=Path, default=Path("results/runtime_profile_live_raw.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/runtime_profile_live_summary.csv"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    samples_path = args.samples if args.samples.is_absolute() else repo_root / args.samples
    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    if samples_path.exists():
        with open(samples_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                grouped[row["component"]].append(float(row["duration_ms"]))

    fields = ["component", "mean_ms", "median_ms", "p95_ms", "max_ms", "num_samples"]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for component, values in sorted(grouped.items()):
            writer.writerow({
                "component": component,
                "mean_ms": f"{statistics.mean(values):.3f}",
                "median_ms": f"{statistics.median(values):.3f}",
                "p95_ms": f"{percentile(values, 95):.3f}",
                "max_ms": f"{max(values):.3f}",
                "num_samples": len(values),
            })

    if grouped:
        print(f"Wrote {len(grouped)} runtime summary rows to {output}")
    else:
        print(f"No runtime samples found at {samples_path}; wrote header to {output}")


if __name__ == "__main__":
    main()
