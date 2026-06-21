#!/usr/bin/env python3
"""Write per-environment and combined summaries for nav_evaluator outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from eval_common import repo_root_from_script


def env_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "house2" in stem or "env1" in stem:
        return "env1"
    if "house3" in stem or "env2" in stem:
        return "env2"
    return "unknown"


def load_rows(paths):
    rows = []
    for path in paths:
        env = env_from_path(path)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "environment": env,
                    "method": row["method"],
                    "success": int(row["success"]),
                    "collision": int(row["collision"]),
                    "costmap_failure": int(row["costmap_failure"]),
                    "time_s": float(row["time_s"]),
                    "path_dist_m": float(row["path_dist_m"]),
                })
    return rows


def summarise(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], row["method"])].append(row)
        grouped[("combined", row["method"])].append(row)

    summary = []
    for (environment, method), values in sorted(grouped.items()):
        n = len(values)
        summary.append({
            "environment": environment,
            "method": method,
            "n": n,
            "success_pct": 100.0 * sum(v["success"] for v in values) / n,
            "stuck_recovery_proxy_pct": 100.0 * sum(v["collision"] for v in values) / n,
            "costmap_failure_pct": 100.0 * sum(v["costmap_failure"] for v in values) / n,
            "avg_time_s": sum(v["time_s"] for v in values) / n,
            "avg_path_dist_m": sum(v["path_dist_m"] for v in values) / n,
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = repo_root_from_script()
    parser.add_argument(
        "csvs",
        nargs="*",
        type=Path,
    )
    parser.add_argument("--output", type=Path, default=Path("results/navigation_summary.csv"))
    args = parser.parse_args()

    if not args.csvs:
        raise SystemExit(
            "Pass one or more nav_evaluator CSV files, e.g. "
            "python3 scripts/summarize_nav_results.py results/nav_env*_reset.csv "
            "--output results/navigation_summary_reset.csv"
        )

    paths = [p if p.is_absolute() else repo_root / p for p in args.csvs]
    rows = load_rows(paths)
    summary = summarise(rows)

    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "environment",
        "method",
        "n",
        "success_pct",
        "stuck_recovery_proxy_pct",
        "costmap_failure_pct",
        "avg_time_s",
        "avg_path_dist_m",
    ]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            row = dict(row)
            for key in (
                "success_pct",
                "stuck_recovery_proxy_pct",
                "costmap_failure_pct",
                "avg_time_s",
                "avg_path_dist_m",
            ):
                row[key] = f"{row[key]:.2f}"
            writer.writerow(row)

    print(f"Wrote {len(summary)} rows to {output}")


if __name__ == "__main__":
    main()
