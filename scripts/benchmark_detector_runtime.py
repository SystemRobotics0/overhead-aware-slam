#!/usr/bin/env python3
"""Benchmark the TFLite detector on this machine with synthetic input.

This is a detector microbenchmark, not a full ROS camera pipeline benchmark.
It is still useful for reporting CPU inference cost on the development host.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

from eval_common import repo_root_from_script


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = repo_root_from_script()
    parser.add_argument("--model", type=Path, default=repo_root / "src/sensor_fusion_nodes/models/detect1.tflite")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/runtime_profile.csv"))
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit(f"TensorFlow is not importable, cannot benchmark detector: {exc}")

    model_path = args.model.resolve()
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    shape = tuple(int(v) for v in input_detail["shape"])
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        sample = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    else:
        sample = np.random.randint(0, 255, size=shape, dtype=dtype)

    for _ in range(args.warmup):
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        for detail in output_details:
            interpreter.get_tensor(detail["index"])

    timings = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        for detail in output_details:
            interpreter.get_tensor(detail["index"])
        timings.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(timings)
    median_ms = statistics.median(timings)
    p95_ms = percentile(timings, 95)
    max_ms = max(timings)
    fps = 1000.0 / mean_ms if mean_ms > 0.0 else 0.0

    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "component",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "max_ms",
        "num_samples",
        "approx_fps",
        "notes",
    ]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "component": "tflite_detector_invoke_cpu_synthetic_input",
            "mean_ms": f"{mean_ms:.3f}",
            "median_ms": f"{median_ms:.3f}",
            "p95_ms": f"{p95_ms:.3f}",
            "max_ms": f"{max_ms:.3f}",
            "num_samples": len(timings),
            "approx_fps": f"{fps:.2f}",
            "notes": f"model={model_path.name}; tf.lite.Interpreter CPU path; synthetic input shape={shape}",
        })

    print(f"Wrote detector runtime benchmark to {output}")
    print(f"Detector synthetic-input CPU mean: {mean_ms:.3f} ms ({fps:.2f} FPS)")


if __name__ == "__main__":
    main()
