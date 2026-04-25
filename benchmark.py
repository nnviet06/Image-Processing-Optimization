# !/usr/bin/env python3
# Benchmark automation script for Image Processing Optimization project.
# Runs all CPU/GPU implementations 5 times each, captures timings, generates CSV.

import subprocess
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
EXECUTABLES = [
    "gaussian_naive",
    "gaussian_simd",
    "gaussian_threaded",
    "gaussian_gpu_kernel",
    "gaussian_gpu_optimized",
    "gaussian_gpu_batch",
]

TEST_IMAGES = ["test1.png", "test2.png"]
NUM_RUNS = 5
OUTPUT_CSV = "benchmarks_results.csv"


def find_timing(stdout: str) -> float:
    """
    Parse timing from executable stdout.
    Expected format: "X ms" or "avg X ms" or similar
    Returns timing in milliseconds as float.
    """
    # Look for pattern like "352.238 ms" or "352.24 ms"
    match = re.search(r'(\d+\.\d+)\s*ms', stdout)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not parse timing from output:\n{stdout}")


def run_executable(exe_name: str, image_path: str, runs: int = NUM_RUNS) -> List[float]:
    """
    Run executable 5 times, return list of timings (in ms).
    """
    timings = []
    exe_path = Path(exe_name).resolve()

    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")

    for run_num in range(runs):
        try:
            result = subprocess.run(
                [str(exe_path), image_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per run
            )

            if result.returncode != 0:
                print(f"  Run {run_num + 1}: ERROR (code {result.returncode})")
                print(f"    stderr: {result.stderr[:200]}")
                raise RuntimeError(f"Executable failed: {result.stderr}")

            timing = find_timing(result.stdout)
            timings.append(timing)
            print(f"  Run {run_num + 1}: {timing:.2f} ms")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Timeout running {exe_name}")

    return timings


def main():
    """Main benchmark harness."""
    cwd = Path.cwd()
    repo_root = cwd.parent if cwd.name == "build" else cwd

    print(f"Working directory: {cwd}")
    print(f"Repo root: {repo_root}")
    print(f"Running {NUM_RUNS} iterations per executable per image")
    print()

    # Collect all results
    results = []

    # Naive baseline timings (per image) for speedup calculation
    baselines = {}

    for image in TEST_IMAGES:
        image_path = repo_root / "data" / image
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}, skipping")
            continue

        print(f"\n{'=' * 80}")
        print(f"Testing: {image} ({image_path.stat().st_size / 1e6:.1f} MB)")
        print("=" * 80)

        for exe_name in EXECUTABLES:
            print(f"\n{exe_name}:")
            try:
                timings = run_executable(exe_name, str(image_path), NUM_RUNS)
                avg_time = statistics.mean(timings)
                stddev = statistics.stdev(timings) if len(timings) > 1 else 0.0

                # Store baseline (naive) for this image
                if exe_name == "gaussian_naive":
                    baselines[image] = avg_time

                # Calculate speedup
                baseline = baselines.get(image, avg_time)
                speedup = baseline / avg_time if avg_time > 0 else 1.0

                # Build result row
                row = {
                    "image": image,
                    "implementation": exe_name,
                    "run_1_ms": f"{timings[0]:.2f}",
                    "run_2_ms": f"{timings[1]:.2f}",
                    "run_3_ms": f"{timings[2]:.2f}",
                    "run_4_ms": f"{timings[3]:.2f}",
                    "run_5_ms": f"{timings[4]:.2f}",
                    "avg_ms": f"{avg_time:.2f}",
                    "stddev_ms": f"{stddev:.2f}",
                    "speedup": f"{speedup:.1f}x",
                }
                results.append(row)

                print(f"  Average: {avg_time:.2f} ± {stddev:.2f} ms")
                print(f"  Speedup: {speedup:.1f}x")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    # Write CSV
    print(f"\n{'=' * 80}")
    print("Writing results to CSV...")
    csv_path = cwd / OUTPUT_CSV

    if results:
        fieldnames = [
            "image",
            "implementation",
            "run_1_ms",
            "run_2_ms",
            "run_3_ms",
            "run_4_ms",
            "run_5_ms",
            "avg_ms",
            "stddev_ms",
            "speedup",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to: {csv_path}")
    else:
        print("No results to write")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()