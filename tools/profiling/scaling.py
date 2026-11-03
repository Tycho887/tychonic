#!/usr/bin/env python3
"""Calibrate fixed work and compare working-set sizes using the uninstrumented runner."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernels", default="range-rate,az-el-rotation,rotate-matrix,full-jacobian,station-geometry")
    parser.add_argument("--patterns", default="shared-repeated,unique-unique")
    parser.add_argument("--sizes", default="1,16,256,1024,16384,65536")
    parser.add_argument("--modes", default="reuse,stream")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    sizes = [int(n) for n in args.sizes.split(",")]
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.repetitions < 1 or any(n <= 0 for n in sizes):
        parser.error("positive duration, repetitions and sizes are required")
    env = os.environ.copy()
    env["SATKIT_OFFLINE"] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metadata.json").write_text(json.dumps({
        "binary": str(args.binary.resolve()),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "kernels": args.kernels, "modes": args.modes, "patterns": args.patterns, "sizes": sizes,
        "target_seconds": args.seconds, "repetitions": args.repetitions,
        "warmup_seconds": 0.2,
        "method": "median of repeated fixed-work run means; no confidence intervals",
    }, indent=2) + "\n")
    records = []
    for kernel in args.kernels.split(","):
        variants = args.patterns.split(",") if kernel in {"station-geometry", "prepared-geometry"} else (
            ["stream"] if kernel in {"full-jacobian", "batch-owned", "batch-reused", "batch-reference"} else args.modes.split(","))
        for variant in variants:
            for rows in sizes:
                command = [str(args.binary.resolve()), "--kernel", kernel, "--rows", str(rows),
                           "--warmup-seconds", "0.2"]
                command += ["--pattern" if kernel in {"station-geometry", "prepared-geometry"} else "--mode", variant]

                def run(iterations):
                    result = json.loads(subprocess.check_output(command + ["--iterations", str(iterations)],
                                                               env=env, text=True))
                    if result["measurement"] != "timing":
                        raise RuntimeError("scaling requires the uninstrumented timing binary")
                    return result

                calibration = run(100)
                batch_ns = calibration["elapsed_ns"] / calibration["iterations"]
                iterations = max(1, round(args.seconds * 1e9 / batch_ns))
                runs = [run(iterations) for _ in range(args.repetitions)]
                means = [r["ns_per_row"] for r in runs]
                median = statistics.median(means)
                records.append({"kernel": kernel, "variant": variant, "rows": rows,
                    "median_ns_per_row": median, "repeat_spread_percent":
                    100 * (max(means) - min(means)) / median,
                    "calibration": calibration, "runs": runs})
                print(f"{kernel}/{variant}/{rows}: {median:.3f} ns/row", flush=True)
                # Save partial progress so interruption does not discard completed measurements.
                (args.output / "scaling.json").write_text(json.dumps(records, indent=2) + "\n")
    with (args.output / "scaling.csv").open("w", newline="") as file:
        fields = ["kernel", "variant", "rows", "median_ns_per_row", "repeat_spread_percent"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
