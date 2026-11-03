#!/usr/bin/env python3
"""Run repeatable Criterion baselines and export per-row timing/allocation tables."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def capture(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="1,16,256,1024,16384,65536")
    parser.add_argument("--filter", default="", help="Criterion benchmark regex")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--skip-allocations", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or any(int(n) <= 0 for n in args.sizes.split(",")):
        parser.error("positive sizes and repetitions are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["SATKIT_OFFLINE"] = "1"
    env["TYCHONIC_BENCH_SIZES"] = args.sizes
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "fixture_seed": 0x747963686f6e6963,
        "sizes": args.sizes, "filter": args.filter, "repetitions": args.repetitions,
        "warmup_seconds": 2, "measurement_seconds": 5, "sample_size": 30,
        "environment": {k: env.get(k) for k in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_TARGET_DIR", "SATKIT_DATA", "SATKIT_OFFLINE", "TYCHONIC_BENCH_SIZES"]},
        "tools": [capture(c) for c in [["rustc", "-Vv"], ["lscpu"],
            ["cargo", "tree", "--locked"]]],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "tools/Cargo.toml", ROOT / "tools/Cargo.lock", *sorted((ROOT / "src").rglob("*.rs")),
                ROOT / "benches/support/mod.rs", ROOT / "benches/kernels.rs",
                ROOT / "tools/src/bin/kernel_profile.rs", ROOT / "tools/profiling/allocations.rs"]},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = {}
    for repeat in range(1, args.repetitions + 1):
        home = output / f"repeat-{repeat}"
        env["CRITERION_HOME"] = str(home)
        command = ["cargo", "bench", "--locked", "--bench", "kernels", "--", "--noplot"]
        if args.filter:
            command.append(args.filter)
        print(f"Criterion repetition {repeat}/{args.repetitions}; log: {output / f'repeat-{repeat}.log'}", flush=True)
        with (output / f"repeat-{repeat}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        found = 0
        for estimates_path in home.glob("**/new/estimates.json"):
            directory = estimates_path.parent
            benchmark = json.loads((directory / "benchmark.json").read_text())
            estimates = json.loads(estimates_path.read_text())
            key = benchmark["full_id"]
            rows = int(benchmark["value_str"])
            group = benchmark["group_id"]
            kernel, variant = group.split("/", 1)
            record = records.setdefault(key, {"benchmark": key, "kernel": kernel,
                "variant": variant, "rows": rows, "runs": []})
            mean = estimates["mean"]
            record["runs"].append({"repetition": repeat,
                "batch_ns": mean["point_estimate"],
                "ns_per_row": mean["point_estimate"] / rows,
                "ci_lower_ns_per_row": mean["confidence_interval"]["lower_bound"] / rows,
                "ci_upper_ns_per_row": mean["confidence_interval"]["upper_bound"] / rows})
            found += 1
        if not found:
            raise RuntimeError("filter selected no benchmarks; inspect the repetition log")
    for record in records.values():
        means = [r["ns_per_row"] for r in record["runs"]]
        median = statistics.median(means)
        record["median_ns_per_row"] = median
        record["rows_per_second"] = 1e9 / median
        record["repeat_spread_percent"] = 100 * (max(means) - min(means)) / median
        ci_width = max(100 * (r["ci_upper_ns_per_row"] - r["ci_lower_ns_per_row"])
                       / (2 * r["ns_per_row"]) for r in record["runs"])
        record["suggested_regression_threshold_percent"] = (
            2 * max(record["repeat_spread_percent"], ci_width) if len(means) >= 3 else None)
    # Preserve both binaries so a later feature build cannot contaminate timing runs.
    subprocess.run(["cargo", "build", "--manifest-path", "tools/Cargo.toml", "--locked", "--release", "--bin", "kernel_profile"],
                   cwd=ROOT, env=env, check=True)
    target = Path(json.loads(subprocess.check_output(
        ["cargo", "metadata", "--manifest-path", "tools/Cargo.toml", "--locked", "--no-deps", "--format-version", "1"],
        cwd=ROOT, text=True))["target_directory"])
    shutil.copy2(target / "release/kernel_profile", output / "kernel_profile")
    if not args.skip_allocations:
        subprocess.run(["cargo", "build", "--manifest-path", "tools/Cargo.toml", "--locked", "--release", "--bin", "kernel_profile",
                        "--features", "profile-allocations"], cwd=ROOT, env=env, check=True)
        allocation_binary = output / "kernel_profile_allocations"
        shutil.copy2(target / "release/kernel_profile", allocation_binary)
        for record in records.values():
            command = [str(allocation_binary), "--kernel", record["kernel"], "--rows",
                       str(record["rows"]), "--iterations", "3", "--warmup-seconds", "0.05"]
            if record["kernel"] in {"station-geometry", "prepared-geometry"}:
                command += ["--pattern", record["variant"]]
            else:
                command += ["--mode", record["variant"]]
            record["allocation_profile"] = json.loads(subprocess.check_output(
                command, cwd=ROOT, env=env, text=True))
    ordered = sorted(records.values(), key=lambda r: (r["kernel"], r["variant"], r["rows"]))
    (output / "summary.json").write_text(json.dumps(ordered, indent=2) + "\n")
    fields = ["benchmark", "rows", "median_ns_per_row", "rows_per_second",
              "repeat_spread_percent", "suggested_regression_threshold_percent",
              "allocations_per_batch", "requested_bytes_per_row"]
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in ordered:
            row = {k: record[k] for k in fields if k in record}
            if "allocation_profile" in record:
                alloc = record["allocation_profile"]
                row["allocations_per_batch"] = alloc["allocations"] / alloc["iterations"]
                row["requested_bytes_per_row"] = alloc["bytes_per_row"]
            writer.writerow(row)
    print(f"Saved {len(ordered)} cases to {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
