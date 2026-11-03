# Numerical kernel profiling

This harness compares checked public kernels, prepared kernels and observation
assembly. The original checked cases remain available for baseline comparisons.
Propagation, solver work and fixture generation are outside timed loops. Station
preprocessing and full-Jacobian measurements include their output allocations
and deallocations. Scalar results are fully consumed through `black_box`, including
Jacobian components that a prediction-only caller might otherwise discard.
The [initial findings](findings.md) include measured costs, scaling and profile
evidence from this workspace.

## Benchmarks and reproducible baselines

Provision satkit data as described in the project README, then run offline:

```sh
SATKIT_OFFLINE=1 cargo bench --locked --bench kernels
SATKIT_OFFLINE=1 cargo test --locked --test kernel_stress
python3 tools/profiling/baseline.py --output target/profiling/full-baseline
```

Defaults are six row counts (1, 16, 256, 1,024, 16,384, 65,536), two seconds of
warm-up, five seconds of measurement, 30 samples, and three independent benchmark
repetitions. The complete baseline takes roughly an hour depending on hardware.
Use a smaller selection for an initial investigation:

```sh
python3 tools/profiling/baseline.py --output target/profiling/initial --sizes 1024
TYCHONIC_BENCH_SIZES=16,1024,65536 SATKIT_OFFLINE=1 cargo bench --locked --bench kernels -- 'full-jacobian|station-geometry'
```

The Python script creates a new output directory, captures hardware/compiler/
dependency metadata and source hashes, preserves Criterion's raw estimates and
samples for each repetition, and writes `summary.json` and `summary.csv`.
It also saves separate timing and allocation-counting runner binaries. Allocation
counts cover three post-warm-up invocations of each selected case. These counts
describe requested Rust heap allocations, not RSS, stack usage or allocator
metadata. Reallocation bytes count the full requested new size.

`median_ns_per_row` is the median of the repetitions' mean batch times divided
by row count. Criterion confidence intervals describe average batch time; they
are not individual-call latency percentiles. Tables include the between-run
spread and an exploratory regression threshold equal to twice the larger of
that spread or the relative within-run CI half-width. No threshold is suggested
with fewer than three repetitions. Confirm thresholds across additional runs on
the same idle host before using them to gate changes.

## Inputs and interpretation

- Scalar groups: range-rate and angle predictions/Jacobians; direct versus
  precomputed LOS conversion; quaternion/matrix vector rotation; inverse ENU
  rotation. `traversal-control` provides a lower bound for loop/black-box overhead;
  do not subtract it automatically from the measurements.
- `reuse` cycles through at most 16 rows; `stream` traverses every row in order.
  Each timed invocation covers the requested row count. Repeated small streaming
  batches are also warm in cache; these are working-set experiments, not forced
  cold-cache measurements. Reuse also reduces input diversity, so branch and
  transcendental-function behavior can change. Attribute differences using
  profiles and counters rather than assuming every gap comes from cache misses.
- Full Jacobians use synthetic nonidentity STMs and drag columns. They qualify
  and time chain-rule assembly without any numerical propagation. Large fixtures
  exceed the current machine's last-level cache.
- Geometry patterns vary distinct epochs and stations independently:
  `shared-repeated` has one epoch and up to 16 sites; `shared-unique` one epoch
  and unique sites; `unique-repeated` unique epochs and up to 16 sites;
  `unique-unique` unique epochs/sites; `shuffled` shuffles rows having up to
  16 sites and roughly one epoch per 16 rows. Pattern names describe reuse
  within each call; the constructor's caches are rebuilt each invocation.
- Fixtures use SplitMix64 with seed `0x747963686f6e6963`, all azimuth quadrants,
  elevations between ±1.4 rad and ranges from 1 km to about 40,000 km. Extra
  correctness tests cover branch crossings, near-zenith/nadir behavior,
  the rejection threshold, invalid inputs and a 65,536-row batch. Error paths
  are tested separately and do not contribute to successful-call throughput.

Keep the machine idle during timing, avoid concurrent benchmark runs, and record
power settings or CPU affinity changes with your results. The baseline uses the
existing optimization defaults plus debug symbols. To investigate CPU-specific
compilation, use a separate output/target directory and record the flags:

```sh
RUSTFLAGS='-C target-cpu=native' CARGO_TARGET_DIR=target/native python3 tools/profiling/baseline.py --output target/profiling/native --sizes 1024
```

## Fixed-work runner and allocation counts

```sh
cargo build --manifest-path tools/Cargo.toml --locked --release --bin kernel_profile
SATKIT_OFFLINE=1 tools/target/release/kernel_profile --kernel az-el-rotation --rows 16384 --iterations 10000
tools/target/release/kernel_profile --help
cargo build --manifest-path tools/Cargo.toml --locked --release --bin kernel_profile --features profile-allocations
SATKIT_OFFLINE=1 tools/target/release/kernel_profile --kernel full-jacobian --rows 65536 --iterations 3
```

The runner emits one JSON record after setup, warm-up and fixed work. Timing
builds report elapsed time and throughput; allocation builds report counts and
bytes, without elapsed-time results. Rebuild without the feature before timing,
or use the binaries preserved by `baseline.py`. Batch kernels require stream
mode; `--pattern` applies only to station preprocessing.

## Hardware counters and call stacks

`perf.py` starts the runner, waits for its post-warm-up `READY` signal, attaches
perf with events disabled, waits for perf to acknowledge enabling them, then
releases the measured loop. Fixture generation and warm-up are excluded.
Small startup/unblocking and final reporting overhead remains; use enough fixed
work to last several seconds. Calibrate iterations with an unprofiled run.
Counter collection and sampling should be separate from baseline timing.

For an exploratory size sweep using the saved timing binary, run:

```sh
python3 tools/profiling/scaling.py --binary target/profiling/initial/kernel_profile --output target/profiling/scaling
```

This separately calibrates fixed work to approximately one second per case,
then runs three repetitions after 0.2-second warm-ups. It saves every raw runner
record, a CSV summary and the binary hash. It covers both scalar memory modes,
all six sizes, full Jacobians, and shared/repeated versus unique/unique station
preprocessing. These shorter runs are exploratory scaling measurements; use
Criterion for confidence intervals and regression qualification.

```sh
python3 tools/profiling/perf.py --kind stat --output target/profiling/angles.stat -- target/profiling/initial/kernel_profile --kernel az-el-rotation --rows 16384 --iterations 10000
python3 tools/profiling/perf.py --kind record --output target/profiling/angles.data -- target/profiling/initial/kernel_profile --kernel az-el-rotation --rows 16384 --iterations 10000
perf report --stdio --no-children --input target/profiling/angles.data
perf report --stdio --children --input target/profiling/angles.data
```

Default counters are user-space cycles, instructions, branches, branch misses
and cache misses. Check perf's running percentages when events are multiplexed;
repeat smaller event sets if necessary. Sampling uses user-space CPU-clock at
499 Hz with DWARF call stacks and avoids writing to the global build-ID cache.
Self-cost identifies expensive instructions/functions; inclusive cost identifies
their callers. For numerical library internals, `perf annotate` can distinguish
validation, transcendental math, small matrix arithmetic and allocation work.

WSL2 and managed machines may lack particular counters or sampling permissions.
The helper reports failures instead of silently replacing counters. If counters
are unavailable, request software events explicitly (`--events task-clock:u`) and
retain Criterion timings; gather hardware evidence on native Linux later.

Some standalone WSL2 perf builds can capture DWARF stacks but cannot decode ELF
symbols or unwind those stacks. Use a perf build linked with libelf and libdw or
libunwind for reporting. Existing captures can be decoded by that build without
repeating measurements. `perf.py --perf-binary /path/to/perf` selects an alternative
collector. A local wrapper may set `LD_LIBRARY_PATH` for unpacked dependencies;
this does not require installing packages system-wide.

## Investigation order

1. Compare direct and precomputed coordinate paths, and inspect rotation-basis
   validation in the angle and stored-LOS kernels.
2. Inspect `hypot`, `atan2` and trigonometric cost versus validation and vector
   operations; compare kernels only at matching sizes and memory modes.
3. Inspect chained-Jacobian allocations and scaling. Distinguish traversal of
   large sensitivity inputs from output matrix writes.
4. Compare geometry patterns to locate station/epoch lookup and frame-transform
   costs. Normalize by both rows and distinct stations/epochs where useful.
5. Rank findings by measured cost and likely call frequency. Keep optimizations
   as a separate change, preserving validation and numerical tolerances.

The prepared-kernel experiments add `az-el-prepared`, `los-prepared`,
`chain-separate`, `chain-combined`, `chain-augmented`, `prepared-geometry`,
`batch-reference`, `batch-owned` and `batch-reused` to both runners.
`batch-reference` reconstructs the pre-refactor observation assembly from
sampled sensitivities, including duplicate predictions and intermediate full
Jacobians. All three batch cases produce equivalent whitened residuals and
Jacobians; their timings exclude propagation. `batch-reused` primes output
storage before timing/counting, so repeated observation assembly needs no
output allocations. This does not measure propagation allocation costs.

Chain experiments isolate four separate products, two combined-observable
products, and one augmented-sensitivity product including augmentation assembly.
They use the same prebuilt instantaneous partials and consume all outputs.

For a three-repetition comparison across small and large batches:

```sh
cargo build --manifest-path tools/Cargo.toml --locked --release --bin kernel_profile
python3 tools/profiling/scaling.py --binary tools/target/release/kernel_profile --output target/profiling/prepared-comparison --kernels az-el-rotation,az-el-prepared,chain-separate,chain-combined,chain-augmented,batch-reference,batch-owned,batch-reused,station-geometry,prepared-geometry --sizes 16,1024,65536 --modes stream
```

The original baseline findings describe the source before this refactor;
new results and limitations are recorded in `prepared-findings.md`. End-to-end
propagation and estimator deadlines remain separate profiling work.
