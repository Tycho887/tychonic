# 0.9.2 qualification

Baseline: 0.9.1 at `65eb42e72a5677068d241bce4948e3b451da3d57`.
The comparison uses Numeris 0.6.0 and Satkit 0.24.1.

## Complexity report

No standalone Rust complexity analyzer was installed. These are conservative
structural estimates: one plus each `if`, loop, boolean conjunction/disjunction,
and match arm, including closures and guards, excluding comments, strings, and
test modules. This is a repeatable source count, not a control-flow graph proof.
The same counting rule was applied to the baseline and implementation.
All production functions now have an estimate at most 10.

| Function | Before | After |
| --- | ---: | ---: |
| `src/optim/passive_rf.rs` evaluate | 31 | 1 |
| `src/estimate/passive_rf.rs` update_with_differential_baseline | 25 | 7 |
| `src/estimate/ekf.rs` check_covariance | 21 | 5 |
| `src/optim/passive_rf.rs` fit_passive_rf | 20 | 10 |
| `src/kernels/observation_models.rs` evaluate_rows | 18 | 2 |
| `src/measurements.rs` JointBatch::evaluate | 16 | 2 |
| `src/estimate/ekf.rs` filter_batches | 15 | 4 |
| `src/kernels/rotation.rs` PreparedGeometry::with_rotation | 15 | 5 |
| `src/optim/varpro.rs` ContactId::parse | 15 | 4 |
| `src/optim/lm.rs` run_lm_with_drag | 14 | 7 |
| `src/optim/passive_rf.rs` PassiveRfData::validate | 14 | 8 |
| `src/estimate/ekf.rs` white_acceleration_noise | 13 | 4 |
| `src/measurements.rs` joint | 13 | 6 |
| `src/measurements.rs` Model::new | 12 | 4 |
| `src/kernels/observation_models.rs` evaluate_mapped_into | 11 | 10 |
| `src/kernels/rotation.rs` check_basis | 11 | 5 |

Extracted: geometry construction cache and fixed-station setup; observable
validation; row prediction, sensitivity composition and whitening; RF prior and
covariance assembly; EKF scheduling and measurement dispatch; covariance scaling
and semidefinite checking; UUID format and hex-digit validation; process-noise
input checking. The local LM driver replaces the former callback bridge.

Behavior verification covers finite-difference Jacobians, all fifteen joint
observable selections, noiseless/noisy fits, variable projection, covariance,
OU drift, filter gating/history/rollback, and TCA sigma points. The local solver
also matches Numeris on a nonlinear regression and checks malformed evaluations,
negative-drag rejection and error preservation. Borrow lifetime and cache
exclusivity are checked with compile-fail documentation tests.

## Performance

Measurements used Rust 1.92.0 release builds on an Intel Core Ultra 5 125U
under Linux/WSL, with child processes pinned to logical CPU 0. Timing and
allocation instrumentation were separate. Each kernel used 1,024 rows and
1,000 iterations, a 0.2 s warm-up, five alternating baseline/current runs, and
the existing deterministic profiler fixture. Geometry used the `shuffled`
pattern. Values below are medians; the raw samples are in
[qualification_0_9_2.json](qualification_0_9_2.json).

| Kernel | 0.9.1 ns/row | 0.9.2 ns/row | Change |
| --- | ---: | ---: | ---: |
| range-rate | 24.45 | 23.72 | -3.0% |
| prepared-geometry | 166.64 | 160.08 | -3.9% |
| batch-owned | 228.28 | 249.11 | +9.1% |
| batch-reused | 231.07 | 235.35 | +1.9% |

Allocation runs used ten iterations after one warm-up call. Counts are the
same for both versions: range rate and reused batch evaluation allocate zero
buffers; owned batch evaluation allocates two buffers (196,608 bytes per
1,024-row batch); shuffled geometry construction allocates 15 times and
reallocates six times (399,376 requested bytes per schedule).

The uninstrumented `batch_fit` example was run five times per version, alternating
order, with `SATKIT_OFFLINE=1`. Its end-to-end median includes fixture generation,
propagation, fitting and process startup: **1.134 s → 1.152 s (+1.6%)**. Both
versions converge in five iterations to RSS `2.876314850635779e-13` and the same
reported state/drag accuracy.

Separately, the existing counting allocator was installed in temporary copies
of each version's example and enabled only around `fit_lm`. These runners used
the same locked dependency versions as the release builds. Requested allocation
traffic was **117,591,720 → 117,490,824 bytes** (100,896 bytes less, about 0.09%).
Allocation calls fell by 60 (1,580,965 → 1,580,905); reallocation calls were
unchanged at 1,068,552. These counts include numerical dependencies, whose
propagation allocations dominate this fixture. Requested bytes are cumulative
allocation traffic, not peak memory or retained memory. Allocation-instrumented
timing is excluded from the timing comparison.

The measurements support accepting the modest batch-evaluation cost in exchange
for the simpler ownership contract. Their spread overlaps; small timing changes
are observations on this machine, not portable speed claims. No performance
threshold is enforced in CI.

To reproduce the kernel runs in each version's checkout:

```sh
cargo build --locked --release --manifest-path tools/Cargo.toml --bin kernel_profile
tools/target/release/kernel_profile --kernel batch-reused --rows 1024 --iterations 1000 --warmup-seconds 0.2
cargo build --locked --release --manifest-path tools/Cargo.toml --bin kernel_profile --features profile-allocations
tools/target/release/kernel_profile --kernel batch-reused --rows 1024 --iterations 10 --warmup-seconds 0
cargo build --locked --release --example batch_fit
SATKIT_OFFLINE=1 target/release/examples/batch_fit
```

Select the other kernel names from the table, run each version on the same CPU,
and preserve the timing executable before building with allocation instrumentation.
The archived 0.9.1 tools lockfile has an old path-package version; refresh that
one entry offline before using `--locked`. No numerical dependency version was
changed for this comparison.

## Release verification

The standard offline suite passes 101 runtime tests and five documentation
tests, including two compile-fail tests. Subsequent focused geometry and solver
runs pass after the final guards. Formatting, production policy, root Clippy
with warnings denied, all tools targets/features, API documentation and the
release library build pass. The ignored 24-hour state/drag bootstrap passes in release mode (75.92 s).
`cargo package --locked --offline --allow-dirty` builds the extracted source.
A separate downstream executable using that extracted package passes borrowed
geometry/batches/RF inputs, sequential fitting and exclusive cache evaluation.
The archive excludes development tools, profiling allocator code and policy/CI
scripts. `--allow-dirty` permits qualification of this uncommitted workspace;
publication still uses a reviewed clean release commit.
