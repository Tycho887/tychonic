# Kernel profiling baseline — 2 November 2026

The scalar kernels allocate no heap memory during measured calls. The clearest
repeated work is rotation-basis validation: it accounts for about 24% of the
isolated angle kernel's sampled CPU time including its callees. Chained Jacobian
assembly spends about 33% in small matrix multiplication. These are useful
targets for controlled optimization experiments; the profiles do not establish
that the multiplication dispatcher itself is inefficient.

## Method and results

Intel Core i9-9900K under WSL2, single-threaded optimized Rust builds with debug
symbols, existing compiler optimization defaults, satkit data loaded offline.
Inputs are deterministic and use seed `0x747963686f6e6963`. Timing excludes
propagation, solver work, fixture creation and warm-up. Batch timings include
output creation and destruction.

The initial Criterion baseline covers every kernel/memory-mode/geometry-pattern
combination at 1,024 rows: 24 cases, three repetitions, two-second warm-ups,
five-second measurement windows and 30 samples per repetition. A separate
fixed-work sweep covers all six planned sizes for range-rate, stored-rotation
angles, matrix rotation, full Jacobians and two preprocessing patterns, with
three approximately one-second repetitions per case. All 144 configurations in
the complete Criterion suite were also exercised in smoke-test mode.

The table uses the median of the three Criterion mean times divided by rows.
Scalar entries use stream mode; geometry entries describe epoch/site reuse.

| Kernel / pattern | ns per row | Heap allocations per batch |
|---|---:|---:|
| Range-rate prediction/Jacobian | 26.9 | 0 |
| Angle prediction/Jacobian, basis-vector wrapper | 116.5 | 0 |
| Angle prediction/Jacobian, stored rotation | 111.8 | 0 |
| LOS conversion, direct frame path | 528.4 | 0 |
| LOS conversion, stored rotation | 52.5 | 0 |
| Quaternion vector rotation | 3.60 | 0 |
| Matrix vector rotation | 2.29 | 0 |
| Inverse rotation with station-position subtraction | 5.07 | 0 |
| Chained three-observable, seven-column Jacobians | 282.2 | 3 |
| Geometry: shared epoch, repeated sites | 65.0 | 11 |
| Geometry: shared epoch, unique sites | 319.3 | 17 |
| Geometry: unique epochs, repeated sites | 388.8 | 181 |
| Geometry: unique epochs and sites | 650.1 | 187 |
| Geometry: shuffled mixed epochs/sites | 101.0 | 19 |

The traversal/black-box control is about 1.06 ns per row. Do not subtract it from
the kernel measurements. Allocation measurements use a separate instrumented
binary after warm-up and count Rust heap requests, not resident memory.

## Findings and next experiments

1. **Avoid repeated basis validation on an explicitly validated internal path.**
   The angle profile attributes 18.1% self cost and 24.0% inclusive cost to
   `check_basis`, including small dot products. A future immutable validated
   geometry wrapper could check rotations once, then use an internal trusted
   helper. Keep the current public kernels' rejection of malformed rotations.
   Measure the improvement rather than treating the sampled share as a promised
   speedup. In full Jacobians the same validation is about 7.4% inclusive.
2. **Benchmark specialization or combination of chained matrix products.**
   Small matrix multiplication accounts for 32.7% of the full-Jacobian profile;
   approximately 31.3% is in its SSE2 arithmetic implementation. Compare the
   current four products per row with a combined three-observable product or
   fixed-size specialization. Also inspect finite-value validation: two
   out-of-line instantiations together account for about 11% self cost. Preserve
   validation guarantees and derivative accuracy in any subsequent change.
3. **Retain and use precomputed coordinate geometry.** Stored LOS conversion is
   about 10.1× faster than the direct path. The direct-path call stacks include
   frame rotation, Earth-orientation lookup, time conversion and ENU conversion.
   Batch OD already uses stored rotations; standalone repeated callers should
   use that API too. Primitive matrix rotation is only about 2.3 ns per row, so
   the surrounding checks and angle math offer more scope for improvement.
4. **Investigate large-batch buffers and traversal separately from arithmetic.**
   Full Jacobians allocate three buffers requesting 168 bytes per row: 10.5 MiB
   at 65,536 rows. The fixed-work sweep rises from 292.5 ns/row at 1,024 rows to
   359.5 at 65,536 rows, about 23%. A future caller-owned output-buffer variant
   could isolate allocation/reuse effects. Current evidence does not assign
   this entire increase to allocation; input traversal and output writes also
   change with size.
5. **Preserve epoch and station reuse in preprocessing.** At 1,024 rows, unique
   epochs/sites cost about 10× the shared/repeated case. Profiles show hashing,
   B-tree lookup/insertion and frame conversion. In the size sweep, shared/
   repeated geometry rises from 64.4 to 109.0 ns/row between 1,024 and 65,536
   rows; unique/unique geometry rises from 629.8 to 827.6. Test capacity planning
   and cache lifetime in a separate change before replacing lookup structures.

Angle math is also substantial: `hypot` accounts for 27.1% self cost and `atan2`
for approximately 29.8% inclusive cost in the isolated angle profile. Any faster
norm or angle implementation needs to retain behavior near singularities and
across the tested range envelope. The basis-vector wrapper is slightly slower
than passing the stored rotation directly; both still validate the basis.

## Evidence, limits and reproduction

The 33-test suite passed, including randomized central-difference measurement
and chained derivatives, quaternion/matrix and LOS roundtrips, branch crossings,
near-zenith/nadir thresholds, invalid inputs and the 65,536-row batch. All four
new stress tests also passed in the optimized release build. Clippy passed for
all targets/features with warnings denied; formatting checks passed. No numerical
failures were observed in these cases. Kernel APIs and algorithms were not
modified by this profiling work.

Hardware events reported 100% running time, and the four sampling captures
reported zero lost samples. The bundled static WSL2 perf can capture stacks but
lacks ELF/DWARF decoding. An Ubuntu perf package and its missing libtraceevent
dependency were unpacked locally; that build decoded the existing captures.
This required no system package installation. The local wrapper is
`target/profiling/perf-tools/perf`; its package hashes and version are preserved
in `provenance.json`. Some libc/math implementation addresses remain unnamed,
but recovered caller stacks identify their computation paths.

Generic cache-miss counts are saved as diagnostics; confirm their cache-level
interpretation on native Linux before drawing hardware-specific conclusions.
Reuse versus stream changes input diversity as well as working-set size, so a
timing gap can include branch/transcendental behavior. Between-run spread is
about 0.7–6.5% for the substantive 1,024-row cases. Saved thresholds are exploratory
and require additional host-specific runs before becoming regression gates.
These measurements describe isolated kernels, not estimator deadline capacity.

Generated evidence remains under `target/profiling/`:

- `baseline-20261102/`: source/compiler/dependency metadata, three sets of raw
  Criterion estimates/samples, timing/allocation binaries, `summary.csv` and
  `summary.json`.
- `scaling-20261102/`: all raw calibrated runs, binary hash and `scaling.csv`.
- `profiles-20261102/`: counter files and normalized `counters.json`, four raw
  DWARF captures, self/inclusive tables, call graphs and runner records.

The captured manifest is saved as `Cargo.toml.benchmark`. Metadata records the
later addition of `default-run = "tychonic"`, which preserves the demo command
and does not change kernel compilation settings.

Reproduce measurements with the commands in [the profiling guide](README.md).
The complete six-size Criterion baseline remains available through
`baseline.py` with its defaults; the recorded Criterion run intentionally uses
1,024 rows, with the broader size investigation recorded separately.

Integration work remains a later profiling phase: `ODProblem::eval` computes
predictions/derivatives twice, validates the problem each evaluation and builds
additional whitened output buffers; `CachedProblem` clones owned results.
These are source observations, not costs measured by the isolated harness.
