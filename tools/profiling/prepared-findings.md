# Prepared kernel results — 3 November 2026

The refactor validates reusable rotations once, computes observations once per
state, composes their partials through the STM, and writes directly into whitened
outputs. It supports six-state and seven-parameter batches without changing the
measurement equations or numerical tolerances.

## Observation results

Median mean nanoseconds per row; three fixed-work repetitions:

| Kernel | 16 rows | 1,024 rows | 65,536 rows |
|---|---:|---:|---:|
| Checked stored-rotation angles | 109.9 | 129.0 | 134.3 |
| Prepared angles | 65.2 | 84.9 | 99.0 |
| Checked stored-rotation LOS | 53.5 | 55.5 | 63.8 |
| Prepared LOS | 26.2 | 32.5 | 39.4 |
| Reference observation assembly | 502.5 | 516.1 | 613.3 |
| Prepared owned assembly | 239.7 | 244.5 | 292.2 |
| Prepared reused assembly | 222.0 | 260.5 | 293.2 |

Prepared angles reduce the measured cost by about 34% at 1,024 rows. Owned
observation assembly is about 52–53% faster than the reconstructed pre-refactor
assembly at all three sizes. The reference and prepared batch paths produce
equivalent whitened residuals and Jacobians from the same sampled sensitivities.
The reference includes duplicate predictions/partials and three intermediate
Jacobian buffers, as the previous `ODProblem::eval` did. Propagation and repeated
problem validation are excluded from both assembly measurements.

Reuse eliminates output allocations, but this sweep does not show a consistent
additional timing improvement over owned output. Allocation requests at 1,024
rows, measured separately after warm-up:

| Assembly | Allocations per batch | Requested bytes per row |
|---|---:|---:|
| batch-reference | 5 | 360 |
| batch-owned | 2 | 192 |
| batch-reused | 0 | 0 |

The old outputs request 360 bytes/row including intermediates; owned prepared
outputs request 192 bytes/row. Reused outputs make zero allocation or
reallocation requests for every measured size. Propagation and cache-key/result
storage still allocate outside this isolated assembly path. Shape changes replace
the affected output buffer because numeris exposes no public dynamic resize API.

## Chain products and geometry

The isolated four-product chain costs 90.9 ns/row at 1,024 rows; combining the
observable rows into two products costs 86.5, and forming an augmented 6×7
sensitivity followed by one product costs 72.4. At 65,536 rows these are 114.0,
94.2 and 89.1 ns/row. The selected implementation uses one `H*S` product with
fixed-size specialization for each measurement type and parameter count.
Microbenchmark improvements do not imply equal end-to-end improvements.

Final geometry measurements use the selected monotonic-epoch path:

| Constructor / pattern | 16 rows | 1,024 rows | 65,536 rows |
|---|---:|---:|---:|
| Raw / shared-repeated | 339.5 | 61.8 | 100.3 |
| Prepared / shared-repeated | 332.1 | 40.3 | 37.4 |
| Raw / unique-unique | 663.0 | 682.3 | 753.4 |
| Prepared / unique-unique | 593.9 | 637.8 | 784.8 |

Strictly increasing or decreasing epochs cannot repeat, so that path reserves
its final entry buffer and omits epoch/pair lookup tables. Other schedules hash
satkit's exact TAI microsecond value and an interned station index. Duplicate
station/epoch rows reference the same validated entry, and frame rotations are
computed once per distinct epoch.

At 65,536 repeated rows, prepared geometry is about 2.7× faster. Entirely unique
ordered geometry is about 4.2% slower than raw preprocessing at that size, while
being faster at the two smaller sizes. Prepared construction includes basis and
state validation that raw preprocessing does not perform. This remains a setup
tradeoff; the geometry is reused across parameter evaluations.

At 1,024 shared/repeated rows, requested allocation bytes fall from 165.3 to
47.5 per row. At 1,024 unique rows, allocation requests fall from 187 to 11, but
requested bytes rise from 562.3 to 608.6 per row because the fixed-station cache
stores more data. These counts include temporary constructor caches, not just
retained geometry; fewer allocation requests do not imply lower total memory.

The earlier tree-based candidate cost 1,071.8 ns/row for 65,536 unique rows.
Replacing the trees with hashes alone still cost 1,059.1 in the next sweep.
Removing redundant tables for ordered epochs brought this to 784.8. The
comparison with the raw constructor in that final sweep is 753.4 ns/row.
Unsorted unique schedules continue to use the general cache path; their timing
is not separately qualified by this two-pattern sweep.

## Verification and evidence

All 40 tests pass with all features, including finite-difference derivatives for
each measurement type and both parameter counts, frame/LOS checks, fractional
and duplicate epochs, forward/backward STM sampling, mixed-schedule stacking,
cache invalidation, invalid input and singularity rejection, output reuse, and
six- and seven-parameter synthetic fits. The 65,536-row stress test also checks
prepared outputs against selected raw Jacobian rows. Release prepared/stress
checks pass; the release demo converges in five iterations with approximately
one micrometre position error. Clippy with warnings denied and formatting checks
pass. All 252 Criterion kernel/size/mode/pattern cases pass smoke testing.

Measurements use the existing Intel i9-9900K/WSL2 host, default release flags,
seed `0x747963686f6e6963`, offline satkit data, 0.2-second warm-ups and calibrated
work targeting one second per repetition. They ran sequentially without tests
or builds competing for CPU time. They are repeated mean timings without
Criterion confidence intervals, not latency percentiles or estimator deadlines.
Between-run spread in the observation/chain sweep ranges from roughly 1% to 23%.
Use fresh host-specific repetitions before setting regression gates.

Generated evidence remains under `target/profiling/`:

- `prepared-first-20261103/`: first two-product/tree-cache candidate and timing binary.
- `prepared-final-20261103/`: full 42-case scalar/chain/assembly/hash-cache sweep,
  raw runs, timing binary and source/hardware metadata.
- `prepared-geometry-selected-20261103/`: 12-case final geometry sweep, timing
  and allocation binaries, all 42 allocation profiles, source metadata and the
  252-case Criterion smoke log.

The collector was subsequently corrected to print the pattern for both geometry
kernels. Original timing records are retained; their outer `variant` identifies
prepared-geometry patterns. Final allocation records contain the corrected field.
Reproduction commands are in [the profiling guide](README.md).
