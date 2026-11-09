# Migrating to 0.9.2

Version 0.9.2 intentionally changes ownership and validated-input APIs. The
measurement equations, signs, units, parameter layouts, and numerical thresholds
are preserved. See [safety contracts](safety.md) for the dependency boundary.

## Geometry and sensors

Pass an owned `PreparedGeometry` to transfer it into a batch, or `&geometry` to
share one schedule across batches. Replace `Arc<PreparedGeometry>` and its clones
with ordinary ownership and borrows. Batches have type `PreparedBatch<'g>`;
owned batches can have lifetime `'static`. Their `geometry()` accessor returns
`&PreparedGeometry`.

Sensor constructors likewise accept a station value or `&StationGeometry`.
`CalibratedSensor<'g, M>`, `GroundSensor<'g, M>`, and `PhaseSensor<'g>` carry the
borrow lifetime. Batch-created calibrated sensors borrow both the model and
geometry. In a `PhaseSensor` literal, use `geometry: station.into()`.

```rust
# use tychonic::kernels::*;
# fn example() -> Result<()> {
let geometry = PreparedGeometry::new(&[], &[])?;
// For real, nonempty measurements:
// let rr = PreparedBatch::range_rate(&geometry, values, sigma)?;
// let angles = PreparedBatch::azimuth_elevation(&geometry, az, el, sigma_az, sigma_el)?;
# Ok(()) }
```

## Validated inputs and reusable workspaces

Replace `Sensitivities` literals with
`Sensitivities::new(timestamps, states, stm, state_dcd)?`. Read arrays through
`timestamps()`, `states()`, `stm()`, and `state_dcd()`, or use `row()` and
`iter_rows()`. `into_parts()` moves buffers out when preparing changed samples;
construct a new validated value afterward.

Replace `PassiveRfData` literals with
`PassiveRfData::new(geometry, interferometer, doppler, phase, sigma_doppler, sigma_phase)?`.
Its fields are read through named accessors. Replace `RfPrior` literals with
`RfPrior::new(mean, sigma)?`, followed by `mean()` and `sigma()` for reads.
Invalid data now fail at construction. `RfExtensions` remains an unchecked setup
value whose combinations are validated when constructing or feeding an estimator.

`CachedProblem::evaluate` requires `&mut self` and returns `&Evaluation`.
The returned borrow ends before another mutable cache access. Store fitted
results or explicitly owned output when independent persistence is required.
A failed refresh invalidates the cache. `evaluate_into` methods reuse caller-owned
buffers; a failed evaluation requires a successful reevaluation before consumption.

## Problems, fitting, and filters

`ODProblem::new` accepts an owned vector of batches, a borrowed vector, or a
borrowed slice. Its lifetime tracks both geometry and borrowed batches.
`fit_lm(&problem, ...)` and `fit_varpro(&problem, ...)` borrow their prepared
problems. The fit facade also accepts owned or borrowed batches.

`PassiveRfProblem::new` / `new_extended` accept owned or borrowed RF data, priors,
and extensions. They own their propagation configuration. `RfBuilder::build`,
`fit`, and `fit_with_options` borrow the builder, so it can be reused without
cloning. Builder configuration methods continue consuming and returning the
builder; configure it before borrowing its built problem.

`FitError::Solver` now carries `numeris::optim::OptimError`, including
`MaxIterations`, rather than a string. `KernelError::Numerical` identifies detected
arithmetic failures separately from invalid input and singular systems.

Filter `steps()` returns `&FilterHistory<N>`. Use `len()`, `get(i)`, or `iter()`;
replace `steps()[i]` with a checked `get(i)`. For direct Numeris RTS calls,
`steps().to_records()` explicitly materializes a contiguous record vector.
Normal `filter.smooth()` handles this conversion. Filter and TCA estimator
cloning is removed; initialize an independent estimator from its state and
covariance when required.

If the same configuration must initialize independent estimators, construct each
`PropagationConfig` explicitly. Copying borrowed Satkit settings at that boundary
is supported; configuration cloning inside numerical evaluation is avoided.
