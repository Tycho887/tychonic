# tychonic

Passive orbit determination in Rust using **satkit** for astrodynamics and
**numeris** for optimization, filtering, and smoothing. Tychonic predicts
observables and analytic state derivatives, fits batches, and updates an orbit
as new measurements arrive.

Version **0.9.2** accepts extracted Doppler, phase difference, differential
Doppler, and azimuth/elevation. Each calibrated observable works independently,
and any combination can be fused with full covariance. Fitting requires an
initial orbit guess; initialization without one is future work. IQ processing,
phase unwrapping, and cycle-slip detection belong upstream.

## Install

Add tychonic from [crates.io](https://crates.io/crates/tychonic):

```toml
[dependencies]
tychonic = "0.9.2"
```

See the [API documentation](https://docs.rs/tychonic) for available types and
functions. `tychonic::satkit` and `tychonic::numeris` re-export the dependency
versions accepted by the API.

## Acquire with BC-MPC

`tychonic::detection` provides Bayesian Continuation MPC acquisition kernels
and `BcMpc` state. Supply a nominal TLE or OMM, explicit particles and weights,
and a communications model implementing `DetectionModel`. Initialization
privately builds a feasible along-track continuation; each plan optimizes its
prefix against acquisition probability over the entire remaining search.

The caller owns the antenna, receiver, clock, and resampling randomness:

```rust,no_run
use tychonic::detection::{Action, BcMpc, DetectionModel, Observation,
    Result, SearchStatus};

fn acquire(mut controller: BcMpc, receiver: &mut impl DetectionModel,
    mut execute: impl FnMut(Action) -> Result<Observation>,
    mut uniform: impl FnMut() -> f64) -> Result<SearchStatus> {
    while !controller.finished() {
        let action = controller.plan(receiver)?.actions[0];
        let observation = execute(action)?;
        controller.observe(observation, &mut uniform)?;
    }
    Ok(controller.status())
}
```

Execute only the first returned action. Report a miss only after successful
execution of the full listening interval; an actuator or API error leaves it
pending. Confirmation ends acquisition. Pointings and actuator limits use
radians, distances metres, and durations seconds. See [acquisition API and
defaults](docs/acquisition.md) and the [reference integration test](tests/acquisition.rs).

## Prepare individual measurements

```rust,no_run
use tychonic::{kernels::{GroundStation, Interferometer, PreparedGeometry, v3},
    measurements as m, satkit::Instant};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let times = vec![Instant::from_unixtime(1704110400.25)];
let stations = vec![GroundStation { lat_deg: 40.0, lon_deg: -75.0, alt_m: 100.0 }];
let geometry = PreparedGeometry::new(&times, &stations)?;
let baseline = Interferometer::new(v3(59.0, 3.0, 0.2), 2.23e9)?;

let doppler = m::doppler(&geometry, 2.23e9, vec![-12500.0], 0.5)?;
let phase = m::phase_difference(&geometry, baseline, vec![12.4], 0.01)?;
let differential = m::differential_doppler(&geometry, baseline, vec![0.03], 0.7)?;
let angles = m::azimuth_elevation(&geometry, vec![1.2], vec![0.7], 1e-4, 2e-4)?;
# Ok(()) }
```

These constructors assume calibrated receiver offsets. For known nonzero
offsets or correlated components, use `m::joint` with `m::Observable` models.
Each values row and covariance is ordered **Doppler, phase difference,
differential Doppler, azimuth, elevation**, omitting absent components.
Covariance is supplied per epoch in physical units. Ordinary and differential
Doppler from the same antenna share noise; use
`m::doppler_differential_covariance(sigma1, sigma2)` for their 2×2 block.
See [the complete joint example](examples/passive_fusion.rs).

## Fit and stream

`tychonic::fit(config, batches, &guess)` stacks independent schedules and
shares propagation over their epoch union. `fit_with_options` also accepts
parameter scales and numeris LM settings. Both return fitted parameters,
residual sum of squares, and iteration count. Choose tolerances appropriate
to measurement noise and propagation accuracy.

```rust,no_run
use tychonic::{OrbitFilter, estimate::UpdateOptions,
    kernels::{PreparedBatch, PropagationConfig, V6, M6}, numeris::DynVector};

fn fit_and_stream(config: PropagationConfig, batches: Vec<PreparedBatch<'_>>,
    guess: V6, prior_covariance: M6) -> Result<(), Box<dyn std::error::Error>> {
    let fit = tychonic::fit(PropagationConfig::new(config.reference_epoch(), config.settings().clone(), config.mode())?, &batches,
        &DynVector::from_slice(guess.as_slice()))?;
    let mut filter = OrbitFilter::new(config, guess, prior_covariance)?;
    let updates = filter.filter_batches(&batches, |_, _, _| Ok(M6::zeros()),
        UpdateOptions::default())?;
    let smoothed = filter.smooth()?;
    Ok(())
}
```

The prior must precede or equal the earliest streaming observation. Updates
sort exact epochs stably, apply process noise once per new epoch, and preserve
full covariance within joint observations. `UpdateOptions::gate` enables NIS
rejection. A failed observation rolls back its prediction; a rejected reading
retains it. `OrbitFilterWithDrag` adds constant C_d A/m to the six orbital states.

For individual readings, the named `m::*_sensor` constructors create models
for `estimate::Observation` and `filter.observe`. A prepared facade batch also
supplies `batch.joint_observation::<M>(row)` with values, full covariance, epoch,
and model; M is the component count, from 1 through 5.

## Correct a TLE epoch

`estimate::TcaUkf` estimates `[dT_seconds, range_rate_bias_m_per_s]` for one
pass from combined range-rate and azimuth/elevation readings. Positive `dT`
moves the prior TLE epoch later while preserving its other elements. Satellite
propagation, station geometry, and frame conversion all use the actual reading
time. Range-rate is positive when receding; angles are radians.

```rust,no_run
use tychonic::{
    estimate::{TcaUkf, UpdateOptions, tca_measurement_noise},
    kernels::{KernelError, Measurement, Result},
    numeris::{Matrix, Vector},
    satkit::TLE,
};

fn correct_pass(prior: TLE, readings: &[Measurement]) -> Result<TcaUkf> {
    let epoch = readings.first().ok_or(KernelError::InvalidInput("empty pass"))?.time;
    // Standard deviations: 150 seconds of epoch error and 200 m/s of bias.
    let covariance = Matrix::new([[150.0_f64.powi(2), 0.0], [0.0, 200.0_f64.powi(2)]]);
    let mut filter = TcaUkf::new(prior, epoch, Vector::zeros(), covariance)?;
    let noise = tca_measurement_noise(0.5)?; // 0.5 m/s RR; 2.5-degree az/el.
    for reading in readings {
        filter.observe(reading, &noise, None, UpdateOptions::default())?;
    }
    let corrected_tle = filter.corrected_tle()?;
    // Persist corrected_tle and filter.state()[1] as separate orbit/sensor results.
    Ok(filter)
}
```

`tca_measurement_noise` treats 2.5 degrees as an independent Gaussian 1σ for
each angle; its returned `MeasurementNoise` can be adjusted in radians. There
is no hard angular bound. State and bias remain constant between readings
unless a discretized 2×2 process covariance is supplied. Readings must be
chronological; same-time updates add no process noise. NIS gating is disabled
by default to allow acquisition with a distant prior. A failed reading rolls
back prediction and update; a gated reading retains prediction.

Choose prior covariance to cover the expected epoch error. The UKF uses five
sigma points and a circular azimuth mean; it assumes one roughly Gaussian
solution and does not search across passes or fit other orbital elements.

## Unknown RF offsets and drift

`m::RfBuilder` accepts `RfPriors` with named orbit, frequency-offset and
phase-offset priors plus an explicit `OuPhaseNoise` model. Use `build()` for the
advanced batch problem, `fit()` or `fit_with_options()` for fitting, and
`filter::<9>()` / `filter::<10>()` for streaming without / with estimated drag.
Optional baseline calibration adds three states per baseline.

This path retains the existing Doppler, Doppler/phase, and
Doppler/phase/differential combinations. Angle fusion with unknown RF drift
is future work. See [the RF setup example](examples/rf_offsets.rs) and
[the phase and drift model](docs/phase_difference.md).

## Units and numerical scope

- States: GCRF position in metres and velocity in m/s. Drag is C_d A/m in m²/kg.
- Doppler: shift relative to the nominal carrier in Hz; geometric shift is
  `−fc * range_rate / c`. Positive range rate means increasing distance.
- Baselines: **antenna 1 minus antenna 2**, in antenna 1's ENU frame in metres.
  Differential Doppler is **f1 minus f2**, using both moving antenna sites.
- Phase: continuous radians with ordinary subtraction. Azimuth: eastward from
  north, wrapped residuals in [−π, π); elevation and angle sigmas: radians.
- The Henault/Guimond far-field phase model is the default;
  `Interferometer::with_spherical_phase()` selects the existing exact range
  difference model. Light-time and atmospheric measurement corrections are
  outside the current models.

Version 0.9.2 updates ownership and validated-input APIs. See the
[migration guide](docs/migration_0_9_2.md) and [safety contracts](docs/safety.md).
`MeasurementKind::Joint` must be handled in exhaustive matches.
See [the advanced API](docs/advanced_api.md) for propagation limits, prepared
geometry, variable projection, process noise, and numerical verification, and
[differential Doppler](docs/differential_doppler.md) for covariance and baselines.

## Development and release

Frame transforms and propagation require Satkit's Earth-orientation, gravity,
and ephemeris data. Provision them with Satkit's documented setup, for example
`satkit::utils::update_datafiles(None, false)`, or select an existing cache with
`SATKIT_DATA`. `SATKIT_OFFLINE=1` uses cached data without network downloads.
Tests fail when required data is missing rather than silently skipping checks.

```sh
SATKIT_OFFLINE=1 cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --all -- --check
cargo doc --locked --no-deps --lib
cargo bench --locked --bench kernels
```

[Research tools](https://github.com/Tycho887/tychonic/blob/main/tools/README.md) live in a separate development crate. Generated
runs stay locally under ignored `artifacts/`; local paper PDFs stay under
ignored `references/`. Selected documentation figures belong in `docs/assets/`.
See [references](references/README.md) and [release packaging](docs/publishing.md).

Licensed under **MIT OR Apache-2.0**.
