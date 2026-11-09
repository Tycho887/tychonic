# Advanced numerical API

The low-level kernels complement the measurement facade. Version 0.9.2 changes
ownership and validated-input APIs; see the [migration guide](migration_0_9_2.md).

## Conventions and API

- Cartesian states are GCRF, with positions in metres and velocities in m/s.
- The seventh parameter is **C_d A/m in m²/kg**, rather than dimensionless C_d.
  SRP susceptibility is zero in `compute_sensitivities` and `propagate_with_stm`.
- Azimuth is eastward from north; angles and angle standard deviations are
  radians. Positive range-rate means increasing distance, including station motion.
- `PreparedBatch::evaluate` returns an `Evaluation` with a whitened residual
  vector and a six- or seven-column Jacobian. Residuals are
  `(prediction − observation) / sigma`; azimuth differences wrap to [-π, π).
  Range-rate batches contain N rows; angle batches contain all N azimuths,
  then N elevations. Combined batches retain RR, AZ, EL ordering.
- Epochs are `Instant` values, including fractional seconds. Batches can have
  any size, and observation order may be unsorted or contain duplicates.
  Each `Measurement` carries its own `GroundStation` latitude/longitude in
  degrees and ellipsoidal altitude in metres.
- Public kernels return `Result`. Invalid inputs, zero range and azimuth
  singularities produce errors rather than NaNs or panics. Exact zenith/nadir
  requires dropping the azimuth observable or using a LOS model; the combined
  az/el kernel rejects horizontal/range ratios at or below 1e-12.

The implementation separates reusable components:

| Module | Responsibility |
|---|---|
| `kernels::stm` | Dense state/STM sampling and optional drag sensitivities |
| `kernels::rotation` | Checked rotations and immutable prepared station geometry |
| `kernels::observation_models` | Instantaneous observables and batch residuals/Jacobians |
| `kernels` | Shared numerical types, errors and `chain_rule(H, S)` |
| `optim` | Epoch-union sampling, batch stacking, evaluation cache and LM adapter |
| `optim::varpro` | Variable projection for per-contact range-rate biases and weak pointing constraints |
| `estimate::ekf` | STM-based EKF, heterogeneous sensors, process noise and RTS history |

Use `range_rate_partial` and `az_el_partial` directly for instantaneous EKF
measurement derivatives. For a filter with a constant drag parameter,
`compute_sensitivities` followed by `augmented_transition(i)` gives
`[Phi, dx/d(C_d A/m); 0, 1]`. `OrbitEkf` uses these transitions for prediction
and the instantaneous partials for measurement updates. Batch models compose the same instantaneous partials with
`S = Phi` or `S = [Phi, dx/d(C_d A/m)]`.

## Sequential estimation and sensor fusion

`OrbitEkf<6>` estimates position/velocity with fixed drag; `OrbitEkf<7>` also
estimates C_d A/m. Both use `PropagationConfig` and relinearize the STM at the
latest updated state for each prediction. Each sensor update invokes
`numeris::estimate::Ekf` with that sensor's measurement dimension, including
its Joseph-form covariance update. Measurement derivatives come from the
existing kernels, with no finite differencing of measurements.

Reuse prepared batches from an `ODProblem`, or supply batches directly:

```rust
use tychonic::estimate::{OrbitEkf, UpdateOptions, white_acceleration_noise};
use tychonic::kernels::{ParameterMode, PropagationConfig, od_prop_settings, v3};

// The EKF prior epoch must be at or before the earliest observation. Unlike
// batch estimation, the sequential filter cannot start at an arc's final epoch.
let config = PropagationConfig::new(
    initial_epoch, od_prop_settings(),
    ParameterMode::StateOnly { fixed_drag: 0.02 },
)?;
let mut filter = OrbitEkf::<6>::new(config, initial_state, initial_covariance)?;
let updates = filter.filter_batches(
    problem.batches(),
    |_, from, to| white_acceleration_noise(
        (to - from).as_seconds(), &v3(1e-8, 1e-8, 1e-8), 0.0,
    ),
    UpdateOptions::default(),
)?;
let smoothed = filter.smooth()?;
// Each (state, covariance) aligns with filter.timestamps().
// Raw records can also be passed directly to numeris::estimate::rts_smooth.
```

`filter_batches` accepts range-rate, azimuth/elevation and combined batches from
different stations and schedules. It sorts by exact `Instant`; ties retain
batch/row order. Noise sigmas become diagonal measurement covariances. The
process-noise callback runs once per new epoch, using the most recent estimate.
Pass a callback returning a zero matrix for deterministic dynamics. The white
acceleration helper discretizes GCRF acceleration spectral densities in m²/s³
and, for seven-state filters, a drag random walk in (m²/kg)²/s.

For streaming or future sensor types, use `Observation<M, Model>` and
`filter.observe(&observation, q, options)`, or call `predict_to` and `update`
separately. Each observation carries a full positive-definite covariance R in
physical units, so correlated components can be fused together. Q is an
already-discretized positive-semidefinite covariance in endpoint GCRF units;
the filter adds it once after propagation. Same-epoch predictions require zero
Q. `GroundSensor<1/2/3>` uses a cloned `StationGeometry` for RR, AZ/EL or combined
readings respectively; prepare it for the observation's epoch.

Implement `MeasurementModel<M>::evaluate` to return a prediction and its M×6
instantaneous state partials. A closure with this signature also works. Override
`innovation` for periodic or other non-Euclidean readings; ground sensors wrap
azimuth to [-π, π). The default innovation is observation minus prediction.
In seven-state filters the direct measurement drag column is zero, and drag
updates through the propagated cross-covariance.

Set `UpdateOptions { gate: Some(threshold) }` to reject readings by normalized
innovation squared (NIS), choosing a chi-squared threshold for the sensor's
dimension. Update results report innovation, NIS and acceptance; batch results
identify each original batch/row. Errors leave an individual prediction/update
unchanged, and `observe` rolls back the prediction if its update fails. A gated
reading retains the prediction. A batch failure preserves earlier successful
readings.

RTS history contains the initial epoch and one record per later prediction
epoch, including epochs with rejected readings or no update. Multiple sensors
at the same epoch revise that record's updated estimate without extra dynamics
or process noise. `steps()` exposes numeris `EkfStep` records, and `smooth()`
calls its built-in `rts_smooth`. The last smoothed estimate equals the last
filtered estimate. Initial and filtered covariances must be positive definite
for the smoother. The inherited approximate drag STM and force-model limits
described below apply to this estimator as well.

## Prepared observation batches

Prepare geometry and observations once, then evaluate changing reference states:

```rust
use tychonic::kernels::*;

let geometry = PreparedGeometry::new(&times, &stations)?;
let range_rates = PreparedBatch::range_rate(&geometry, observed_rr, 0.05)?;
let angles = PreparedBatch::azimuth_elevation(
    &geometry, observed_az, observed_el, 1e-4, 2e-4,
)?;
let config = PropagationConfig::new(
    reference_epoch, od_prop_settings(),
    ParameterMode::StateOnly { fixed_drag: 0.02 },
)?;
let mut output = Evaluation::default();
range_rates.evaluate_into(&initial_state_parameters, &config, &mut output)?;

// Reuse propagation across observation models on the same schedule.
let sensitivities = config.sample(&initial_state_parameters, range_rates.timestamps())?;
angles.evaluate_from_sensitivities_into(&sensitivities, &mut output)?;
```

`StateOnly` accepts six physical state parameters and samples the nominal
state/STM without drag perturbation arcs. `StateAndDrag { delta_cd: 1e-4 }`
accepts seven parameters and adds the drag finite-difference column.
`evaluate` allocates owned output; `evaluate_into` reuses its residual and
Jacobian storage while their shapes match. A shape change replaces the affected
buffer because numeris does not expose public resizing. Propagation still
allocates its own sampling storage. After an evaluation error, output contents
are unspecified; the next successful evaluation overwrites them.

Construct the existing combined batch with
`PreparedBatch::from_measurements(&measurements, noise)` or
`PreparedBatch::combined(geometry, observations)`. Measurement records preserve
each station's latitude, longitude and ellipsoidal altitude. Individual batches
have uniform measurement types and per-observable standard deviations.

The geometry constructor caches frame rotations once per exact epoch, fixed
ITRF geometry once per station, and station states/ENU matrices once per distinct
station–epoch pair. Duplicate rows reference the same immutable entry.
`geometry.row(i)` exposes read-only position, velocity and `ValidatedRotation`;
`geometry.stations()` retains input station coordinates. Caches live with the
prepared geometry and can be shared through immutable borrows. Rebuild geometry when station
coordinates or external frame data change.

`ValidatedRotation::new(matrix)` checks a finite, orthonormal, right-handed
basis once. `az_el_partial_prepared` and `convert_az_el_to_gcrf_prepared` then
reuse it without repeated basis validation. Raw-matrix and basis-vector kernels
retain their input checks. Rotations have east, north and up as columns;
`to_gcrf` performs the forward rotation and `to_enu` its transpose.

The existing raw `precompute_station_geometry`, `compute_station_trajectory`
and `compute_enu_basis` convenience functions remain available. External raw
trajectories can be checked and consumed using
`PreparedGeometry::from_trajectories(station, enu)`.

## Stacking and fitting

```rust
use tychonic::optim::{ODProblem, fit_lm};

let problem = ODProblem::new(config, vec![range_rates, angles])?;
let evaluation = problem.evaluate(&initial_state_parameters)?;
let fit = fit_lm(&problem, &initial_state_parameters, None, &lm_settings)?;
```

`ODProblem` prepares a union of exact epochs, samples sensitivities once per
parameter evaluation, and writes batch outputs directly into stacked buffers.
Batch order follows the supplied vector; each batch retains its original row
order, including duplicate and unsorted epochs. Its `evaluate_into` variant
reuses output storage. Geometry, batches and propagation configuration are
immutable and accessible through getters.

`CachedProblem::evaluate(&mut self, parameters)` returns `&Evaluation` from
reusable owned buffers. A changed parameter vector refreshes the cache; errors
invalidate the cache until another evaluation succeeds. Rust prevents another
mutable cache access while a returned evaluation remains in use.
`fit_lm(&problem, ...)` uses a sequential, fallible Tychonic LM driver. It evaluates
residuals and Jacobians together, solves scaled corrections, rejects negative-drag
trials, and propagates kernel errors directly.
Default scales are `[100, 100, 100, 0.1, 0.1, 0.1, 0.01]`, truncated for
state-only fits; callers may supply positive scales explicitly. Choose stopping
tolerances consistent with propagation accuracy. The noisy regression uses
`grad_tol=1e-2`, `x_tol=1e-5`, `f_tol=1e-5` in scaled coordinates.

The former mutable `ODProblem` struct and tuple-valued `eval` API have been
replaced. Import problem/cache types from `tychonic::optim` and access
`evaluation.residual` and `evaluation.jacobian`.

## Per-contact range-rate biases

A 24 h bootstrap fit combines many short range-rate contacts, each with its own
constant range-rate offset. `optim::varpro` treats those offsets as linear
parameters and eliminates them by variable projection, so the nonlinear fit
keeps only the six state parameters or the seven state-plus-drag parameters.

```rust
use numeris::optim::LmSettings;
use tychonic::kernels::{
    MeasurementNoise, ParameterMode, PropagationConfig, od_prop_settings,
};
use tychonic::optim::{
    ContactBiasProblem, ContactId, ContactRangeRate, PointingConstraint, fit_varpro,
};

let contacts = vec![
    ContactRangeRate {
        time,
        station,
        contact: ContactId::parse("00112233-4455-6677-8899-aabbccddeeff")?,
        range_rate: -7421.5,
    },
    // one row per observed range-rate sample, across all contacts
];
let pointing = vec![
    PointingConstraint { time, station, azimuth, elevation },
    // weak az/el pseudo-measurements
];
let noise = MeasurementNoise { sigma_rr: 0.05, sigma_az: 1e-3, sigma_el: 1e-3 };
let config = PropagationConfig::new(
    mid_arc_epoch,
    od_prop_settings(),
    ParameterMode::StateAndDrag { delta_cd: 1e-4 },
)?;
let problem = ContactBiasProblem::new(config, &contacts, &pointing, noise)?;
let fit = fit_varpro(&problem, &guess, None, &LmSettings::default())?;
// fit.parameters: 7 nonlinear parameters
// fit.biases: one constant m/s offset per contact
// fit.covariance: state block of the joint covariance
```

`fit_bootstrap(config, range_rates, pointing, noise, guess, scales, settings)`
is the one-call equivalent of `ContactBiasProblem::new` plus `fit_varpro`.
`ContactBiasProblem::contacts`, `inner`, `bias_map`, `evaluate_reduced`,
`solve_biases` and `covariance` expose the stacked problem and its reduced
evaluation for staged use.

Range-rate rows are stacked before the `2 * n_pointing` angle rows, and only
range-rate rows receive a bias column. The bias is a constant added to the
range-rate prediction, in m/s; the optimum is
`b* = -(AᵀA)⁻¹Aᵀr` on the whitened residual. `VarproFit::biases` reports one
value per contact in first-appearance order across the input rows; reordering
rows changes that order. `ContactId` accepts 32 bare hex digits or the
hyphenated UUID form.

Azimuth and elevation pseudo-measurements carry no bias column. They are weak
constraints for directions that sparse contacts leave poorly observed, not
precision measurements. Set `sigma_az` and `sigma_el` to the trust those angles
deserve: small sigmas make them hard observations, large sigmas let the
range-rate data dominate.

`VarproFit::covariance` is the state block of the joint covariance after
eliminating the biases, scaled by `RSS / (rows - columns - contacts)` and
multiplied by `(JᵀPJ)⁻¹` of the projected Jacobian. It requires positive
residual degrees of freedom; malformed inputs and non-positive dof return
`KernelError::InvalidInput`, and a singular normal matrix returns
`KernelError::Singular`. For a 24 h multi-contact span, use a mid-arc reference
epoch so the state and its STM stay centred on the data instead of
extrapolating from one end of the day.

## Numerical scope and verification

`od_prop_settings()` uses spherical-harmonic Earth gravity, Sun/Moon gravity,
and atmospheric drag with tight integration tolerances. Satkit's STM omits
tide and relativity derivatives; these kernels explicitly reject those force
settings and integrators without STM support. Satkit approximates the drag
density gradient using altitude only, so its drag STM is approximate. One-hour
tests bound linearized errors below 2 mm and 5 µm/s for 10 m position or 0.01 m/s
velocity perturbations in the tested orbits. Longer arcs, very low orbits, and
different force settings need additional qualification.

The drag column uses central differences, with a second-order forward stencil
near zero. `delta_cd` is an absolute step in m²/kg. Check step-size stability for
your orbit, arc length and integration tolerances. Batch propagation integrates
complete forward/backward arcs and samples dense output, avoiding a separate
integration from the reference epoch for every observation.

Station states, ENU bases and angle-to-LOS conversion consistently use satkit's
approximate IAU-76/FK5 frame chain (roughly arcsecond accuracy). This is a
geometric measurement model: light-time, refraction, station displacement,
clock errors and sensor biases are not modelled. Synthetic fits verify internal
consistency, rather than absolute accuracy against real tracking data.

Regression coverage includes analytic versus central-difference measurement
Jacobians across quadrants, branch crossings, rotations and ranges; zenith/nadir
and invalid inputs; subsecond station motion; the published Vallado SGP4 case;
forward/backward LEO, eccentric and GEO STM checks; independent endpoint versus
dense output; zero-time identity; drag sensitivity and step stability; augmented
transition composition; batch Jacobian whitening and residual signs; sizes
through 1024; cache behavior; and noisy/noiseless seven-parameter orbit fits.
Mixed-station checks cover shared-epoch quaternion reuse, direct quaternion
versus matrix rotations, altitude-dependent station states, batch Jacobians,
and simultaneous observations from two stations in an orbit fit.
