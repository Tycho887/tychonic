# Python API reference · `tychonic.low_level`

Use `from tychonic import low_level as ll` for the names below. These adapters operate on individual measurement batches, filters, and acquisition controllers. Pass times as `satkit.time` objects, for example `satkit.time.from_rfc3339("2024-01-01T12:00:00Z")`. A bare `*` introduces keyword-only arguments. For the spacecraft catalogue and job-based interfaces, see the [main API reference](python-api.md).

| Task | Look up |
| --- | --- |
| [Describe a sensor](#describe-a-sensor) | `GroundStation`, `Interferometer`, `Observable` and its factories |
| [Prepare and predict measurements](#prepare-and-predict-measurements) | `MeasurementBatch`, `measurements.joint`, `doppler`, `phase_difference`, `differential_doppler`, `azimuth_elevation`, `doppler_differential_covariance` |
| [Configure propagation and fitting](#configure-propagation-and-fitting) | `PropagationSettings`, `PropagationConfig`, `LMOptions`, `fit_orbit`, `FitResult` |
| [Filter and smooth an orbit](#filter-and-smooth-an-orbit) | `OrbitEKF`, `FilterOptions`, `WhiteAccelerationNoise`, `UpdateResult`, `BatchUpdate`, `FilterHistory` |
| [Correct a TLE during a pass](#correct-a-tle-during-a-pass) | `TcaUKF`, `TcaMeasurementNoise`, `TcaCorrection` |
| [Plan with a receiver callback](#plan-with-a-receiver-callback) | `BCMPC`, `MPCConfig`, `AntennaLimits`, `SearchSetup`, `ParticleGeometry`, `Action`, `ActionPlan`, `Observation`, `ObservationUpdate`, `SearchStatus` |
| [Interpret errors and versions](#errors-and-versions) | `TychonicError`, `DataError`, `NumericalError`, `SolverError`, `AcquisitionStateError`, `__version__`, `__core_version__` |

## Describe a sensor

### GroundStation and Interferometer

```text
ll.GroundStation(lat_deg, lon_deg, alt_m=0.0)
ll.Interferometer(baseline_enu_m, carrier_hz, spherical_phase=False)
```

A low-level station holds geodetic latitude and longitude (degrees) and altitude (metres). An interferometer combines a positive RF carrier in Hz with a three-component baseline from antenna 2 to antenna 1, expressed in antenna 1's east, north, up frame in metres. The phase model uses the far-field approximation by default. Set `spherical_phase=True` to compute the exact range difference.

### Observable

```text
ll.Observable(kind, carrier_hz=None, offset_hz=0.0, baseline_enu_m=None,
              spherical_phase=False, frequency_offset_hz=0.0, phase_offset_rad=0.0)
ll.Observable.doppler(carrier_hz, *, offset_hz=0.0) -> Observable
ll.Observable.phase_difference(interferometer, *, frequency_offset_hz=0.0,
                               phase_offset_rad=0.0) -> Observable
ll.Observable.differential_doppler(interferometer) -> Observable
ll.Observable.azimuth_elevation() -> Observable
```

Use the named factories to configure a model for a joint batch. The direct constructor's `kind` is the corresponding factory name as a string.

| Factory | Model and calibration arguments |
| --- | --- |
| `doppler` | Signed frequency shift in Hz for the given carrier. The geometric term is `-carrier_hz * range_rate / c`. `offset_hz` adds a known receiver offset. |
| `phase_difference` | Continuous, unwrapped phase in radians using the supplied `Interferometer`. `frequency_offset_hz` adds a known offset to the predicted received frequency used in the phase model and `phase_offset_rad` adds a constant phase offset. |
| `differential_doppler` | Antenna-1 minus antenna-2 Doppler in Hz, using that interferometer's carrier and baseline. |
| `azimuth_elevation` | Two pointing angles in radians, with azimuth eastward from north, elevation above the horizon, and a wrapped azimuth residual. |

## Prepare and predict measurements

### MeasurementBatch

```text
ll.MeasurementBatch(times, stations, observables, values, covariance)
ll.measurements.joint(times, stations, observables, values, covariance) -> MeasurementBatch
batch.predict(x, *, propagation) -> ndarray
```

The constructor and `joint` build an owned batch with the same arguments. Inputs are copied. Use separate batches for independent schedules.

| Argument | Expected input |
| --- | --- |
| `times`, `stations` | A sequence of `N` Satkit epochs and one `ll.GroundStation` for all rows, or a sequence of stations aligned with the rows |
| `observables` | A sequence of distinct configured `Observable` models. Azimuth/elevation contributes two components. |
| `values` | Finite array of shape `(N, M)`. Regardless of model order, put the `M` columns in Doppler, phase difference, differential Doppler, azimuth, elevation order, skipping missing channels. |
| `covariance` | Symmetric positive-definite measurement covariance in the same units and component order. Shape `(M, M)` shares one matrix across rows and `(N, M, M)` supplies one per row. |

Read `batch.times` for its tuple of Satkit epochs, `batch.components` for `M`, and `len(batch)` for `N`. To simulate measurements, pass a six- or seven-component GCRF state `x` at the reference epoch of a `PropagationConfig` to `predict`. It returns an `(N, M)` array in measurement units.

### Measurement helpers

These functions live on `ll.measurements`. The four measurement-specific factories return a `MeasurementBatch`. Each takes the same `times` and `stations` as `joint`, a vector of observations per measured component, and positive finite scalar standard deviations shared by every row.

| Call | When to use it |
| --- | --- |
| `ll.measurements.doppler(times, stations, carrier_hz, values, sigma_hz)` | Supply a positive carrier in Hz, signed downlink shifts in Hz, and their sigma in Hz. |
| `ll.measurements.phase_difference(times, stations, interferometer, values, sigma_rad)` | Supply an `Interferometer`, unwrapped phase differences in radians, and phase sigma in radians. |
| `ll.measurements.differential_doppler(times, stations, interferometer, values, sigma_hz)` | Supply an `Interferometer`, antenna-1 minus antenna-2 shifts in Hz, and differential sigma in Hz. |
| `ll.measurements.azimuth_elevation(times, stations, azimuth, elevation, sigma_az, sigma_el)` | Supply aligned angle vectors and their independent standard deviations, all in radians. |
| `ll.measurements.doppler_differential_covariance(sigma1, sigma2)` | Given independent antenna Doppler standard deviations in Hz, return the 2×2 covariance for antenna-1 Doppler and their difference in Hz². The diagonal is `[sigma1**2, sigma1**2 + sigma2**2]` and both cross terms are `sigma1**2`. Pass this block to `joint` when using both channels. |

## Configure propagation and fitting

### PropagationSettings and PropagationConfig

```text
ll.PropagationSettings(gravity_degree=4, gravity_order=4, gravity_model="EGM2008",
                       abs_error=1e-10, rel_error=1e-13, use_spaceweather=True,
                       use_sun_gravity=True, use_moon_gravity=True, enable_interp=True,
                       integrator="RKV98", max_steps=1_000_000,
                       require_eop_coverage=False, initial_step_secs=None)
ll.PropagationConfig(epoch, settings=PropagationSettings(), estimate_drag=False,
                     fixed_drag=0.0, delta_cd=1e-4)
```

| Settings | Meaning |
| --- | --- |
| `gravity_degree`, `gravity_order`, `gravity_model` | Maximum spherical harmonic degree and order, and geopotential model. Model names include `"EGM2008"`, `"EGM96"`, `"JGM2"`, `"JGM3"`, and `"ITUGrace16"`. |
| `abs_error`, `rel_error` | Absolute and relative integration error tolerances for the GCRF SI state. |
| `use_spaceweather`, `use_sun_gravity`, `use_moon_gravity` | Enable space-weather inputs to the atmosphere and the solar and lunar gravity terms, respectively. |
| `enable_interp`, `integrator`, `max_steps`, `initial_step_secs` | Enable dense interpolation, choose the integration method, bound the step count, and optionally supply an initial step hint in seconds. Adaptive integrator names include `"RKV98"`, `"RKV98NoInterp"`, `"RKV87"`, `"RKV65"`, and `"RKTS54"`. |
| `require_eop_coverage` | Require Earth orientation parameter data covering the propagation interval. |

Use `PropagationConfig` to pair those settings with the Satkit reference `epoch` of the input state. In the default six-state model, the order is GCRF x, y, z (m) followed by vx, vy, vz (m/s), with `fixed_drag` supplying drag coefficient times area/mass in m²/kg. With `estimate_drag=True`, append that drag parameter to the state. `delta_cd` is the finite-difference step for its sensitivity in m²/kg. Read `propagation.dimension` for the required state length, 6 or 7.

### fit_orbit, LMOptions, and FitResult

```text
ll.fit_orbit(batches, x0, *, propagation, options=None) -> FitResult
ll.LMOptions(max_iter=100, grad_tol=1e-8, x_tol=1e-12, f_tol=1e-12, mu_init=1e-3,
             mu_min=1e-10, mu_max=1e10, mu_increase=10.0, mu_decrease=0.1,
             parameter_scales=None)
ll.FitResult(x, rss, nit)
```

Use `fit_orbit` to fit one reference state to a sequence of calibrated `MeasurementBatch` objects. Pass the starting vector `x0` at `propagation.epoch`. The solver weights residuals by the batch covariances. `options=None` uses `LMOptions()`. For a fitted state with absolute covariance, see [CalibratedLM in the main API](python-api.md#solver-profiles).

- `max_iter` caps solver iterations. `grad_tol`, `x_tol`, and `f_tol` set stopping tolerances for gradient, parameter step, and cost change in scaled coordinates.
- `mu_init`, `mu_min`, and `mu_max` set initial, minimum, and maximum damping. `mu_increase` and `mu_decrease` multiply the damping on adjustment.
- `parameter_scales` supplies one scale per fitted parameter. The defaults are 100 m for each position, 0.1 m/s for each velocity, and 0.01 m²/kg for estimated drag.

Read the fitted state at the reference epoch from `result.x`, the weighted residual sum of squares from `result.rss`, and the iteration count from `result.nit`. The `result.cost` property is half the RSS.

## Filter and smooth an orbit

### FilterOptions and WhiteAccelerationNoise

```text
ll.FilterOptions(gate=None)
ll.WhiteAccelerationNoise(acceleration_psd=0.0, drag_psd=0.0)
noise(state, previous, next_epoch) -> ndarray
```

`gate` sets a finite nonnegative normalized innovation squared (NIS) threshold on either low-level filter. Select a chi-squared threshold for the measurement dimension or use `None` to accept readings without gating.

A `WhiteAccelerationNoise` instance is a process-noise callback. Give `acceleration_psd` as a common scalar or a triple of GCRF axis power spectral densities (m²/s³). Optional `drag_psd` drives a drag random walk in (m²/kg)²/s for seven-state propagation. Calling the instance with the current state vector and two Satkit epochs returns a discretized, square process covariance for that time step.

### OrbitEKF

```text
ll.OrbitEKF(x0, covariance, *, propagation, process_noise=None, options=None)
ekf.predict_to(time, q=None) -> None
ekf.observe(batch, row=0, q=None) -> UpdateResult
ekf.filter_batches(batches) -> list[BatchUpdate]
ekf.smooth() -> FilterHistory
```

Seed the extended Kalman filter with `x0` and its square `covariance`, dimensioned for `propagation`. The initial epoch is `propagation.epoch`. `options` defaults to `FilterOptions()`. Read `ekf.state`, `ekf.covariance`, and `ekf.epoch` for the current snapshot, `ekf.options` for its gate settings, and `ekf.history` for the retained forward history.

- `predict_to` advances to the given Satkit time. `observe` predicts to the selected zero-based row of a `MeasurementBatch` and jointly updates its components. For either call, `q` is an already discretized square process covariance that overrides `process_noise`. A new epoch uses the callback, or zero noise by default. Prediction at the current epoch requires zero process covariance.
- A custom `process_noise(state, previous, next_epoch)` receives the supplied state and Satkit epochs. Return a matrix with shape `(propagation.dimension, propagation.dimension)`. Use that state inside the callback to avoid reentering the filter.
- `filter_batches` processes a sequence of batches in stable chronological order, applying process noise once per new epoch. It returns one record per input row. A gated row retains the prediction. If a row raises a numerical error, earlier successful rows remain committed.
- `smooth` runs Rauch–Tung–Striebel smoothing and returns snapshots aligned with the epochs in `ekf.history`, including the initial epoch.

This update compares one measured Doppler shift with the shift predicted from the prior orbit. The diagonal prior covariance gives position variances in m² and velocity variances in (m/s)².

```python
import numpy as np
import satkit
from tychonic import low_level as ll

epoch = satkit.time.from_rfc3339("2024-01-01T12:00:00.250000Z")
station = ll.GroundStation(40.0, -75.0, 100.0)
propagation = ll.PropagationConfig(epoch, settings=ll.PropagationSettings(use_spaceweather=False))
batch = ll.measurements.doppler([epoch], station, 2.23e9, [-8504.0], sigma_hz=0.5)
ekf = ll.OrbitEKF(
    [7e6, 0, 1e5, 0, 7400, 100], np.diag([1e4] * 3 + [0.01] * 3),
    propagation=propagation,
)
update = ekf.observe(batch)
innovation_hz = update.innovation[0]
```

### UpdateResult, BatchUpdate, and FilterHistory

```text
ll.UpdateResult(innovation, nis, accepted)
ll.BatchUpdate(epoch, batch, row, nis, accepted)
ll.FilterHistory(times, states, covariances)
```

| Record | Fields |
| --- | --- |
| `UpdateResult` | `innovation` is the measurement residual vector before correction, in measurement order and units. `nis` is its normalized squared magnitude. `accepted` tells whether the correction was applied. |
| `BatchUpdate` | Satkit `epoch`, zero-based source `batch` and `row` indices, NIS, and the gate decision for one reading. |
| `FilterHistory` | A tuple of distinct Satkit `times`, a `(K, D)` array of states, and `(K, D, D)` covariances for `K` epochs and state dimension `D`. |

## Correct a TLE during a pass

```text
ll.TcaUKF(tle, epoch, x0, covariance, *, options=None)
ukf.predict_to(time, q=None) -> None
ukf.observe(time, station, values, *, noise=None, q=None) -> UpdateResult
ll.TcaMeasurementNoise(sigma_rr=0.5, sigma_az=radians(2.5), sigma_el=radians(2.5))
ll.TcaCorrection(epoch_offset_seconds, range_rate_bias_m_per_s, covariance, corrected_omm)
```

Pass a TLE as a single multiline string, a Satkit filter epoch, the initial pair `x0 = [epoch_offset_seconds, range_rate_bias_m_per_s]`, and its 2×2 covariance. A positive offset moves the TLE epoch later. `options` takes a `FilterOptions` gate configuration, with gating disabled by default.

- `predict_to` advances the two-state estimate to `time`. Both methods accept an optional discretized 2×2 process covariance `q` (default zero).
- `observe` updates at `time` from an `ll.GroundStation` and a three-vector of range rate (m/s), azimuth (rad), and elevation (rad), in that order. `noise` takes a `TcaMeasurementNoise` giving independent standard deviations in those units. By default these are 0.5 m/s and 2.5° in radians for each angle. Returns an `UpdateResult`.
- `ukf.state`, `ukf.covariance`, and `ukf.epoch` expose the current estimates and time. `ukf.options` holds the gate settings. `ukf.corrected_omm` returns the corrected mean elements as a dictionary. `ukf.correction` packages the epoch offset, range-rate bias, their covariance, and that dictionary in a `TcaCorrection`.

## Plan with a receiver callback

### MPCConfig, AntennaLimits, and SearchSetup

```text
ll.MPCConfig(search_duration_s=192.0, move_s=4.0, settle_s=0.5, listen_s=1.5,
             horizon=4, beam_width=40, candidate_count=17, candidate_extent_sigma=2.8,
             ess_threshold=0.5)
ll.AntennaLimits(rate_rad_s=radians(1.10), acceleration_rad_s2=radians(1))
ll.SearchSetup(station, start, config=MPCConfig(), antenna=AntennaLimits(),
               elevation_mask_rad=radians(10), initial_direction=None)
```

| Settings | Meaning |
| --- | --- |
| `MPCConfig` timing | Total search budget and the move, settle, and listen durations in seconds. The budget must contain a whole number of these cycles. |
| `horizon`, `beam_width` | Number of cycles to optimize in the prefix and maximum number of partial plans to retain. |
| `candidate_count`, `candidate_extent_sigma`, `ess_threshold` | Positive odd number of moving pointing candidates, their extent in standard deviations of the prior, and effective sample size threshold for miss resampling as a fraction of the active particle count. |
| `AntennaLimits` | Great-circle rest-to-rest slew rate (rad/s) and acceleration (rad/s²). |
| `SearchSetup` | Low-level `GroundStation`, Satkit `start` time, planning `config`, and `antenna` limits. Supply the visibility mask in radians. `initial_direction` optionally gives the antenna's starting ENU unit three-vector above the mask. The default starts at nominal pointing. |

### BCMPC and ParticleGeometry

```text
ll.BCMPC(nominal, particles, weights, *, setup)
search.plan(probabilities) -> ActionPlan
search.observe(outcome, rng=None) -> ObservationUpdate
ll.ParticleGeometry(directions_enu, ranges_m, elevations_rad, visible)
```

Construct the controller with a nominal orbit, a nonempty sequence of particle orbits (multiline TLE strings or OMM mappings), and one finite nonnegative weight per particle with positive total mass. Pass an `ll.SearchSetup` for the station and planning parameters.

`plan` accepts a receiver callback `probabilities(command, geometry, listen_s)`. At a listening midpoint, `command` is an ENU unit three-vector and `listen_s` is the listening duration. For `P` active particles, `ParticleGeometry` supplies `(P, 3)` ENU unit directions, `(P,)` ranges in metres, elevation angles in radians, and a boolean visibility mask. Return a finite length-`P` vector of acquisition probabilities in [0, 1], with zero for invisible particles. The callback must give consistent answers within a single plan.

Execute the first action of the returned plan. Once its listening interval completes, call `observe` with the outcome enum or its string value. An optional NumPy `Generator` controls resampling after a miss. By default the controller uses its own generator. `observe` consumes the pending action and returns an `ObservationUpdate`. If execution is interrupted, the action stays pending.

| Controller property | Value |
| --- | --- |
| `search.status`, `search.finished`, `search.cycle` | Current `SearchStatus`, whether acquisition or exhaustion has ended the search, and zero-based cycle counter. |
| `search.direction_enu` | Current antenna direction as an ENU unit three-vector. |
| `search.weights`, `search.particle_indices`, `search.effective_sample_size` | Normalized weights, corresponding indices in the original particle pool, and effective sample size of the current belief. |

### Acquisition actions and outcomes

```text
ll.Action(cycle, direction_enu, listen_start, listen_end, candidate_index)
ll.ActionPlan(actions, acquisition_probability, retained_acquisition_probability)
ll.ObservationUpdate(resampled, status)
ll.Observation(value)
ll.SearchStatus(value)
```

- `Action` carries the cycle index, commanded ENU unit vector, Satkit listening start/end times, and optional candidate index. `action.azimuth_elevation` converts the command to a two-element array of angles in radians (eastward from north, above the horizon).
- `ActionPlan.actions` is the tuple of commands over the remaining cycles. `acquisition_probability` scores that plan. `retained_acquisition_probability` scores the previous continuation under the same current belief and receiver.
- `Observation` takes `"miss"` (`MISS`) or `"acquired"` (`ACQUIRED`). `ObservationUpdate.resampled` reports whether the active particle pool was resampled and `.status` gives the resulting `SearchStatus`.
- `SearchStatus` takes `"ready"` (`READY`), `"awaiting_observation"` (`AWAITING_OBSERVATION`), `"acquired"` (`ACQUIRED`), or `"exhausted"` (`EXHAUSTED`). Plan while ready and report the completed listen while awaiting observation.

For runnable workflows with synthetic data, see [fitting and smoothing](../python/examples/fit_and_fuse.py), [TLE correction](../python/examples/tca_correction.py), and [acquisition with a custom receiver](../python/examples/acquire.py).

## Errors and versions

| Name | When to use it |
| --- | --- |
| `ll.TychonicError(*args)` | Catch the base runtime exception for native Tychonic failures. |
| `ll.DataError(*args)` | Required Satkit data or epoch coverage is unavailable. The error identifies the missing data or interval. |
| `ll.NumericalError(*args)` | A numerical operation failed, such as propagation or matrix factorization. |
| `ll.SolverError(*args)` | An optimizer failed, including reaching its iteration limit. |
| `ll.AcquisitionStateError(*args)` | A search operation is invalid in its current state, such as planning while awaiting feedback. |
| `ll.__version__`, `ll.__core_version__` | Version strings for the Python adapters and the linked Rust core, respectively. |

Argument validation can also raise standard Python exceptions such as `ValueError` for inconsistent dimensions or `TypeError` for an unsupported epoch type.
