# Python API reference · `tychonic`

Use `import tychonic as tc` for the names below. Methods are shown on an instance unless prefixed with a class name. A bare `*` introduces keyword-only arguments. Constructor arguments are also the field names on the data objects. For installation and a walkthrough, see the [Python guide](python.md). For the adapters that take `satkit.time`, see the [low-level reference](python-low-level.md).

| Task | Look up |
| --- | --- |
| [Prepare observations](#prepare-observations) | `GroundStation`, `ContactSamples`, `ObservationBatch`, `JobData`, Polars and TDM exports |
| [Initialize or publish an orbit](#initialize-or-publish-an-orbit) | `ForceModel`, `GaussianPrior`, `TleAgeUncertainty`, `TleInitialization`, `StateEstimate`, `OrbitalMarginal`, `TleCorrection` |
| [Fit observations](#fit-observations) | `ODProblem`, `OptimizerProfile`, `CalibratedLM`, `ContactBiasFit`, `PassiveRFFit`, `ConstrainedFit`, `FitResult` |
| [Update a spacecraft catalogue](#update-a-spacecraft-catalogue) | `EnsembleFilter`, `OrbitEKFCatalogue`, `TleEpochUKFCatalogue`, `OrbitProcessNoise`, `TleProcessNoise` |
| [Read updates and resume processing](#read-updates-and-resume-processing) | `ReadingRef`, `FilterUpdate`, `FilterCheckpoint`, `SmoothedStep` |
| [Plan an acquisition](#plan-an-acquisition) | `SatFinder`, `OrbitUncertainty`, `AcquisitionReceiver`, `SearchSetup`, `SearchAction`, `SearchPlan`, `SearchState` |
| [Interpret types, errors, and versions](#types-errors-and-versions) | `Epoch`, `TLE`, `OMM`, `ParticleSampler`, `FitError`, `CovarianceError`, `RankDeficientError`, `__version__`, `__core_version__` |

## Prepare observations

### GroundStation and ContactSamples

```text
tc.GroundStation(antenna_id, latitude_deg, longitude_deg, altitude_m)
tc.ContactSamples(contact_id, station, epochs, carrier_hz=None, lock_state=None,
                  ebn0_db=None, observation_ids=None)
tc.ContactSamples.from_unix_seconds(contact_id, station, epoch_unix_s, *,
                                    carrier_hz=None, observation_ids=None,
                                    lock_state=None, ebn0_db=None) -> ContactSamples
```

A station identifies an antenna by string ID and WGS84 geodetic latitude, longitude (degrees), and altitude (metres). Group simultaneous readings from one contact UUID and station in a `ContactSamples`. Pass a sequence of UTC RFC 3339 strings in `epochs`, or use the factory with a finite, one-dimensional numeric column of POSIX seconds. The factory retains the numeric column for ingestion and exposes UTC labels through `.epochs` at microsecond resolution. Use UTC labels for leap seconds.

`carrier_hz` is the positive downlink carrier for RF measurements. Optical pointing alone can omit it. Optional columns each have one entry per epoch. `lock_state` records receiver lock (boolean), `ebn0_db` records measured Eb/N0 in dB, and `observation_ids` holds source-stable strings. Catalogue updates require these IDs. Give distinct readings at the same epoch distinct IDs within a contact and antenna.

### ObservationBatch

```text
tc.ObservationBatch(*, samples, covariance, doppler_hz=None, phase_difference_rad=None,
                    differential_doppler_hz=None, azimuth_rad=None, elevation_rad=None,
                    phase_baseline_enu_m=None, differential_baseline_enu_m=None)
```

Use one batch for simultaneous channels that share a `ContactSamples`. Each supplied measurement column has shape `(N,)` for `N` epochs. The covariance axes follow this order, skipping absent columns.

| Column | Measurement |
| --- | --- |
| `doppler_hz` | Signed downlink shift relative to the carrier, negative for recession, in Hz |
| `phase_difference_rad` | Continuous, unwrapped antenna-1 minus antenna-2 phase difference in radians |
| `differential_doppler_hz` | Antenna-1 minus antenna-2 Doppler shift in Hz |
| `azimuth_rad`, `elevation_rad` | Paired pointing angles in radians, with azimuth eastward from north and elevation above the horizon |

Supply at least one epoch and one channel. For `M` measurement components, `covariance` is the absolute measurement covariance in the corresponding units, shape `(M, M)` for all rows or `(N, M, M)` for individual rows. Each interferometric channel requires its own baseline argument. Both use antenna 1 minus antenna 2 in antenna 1's local east, north, up (ENU) coordinates in metres, shape `(3,)` or `(N, 3)`.

### JobData

```text
tc.JobData(spacecraft_id, spacecraft_name=None)
job.add_joint(batch) -> None
job.add_doppler(samples, doppler_hz, *, sigma_hz) -> None
job.add_phase_difference(samples, phase_rad, *, baseline_enu_m, sigma_rad) -> None
job.add_differential_doppler(samples, differential_hz, *, baseline_enu_m, sigma_hz) -> None
```

Create one job per spacecraft UUID. `spacecraft_name` is an optional display name. Each `add_*` call appends to the job's initially empty `batches` list. Pass an `ObservationBatch` to `add_joint` to preserve joint uncertainty. The three scalar-channel helpers take a `ContactSamples`, a measurement vector in the units above, and a positive finite standard deviation, either a scalar or one per row. They square the sigma to build the covariance. For phase and differential Doppler, `baseline_enu_m` has the same meaning and shapes as the corresponding batch baseline.

For example, two Doppler readings from one contact can share the same uncertainty.

```python
from uuid import UUID

import numpy as np
import tychonic as tc

station = tc.GroundStation("antenna-1", 40.0, -75.0, 100.0)
samples = tc.ContactSamples(
    UUID(int=1), station,
    ["2024-01-01T12:00:00Z", "2024-01-01T12:00:10Z"],
    carrier_hz=2.23e9, observation_ids=["read-1", "read-2"],
)
job = tc.JobData(UUID(int=7))
job.add_doppler(samples, np.array([-910.0, -920.0]), sigma_hz=0.5)
```

| Export method | Arguments and result |
| --- | --- |
| `job.to_polars()` | Returns a Polars DataFrame with a row per batch and epoch, including IDs, telemetry, measurements, baselines, and covariance. Install the `polars` extra. |
| `job.export_to_ccsds_tdm(directory, *, originator, creation_epoch, spacecraft_participant, antenna_participants)` | Writes CCSDS TDM 2.0 KVN files under `directory` and returns a tuple of their paths. Supply the issuing organization, UTC creation time, spacecraft participant name, and a mapping from antenna ID to ground participant name. Each contact/antenna pair with Doppler produces one file of one-way range rates in km/s. An existing destination raises `FileExistsError`. Install the `tdm` extra. |

## Initialize or publish an orbit

### ForceModel

```text
tc.ForceModel(*, gravity_model="EGM2008", gravity_degree=4, gravity_order=4,
              sun_gravity=True, moon_gravity=True, use_space_weather=True,
              fixed_drag_m2_kg=0.0, integrator=None, abs_tolerance=1e-10,
              rel_tolerance=1e-13)
```

Pass a force model to a fit, EKF factory, or Cartesian arc export. `gravity_model` selects the geopotential and `gravity_degree` and `gravity_order` truncate its harmonics. The next three switches control solar gravity, lunar gravity, and space-weather inputs to the atmosphere. `fixed_drag_m2_kg` is the drag coefficient times area divided by mass. `integrator` selects a Satkit integration method (`None` uses the native OD default). `abs_tolerance` and `rel_tolerance` control integration error. For model and integrator names, see [PropagationSettings](python-low-level.md#propagationsettings-and-propagationconfig).

### Priors and TLE initialization

```text
tc.GaussianPrior(mean, sigma)
tc.TleAgeUncertainty(*, sigma_at_epoch, sigma_growth_per_day)
tc.TleInitialization(reference_tle, uncertainty)
```

- `GaussianPrior` supplies a mean and one standard deviation in the receiving argument's units, such as Hz for a Doppler bias or seconds for a TLE epoch offset.
- `TleAgeUncertainty` describes independent radial, along-track, and cross-track errors. Each vector has six entries, position (m) then velocity (m/s) in that axis order. Provide positive epoch sigmas and nonnegative growth in those units per day. At age `d` days each sigma is `sigma_at_epoch + abs(d) * sigma_growth_per_day`. The recipe exposes `name="rtn_linear_sigma"` and `version=1`.
- `TleInitialization` records the original two-line `reference_tle` and its `TleAgeUncertainty` recipe. Read this record from a TLE-seeded estimate to recover its inputs.

### StateEstimate and OrbitalMarginal

```text
tc.StateEstimate(epoch, position_m, velocity_m_s, *, covariance=None, drag_m2_kg=None,
                 doppler_bias_hz=None, tle_initialization=None)
tc.StateEstimate.from_tle(tle, *, at, uncertainty) -> StateEstimate
tc.OrbitalMarginal(epoch, state_si, covariance_si)
```

Construct a Cartesian estimate from its UTC epoch and GCRF position and velocity three-vectors in metres and m/s. Optional `drag_m2_kg` and `doppler_bias_hz` add estimated drag and additive Doppler bias. `estimate.covariance_order` lists the axes in order, starting with Cartesian position and velocity, followed by drag and bias when supplied. `covariance` is square in that order. `None` represents a point estimate. `tle_initialization` optionally records provenance.

Use `from_tle` with two TLE lines, evaluation epoch `at`, and a `TleAgeUncertainty` to obtain an SGP4/GCRF state. The factory rotates the six-axis uncertainty into GCRF and attaches the initialization record.

| Method | Arguments and result |
| --- | --- |
| `estimate.orbital_marginal()` | Returns an `OrbitalMarginal` at the same `epoch`. Its `state_si` is a six-vector and `covariance_si` is a copy of the 6×6 Cartesian block. Raises `CovarianceError` for a point estimate. |
| `estimate.to_tle(epochs, *, force_model)` | Propagates with the given force model at a sequence of UTC epochs spanning the estimate epoch, fits SGP4 mean elements to the arc, and returns two TLE lines. A short arc can fail to constrain the fit. |
| `estimate.to_omm(epochs, *, force_model)` | Fits the same arc and returns its mean elements as an OMM mapping. |
| `estimate.to_oem(epochs, *, force_model, object_name, object_id, originator)` | Returns CCSDS OEM text with Cartesian GCRF samples in km and km/s. Supply strictly increasing UTC epochs, the propagation model, object name and identifier, and issuing organization. |

### TleCorrection

```text
tc.TleCorrection(reference_tle, epoch, *, epoch_offset_s, doppler_bias_hz, covariance)
correction.to_tle() -> TLE
correction.to_omm() -> OMM
```

The TLE catalogue uses this two-state estimate at observation time `epoch`. A positive `epoch_offset_s` moves the element epoch later. `doppler_bias_hz` is the additive receiver bias. Supply the 2×2 covariance in offset/bias order. Both export methods shift the epoch of `reference_tle` and retain its other mean elements. Use `to_tle()` for two text lines or `to_omm()` for a mapping with the internal epoch precision. The receiver bias stays on the correction object.

## Fit observations

### ODProblem

```text
tc.ODProblem(data, *, initial_guess, force_model, profile, weighted_prior=None)
problem.solve(*, output_epoch=None) -> FitResult
```

Pass a spacecraft's `JobData`, a starting `StateEstimate`, a `ForceModel`, and one of the four profiles below. The optimizer starts at `initial_guess`. Only `weighted_prior` contributes independent Gaussian information. Supply it at the guess epoch, with positive-definite covariance over the exact fitted orbit axes (including drag when fitted). When reprocessing data already represented in a seed, pass the seed as the initial guess alone.

`solve` returns the estimate at `output_epoch` (UTC, default `initial_guess.epoch`), deterministically propagating the fitted covariance and cross terms when available. The latest contributing observation time is recorded separately.

### Solver profiles

```text
tc.OptimizerProfile(*, max_iterations=None)
tc.CalibratedLM(*, max_iterations=None, fit_drag=False, covariance=True,
                gradient_tolerance=None, step_tolerance=None, cost_tolerance=None)
tc.ContactBiasFit(*, max_iterations=None, fit_drag=False)
tc.PassiveRFFit(*, max_iterations=None, frequency_offset_hz, phase_offset_rad,
                drift_correlation_s, drift_sigma_rad, fit_drag=False,
                phase_baseline_sigma_m=None, differential_baseline_sigma_m=None)
tc.ConstrainedFit(*, max_iterations=None, full_fov_rad, pointings_per_contact=3)
```

`OptimizerProfile` is the settings base. Select a concrete profile for `ODProblem`. Each accepts an iteration limit (`None` uses the solver default). Where present, `fit_drag` adds drag to the fitted parameters. Its initial value comes from the guess or the force model's fixed drag.

| Profile | Use and arguments |
| --- | --- |
| `CalibratedLM` | Fit measurements with full calibrated covariance. `covariance=True` requests absolute state covariance, or posterior covariance with a weighted prior. Set it to `False` for a point fit. Optional gradient, parameter-step, and cost tolerances override native stopping criteria. |
| `ContactBiasFit` | Fit the orbit and one Doppler bias per contact, optionally with soft pointing. Use a single carrier per contact, a common sigma after conversion to range rate, and independent, constant azimuth and elevation sigmas. Returns residual-scaled Schur covariance. |
| `PassiveRFFit` | Fit Doppler with optional continuous phase, and optionally differential Doppler alongside phase. Supply a diagonal weighted orbit prior, Gaussian priors on frequency (Hz) and phase (rad) offsets, and the correlation time (s) and stationary sigma (rad) of the phase drift. Optional baseline sigmas are three ENU standard deviations in metres for fitting the corresponding baseline corrections. Batches share the carrier, channels, and measurement covariance. In the three-channel case Doppler and differential Doppler share the antenna-1 noise. |
| `ConstrainedFit` | Fit orbit, drag, and contact biases under hard pointing cones. `full_fov_rad` is the full angular width. `pointings_per_contact` selects a count near closest approach or `"all"`. Each Doppler row can have its own sigma. The result is a point estimate. |

### FitResult

```text
tc.FitResult(*, estimate, last_contributing_epoch, used_weighted_prior, whitened_rss,
             iterations, covariance_method=None, rank=None, contact_bias_hz={},
             rf_parameters={}, rf_covariance=None, phase_drift_rad=None,
             max_pointing_violation_rad=None)
```

Read these fields on the result of `solve()`. The two empty mappings shown above are created separately for each result.

| Field or property | Meaning |
| --- | --- |
| `estimate`, `output_epoch` | Fitted `StateEstimate` and its UTC epoch |
| `last_contributing_epoch` | Latest observation used in the objective or in a retained constraint |
| `used_weighted_prior`, `whitened_rss`, `iterations` | Whether an orbit prior contributed, weighted residual sum of squares, and solver iteration count |
| `covariance_method`, `rank` | Covariance method label and fitted system rank when reported |
| `contact_bias_hz` | Fitted Doppler biases keyed by contact UUID |
| `rf_parameters`, `rf_covariance`, `phase_drift_rad` | Full passive RF fit keyed by parameter name, covariance in that key order, and phase drift in input row order |
| `max_pointing_violation_rad` | Worst retained pointing-cone violation in a constrained fit, in radians |

## Update a spacecraft catalogue

### EnsembleFilter

```text
tc.EnsembleFilter.ekf(*, force_model, noise=OrbitProcessNoise(), gate=None,
                      retain_history=False) -> OrbitEKFCatalogue
tc.EnsembleFilter.tle_epoch_ukf(*, noise, gate=None) -> TleEpochUKFCatalogue
tc.OrbitProcessNoise(*, acceleration_psd_m2_s3=0.0, bias_psd_hz2_s=0.0)
tc.TleProcessNoise(*, offset_psd_s2_s, bias_psd_hz2_s, max_offset_sigma_s, max_bias_sigma_hz)
```

Use the factories on the abstract `EnsembleFilter` to create an empty catalogue. `ekf` takes a `ForceModel` and tracks a Cartesian orbit plus a Doppler bias per spacecraft. `OrbitProcessNoise` gives the common white-acceleration power spectral density per axis (m²/s³) and the independent bias random-walk rate (Hz²/s). Set `retain_history=True` to enable smoothing.

`tle_epoch_ukf` tracks an epoch offset and Doppler bias against a reference TLE. `TleProcessNoise` gives their random-walk rates in s²/s and Hz²/s and positive caps on their marginal standard deviations in s and Hz. For both factories, `gate` is a nonnegative normalized innovation squared (NIS) threshold. `None` disables gating.

| Shared method | Arguments and effect |
| --- | --- |
| `catalogue.add_spacecraft(spacecraft_id, initial)` | Register a new UUID with its seed. The concrete catalogues below specify the seed type and EKF-specific keywords. |
| `catalogue.drop_spacecraft(spacecraft_id)` | Remove the entry for this UUID. Returns `None`. |
| `catalogue.update(data)` | Apply one `JobData` to its spacecraft in stable epoch order and return a tuple of `FilterUpdate` records. Streamed rows require observation IDs. A rejected row retains its prediction. On a numerical row failure, earlier committed rows remain applied and the exception exposes `last_processed_reading`. |
| `catalogue.predict_to(epoch, *, ids=None)` | Advance the selected sequence of UUIDs, or every spacecraft by default, to the supplied UTC epoch. Returns `None`. |
| `catalogue.get_state(spacecraft_id)` | Return an independent current snapshot for this UUID, a `StateEstimate` for the EKF or `TleCorrection` for the UKF. |
| `catalogue.checkpoint(spacecraft_id)` | Return a `FilterCheckpoint` for this UUID including reading cursors and any retained history. |
| `catalogue.restore(checkpoint)` | Insert the saved entry into a catalogue with matching kind and settings. Requires a supported checkpoint version and a free UUID. Returns `None`. |

### OrbitEKFCatalogue

```text
tc.OrbitEKFCatalogue(*, force_model, noise, gate=None, retain_history=False)
catalogue.add_spacecraft(spacecraft_id, initial, *, bias_prior_hz=None,
                         last_contributing_epoch=None) -> None
catalogue.smooth(spacecraft_id) -> tuple[SmoothedStep, ...]
```

Direct construction takes the same settings as `EnsembleFilter.ekf`, with an explicit `OrbitProcessNoise`. Seed the UUID with a covariance-bearing `StateEstimate`. When the seed has a bias, its cross-covariances are retained. Otherwise supply an independent `GaussianPrior` in Hz via `bias_prior_hz`. A fitted drag value becomes that object's fixed propagation drag. Pass the fit's `last_contributing_epoch` to carry forward its evidence time. With retained history, `smooth` returns chronological RTS-smoothed (Rauch–Tung–Striebel) estimates for the UUID, including the prior epoch. See also the [shared catalogue methods](#ensemblefilter).

For a batch-to-live handoff, take `history` and `next_contact` from the same spacecraft and choose an `output_epoch` between the historical and incoming readings. Here `seed` is a Cartesian estimate, `forces` is a force model, and `profile` is a calibrated LM profile with covariance enabled.

```python
import tychonic as tc

def fit_and_update(history, next_contact, seed, forces, profile, output_epoch):
    problem = tc.ODProblem(history, initial_guess=seed, force_model=forces, profile=profile)
    fit = problem.solve(output_epoch=output_epoch)
    catalogue = tc.EnsembleFilter.ekf(force_model=forces)
    catalogue.add_spacecraft(
        history.spacecraft_id, fit.estimate, bias_prior_hz=tc.GaussianPrior(0.0, 5.0),
        last_contributing_epoch=fit.last_contributing_epoch,
    )
    updates = catalogue.update(next_contact)
    return updates, catalogue.get_state(history.spacecraft_id)
```

### TleEpochUKFCatalogue

```text
tc.TleEpochUKFCatalogue(*, noise, gate=None)
catalogue.add_spacecraft(spacecraft_id, initial) -> None
catalogue.replace_tle(spacecraft_id, tle, *, epoch, offset_prior_s) -> None
```

Direct construction takes the same settings as `EnsembleFilter.tle_epoch_ukf`. Seed the UUID with a `TleCorrection` whose initial offset and bias uncertainties fit within the configured caps. Updates take joint Doppler, azimuth, and elevation batches. To switch reference elements, pass the new two-line `tle`, the handoff `epoch`, and a `GaussianPrior` for the new offset in seconds. The handoff predicts to that epoch, keeps the predicted bias and its variance, and resets offset/bias cross-covariance to zero. See the [shared catalogue methods](#ensemblefilter).

## Read updates and resume processing

```text
tc.ReadingRef(contact_id, antenna_id, observation_id, epoch)
tc.FilterUpdate(batch_index, row_index, reading, accepted, normalized_innovation_squared)
tc.FilterCheckpoint(*, version, kind, spacecraft_id, estimate_epoch, last_contributing_epoch,
                    last_processed_reading, last_accepted_observation, payload)
tc.SmoothedStep(estimate, readings)
```

| Record | How to read it |
| --- | --- |
| `ReadingRef` | Source contact UUID, antenna and observation ID strings, and UTC reading time. This identifies a row even when two observations share an epoch. |
| `FilterUpdate` | Zero-based batch and row indices in the supplied job, the source `ReadingRef`, whether the correction was accepted, and its NIS statistic. |
| `FilterCheckpoint` | Persist the entire record, including opaque `payload` bytes. `version` is 1 and `kind` is `"orbit_ekf"` or `"tle_epoch_ukf"`. `spacecraft_id` identifies the entry. `estimate_epoch` includes prediction through gaps. `last_contributing_epoch` tracks the latest evidence. `last_processed_reading` includes gated rows and `last_accepted_observation` tracks accepted corrections. These last three fields may be `None`. |
| `SmoothedStep` | Augmented `StateEstimate` paired with a tuple of forward-pass `FilterUpdate` records at that epoch. A prediction-only epoch has an empty tuple. |

## Plan an acquisition

### OrbitUncertainty, AcquisitionReceiver, and SearchSetup

Angle defaults below use `radians(degrees)` to show their values in radians.

```text
tc.OrbitUncertainty(*, phase_sigma_s=20.0, inclination_sigma_deg=0.025,
                    raan_sigma_deg=0.035, mean_motion_sigma_rev_day=2.5e-4)
tc.AcquisitionReceiver(*, beam_fwhm_rad=radians(3), cutoff_rad=radians(3),
                       on_axis_ebn0_db_at_1000_km=20.0, acquire_ebn0_db=5.0,
                       acquisition_s=0.3)
tc.SearchSetup(station, start, *, receiver=AcquisitionReceiver(), search_duration_s=192.0,
               move_s=4.0, settle_s=0.5, listen_s=1.5, rate_rad_s=radians(1.10),
               acceleration_rad_s2=radians(1), elevation_mask_rad=radians(10),
               initial_az_el_rad=None, horizon=4, beam_width=40, candidate_count=17,
               candidate_extent_sigma=2.8, resample_ess_fraction=0.5)
```

- `OrbitUncertainty` sets independent Gaussian perturbations to orbital phase (seconds along the orbit, applied to mean anomaly), inclination (degrees), right ascension of the ascending node (degrees), and mean motion (revolutions/day).
- `AcquisitionReceiver` sets the full width at half maximum of the power beam, maximum off-axis angle, on-axis Eb/N0 at 1000 km, minimum Eb/N0 to acquire, and required acquisition dwell time in seconds.
- `SearchSetup` takes a `GroundStation`, UTC start epoch, and receiver. The search budget must contain whole move/settle/listen cycles. Rate and acceleration limit antenna slews in rad/s and rad/s². The elevation mask is in radians. To start away from nominal, give the antenna's initial (azimuth, elevation) pair in radians above the mask.
- `horizon` is the number of cycles optimized in the plan prefix. `beam_width` bounds the number of partial plans retained. `candidate_count` is a positive odd number of pointing candidates and `candidate_extent_sigma` sets their extent in prior standard deviations. A miss can trigger resampling when effective sample size falls below `resample_ess_fraction` times the particle count.

### SatFinder

```text
tc.SatFinder.from_tle(tle, *, setup, n_particles=160, distribution=OrbitUncertainty(),
                      rng=None) -> SatFinder
tc.SatFinder.from_hypotheses(nominal, hypotheses, weights, *, setup, rng=None) -> SatFinder
tc.SatFinder(*, nominal, setup, rng, n_particles, distribution=None, hypotheses=(), weights=None)
finder.plan() -> SearchPlan
finder.observe(*, acquired) -> None
finder.read_state() -> SearchState
```

Use `from_tle` to draw `n_particles` equally weighted orbits around the input TLE with an `OrbitUncertainty` or custom `ParticleSampler`. For an explicit pool, call `from_hypotheses` with a nominal TLE or OMM, a nonempty sequence of hypotheses in either format, and a vector of finite nonnegative weights with positive total mass. Both factories take a `SearchSetup` and an optional NumPy `Generator` to control sampling and resampling. For direct construction, supply the generator and count, plus either a distribution or a matching pool and weights.

Call `plan()` when ready, then execute the first timed action. After completing its listening interval call `observe(acquired=True)` for confirmed acquisition or `observe(acquired=False)` for a miss. If execution is interrupted, the action stays pending. Read the status and particle weights with `read_state()` before planning another cycle.

### SearchAction, SearchPlan, and SearchState

```text
tc.SearchAction(azimuth_rad, elevation_rad, listen_start, listen_end)
tc.SearchPlan(actions, acquisition_probability)
tc.SearchState(status, cycle, weights, ancestor_indices)
```

A `SearchAction` specifies the commanded azimuth and elevation in radians and the UTC start and end of listening. A `SearchPlan` holds a tuple of actions and the predicted acquisition probability over the remaining plan. `SearchState.status` is `"ready"`, `"awaiting_observation"`, `"acquired"`, or `"exhausted"`. `cycle` is the zero-based cycle counter. The normalized weight at index `i` belongs to the original hypothesis numbered `ancestor_indices[i]`, including after resampling.

## Types, errors, and versions

| Name | Meaning |
| --- | --- |
| `Epoch` | UTC RFC 3339 string, e.g. `"2024-01-01T12:00:00.250000Z"`. |
| `TLE`, `OMM` | A two-string tuple (element lines 1 and 2), or an Orbit Mean-Elements Message mapping with string keys and string or numeric values, respectively. |
| `ParticleSampler` | Callable `sampler(rng, count)` returning a finite NumPy array of shape `(count, 4)`. Columns are phase offset (s), inclination offset (deg), RAAN offset (deg), and mean-motion offset (rev/day). |
| `FitError(*args)` | Runtime error for a solver failure during a fit. Catch this base class to include covariance failures. |
| `CovarianceError(*args)` | A `FitError` raised when the requested covariance is unavailable. |
| `RankDeficientError(rank, parameters)` | A `CovarianceError` carrying the rank of the whitened system and number of fitted parameters. Inspect `.rank` and `.parameters` on the exception. |
| `tc.__version__`, `tc.__core_version__` | Version strings for the Python package and the linked Rust core, respectively. |
