# Python orbit estimation

| To look up | Reference |
| --- | --- |
| Observation columns, solver profiles, spacecraft catalogues, exports, or satellite acquisition | [`tychonic` API reference](python-api.md) |
| Measurement factories, propagation settings, individual filters, or a custom receiver callback | [`tychonic.low_level` API reference](python-low-level.md) |

## Preparing observations

Create one `JobData` per spacecraft UUID. Group simultaneous columns by contact,
antenna, carrier and available observables. `ContactSamples.from_unix_seconds`
accepts numeric UTC POSIX seconds and preserves a numeric column for native
ingestion. Text inputs use UTC RFC 3339 at microsecond resolution. POSIX omits
leap seconds. Supply stable string observation IDs for streamed updates.

`add_doppler(samples, hz, sigma_hz=...)` takes signed downlink frequency shift
in Hz, negative for recession. The contact carries the positive carrier in Hz.
`add_joint(ObservationBatch(...))` takes shared or per-epoch absolute covariance.
Component order is Doppler Hz, phase difference rad, differential Doppler Hz,
azimuth rad and elevation rad, skipping absent components. Optical angle pairs
can omit the carrier. Each interferometric baseline is antenna 1 minus antenna 2
in antenna 1's ENU frame in metres. For inertial baseline input, rotate at each
observation epoch. Absolute ranging has no channel in this release.

## Fitting and publication

Use `StateEstimate.from_tle(tle, at=..., uncertainty=TleAgeUncertainty(...))` for
an SGP4/GCRF seed. Supply six positive epoch standard deviations and six
nonnegative daily growth rates in RTN position and velocity order. The seed
records the input elements and version 1 recipe. Coefficients are caller calibrated.
Cartesian input accepts metres and metres per second.

- `CalibratedLM()` requests absolute covariance. Only `weighted_prior=` adds
  independent information. A covariance-bearing `initial_guess=` has zero
  statistical weight, including when reprocessing observations. Rank failures
  expose `RankDeficientError.rank` and `.parameters`.
- `ContactBiasFit` projects out a bias per contact, with residual-scaled Schur
  covariance. The native solver accepts a common range-rate sigma and constant,
  independent angular sigmas. Conversion to range rate is internal. Each contact
  uses one carrier and the returned biases are in Hz.
- `PassiveRFFit` takes a diagonal orbit prior, RF offset priors and an OU phase
  drift model. Batches share a carrier, channels and measurement covariance.
  Supported combinations start with Doppler, optionally add phase, and can then
  add differential Doppler. The three-channel model shares antenna-1 noise
  between Doppler and differential Doppler. Returned drift follows input row order.
- `ConstrainedFit` fits seven orbital parameters and contact biases using hard
  pointing cones. It accepts per-row rate uncertainty. Its result includes the
  worst retained cone violation and has no covariance.

`solve(output_epoch=...)` returns the orbit at the requested epoch. The full
variational transition transports covariance, including drag and nuisance cross
terms. `last_contributing_epoch` is the newest measurement used in the objective
or retained constraints. Publish `estimate.orbital_marginal()` for the Cartesian
six-state and 6×6 covariance. Converting the entire SI state to km and km/s
scales its covariance by 1e-6.

## Catalogues, history and exports

`EnsembleFilter.ekf(...)` stores independent orbit and additive Hz-bias estimates,
with drag held fixed per spacecraft. A new bias needs its own Gaussian prior.
`EnsembleFilter.tle_epoch_ukf(...)` uses a persistent two-state epoch-offset and
Hz-bias estimate with joint Doppler/azimuth/elevation updates. Its bounded process
noise caps marginal variance growth. Replacing the TLE resets the offset prior
and its cross-covariance after prediction to the handoff epoch.

Store the versioned checkpoint envelope and opaque bytes together. Restore into
a compatible catalogue with an unoccupied spacecraft ID. A gated reading advances
the prediction and processed cursor. An accepted correction also advances the
accepted cursor and evidence epoch. On a numerical update failure, successful
earlier readings remain committed and the exception carries the processed cursor.
Retained EKF history supports native RTS smoothing, with readings and gate
outcomes linked to each epoch. Disabling history keeps one forward record per object.

`to_oem` samples the Cartesian arc in GCRF and writes km and km/s.
`to_tle` and `to_omm` fit SGP4 mean elements over a user-selected arc spanning
the estimate epoch. The fit can fail to converge on a short or weakly informative
arc. Use `JobData.export_to_ccsds_tdm` for one-way Doppler KVN and `to_polars()`
for a table retaining telemetry and joint covariances.

`SatFinder.from_tle` samples phase, inclination, RAAN, and mean-motion offsets
using the supplied NumPy generator. `from_hypotheses` accepts explicit weighted
TLE/OMM hypotheses. Call `plan()`, execute the first action and pass the completed
listening outcome to `observe(acquired=...)`. `read_state()` supplies the updated
weights and original ancestor indices.

## Local builds

The private extension depends on the root `tychonic` crate and has its own
Cargo workspace and lockfile in `python/`. From the
repository root, run `maturin build --release --locked` or `maturin sdist`.
Install the wheel into a fresh environment and run `pytest -q python/tests`
with both optional extras installed. Python release tags use `python-v<VERSION>`.
