# Changelog

## 0.9.4

- Add covariance-enabled calibrated batch LM and facade APIs without changing
  existing fit functions or `FitError` variants. Return physical parameter
  covariance, numerical rank, and its statistical interpretation.
- Compute absolute whitened-information covariance from the final retained LM
  evaluation using column-normalized pivoted QR and compact R SVD. Exclude
  residual scaling and solver damping; retain Varpro Schur covariance behavior.
- Return structured rank and numerical errors instead of regularization or
  pseudo-priors. Keep covariance storage linear in observation count.
- Verify analytic correlated-noise references, units and optimizer-scale
  invariance, six/seven-parameter fits, singular geometry, arithmetic overflow,
  and noisy mixed-observable recovery. Add a reproducible downstream fixture.

## 0.9.3

Release contents include all additions since 0.9.2. See the
[feature outline](docs/release_0_9_3.md) for interfaces, integration, and validation.

- Add the public `detection` module and `BcMpc` controller for Bayesian
  Continuation MPC satellite acquisition from explicit weighted TLE/OMM hypotheses.
- Build a private feasible ATUS continuation, then optimize a moving candidate
  prefix against acquisition probability over the entire remaining search.
  Preserve the continuation boundary and retain the incumbent plan when beam
  search cannot improve its score.
- Add particle-belief normalization, log-space Bayesian miss updates, effective
  sample size, and caller-seeded systematic resampling. Cache geometry for the
  original ephemeris pool and select active hypotheses by ancestor indices.
- Add ENU pointing geometry, along-track candidates, elevation-mask checks,
  rest-to-rest antenna slew constraints, and partial slews for initialization.
- Add timed actions, acquisition/miss observations, explicit ready/pending/
  acquired/exhausted states, and a caller-supplied `DetectionModel`. Planning
  failures, unsupported misses, and invalid resampling draws preserve controller
  state; execution failures leave the action pending rather than reporting a miss.
- Add checked batch `kernels::propagate_sgp4_gcrf` for any Satkit `SGP4Source`,
  including TLE and OMM, preserving epoch order and checking every propagation
  status, output shape, frame conversion, and state. Reuse it in the existing
  TLE initializer and combine it with prepared station geometry for acquisition.
- Add a shared simulated plant and runnable `acquisition` example with an
  independent true orbit, erroneous planning orbit, Gaussian receiver model,
  separate seeded receiver/resampling streams, and per-cycle diagnostics.
  The default fixture discovers the satellite in 12 simulated seconds.
- Add acquisition regressions for kernel/reference agreement, constrained
  planning, belief/resampling behavior, state transitions and error atomicity,
  TLE/OMM geometry, reproducible simulated discovery, budget exhaustion, and
  failed-execution retry behavior. Retain the forced-feedback controller test.
- Add the acquisition API/defaults guide, README integration example, simulated
  discovery instructions, and the satellite-search paper reference.

## 0.9.2

- Replace reference counting and interior mutability in Tychonic-owned code
  with owned or borrowed geometry, observation data and problems, plus reusable
  mutable evaluation buffers. Kernels execute sequentially on the caller's thread.
- Add a local fallible LM driver for ordinary, RF and variable-projection fits,
  preserving damping and convergence equations while evaluating residuals and
  Jacobians together and rejecting negative-drag trials before propagation.
- Validate sensitivity arrays, RF data, priors, epochs and integration controls
  before use. Return numerical and solver errors without panic extraction.
- Store filter history with an always-present current record; restore small
  checkpoints on failed observations and materialize RTS records explicitly.
- Retain owned covariance factors and share one factor across independent joint
  rows. Remove whole-estimator and whole-evaluation cloning.
- Forbid unsafe library code and enforce ownership/error policies with Clippy,
  a source check and CI. Pin and document the Numeris/Satkit trust boundary and
  narrowly scoped external settings/TLE copies.
- Intentionally break ownership, validated-input and history APIs; update
  examples, tests and tools. See [migration](docs/migration_0_9_2.md),
  [safety contracts](docs/safety.md) and [qualification](docs/qualification_0_9_2.md).

- Add a two-state TCA UKF for TLE epoch correction and a constant pass
  range-rate bias, using combined range-rate and soft azimuth/elevation readings.
- Default TCA pointing noise to independent 2.5-degree standard deviations;
  retain observation-time station geometry and invalidate shifted SGP4 caches.

## 0.9.0

- Add a calibrated facade for standalone Doppler Hz, continuous phase difference,
  exact differential Doppler, and azimuth/elevation.
- Fuse any selection with full per-epoch covariance in batch fits and EKF
  updates, retaining shared ordinary/differential Doppler noise.
- Add fit convenience functions, named orbit filter aliases, and an RF builder
  with named priors for existing offset, drift and baseline-calibration models.
- Preserve analytic Jacobians, prepared geometry, propagation, variable
  projection, augmented RF estimators and RTS smoothing.
- Preserve existing public functions; extend `MeasurementKind` with `Joint`.
  Downstream exhaustive matches must handle the new variant.
- Move research/profiling executables and plotting scripts to the development
  tools crate. Replace the default executable with the `batch_fit` example.
- Keep generated experiments and reference PDFs locally, outside Git tracking
  and release packages. Add MIT/Apache-2.0 licensing and release metadata.
- Clarify that fitting requires an initial orbit guess; this release does not
  add orbit initialization without a guess or receiver signal extraction.

Earlier experimental implementations are not assigned release versions here.
