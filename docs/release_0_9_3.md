# Intended features for 0.9.3

Version 0.9.3 is intended to add satellite acquisition to Tychonic's existing
orbit-determination tools. This outline covers all additions in the current
source tree since the 0.9.2 baseline, including the simulated discovery work.
The release remains unreleased; the package manifest still identifies 0.9.2.

The comparison baseline is commit
[`dc8cece`](https://github.com/Tycho887/tychonic/commit/dc8cece), which contains
the 0.9.2 manifest, changelog, migration guide, and qualification report.
There is no local `v0.9.2` tag. TCA UKF and the ownership/error-handling overhaul
are already part of that baseline and remain documented under 0.9.2.

## Acquisition controller and public interfaces

- **BC-MPC controller:** the new `tychonic::detection` module exposes `BcMpc`
  to plan acquisition from a nominal orbit, explicit hypotheses, a station, and
  a caller's communications model. A private ATUS sweep builds the initial
  feasible continuation; subsequent plans optimize a prefix against probability
  of acquisition over the full remaining contact.
- **Orbit hypotheses and belief:** `Ephemeris` accepts TLE and OMM, including
  mixed ensembles. `ParticleBelief` normalizes explicit weights, applies Bayesian
  miss likelihoods in log space, computes effective sample size, and optionally
  performs systematic resampling. Active hypotheses use indices into the original
  pool; cached geometry and candidate spread remain tied to that pool.
- **Constrained planning:** moving along-track candidates use nominal phase
  geometry and the prior's angular spread. Beam search supports holding and
  retained commands, preserves the fixed continuation beyond its horizon, checks
  the transition into that continuation, and retains the incumbent score.
  Standalone candidate construction and schedule-scoring kernels are also exposed.
- **Antenna geometry:** `EnuDirection`, `PointingGeometry`, and `AntennaLimits`
  provide validated ENU pointings, angular separation, visibility, range,
  rest-to-rest slew time, reachability, and partial slews. Commands respect the
  elevation mask and configured move duration.
- **Timed execution and feedback:** `SearchSetup` and `MpcConfig` specify the
  station, start, initial pointing, move/settle/listen timing, search budget,
  horizon, beam width, candidates, and resampling threshold. `Action` and
  `ActionPlan` describe timed pointings and predicted acquisition probabilities;
  `Observation`, `ObservationUpdate`, and `SearchStatus` represent misses,
  confirmation, resampling, and ready/pending/acquired/exhausted states.
- **Receiver and error boundaries:** `DetectionModel` evaluates per-particle
  acquisition probabilities. The caller executes only the first action, supplies
  receiver feedback, and owns resampling randomness. `AcquisitionError` reports
  invalid inputs/states, propagation/model failures, and impossible observations.
  Failed planning or observation updates preserve controller state; unsuccessful
  execution leaves an action pending for retry.

The defaults are a 192-second budget with 32 cycles, 4/0.5/1.5-second
move/settle/listen intervals, a four-cycle horizon, beam width 40, 17 candidates
spanning ±2.8 prior standard deviations, and a 0.5 ESS threshold. The antenna
defaults to 1.10°/s rate and 1.00°/s² acceleration with a 10° elevation mask.
Public acquisition angles are radians, distances metres, and durations seconds;
ground-station latitude/longitude fields remain degrees.

See the [acquisition guide](acquisition.md) for complete integration contracts.

## Shared propagation and geometry

- Add `kernels::propagate_sgp4_gcrf`, accepting any Satkit `SGP4Source` for batch
  GCRF position/velocity in caller epoch order, including repeated epochs and
  empty schedules. Check epochs, output dimensions, each SGP4 status, frame
  conversion, and finite state values.
- Route `initial_state_from_tle` through that checked batch kernel.
- Add `detection::propagate_pointing_grid` to reuse `PreparedGeometry` station
  schedules, and `pointing_geometry` for externally propagated GCRF positions.
  Geometry follows the existing approximate station-rotation convention and is
  evaluated at each listening midpoint.

These additions retain the existing batch-fitting, measurement, EKF, RF,
smoothing, and TCA interfaces. Numeris and Satkit dependency versions are unchanged.

## Simulated discovery and regression coverage

The [acquisition example](../examples/acquisition.rs) and integration tests share
a simulated plant under `examples/support/`. Its independent true ISS orbit is
hidden from planning; the nominal orbit has a 10-second along-track error, and
seven equally weighted hypotheses span ±30 seconds around it. The receiver
uses the Python prototype's 2° FWHM Gaussian beam, 0.8 reference probability,
and 1,000 km reference range with elevation and listening-duration dependence.

Receiver seed 42 and resampling seed 43 use separate SplitMix64 streams.
The plant validates command order, timing, mask, and slew feasibility, samples
acquisition at the listening midpoint, and advances simulated time to listen end.
The example prints pointing error, receiver probability/draw, slew duration,
predicted remaining acquisition probability, resampling, and outcome.
The default run starts about 5.26° off target and acquires after two cycles
(12 simulated seconds); it returns an error on budget exhaustion.

The [acquisition integration suite](../tests/acquisition.rs) covers:

- Weight validation/normalization, Bayesian miss updates, ESS, systematic
  resampling, original-pool preservation, and resampled geometry indices.
- Direction conventions, antenna slew profiles, candidate spread and degenerate
  geometry, elevation masks, partial-slew initialization, and invalid inputs.
- Full-contact scoring against exhaustive search and the Python reference,
  continuation feasibility, incumbent preservation, holding commands, and
  numerical stability for zero/one/tiny probabilities and long schedules.
- TLE/OMM pointing agreement against independent Satkit transforms, checked
  propagation errors, and shared-initializer consistency.
- Controller state transitions, terminal behavior, model failures, impossible
  misses, invalid draws, and rollback. The forced-feedback loop remains a
  separate controller-contract regression.
- Reproducible sampled discovery, feasible timed actions, synchronized plant/
  controller progress, immediate termination after confirmation, a zero-probability
  receiver exhausting all 32 cycles, and execution failures preserving the pending
  action and receiver random sequence.

Run with Satkit data already provisioned:

```sh
SATKIT_OFFLINE=1 cargo run --locked --example acquisition
SATKIT_OFFLINE=1 cargo test --locked --test acquisition
```

The focused suite currently contains 18 passing tests. This verifies the fixture
and acquisition contracts; full release qualification remains a release-preparation
step using the [standard checks](publishing.md).

## Documentation and boundaries

The release adds the acquisition API/defaults guide, README controller example,
simulated discovery instructions, this feature outline, and the
[*Optimal Search for Finding Satellites Post-Launch* reference](optimalSatelliteSearch.pdf).
The [changelog](../CHANGELOG.md) summarizes these additions under 0.9.3.

The caller continues to supply orbit hypotheses, contact selection, receiver/link
model, antenna execution, and random draws. The simulated plant and Gaussian
receiver are example/test support. Automatic prior sampling, multi-contact
scheduling, receiver signal extraction, actuator imperfections, lock maintenance,
and fleet-wide Monte Carlo claims are outside the intended 0.9.3 additions.
