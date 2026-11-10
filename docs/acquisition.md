# BC-MPC acquisition

The `detection` module implements the acquisition math in *Optimal Search for
Finding Satellites Post-Launch*, particularly the Bayesian miss update in Eq. (3)
and the full-contact continuation objective in Eqs. (9–10). The Python prototype
in `satfinder-MPC` supplies the candidate construction and approximate beam-search
reference. Antenna execution remains caller-owned.

## Inputs and controller loop

Construct `ParticleBelief::new(particles, weights)` from explicit TLE/OMM
`Ephemeris` values. Particle count comes from this input; weights are normalized
and must be finite, nonnegative, and have positive support. Supply the nominal
orbit separately. The caller is responsible for choosing hypotheses that express
its uncertainty; no Gaussian prior is imposed or sampled.

`SearchSetup::new(station, start)` selects the defaults below. Set its optional
`initial_direction` if the antenna is already positioned away from nominal.
The initial direction must be above the elevation mask. The antenna is assumed
to be at rest at that direction before the search clock starts.

Create `BcMpc::new(nominal, belief, setup)`. Repeatedly call `plan(model)`, copy
the first `Action`, execute its slew/settle/listen interval, then call
`observe(outcome, uniform_draw)`. The returned plan borrows controller-owned
storage and describes every remaining cycle. Its probability and the retained
plan's probability are evaluated under the same current belief and model.

`SearchStatus` distinguishes readiness, a pending observation, confirmation, and
budget exhaustion. A pending action must be observed before another plan can be
requested. `Observation::Acquired` terminates search without updating the miss
belief. `Observation::Miss` updates weights, optionally resamples, then shifts
the continuation one cycle. There is no nominal-tracking or ATUS policy switch;
ATUS is a private initialization step.

Execution failures are not misses. Keep the copied action pending while retrying
execution, or abandon the controller. Probability evaluation failures,
unsupported misses, and invalid resampling draws preserve controller state.
Side effects inside a caller's model or random generator cannot be rolled back.

## Receiver boundary

Implement `DetectionModel::probabilities(command, particles, listen_s, output)`.
Overwrite every output with an acquisition probability in `[0, 1]`, one for each
current particle, and use zero for invisible particles. Geometry provides ENU
direction, range, elevation, and visibility. Evaluate deterministically within a
single plan: commands may be visited repeatedly, and equal directions share a
likelihood cache. A model may change between calls to `plan`.

The library includes no radio/link-budget model. The reference test supplies
the Python prototype's Gaussian beam with its explicit on-axis range/elevation
function. Geometry is held at the listening midpoint and successive attempts
are assumed independent conditional on an orbit hypothesis. Commands remain
fixed through listening, and confirmation is reported at its end.

## Geometry and optimization kernels

`propagate_sgp4_gcrf` in `kernels` accepts any Satkit `SGP4Source`, checks every
status, and returns batch GCRF states. `propagate_pointing_grid` combines it with
`PreparedGeometry`; `pointing_geometry` also accepts externally propagated GCRF
positions and a prepared station row. Station positions and rotations retain
the existing `qitrf2gcrf_approx` convention. Thus comparisons with direct,
full-accuracy TEME-to-ITRF pointing may include that approximation's error.
Zenith has a valid acquisition direction even though an azimuth Jacobian is
undefined there. Zero range and propagation failures are errors.

`along_track_candidates` uses the initial unweighted prior's population spread
in the nominal pointing's spherical tangent plane. Positive phase comes from
nominal ±1-second mean-anomaly shifts at the same observation time, leaving the
element epoch unchanged. Degenerate phase geometry uses the dominant tangent
axis; coincident particles produce coincident candidates. Antipodal prior
directions and candidate extents of π or more are rejected.

`AntennaLimits` supplies great-circle rest-to-rest slew time, reachability within
the move budget, and a partial slew. The private bootstrap follows the paper's
along-track sweep, choosing reachable above-mask candidates or a partial slew,
and holding when no valid target is available.

`optimize_prefix` accepts current geometry/candidates, a feasible retained
schedule, normalized or unnormalized weights, `PlannerSettings`, and a model.
It changes only the next `min(horizon, remaining_cycles)` commands, preserving
the suffix and checking the transition into it. Candidates include moving
pointings, retained commands, and holding at the current pointing. Stable beam
ties follow that order. Incomplete prefixes are ranked with the fixed suffix
beyond the horizon, matching the prototype's approximation. The old schedule
remains an incumbent, guaranteeing a returned score at least as good under the
current belief; beam search does not guarantee a globally optimal sequence.

`schedule_acquisition_probability` scores detection-probability tables directly.
Scoring and Bayesian updates use log-space arithmetic, including exact zero and
one probabilities. `normalize_weights` and `systematic_resampling_indices` are
independently usable kernels.

The belief owns its original ephemeris pool. Active hypotheses are indices into
that pool, so resampling does not copy or modify ephemerides. Geometry is cached
for the original pool over the full search; active rows follow the resampled
indices. Candidate spread remains tied to the original prior. Likelihood caches
last one plan, and its first action's likelihoods are retained for observation.

## Defaults and limits

| Setting | Default |
| --- | --- |
| Search budget | 192 seconds, 32 complete cycles |
| Move / settle / listen | 4 / 0.5 / 1.5 seconds |
| Optimized prefix / beam width | 4 cycles / 40 |
| Moving candidates / extent | 17 / ±2.8 population standard deviations |
| ESS resampling threshold | 0.5 × current particle count |
| Antenna rate / acceleration | 1.10°/s / 1.00°/s², represented in radians |
| Elevation mask | 10°, represented in radians |

Azimuth is eastward from north in `[0, 2π)`, with zero at zenith/nadir. Public
acquisition angles are radians; existing ground-station latitude and longitude
fields remain degrees. Distances are metres, durations seconds, and times
Satkit `Instant` values. Durations must contain complete cycles and be
representable by Satkit's microsecond clock.

The resampling callback supplies a finite uniform value in `[0, 1)` and is called
only when resampling is needed. The caller can use an independently seeded RNG
for planning/resampling and for physical or simulated receiver outcomes.

Contact selection, prior sampling, multi-contact scheduling, numerical orbit
hypotheses, mount axis limits, actuator imperfections, and lock maintenance are
outside this controller. Existing Satkit data provisioning/offline behavior is
unchanged. The reference test verifies a deterministic loop and kernel contracts;
it does not reproduce the paper's fleet statistics or promise acquisition.
