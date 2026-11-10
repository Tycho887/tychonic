# Behavioral antenna simulation

The `simulation` module provides caller-clocked mount motion, Gaussian beam
attenuation, hard reception exclusion, receiver lock hysteresis, seeded Doppler
noise, and actual-epoch orbital observations. It does not own HTTP, scheduling,
hardware access, or truth disclosure. These are Development models; no physical
antenna or receiver calibration is claimed.

All numerical time advances explicitly in seconds. Angles use radians, distances
metres, and Doppler Hz with negative recession. A `Mount` exposes actual aperture
angles. A `Receiver` exposes no Doppler while unlocked. The default reception
cutoff is an inclusive 3 degrees, with immediate exclusion outside that cone.
The receiver uses 1000 km as its reference link range, a Gaussian power beam, and
configurable acquisition/hold thresholds and acquisition dwell.

`orbital_sample` computes spacecraft direction and frequency from a supplied TLE.
Caller code must keep spacecraft truth separate from a program-track ephemeris.
`shifted_epoch` invalidates the cloned propagation cache and preserves observation
times. It is distinct from evaluating a program track ahead in time.

The independent tests check the reception boundary, loss, seeded replay, mount
motion and limits, and rejected input. Existing acquisition and UKF tests remain
required. Use `cargo test --locked` with provisioned Satkit data and
`SATKIT_OFFLINE=1` before consuming a published release.
