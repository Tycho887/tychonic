"""Synthetic calibrated batch fitting, mixed-sensor EKF, and RTS smoothing."""

import numpy as np
import satkit
from tychonic import low_level as tc


def main():
    epoch = satkit.time.from_rfc3339("2024-01-01T12:00:00.250000Z")
    times = [epoch + satkit.duration.from_seconds(i * 10) for i in range(7)]
    station = tc.GroundStation(40, -75, 100)
    propagation = tc.PropagationConfig(
        epoch,
        fixed_drag=0.02,
        settings=tc.PropagationSettings(use_spaceweather=False),
    )
    truth = np.array([7e6, 0, 1e5, 0, 7400, 100])
    baseline = tc.Interferometer((59, 3, 0.2), 2.23e9)
    models = [
        tc.Observable.doppler(2.23e9),
        tc.Observable.phase_difference(baseline),
        tc.Observable.differential_doppler(baseline),
        tc.Observable.azimuth_elevation(),
    ]
    sigmas = np.array([0.5, 0.01, 0.7, 1e-4, 2e-4])
    covariance = np.outer(sigmas, sigmas) * (0.85 * np.eye(5) + 0.15)
    template = tc.measurements.joint(
        times, station, models, np.zeros((7, 5)), covariance
    )
    readings = template.predict(truth, propagation=propagation)
    batch = tc.measurements.joint(times, station, models, readings, covariance)
    x0 = truth + [25, -30, 20, 0.02, -0.01, 0.03]

    result = tc.fit_orbit(
        [batch], x0, propagation=propagation, options=tc.LMOptions(max_iter=300)
    )
    print(f"Fit: {result.nit} iterations; weighted RSS {result.rss:.6g}")
    print(f"Position error: {np.linalg.norm(result.x[:3] - truth[:3]):.6g} m")

    filter = tc.OrbitEKF(
        x0,
        np.diag([1e4] * 3 + [0.01] * 3),
        propagation=propagation,
        process_noise=tc.WhiteAccelerationNoise(1e-6),
    )
    updates = filter.filter_batches([batch])
    smoothed = filter.smooth()
    print(f"EKF: {sum(update.accepted for update in updates)} accepted updates")
    print(f"Smoothing: {len(smoothed.times)} epochs, states {smoothed.states.shape}")


if __name__ == "__main__":
    main()
