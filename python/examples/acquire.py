"""Seeded, simulated BC-MPC receiver outcomes; no hardware calls or waiting."""

import numpy as np
import satkit
from tychonic import low_level as tc

TLE = (
    "1 25544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992\n"
    "2 25544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487617"
)


def gaussian_receiver(command, geometry, listen_s):
    error = np.arccos(np.clip(geometry.directions_enu @ command, -1, 1))
    beam = np.exp(-4 * np.log(2) * (error / np.deg2rad(2)) ** 2)
    per_second = np.where(
        geometry.visible,
        0.8
        * np.minimum((1e6 / geometry.ranges_m) ** 2, 1)
        * np.sin(geometry.elevations_rad)
        * beam,
        0,
    )
    return -np.expm1(listen_s * np.log1p(-per_second))


def main():
    setup = tc.SearchSetup(
        tc.GroundStation(19.43636457546619, -30.817318939448704),
        satkit.time.from_rfc3339("2024-12-21T14:12:41.190144Z"),
        config=tc.MPCConfig(
            search_duration_s=18, horizon=2, beam_width=5, candidate_count=5
        ),
    )
    true_orbit = satkit.TLE.from_lines(TLE.splitlines())[0]
    nominal = true_orbit.to_omm()
    epoch = satkit.time.from_rfc3339(nominal["EPOCH"])
    particles = [
        dict(nominal, EPOCH=(epoch + satkit.duration.from_seconds(offset)).to_rfc3339())
        for offset in (-10, 0, 10)
    ]
    search = tc.BCMPC(nominal, particles, [1, 1, 1], setup=setup)
    receiver_rng = np.random.default_rng(2)
    resampling_rng = np.random.default_rng(43)
    station = satkit.itrfcoord(
        latitude_deg=setup.station.lat_deg,
        longitude_deg=setup.station.lon_deg,
        altitude=setup.station.alt_m,
    )
    while not search.finished:
        plan = search.plan(gaussian_receiver)
        action = plan.actions[0]
        midpoint = action.listen_start + (action.listen_end - action.listen_start) / 2
        position, velocity = satkit.sgp4(true_orbit, midpoint)
        position, _ = satkit.frametransform.transform_state(
            satkit.frame.TEME,
            satkit.frame.ITRF,
            midpoint,
            position,
            velocity,
        )
        enu = satkit.itrfcoord(position).to_enu(station)
        distance = np.linalg.norm(enu)
        direction = enu / distance
        elevation = np.arctan2(enu[2], np.hypot(enu[0], enu[1]))
        geometry = tc.ParticleGeometry(
            direction[None, :],
            np.array([distance]),
            np.array([elevation]),
            np.array([elevation >= setup.elevation_mask_rad]),
        )
        probability = gaussian_receiver(
            action.direction_enu, geometry, setup.config.listen_s
        )[0]
        acquired = receiver_rng.random() < probability
        outcome = tc.Observation.ACQUIRED if acquired else tc.Observation.MISS
        search.observe(outcome, resampling_rng)
        print(action.cycle, action.azimuth_elevation, action.listen_end, outcome.value)
    print(search.status.value)


if __name__ == "__main__":
    main()
