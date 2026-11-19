"""TCA UKF on three synthetic Rust-generated range-rate/angle readings."""

import numpy as np
import satkit
from tychonic import low_level as tc

TLE = (
    "1 25544U 98067A   24356.58519896  .00014389  00000-0  25222-3 0  9992\n"
    "2 25544  51.6403 106.8969 0007877   6.1421 113.2479 15.50801739487617"
)
READINGS = [
    (
        "2024-12-21T14:07:41.315144Z",
        [6553.637361915445, 1.7364277033493494, -0.09232627218207072],
    ),
    (
        "2024-12-21T14:07:51.315144Z",
        [6560.94754096105, 1.743877629534252, -0.10056042487865696],
    ),
    (
        "2024-12-21T14:08:01.315144Z",
        [6567.080590176417, 1.751045996591249, -0.10868158018630916],
    ),
]


def main():
    filter = tc.TcaUKF(
        TLE,
        satkit.time.from_rfc3339(READINGS[0][0]),
        [0, 0],
        np.diag([150**2, 200**2]),
    )
    station = tc.GroundStation(40, -75, 100)
    for epoch, values in READINGS:
        filter.observe(satkit.time.from_rfc3339(epoch), station, values)
    correction = filter.correction
    print(f"Epoch correction: {correction.epoch_offset_seconds:.6g} seconds")
    print(f"Range-rate bias: {correction.range_rate_bias_m_per_s:.6g} m/s")
    print(f"Corrected OMM epoch: {correction.corrected_omm['EPOCH']}")


if __name__ == "__main__":
    main()
