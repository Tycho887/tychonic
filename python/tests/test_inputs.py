from datetime import datetime

import numpy as np
import pytest
import satkit
from tychonic import _native
from tychonic import low_level as tc


@pytest.mark.parametrize(
    "label",
    [
        "2024-01-01T00:00:00.000001Z",
        "2016-12-31T23:59:59.999999Z",
        "2016-12-31T23:59:60.123456Z",
        "2017-01-01T00:00:00.000001Z",
    ],
)
def test_time_roundtrip_preserves_microseconds_and_leap_seconds(label):
    time = satkit.time.from_rfc3339(label)
    result = satkit.time.from_rfc3339(_native.roundtrip_time(time.to_rfc3339()))
    assert time == result


def test_reject_implicit_time_and_unsupported_settings():
    with pytest.raises(TypeError, match="satkit.time"):
        tc.PropagationConfig(datetime(2024, 1, 1))
    with pytest.raises(ValueError, match="integrator"):
        tc.PropagationConfig(
            satkit.time(2024, 1, 1),
            settings=tc.PropagationSettings(integrator="RODAS4"),
        )
    with pytest.raises(ValueError):
        tc.PropagationConfig(
            satkit.time(2024, 1, 1), settings=tc.PropagationSettings(gravity_order=8)
        )
    with pytest.raises(ValueError):
        tc.FilterOptions(gate=-1)


def test_noncontiguous_inputs_and_owned_batch(orbit_case):
    data, propagation, station, models, times, _, _ = orbit_case
    values = np.asarray(data["values"])
    padded = np.repeat(values, 2, axis=1)
    sliced = padded[:, ::2]
    assert not sliced.flags.c_contiguous
    covariance = np.asarray(data["covariance"]).T
    batch = tc.measurements.joint(times, station, models, sliced, covariance)
    sliced[:] = 123
    predicted = batch.predict(data["truth"], propagation=propagation)
    np.testing.assert_allclose(predicted, values, rtol=1e-10, atol=1e-9)


def test_invalid_measurement_inputs_and_filter_dimensions(orbit_case):
    data, propagation, station, models, times, _, prior = orbit_case
    for covariance in (np.zeros((5, 5)), np.ones((5, 4)), np.eye(6)):
        with pytest.raises(ValueError):
            tc.measurements.joint(times, station, models, data["values"], covariance)
    for values in (np.zeros((2, 5)), np.full((7, 5), np.nan)):
        with pytest.raises(ValueError):
            tc.measurements.joint(times, station, models, values, data["covariance"])
    with pytest.raises(ValueError):
        tc.OrbitEKF([1, 2], prior, propagation=propagation)
    with pytest.raises(ValueError):
        tc.measurements.doppler(times, station, 2.23e9, [0] * len(times), -1)


def test_covariance_helper_and_noise_units():
    np.testing.assert_array_equal(
        tc.measurements.doppler_differential_covariance(2, 3), [[4, 4], [4, 13]]
    )
    epoch = satkit.time(2024, 1, 1)
    q = tc.WhiteAccelerationNoise((1, 2, 3), 4)(
        np.zeros(7), epoch, epoch + satkit.duration.from_seconds(2)
    )
    assert q[0, 0] == pytest.approx(8 / 3)
    assert q[0, 3] == 2
    assert q[3, 3] == 2
    assert q[6, 6] == 8
    with pytest.raises(ValueError):
        tc.WhiteAccelerationNoise(1, 4)(
            np.zeros(6), epoch, epoch + satkit.duration.from_seconds(2)
        )


def test_unavailable_eop_coverage_has_actionable_data_error():
    epoch = satkit.time(1900, 1, 1)
    propagation = tc.PropagationConfig(
        epoch, settings=tc.PropagationSettings(require_eop_coverage=True)
    )
    batch = tc.measurements.doppler(
        [epoch + satkit.duration.from_seconds(10)],
        tc.GroundStation(40, -75),
        2.23e9,
        [0],
        1,
    )
    with pytest.raises(tc.DataError, match="SATKIT_DATA"):
        batch.predict([7e6, 0, 1e5, 0, 7400, 100], propagation=propagation)
