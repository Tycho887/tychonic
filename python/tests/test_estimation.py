import numpy as np
import pytest
import satkit
from tychonic import low_level as tc


@pytest.mark.parametrize("mask", range(1, 16))
def test_all_sensor_combinations_update_six_and_seven_state_filters(orbit_case, mask):
    data, propagation, station, reversed_models, times, _, prior = orbit_case
    models = list(reversed(reversed_models))
    selected = [model for i, model in enumerate(models) if mask & (1 << i)]
    columns = [i for i, bit in enumerate((1, 2, 4, 8, 8)) if mask & bit]
    values = np.asarray(data["values"])[0:1, columns]
    covariance = np.asarray(data["covariance"])[np.ix_(columns, columns)]
    batch = tc.measurements.joint([times[0]], station, selected, values, covariance)
    filter = tc.OrbitEKF(data["truth"], prior, propagation=propagation)
    update = filter.observe(batch)
    assert update.accepted
    np.testing.assert_allclose(update.innovation, 0, atol=1e-8)
    assert np.isfinite(filter.covariance).all()


def test_failed_update_rolls_back_its_prediction(orbit_case):
    data, propagation, station, models, times, _, prior = orbit_case
    batch = tc.measurements.joint(
        [times[1]], station, models, np.full((1, 5), 1e308), data["covariance"]
    )
    filter = tc.OrbitEKF(data["truth"], prior, propagation=propagation)
    with pytest.raises(tc.NumericalError):
        filter.observe(batch, q=prior * 1e-5)
    np.testing.assert_array_equal(filter.state, data["truth"])
    np.testing.assert_array_equal(filter.covariance, prior)
    assert filter.epoch == times[0]
    assert len(filter.history.times) == 1


def test_batch_predictions_and_fit_match_rust(orbit_case):
    data, propagation, _, _, _, batch, _ = orbit_case
    np.testing.assert_allclose(
        batch.predict(data["truth"], propagation=propagation),
        data["values"],
        rtol=1e-10,
        atol=1e-9,
    )
    fit = tc.fit_orbit(
        [batch], data["x0"], propagation=propagation, options=tc.LMOptions(max_iter=300)
    )
    np.testing.assert_allclose(fit.x, data["fit"]["x"], rtol=1e-10, atol=1e-6)
    assert fit.nit == data["fit"]["nit"]
    assert fit.rss == pytest.approx(data["fit"]["rss"], rel=1e-5, abs=1e-12)
    assert fit.cost == fit.rss / 2
    with pytest.raises(tc.SolverError):
        tc.fit_orbit(
            [batch],
            data["x0"],
            propagation=propagation,
            options=tc.LMOptions(max_iter=0),
        )


def test_ekf_and_smoothing_match_rust(orbit_case):
    data, propagation, _, _, _, batch, prior = orbit_case
    noise = tc.WhiteAccelerationNoise(1e-6, 1e-10 if propagation.estimate_drag else 0)
    filter = tc.OrbitEKF(
        data["x0"], prior, propagation=propagation, process_noise=noise
    )
    updates = filter.filter_batches([batch])
    expected = data["filter"]
    np.testing.assert_allclose(filter.state, expected["state"], rtol=1e-11, atol=1e-7)
    np.testing.assert_allclose(
        filter.covariance, expected["covariance"], rtol=1e-8, atol=1e-7
    )
    np.testing.assert_allclose(
        [u.nis for u in updates], expected["nis"], rtol=1e-7, atol=1e-9
    )
    np.testing.assert_allclose(
        filter.history.states, expected["states"], rtol=1e-11, atol=1e-7
    )
    smooth = filter.smooth()
    assert smooth.times == filter.history.times
    np.testing.assert_allclose(
        smooth.states, expected["smooth_states"], rtol=1e-10, atol=1e-6
    )
    np.testing.assert_allclose(
        smooth.covariances, expected["smooth_covariances"], rtol=1e-7, atol=1e-6
    )
    snapshot = filter.state
    snapshot[:] = 0
    assert filter.state[0] != 0


def test_individual_and_batch_updates_agree(orbit_case):
    data, propagation, _, _, times, batch, prior = orbit_case
    noise = tc.WhiteAccelerationNoise(1e-6)
    one = tc.OrbitEKF(data["x0"], prior, propagation=propagation, process_noise=noise)
    many = tc.OrbitEKF(data["x0"], prior, propagation=propagation, process_noise=noise)
    updates = [one.observe(batch, row) for row in range(len(times))]
    many.filter_batches([batch])
    np.testing.assert_allclose(one.state, many.state, atol=1e-8)
    np.testing.assert_allclose(one.covariance, many.covariance, atol=1e-8)
    assert all(update.accepted for update in updates)


def test_noise_once_per_epoch_stable_order_and_partial_progress(orbit_case):
    data, propagation, _, _, times, batch, prior = orbit_case
    calls = []

    def noise(state, previous, next_epoch):
        calls.append((previous, next_epoch))
        assert state.size == propagation.dimension
        return np.zeros_like(prior)

    filter = tc.OrbitEKF(
        data["truth"], prior, propagation=propagation, process_noise=noise
    )
    updates = filter.filter_batches([batch, batch])
    assert len(calls) == len(times) - 1
    assert [(u.batch, u.row) for u in updates[:4]] == [(0, 0), (1, 0), (0, 1), (1, 1)]

    error = RuntimeError("receiver noise unavailable")

    def failure(state, previous, next_epoch):
        raise error

    failed = tc.OrbitEKF(
        data["truth"], prior, propagation=propagation, process_noise=failure
    )
    with pytest.raises(RuntimeError) as exc:
        failed.filter_batches([batch])
    assert exc.value is error
    assert failed.epoch == times[0]
    assert np.trace(failed.covariance) < np.trace(prior)  # first row committed
    assert len(failed.history.times) == 1


def test_reentrant_mutation_is_rejected(orbit_case):
    data, propagation, _, _, _, batch, prior = orbit_case

    def noise(state, previous, next_epoch):
        filter.predict_to(next_epoch)
        return np.zeros_like(prior)

    filter = tc.OrbitEKF(
        data["truth"], prior, propagation=propagation, process_noise=noise
    )
    with pytest.raises(RuntimeError, match="already in progress"):
        filter.filter_batches([batch])


def test_gating_rollback_chronology_and_explicit_q(orbit_case):
    data, propagation, station, models, times, batch, prior = orbit_case
    filter = tc.OrbitEKF(
        data["truth"], prior, propagation=propagation, options=tc.FilterOptions(gate=0)
    )
    bad = tc.measurements.joint(
        [times[1]],
        station,
        models,
        np.asarray(data["values"])[1:2] + 1,
        data["covariance"],
    )
    result = filter.observe(bad)
    assert not result.accepted
    assert filter.epoch == times[1]
    saved = filter.state.copy(), filter.covariance.copy(), filter.epoch
    with pytest.raises(ValueError, match="chronological"):
        filter.observe(batch, 0)
    np.testing.assert_array_equal(filter.state, saved[0])
    np.testing.assert_array_equal(filter.covariance, saved[1])
    assert filter.epoch == saved[2]
    with pytest.raises(ValueError, match="zero process noise"):
        filter.predict_to(filter.epoch, q=np.eye(propagation.dimension))

    def failure(*args):
        raise RuntimeError("should not run for explicit Q")

    explicit = tc.OrbitEKF(
        data["truth"], prior, propagation=propagation, process_noise=failure
    )
    explicit.observe(batch, 1, q=np.zeros_like(prior))
    assert explicit.epoch == times[1]


def test_tca_matches_rust_and_returns_separate_correction(reference):
    data = reference["tca"]
    filter = tc.TcaUKF(
        data["tle"],
        satkit.time.from_rfc3339(data["start"]),
        [0, 0],
        np.diag([22500, 40000]),
    )
    updates = [
        filter.observe(
            satkit.time.from_rfc3339(row["time"]),
            tc.GroundStation(40, -75, 100),
            row["values"],
        )
        for row in data["readings"]
    ]
    np.testing.assert_allclose(filter.state, data["state"], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(
        filter.covariance, data["covariance"], rtol=1e-9, atol=1e-7
    )
    np.testing.assert_allclose(
        [u.nis for u in updates], [row["nis"] for row in data["readings"]], rtol=1e-8
    )
    assert filter.corrected_omm == data["corrected_omm"]
    assert filter.correction.range_rate_bias_m_per_s == filter.state[1]
    state = filter.state.copy()
    covariance = filter.covariance.copy()
    with pytest.raises(ValueError):
        filter.observe(
            filter.epoch,
            tc.GroundStation(40, -75),
            [0, 0, 0],
            noise=tc.TcaMeasurementNoise(sigma_rr=-1),
        )
    np.testing.assert_array_equal(filter.state, state)
    np.testing.assert_array_equal(filter.covariance, covariance)
