import copy

import numpy as np
import pytest
from tychonic import low_level as tc


def controller(reference, setup):
    data = reference["acquisition"]
    return tc.BCMPC(data["tle"], data["particles"], [1, 1, 1], setup=setup)


def zero(command, geometry, listen_s):
    return np.zeros(geometry.ranges_m.size)


def test_plan_matches_rust_and_snapshots_are_independent(reference, acquisition_setup):
    search = controller(reference, acquisition_setup)
    plan = search.plan(zero)
    expected = reference["acquisition"]
    np.testing.assert_allclose(
        [a.direction_enu for a in plan.actions], expected["directions"], atol=1e-12
    )
    assert [a.listen_start.to_rfc3339() for a in plan.actions] == expected[
        "listen_starts"
    ]
    assert plan.acquisition_probability == expected["probability"]
    assert search.status == tc.SearchStatus.AWAITING_OBSERVATION
    with pytest.raises(tc.AcquisitionStateError):
        search.plan(zero)
    plan.actions[0].direction_enu[:] = 0
    search.observe(tc.Observation.MISS)
    assert np.linalg.norm(search.direction_enu) == pytest.approx(1)
    search.plan(zero)
    search.observe(tc.Observation.ACQUIRED)
    assert search.finished and search.status == tc.SearchStatus.ACQUIRED
    with pytest.raises(tc.AcquisitionStateError):
        search.plan(zero)


def test_misses_exhaust_budget_without_drawing_rng(reference, acquisition_setup):
    search = controller(reference, acquisition_setup)
    rng = np.random.default_rng(42)
    initial_rng_state = copy.deepcopy(rng.bit_generator.state)
    while not search.finished:
        search.plan(zero)
        search.observe("miss", rng)
    assert search.status == tc.SearchStatus.EXHAUSTED
    assert search.cycle == 3
    assert rng.bit_generator.state == initial_rng_state


def test_receiver_errors_preserve_controller_and_python_exception(
    reference, acquisition_setup
):
    search = controller(reference, acquisition_setup)
    error = RuntimeError("radio model failed")

    def fail(*args):
        raise error

    with pytest.raises(RuntimeError) as caught:
        search.plan(fail)
    assert caught.value is error
    assert search.status == tc.SearchStatus.READY
    with pytest.raises(ValueError):
        search.plan(lambda *args: [np.nan] * 3)
    assert search.status == tc.SearchStatus.READY
    search.plan(zero)
    assert search.status == tc.SearchStatus.AWAITING_OBSERVATION


def test_seeded_resampling_is_reproducible(reference, acquisition_setup):
    from dataclasses import replace

    setup = replace(
        acquisition_setup, config=replace(acquisition_setup.config, ess_threshold=1)
    )

    def asymmetric(command, geometry, listen_s):
        return np.where(geometry.visible, [0, 0.99, 0.999], 0)

    snapshots = []
    for _ in range(2):
        search = controller(reference, setup)
        rng = np.random.default_rng(43)
        before = copy.deepcopy(rng.bit_generator.state)
        search.plan(asymmetric)
        update = search.observe("miss", rng)
        assert update.resampled
        assert rng.bit_generator.state != before
        snapshots.append((search.weights, search.particle_indices))
    np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
    np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])


def test_reentrant_planning_is_rejected(reference, acquisition_setup):
    search = controller(reference, acquisition_setup)

    def recursive(*args):
        search.plan(zero)

    with pytest.raises(RuntimeError, match="already in progress"):
        search.plan(recursive)
    assert search.status == tc.SearchStatus.READY
