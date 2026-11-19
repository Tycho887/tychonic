"""Deterministic fixtures produced by the Rust API in examples/reference.rs."""

import json
from pathlib import Path

import numpy as np
import pytest
import satkit
from tychonic import low_level as tc


@pytest.fixture(scope="session")
def reference():
    return json.loads((Path(__file__).parent / "data/reference.json").read_text())


@pytest.fixture(params=["orbit", "drag"])
def orbit_case(reference, request):
    data = reference[request.param]
    propagation = tc.PropagationConfig(
        satkit.time.from_rfc3339(data["epoch"]),
        settings=tc.PropagationSettings(use_spaceweather=False),
        estimate_drag=request.param == "drag",
        fixed_drag=0.02,
    )
    station = tc.GroundStation(40, -75, 100)
    baseline = tc.Interferometer((59, 3, 0.2), 2.23e9)
    models = [
        tc.Observable.azimuth_elevation(),
        tc.Observable.differential_doppler(baseline),
        tc.Observable.phase_difference(baseline),
        tc.Observable.doppler(2.23e9),
    ]
    times = [satkit.time.from_rfc3339(time) for time in data["times"]]
    batch = tc.measurements.joint(
        times, station, models, data["values"], data["covariance"]
    )
    prior = np.diag(
        [1e4] * 3 + [0.01] * 3 + ([1e-4] if propagation.estimate_drag else [])
    )
    return data, propagation, station, models, times, batch, prior


@pytest.fixture
def acquisition_setup(reference):
    data = reference["acquisition"]
    return tc.SearchSetup(
        tc.GroundStation(*data["station"]),
        satkit.time.from_rfc3339(data["start"]),
        config=tc.MPCConfig(
            search_duration_s=18, horizon=2, beam_width=5, candidate_count=5
        ),
    )
