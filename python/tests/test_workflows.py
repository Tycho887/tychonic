"""Release checks through the high-level API and installed native module."""

import json
from dataclasses import replace
from itertools import groupby
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
import tychonic as tc

FIXTURE = json.loads((Path(__file__).parent / "data/two_contacts.json").read_text())
KEY = UUID(int=7)
# Both Rust reference generators use a fixed atmosphere for these synthetic passes.
FORCES = tc.ForceModel(
    fixed_drag_m2_kg=FIXTURE["fixed_drag_m2_kg"], use_space_weather=False
)
COLUMNS = (
    "doppler_hz",
    "phase_difference_rad",
    "differential_doppler_hz",
    "azimuth_rad",
    "elevation_rad",
)


def job(start, stop):
    result = tc.JobData(KEY)
    for contact, group in groupby(
        FIXTURE["observations"][start:stop], lambda row: row["contact_id"]
    ):
        readings = list(group)

        def column(name):
            return np.array([r[name] for r in readings])

        site = readings[0]["station"]
        station = tc.GroundStation(
            readings[0]["antenna_id"], site["lat_deg"], site["lon_deg"], site["alt_m"]
        )
        samples = tc.ContactSamples.from_unix_seconds(
            UUID(contact),
            station,
            column("epoch_unix_s"),
            carrier_hz=FIXTURE["carrier_hz"],
            observation_ids=column("observation_id"),
        )
        result.add_joint(
            tc.ObservationBatch(
                samples=samples,
                phase_baseline_enu_m=column("baseline_enu_m"),
                differential_baseline_enu_m=column("baseline_enu_m"),
                covariance=column("measurement_covariance"),
                **{k: column(k) for k in COLUMNS},
            )
        )
    return result


def problem(history):
    guess = np.array(FIXTURE["initial_state_gcrf"])
    seed = tc.StateEstimate("2024-01-01T12:00:00.250000Z", guess[:3], guess[3:])
    opts = {
        k: FIXTURE[k]
        for k in (
            "max_iterations",
            "gradient_tolerance",
            "step_tolerance",
            "cost_tolerance",
        )
    }
    return tc.ODProblem(
        history, initial_guess=seed, profile=tc.CalibratedLM(**opts), force_model=FORCES
    )


def catalogue(**kwargs):
    return tc.EnsembleFilter.ekf(force_model=FORCES, retain_history=True, **kwargs)


def test_batch_to_live_resume():
    history = job(0, 39)
    task = problem(history)
    at_seed = task.solve()
    assert at_seed.output_epoch == task.initial_guess.epoch
    assert at_seed.last_contributing_epoch == "2024-01-01T12:05:04.250000Z"
    assert at_seed.output_epoch != at_seed.last_contributing_epoch
    result = task.solve(output_epoch=at_seed.last_contributing_epoch)
    assert (
        result.output_epoch
        == result.last_contributing_epoch
        == at_seed.last_contributing_epoch
    )
    marginal = result.estimate.orbital_marginal()
    assert marginal.state_si.shape == (6,)
    native = json.loads(
        (Path(__file__).parent / "data/handoff_reference.json").read_text()
    )
    np.testing.assert_allclose(
        marginal.state_si, native["published_state"], rtol=1e-7, atol=0.02
    )
    np.testing.assert_allclose(
        marginal.covariance_si,
        np.array(native["published_covariance"]).reshape(6, 6),
        rtol=1e-4,
        atol=1e-7,
    )
    publication = marginal.covariance_si * 1e-6
    np.linalg.cholesky(publication)
    np.testing.assert_allclose(publication * 1e6, result.estimate.covariance)
    assert 6000 < np.linalg.norm(marginal.state_si[:3] * 1e-3) < 8000
    running = catalogue()
    running.add_spacecraft(
        KEY,
        result.estimate,
        bias_prior_hz=tc.GaussianPrior(-0.5, 5.0),
        last_contributing_epoch=result.last_contributing_epoch,
    )
    restored = catalogue()
    restored.restore(running.checkpoint(KEY))
    held = job(39, 42)
    assert running.update(held) == restored.update(held)
    saved = running.checkpoint(KEY)
    assert saved.last_contributing_epoch == held.batches[0].samples.epochs[-1]
    assert saved.last_accepted_observation == saved.last_processed_reading
    resumed = restored.checkpoint(KEY)
    assert resumed.last_contributing_epoch == saved.last_contributing_epoch
    assert resumed.last_processed_reading == saved.last_processed_reading
    assert resumed.last_accepted_observation == saved.last_accepted_observation
    state = restored.get_state(KEY)
    assert np.any(state.covariance[:6, 6] != 0)
    np.testing.assert_allclose(
        state.covariance,
        np.array(native["final_covariance"]).reshape(7, 7),
        rtol=1e-4,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.r_[state.position_m, state.velocity_m_s, state.doppler_bias_hz],
        native["final_state"],
        rtol=1e-7,
        atol=0.02,
    )
    np.testing.assert_array_equal(state.covariance, running.get_state(KEY).covariance)
    # Resume again from a nonempty retained history.
    again = catalogue()
    again.restore(saved)
    for a, b in zip(again.smooth(KEY), restored.smooth(KEY), strict=True):
        assert a.readings == b.readings and a.estimate.epoch == b.estimate.epoch
        np.testing.assert_array_equal(a.estimate.covariance, b.estimate.covariance)
        np.testing.assert_array_equal(a.estimate.position_m, b.estimate.position_m)
        assert a.estimate.doppler_bias_hz == b.estimate.doppler_bias_hz


def test_prior_and_gating():
    task = problem(job(0, 1))
    with pytest.raises(tc.RankDeficientError) as failure:
        task.solve()
    assert failure.value.rank < failure.value.parameters == 6
    P = np.diag([100, 100, 100, 0.1, 0.1, 0.1])
    P[0, 3] = P[3, 0] = 0.3
    fit = replace(
        task, weighted_prior=replace(task.initial_guess, covariance=P)
    ).solve()
    assert fit.used_weighted_prior and fit.covariance_method == "gaussian_posterior"
    cat = catalogue(gate=0)
    cat.add_spacecraft(
        KEY,
        fit.estimate,
        bias_prior_hz=tc.GaussianPrior(1, 3),
        last_contributing_epoch=fit.last_contributing_epoch,
    )
    (update,) = cat.update(job(1, 2))
    assert not update.accepted
    saved = cat.checkpoint(KEY)
    assert saved.last_accepted_observation is None
    assert saved.last_processed_reading == update.reading
    assert saved.last_contributing_epoch == fit.last_contributing_epoch
    assert saved.estimate_epoch == update.reading.epoch


def test_exports(tmp_path):
    observations = job(0, 3)
    batch = observations.batches[0]
    (path,) = observations.export_to_ccsds_tdm(
        tmp_path,
        originator="Telemetry",
        creation_epoch=batch.samples.epochs[0],
        spacecraft_participant="SAT",
        antenna_participants={"antenna-0": "GS"},
    )
    entries = [
        line.split()
        for line in path.read_text().splitlines()
        if line.startswith("DOPPLER_INSTANTANEOUS")
    ]
    np.testing.assert_allclose(
        [float(e[-1]) for e in entries],
        batch.doppler_hz * (-299792.458 / FIXTURE["carrier_hz"]),
    )
    seed = problem(observations).initial_guess
    text = seed.to_oem(
        batch.samples.epochs,
        force_model=FORCES,
        object_name="SAT",
        object_id="1999-001A",
        originator="Telemetry",
    )
    state = np.fromstring(
        text.split("META_STOP\n")[1].splitlines()[0].split(" ", 1)[1], sep=" "
    )
    np.testing.assert_allclose(state[:3], seed.position_m / 1000)


def test_tle_prior_and_acquisition(reference):
    d = reference["acquisition"]
    tle = tuple(d["tle"].strip().splitlines()[-2:])
    model = tc.TleAgeUncertainty(
        sigma_at_epoch=(100, 200, 300, 1, 2, 3), sigma_growth_per_day=(1,) * 6
    )
    seed = tc.StateEstimate.from_tle(tle, at=d["start"], uncertainty=model)
    assert seed.tle_initialization.uncertainty.version == 1
    np.linalg.cholesky(seed.covariance)
    setup = tc.SearchSetup(
        tc.GroundStation("site", *d["station"]),
        d["start"],
        horizon=2,
        beam_width=5,
        candidate_count=5,
        search_duration_s=18,
    )
    finder = tc.SatFinder.from_tle(
        tle, setup=setup, n_particles=8, rng=np.random.default_rng(17)
    )
    assert len(finder.plan().actions) > 0
    finder.observe(acquired=True)
    assert finder.read_state().status == "acquired"


def test_ukf_carrier_checkpoint_and_replacement(reference):
    data = reference["tca"]
    tle = tuple(data["tle"].strip().splitlines()[-2:])
    noise = tc.TleProcessNoise(
        offset_psd_s2_s=0.01,
        bias_psd_hz2_s=0.04,
        max_offset_sigma_s=60,
        max_bias_sigma_hz=200,
    )
    seed = tc.TleCorrection(
        tle,
        data["start"],
        epoch_offset_s=0,
        doppler_bias_hz=-7,
        covariance=np.array([[400.0, 5.0], [5.0, 100.0]]),
    )

    def catalogue():
        return tc.EnsembleFilter.tle_epoch_ukf(noise=noise)

    cat = catalogue()
    cat.add_spacecraft(KEY, seed)
    restored = catalogue()
    restored.restore(cat.checkpoint(KEY))
    for i, reading in enumerate(data["readings"][:2]):
        carrier = (2.23 + i) * 1e9
        rr, az, el = reading["values"]
        single = tc.JobData(KEY)
        sample = tc.ContactSamples(
            UUID(int=i + 1),
            tc.GroundStation("gs", 40, -75, 100),
            [reading["time"]],
            carrier_hz=carrier,
            observation_ids=[str(i)],
        )
        sigma = np.array([2.0, 0.01, 0.02])
        P = np.outer(sigma, sigma) * (0.1 + 0.9 * np.eye(3))
        single.add_joint(
            tc.ObservationBatch(
                samples=sample,
                covariance=P,
                doppler_hz=np.array([-rr * carrier / 299792458.0]),
                azimuth_rad=[az],
                elevation_rad=[el],
            )
        )
        assert cat.update(single) == restored.update(single)
    before = cat.get_state(KEY)
    cat.replace_tle(
        KEY, tle, epoch=before.epoch, offset_prior_s=tc.GaussianPrior(0, 10)
    )
    after = cat.get_state(KEY)
    assert after.doppler_bias_hz == before.doppler_bias_hz
    assert after.covariance[0, 1] == 0
    assert after.covariance[1, 1] == before.covariance[1, 1]
    cat.predict_to("2025-12-21T14:07:41.315144Z")
    np.testing.assert_allclose(
        np.sqrt(np.diag(cat.get_state(KEY).covariance)), [60, 200]
    )


@pytest.mark.parametrize("offset_s", [0.0, 299.875])
def test_mean_elements_exports(reference, offset_s):
    from datetime import datetime, timedelta

    data = reference["acquisition"]
    tle = tuple(data["tle"].strip().splitlines()[-2:])
    start = datetime.fromisoformat(data["start"]) + timedelta(seconds=offset_s)
    epoch = start.isoformat().replace("+00:00", "Z")
    uncertainty = tc.TleAgeUncertainty(
        sigma_at_epoch=(1,) * 6, sigma_growth_per_day=(0,) * 6
    )
    seed = tc.StateEstimate.from_tle(
        tle,
        at=epoch,
        uncertainty=uncertainty,
    )
    # A three-hour arc constrains the fit; a ten-minute arc can exhaust the solver.
    epochs = [
        (start + timedelta(minutes=(i - 30) * 3)).isoformat().replace("+00:00", "Z")
        for i in range(61)
    ]
    exported = seed.to_tle(epochs, force_model=FORCES)
    assert len(exported) == 2
    restored = tc.StateEstimate.from_tle(exported, at=epoch, uncertainty=uncertainty)
    assert np.linalg.norm(restored.position_m - seed.position_m) < 500
    assert float(seed.to_omm(epochs, force_model=FORCES)["MEAN_MOTION"]) > 0


def test_polars_and_optical_batches():
    data = job(0, 2)
    rf = data.batches[0]
    samples = replace(rf.samples, carrier_hz=None)
    data.add_joint(
        tc.ObservationBatch(
            samples=samples,
            azimuth_rad=rf.azimuth_rad,
            elevation_rad=rf.elevation_rad,
            covariance=np.array([[1e-8, 3e-9], [3e-9, 4e-8]]),
        )
    )
    table = data.to_polars()
    assert table.height == 4
    assert table["carrier_hz"].null_count() == 2
    assert table["observation_id"].to_list() == list(samples.observation_ids) * 2
    np.testing.assert_array_equal(
        np.array(table["covariance"].to_list()[0]), rf.covariance[0]
    )


def test_doppler_bias_and_constrained_profiles():
    reference = json.loads(
        (Path(__file__).parent / "data/contact_reference.json").read_text()
    )
    x = np.array(reference["truth"])
    seed = tc.StateEstimate(reference["epoch"], x[:3], x[3:6], drag_m2_kg=x[6])
    # The measured range rates include the known pass biases of the native fixture.
    data = tc.JobData(KEY)
    carrier = 2.23e9
    for contact, group in groupby(reference["rows"], lambda r: r["id"]):
        rows = list(group)
        site = tc.GroundStation(contact, *rows[0]["station"])
        samples = tc.ContactSamples(
            UUID(contact), site, [r["epoch"] for r in rows], carrier
        )
        data.add_joint(
            tc.ObservationBatch(
                samples=samples,
                doppler_hz=np.array([r["rate"] for r in rows]) * (-carrier / 299792458),
                azimuth_rad=[r["az"] for r in rows],
                elevation_rad=[r["el"] for r in rows],
                covariance=np.diag([(0.1 * carrier / 299792458) ** 2, 1e-6, 1e-6]),
            )
        )
    for profile in (tc.ContactBiasFit(), tc.ConstrainedFit(full_fov_rad=0.0003)):
        fit = tc.ODProblem(
            data, initial_guess=seed, force_model=FORCES, profile=profile
        ).solve()
        np.testing.assert_allclose(fit.estimate.position_m, x[:3], atol=2, rtol=0)
        actual = [fit.contact_bias_hz[UUID(int=i)] for i in range(3)]
        np.testing.assert_allclose(
            actual, np.array(reference["biases"]) * (-carrier / 299792458), atol=0.05
        )
        if isinstance(profile, tc.ConstrainedFit):
            assert fit.estimate.covariance is None
            assert fit.max_pointing_violation_rad <= 1e-8


def test_rf_fit_and_propagated_output():
    data = job(0, 39)
    for i, b in enumerate(data.batches):
        data.batches[i] = replace(
            b,
            azimuth_rad=None,
            elevation_rad=None,
            differential_doppler_hz=None,
            differential_baseline_enu_m=None,
            covariance=np.diag([0.25, 1e-4]),
        )
    task = problem(data)
    prior = replace(task.initial_guess, covariance=np.diag([1e4] * 3 + [0.01] * 3))
    profile = tc.PassiveRFFit(
        frequency_offset_hz=tc.GaussianPrior(0, 2),
        phase_offset_rad=tc.GaussianPrior(0, 0.5),
        drift_correlation_s=40,
        drift_sigma_rad=0.03,
    )
    result = replace(task, profile=profile, weighted_prior=prior).solve(
        output_epoch=data.batches[-1].samples.epochs[-1]
    )
    assert result.rf_covariance.shape == (8, 8)
    assert result.phase_drift_rad.shape == (39,)
    assert tuple(result.rf_parameters)[-2:] == (
        "frequency_offset_hz",
        "phase_offset_rad",
    )
    np.testing.assert_array_equal(
        result.estimate.covariance, result.rf_covariance[:6, :6]
    )
