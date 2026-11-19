"""Dispatch reviewed solver profiles to the corresponding native objective."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from uuid import UUID

import numpy as np

from . import _native, api

_C = 299792458.0


@dataclass
class _Fit:
    x: np.ndarray
    covariance: np.ndarray | None
    rss: float
    iterations: int
    evidence: str
    details: dict


def _calibrated(problem, config, guess):
    p = problem.profile
    prior = problem.weighted_prior
    args = None if prior is None else (prior._parameters(p.fit_drag), prior.covariance)
    batches = [api._batch(b) for b in problem.data.batches]
    try:
        x, cov, rss, it, rank, method, epoch = _native.calibrated(
            batches,
            config,
            guess,
            p.covariance,
            (
                p.max_iterations,
                p.gradient_tolerance,
                p.step_tolerance,
                p.cost_tolerance,
            ),
            args,
        )
    except _native.RankFailure as error:
        raise api.RankDeficientError(*error.args) from error
    except _native.CovarianceFailure as error:
        raise api.CovarianceError(str(error)) from error
    return _Fit(
        np.array(x),
        None if cov is None else np.array(cov),
        rss,
        it,
        epoch,
        dict(rank=rank, covariance_method=method),
    )


def _site(batch):
    s = batch.samples.station
    return (s.latitude_deg, s.longitude_deg, s.altitude_m)


def _covariances(batch):
    n = len(batch.samples.epochs)
    matrix = np.asarray(batch.covariance, dtype=float)
    if not np.isfinite(matrix).all() or np.any(
        np.diagonal(matrix, axis1=-2, axis2=-1) <= 0
    ):
        raise ValueError(
            "measurement covariance needs finite entries and positive variances"
        )
    return np.broadcast_to(matrix, (n, *matrix.shape[-2:]))


def _soft_noise(sigmas):
    if not sigmas:
        return (1.0, 1.0)
    if not np.allclose(sigmas, sigmas[0], atol=0, rtol=1e-12):
        raise ValueError(
            "contact-bias fit requires constant soft pointing uncertainties"
        )
    return sigmas[0]


def _contact_rows(batch, carriers, rates, angles, sigmas):
    if (
        batch.phase_difference_rad is not None
        or batch.differential_doppler_hz is not None
    ):
        raise ValueError("contact profiles accept only Doppler and pointing")
    matrices = _covariances(batch)
    m = matrices.shape[-1]
    if np.any(matrices[:, ~np.eye(m, dtype=bool)]):
        raise ValueError("contact profiles require independent rate and pointing")
    contact = batch.samples.contact_id
    carrier = batch.samples.carrier_hz
    if (
        batch.doppler_hz is not None
        and carriers.setdefault(contact, carrier) != carrier
    ):
        raise ValueError("contact profiles require one carrier per contact")
    for i, epoch in enumerate(batch.samples.epochs):
        meta = (str(contact), epoch, _site(batch))
        sigma = np.sqrt(np.diag(matrices[i]))
        if batch.doppler_hz is not None:
            rates.append(
                (*meta, -_C * batch.doppler_hz[i] / carrier, _C * sigma[0] / carrier)
            )
        if batch.azimuth_rad is not None:
            angles.append((*meta, batch.azimuth_rad[i], batch.elevation_rad[i]))
            sigmas.append(sigma[-2:])


def _contacts(problem, config, guess):
    p = problem.profile
    carriers, rates, angles, sigmas = {}, [], [], []
    for batch in problem.data.batches:
        _contact_rows(batch, carriers, rates, angles, sigmas)
    hard = isinstance(p, api.ConstrainedFit)
    n = p.pointings_per_contact if hard else None
    x, cov, bias, rss, it, epoch, violation = _native.contacts(
        config,
        guess,
        rates,
        angles,
        (1.0, 1.0) if hard else _soft_noise(sigmas),
        p.max_iterations,
        p.full_fov_rad if hard else None,
        None if n == "all" else n,
    )
    details = dict(
        contact_bias_hz={UUID(k): -v * carriers[UUID(k)] / _C for k, v in bias},
        max_pointing_violation_rad=violation,
        covariance_method=None if hard else "residual_scaled_schur",
    )
    return _Fit(
        np.array(x), None if cov is None else np.array(cov), rss, it, epoch, details
    )


def _rf_noise(covariance):
    count = len(covariance)
    expected = np.diag(np.diag(covariance))
    sigmas = np.ones(3)
    sigmas[:count] = np.sqrt(np.diag(covariance))
    if count == 3:
        expected[0, 2] = expected[2, 0] = expected[0, 0]
        if expected[2, 2] <= expected[0, 0]:
            raise ValueError("differential Doppler requires positive antenna-2 noise")
        sigmas[2] = np.sqrt(expected[2, 2] - expected[0, 0])
    if not np.allclose(expected, covariance, rtol=1e-12, atol=0):
        raise ValueError(
            "measurement covariance differs from the passive RF noise model"
        )
    return sigmas.tolist()


def _channels(batch):
    return [
        name
        for name in (
            "doppler_hz",
            "phase_difference_rad",
            "differential_doppler_hz",
            "azimuth_rad",
        )
        if getattr(batch, name) is not None
    ]


def _rf_columns(batches):
    if not batches:
        raise ValueError("passive RF fit requires observations")
    first = batches[0]
    carrier = first.samples.carrier_hz
    names = _channels(first)
    if names not in (
        ["doppler_hz"],
        ["doppler_hz", "phase_difference_rad"],
        ["doppler_hz", "phase_difference_rad", "differential_doppler_hz"],
    ):
        raise ValueError("unsupported passive RF channel combination")
    reference = _covariances(first)[0]
    for b in batches:
        present = _channels(b)
        if present != names or b.samples.carrier_hz != carrier:
            raise ValueError("passive RF fit requires common carrier and channels")
        if not np.allclose(_covariances(b), reference, rtol=1e-12, atol=0):
            raise ValueError("passive RF fit requires common measurement covariance")
    return names, carrier, _rf_noise(reference)


def _baselines(batches, name):
    arrays = []
    for b in batches:
        baseline = getattr(b, name)
        n = len(b.samples.epochs)
        arrays.append(
            np.broadcast_to([0.0, 0.0, 0.0] if baseline is None else baseline, (n, 3))
        )
    return np.vstack(arrays)


def _rf(problem, config, guess):
    p = problem.profile
    batches = problem.data.batches
    names, carrier, sigmas = _rf_columns(batches)
    epochs = [e for b in batches for e in b.samples.epochs]
    sites = [_site(b) for b in batches for _ in b.samples.epochs]
    columns = [np.concatenate([getattr(b, k) for b in batches]) for k in names]
    prior = problem.weighted_prior
    opts = dict(
        carrier=carrier,
        sigmas=sigmas,
        offset=[p.frequency_offset_hz.mean, p.frequency_offset_hz.sigma],
        phase=[p.phase_offset_rad.mean, p.phase_offset_rad.sigma],
        drift=[p.drift_correlation_s, p.drift_sigma_rad],
        baseline_sigma=p.phase_baseline_sigma_m,
        differential_sigma=p.differential_baseline_sigma_m,
        iterations=p.max_iterations,
    )
    values, P, rss, it, drift, epoch = _native.passive(
        config,
        guess,
        (prior._parameters(p.fit_drag), np.sqrt(np.diag(prior.covariance))),
        epochs,
        sites,
        columns,
        _baselines(batches, "phase_baseline_enu_m"),
        _baselines(batches, "differential_baseline_enu_m"),
        json.dumps(opts, default=lambda a: a.tolist()),
    )
    axes = list(prior.covariance_order) + ["frequency_offset_hz", "phase_offset_rad"]
    for name, sigma in (
        ("phase", p.phase_baseline_sigma_m),
        ("differential", p.differential_baseline_sigma_m),
    ):
        if sigma is not None:
            axes.extend(f"{name}_{axis}_m" for axis in ("east", "north", "up"))
    return _Fit(
        np.array(values),
        np.array(P),
        rss,
        it,
        epoch,
        dict(
            rf_parameters=dict(zip(axes, values, strict=True)),
            phase_drift_rad=np.array(drift),
            covariance_method="gaussian_posterior",
        ),
    )


def solve(problem, output_epoch):
    p = problem.profile
    seed = problem.initial_guess
    fit_drag = isinstance(p, api.ConstrainedFit) or p.fit_drag
    config = problem.force_model._config(seed.epoch, fit_drag)
    if fit_drag and seed.drag_m2_kg is None:
        seed = replace(
            seed, drag_m2_kg=problem.force_model.fixed_drag_m2_kg, covariance=None
        )
    guess = seed._parameters(fit_drag)
    handlers = {
        api.CalibratedLM: _calibrated,
        api.ContactBiasFit: _contacts,
        api.ConstrainedFit: _contacts,
        api.PassiveRFFit: _rf,
    }
    try:
        fit = handlers[type(p)](problem, config, guess)
    except _native.SolverError as error:
        raise api.FitError(str(error)) from error
    epoch = seed.epoch if output_epoch is None else output_epoch
    n = len(guess)
    _transport(fit, config, seed.epoch, epoch)
    if isinstance(p, api.PassiveRFFit):
        fit.details.update(
            rf_covariance=fit.covariance,
            rf_parameters=dict(zip(fit.details["rf_parameters"], fit.x, strict=True)),
        )
    estimate = api.StateEstimate(
        epoch,
        fit.x[:3].copy(),
        fit.x[3:6].copy(),
        covariance=None if fit.covariance is None else fit.covariance[:n, :n].copy(),
        drag_m2_kg=fit.x[6] if fit_drag else None,
        tle_initialization=seed.tle_initialization,
    )
    return api.FitResult(
        estimate=estimate,
        last_contributing_epoch=fit.evidence,
        used_weighted_prior=problem.weighted_prior is not None,
        whitened_rss=fit.rss,
        iterations=fit.iterations,
        **fit.details,
    )


def _transport(fit, config, previous, epoch):
    if _native.elapsed(previous, epoch) != 0:
        state, covariance = _native.transport(config, fit.x, epoch, fit.covariance)
        fit.x = np.asarray(state)
        fit.covariance = None if covariance is None else np.asarray(covariance)
