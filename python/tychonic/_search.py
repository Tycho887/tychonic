"""Translate catalogue and particle inputs to the native acquisition planner."""

import json
from dataclasses import asdict

import numpy as np

from . import _native, api


def _ephemeris(value):
    return "\n".join(value) if isinstance(value, tuple) else json.dumps(value)


def _hypotheses(finder):
    if finder.hypotheses:
        return [_ephemeris(p) for p in finder.hypotheses], finder.weights
    count = finder.n_particles
    distribution = finder.distribution
    if callable(distribution):
        offsets = distribution(finder.rng, count)
    else:
        offsets = finder.rng.normal(
            scale=list(asdict(distribution).values()), size=(count, 4)
        )
    offsets = np.asarray(offsets, dtype=float)
    if offsets.shape != (count, 4) or not np.isfinite(offsets).all():
        raise ValueError("particle sampler must return finite (count, 4) offsets")
    return _native.particles(_ephemeris(finder.nominal), offsets), np.ones(count)


def controller(finder):
    setup = finder.setup
    s = setup.station
    settings = {
        k: getattr(setup, k)
        for k in (
            "search_duration_s",
            "move_s",
            "settle_s",
            "listen_s",
            "horizon",
            "beam_width",
            "candidate_count",
            "candidate_extent_sigma",
        )
    }
    settings["ess_threshold"] = setup.resample_ess_fraction
    direction = None
    if setup.initial_az_el_rad is not None:
        az, el = setup.initial_az_el_rad
        direction = [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)]
    hypotheses, weights = _hypotheses(finder)
    return _native.NativeBCMPC(
        _ephemeris(finder.nominal),
        hypotheses,
        weights,
        json.dumps(
            dict(
                station=[s.latitude_deg, s.longitude_deg, s.altitude_m],
                start=setup.start,
                config=settings,
                antenna=dict(
                    rate_rad_s=setup.rate_rad_s,
                    acceleration_rad_s2=setup.acceleration_rad_s2,
                ),
                elevation_mask_rad=setup.elevation_mask_rad,
                initial_direction=direction,
            )
        ),
    )


def plan(finder):
    actions, probability, _ = finder._controller.plan_receiver(
        list(asdict(finder.setup.receiver).values())
    )

    def action(row):
        _, (e, n, u), start, end, _ = row
        return api.SearchAction(
            np.arctan2(e, n) % (2 * np.pi), np.arctan2(u, np.hypot(e, n)), start, end
        )

    return api.SearchPlan(tuple(action(row) for row in actions), probability)


def state(finder):
    status, cycle, weights, indices, _, _ = finder._controller.snapshot()
    return api.SearchState(
        status, cycle, np.array(weights), np.array(indices, dtype=np.intp)
    )
