"""Per-object filter ownership, durable reading identities and checkpoint v1."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from threading import RLock
from uuid import UUID

import numpy as np

from . import _native, api


@dataclass
class _Entry:
    native: object
    model: object
    evidence: str | None
    processed: api.ReadingRef | None = None
    accepted: api.ReadingRef | None = None
    readings: dict = field(default_factory=dict)
    lock: RLock = field(default_factory=RLock)


def _insert(cat, key, entry):
    with cat._lock:
        if key in cat._entries:
            raise ValueError(f"spacecraft {key} already exists")
        cat._entries[key] = entry


@contextmanager
def _locked_entry(cat, key):
    # Locking the entry before rechecking membership permits other objects to progress.
    with cat._lock:
        entry = cat._entries[key]
    with entry.lock:
        with cat._lock:
            if cat._entries.get(key) is not entry:
                raise KeyError(key)
        yield entry


def _validate(cat):
    if cat.gate is not None and (not np.isfinite(cat.gate) or cat.gate < 0):
        raise ValueError("gate must be finite and nonnegative")
    noise = list(asdict(cat.noise).values())
    if not np.all(np.isfinite(noise)) or min(noise) < 0:
        raise ValueError("invalid process noise")


def add_orbit(cat, key, initial, prior, evidence):
    _validate(cat)
    if initial.covariance is None:
        raise api.CovarianceError("an EKF seed requires covariance")
    model = cat.force_model
    if initial.drag_m2_kg is not None:
        model = replace(model, fixed_drag_m2_kg=initial.drag_m2_kg)
    x = np.empty(7)
    x[:6] = initial._parameters()
    if initial.doppler_bias_hz is None:
        if prior is None or not np.isfinite(prior.sigma) or prior.sigma <= 0:
            raise ValueError("a positive independent Doppler bias prior is required")
        x[6] = prior.mean
        p = np.zeros((7, 7))
        p[:6, :6] = initial.covariance[:6, :6]
        p[6, 6] = prior.sigma**2
    else:
        if prior is not None:
            raise ValueError("estimate already carries a Doppler bias")
        x[6] = initial.doppler_bias_hz
        axes = [0, 1, 2, 3, 4, 5, len(initial.covariance) - 1]
        p = np.asarray(initial.covariance)[np.ix_(axes, axes)]
    native = _native.NativeOrbitEKF(x, p, model._config(initial.epoch), True)
    _insert(cat, key, _Entry(native, model, evidence))


def add_tle(cat, key, initial):
    _validate(cat)
    caps = np.array([cat.noise.max_offset_sigma_s, cat.noise.max_bias_sigma_hz])
    if (caps <= 0).any() or (np.diag(initial.covariance) > caps**2).any():
        raise ValueError("TCA initial uncertainty exceeds its configured cap")
    _insert(cat, key, _Entry(initial._filter(), initial.reference_tle, None))


def _is_orbit(cat):
    return isinstance(cat, api.OrbitEKFCatalogue)


def _orbit(x, p, epoch):
    x = np.array(x)
    return api.StateEstimate(
        epoch, x[:3], x[3:6], doppler_bias_hz=x[6], covariance=np.array(p)
    )


def get(cat, key):
    with _locked_entry(cat, key) as entry:
        x, p, epoch = entry.native.snapshot()
        if _is_orbit(cat):
            return _orbit(x, p, epoch)
        return api.TleCorrection(
            entry.model,
            epoch,
            epoch_offset_s=x[0],
            doppler_bias_hz=x[1],
            covariance=np.array(p),
        )


def drop(cat, key):
    with _locked_entry(cat, key), cat._lock:
        del cat._entries[key]


def update(cat, data):
    batches = [api._batch(b) for b in data.batches]
    for batch in data.batches:
        if batch.samples.observation_ids is None:
            raise ValueError("streamed updates require stable observation_ids")
    with _locked_entry(cat, data.spacecraft_id) as entry:
        if _is_orbit(cat):
            rows, error = entry.native.update_bias(
                batches,
                cat.noise.acceleration_psd_m2_s3,
                cat.noise.bias_psd_hz2_s,
                cat.retain_history,
                cat.gate,
            )
        else:
            rows, error = entry.native.update_hz(
                batches, list(asdict(cat.noise).values()), cat.gate
            )
        updates = tuple(_record(cat, entry, data, row) for row in rows)
        if error is not None:
            error.last_processed_reading = entry.processed
            raise error
        return updates


def _record(cat, entry, data, receipt):
    index, row, epoch, nis, accepted = receipt
    samples = data.batches[index].samples
    reading = api.ReadingRef(
        samples.contact_id,
        samples.station.antenna_id,
        samples.observation_ids[row],
        epoch,
    )
    event = api.FilterUpdate(index, row, reading, accepted, nis)
    entry.processed = reading
    if accepted:
        entry.accepted = reading
        if entry.evidence is None or _native.elapsed(entry.evidence, epoch) >= 0:
            entry.evidence = epoch
    if _is_orbit(cat) and cat.retain_history:
        entry.readings.setdefault(epoch, []).append(event)
    return event


def predict(cat, epoch, ids):
    with cat._lock:
        keys = tuple(cat._entries if ids is None else ids)
    for key in keys:
        with _locked_entry(cat, key) as entry:
            if _is_orbit(cat):
                dt = _native.elapsed(entry.native.snapshot()[2], epoch)
                q = _native.white_acceleration_noise(
                    dt,
                    [cat.noise.acceleration_psd_m2_s3] * 3,
                    cat.noise.bias_psd_hz2_s,
                    True,
                )
                entry.native.predict_to(epoch, np.array(q))
                if not cat.retain_history:
                    entry.native.discard_history()
            else:
                entry.native.predict_bounded(epoch, list(asdict(cat.noise).values()))


def replace_tle(cat, key, tle, epoch, prior):
    with _locked_entry(cat, key) as entry:
        state = get(cat, key)
        candidate = state._filter()
        candidate.predict_bounded(epoch, list(asdict(cat.noise).values()))
        x, p, _ = candidate.snapshot()
        if not 0 < prior.sigma <= cat.noise.max_offset_sigma_s:
            raise ValueError("offset prior must have positive sigma within the cap")
        replacement = api.TleCorrection(
            tle,
            epoch,
            epoch_offset_s=prior.mean,
            doppler_bias_hz=x[1],
            covariance=np.diag([prior.sigma**2, p[1][1]]),
        )
        entry.native = replacement._filter()
        entry.model = tle


def smooth(cat, key):
    with _locked_entry(cat, key) as entry:
        if not cat.retain_history:
            raise ValueError("smoothing requires retain_history=True")
        epochs, states, covariances = entry.native.smooth()
        return tuple(
            api.SmoothedStep(_orbit(x, p, t), tuple(entry.readings.get(t, ())))
            for t, x, p in zip(epochs, states, covariances, strict=True)
        )


def _settings(cat):
    settings = {"noise": asdict(cat.noise), "gate": cat.gate}
    if _is_orbit(cat):
        settings.update(
            force_model=asdict(cat.force_model), retain_history=cat.retain_history
        )
    return settings


def checkpoint(cat, key):
    with _locked_entry(cat, key) as entry:
        x, p, epoch = entry.native.snapshot()
        payload = dict(
            settings=_settings(cat),
            x=x,
            p=p,
            model=asdict(entry.model) if _is_orbit(cat) else entry.model,
        )
        if _is_orbit(cat) and cat.retain_history:
            payload.update(
                steps=entry.native.saved_steps(),
                readings={
                    t: [asdict(r) for r in rs] for t, rs in entry.readings.items()
                },
            )
        return api.FilterCheckpoint(
            version=1,
            kind="orbit_ekf" if _is_orbit(cat) else "tle_epoch_ukf",
            spacecraft_id=key,
            estimate_epoch=epoch,
            last_contributing_epoch=entry.evidence,
            last_processed_reading=entry.processed,
            last_accepted_observation=entry.accepted,
            payload=json.dumps(payload, allow_nan=False, default=str).encode(),
        )


def _event(value):
    reading = value["reading"]
    reading = api.ReadingRef(**{**reading, "contact_id": UUID(reading["contact_id"])})
    return api.FilterUpdate(**{**value, "reading": reading})


def restore(cat, checkpoint):
    kind = "orbit_ekf" if _is_orbit(cat) else "tle_epoch_ukf"
    if checkpoint.version != 1 or checkpoint.kind != kind:
        raise ValueError("incompatible checkpoint version or filter kind")
    payload = json.loads(checkpoint.payload)
    if payload["settings"] != _settings(cat):
        raise ValueError("checkpoint settings differ from catalogue settings")
    x, p, epoch = (
        np.array(payload["x"]),
        np.array(payload["p"]),
        checkpoint.estimate_epoch,
    )
    native, model = _restore_native(cat, payload, x, p, epoch)
    entry = _Entry(
        native,
        model,
        checkpoint.last_contributing_epoch,
        checkpoint.last_processed_reading,
        checkpoint.last_accepted_observation,
    )
    if _is_orbit(cat) and cat.retain_history:
        native.restore_steps([tuple(step) for step in payload["steps"]])
        entry.readings = {
            t: [_event(v) for v in values] for t, values in payload["readings"].items()
        }
    _insert(cat, checkpoint.spacecraft_id, entry)


def _restore_native(cat, payload, x, p, epoch):
    if _is_orbit(cat):
        model = api.ForceModel(**payload["model"])
        native = _native.NativeOrbitEKF(x, p, model._config(epoch), True)
    else:
        model = tuple(payload["model"])
        native = api.TleCorrection(
            model, epoch, epoch_offset_s=x[0], doppler_bias_hz=x[1], covariance=p
        )._filter()
    return native, model
