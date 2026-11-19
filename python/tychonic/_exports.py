"""Observation and ephemeris interchange in service-independent units."""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np

from . import _native

_COLUMNS = (
    "doppler_hz",
    "phase_difference_rad",
    "differential_doppler_hz",
    "azimuth_rad",
    "elevation_rad",
)


def to_polars(job):
    import polars as pl

    frames = []
    for index, batch in enumerate(job.batches):
        samples = batch.samples
        n = len(samples.epochs)
        cov = np.asarray(batch.covariance)
        table = dict(
            batch_index=[index] * n,
            spacecraft_id=str(job.spacecraft_id),
            contact_id=str(samples.contact_id),
            antenna_id=samples.station.antenna_id,
            epoch=list(samples.epochs),
            carrier_hz=samples.carrier_hz,
            observation_id=samples.observation_ids,
            lock_state=samples.lock_state,
            ebn0_db=samples.ebn0_db,
            latitude_deg=samples.station.latitude_deg,
            longitude_deg=samples.station.longitude_deg,
            altitude_m=samples.station.altitude_m,
            covariance=list(np.broadcast_to(cov, (n, *cov.shape[-2:])).tolist()),
        )
        table.update({k: getattr(batch, k) for k in _COLUMNS})
        for k in ("phase_baseline_enu_m", "differential_baseline_enu_m"):
            b = getattr(batch, k)
            table[k] = None if b is None else np.broadcast_to(b, (n, 3)).tolist()
        frames.append(pl.DataFrame(table))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _metadata(value):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.splitlines()) != 1
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError("CCSDS metadata requires nonempty single-line text")
    return value


def _utc(value):
    return _native.roundtrip_time(value).removesuffix("Z")


def _tdm_text(originator, creation_epoch, metadata, observations):
    from ._tdm import TDM, Header, Observation, Segment, dumps

    rows = []
    for epoch, value in sorted(observations):
        if not np.isfinite(value):
            raise ValueError("Doppler must be finite")
        rows.append(Observation("DOPPLER_INSTANTANEOUS", epoch, value))
    segment = Segment({k: _metadata(v) for k, v in metadata.items()}, rows)
    header = Header(originator=_metadata(originator), creation_date=creation_epoch)
    return dumps(TDM(header, [segment]), timestamp_precision=6)


def write_tdm(job, directory, originator, creation_epoch, spacecraft, participants):
    groups = defaultdict(list)
    for b in job.batches:
        if b.doppler_hz is None:
            continue
        s = b.samples
        km_s = -299792.458 * np.asarray(b.doppler_hz) / s.carrier_hz
        groups[s.contact_id, s.station.antenna_id].extend(
            zip(map(_utc, s.epochs), km_s.tolist(), strict=True)
        )
    texts = {}
    for (contact, antenna), observations in groups.items():
        path = Path(directory) / f"{contact}_{quote(antenna, safe='')}.tdm"
        meta = {
            "TIME_SYSTEM": "UTC",
            "PARTICIPANT_1": spacecraft,
            "PARTICIPANT_2": participants[antenna],
            "MODE": "SEQUENTIAL",
            "PATH": "1,2",
        }
        texts[path] = _tdm_text(originator, creation_epoch, meta, observations)
    for path in texts:
        if path.exists():
            raise FileExistsError(path)
    for path, content in texts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8"
        ) as temp:
            temp.write(content)
            temp.flush()
            os.link(temp.name, path)
    return tuple(texts)


def oem(estimate, epochs, forces, name, object_id, originator):
    times = [_utc(t) for t in epochs]
    if not times or times != sorted(set(times)):
        raise ValueError("OEM epochs must be nonempty and strictly chronological")
    states = np.asarray(_native.arc(*estimate._export_args(epochs, forces))) / 1000
    creation = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")
    lines = [
        "CCSDS_OEM_VERS = 2.0",
        f"CREATION_DATE = {creation}",
        f"ORIGINATOR = {_metadata(originator)}",
        "META_START",
        f"OBJECT_NAME = {_metadata(name)}",
        f"OBJECT_ID = {_metadata(object_id)}",
        "CENTER_NAME = EARTH",
        "REF_FRAME = GCRF",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {times[0]}",
        f"STOP_TIME = {times[-1]}",
        "META_STOP",
    ]
    lines.extend(
        t + " " + " ".join(format(v, ".17g") for v in row)
        for t, row in zip(times, states, strict=True)
    )
    return "\n".join(lines) + "\n"
