"""KVN generation copied from TDM-writer a1eed59, limited to the writer path."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

import pandas as pd

Scalar = str | int | float | Decimal | bool


@dataclass(slots=True)
class ValidationIssue:
    """One validation problem associated with a document path."""

    path: str
    message: str
    code: str = "invalid"
    severity: Literal["error", "warning"] = "error"


@dataclass(slots=True)
class Header:
    """The TDM header preceding the first metadata segment."""

    version: str = "2.0"
    creation_date: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp.now(tz="UTC").floor("s")
    )
    originator: str = "UNKNOWN"
    comments: list[str] = field(default_factory=list)
    extensions: MutableMapping[str, Scalar] = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        self.creation_date = normalize_timestamp(
            self.creation_date, assume_timezone="UTC"
        )
        self.extensions = OrderedDict(self.extensions)


@dataclass(slots=True)
class Observation:
    """A single timestamped TDM observation."""

    keyword: str
    epoch: pd.Timestamp
    value: Scalar
    raw_value: str | None = None
    commented: bool = False

    def __post_init__(self) -> None:
        self.keyword = self.keyword.upper()
        self.epoch = normalize_timestamp(self.epoch, assume_timezone="UTC")


@dataclass(slots=True)
class Segment:
    metadata: Mapping[str, Scalar]
    observations: list[Observation]
    metadata_comments: tuple = ()
    data_comments: tuple = ()

    def serialized_observations(self):
        return self.observations


@dataclass(slots=True)
class TDM:
    header: Header
    segments: list[Segment]


class TDMError(Exception):
    """Invalid TDM document."""


class TDMValidationError(TDMError, ValueError):
    """Raised when a document fails requested validation."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        message = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        super().__init__(message or "TDM validation failed")


def normalize_timestamp(
    value: Any, *, assume_timezone: str | None = None
) -> pd.Timestamp:
    """Convert a timestamp-like value to a timezone-aware UTC Timestamp."""

    timestamp = pd.Timestamp(value)
    if not isinstance(timestamp, pd.Timestamp):
        raise ValueError("timestamp cannot be NaT")
    if timestamp.tzinfo is None:
        if assume_timezone is None:
            raise ValueError("naive timestamps require assume_timezone")
        timestamp = timestamp.tz_localize(assume_timezone)
    return timestamp.tz_convert("UTC")


def validate_structure(document: TDM) -> list[ValidationIssue]:
    """Validate the format-neutral invariants needed to write valid KVN."""

    issues: list[ValidationIssue] = []
    if not document.header.version:
        issues.append(ValidationIssue("header.version", "CCSDS_TDM_VERS is required"))
    if not document.header.originator:
        issues.append(ValidationIssue("header.originator", "ORIGINATOR is required"))
    if not document.segments:
        issues.append(ValidationIssue("segments", "at least one segment is required"))
    for index, segment in enumerate(document.segments):
        if not segment.metadata:
            issues.append(
                ValidationIssue(
                    f"segments[{index}].metadata", "metadata cannot be empty"
                )
            )
        for keyword in segment.metadata:
            if not _valid_keyword(keyword):
                issues.append(
                    ValidationIssue(
                        f"segments[{index}].metadata.{keyword}", "invalid KVN keyword"
                    )
                )
        for observation in segment.serialized_observations():
            if not _valid_keyword(observation.keyword):
                issues.append(
                    ValidationIssue(
                        f"segments[{index}].data.{observation.keyword}",
                        "invalid KVN keyword",
                    )
                )
    return issues


def _valid_keyword(keyword: str) -> bool:
    return bool(keyword) and all(
        character.isupper() or character.isdigit() or character == "_"
        for character in keyword
    )


def dumps(document: TDM, *, timestamp_precision: int | None = None) -> str:
    """Serialize a document to deterministic KVN text."""

    issues = validate_structure(document)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise TDMValidationError(errors)

    lines = [f"CCSDS_TDM_VERS = {document.header.version}"]
    lines.extend(f"COMMENT {comment}" for comment in document.header.comments)
    created = _format_epoch(document.header.creation_date, timestamp_precision)
    lines.append(f"CREATION_DATE = {created}")
    lines.append(f"ORIGINATOR = {document.header.originator}")
    lines.extend(
        f"{keyword} = {_format_scalar(value)}"
        for keyword, value in document.header.extensions.items()
    )

    for segment in document.segments:
        lines.extend(_segment_lines(segment, timestamp_precision))
    return "\n".join(lines) + "\n"


def _segment_lines(segment, timestamp_precision):
    lines = []
    lines.append("META_START")
    lines.extend(f"COMMENT {comment}" for comment in segment.metadata_comments)
    lines.extend(
        f"{keyword} = {_format_scalar(value)}"
        for keyword, value in segment.metadata.items()
    )
    lines.append("META_STOP")
    lines.append("DATA_START")
    lines.extend(f"COMMENT {comment}" for comment in segment.data_comments)
    for observation in segment.serialized_observations():
        prefix = "COMMENT " if observation.commented else ""
        value = observation.raw_value or _format_scalar(observation.value)
        lines.append(
            f"{prefix}{observation.keyword} = "
            f"{_format_epoch(observation.epoch, timestamp_precision)}    {value}"
        )
    lines.append("DATA_STOP")
    return lines


def _format_epoch(timestamp: Any, precision: int | None) -> str:
    value = normalize_timestamp(timestamp, assume_timezone="UTC").tz_localize(None)
    base = value.strftime("%Y-%m-%dT%H:%M:%S")
    fraction = value.microsecond * 1_000 + value.nanosecond
    if precision is not None:
        if not 0 <= precision <= 9:
            raise ValueError("timestamp_precision must be between 0 and 9")
        if precision == 0:
            return base
        return f"{base}.{fraction:09d}"[: len(base) + 1 + precision]
    if fraction == 0:
        return base
    return f"{base}.{fraction:09d}".rstrip("0")


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        return repr(value)
    return str(value)
