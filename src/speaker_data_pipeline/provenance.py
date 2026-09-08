"""Portable transformation provenance records and atomic report writers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


PROVENANCE_FILENAME = "normalization_provenance.csv"
SUMMARY_FILENAME = "normalization_summary.json"
NORMALIZATION_METADATA_FILES = frozenset({PROVENANCE_FILENAME, SUMMARY_FILENAME})


@dataclass(frozen=True)
class ProvenanceRecord:
    """One deterministic, dataset-relative normalization result."""

    relative_path: str
    speaker_id: str
    source_sample_rate: int | None
    source_channels: int | None
    source_frames: int | None
    source_duration_seconds: float | None
    normalized_duration_seconds: float | None
    target_sample_rate: int
    target_channels: int
    target_samples: int
    target_duration_seconds: float
    resampled: bool | None
    channel_converted: bool | None
    operation: str
    crop_start_seconds: float | None = None
    crop_end_seconds: float | None = None
    padding_left_seconds: float | None = None
    padding_right_seconds: float | None = None
    selected_window_speech_ratio: float | None = None
    best_window_speech_ratio: float | None = None
    candidate_window_count: int | None = None
    eligible_candidate_count: int | None = None
    vad_status: str = "not_applicable"
    seed: int = 0
    status: str = "processed"
    error: str = ""


PROVENANCE_FIELDS = tuple(ProvenanceRecord.__dataclass_fields__)


def _temporary_sibling(path: Path, suffix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent, delete=False
    ) as temporary:
        return Path(temporary.name)


def write_provenance_csv(
    path: Path | str, records: Iterable[ProvenanceRecord]
) -> Path:
    """Write records in portable relative-path order as UTF-8 CSV."""
    target = Path(path)
    ordered = sorted(records, key=lambda record: record.relative_path)
    temporary = _temporary_sibling(target, ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PROVENANCE_FIELDS, lineterminator="\n")
            writer.writeheader()
            for record in ordered:
                writer.writerow(asdict(record))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_json_atomic(path: Path | str, payload: dict[str, Any]) -> Path:
    """Write stable UTF-8 JSON through a sibling temporary file."""
    target = Path(path)
    temporary = _temporary_sibling(target, ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_dict_csv(
    path: Path | str, rows: Iterable[dict[str, Any]], fieldnames: Iterable[str]
) -> Path:
    """Write dictionary rows atomically with explicit column order."""
    target = Path(path)
    temporary = _temporary_sibling(target, ".tmp")
    columns = tuple(fieldnames)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
