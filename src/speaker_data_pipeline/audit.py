"""Read-only inventory and compatibility auditing for speaker-folder WAV data."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from .audio_io import AudioIOError, inspect_wav
from .provenance import write_dict_csv, write_json_atomic


class AuditError(RuntimeError):
    """Raised when an audit cannot safely start or publish its reports."""


@dataclass(frozen=True)
class AuditResult:
    """Audit summary and generated report locations."""

    summary: dict[str, Any]
    summary_path: Path
    speaker_summary_path: Path
    inventory_path: Path


INVENTORY_FIELDS = (
    "relative_path",
    "speaker_id",
    "sample_rate",
    "channels",
    "frames",
    "duration_seconds",
    "subtype",
    "format",
    "file_size_bytes",
    "status",
    "error",
)

SPEAKER_FIELDS = (
    "speaker_id",
    "wav_count",
    "readable_count",
    "unreadable_count",
    "total_duration_seconds",
)


def _is_inside_or_equal(candidate: Path, parent: Path) -> bool:
    candidate_text = os.path.normcase(str(candidate.resolve(strict=False)))
    parent_text = os.path.normcase(str(parent.resolve(strict=False)))
    try:
        return os.path.commonpath((candidate_text, parent_text)) == parent_text
    except ValueError:
        return False


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "minimum": None,
            "mean": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
        }
    data = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(data)),
        "mean": float(np.mean(data)),
        "p25": float(np.percentile(data, 25)),
        "median": float(np.median(data)),
        "p75": float(np.percentile(data, 75)),
        "p95": float(np.percentile(data, 95)),
        "maximum": float(np.max(data)),
    }


def audit_dataset(
    input_root: Path | str,
    output_dir: Path | str,
    target_duration_seconds: float | None = None,
) -> AuditResult:
    """Inspect WAV metadata and structure without decoding or changing source audio."""
    source_root = Path(input_root).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise AuditError(f"input root is not a directory: {source_root}")
    report_dir = Path(output_dir).expanduser().resolve(strict=False)
    if _is_inside_or_equal(report_dir, source_root):
        raise AuditError("audit output directory must not be inside the input dataset")
    if target_duration_seconds is not None and (
        not np.isfinite(target_duration_seconds) or target_duration_seconds <= 0
    ):
        raise ValueError("target duration must be finite and positive")

    all_files = sorted(
        (path for path in source_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    wav_files = [path for path in all_files if path.suffix.lower() == ".wav"]
    non_wav_files = [path for path in all_files if path.suffix.lower() != ".wav"]
    immediate_speaker_dirs = sorted(
        (path for path in source_root.iterdir() if path.is_dir()), key=lambda path: path.name
    )

    inventory: list[dict[str, Any]] = []
    durations: list[float] = []
    sample_rates: Counter[int] = Counter()
    channel_counts: Counter[int] = Counter()
    speaker_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unreadable_paths: list[str] = []
    zero_frame_paths: list[str] = []
    root_level_paths: list[str] = []
    deeper_paths: list[str] = []
    exact_count = shorter_count = longer_count = 0

    for path in wav_files:
        relative = path.relative_to(source_root)
        relative_text = relative.as_posix()
        speaker_id = path.parent.name
        if len(relative.parts) == 1:
            root_level_paths.append(relative_text)
        elif len(relative.parts) > 2:
            deeper_paths.append(relative_text)
        try:
            file_size = path.stat().st_size
        except OSError:
            file_size = None
        row: dict[str, Any] = {
            "relative_path": relative_text,
            "speaker_id": speaker_id,
            "sample_rate": "",
            "channels": "",
            "frames": "",
            "duration_seconds": "",
            "subtype": "",
            "format": "",
            "file_size_bytes": file_size if file_size is not None else "",
            "status": "unreadable",
            "error": "",
        }
        try:
            metadata = inspect_wav(path)
            row.update(
                sample_rate=metadata.sample_rate,
                channels=metadata.channels,
                frames=metadata.frames,
                duration_seconds=metadata.duration_seconds,
                subtype=metadata.subtype,
                format=metadata.format,
                status="readable",
            )
            durations.append(metadata.duration_seconds)
            sample_rates[metadata.sample_rate] += 1
            channel_counts[metadata.channels] += 1
            if metadata.frames == 0:
                zero_frame_paths.append(relative_text)
            if target_duration_seconds is not None:
                target_frames = round(metadata.sample_rate * target_duration_seconds)
                if metadata.frames == target_frames:
                    exact_count += 1
                elif metadata.frames < target_frames:
                    shorter_count += 1
                else:
                    longer_count += 1
        except (AudioIOError, OSError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            unreadable_paths.append(relative_text)
        inventory.append(row)
        speaker_rows[speaker_id].append(row)

    per_speaker: list[dict[str, Any]] = []
    for speaker_id in sorted(speaker_rows):
        rows = speaker_rows[speaker_id]
        readable = [row for row in rows if row["status"] == "readable"]
        per_speaker.append(
            {
                "speaker_id": speaker_id,
                "wav_count": len(rows),
                "readable_count": len(readable),
                "unreadable_count": len(rows) - len(readable),
                "total_duration_seconds": sum(float(row["duration_seconds"]) for row in readable),
            }
        )

    utterance_counts = [int(row["wav_count"]) for row in per_speaker]
    empty_speaker_dirs = [
        directory.relative_to(source_root).as_posix()
        for directory in immediate_speaker_dirs
        if not any(directory.iterdir())
    ]
    total_bytes = 0
    stat_failure_paths: list[str] = []
    for path in all_files:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            stat_failure_paths.append(path.relative_to(source_root).as_posix())

    speaker_statistics: dict[str, int | float | None] = {
        "total_speakers": len(per_speaker),
        "utterances_minimum": min(utterance_counts) if utterance_counts else None,
        "utterances_mean": mean(utterance_counts) if utterance_counts else None,
        "utterances_p25": float(np.percentile(utterance_counts, 25)) if utterance_counts else None,
        "utterances_median": median(utterance_counts) if utterance_counts else None,
        "utterances_p75": float(np.percentile(utterance_counts, 75)) if utterance_counts else None,
        "utterances_maximum": max(utterance_counts) if utterance_counts else None,
    }
    summary: dict[str, Any] = {
        "dataset": {
            "total_files": len(all_files),
            "total_wav_files": len(wav_files),
            "non_wav_file_count": len(non_wav_files),
            "total_speaker_folders": len(immediate_speaker_dirs),
            "total_bytes": total_bytes,
        },
        "audio_distributions": {
            "sample_rate_counts": {str(key): sample_rates[key] for key in sorted(sample_rates)},
            "channel_counts": {str(key): channel_counts[key] for key in sorted(channel_counts)},
            "duration_seconds": _distribution(durations),
        },
        "speaker_statistics": speaker_statistics,
        "structural_issues": {
            "root_level_wavs": root_level_paths,
            "unexpected_deeper_wavs": deeper_paths,
            "empty_speaker_directories": empty_speaker_dirs,
            "unreadable_wavs": unreadable_paths,
            "zero_frame_wavs": zero_frame_paths,
            "non_wav_files": [path.relative_to(source_root).as_posix() for path in non_wav_files],
            "file_stat_failures": stat_failure_paths,
        },
    }
    if target_duration_seconds is not None:
        summary["target_duration_comparison"] = {
            "target_duration_seconds": target_duration_seconds,
            "exactly_target_count": exact_count,
            "shorter_than_target_count": shorter_count,
            "longer_than_target_count": longer_count,
        }

    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_json_atomic(report_dir / "audit_summary.json", summary)
    speaker_path = write_dict_csv(report_dir / "speaker_summary.csv", per_speaker, SPEAKER_FIELDS)
    inventory_path = write_dict_csv(report_dir / "file_inventory.csv", inventory, INVENTORY_FIELDS)
    return AuditResult(summary, summary_path, speaker_path, inventory_path)
