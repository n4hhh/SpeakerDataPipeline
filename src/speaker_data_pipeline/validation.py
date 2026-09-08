"""Complete invariant validation for processed speaker-audio output datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

from .audio_io import AudioIOError, read_wav
from .provenance import NORMALIZATION_METADATA_FILES, write_dict_csv, write_json_atomic


class ValidationError(RuntimeError):
    """Raised when validation cannot safely start or publish reports."""


@dataclass(frozen=True)
class ValidationResult:
    """Validation outcome and generated report locations."""

    passed: bool
    summary: dict[str, Any]
    summary_path: Path
    failures_path: Path


FAILURE_FIELDS = ("relative_path", "check", "expected", "actual", "error")


def _is_inside_or_equal(candidate: Path, parent: Path) -> bool:
    candidate_text = os.path.normcase(str(candidate.resolve(strict=False)))
    parent_text = os.path.normcase(str(parent.resolve(strict=False)))
    try:
        return os.path.commonpath((candidate_text, parent_text)) == parent_text
    except ValueError:
        return False


def _failure(
    relative_path: str,
    check: str,
    expected: object = "",
    actual: object = "",
    error: str = "",
) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "check": check,
        "expected": expected,
        "actual": actual,
        "error": error,
    }


def _casefold_duplicates(paths: list[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        groups[path.casefold()].append(path)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def validate_dataset(
    input_root: Path | str,
    output_dir: Path | str,
    target_sample_rate: int,
    target_channels: int,
    target_duration_seconds: float,
    source_root: Path | str | None = None,
) -> ValidationResult:
    """Validate every output WAV and optionally enforce source path-set equality."""
    processed_root = Path(input_root).expanduser().resolve(strict=True)
    if not processed_root.is_dir():
        raise ValidationError(f"input root is not a directory: {processed_root}")
    report_dir = Path(output_dir).expanduser().resolve(strict=False)
    if _is_inside_or_equal(report_dir, processed_root):
        raise ValidationError("validation output directory must not be inside the input dataset")
    if target_sample_rate <= 0:
        raise ValueError("target sample rate must be positive")
    if target_channels < 1:
        raise ValueError("target channels must be at least 1")
    if not np.isfinite(target_duration_seconds) or target_duration_seconds <= 0:
        raise ValueError("target duration must be finite and positive")
    target_samples = round(target_sample_rate * target_duration_seconds)
    if target_samples <= 0:
        raise ValueError("configured target duration produces no samples")

    source_path: Path | None = None
    if source_root is not None:
        source_path = Path(source_root).expanduser().resolve(strict=True)
        if not source_path.is_dir():
            raise ValidationError(f"source root is not a directory: {source_path}")
        if _is_inside_or_equal(report_dir, source_path):
            raise ValidationError("validation output directory must not be inside the source dataset")

    all_files = sorted(
        (path for path in processed_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(processed_root).as_posix(),
    )
    wav_files = [path for path in all_files if path.suffix.lower() == ".wav"]
    relative_wavs = [path.relative_to(processed_root).as_posix() for path in wav_files]
    failures: list[dict[str, object]] = []
    readable_count = 0
    zero_frame_count = 0

    non_wav_files = []
    for path in all_files:
        if path.suffix.lower() == ".wav":
            continue
        relative = path.relative_to(processed_root).as_posix()
        if relative not in NORMALIZATION_METADATA_FILES:
            non_wav_files.append(relative)
            failures.append(_failure(relative, "unexpected_non_wav", "no unexpected file", "present"))

    for duplicate_group in _casefold_duplicates(relative_wavs):
        failures.append(
            _failure(" | ".join(duplicate_group), "duplicate_relative_path", "unique", "duplicate")
        )

    for path, relative_text in zip(wav_files, relative_wavs):
        relative = path.relative_to(processed_root)
        if len(relative.parts) == 1:
            failures.append(
                _failure(relative_text, "structure", "speaker/WAV", "root-level WAV")
            )
        elif len(relative.parts) > 2:
            failures.append(
                _failure(relative_text, "structure", "speaker/WAV", "deeper nesting")
            )
        if path.is_symlink():
            failures.append(_failure(relative_text, "symbolic_link", "regular file", "symbolic link"))
            continue
        try:
            audio = read_wav(path)
            metadata = audio.metadata
            readable_count += 1
        except (AudioIOError, OSError) as exc:
            failures.append(
                _failure(relative_text, "readable", "readable WAV", "unreadable", str(exc))
            )
            continue
        if metadata.frames == 0:
            zero_frame_count += 1
            failures.append(_failure(relative_text, "zero_frames", "> 0", 0))
        if metadata.sample_rate != target_sample_rate:
            failures.append(
                _failure(relative_text, "sample_rate", target_sample_rate, metadata.sample_rate)
            )
        if metadata.channels != target_channels:
            failures.append(_failure(relative_text, "channels", target_channels, metadata.channels))
        if metadata.frames != target_samples:
            failures.append(_failure(relative_text, "sample_count", target_samples, metadata.frames))
        if metadata.subtype != "PCM_16":
            failures.append(_failure(relative_text, "subtype", "PCM_16", metadata.subtype))
        duration_tolerance = 0.5 / target_sample_rate + np.finfo(float).eps
        if abs(metadata.duration_seconds - target_duration_seconds) > duration_tolerance:
            failures.append(
                _failure(
                    relative_text,
                    "duration_seconds",
                    target_duration_seconds,
                    metadata.duration_seconds,
                )
            )

    missing_source_paths: list[str] = []
    extra_output_paths: list[str] = []
    source_wav_count: int | None = None
    if source_path is not None:
        source_wavs = sorted(
            path.relative_to(source_path).as_posix()
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        )
        source_wav_count = len(source_wavs)
        for duplicate_group in _casefold_duplicates(source_wavs):
            failures.append(
                _failure(
                    " | ".join(duplicate_group),
                    "source_duplicate_relative_path",
                    "unique",
                    "duplicate",
                )
            )
        source_set = set(source_wavs)
        output_set = set(relative_wavs)
        missing_source_paths = sorted(source_set - output_set)
        extra_output_paths = sorted(output_set - source_set)
        failures.extend(
            _failure(path, "source_path_set", "present in output", "missing")
            for path in missing_source_paths
        )
        failures.extend(
            _failure(path, "source_path_set", "present in source", "extra in output")
            for path in extra_output_paths
        )

    failures.sort(key=lambda row: (str(row["relative_path"]), str(row["check"])))
    summary: dict[str, Any] = {
        "passed": not failures,
        "total_wav_files": len(wav_files),
        "readable_wav_files": readable_count,
        "zero_frame_count": zero_frame_count,
        "unexpected_non_wav_count": len(non_wav_files),
        "failure_count": len(failures),
        "expected": {
            "sample_rate": target_sample_rate,
            "channels": target_channels,
            "samples": target_samples,
            "duration_seconds": target_duration_seconds,
            "subtype": "PCM_16",
        },
        "source_path_comparison": {
            "enabled": source_path is not None,
            "source_wav_count": source_wav_count,
            "missing_output_count": len(missing_source_paths),
            "extra_output_count": len(extra_output_paths),
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_json_atomic(report_dir / "validation_summary.json", summary)
    failures_path = write_dict_csv(
        report_dir / "validation_failures.csv", failures, FAILURE_FIELDS
    )
    return ValidationResult(not failures, summary, summary_path, failures_path)
