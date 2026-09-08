"""Deterministic, separate-output speaker-audio normalization."""

from __future__ import annotations

from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random
from typing import Callable

import numpy as np

from .audio_io import (
    AudioMetadata,
    convert_channels,
    inspect_wav,
    read_wav,
    resample_waveform,
    write_wav_pcm16_atomic,
)
from .provenance import (
    NORMALIZATION_METADATA_FILES,
    PROVENANCE_FILENAME,
    SUMMARY_FILENAME,
    ProvenanceRecord,
    write_json_atomic,
    write_provenance_csv,
)
from .vad import VAD_FRAME_DURATIONS_MS, VAD_SAMPLE_RATES, WebRtcVad


class NormalizationError(RuntimeError):
    """Raised when a normalization run violates a safety invariant."""


@dataclass(frozen=True)
class NormalizationConfig:
    """Explicit normalization settings; no dataset path is implicit."""

    input_root: Path
    output_root: Path
    target_sample_rate: int = 16000
    target_channels: int = 1
    target_duration_seconds: float = 3.0
    seed: int = 2026
    vad_aggressiveness: int = 1
    vad_frame_ms: int = 30
    candidate_quality_ratio: float = 0.95
    workers: int = 1
    resume: bool = False

    @property
    def target_samples(self) -> int:
        return round(self.target_sample_rate * self.target_duration_seconds)

    def validate(self) -> None:
        if self.target_sample_rate not in VAD_SAMPLE_RATES:
            raise ValueError(
                f"target sample rate must support WebRTC VAD: {sorted(VAD_SAMPLE_RATES)}"
            )
        if self.target_channels < 1:
            raise ValueError("target channels must be at least 1")
        if not np.isfinite(self.target_duration_seconds) or self.target_duration_seconds <= 0:
            raise ValueError("target duration must be finite and positive")
        if self.target_samples <= 0:
            raise ValueError("configured target duration produces no samples")
        if self.vad_aggressiveness not in range(4):
            raise ValueError("VAD aggressiveness must be 0, 1, 2, or 3")
        if self.vad_frame_ms not in VAD_FRAME_DURATIONS_MS:
            raise ValueError("VAD frame duration must be 10, 20, or 30 ms")
        if not 0 < self.candidate_quality_ratio <= 1:
            raise ValueError("candidate quality ratio must be in (0, 1]")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")


@dataclass(frozen=True)
class NormalizationResult:
    """Run summary and portable report locations."""

    summary: dict[str, object]
    provenance_path: Path
    summary_path: Path
    failures: tuple[ProvenanceRecord, ...]


def _normalized_path_text(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _roots_overlap(first: Path, second: Path) -> bool:
    first_text = _normalized_path_text(first)
    second_text = _normalized_path_text(second)
    try:
        common = os.path.commonpath((first_text, second_text))
    except ValueError:
        return False
    return common == first_text or common == second_text


def _prepare_roots(config: NormalizationConfig) -> tuple[Path, Path]:
    input_root = config.input_root.expanduser().resolve(strict=True)
    if not input_root.is_dir():
        raise NormalizationError(f"input root is not a directory: {input_root}")
    output_root = config.output_root.expanduser().resolve(strict=False)
    if _roots_overlap(input_root, output_root):
        raise NormalizationError(
            "input and output roots must be distinct and neither may contain the other"
        )
    if output_root.exists() and not output_root.is_dir():
        raise NormalizationError(f"output root is not a directory: {output_root}")
    existing_files = [path for path in output_root.rglob("*") if path.is_file()] if output_root.exists() else []
    if existing_files and not config.resume:
        raise NormalizationError(
            "output root already contains files; use a new root or explicitly enable --resume"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return input_root, output_root


def _scan_sources(input_root: Path) -> list[tuple[Path, Path]]:
    files = sorted(
        (path for path in input_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(input_root).as_posix(),
    )
    unsupported = [path.relative_to(input_root).as_posix() for path in files if path.suffix.lower() != ".wav"]
    if unsupported:
        preview = ", ".join(unsupported[:5])
        raise NormalizationError(
            f"input contains {len(unsupported)} unsupported non-WAV file(s): {preview}"
        )
    sources: list[tuple[Path, Path]] = []
    for source in files:
        relative = source.relative_to(input_root)
        if source.is_symlink():
            raise NormalizationError(f"symbolic-link audio input is not supported: {relative.as_posix()}")
        if len(relative.parts) != 2:
            raise NormalizationError(
                "expected speaker-folder/WAV layout; invalid relative path: "
                f"{relative.as_posix()}"
            )
        sources.append((source, relative))
    if not sources:
        raise NormalizationError("input dataset contains no WAV files")
    return sources


def _validate_resume_contents(output_root: Path, expected: set[str]) -> None:
    if not output_root.exists():
        return
    unexpected: list[str] = []
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        if path.suffix.lower() == ".wav":
            if relative not in expected:
                unexpected.append(relative)
        elif relative not in NORMALIZATION_METADATA_FILES:
            unexpected.append(relative)
    if unexpected:
        preview = ", ".join(sorted(unexpected)[:5])
        raise NormalizationError(
            f"resume output contains {len(unexpected)} unexpected file(s): {preview}"
        )


def stable_file_seed(global_seed: int, relative_path: str) -> int:
    """Derive a process- and worker-independent RNG seed with SHA-256."""
    digest = hashlib.sha256(f"{global_seed}:{relative_path}".encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _base_record(
    config: NormalizationConfig,
    relative: Path,
    metadata: AudioMetadata | None,
    **overrides: object,
) -> ProvenanceRecord:
    values: dict[str, object] = {
        "relative_path": relative.as_posix(),
        "speaker_id": relative.parent.name,
        "source_sample_rate": metadata.sample_rate if metadata else None,
        "source_channels": metadata.channels if metadata else None,
        "source_frames": metadata.frames if metadata else None,
        "source_duration_seconds": metadata.duration_seconds if metadata else None,
        "normalized_duration_seconds": (
            round(metadata.frames * config.target_sample_rate / metadata.sample_rate)
            / config.target_sample_rate
            if metadata and metadata.sample_rate > 0
            else None
        ),
        "target_sample_rate": config.target_sample_rate,
        "target_channels": config.target_channels,
        "target_samples": config.target_samples,
        "target_duration_seconds": config.target_duration_seconds,
        "resampled": metadata.sample_rate != config.target_sample_rate if metadata else None,
        "channel_converted": metadata.channels != config.target_channels if metadata else None,
        "operation": "failed",
        "seed": config.seed,
        "status": "failed",
    }
    values.update(overrides)
    return ProvenanceRecord(**values)  # type: ignore[arg-type]


def _validate_existing_output(path: Path, config: NormalizationConfig) -> None:
    audio = read_wav(path)
    metadata = audio.metadata
    problems: list[str] = []
    if metadata.sample_rate != config.target_sample_rate:
        problems.append(f"sample rate {metadata.sample_rate}")
    if metadata.channels != config.target_channels:
        problems.append(f"channels {metadata.channels}")
    if metadata.frames != config.target_samples:
        problems.append(f"frames {metadata.frames}")
    if metadata.subtype != "PCM_16":
        problems.append(f"subtype {metadata.subtype}")
    if problems:
        raise NormalizationError(
            f"existing resume output is invalid ({'; '.join(problems)}): {path}"
        )


def _process_one(
    source: Path,
    relative: Path,
    output_root: Path,
    config: NormalizationConfig,
) -> ProvenanceRecord:
    metadata = inspect_wav(source)
    if metadata.frames <= 0:
        raise NormalizationError(f"source WAV has no audio frames: {relative.as_posix()}")
    target = output_root / relative
    target_parent = target.parent.resolve(strict=False)
    try:
        target_parent.relative_to(output_root)
    except ValueError as exc:
        raise NormalizationError(f"output path escapes output root: {relative.as_posix()}") from exc

    if target.exists():
        if not config.resume:
            raise NormalizationError(f"refusing to overwrite existing output: {relative.as_posix()}")
        _validate_existing_output(target, config)
        return _base_record(
            config,
            relative,
            metadata,
            operation="resume",
            status="resume_existing_valid",
        )

    audio = read_wav(source)
    waveform = convert_channels(audio.waveform, config.target_channels)
    waveform = resample_waveform(waveform, metadata.sample_rate, config.target_sample_rate)
    normalized_duration = waveform.shape[0] / config.target_sample_rate
    fields: dict[str, object] = {"normalized_duration_seconds": normalized_duration}

    if waveform.shape[0] > config.target_samples:
        mono_guide = np.mean(waveform, axis=1, dtype=np.float32)
        detector = WebRtcVad(
            sample_rate=config.target_sample_rate,
            aggressiveness=config.vad_aggressiveness,
            frame_ms=config.vad_frame_ms,
        )
        candidates = detector.score_target_windows(mono_guide, config.target_samples)
        best_score = max(candidate.speech_ratio for candidate in candidates)
        if best_score == 0.0:
            eligible = candidates
            vad_status = "vad_no_speech_random_fallback"
        else:
            threshold = config.candidate_quality_ratio * best_score
            eligible = [
                candidate for candidate in candidates if candidate.speech_ratio + 1e-12 >= threshold
            ]
            vad_status = "vad_scored"
        rng = random.Random(stable_file_seed(config.seed, relative.as_posix()))
        selected = rng.choice(eligible)
        crop_end = selected.start_sample + config.target_samples
        waveform = waveform[selected.start_sample:crop_end, :]
        fields.update(
            operation="crop",
            crop_start_seconds=selected.start_sample / config.target_sample_rate,
            crop_end_seconds=crop_end / config.target_sample_rate,
            selected_window_speech_ratio=selected.speech_ratio,
            best_window_speech_ratio=best_score,
            candidate_window_count=len(candidates),
            eligible_candidate_count=len(eligible),
            vad_status=vad_status,
        )
    elif waveform.shape[0] == config.target_samples:
        fields.update(operation="preserve_duration")
    else:
        missing = config.target_samples - waveform.shape[0]
        left = missing // 2
        right = missing - left
        waveform = np.pad(waveform, ((left, right), (0, 0)), mode="constant")
        fields.update(
            operation="pad",
            padding_left_seconds=left / config.target_sample_rate,
            padding_right_seconds=right / config.target_sample_rate,
        )

    if waveform.shape != (config.target_samples, config.target_channels):
        raise NormalizationError(
            f"exact output invariant failed for {relative.as_posix()}: {waveform.shape}"
        )
    if target.exists():
        raise NormalizationError(f"refusing to overwrite existing output: {relative.as_posix()}")
    write_wav_pcm16_atomic(target, waveform, config.target_sample_rate)
    return _base_record(config, relative, metadata, status="processed", **fields)


def normalize_dataset(
    config: NormalizationConfig,
    progress: Callable[[int, int], None] | None = None,
) -> NormalizationResult:
    """Normalize all source WAVs without advancing to output validation."""
    config.validate()
    input_root, output_root = _prepare_roots(config)
    sources = _scan_sources(input_root)
    expected = {relative.as_posix() for _, relative in sources}
    if config.resume:
        _validate_resume_contents(output_root, expected)

    records: list[ProvenanceRecord] = []
    metadata_by_relative: dict[str, AudioMetadata] = {}

    def run(source: Path, relative: Path) -> ProvenanceRecord:
        try:
            metadata_by_relative[relative.as_posix()] = inspect_wav(source)
            return _process_one(source, relative, output_root, config)
        except Exception as exc:  # Collect every per-file failure for a complete run report.
            return _base_record(
                config,
                relative,
                metadata_by_relative.get(relative.as_posix()),
                error=f"{type(exc).__name__}: {exc}",
            )

    completed = 0
    if config.workers == 1:
        for source, relative in sources:
            records.append(run(source, relative))
            completed += 1
            if progress:
                progress(completed, len(sources))
    else:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(run, source, relative): relative
                for source, relative in sources
            }
            for future in as_completed(futures):
                records.append(future.result())
                completed += 1
                if progress:
                    progress(completed, len(sources))

    records.sort(key=lambda record: record.relative_path)
    failures = tuple(record for record in records if record.status == "failed")
    successful = [record for record in records if record.status != "failed"]
    summary: dict[str, object] = {
        "total_files": len(records),
        "processed_files": len(successful),
        "preserve_count": sum(record.operation == "preserve_duration" for record in records),
        "crop_count": sum(record.operation == "crop" for record in records),
        "pad_count": sum(record.operation == "pad" for record in records),
        "resume_count": sum(record.status == "resume_existing_valid" for record in records),
        "failure_count": len(failures),
        "resampled_count": sum(record.resampled is True and record.status != "failed" for record in records),
        "channel_converted_count": sum(
            record.channel_converted is True and record.status != "failed" for record in records
        ),
        "configuration": {
            "target_sample_rate": config.target_sample_rate,
            "target_channels": config.target_channels,
            "target_duration_seconds": config.target_duration_seconds,
            "target_samples": config.target_samples,
            "seed": config.seed,
            "vad_aggressiveness": config.vad_aggressiveness,
            "vad_frame_ms": config.vad_frame_ms,
            "candidate_quality_ratio": config.candidate_quality_ratio,
            "workers": config.workers,
            "resume": config.resume,
        },
    }
    provenance_path = write_provenance_csv(output_root / PROVENANCE_FILENAME, records)
    summary_path = write_json_atomic(output_root / SUMMARY_FILENAME, summary)
    return NormalizationResult(
        summary=summary,
        provenance_path=provenance_path,
        summary_path=summary_path,
        failures=failures,
    )
