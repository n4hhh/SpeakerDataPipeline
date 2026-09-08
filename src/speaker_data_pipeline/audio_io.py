"""Safe WAV metadata, loading, channel conversion, resampling, and writing."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly
import soundfile as sf


FloatWaveform = NDArray[np.float32]
WAV_CONTAINER_FORMATS = frozenset({"WAV", "WAVEX", "RF64"})


class AudioIOError(RuntimeError):
    """Raised when an audio file cannot satisfy the pipeline contract."""


@dataclass(frozen=True)
class AudioMetadata:
    """Portable audio properties reported by libsndfile."""

    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    subtype: str
    format: str


@dataclass(frozen=True)
class AudioData:
    """Decoded audio using the invariant ``[samples, channels]`` layout."""

    waveform: FloatWaveform
    metadata: AudioMetadata


def inspect_wav(path: Path | str) -> AudioMetadata:
    """Read WAV metadata without decoding audio samples."""
    source = Path(path)
    try:
        info = sf.info(str(source))
    except (RuntimeError, OSError) as exc:
        raise AudioIOError(f"cannot read WAV metadata for {source}: {exc}") from exc
    if info.format.upper() not in WAV_CONTAINER_FORMATS:
        raise AudioIOError(
            f"unsupported audio container for {source}: {info.format!r}; expected WAV"
        )
    return AudioMetadata(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        duration_seconds=float(info.duration),
        subtype=str(info.subtype),
        format=str(info.format),
    )


def read_wav(path: Path | str) -> AudioData:
    """Decode a WAV as finite float32 samples with shape ``[samples, channels]``."""
    source = Path(path)
    metadata = inspect_wav(source)
    try:
        waveform, sample_rate = sf.read(
            str(source), dtype="float32", always_2d=True
        )
    except (RuntimeError, OSError) as exc:
        raise AudioIOError(f"cannot decode WAV {source}: {exc}") from exc
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 2:
        raise AudioIOError(f"decoded WAV has invalid shape {waveform.shape}: {source}")
    if int(sample_rate) != metadata.sample_rate:
        raise AudioIOError(f"sample-rate metadata changed while reading {source}")
    if waveform.shape != (metadata.frames, metadata.channels):
        raise AudioIOError(
            f"decoded shape {waveform.shape} disagrees with metadata "
            f"({metadata.frames}, {metadata.channels}) for {source}"
        )
    if not np.isfinite(waveform).all():
        raise AudioIOError(f"WAV contains non-finite samples: {source}")
    return AudioData(waveform=waveform, metadata=metadata)


def convert_channels(waveform: FloatWaveform, target_channels: int) -> FloatWaveform:
    """Convert channels deterministically while preserving sample order.

    Multi-channel input is converted to mono with an arithmetic mean. Mono input
    can be explicitly duplicated to a requested multi-channel layout. Conversion
    between two unequal multi-channel layouts is intentionally rejected because
    no unambiguous generic mapping exists.
    """
    if waveform.ndim != 2 or waveform.shape[1] < 1:
        raise AudioIOError(f"waveform must have shape [samples, channels], got {waveform.shape}")
    if target_channels < 1:
        raise ValueError("target_channels must be at least 1")
    source_channels = waveform.shape[1]
    if source_channels == target_channels:
        return np.asarray(waveform, dtype=np.float32)
    if target_channels == 1:
        return np.mean(waveform, axis=1, keepdims=True, dtype=np.float32)
    if source_channels == 1:
        return np.repeat(waveform, target_channels, axis=1).astype(np.float32)
    raise AudioIOError(
        f"unsupported channel conversion: {source_channels} -> {target_channels}"
    )


def resample_waveform(
    waveform: FloatWaveform, source_rate: int, target_rate: int
) -> FloatWaveform:
    """Resample along the sample axis using reduced rational polyphase factors."""
    if waveform.ndim != 2:
        raise AudioIOError(f"waveform must be two-dimensional, got {waveform.shape}")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate:
        return np.asarray(waveform, dtype=np.float32)
    divisor = gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    expected_samples = round(waveform.shape[0] * target_rate / source_rate)
    result = np.asarray(resample_poly(waveform, up, down, axis=0), dtype=np.float32)
    correction = expected_samples - result.shape[0]
    if abs(correction) > 1:
        raise AudioIOError(
            "resampling produced an unexpectedly large sample-count correction: "
            f"expected {expected_samples}, received {result.shape[0]}"
        )
    if correction < 0:
        result = result[:expected_samples, :]
    elif correction > 0:
        result = np.pad(result, ((0, correction), (0, 0)), mode="constant")
    if not np.isfinite(result).all():
        raise AudioIOError("resampling produced non-finite samples")
    return np.asarray(result, dtype=np.float32)


def write_wav_pcm16_atomic(
    path: Path | str, waveform: FloatWaveform, sample_rate: int
) -> None:
    """Validate and atomically publish a PCM-16 WAV at ``path``.

    A sibling temporary file is written and verified before ``os.replace`` makes
    it visible at the final path. Callers must separately decide whether replacing
    an existing target is permitted.
    """
    target = Path(path)
    if waveform.ndim != 2 or waveform.shape[1] < 1:
        raise AudioIOError(f"waveform must have shape [samples, channels], got {waveform.shape}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.isfinite(waveform).all():
        raise AudioIOError("refusing to write non-finite audio samples")
    safe_waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}.", suffix=".tmp.wav", dir=target.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        sf.write(
            str(temporary_path),
            safe_waveform,
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        metadata = inspect_wav(temporary_path)
        if (
            metadata.sample_rate != sample_rate
            or metadata.channels != safe_waveform.shape[1]
            or metadata.frames != safe_waveform.shape[0]
            or metadata.subtype != "PCM_16"
        ):
            raise AudioIOError(f"temporary WAV verification failed for {target}")
        os.replace(temporary_path, target)
        temporary_path = None
    except (RuntimeError, OSError) as exc:
        if isinstance(exc, AudioIOError):
            raise
        raise AudioIOError(f"cannot write WAV {target}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
