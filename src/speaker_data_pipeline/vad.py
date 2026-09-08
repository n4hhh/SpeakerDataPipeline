"""Deterministic WebRTC VAD masks and contiguous-window speech scoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray
import webrtcvad


VAD_SAMPLE_RATES = frozenset({8000, 16000, 32000, 48000})
VAD_FRAME_DURATIONS_MS = frozenset({10, 20, 30})


@dataclass(frozen=True)
class WindowScore:
    """Speech occupancy for one contiguous candidate window."""

    start_sample: int
    speech_ratio: float


class WebRtcVad:
    """Small WebRTC VAD wrapper that never changes waveform chronology."""

    def __init__(self, sample_rate: int, aggressiveness: int = 1, frame_ms: int = 30):
        if sample_rate not in VAD_SAMPLE_RATES:
            raise ValueError(
                f"WebRTC VAD sample rate must be one of {sorted(VAD_SAMPLE_RATES)}"
            )
        if aggressiveness not in range(4):
            raise ValueError("VAD aggressiveness must be 0, 1, 2, or 3")
        if frame_ms not in VAD_FRAME_DURATIONS_MS:
            raise ValueError("WebRTC VAD frame duration must be 10, 20, or 30 ms")
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000
        self._vad = webrtcvad.Vad(aggressiveness)

    def frame_mask(self, mono_waveform: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Return one deterministic voiced flag per frame, padding only the final frame."""
        waveform = np.asarray(mono_waveform, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError(f"VAD input must be mono and one-dimensional, got {waveform.shape}")
        if not np.isfinite(waveform).all():
            raise ValueError("VAD input contains non-finite samples")
        if waveform.size == 0:
            return np.zeros(0, dtype=np.bool_)
        frame_count = ceil(waveform.size / self.frame_samples)
        padded_count = frame_count * self.frame_samples
        if padded_count != waveform.size:
            waveform = np.pad(waveform, (0, padded_count - waveform.size), mode="constant")
        pcm = np.rint(np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
        mask = np.empty(frame_count, dtype=np.bool_)
        for index in range(frame_count):
            start = index * self.frame_samples
            end = start + self.frame_samples
            mask[index] = self._vad.is_speech(pcm[start:end].tobytes(), self.sample_rate)
        return mask

    def score_window(
        self,
        mask: NDArray[np.bool_],
        start_sample: int,
        window_samples: int,
        total_samples: int,
    ) -> float:
        """Return the voiced-sample fraction of one contiguous window."""
        if start_sample < 0 or window_samples <= 0:
            raise ValueError("window bounds must be positive")
        end_sample = start_sample + window_samples
        if end_sample > total_samples:
            raise ValueError("candidate window extends beyond the waveform")
        first_frame = start_sample // self.frame_samples
        last_frame = (end_sample - 1) // self.frame_samples
        voiced_samples = 0
        for frame_index in range(first_frame, last_frame + 1):
            frame_start = frame_index * self.frame_samples
            frame_end = min(frame_start + self.frame_samples, total_samples)
            overlap = max(0, min(end_sample, frame_end) - max(start_sample, frame_start))
            if overlap and frame_index < mask.size and bool(mask[frame_index]):
                voiced_samples += overlap
        return voiced_samples / window_samples

    def score_target_windows(
        self, mono_waveform: NDArray[np.float32], target_samples: int
    ) -> list[WindowScore]:
        """Score frame-stride candidates plus the final valid contiguous window."""
        waveform = np.asarray(mono_waveform, dtype=np.float32)
        if target_samples <= 0:
            raise ValueError("target_samples must be positive")
        if waveform.ndim != 1 or waveform.size < target_samples:
            raise ValueError("waveform must be mono and at least as long as the target")
        last_start = waveform.size - target_samples
        starts = list(range(0, last_start + 1, self.frame_samples))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        mask = self.frame_mask(waveform)
        return [
            WindowScore(
                start_sample=start,
                speech_ratio=self.score_window(mask, start, target_samples, waveform.size),
            )
            for start in starts
        ]
