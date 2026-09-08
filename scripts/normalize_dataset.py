"""Normalize a speaker WAV dataset into a separate, deterministic output tree."""

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from speaker_data_pipeline.normalize import (  # noqa: E402
    NormalizationConfig,
    normalize_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--target-channels", type=int, default=1)
    parser.add_argument("--target-duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--vad-aggressiveness", type=int, choices=range(4), default=1)
    parser.add_argument("--vad-frame-ms", type=int, choices=(10, 20, 30), default=30)
    parser.add_argument("--candidate-quality-ratio", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    """Parse arguments and run normalization without invoking validation."""
    args = build_parser().parse_args()
    config = NormalizationConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        target_sample_rate=args.target_sample_rate,
        target_channels=args.target_channels,
        target_duration_seconds=args.target_duration,
        seed=args.seed,
        vad_aggressiveness=args.vad_aggressiveness,
        vad_frame_ms=args.vad_frame_ms,
        candidate_quality_ratio=args.candidate_quality_ratio,
        workers=args.workers,
        resume=args.resume,
    )
    next_progress = 1

    def report_progress(completed: int, total: int) -> None:
        nonlocal next_progress
        interval = max(1, min(100, total // 10))
        if completed == total or completed >= next_progress:
            print(f"Progress: {completed}/{total}")
            next_progress = completed + interval

    try:
        result = normalize_dataset(config, progress=report_progress)
    except Exception as exc:
        print(f"Normalization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    labels = (
        ("Total files", "total_files"),
        ("Crop count", "crop_count"),
        ("Preserve count", "preserve_count"),
        ("Pad count", "pad_count"),
        ("Resample count", "resampled_count"),
        ("Channel conversion count", "channel_converted_count"),
        ("Resume count", "resume_count"),
        ("Failure count", "failure_count"),
    )
    for label, key in labels:
        print(f"{label}: {summary[key]}")
    print(f"Provenance: {result.provenance_path}")
    print(f"Summary: {result.summary_path}")
    print(f"Output root: {config.output_root.resolve(strict=False)}")
    if result.failures:
        for failure in result.failures[:10]:
            print(f"Failed {failure.relative_path}: {failure.error}", file=sys.stderr)
        if len(result.failures) > 10:
            print(
                f"...and {len(result.failures) - 10} additional failure(s); see provenance",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
