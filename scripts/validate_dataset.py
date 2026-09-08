"""Validate a processed speaker WAV dataset against its exact output contract."""

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from speaker_data_pipeline.validation import validate_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--target-sample-rate", required=True, type=int)
    parser.add_argument("--target-channels", required=True, type=int)
    parser.add_argument("--target-duration", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    return parser


def main() -> int:
    """Parse arguments, run only validation, and return its pass/fail status."""
    args = build_parser().parse_args()
    try:
        result = validate_dataset(
            input_root=args.input_root,
            output_dir=args.output_dir,
            target_sample_rate=args.target_sample_rate,
            target_channels=args.target_channels,
            target_duration_seconds=args.target_duration,
            source_root=args.source_root,
        )
    except Exception as exc:
        print(f"Validation failed to run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Validation: {'PASS' if result.passed else 'FAIL'}")
    print(f"WAV files: {result.summary['total_wav_files']}")
    print(f"Failures: {result.summary['failure_count']}")
    print(f"Summary: {result.summary_path}")
    print(f"Failure report: {result.failures_path}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
