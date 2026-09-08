"""Run a read-only metadata and structure audit for a speaker WAV dataset."""

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from speaker_data_pipeline.audit import audit_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-duration", type=float)
    return parser


def main() -> int:
    """Parse arguments, run only the audit stage, and print report locations."""
    args = build_parser().parse_args()
    try:
        result = audit_dataset(args.input_root, args.output_dir, args.target_duration)
    except Exception as exc:
        print(f"Audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    dataset = result.summary["dataset"]
    issues = result.summary["structural_issues"]
    print(f"Total files: {dataset['total_files']}")
    print(f"WAV files: {dataset['total_wav_files']}")
    print(f"Speaker folders: {dataset['total_speaker_folders']}")
    print(f"Unreadable WAVs: {len(issues['unreadable_wavs'])}")
    print(f"Summary: {result.summary_path}")
    print(f"Speaker summary: {result.speaker_summary_path}")
    print(f"File inventory: {result.inventory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
