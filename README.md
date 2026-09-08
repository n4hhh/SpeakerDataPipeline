# SpeakerDataPipeline

SpeakerDataPipeline audits, normalizes, and validates speaker-audio datasets
before they enter downstream speaker-verification or model-training projects. It
does not create data splits, training manifests, features, or models.

The three stages are production-usable but intentionally separate:

```text
AUDIT -> user reviews reports -> NORMALIZE -> VALIDATE
```

No command automatically starts the next stage.

## Supported dataset layout

The first production version accepts WAV files arranged exactly one folder below
the dataset root. A WAV's direct parent folder is its `speaker_id`:

```text
dataset/
|-- speaker_001/
|   |-- utterance_001.wav
|   `-- utterance_002.wav
`-- speaker_002/
    `-- utterance_001.wav
```

Normalization is intentionally strict: root-level WAVs, deeper-nested WAVs, and
non-WAV files cause it to fail rather than silently omit data. Audit reports these
conditions, and validation rejects them in processed output. Common PCM WAV
formats readable by `soundfile` are supported; other audio containers are not.

## Source-data safety

> **Source datasets are immutable. SpeakerDataPipeline never modifies or deletes
> source files.**

Normalization always writes a separate tree while preserving speaker folders and
relative WAV filenames:

```text
Input:  D:\datasets\my_dataset
Output: D:\datasets\my_dataset_processed
```

The input and output roots must differ. Neither may contain the other, preventing
recursive discovery of generated files and accidental publication over source
audio. WAVs are written to a temporary sibling and verified before atomic
publication as completed outputs.

## Installation

SpeakerDataPipeline targets Python 3.10+ and uses a small CPU-only stack:

- `numpy` for array operations
- `scipy` for polyphase resampling
- `soundfile` for WAV metadata, decoding, and PCM-16 writing
- `webrtcvad-wheels` (imported as `webrtcvad`) for speech detection

Windows PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The bounded versions are recorded in `requirements.txt`. PyTorch, torchaudio,
SpeechBrain, and librosa are not dependencies.

## 1. Audit

Audit reads filesystem and WAV metadata without decoding or changing source audio.
It derives speaker identity from each WAV's direct parent folder.

```powershell
python scripts\audit_dataset.py `
    --input-root "D:\datasets\my_dataset" `
    --output-dir "D:\dataset_reports\my_dataset_audit" `
    --target-duration 3.0
```

The report directory must not be inside the input dataset. Audit produces:

- `audit_summary.json`: dataset totals, format/duration distributions, target-
  duration comparison, speaker statistics, and structural issues
- `speaker_summary.csv`: deterministic per-speaker WAV/readability totals
- `file_inventory.csv`: relative path, speaker, metadata, size, and readable status

Unreadable and zero-frame WAVs, root-level/deeper nesting, empty speaker folders,
and non-WAV files are reported explicitly. Audio content is not hashed.

Review these reports before intentionally invoking normalization.

## 2. Normalize

```powershell
python scripts\normalize_dataset.py `
    --input-root "D:\datasets\my_dataset" `
    --output-root "D:\datasets\my_dataset_3s" `
    --target-sample-rate 16000 `
    --target-channels 1 `
    --target-duration 3.0 `
    --vad-aggressiveness 1 `
    --vad-frame-ms 30 `
    --candidate-quality-ratio 0.95 `
    --seed 2026 `
    --workers 8
```

The defaults are 16 kHz, mono, 3.0 seconds, seed 2026, VAD aggressiveness 1,
30 ms VAD frames, candidate quality ratio 0.95, and one worker. All are explicit
CLI/configuration values rather than dataset-specific constants.

For each WAV, processing order is:

```text
decode -> channel conversion -> resample -> duration normalization -> PCM-16 WAV
```

The exact output sample count is:

```text
round(target_sample_rate * target_duration_seconds)
```

Every successful output must satisfy that invariant.

### Channel and resampling policy

Mono conversion uses the deterministic arithmetic mean of all source channels;
it never silently selects one channel. Mono can be explicitly duplicated to a
multi-channel target. An ambiguous unequal multi-channel conversion is rejected.

Sample-rate conversion uses `scipy.signal.resample_poly` with rational factors
reduced by their greatest common divisor. The expected resampled length is
calculated deterministically; a one-sample library rounding difference is trimmed
at the end or zero-padded at the end, while a larger discrepancy fails.

### VAD-guided crop algorithm

Audio longer than the target is downmixed to a mono guidance signal for WebRTC
VAD. VAD accepts target rates 8000, 16000, 32000, or 48000 Hz and frame sizes 10,
20, or 30 ms.

The normalizer scores contiguous target-length windows at VAD-frame stride plus
the final valid boundary. It finds the best speech occupancy, retains candidates
whose score is at least `candidate_quality_ratio * best_score`, and chooses one
eligible window with a stable per-file random generator. It never concatenates
separated speech regions or substitutes amplitude maxima for speech detection.

If every candidate contains zero VAD speech, the same deterministic generator
selects a valid contiguous window and provenance records
`vad_no_speech_random_fallback`.

The per-file generator seed is derived from SHA-256 of:

```text
<global_seed>:<dataset-relative-path>
```

Consequently, a file and global seed select the same crop regardless of filesystem
order, worker count, or worker completion order. Changing the seed can select a
different qualifying crop.

### Exact-length and short audio

Audio already at the target sample count is preserved in full and recorded as
`preserve_duration` (while any sample-rate or channel transformation remains
separately recorded).

Short audio is symmetrically zero-padded only at its boundaries:

```text
left = floor(missing_samples / 2)
right = missing_samples - left
output = left zeros + original audio + right zeros
```

No silence is inserted in the middle.

### Provenance and summary

Normalization writes these portable metadata files at the output root:

- `normalization_provenance.csv`
- `normalization_summary.json`

The CSV is sorted by dataset-relative path and records source properties, target
contract, normalized duration, channel/resampling flags, crop or padding values,
actual selected and best VAD occupancy, candidate counts, seed, status, and errors.
It does not require machine-specific absolute dataset roots.

The JSON contains aggregate crop, preserve, pad, resume, failure, resampling, and
channel-conversion counts plus the effective configuration.

If any file fails, the remaining file errors are collected, reports are written,
and the command returns nonzero. A partial run is never reported as successful.

### Safe resume

Resume is disabled by default. Without `--resume`, a nonempty output root is
rejected. To continue an interrupted run:

```powershell
python scripts\normalize_dataset.py `
    --input-root "D:\datasets\my_dataset" `
    --output-root "D:\datasets\my_dataset_3s" `
    --resume
```

An existing expected WAV is skipped only after full decoding confirms finite
audio, the configured sample rate and channels, the exact target sample count,
and PCM-16 subtype. It is recorded as `resume_existing_valid`. Invalid existing
or unexpected output files fail closed and are never silently overwritten.

## 3. Validate

Validation is a separate, intentional command:

```powershell
python scripts\validate_dataset.py `
    --input-root "D:\datasets\my_dataset_3s" `
    --source-root "D:\datasets\my_dataset" `
    --target-sample-rate 16000 `
    --target-channels 1 `
    --target-duration 3.0 `
    --output-dir "D:\dataset_reports\my_dataset_validation"
```

`--source-root` is optional. When supplied, validation requires exact equality of
the source and output relative WAV path sets.

Validation fully decodes every output WAV and checks readability, finite samples,
sample rate, channels, exact sample count, duration, PCM-16 subtype, zero frames,
layout, portable duplicate paths, and unexpected non-WAV files. The two known
normalization metadata files are allowed. It produces:

- `validation_summary.json`
- `validation_failures.csv`

The command returns exit code 0 only when every required invariant passes. Its
report directory must remain outside both the processed input and optional source
dataset.

## Manual end-to-end workflow

1. Run `audit_dataset.py` into a separate report directory.
2. Review the audit and decide whether normalization settings are appropriate.
3. Run `normalize_dataset.py` into a distinct output dataset.
4. Run `validate_dataset.py` and require a complete pass before downstream use.

Do not point output at input, delete source data without an independent backup
policy, skip validation, or edit provenance to conceal rejected or failed files.

## Configuration example

`configs/example.yaml` documents the implemented settings for teams that manage
configuration externally. The current CLIs are argument-driven and do not load
YAML.

## Current status and limitations

This is the first production implementation for strict folder-per-speaker,
WAV-only datasets. It does not implement content hashing, duplicate-audio
detection, alternative dataset adapters, augmentation, speaker splits, manifests,
feature extraction, model training, or evaluation.
