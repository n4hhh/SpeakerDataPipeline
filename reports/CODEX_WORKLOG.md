# Codex project worklog

This file is append-only. Later tasks must preserve historical entries and append
corrections or new results rather than rewriting earlier records.

## Project baseline

- SpeakerDataPipeline is a new standalone repository.
- Source datasets are external and immutable.
- The current task covers scaffold, governance, and documentation only.
- No dataset has been accessed.
- No preprocessing has been implemented.
- No commit or push has been performed.

## 2026-09-08 — Task 0: repository scaffold and governance

### Goal

Establish the minimal reusable repository structure, source-data safety rules,
public-facing documentation, and honest placeholders for later approved stages.

### Repository state inspected

The repository was inspected before editing. It contained a Git directory and
two tracked, empty files: `README.md` and `.gitignore`. The working tree was clean
on branch `main`, tracking `origin/main`.

### Files created

- `AGENTS.md`
- `requirements.txt`
- `configs/example.yaml`
- `src/speaker_data_pipeline/__init__.py`
- `src/speaker_data_pipeline/audio_io.py`
- `src/speaker_data_pipeline/audit.py`
- `src/speaker_data_pipeline/vad.py`
- `src/speaker_data_pipeline/normalize.py`
- `src/speaker_data_pipeline/provenance.py`
- `src/speaker_data_pipeline/validation.py`
- `scripts/audit_dataset.py`
- `scripts/normalize_dataset.py`
- `scripts/validate_dataset.py`
- `reports/CODEX_WORKLOG.md`
- `tests/__init__.py`

### Files modified

- `README.md`
- `.gitignore`

### Important design decisions

- Source datasets are immutable and output must use a distinct root.
- Audit, normalize, and validate remain separately approved stages.
- Dataset-specific behavior belongs in CLI arguments or configuration.
- Transformations must fail clearly, be deterministic when applicable, and
  produce source-linked provenance.
- Scaffold scripts report that they are unimplemented instead of simulating work.

### Commands/tests executed

- Inspected the top-level directory, tracked files, and Git status.
- Enumerated repository files without traversing external datasets or repositories.
- Compiled all scaffold Python files with Python's syntax compiler.
- Invoked each placeholder CLI with representative arguments and confirmed a
  nonzero, explicit not-implemented result.
- Inspected `git diff`, ran `git diff --check`, and ran `git status --short`.

### Result

Task 0 repository scaffolding and documentation were completed. Syntax validation
passed, and no processing capability was represented as implemented.

### Explicitly deferred work

- Dataset auditing and report generation
- Audio loading, metadata extraction, resampling, and channel conversion
- VAD selection or implementation
- Cropping, padding, normalization, and reject-policy implementation
- Provenance schema implementation
- Output validation implementation
- Runtime dependency selection and pinning
- Dataset access, manifests, splits, feature extraction, model training, evaluation,
  commits, and pushes

## 2026-09-08 — First production SpeakerDataPipeline implementation

### Goal

Replace the Task 0 placeholders with reusable implementations for the separately
invoked AUDIT, NORMALIZE, and VALIDATE stages without processing a real dataset.

### Files inspected

- `AGENTS.md`, `reports/CODEX_WORKLOG.md`, `README.md`, `requirements.txt`, and
  `configs/example.yaml`
- Every existing file under `src/speaker_data_pipeline/` and `scripts/`
- Repository file inventory, branch, working-tree status, and final diff

### Files created or modified

No new repository paths were added. The following existing files were modified:

- `README.md`, `requirements.txt`, and `configs/example.yaml`
- `src/speaker_data_pipeline/__init__.py`
- `src/speaker_data_pipeline/audio_io.py`
- `src/speaker_data_pipeline/audit.py`
- `src/speaker_data_pipeline/vad.py`
- `src/speaker_data_pipeline/normalize.py`
- `src/speaker_data_pipeline/provenance.py`
- `src/speaker_data_pipeline/validation.py`
- `scripts/audit_dataset.py`
- `scripts/normalize_dataset.py`
- `scripts/validate_dataset.py`
- `reports/CODEX_WORKLOG.md` (this appended entry only)

`AGENTS.md`, `.gitignore`, and `tests/__init__.py` were preserved without changes
from Task 0.

### Dependencies selected

The CPU-only runtime stack is bounded to NumPy 1.24–2.x, SciPy 1.10–1.x,
SoundFile 0.12–0.x, and `webrtcvad-wheels` 2.0.14–2.x. No packages were installed.

### Audio I/O policy

Audio is decoded through SoundFile as finite float32 arrays using one consistent
`[samples, channels]` convention. Outputs are clipped to the valid float audio
range, written as PCM-16 WAV to a sibling temporary file, verified, and atomically
published. Source WAVs are never opened for writing.

### Resampling policy

SciPy `resample_poly` uses GCD-reduced rational up/down factors. The expected
sample count is calculated deterministically; only a one-sample rounding mismatch
may be trimmed or zero-padded at the resampling boundary, and larger corrections
fail.

### Channel-conversion policy

Multi-channel input converts to target mono by arithmetic mean. Mono may be
explicitly duplicated to a multi-channel target. Ambiguous conversion between two
unequal multi-channel layouts fails.

### VAD algorithm

WebRTC VAD accepts aggressiveness 0–3, compatible rates 8/16/32/48 kHz, and
10/20/30 ms frames. It converts a finite normalized mono guidance waveform to
PCM-16 bytes and returns a deterministic frame mask. VAD only scores contiguous
windows and never removes or concatenates speech regions.

### Deterministic crop policy

Target-length windows are evaluated at VAD-frame stride plus the final valid
boundary. Candidates at least the configured ratio of the best speech occupancy
are eligible. Selection uses a random generator seeded from SHA-256 of the global
seed and portable relative path. Zero-speech candidates use the same deterministic
random fallback and record that condition. Selected-window provenance records the
actual selected score.

### Short-audio padding policy

Post-resample audio shorter than the target is padded with zeros at its boundaries:
floor of the missing samples on the left and the remainder on the right. Silence
is never inserted in the middle.

### Provenance schema

UTF-8 CSV rows use portable relative paths, deterministic ordering, source and
target audio properties, normalized duration, transformation flags, crop/padding
details, VAD scores and candidate counts, global seed, status, and error text.
`normalization_summary.json` records required aggregate counts and effective
configuration without absolute dataset roots.

### Resume behavior

Non-resume runs reject populated output roots. Resume accepts an existing expected
WAV only after full decoding and verification of finite samples, target rate,
channels, exact frames, and PCM-16 subtype. Invalid or unexpected output files
fail closed; valid files record `resume_existing_valid`.

### Validation behavior

Validation fully decodes every processed WAV; checks structure, readability,
finiteness, PCM-16 subtype, rate, channels, exact samples and duration; reports
zero-frame, duplicate-path, and unexpected-file failures; and optionally requires
source/output relative-WAV path-set equality. Its CLI returns zero only for a
complete pass.

### Commands executed

- Inspected all required repository files and ran `git status --short --branch`.
- Ran Python syntax compilation across `src`, `scripts`, and `tests`.
- Ran all three CLI commands with `--help` successfully.
- Confirmed NumPy, SciPy, SoundFile, and WebRTC VAD were already importable; no
  dependency installation was performed.
- Created a temporary repository-local synthetic dataset with one short stereo
  8 kHz WAV, one exact-length mono 16 kHz WAV, and one long mono 16 kHz WAV.
- Ran audit, two-worker normalization, validation with source path comparison,
  and safe resume. All completed as expected with crop/preserve/pad counts of
  1/1/1, zero failures, and validation PASS.
- Repeated normalization with one worker and confirmed all output WAV SHA-256
  hashes matched the two-worker run.
- Confirmed equal, descendant, and ancestor input/output roots fail, and confirmed
  a populated output fails without resume.
- Removed only the verified temporary synthetic test directory.
- Inspected the final diff, ran `git diff --check`, and ran final Git status.

### Result

The first production implementation is complete for strict folder-per-speaker,
WAV-only input. Audit, normalize, and validate remain independent commands. No
real or external dataset was accessed, and no commit or push was performed.

### Explicitly deferred work

- Alternative dataset-layout adapters and non-WAV input formats
- Content hashing and duplicate-audio detection
- Additional channel-matrix policies beyond mean-to-mono and mono duplication
- Audio augmentation and learned VAD models
- Speaker splits, manifests, feature extraction, model training, and evaluation
- Automated stage chaining, package publishing, commits, and pushes
