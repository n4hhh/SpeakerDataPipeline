# SpeakerDataPipeline agent guidance

These rules apply to all work in this repository.

## Working practice

- Inspect the repository and Git status before editing.
- Keep every task limited to its explicitly approved stage.
- Never automatically continue into another pipeline stage.
- Do not commit or push unless explicitly instructed.
- Do not claim functionality is implemented when only scaffold code exists.

## Dataset and transformation safety

- Treat source datasets as immutable.
- Never write into the input dataset root or overwrite source audio.
- Require input and output roots to resolve to distinct paths.
- Do not silently resample, convert channels, crop, pad, normalize, augment, or
  delete files. Every transformation requires explicit task approval.
- Fail closed on unexpected audio or data conditions unless an explicitly
  approved reject policy exists.
- Use CLI arguments or configuration for dataset-specific paths, counts, and
  settings; never hard-code them into reusable core modules.
- Expose and record the random seed for deterministic processing.
- Make every generated output traceable to its source.
- Validate processed output before considering it ready for downstream use.

## Approved conceptual stages

- **AUDIT**: read-only dataset inventory and compatibility analysis.
- **NORMALIZE**: explicitly approved transformations written to a separate output.
- **VALIDATE**: complete checks of the processed output dataset.

Each stage requires separate approval and must not trigger the next stage
automatically.

## Repository hygiene and history

- Do not commit source datasets, generated processed audio, virtual environments,
  caches, large generated artifacts, or machine-specific files and paths.
- Treat `reports/CODEX_WORKLOG.md` as append-only historical task documentation.
- Never rewrite historical worklog entries to make later results look cleaner;
  append a correction or follow-up entry instead.
