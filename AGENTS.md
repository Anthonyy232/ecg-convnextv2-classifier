# Working on ECG ConvNeXt V2 classifier

A PyTorch learning project that classifies arrhythmia in MIT-BIH ECG windows from their S-Transform with a ConvNeXt V2 network. Everything runs through `python ecg_cli.py <command>`.

## Code

- Simple and direct. No speculative abstractions, options or extension points.
- Stay in scope; suggest extras instead of adding them.
- Use what the platform, framework, dependencies and this repo already provide before writing a helper.
- Validate only at real boundaries (user input, storage, network, other processes). No checks, fallbacks or try/catch for cases that can't happen.
- One source of truth for each value or rule.
- Delete dead code you find or create.
- Match the surrounding code.
- Fix root causes, and fix the same pattern everywhere it occurs.

## Comments

- Only where the reason isn't clear from the code: a quirk, a race, a non-obvious decision.
- Short fragments, not sentences. Never restate code, narrate, or label sections.
- Delete comments that no longer match.

## Tests

- Behaviour and the edges that can break. No trivial, duplicate, obvious or low-signal tests.
- A regression test fails without its fix.
- Deterministic: fake time and observable events, no sleeps.
- A failing test is a bug until shown otherwise.

## Project

- Scripts in `src/` import each other as top-level modules and run with `src/` as the working directory; the CLI handles that.
- Cached windows in `data/` go stale when preprocessing or `src/config.py` changes: delete them before retraining.
- Keep the patient-wise split: windows from one record never appear in both training and validation, and the held-out records are never trained on.
- No dataset files, derived data, checkpoints or ONNX models in Git. MIT-BIH is downloaded at run time.
- Results quoted in the README must come from a run whose output is committed or linked.
- `src/model.py` adapts MIT-licensed ConvNeXt V2 code: keep its notice. Don't add the CC-BY-NC pretrained weights.

## Done means

- Diff reviewed against this file.
- `python ecg_cli.py test` passes with zero warnings.
- UI changes seen in the running app, light and dark. Docs and metadata changes need no end-to-end run.
- README updated for anything users will notice; behaviour changes called out.

## Docs

- README is for users: what it does, how to get it, how to use it. No internals, no agent instructions.
- This file is for agents: only what can't be found quickly in the code.
- No meta docs in the repo (notes, audits, plans, reports).

## Git

- `type: summary` or `type(scope): summary`: lowercase, imperative, no full stop, about 70 characters at most. Types: `feat` `fix` `refactor` `perf` `test` `docs` `style` `build` `ci` `chore`.
- Body only to say why. `Fixes #123` closes an issue.
- One logical change per commit. Squash fix-ups before merging.
- Branch off `master`. A single commit fast-forwards; several merge with `merge: <summary>`.
- No AI attribution in commits, branches, PRs or releases.
- One maintainer: amending and force-pushing to tidy history is fine. Keep contributors as authors. Delete stale branches.

## Pull requests

- Title in commit format. Body: what changed, why, how it was checked. Short.

## Releases

- None so far. If one is made: bump commit `chore: release vX.Y.Z`; tag `vX.Y.Z`; title `ECG ConvNeXt V2 classifier X.Y.Z`.
- Notes: banner image first, then `### New`, `### Improved`, `### Fixed` (only those that apply). One line per change, written for users; no internals, dependency lists or test counts. End with the full changelog link.
- Banner: generated and hosted on an orphan `media` branch.

## Agents

- Pushing, amending and force-pushing are allowed. Ask before publishing a release or submitting to a store.
- Leave changes you didn't make alone. Clean up scratch files, processes and emulators.
- Say what was checked and how, and what wasn't.
- Short replies; one recommendation, not a list of options.
