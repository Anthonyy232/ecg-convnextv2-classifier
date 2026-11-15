# ecg-convnextv2-classifier media

Source for the pipeline figure in the [ecg-convnextv2-classifier](https://github.com/Anthonyy232/ecg-convnextv2-classifier) README: `.github/readme/pipeline-light.png` and `pipeline-dark.png` on `master`.

## Rebuild

Needs Node only (no Python). From a checkout of `master`:

```bash
git worktree add ../ecg-media media
cd ../ecg-media
npm ci && npx playwright install chromium
node readme/build.cjs ../ecg-convnextv2-classifier
```

That writes both PNGs (2048x1026, 1280x641 CSS px at 1.6x, rounded card on transparency) into `../ecg-convnextv2-classifier/.github/readme/`. Commit them on `master`. No network is needed after the install and there are no input files: everything is drawn by `readme/build.cjs`.

## What the script draws

- **ECG trace (01):** a synthetic 10 s window at 100 Hz, made in closed form from Gaussian P, Q, R, S and T waves on fixed beat times, with one early beat, then min-max scaled as `src/preprocess.py` does. It is not MIT-BIH data.
- **S-Transform panel (02):** a JavaScript port, for illustration, of `perform_s_transform` in `src/signal_processing.py`, run on that trace from 0 to 40 Hz. It shows the magnitude with each frequency row scaled to its peak, 0 Hz at the bottom. If that function changes, update the port in `build.cjs` to match.
- **ConvNeXt V2 (03):** blocks per stage from `src/model.py` (2, 2, 6, 2). Bar heights are a fixed design, not to scale.
- The text in the cards and the training line restate the README; keep them in step with it.

## Layout

- `readme/build.cjs`: computes the trace and the panel, then draws the figure in Chromium.
- `fonts/`: Instrument Serif regular and italic (latin subset), under the SIL Open Font License in `fonts/OFL.txt`.
