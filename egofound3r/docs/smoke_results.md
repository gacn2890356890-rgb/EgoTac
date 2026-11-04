# Smoke test results

Run on `msc-vision` (2 x TITAN RTX, 24 GB). Official weights versus 600 steps of fine-tuning from them,
batch size 2, stride 5, 256x256 center crop. Training data: 117 clips from 12 OakInk-v2 sequences.

- `official`: official weights, no training.
- `baseline`: official weights plus fine-tuning, paper objective.
- `diffusion`: same, but the tactile head starts **untrained** and replaces Eq. 5-6.

## How to read this

This shows the pipeline works. It does not show which variant is better.

- The baseline's contact head starts trained and the diffusion head starts from nothing, so 600 steps
  favors the baseline.
- There are 29 OakInk windows and 51 TACO windows, from a handful of sequences. Differences of this
  size can be noise: after only six steps at a near-zero learning rate, the two variants already differed
  by about 1 mm RR-MPJPE on a 4-window check.
- Geometry training is the same in both variants, yet RR-MPJPE differs by about 2 mm on OakInk. The
  diffusion loss also reaches the shared features, which may contribute, but this was not separated
  from noise.
- TACO was not trained on, and every number there is worse than the official weights. Fine-tuning on
  one dataset alone costs generality, which is why full training must mix datasets.

## Converter validation: official weights against the paper

Official weights on our converted clips. This tests the converters, not the model. The sample is small,
and these sequences may be in the authors' training set, so it checks magnitude only.

| | OakInk-v2 paper | OakInk-v2 ours | TACO paper | TACO ours |
|---|---|---|---|---|
| RR-MPJPE (mm) | 17.77 | 17.15 | 18.54 | 22.14 |
| PA-MPJPE (mm) | 8.02 | 7.14 | 8.13 | 8.78 |
| Contact precision, joints | 0.510 | 0.576 | 0.654 | 0.790 |
| Contact recall, joints | 0.665 | 0.698 | 0.777 | 0.726 |
| Contact F1, joints | 0.493 | 0.583 | 0.697 | 0.746 |
| Visibility F1, joints | 0.802 | 0.831 | 0.880 | 0.816 |
| Windows | 400 | 32 | 400 | 51 |

"Ours" here is 512-wide inference, on held-out clips (OakInk) and a random 24-sequence subset (TACO).

## Input/output contract

Official weights with and without the head, on a real video: 191 tensors compared, with no difference in
names, shapes, dtypes, or output files. 50 tactile tensors changed and 0 others did, bit for bit.

## Comparison

### oakink2

| Metric | official | baseline | diffusion |
|---|---|---|---|
| RR-MPJPE (mm) | 16.90 | 17.99 | 19.92 |
| PA-MPJPE (mm) | 7.32 | 7.50 | 7.48 |
| RR-MPVPE (mm) | 17.96 | 18.30 | 19.09 |
| Raw MPJPE (mm) | 83.64 | 81.83 | 99.18 |
| Contact F1, joints | 0.461 | 0.558 | 0.469 |
| Contact AP, joints | 0.775 | 0.825 | 0.766 |
| Contact F1, vertices | 0.442 | 0.553 | 0.457 |
| Contact AP, vertices | 0.725 | 0.795 | 0.747 |
| Distance MAE all, joints (mm) | 40.07 | 42.91 | 35.83 |
| Distance MAE GT-contact, joints (mm) | 8.80 | 10.55 | 11.13 |
| Distance MAE all, vertices (mm) | 40.27 | 43.22 | 35.47 |
| Distance MAE GT-contact, vertices (mm) | 8.92 | 11.18 | 10.99 |
| Visibility F1, joints | 0.839 | 0.857 | 0.853 |
| Visibility F1, vertices | 0.687 | 0.831 | 0.831 |
| Windows | 29 | 29 | 29 |
### taco

| Metric | official | baseline | diffusion |
|---|---|---|---|
| RR-MPJPE (mm) | 24.32 | 32.05 | 31.50 |
| PA-MPJPE (mm) | 9.62 | 12.64 | 12.52 |
| RR-MPVPE (mm) | 22.75 | 28.84 | 29.37 |
| Raw MPJPE (mm) | 123.31 | 126.19 | 133.97 |
| Contact F1, joints | 0.750 | 0.595 | 0.471 |
| Contact AP, joints | 0.817 | 0.761 | 0.747 |
| Contact F1, vertices | 0.671 | 0.486 | 0.474 |
| Contact AP, vertices | 0.730 | 0.695 | 0.635 |
| Distance MAE all, joints (mm) | 20.00 | 30.00 | 24.22 |
| Distance MAE GT-contact, joints (mm) | 5.93 | 17.11 | 16.75 |
| Distance MAE all, vertices (mm) | 21.65 | 29.73 | 25.21 |
| Distance MAE GT-contact, vertices (mm) | 6.84 | 18.54 | 16.74 |
| Visibility F1, joints | 0.791 | 0.764 | 0.766 |
| Visibility F1, vertices | 0.685 | 0.717 | 0.705 |
| Windows | 51 | 51 | 51 |
