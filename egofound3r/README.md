# EgoFound3R training and a diffusion tactile head

Training and evaluation for [EgoFound3R](https://github.com/MINT-SJTU/EgoFound3R), whose official
release is inference only, in two variants that share all code:

- `baseline`: the original model trained with the paper's objective (Eq. 3-8, Table 5).
- `diffusion`: the same, except the contact and distance losses (Eq. 5-6) are replaced by a
  rectified-flow head over the 216 points of each hand.

The official code is not modified. The head attaches through PyTorch hooks and returns the same
fields, so downstream code, including overlay videos, cannot tell the difference.

Measured results are in [docs/smoke_results.md](docs/smoke_results.md).

## Setup

```bash
bash scripts/setup.sh
```

You also need three things this repository cannot redistribute:

| File | Source |
|---|---|
| `EgoFound3R.pt` | [Frank-F2021/EgoFound3R](https://huggingface.co/Frank-F2021/EgoFound3R), gated, CC BY-NC-SA 4.0 |
| `vggt_omega_1b_512.pt` | [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) |
| `MANO_LEFT.pkl`, `MANO_RIGHT.pkl` | [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), requires accepting its license |

The paths in `configs/` and the defaults in `scripts/` point at the `msc-vision` server. Change them,
or set `EGOTAC_ROOT`, `EGOTAC_PYTHON`, and `EGOFOUND3R_ROOT`.

## Commands

```bash
# Original versus diffusion on the same data and steps, then a comparison table.
IO_VIDEO=/path/to/video.mp4 bash scripts/smoke_test.sh 0 1 600

# Convert datasets. Both resume when rerun.
bash scripts/convert_oakink_full.sh /path/to/mano 6 0
bash scripts/convert_taco_full.sh /path/to/mano 6 0

# Full training, one variant per run. Rerunning resumes from latest.pt.
bash scripts/train_full.sh baseline 8
bash scripts/train_full.sh diffusion 8
sbatch scripts/slurm_train.sbatch baseline

# Evaluate anything, including the official weights.
python -m egotac.evaluate --upstream-config <default.toml> --checkpoint <file.pt> \
    [--variant diffusion] --clips <clips_root> --split test --image-size 512 --image-mode aspect

# Inference on a video through the official script, with the head installed.
python -m egotac.infer_video --tactile-checkpoint <diffusion.pt> \
    --config <default.toml> --checkpoint <EgoFound3R.pt> --video-path <video.mp4> --output-dir <dir>
```

A training checkpoint keeps the official layout, so the official inference scripts load it directly.
A `diffusion` checkpoint loaded that way gives the geometry but the untrained original contact heads,
so use `egotac.infer_video` for its tactile output.

## Dataset status

| Dataset | Converter | Access |
|---|---|---|
| OakInk-v2 | Done, checked against the paper | Public, CC BY-SA 4.0 |
| TACO | Done, checked against the paper | Public, CC BY 4.0 |
| HOT3D Aria | Not written | Public mirror, under its own license agreement |
| ARCTIC | Not written | Requires registration |
| H2O | Not written | Requires a username and password from the authors |
| HOI4D | Not written | Requires its own application |
| STERA-10M | Not written | Access not confirmed |

A converter is written against real files and only then trusted, because a wrong convention produces
plausible-looking labels. To add a dataset, produce `FrameInput`s and call `build_clip`, then pass both
checks: `scripts/verify_labels.py` must show labels on the hands, and the official weights must score
close to the paper on the converted clips. Add the dataset to `[data.weights]`; a listed dataset with
no clips stops training at startup.

## Values the paper does not publish

These are our choices, all in `configs/` or at the top of `egotac/losses.py`:

- All loss weights except `geom`, and the confidence weight `alpha`.
- Sampling shares per dataset, and stride proportions.
- The visibility definition: a z-buffer test against hands and objects, with tolerances of 12 mm for
  vertices and 25 mm for joints.
- How 256x256 is formed from a wide image. We use a center crop, which loses some off-center hands.
- The train/test split. Sequences are assigned by hash, not the authors' split.
- Extra terms: hand scale, root translation, and in-view. The first exists because Eq. 3 normalizes
  scale away.

## Known limits

- **Metric head:** `L_3R` here covers camera rotation and translation. Depth is not converted, so the
  scene depth term and metric-scale supervision are missing. Camera-space hands do not depend on them,
  but world-space metrics will be affected.
- **Temporal:** the head generates each frame independently; nothing models time yet.
- **Slurm:** `slurm_train.sbatch` has not run on a real cluster. Multi-GPU was tested on 2 GPUs, one node.
- **Stride:** on a 24 GB GPU, stride 1 and 2 run out of memory at batch size 2, so `full.toml`'s default
  strides need a larger GPU. Change `train.strides` otherwise.
- **Not tested end to end:** `convert_oakink_full.sh` and `convert_taco_full.sh`. Their converters ran
  on subsets; the streaming and full-fetch modes did not.

## Silent memory corruption

On `msc-vision`, reading the same intact file twice gave different bytes, and a pattern test found
flipped bits in RAM, with ECC disabled. Run `memtest86+` there. Until it is fixed, `egotac/integrity.py`
loads weights by majority vote across reads and repairs the live copy, and training skips steps with a
non-finite loss or gradient rather than ending the run. Each run writes `weight_integrity.json`.
Clips and checkpoints written on that machine have no such protection.
