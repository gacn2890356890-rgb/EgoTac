# EgoFound3R Training

**[EgoFound3R](https://github.com/MINT-SJTU/EgoFound3R) 的训练代码。** 官方仓库只发布了推理代码，这里按论文（Eq. 3-8、Table 5、Table 7）补写了训练、数据转换和评估，可以训练两个版本：

| 版本 | 代码目录 | 内容 |
|---|---|---|
| `baseline` | `EgoFound3R_original/` | 官方架构，代码与官方一字不差 |
| `world` | `EgoFound3R_world/` | 世界模型架构，输入输出与官方相同 |

下面的命令都在 `training/egofound3r/` 目录下运行。

## Installation

```bash
bash scripts/setup.sh
```

Python 写死为 3.10.12，其他版本会直接报错。依赖版本全部写死在 `requirements.txt`，环境装在 `.venv/`。

然后创建配置文件，填上存储路径和数据路径：

```bash
cp cluster.env.example cluster.env
```

| 变量 | 内容 |
|---|---|
| `EGOTAC_ROOT` | 大容量存储，放权重、转换后的数据和训练结果 |
| `MANO_DIR` | `MANO_LEFT.pkl` 和 `MANO_RIGHT.pkl` 所在目录，从 [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) 获取 |
| `HF_TOKEN` 或 `VGGT_CHECKPOINT` | 用于获取 VGGT-Omega 权重，见 [Checkpoints](#checkpoints) |
| `DATA_*` | 服务器上各数据集的路径，见 [Data](#data) |

## Checkpoints

- **VGGT-Omega**：冻结的几何主干，训练必需。开始训练时会自动下载 [`vggt_omega_1b_512.pt`](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt)（4.6 GB），已有就跳过。这个文件要申请权限，所以要在 `cluster.env` 里填有权限账号的 `HF_TOKEN`；如果服务器上已有这个文件，填 `VGGT_CHECKPOINT` 指向它就不会下载。
- **EgoFound3R**：训练不需要。除 VGGT-Omega 外，其余部分都从随机初始化开始训练。如果想从官方权重开始，把 `configs/full.toml` 里的 `checkpoint` 指向 [`EgoFound3R.pt`](https://huggingface.co/Frank-F2021/EgoFound3R)。

也可以提前手动下载：

```bash
bash scripts/download_weights.sh
```

## Data

代码不下载数据，只读取 `DATA_*` 指向的目录，转换结果写到 `$EGOTAC_ROOT/data/clips_full`。序列数和窗口数按论文 Table 7，每个转换器开头写了它期望的目录结构。

```bash
bash scripts/convert_all.sh
```

| 数据集 | 路径变量 | 训练序列 | 评估窗口 | 转换器 |
|---|---|---|---|---|
| OakInk-v2 | `DATA_OAKINK2` | 565 | 400 | 跑过少量真实数据 |
| TACO | `DATA_TACO` | 839 | 400 | 跑过少量真实数据 |
| HOT3D Aria | `DATA_HOT3D` | 136 | 400 | 跑过少量真实数据 |
| HOI4D | `DATA_HOI4D` | 1431 | 461 | 只用合成数据测过 |
| H2O | `DATA_H2O`、`DATA_H2O_OBJECTS` | 138 | 283 | 只用合成数据测过 |
| ARCTIC | `DATA_ARCTIC` | 226 | 434 | 只用合成数据测过 |
| STERA-10M | 无 | 全部 | 无 | **没有转换器，目前不参与训练** |

只用合成数据测过的数据集，全量转换前先转 3 条，确认关节点落在手上：

```bash
. scripts/env.sh; "$PY" -m egotac.data.convert_h2o --root "$DATA_H2O" --object-dir "$DATA_H2O_OBJECTS" --output /tmp/h2o_check --mano-dir "$MANO_DIR" --limit 3
```

```bash
. scripts/env.sh; "$PY" scripts/check_clip.py <某个 .npz> check.jpg
```

## Usage

训练世界模型版，最后的数字是 GPU 数量：

```bash
bash scripts/train_full.sh world 8
```

训练原版：

```bash
bash scripts/train_full.sh baseline 8
```

只用其中几张卡时：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_full.sh world 4
```

启动后会先获取 VGGT-Omega 权重，再检查权重、数据和序列数，有问题直接报错。默认配置来自论文 Table 5：AdamW，学习率 2.5e-4 余弦衰减到 1e-5，warmup 100 步，共 3200 步，输入 256x256，梯度裁剪 1.0，总 batch 64。

结果写到 `$EGOTAC_ROOT/outputs/full_<baseline|world>/`：

| 文件 | 内容 |
|---|---|
| `final.pt` | 训练完成后的权重，用于评估和推理 |
| `latest.pt` | 每 200 步保存一次，中断后重新运行同一条命令会从这里续训 |
| `train_log.jsonl` | 每 20 步的损失和学习率 |
| `eval_log.jsonl` | 每 400 步的评估指标 |
| `config.json` | 这次训练的完整配置 |
| `weight_integrity.json` | 加载权重时的核对结果 |

常用参数：

| 参数 | 作用 |
|---|---|
| `--set train.batch_size=4` | 每张卡的 batch（默认 8）。调小时梯度累积会自动补足总 batch 64 |
| `--set train.steps=1000` | 训练步数（默认 3200） |
| `--set train.save_every=100` | 保存 `latest.pt` 的间隔（默认 200） |
| `--set train.eval_every=0` | 评估间隔（默认 400），0 表示训练中不评估 |
| `--set data.weights.hoi4d=0` | 不使用某个数据集。这样就不是论文规模了 |

`--set` 写在 GPU 数量后面，例如：

```bash
bash scripts/train_full.sh world 8 --set train.batch_size=4
```

实测峰值显存（TITAN RTX，stride 5，每卡 batch 2）：原版 15.9 GiB，世界模型版 19.3 GiB。stride 越小越占显存，24 GB 的卡在 stride 1 和 2、batch 2 时就会爆显存。

## Evaluation

```bash
. scripts/env.sh; EGOFOUND3R_ROOT=$(tree_for_variant world) "$PY" -m egotac.evaluate --variant world --upstream-config "$EGOTAC_ROOT/configs/default.toml" --checkpoint "$EGOTAC_ROOT/outputs/full_world/final.pt" --clips "$EGOTAC_ROOT/data/clips_full" --image-mode center_crop --batch-size 4 --output eval_world.json
```

```bash
. scripts/env.sh; EGOFOUND3R_ROOT=$(tree_for_variant baseline) "$PY" -m egotac.evaluate --variant baseline --upstream-config "$EGOTAC_ROOT/configs/default.toml" --checkpoint "$EGOTAC_ROOT/outputs/full_baseline/final.pt" --clips "$EGOTAC_ROOT/data/clips_full" --image-mode center_crop --batch-size 4 --output eval_baseline.json
```

对比两个版本：

```bash
. scripts/env.sh; "$PY" scripts/compare_runs.py baseline=eval_baseline.json world=eval_world.json --output compare.md
```

## Inference

训练完用自己的权重推理，命令和官方一样，输出文件也一样：

```bash
. scripts/env.sh; cd EgoFound3R_world && PYTHONPATH=$PWD "$PY" infer_marker_video.py --config "$EGOTAC_ROOT/configs/default.toml" --video-path path/to/video.mp4 --checkpoint "$EGOTAC_ROOT/outputs/full_world/final.pt" --output-dir outputs/video
```

## Status

- **验证过：** 环境安装，以及两个版本在少量真实数据上的短训练、评估和推理。
- **没有跑过：** 全量训练、全量数据转换、从 Hugging Face 真实下载 VGGT-Omega。
- **没有实现：** STERA-10M 的转换器，以及场景深度和尺度的监督。相机空间的手部结果不依赖后者，世界坐标指标会受影响。
- **论文没公开、由我们决定：** 除 `geom` 外的损失权重、数据集和 stride 的采样比例、可见性定义、256x256 的裁剪方式，以及世界模型的全部设计。官方也没有公开序列名单，所以数据集数量和定义与论文一致，但不保证与作者逐条相同。

## License

官方代码和权重使用 [CC BY-NC-SA 4.0](EgoFound3R_original/LICENSE)，仅限学术研究和教育用途，不可商用。第三方组件保留各自的条款。权重和数据不进仓库。

## Citation

```bibtex
@misc{fu2026egofound3rendtoendegocentrichand,
  title={EgoFound3R: End-to-End Egocentric Hand Reconstruction in World Space with Point-Wise Interaction Attributes},
  author={Hongming Fu and Jingcheng Shi and Wenjia Wang and Binhua Zuo and Bo Zhao},
  year={2026},
  eprint={2610.01210},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.01210}
}
```
