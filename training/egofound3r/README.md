# EgoFound3R：原版与世界模型版

同一个代码库的两个版本，用于在集群上做全量训练。

| 目录 | 内容 |
|---|---|
| `EgoFound3R_original/` | 官方代码（[MINT-SJTU/EgoFound3R](https://github.com/MINT-SJTU/EgoFound3R)，commit `264f4df`），一个字节都没改。`scripts/verify_original.sh` 可以核对。 |
| `EgoFound3R_world/` | 同样的代码，只多了 `egohandmetric_prompt/world_model.py`，并写死了 `requirements.txt` 的版本。 |
| `egotac/` | 两个版本共用的训练、数据和评估代码。 |

**官方仓库只有推理代码**，没有训练脚本、数据加载器和数据配置：它的 `data/` 目录里只有 MANO 相关的小文件，`configs/default.toml` 只有模型和运行参数。所以 `egotac/` 是按论文（Eq. 3-8、Table 5、Table 7）补写的，不是官方代码。

## 世界模型版改了什么

设计来自 [`next_experiments.md`](https://github.com/gacn2890356890-rgb/EgoTac/blob/71ec8b2/training/docs/next_experiments.md)：只看过去，预测未来的视觉和触觉。

```
过去 16 帧 → EgoFound3R 官方模型 → 每只手 216 个点的特征
                                    ├─ 当前帧：rectified flow 生成接触和距离（替换 Eq. 5-6）
                                    └─ 动态模块：未来查询读取过去 → 未来 latent
                                          ├─ 未来 5 帧（0.5 s）的接触和距离
                                          └─ 未来视觉特征（目标来自冻结的 DINO 编码器）
```

- 未来帧不会输入模型，所以不会偷看未来。
- 推理输出的字段、形状和文件都与原版相同，只有 50 个触觉字段的数值不同（`scripts/compare_io.py` 已核对）。
- 新增损失：`tactile_flow`、`future_flow`、`future_visual`。
- 全量配置下可训练参数约 2.64 亿，其中头部 5266 万；原版约 2.12 亿。原版架构不改，否则加载不了官方权重。

## 集群上怎么跑

Python 写死为 3.10.12（见 `.python-version`，其他版本会直接报错），依赖版本全部写死在 `requirements.txt`，即使系统 Python 没有 pip 也能装。在仓库的 `training/egofound3r/` 目录下执行：

```bash
cp cluster.env.example cluster.env   # 填上大容量存储路径 EGOTAC_ROOT 和 MANO_DIR
```

```bash
bash scripts/setup.sh
```

```bash
bash scripts/download_weights.sh     # EgoFound3R.pt 需要先在 Hugging Face 申请权限
```

```bash
sbatch -p <分区> scripts/slurm_convert.sbatch
```

```bash
sbatch -p <分区> scripts/slurm_train.sbatch baseline
```

```bash
sbatch -p <分区> scripts/slurm_train.sbatch world
```

MANO 文件需要自己从 [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) 下载，放到 `MANO_DIR`。`scripts/preflight.py` 会在排队前检查权重、数据和序列数，有问题直接报错。任务被抢占后重新提交会从 `latest.pt` 续训。

## 训练参数

`configs/full.toml` 的大部分值来自论文 Table 5：AdamW，学习率 2.5e-4 余弦衰减到 1e-5，warmup 100 步，3200 步，256x256，梯度裁剪 1.0，总 batch 64。

`train.batch_size` 是每张卡的 batch，默认 8。显存不够就调小，梯度累积会自动补足总 batch 64，例如：

```bash
sbatch -p <分区> scripts/slurm_train.sbatch world --set train.batch_size=4
```

实测峰值显存（TITAN RTX，stride 5，每卡 batch 2）：原版 15.9 GiB，世界模型版 19.3 GiB。stride 越小越占显存：24 GB 卡上 stride 1 和 2 在 batch 2 就会爆显存。

## 数据

规模按论文 Table 7。论文的训练序列共 3335 条，加上全部 STERA-10M；评估共 2378 个窗口。

代码不下载任何数据，只读集群上已有的文件：在 `cluster.env` 里填好 `DATA_*` 路径（见 `cluster.env.example`），再运行 `scripts/slurm_convert.sbatch`。这些目录只读，转换结果写到 `$EGOTAC_ROOT/data/clips_full`。

| 数据集 | 路径变量 | 训练序列 | 评估窗口 | 真实数据验证 |
|---|---|---|---|---|
| OakInk-v2 | `DATA_OAKINK2` | 565 | 400 | 跑过少量序列 |
| TACO | `DATA_TACO` | 839 | 400 | 跑过少量序列 |
| HOT3D Aria | `DATA_HOT3D` | 136 | 400 | 跑过 4 个片段，关节点落在手上 |
| HOI4D | `DATA_HOI4D` | 1431 | 461 | **未验证** |
| H2O | `DATA_H2O`、`DATA_H2O_OBJECTS` | 138 | 283 | **未验证** |
| ARCTIC | `DATA_ARCTIC` | 226 | 434 | **未验证** |
| STERA-10M | 无 | 全部 | 无 | **没有转换器** |

每个转换器开头写了它期望的目录结构。**未验证**的转换器是照官方文档和官方代码写的，只用合成数据测过。第一次在集群上运行时，先转几条并用眼睛检查：

```bash
"$EGOTAC_VENV/bin/python" -m egotac.data.convert_h2o --root "$DATA_H2O" --object-dir "$DATA_H2O_OBJECTS" --output /tmp/h2o_check --mano-dir "$MANO_DIR" --limit 3
```

```bash
"$EGOTAC_VENV/bin/python" scripts/check_clip.py <某个 .npz> check.jpg
```

点要落在手上，否则不要全量转换。某个数据集暂时不用时，加 `--set data.weights.hoi4d=0` 这样的参数，但那样就不是论文规模。

官方仓库没有公开数据处理代码（[issue #1](https://github.com/MINT-SJTU/EgoFound3R/issues/1) 问过训练代码，没有回复），论文也不公开序列名单，所以这里能做到的是**数据集、数量、窗口、标签定义和论文一致**，不能保证与作者逐条相同。HOI4D 的铰接物体（如笔记本）只有手与手的接触标签，因为官方文档没有说明部件和姿态的对应关系。

## 论文没公开、由我们决定的

都在 `configs/` 或 `egotac/losses.py` 的开头：除 `geom` 外的损失权重、数据集和 stride 的采样比例、可见性定义、256x256 的裁剪方式，以及世界模型的全部设计（过去 16 帧、未来 5 帧、头部规模）。

## 验证到哪一步

结果见 [docs/smoke_results.md](docs/smoke_results.md)。

- **没有跑过：** 真正的全量训练、真实集群上的 Slurm（只用替身命令测过脚本）、完整数据的转换（只测过每个数据集 1-2 条序列）。
- **没有实现：** 场景深度和尺度的监督。相机空间的手部结果不依赖它，世界坐标指标会受影响。

官方代码和权重是 CC BY-NC-SA 4.0，不可商用；权重和数据不进仓库。
