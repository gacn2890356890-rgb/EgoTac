# EgoTouch training code

主视角视频派生特征上的触觉 decoder 训练与评价。数据、视觉骨干权重、checkpoint、特征缓存、预测数组和虚拟环境均保存在 Git 外部。

## 当前实现

| 内容 | 状态 |
|---|---|
| v15 全局 / 手部局部特征触觉 decoder | 已有实现，迁移为可配置训练入口 |
| 接触概率、条件幅值及连续响应 p×a | 已实现 |
| 完整小模块训练、验证选模、warm start、断点恢复 | 已实现 |
| MPS 与单 CUDA GPU 的显式设备选择 | 已实现，无 CPU 模型 fallback |
| 未来视觉＋触觉联合预测、双流和接触门控 | 待实现，见 [下一轮方案](docs/next_experiments.md) |
| 腕视频生成及大模型端到端微调 | 本训练包不包含对应实现 |

**当前可训练入口是窗口内同帧触觉估计。** 时间模块读取整个观测窗口，窗口中间的输出可以使用后帧。未来预测需要新的时间索引、合法 prefix 特征和预测 query，不能将此入口的成绩称为世界模型未来预测。

`global` 使用全局主视角特征与从同一视频预测的手部几何；`hand` 在此基础上加入手部裁剪 tokens。两者均不输入真实触觉、真实腕视频或 GT pose。12 通道为当前 HumanTouch 区域读出，输出单位为相对传感器响应。

## 1. 安装

以下命令从仓库根目录执行。建议 Python 3.12；使用项目独立环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./training
```

CUDA 机器先按其驱动安装匹配的 PyTorch，再执行安装命令。MPS 使用支持 Apple Silicon 的 PyTorch。训练主体不导入 VGGT/EgoFound3R/WristWorld，读取预先生成的冻结特征即可。

依赖仅为 PyTorch、NumPy。原本地开发环境记录见 [环境参考](docs/environment_reference.json)，该记录不代表在其他 CUDA 环境已验证，也不是强制 wheel 清单。

## 2. 外置数据与权重

预期目录：

```text
data/humantouch/           # ignored
  dataset.json            # split / train-only stats / relative bundle paths
  bundles/
    <episode>.pt
checkpoints/              # ignored; optional historical initialization
runs/                     # ignored; output from new fits
```

从已有本地 v15 元数据导出可迁移清单，输出仍放在 ignored 目录：

```bash
egotouch-export-manifest \
  --source /absolute/path/to/expanded_train_v15/dataset.json \
  --output data/humantouch/dataset.json
```

该命令只改写 train/val 元数据和相对路径，不复制数据或权重。将对应 bundles 自行放在 `data/humantouch/bundles/`，或通过 `--data-root` 指定外部根目录。[数据字段说明](docs/data_format.md)列出每个 tensor 的 shape 和来源要求。

首次核对数据：

```bash
egotouch-check-data --manifest data/humantouch/dataset.json --split train
egotouch-check-data --manifest data/humantouch/dataset.json --split val
```

这些命令只读 tensor、shape、mask 和时间索引，不运行神经模型。仓库中的 `examples/dataset.example.json` 只有人工占位元数据，不能用于训练或算实验结果。

## 3. 正式训练

本地 MPS 的一次手部增强训练：

```bash
bash training/scripts/run_local.sh --seconds 14400 \
  --report runs/v15-hand-seed42/watchdog.json -- \
  --manifest data/humantouch/dataset.json \
  --config training/configs/v15_hand.json \
  --device mps \
  --output runs/v15-hand-seed42
```

每次最多 10000 步 / 4 小时，按任务、episode、40 帧窗口采样。每500步完整验证，按等任务连续响应 MAE 选 checkpoint，AP打破平局。概率阈值只在val选择。保留0步候选，不强行要求训练后 checkpoint 获胜。

全局对照将配置换为 `training/configs/v15_global.json`，输出换成新目录。用 `--seed 43` 或 `--seed 44`补齐种子。所有实验串行执行。

单 CUDA 卡示例：先确认候选物理卡两次采样均没有其他 compute、利用率≤5%，间隔至少30秒，记录空闲显存，再固定该卡UUID。

```bash
CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-AVAILABLE-UUID \
bash training/scripts/run_local.sh --seconds 14400 \
  --report runs/v15-hand-cuda-seed42/watchdog.json -- \
  --manifest data/humantouch/dataset.json \
  --config training/configs/v15_hand.json \
  --device cuda \
  --output runs/v15-hand-cuda-seed42
```

显式 CUDA 入口要求只暴露一张卡。MPS不会切到CPU运行模型。迁移代码不等于已经在CUDA上验证。

默认小模块随机初始化。如果需要从外置历史 checkpoint 接续，显式增加：

```text
--init-checkpoint /absolute/path/to/old-head.pt
```

warm start 严格匹配模型参数，重置当前 train-only 先验并重新建立 optimizer。历史 v15 结果从上一轮10000步权重继续训练；默认随机初始化不等于复现该历史训练预算或成绩。两组对照必须匹配初始化来源。

## 4. 断点恢复

```bash
bash training/scripts/run_local.sh --seconds 14400 \
  --report runs/v15-hand-seed42/watchdog-resume.json -- \
  --manifest data/humantouch/dataset.json \
  --config training/configs/v15_hand.json \
  --device mps \
  --output runs/v15-hand-seed42 \
  --resume runs/v15-hand-seed42/last.pt
```

恢复模型、optimizer、随机数状态和已完成步数，累计内部时间仍受原预算限制。配置、统计量、数据revision和episode身份必须相同。父进程硬限时可能在两次保存之间中断，恢复从最近完成的 `last.pt` 继续，不能假称恢复未保存的更新。

## 5. 评价

```bash
egotouch-evaluate \
  --manifest data/humantouch/dataset.json \
  --checkpoint runs/v15-hand-seed42/best.pt \
  --split val --device mps \
  --output runs/eval-v15-hand-seed42 \
  --save-predictions
```

保持训练归一化和checkpoint中的val阈值。报告连续MAE/RMSE、接触AP/F1、接触区误差、事件起始F1、抖动及全零/训练均值/中位数基线。`AUPRC`是旧字段，计算的是average precision，与`AP`同义，不混用梯形PR积分。指标先episode平均，再任务等权平均。

仅生成预测可加 `--prediction-only`。新独立测试集另用 `--split test`，测试阶段不调阈值或选模型。旧44episode确认集已经评估过，后续只能作为已见诊断组。

## 6. 产物与源码

```text
training/
  configs/                 global / hand 参数
  src/egotouch/model.py    原v13完整空间、区域、时间触觉头
  src/egotouch/losses.py   v15接触、幅值、响应、时序、峰值监督
  src/egotouch/train.py    可迁移训练、选模与断点恢复
  src/egotouch/evaluate.py 固定checkpoint评价
  scripts/                有限运行入口
  tests/                  指标与数据接口测试
  docs/                   数据格式、协议、下一轮架构计划
```

每次训练输出 `run.json`、`progress.json`、`history.json`、`best.pt`、`last.pt` 和 `status.json`。记录PID、命令、版本、seed、实际更新、参数变化、覆盖episode、退出码、耗时和峰值。MPS真实峰值未知时为null，CUDA报告PyTorch分配峰值。

数据、权重、模型下载凭据均不在本仓库。`.gitignore`只影响未跟踪文件，提交前仍需检查暂存区。

[代码包检查记录](docs/validation.md)包含接口测试、wheel构建和本地MPS合成链路验证的范围。
