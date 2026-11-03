# 外置训练数据格式

仓库只提供 schema 和人工占位示例，不提供任何真实训练数据。当前入口读取先前已经提取的主视角特征，原始视频、触觉、冻结大模型权重和特征提取链路需要自行准备。

## Manifest

`dataset.json` 包含 `task`、`data_revision`、`stats`、`rows`。`task` 必须是 `same_frame_window_estimation`。每条 row 为一个episode，至少包含：

| 字段 | 含义 |
|---|---|
| name | 全局唯一episode身份，不能在多个split重复 |
| task | 任务类型，训练按任务等概率采样 |
| split | train、val；独立评价可另加test或diagnostic |
| file | 相对于data-root的bundle路径，或显式外部绝对路径 |
| frames | 该episode的特征帧数，主要用于记录 |

归一化与先验只能在train计算：

| stats字段 | shape | 定义 |
|---|---|---|
| geometry_mean / geometry_std | [14] | 每一几何特征维度的训练均值/标准差，跨区域共享 |
| contact_prior | [12] | 每区域响应≥1的有效训练占比 |
| positive_log_median | [12] | 正例 `log1p(response)` 中位数，已是log空间 |
| response_scale | [12] | 训练响应误差的正归一化尺度 |
| train_patch_mean / train_patch_median | [12] | 只用于常数响应基线 |

当前入口消费已有训练统计，不自动从val/test重算。`export-manifest` 只复制原train统计及train/val记录。

## Bundle

每个 `.pt` 文件是张量为主的字典，以 `torch.load(weights_only=True, mmap=True)` 读取。应以标准 `torch.save` zip格式保存。

| 字段 | shape | 内容 |
|---|---|---|
| grid | [T,64,1024] | 主RGB全局8×8冻结特征 |
| local_grid | [T,32,1024] | 两只手各4×4裁剪特征 |
| local_uv | [T,32,2] | 局部token映射至主图的归一化坐标 |
| local_valid | [T,32] bool | 局部特征可用性 |
| geometry | [T,12,14] | 主RGB预测的手部区域几何，不是GT pose |
| uv | [T,12,2] | 区域中心归一化主图坐标 |
| geometry_valid | [T,12] bool | 几何可用性 |
| frame_indices | [T] integer | 原视频对应帧索引 |
| timestamps | [T] | 秒，严格递增 |
| target | [T,12] | 非负相对触觉响应，仅监督/评价 |
| target_valid | [T,12] bool | 传感器/源数据有效性 |
| native_bin_peak | [T,12] optional | 原生采样小窗口峰值，仅监督 |
| native_bin_peak_valid | [T,12] bool optional | 峰值有效性，与峰值同时提供 |

数值先清理为有限值，同时保留真实mask。无效标签不能因为填零就变成有效非接触。所有train bundles统一包含或统一省略native峰值监督。

模型forward只接收视觉及预测几何字段。target、target_valid和native峰值不进入模型输入。输入context最多40帧，短episode可评价但不能用于超过其长度的训练窗口。

## 原特征来源

既有管线使用冻结的VGGT-Omega视觉特征和EgoFound3R预测几何，官方代码与权重由各自上游提供。本仓库不再分发第三方模型实现或权重，复用外置bundle不需要下载这些模型。

当前包不包含从原始HumanTouch parquet到这些bundle的独立提取器。原始特征提取还依赖相机处理、模型版本、手部ROI和监督对齐；只有本数据格式的模拟张量不能替代真实实验。

下一轮未来预测要重新核实特征和几何的prefix时间依赖。当前bundle接口没有证明其可直接用于因果未来任务。
