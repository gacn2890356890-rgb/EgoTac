# SuperMimic 训练与评测归档

[下载全部权重、代码及实验记录（GitHub Release）](https://github.com/gacn2890356890-rgb/EgoTac/releases/tag/supermimic-archive-20261115)

归档保留最终 320000 步模型、三小时检查点、早期小规模模型、ONNX 导出、完整优化器/回放/RNG 续训状态，以及训练配置、实际代码、本机 Git 历史、离线 W&B 记录、评测原始逐步样本和失败名单。原始第三方 BONES 动作数据与 Python 环境未打包；数据筛选名单和元数据单独保存。

最终模型在重建名单的全部 129491 条动作上失败 12 条，成功率 99.9907329%，平均归一化回报 0.9337696698。已从全部 47341890 个有效逐步样本独立复算。该结果是训练集覆盖评测，名单与作者未发布的最终名单存在差异。三小时模型对照在初始化阶段中断，没有有效评测数值；未复现的基线、消融及真机结果不计入本结果。

## 下载与恢复

大归档分成四部分，按文件名顺序拼接后解压：

```bash
cat remote-artifacts.tar.gz.part* > remote-artifacts.tar.gz
mkdir restored
tar -xzf remote-artifacts.tar.gz -C restored
```

最终权重位于 `remote-project/logs/SuperMimic-GenCoord-G1/2026-11-13_21-21-37/`。其中 `training_state.pt` 的保存步号是 319999，对应已完成 320000 次采集迭代。恢复训练会重新初始化仿真 episode，不声称逐 bit 延续物理轨迹。

Release 中的 `ARCHIVE_README.md` 说明全部文件用途；`SHA256SUMS`、`LOCAL_FILES.json`、`REMOTE_FILES.json` 提供文件清单和哈希；`RESTORE_VERIFIED.json` 和 `LOCAL_DOWNLOAD_VERIFIED.json` 记录从 GitHub 下载后进行的核验。751 个服务器文件、640 个本机源文件、6 份数据元数据、25 份权重/导出文件以及两份 Git 历史已检查。

按用户要求，核验后清理本机本项目与专用 mjlab 目录及临时归档副本。服务器原件保留。
