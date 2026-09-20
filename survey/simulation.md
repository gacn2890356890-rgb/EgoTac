# Simulation-to-Real / Simulation-to-Data

[主报告与原文截图](experimental_setup.md) · 更新：2026-09-19

| 工作 | 平台/机制 | 数据从哪里来、如何使用 | 与本项目相关的边界 | 原文 |
| --- | --- | --- | --- | --- |
| Tabero | Isaac Sim / Isaac Lab；Taxim等 | 重建原MuJoCo等任务，TCP对齐、高增益PD重放；渲染腕/第三视角RGB-D与GelSight | 原轨迹在新引擎不一定成功，要检查保留率；不是人体指套测量 | [pp.3–4](https://arxiv.org/abs/2605.27886) |
| Tactile Genesis | Genesis World | 同任务/手/策略，比较传感表示、覆盖、分辨率与噪声；teacher→student | 可帮助筛布局，但仿真力或位移需真实标定验证 | [pp.4–7](https://arxiv.org/abs/2606.22332) |
| DEX-X | IsaacLab / PPO / DAgger | 人类视频重建→仿真交互→视觉触觉student→真机 | 有真实30次/条件消融；深度和机器人指尖力不同于人体指套 | [pp.5–7,13–14](https://arxiv.org/abs/2609.07747) |
| DexFlow | Isaac Gym / PhysX | MANO→ShadowHand/Allegro，接触/穿透优化；六向重力100步检验 | 几何接触不是实测力；292k为右手轨迹帧 | [pp.3–6](https://arxiv.org/abs/2505.01083) |
| EgoPHI | SOFA | ARCTIC手物几何生成物理标签，另做真实物体端光学验证 | 手部力图预测与物体端GT需要分开 | [论文及pp.12–13](https://arxiv.org/abs/2608.13014) |
| REFORCE | 仿真交互训练力感知修正 | 人类动作/力作为参考，真机指尖反馈在线修正 | 直接重放、手工导纳、学习修正使用同参考；不能只看轨迹像不像 | [pp.3–7](https://arxiv.org/abs/2608.15560) |
| EgoEngine | Real-to-Sim-to-Real流程 | 人类ego视频→机器人外观/可执行动作→真机 | 相机/机器人坐标对齐仍需标定；未提供人体实测触觉 | [pp.14–15,26](https://arxiv.org/abs/2606.12604) |
| EgoHOI | 视频世界模型 | HOT3D的图像与几何标注驱动HOI视频生成 | 逼真视频不能作力GT | [p.23](https://arxiv.org/abs/2603.13615) |
| UMR | MuJoCo与几何对应 | 跨人体/人形机器人动作表示，Unitree G1评估 | 重点全身重定向；非指套采集 | [p.5](https://arxiv.org/abs/2609.02134) |

## 下一步建议

先确定本项目实际能测到的量：二值接触、原始强度、标定法向力，或光学形变图。再选择能产生同类信号的仿真表示。不要因为仿真能输出全手力，就把稀疏指套观测包装成全手真值。

短期可做小范围离线复用：保存真实手/物体几何与传感器布局，检查仿真接触位置能否与实测通道对应。真实采集的零点、漂移、同步与遮挡验证仍必须保留。本轮没有运行上述模拟器或复现论文性能。
