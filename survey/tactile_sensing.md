# Wearable Tactile Sensing · 触觉采集与标定

[主报告](experimental_setup.md) · 更新：2026-09-19

 **已有指套的型号和单位尚未提供。**  下列文献中的全手触觉手套、姿态手套、光学触觉和腕部F/T不可互换。稀疏指套也不能默认生成真实全手压力图。

| 系统 | 人/机器人端、布局 | 原始读数 / 频率 | 标签处理 | 来源 |
| --- | --- | --- | --- | --- |
| TouchAnything | 双手外层JQ织物手套；内层Rokoko测姿态 | 每手16×16原始阵列，原始Hz未找到 | 重排21×21；统一30 Hz保存最近样本 | [pp.4,15–16](https://arxiv.org/abs/2605.13083) |
| EgoTac | 双手每手60指+72掌力通道 | 132力通道/手；原始Hz未找到 | 去弯曲通道；插值到MANO778顶点，0.05 N接触阈值 | [pp.14–15](https://arxiv.org/abs/2608.15060) |
| EgoTactile（补充） | 全手162点 | 约17 Hz；原文0–350 N | 15 Hz sample-and-hold；部分指段六传感器取均值 | [pp.4,12](https://arxiv.org/abs/2606.09243) |
| EgoPressure / PressureVision | 人裸手，Sensel平面 | 法向压力；0.5 kPa有效阈值 | 平面GT投影到RGB/手网格 | [EgoPressure pp.4–6](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_EgoPressure_A_Dataset_for_Hand_Pressure_and_Pose_Estimation_in_CVPR_2025_paper.pdf) |
| Touch and Go | 手持GelSight局部接触 | 形变图像约27 Hz | 触觉图像、接触事件，未加独立力传感器 | [pp.3–4](https://arxiv.org/abs/2211.12498) |
| REFORCE | 人五指FSR + 机器人XHand指尖 | 标定到N；ACT原始示范83 Hz | 同物理单位比较参考/反馈力 | [pp.5–7,13](https://arxiv.org/abs/2608.15560) |
| FTP-1 | Xense图像 / Contactile阵列等 | 多种不同触觉类型 | 跨传感器表征与微调 | [pp.7,20](https://arxiv.org/abs/2606.13102) |
| Facet-0 | 机器人腕部六轴F/T | 合力与力矩 | 15 Hz训练数据、高频流供控制器 | [pp.19–20](https://arxiv.org/abs/2609.01596) |

## 校准证据与尚缺的内容

EgoPHI给出了可操作的物体端标定流程：棋盘格标定鱼眼；1 g分辨率电子秤；50 g增量加载到1000 g；每级10帧平均亮度，建立N单位查找表。它针对发光亚克力与特定光路，不能原样拿来校准电阻式指套。[原文PDF p.13](https://arxiv.org/abs/2608.13014)

REFORCE明确要求人端/机器人端读数统一到N，但完整加载夹具和拟合曲线本次未找到。EgoTac的MANO映射是空间重采样，不能代替物理标定。EgoTactile的双手一致性试验是代理标签验证，不能代替传感器标定。

## 本项目指套的小试记录（建议，未实施）

1. 记录品牌、型号、感应区、左右手适配、通道编号、原始单位、原始频率和通信方式。
2. 按制造商要求完成预热/归零；保留未换算原始值、标定值和标定版本。
3. 用合适参考力源测加载/卸载、重复性、零漂和饱和；每个通道分别记录，不从其他论文借用系数。
4. 对接触和滑动分开验证。法向力变化不自动等于测得切向力或滑移。
5. 视频与触觉记录共同可识别事件，估计偏移、抖动和长录制漂移；保存采样时间与到达时间（若设备提供），记录丢包。
6. 戴与不戴指套可比较活动受限、接触外观与动作时长；裸手条件没有直接传感器力时，明确标为“无实测力”，而不是补造标签。
