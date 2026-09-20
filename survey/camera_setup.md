# Camera & Mounting Survey · 相机与安装

[主报告与16张原文截图](experimental_setup.md) · 更新：2026-09-19

下表只填核读来源明确支持的规格。RGB-D硬件、实际记录的模态、模型输入三者分开；策略运行频率、训练缩放尺寸不当作相机规格。

| 论文 / 原文位置 | 型号 | 安装与视角 | 模态 / 录制规格 | 标定 / 同步 | 待补查 |
| --- | --- | --- | --- | --- | --- |
| [TouchAnything](https://arxiv.org/abs/2605.13083) pp.3–4,15–16 | 型号未找到 | 主文头戴 + 双腕鱼眼；附录有chest/head表述 | 3路RGB，640×480，30 Hz | 多线程读最近样本，30 Hz共同保存；位姿坐标变换 | 胸/头具体批次、FOV、曝光、时差 |
| [EgoTac](https://arxiv.org/abs/2608.15060) pp.4,14–15 | ZED 2i | 头戴 | RGB-D，1080p，30 FPS；可选点云 | 时间戳重采样；SE(3)/线性插值；固定变换链 | 原始触觉Hz、曝光与残余时差 |
| [EgoTactile](https://arxiv.org/abs/2606.09243) pp.4–5,12 | DJI Action 5 Pro | 头带 / 颈戴；绿幕补光 | RGB 1280×720，30 FPS | 取最近视频和手套样本，15 Hz保存 | 安装角度/距手距离、FOV、时延抖动 |
| [EgoPressure](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_EgoPressure_A_Dataset_for_Hand_Pressure_and_Pose_Estimation_in_CVPR_2025_paper.pdf) pp.4–6 | Azure Kinect ×8 | 铝框固定7台 + 头盔1台 | RGB-D；静态RGB 2560×1440、ego 1920×1080；30 FPS | CM4/PTP+同步触发；主动IR点+PnP | 补充材料中的同步量化；不可替换成RealSense规格 |
| [PressureVision](https://arxiv.org/abs/2203.10385) pp.6–7 | OptiTrack Prime Color ×4 | 固定架上方 | RGB 1080p，FPS未找到 | 已标定与同步，实施细节待查 | 触发、时差、外参/支架尺寸 |
| [PressureVision++](https://openaccess.thecvf.com/content/WACV2024/papers/Grady_PressureVision_Estimating_Fingertip_Pressure_From_Diverse_RGB_Images_WACV_2024_paper.pdf) pp.3–4 | 多个消费级webcam，型号未找到 | 多环境，未明确头戴 | RGB 1080p30 | 压力平面单应性投影 | 相机数、几何布局、时间同步 |
| [Touch and Go](https://arxiv.org/abs/2211.12498) pp.3–4,17 | 普通webcam，型号未找到 | 另一人拍摄近似ego视角 | RGB与GelSight视频约27 Hz | 同电脑并行录制 | 两视频偏移/漂移、尺寸、FOV |
| [EgoPHI](https://arxiv.org/abs/2608.13014) pp.12–13 | Logitech C920；Sony IMX179鱼眼 | 头戴；物体下方 | 30 FPS固定曝光；78°对角 / 近180°鱼眼 | 棋盘格、ArUco；亮度—力标定 | 录制分辨率、同步实现和残差 |
| [EgoScale](https://arxiv.org/abs/2602.16710) p.4 | 对齐采集组型号未找到 | 头戴+双腕；匹配机器人视角 | RGB；大规模源视频30 FPS | 匹配内参；Vive/Manus与视频同步 | 对齐组各流频率、同步实现 |
| [REFORCE](https://arxiv.org/abs/2608.15560) pp.5–6,13 | RealSense L515 | 看腕部AprilTag；非头戴证据 | RGB-D硬件用于腕部追踪 | 人/机器人力统一N | 相机FPS、位置、外参、残余时差；83 Hz是示范数据频率 |
| [DEX-X](https://arxiv.org/abs/2609.07747) pp.6–7 | RealSense，具体型号未找到 | 固定深度相机 | 深度生成场景点云 | 点云/机器人坐标；误差影响见p.14 | 原始FPS、距离、角度；30 Hz是策略频率 |
| [EgoEngine](https://arxiv.org/abs/2606.12604) pp.14–15 | 本次PDF为Aria Gen2 | 人戴眼镜；机器人头部刚性安装 | ego图像、手/相机轨迹 | 桌面AprilTag与机器人正向运动学 | 录制规格、视角对齐容差、时差 |

## 试采时应怎样比较安装方式（建议）

先固定同一任务、物体和照明，对头戴与颈戴记录：完整手/指尖出画比例、物体遮挡、快速动作模糊、支架移动、接触区像素尺寸、穿戴时长与不适情况。保留原始尺寸、FPS和曝光记录；不要只保存256×256训练图。

使用同一型号的设备做安装对照才容易判断安装的影响。上表不同论文的相机、数据集、任务差别很大，不能把论文性能差直接归因于头戴/颈戴。
