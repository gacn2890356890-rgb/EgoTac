# Intel RealSense · 文献依据与试采可行性

[主报告与截图](experimental_setup.md) · 更新：2026-09-19

 **结论：这批文献支持固定RealSense参与手/机器人操作采集，但不足以直接确认我们已有型号适合头戴近距离触觉采集。**  具体型号和支架还需记录；当前没有本项目实测数据。

| 文献 | 实际怎么用 | 能支持的判断 | 不能外推的判断 |
| --- | --- | --- | --- |
| [REFORCE](https://arxiv.org/abs/2608.15560)，PDF pp.5–6 | L515 RGB-D相机追踪手腕附近AprilTag；同时记录Manus和指尖FSR | RealSense可作为该系统腕轨迹采集链的一部分 | 未证明任意型号头戴可用；83 Hz不是L515 RGB-D规格 |
| [DEX-X](https://arxiv.org/abs/2609.07747)，PDF pp.6–7 | 固定RealSense深度相机给FR3+Sharpa系统提供场景点云 | 固定深度视角可配机器人指尖触觉闭环 | 型号、距离、FOV和原始帧率未找到；策略30 Hz不是相机规格 |
| [EgoTac](https://arxiv.org/abs/2608.15060)，PDF p.14 | 头戴ZED 2i RGB-D | 头戴RGB-D是存在的部署路线 | ZED的结果不能替代RealSense型号验证 |
| [EgoPressure](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_EgoPressure_A_Dataset_for_Hand_Pressure_and_Pose_Estimation_in_CVPR_2025_paper.pdf)，PDF pp.4–6 | 头盔Azure Kinect + 七静态Kinect | 多视图标定和时钟/触发值得参考 | Kinect硬件同步方式不能直接套在RealSense上 |

## 最小可行验证（建议）

先固定安装，记录设备型号、工作模式、RGB/深度分辨率和各自FPS。在实际手—物体距离处采几种表面和遮挡条件，记录手部及接触区深度空洞、稳定性、对齐误差。只有型号确认后，再按该型号官方文档确定工作距离、曝光/同步接口；本版不推荐购买新设备。

空间上先做相机内参与工作区坐标检查；时间上分别核查RGB—depth内部对齐、RealSense—指套外部对齐。相同时间戳格式并不保证采样同一时刻。若以后头戴，要重新检查视野、线缆牵扯、支架移动和快速头动影响。

 **记录表：**  型号 / 固件 / 录制模式 / 安装位置 / 到手距离 / 物体表面 / 有效深度比例 / 对齐误差 / 时间偏移及漂移 / 丢帧 / 操作者反馈。各数值待实测，不预填。
