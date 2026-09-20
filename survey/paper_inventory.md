# 论文清单与证据索引

[返回实验部署图文报告](experimental_setup.md) · [可编辑CSV](paper_inventory.csv) · [来源快照记录](source_manifest.json)

起点：《EgoTac (1).pptx》。更新：2026-09-19（Pacific）。去重后24项研究工作；23项取得全文，EgoH4RT只能核对项目页；另补充EgoTactile。OpenTouch只作为HOPE数据源追溯，不加入PPT计数。

逐项核读与本问题相关的采集、硬件、标定、同步、实验章节；不是论文方法或结果的全面复现。优先原文及官方项目页，未核实项不补造。页码均为本次PDF页码。论文可能后续改版，图号与页码以本快照为准。

| 工作 | PPT页 | 类别 | 全文页数 | 重点证据PDF页 | 报告节 | 截图 |
| --- | --- | --- | --- | --- | --- | --- |
| [TouchAnything](https://arxiv.org/abs/2605.13083) | 2 | 人手直接采集 | 24 | 3–4,15–16 | 3.1 | Figure 2 |
| [EgoPressDiff](https://arxiv.org/abs/2606.06872) | 2 | 复用EgoPressure | 5 | 3 | 3.9 | — |
| [HOPE](https://arxiv.org/abs/2608.06192) | 2 | 复用压力/接触数据 | 12 | 4–5,8 | 3.9 | — |
| [EgoTac](https://arxiv.org/abs/2608.15060) | 2 | 人手直接采集+混合数据 | 26 | 3–4,14–15 | 3.2 | Figure X1 |
| [Touch and Go](https://arxiv.org/abs/2211.12498) | 2 | 手持GelSight/两人采集 | 23 | 3–4,17 | 3.7 | Figure 6 |
| [EgoPHI](https://arxiv.org/abs/2608.13014) | 2,4 | 仿真+真实物体端验证 | 18 | 12–13 | 3.8 | Figure 6 |
| [EgoPressure](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_EgoPressure_A_Dataset_for_Hand_Pressure_and_Pose_Estimation_in_CVPR_2025_paper.pdf) | 2 | 裸手压力板/多相机 | 12 | 4–6 | 3.4 | Figures 3–4 |
| [PressureVision](https://arxiv.org/abs/2203.10385) | 2 | 裸手压力板/固定相机 | 18 | 6–7 | 3.5 | Figure 3(a–b) |
| [PressureVision++](https://openaccess.thecvf.com/content/WACV2024/papers/Grady_PressureVision_Estimating_Fingertip_Pressure_From_Diverse_RGB_Images_WACV_2024_paper.pdf) | 2 | 完整压力/弱接触标签 | 11 | 3–4 | 3.6 | Figure 3局部 |
| [EgoScale](https://arxiv.org/abs/2602.16710) | 4 | 人类/机器人对齐采集 | 22 | 3–4 | 4.1 | Figure 2(a) |
| [EgoHOI](https://arxiv.org/abs/2603.13615) | 4 | 已有数据/视频生成 | 32 | 23 | 5.4 | — |
| [DEX-X](https://arxiv.org/abs/2609.07747) | 4 | 仿真→视觉触觉真机 | 22 | 6–7,13–14 | 4.3 | Figure 3 |
| [MINT](https://arxiv.org/abs/2609.04958) | 4 | 离线姿态/相机重建 | 10 | 4,7–8 | 5.4 | — |
| [Facet-0](https://arxiv.org/abs/2609.01596) | 4 | 机器人F/T装配 | 28 | 4–5,19–20 | 4.7 | — |
| [EgoH4RT](https://egoh4rt.pages.dev/) | 4 | 项目页；论文待公开 | 网页 | 项目页 | 5.4 | — |
| [Tabero](https://arxiv.org/abs/2605.27886) | 4 | 仿真多模态生成 | 15 | 3–4 | 5.1 | Figure 2 |
| [Tactile Genesis](https://arxiv.org/abs/2606.22332) | 4 | 仿真触觉/布局消融 | 24 | 2,4–7 | 5.2 | Figure 1 |
| [FTP-1](https://arxiv.org/abs/2606.13102) | 4 | 多机器人/多传感器 | 22 | 7,20 | 4.5 | Figure 6 |
| [VT-MUSE](https://arxiv.org/abs/2608.21290) | 4 | 机器人遥操作/真机对照 | 9 | 4–5 | 4.6 | Figure 3 |
| [UMR](https://arxiv.org/abs/2609.02134) | 4 | 全身动作重定向 | 9 | 5 | 5.4 | — |
| [EgoEngine](https://arxiv.org/abs/2606.12604) | 4 | 人类视频→机器人 | 27 | 14–15,26 | 4.4 | Figure A.1 |
| [DexFlow](https://arxiv.org/abs/2505.01083) | 5 | 几何重定向/仿真 | 7 | 3–6 | 5.3 | — |
| [REFORCE](https://arxiv.org/abs/2608.15560) | 5 | 人体力示范/机器人重放 | 14 | 5–7,13 | 4.2 | Figure 3 |
| [ACE-Ego-Hand](https://arxiv.org/abs/2608.20308) | 6 | 离线视频手重建 | 25 | 5,13,20 | 5.4 | — |
| [EgoTactile](https://arxiv.org/abs/2606.09243) | 补充 | 直接/弱配对人体采集 | 26 | 4–5,12,19 | 3.3 | Figure 2(a–d) |

## 基础表示资源（PPT第3页）

- [MANO](https://mano.is.tue.mpg.de/)：手部参数化表示。
- [SMPL](https://smpl.is.tue.mpg.de/)：人体参数化表示。

二者是表示工具，不是相机、力传感器或实验采集数据。

## 本轮来源处理

主源为arXiv作者全文；EgoPressure与PressureVision++采用CVF正式论文。EgoH4RT原页面已核查，明确paper / arXiv / code coming soon，未虚构论文链接。项目页可浏览但普通下载被拒绝，不表示项目不存在。

保存16幅局部原图，紧邻中文讨论和原文链接。未将完整论文PDF或原始私人PPT放入仓库。图片保持原图含义；局部裁剪在图号中标明。全部截图版权归原作者，当前用于私有研究记录。

## 有待下一轮跟进

优先补设备具体型号、指套绝对标定、残余时差，而不是扩展论文数量。EgoPressure的同步量化可继续核对独立补充材料；HOPE应继续追OpenTouch原始采集文档。MINT单rig硬件、多个机器人系统的相机型号/安装尺寸仍未找到。EgoH4RT待全文公开后再核读。
