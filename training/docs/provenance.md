# 源码来源与迁移范围

本训练目录整理自Ego2WristTouch本地项目的v15训练链路。只提交项目自有训练/评价实现及说明，不提交数据、checkpoint、第三方模型仓库或下载凭据。

| 当前文件 | 原项目模块 | 变化 |
|---|---|---|
| model.py | joint_tactile_head_v13.py | 保留完整模型及state_dict键名 |
| contact_metrics.py | contact_label_metrics.py | 保留原接触/事件定义，无正例的幅值指标返回null |
| losses.py | expanded_v15_common.py | 保留v15损失权重，device从预测读取，补空mask处理 |
| inference.py / metrics.py | expanded_v15_common.py | MPS/CUDA可配置，保留窗口融合和任务宏平均 |
| train.py | train_expanded_v15.py | 外部manifest与输出路径、参数配置、显式warm start、optimizer/RNG恢复、有限预算 |

迁移版不再硬编码414/44/10任务，也不依赖某个用户的绝对路径。实际数据规模在每次run.json记录，训练不会因文档中的历史规模而伪造覆盖数。

历史v15从各条件/seed上一轮10000步checkpoint继续。这里默认随机初始化，只有显式指定外置初始化权重才是warm start。新跑的预算、初始化和数据都需要记录，不能默认宣称复现历史结果。

代码默认选择val等任务连续响应MAE最小的checkpoint，AP打破平局，概率阈值在val选择。最终测试不重新搜索阈值。

当前模型时间编码是窗口内双向上下文。这对离线同帧标签估计成立，不能将其解释为在线因果未来预测。下一轮方案见next_experiments.md。

第三方VGGT-Omega、EgoFound3R和WristWorld只作为既有外部特征/几何链路的来源描述，其代码许可和模型权重授权应依各自上游执行。本代码包没有为第三方权重授予许可。
