# FedVLR 攻防模块说明文档

## 1. 文档目的

本文档用于整理当前项目已经完成的最小攻防实验闭环，作为后续 10 天冲刺阶段的统一依据。本文档只描述当前仓库里已经落地或已经打通链路的内容，并明确区分：

- 已完成实现
- 已接通但仍为占位
- 下一阶段规划

本文档不讨论新的算法设计，也不替代论文式方法说明。

## 2. 当前项目总体架构

当前工作区由四个子项目组成：

### 2.1 `FedVLR`

- 角色：后端训练主仓
- 职责：联邦推荐训练、最小攻防实验编排、结果结构化输出
- 当前状态：已形成最小攻防实验闭环，是当前主开发对象

### 2.2 `FedVLR-API`

- 角色：只读结果接口层
- 职责：读取 `FedVLR` 输出目录中的 JSON 结果，对前端提供只读接口
- 当前状态：已支持健康检查、实验摘要列表、单个实验摘要、单个实验详细结果读取

### 2.3 `FedVLR-Frontend`

- 角色：展示与控制前端
- 职责：实验展示、控制台页面、历史实验查看
- 当前状态：
  - 历史实验列表页已接真实 API 列表
  - 历史实验摘要预览已接真实 summary
  - 历史实验 result 级详情已接真实 result
  - 其他核心页面仍以 mock / 占位数据为主

### 2.4 `research/SplitNN_FSHA-attack-Fork-`

- 角色：参考攻击仓
- 职责：提供 FSHA 相关思想参考
- 当前状态：只参考，不直接整合，不引入 TensorFlow 代码

## 3. `FedVLR` 当前真实训练主链路

当前真实训练主链路已经明确为：

`main.py -> quick_start -> FederatedTrainer 骨架 -> 具体 Trainer`

当前优先关注的 Trainer 主线为：

- `MMFedRAPTrainer`
- `FedRAPTrainer`

其中：

- `main.py` 负责解析训练入口参数并进入 `quick_start`
- `utils/quick_start.py` 负责配置加载、数据准备、模型与 Trainer 动态加载
- `utils/federated/trainer.py` 中的 `FederatedTrainer` 承担联邦训练每轮主循环
- 各具体模型文件中的 Trainer 负责客户端训练细节与聚合实现

## 4. 当前安全实验链路的工程位置

当前最小攻防实验能力，并不是硬编码在训练逻辑里，而是围绕以下位置插入：

- `FederatedTrainer._train_epoch`
- `FederatedTrainer._train_client`
- 聚合前的 `before_aggregation`
- `Trainer.fit` 每轮结束后的统一结果出口

这套设计的核心目标是：

- 默认情况下不改变训练行为
- 先打通扩展骨架和结果结构
- 为后续接入真实 attack / defense / privacy 评估留出稳定插口

## 5. 已有模块分类

当前已形成四类模块：

1. 攻击模块
2. 防御模块
3. 隐私观测 / 评估模块
4. 实验编排与结果输出模块

下面按模块逐个说明。

## 6. 已有攻击模块

### 6.1 `ClientPreferenceLeakageProbe`

- 模块类型：attack-like 只读探针
- 代码位置：`FedVLR/attacks/client_preference_leakage_probe.py`
- 插入位置：`before_aggregation`
- 是否只读：是
- 是否修改 `participant_params`：否
- 作用对象：本轮各客户端上传更新
- 当前作用机制：
  - 读取聚合前的 `participant_params`
  - 基于更新范数、非零程度、top-k 强度集中度构造启发式 leakage score
  - 输出高风险客户端、平均风险、最大风险等结构化结果
  - 可读取本轮 `malicious_clients`，形成 target-aware 风险视角
- 当前实验价值：
  - 作为第一个 FSHA-inspired 攻击样式模块
  - 不做主动攻击，风险低
  - 先验证“从可观测更新推断隐私风险”这条链路在联邦推荐主仓内可行

### 6.2 `ClientUpdateScaleAttack`

- 模块类型：轻量主动攻击模块
- 代码位置：`FedVLR/attacks/client_update_scale_attack.py`
- 插入位置：`before_aggregation`
- 是否只读：否
- 是否修改 `participant_params`：是
- 作用对象：本轮 `malicious_clients` 对应的客户端更新
- 当前作用机制：
  - 读取 `round_state["malicious_clients"]`
  - 对目标客户端更新执行统一比例缩放
  - 使用最小配置项 `attack_scale`
- 当前实验价值：
  - 作为第一个真正会作用于聚合输入的攻击模块
  - 简单、可解释、可回退
  - 可直接与裁剪型防御形成最小攻防对照

### 6.3 `SignFlipAttack`

- 模块类型：轻量主动攻击模块
- 代码位置：`FedVLR/attacks/sign_flip_attack.py`
- 插入位置：`before_aggregation`
- 是否只读：否
- 是否修改 `participant_params`：是
- 作用对象：本轮 `malicious_clients` 对应的客户端更新
- 当前作用机制：
  - 读取 `round_state["malicious_clients"]`
  - 对目标客户端更新执行符号翻转与比例缩放
  - 采用最小规则：`flipped_update = - sign_flip_scale * original_update`
  - 使用最小配置项 `sign_flip_scale`
- 当前实验价值：
  - 作为第二个更接近经典联邦更新攻击的主动攻击模块
  - 与 `ClientUpdateScaleAttack` 保持同一接入风格，便于统一实验编排
  - 能更直接体现“恶意更新方向反转”对聚合输入的影响

### 6.4 `NoOpAttack`

- 模块类型：占位攻击模块
- 代码位置：`FedVLR/attacks/noop_attack.py`
- 插入位置：可通过攻击注册表加载
- 是否只读：是
- 是否修改 `participant_params`：否
- 作用对象：无实际作用对象
- 当前作用机制：
  - 仅用于打通 `config -> registry -> runtime` 链路
- 当前实验价值：
  - 作为工程占位，不参与真实实验结论

## 7. 已有防御模块

### 7.1 `ClientUpdateAnomalyDetector`

- 模块类型：只读检测型防御模块
- 代码位置：`FedVLR/defenses/client_update_anomaly_detector.py`
- 插入位置：`before_aggregation`
- 是否只读：是
- 是否修改 `participant_params`：否
- 作用对象：本轮各客户端上传更新
- 当前作用机制：
  - 基于客户端更新范数做简单异常检测
  - 当前规则为 `update_norm > mean + k * std`
  - 输出可疑客户端列表、阈值、检测规则、客户端分数等
- 当前实验价值：
  - 作为第一个真实但只读的 defense 模块
  - 不干预聚合，便于先做观察型对照实验

### 7.2 `NormClipDefense`

- 模块类型：轻量主动防御模块
- 代码位置：`FedVLR/defenses/norm_clip_defense.py`
- 插入位置：`before_aggregation`
- 是否只读：否
- 是否修改 `participant_params`：是
- 作用对象：本轮所有客户端上传更新
- 当前作用机制：
  - 对每个客户端更新计算全局范数
  - 超过阈值时按比例整体缩放到 `defense_clip_norm` 范围内
  - 不替换聚合算法，只做聚合前预处理
- 当前实验价值：
  - 作为第一个主动防御模块
  - 能与 `ClientUpdateScaleAttack` 或 `SignFlipAttack` 形成最小攻防对照
  - 工程风险低，解释性强

### 7.3 `NoOpDefense`

- 模块类型：占位防御模块
- 代码位置：`FedVLR/defenses/noop_defense.py`
- 插入位置：可通过防御注册表加载
- 是否只读：是
- 是否修改 `participant_params`：否
- 作用对象：无实际作用对象
- 当前作用机制：
  - 仅用于打通 `config -> registry -> runtime` 链路
- 当前实验价值：
  - 作为工程占位，不参与真实实验结论

## 8. 已有隐私观测 / 评估模块

### 8.1 `ClientUpdateNormMetric`

- 模块类型：真实 privacy metric
- 代码位置：`FedVLR/privacy_eval/client_update_norm_metric.py`
- 插入位置：`before_aggregation`
- 是否只读：是
- 是否修改 `participant_params`：否
- 作用对象：本轮各客户端上传更新
- 当前作用机制：
  - 统计客户端更新范数
  - 输出平均值、最大值、最小值、客户端数等
- 当前实验价值：
  - 作为第一个真正可运行的 privacy metric
  - 不干扰训练，只做观察和统计
  - 适合作为 API / 前端展示的安全观测指标

### 8.2 `NoOpPrivacyMetric`

- 模块类型：占位隐私评估模块
- 代码位置：`FedVLR/privacy_eval/noop_metric.py`
- 插入位置：可通过隐私指标注册表加载
- 是否只读：是
- 是否修改 `participant_params`：否
- 作用对象：无实际作用对象
- 当前作用机制：
  - 仅用于打通 `config -> registry -> runtime` 链路
- 当前实验价值：
  - 作为工程占位，不参与真实实验结论

## 9. 实验编排与结果输出模块

### 9.1 `ExperimentHookManager`

- 模块类型：实验编排与运行时容器
- 代码位置：`FedVLR/utils/experiment_hooks.py`
- 插入位置：贯穿训练过程
- 是否只读：不完全只读
- 是否修改 `participant_params`：取决于挂接模块
- 作用对象：
  - 当前实验上下文
  - 本轮 `round_state`
  - 已加载 attack / defense / privacy 模块
- 当前作用机制：
  - 读取配置并加载模块
  - 维护 `round_state`
  - 统一调用 `before_aggregation` 等阶段钩子
  - 生成实验级与轮级结构化结果
- 当前实验价值：
  - 是当前最小攻防实验闭环的核心调度层
  - 保证扩展能力和默认兼容性

### 9.2 malicious client 占位配置

- 模块类型：实验目标链路
- 代码位置：`ExperimentHookManager` 内部逻辑
- 插入位置：`start_round`
- 是否只读：是
- 是否修改 `participant_params`：否
- 作用对象：本轮参与客户端集合
- 当前作用机制：
  - 支持固定 ID 或按比例从参与客户端中生成 `malicious_clients`
  - 当前只作为记录与目标选择依据
- 当前实验价值：
  - 为攻击模块提供统一目标客户端链路
  - 先把实验场景语义和结果结构稳定下来

### 9.3 结果结构与自动写盘

- 模块类型：结果输出能力
- 代码位置：
  - `FedVLR/privacy_eval/result_schema.py`
  - `FedVLR/common/trainer.py`
  - `FedVLR/utils/utils.py`
- 插入位置：训练结束后的统一出口
- 是否只读：是
- 是否修改 `participant_params`：否
- 作用对象：
  - 详细版结果
  - 摘要版结果
- 当前作用机制：
  - 生成 `experiment_result_dict`
  - 生成 `experiment_summary_dict`
  - 自动输出：
    - `*.experiment_result.json`
    - `*.experiment_summary.json`
- 当前实验价值：
  - 让后端结果能被 API 和前端稳定消费
  - 为比赛展示和答辩沉淀统一格式

## 10. 当前 API 层承担的角色

`FedVLR-API` 当前是一个只读结果包装层，不负责训练调度。

当前已提供：

- `GET /health`
- `GET /experiments/summaries`
- `GET /experiments/{experiment_key}/summary`
- `GET /experiments/{experiment_key}/result`

当前 API 的职责是：

- 扫描 `FedVLR` 输出目录
- 读取详细版 / 摘要版 JSON
- 对前端提供稳定、轻量、只读的结果访问接口

当前 API 明确不承担：

- 启动训练
- 停止训练
- 写操作
- 数据库存储
- 鉴权

## 11. 当前前端真实接通到哪一步

`FedVLR-Frontend` 当前真实接通的部分主要集中在历史实验页：

### 已接真实数据

- 历史实验列表页：
  - 优先通过 `GET /experiments/summaries` 获取真实列表
- 历史实验摘要预览：
  - 优先通过 `GET /experiments/{experiment_key}/summary` 获取真实摘要
- 历史实验 result 级详情：
  - 优先通过 `GET /experiments/{experiment_key}/result` 获取真实结果详情
- 历史实验页的来源标识、状态提示、摘要 / 结果层级切换：
  - 已做基本交互优化

### 仍以 mock / 占位为主

- 实验配置页
- 运行监控页
- 结果分析页
- 对比分析页
- 首页大部分展示内容
- 系统架构页

也就是说，当前前端已经打通“历史实验结果查看”这一条真实数据链路，但尚未完成训练控制和实时监控的真实联调。

## 12. 当前未完成 / 未扩展部分

当前还没有完成的内容主要包括：

### 后端训练侧

- 更复杂的主动攻击模块
- 更复杂的主动防御模块
- 更细粒度的 privacy 评估指标
- `MMFedRAP` 主线的标准实验验证

### API 侧

- 训练任务接口
- 实验结果筛选、分页、对比型接口
- 更丰富的错误码和状态接口

### 前端侧

- 训练控制台更多页面接真实数据
- 结果分析页与对比分析页的真实联调
- 更完整的实验详情展示与对照展示

## 13. 建议后续优先级

建议的后续优先级如下：

1. 固化标准实验矩阵，先稳定跑通 `FedRAP + KU` 的标准场景
2. 继续增强结果结构与 API 输出，让前端能稳定展示 baseline / attack / defense / 对照
3. 将结果分析页和对比分析页逐步接入真实结果
4. 在不破坏主链路的前提下，补第二批轻量 attack / defense
5. 最后再考虑把同样的实验矩阵迁移到 `MMFedRAP`

## 14. 已完成内容与下一阶段规划的边界

### 当前已完成实现

- 训练主链路与最小 hooks 骨架
- malicious client 占位配置
- `ClientUpdateNormMetric`
- `ClientUpdateAnomalyDetector`
- `ClientPreferenceLeakageProbe`
- `ClientUpdateScaleAttack`
- `SignFlipAttack`
- `NormClipDefense`
- 详细版 / 摘要版结果结构与自动写盘
- API 只读结果接口
- 前端历史实验页真实列表 / 摘要 / result 接入

### 下一阶段规划

- 更强的攻击形式
- 更强的过滤型防御
- 更丰富的 privacy 观测指标
- `MMFedRAP` 兼容实验
- 训练调度型 API
- 前端控制台其他页面真实联调

以上规划当前均未实现，不应在比赛展示时表述为“已完成功能”。
