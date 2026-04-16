# FedVLR 后续集成规划（仅骨架阶段）

## 为什么当前只做骨架

当前阶段的目标不是立即实现完整 attack / defense / privacy_eval，而是先在 `FedVLR` 内部建立一套清晰、稳定、可扩展的工程边界。这样做有几个原因：

1. 当前真实训练主链路已经明确，核心运行路径是 `main.py -> quick_start -> FederatedTrainer -> 具体 Trainer`。在没有完全确认每一种攻击和防御方案之前，先补齐独立目录和统一接口，更有利于后续平滑接入。
2. 未来结果需要通过 API 提供给前端，因此新增输出结构必须稳定、可序列化、字段清晰。先做结果结构草案，可以避免后面边开发边改协议。
3. 当前训练逻辑已经可运行，本阶段不应直接改动 `main.py`、`quick_start.py`、`FederatedTrainer` 和 `models/*.py`，否则容易把“功能探索”和“工程铺底”混在一起。

## 为什么不直接迁移 TensorFlow attack 参考仓

`research/SplitNN_FSHA-attack-Fork-` 当前只是参考仓，不适合直接迁移到 `FedVLR` 中，原因包括：

1. 该参考仓以 TensorFlow 2 为主，而 `FedVLR` 当前训练主仓是 PyTorch 体系，训练范式、模型组织方式、张量对象和调试方式都不一致。
2. 参考仓面向的是分裂学习攻击，不等价于当前 `FedVLR` 的联邦推荐训练骨架。即使攻击思想可参考，实现位置和数据结构也不能直接照搬。
3. 当前更需要的是“插桩接口”和“结构化结果”，而不是直接复制一套外部实现。先把 FedVLR 的扩展骨架搭好，后续才能有序地把参考思想转换成适合本仓库的 PyTorch 实现。

## 为什么优先以 MMFedRAP / FedRAP 为落点

当前仓库里没有 `FedVLRTrainer`，所以后续联邦安全扩展最适合优先落在以下两条主线：

1. `MMFedRAPTrainer`
   - 更接近当前项目“多模态联邦推荐安全实验平台”的语义。
   - 同时覆盖客户端本地训练、融合层更新、共享表示聚合等关键位置。
   - 后续更适合接入模态攻击、融合层防御、聚合阶段鲁棒处理。

2. `FedRAPTrainer`
   - 结构更简洁，便于先验证攻击、防御和隐私评估的工程插入链路。
   - 如果需要先做轻量的联邦推荐安全基线，再过渡到多模态版本，FedRAP 是更稳妥的中间落点。

## 未来准备插入 attack / defense / privacy_eval 的训练位置

### 1. `_train_epoch`

位置：`utils/federated/trainer.py`

适合插入：

- round 开始前的实验状态记录
- 恶意客户端选择
- 客户端上传前后的全局调度逻辑
- round 级攻击 / 防御 / 隐私指标汇总

### 2. `_train_client`

位置：`utils/federated/trainer.py`

适合插入：

- 客户端本地训练阶段的标签翻转、模型投毒等攻击
- 本地梯度裁剪、本地差分隐私噪声等处理
- 客户端训练后输出更新的额外统计

### 3. `_aggregate_params`

位置：各具体 Trainer，例如 `models/fedrap.py`、`models/mmfedrap.py`

适合插入：

- 聚合前异常更新过滤
- 鲁棒聚合
- 安全聚合包装
- 聚合后防御恢复效果记录

### 4. `fit` 每轮结束后的出口

位置：`common/trainer.py`

适合插入：

- 每轮评估后的统一结果记录
- round-level 的 attack / defense / privacy_eval 汇总
- 形成后续 API / 前端可消费的结构化结果

## 当前阶段边界

本轮只补工程骨架，不接训练逻辑，不修改现有训练主链路。后续真正实现 attack / defense / privacy_eval 时，应优先围绕现有联邦训练骨架做增量接入，而不是重写训练流程。

## 本轮新增的最小非侵入式插桩准备

当前新增了一层极简运行时容器：`utils/experiment_hooks.py` 中的 `ExperimentHookManager`。

它当前只承担以下职责：

1. 维护统一的实验上下文，例如：
   - `enabled`
   - `attacks`
   - `defenses`
   - `privacy_metrics`
   - `round_states`
2. 为未来 attack / defense / privacy_eval 预留统一调用入口。
3. 以 `privacy_eval/result_schema.py` 中的结果结构为基础，收集最小的 round 级占位结果。

### 默认开关与行为

- `enable_experiment_hooks`：默认 `False`
- `collect_round_metrics`：默认 `True`
- `enable_malicious_clients`：默认 `False`
- `malicious_client_mode`：默认 `none`
- `malicious_client_ratio`：默认 `0.0`
- `malicious_client_ids`：默认空列表
- `enabled_attacks`：默认空列表
- `enabled_defenses`：默认空列表
- `enabled_privacy_metrics`：默认空列表

默认情况下：

- 不执行任何真实攻击逻辑
- 不执行任何真实防御逻辑
- 不执行任何真实隐私评估逻辑
- 不修改现有 loss
- 不修改现有聚合算法
- 不修改现有模型参数更新行为

当前只会在内存中记录最基础的 round 信息，供未来 API / 前端接入时复用。
当前的 `malicious_clients` 也只是占位配置与占位记录，不会改变任何客户端的训练行为。
当前 attack / defense / privacy metric 也只支持 NoOp/占位模块，通过注册表和 runtime 加载链路接通，但不会改变训练行为。

### 本轮已预留的调用点

#### 1. `FederatedTrainer._train_epoch`

已接入：

- round 开始时初始化 round state
- round 开始时按配置生成本轮 `malicious_clients` 占位列表
- 本地训练完成后预留 `after_local_train`
- 聚合前预留 `before_aggregation`
- round 结束时记录最基础的 round loss 和 participant 数量

#### 2. `FederatedTrainer._train_client`

已接入：

- 记录单个客户端本地训练的 loss 轨迹占位信息

#### 3. `Trainer.fit` 每轮结束出口

已接入：

- 每轮训练结束后的 `valid_result / test_result / stop_flag` 占位记录
- 训练结束后生成与 `ExperimentResult` 兼容的顶层结果结构

### 未来如何接 attack / defense / privacy_eval

后续如果要真正实现扩展，可沿用当前 manager：

- attack：
  在 `after_local_train` 或 `before_aggregation` 中接入客户端更新篡改、恶意客户端策略等逻辑。当前生成的 `malicious_clients` 列表可直接作为攻击目标集合。
- defense：
  在 `before_aggregation` 或具体 Trainer 的 `_aggregate_params` 周围接入异常检测、鲁棒聚合、安全聚合等逻辑。
- privacy_eval：
  在 round 结束和 `fit` 末尾汇总输出隐私风险评分、隐私预算和 round-level 指标。

也就是说，本轮不是在训练链路里硬编码安全逻辑，而是先把“可插入的位置”和“可输出的结构”准备好，并确保默认情况下训练行为不变。

## 本轮新增的最小注册与加载链路

当前新增了三类配置项：

- `enabled_attacks`
- `enabled_defenses`
- `enabled_privacy_metrics`

它们都支持以名称列表的方式写入 `config_dict / Config`。如果为空，则默认不启用任何模块。

当前三类模块都支持极简 NoOp 示例：

- `NoOpAttack`
- `NoOpDefense`
- `NoOpPrivacyMetric`

这些模块通过各自的注册表注册，并在 `ExperimentHookManager` 初始化时按名称实例化。即使显式加载了 NoOp 模块，也只会做占位记录，不会改动训练、loss、聚合或参数更新。

## 第一个真实 privacy metric：ClientUpdateNormMetric

当前新增了第一个可运行的 privacy metric：`ClientUpdateNormMetric`。

它的职责非常克制：

- 只在联邦训练的聚合前阶段观察 `participant_params`
- 只统计本轮参与客户端上传更新的范数分布
- 只输出结构化统计结果，不修改任何训练数据或参数

当前主要输出：

- `avg_update_norm`
- `max_update_norm`
- `min_update_norm`
- `num_clients`

该 metric 通过现有 `enabled_privacy_metrics` 配置启用，并沿用统一的
`registry -> ExperimentHookManager -> round_metrics.extra` 链路接入。

默认情况下它不会启用，因此训练行为保持不变。即使启用后，它也只是做
观察与统计，不会修改 loss、聚合算法或参数更新。

## 第一个真实但只读的 defense 模块：ClientUpdateAnomalyDetector

当前新增了第一个可运行的 defense 检测模块：`ClientUpdateAnomalyDetector`。

它的职责同样保持克制：

- 只在联邦训练的聚合前阶段观察 `participant_params`
- 只基于客户端更新范数做简单异常检测
- 只记录 `suspicious_clients` 等结构化结果，不执行拦截或过滤

当前采用的检测规则是一个可解释的占位规则：

- `update_norm > mean + std_factor * std`

当前主要输出：

- `suspicious_clients`
- `suspicious_client_count`
- `anomaly_threshold`
- `detection_rule`
- `client_scores`

该 detector 通过 `enabled_defenses` 配置启用，并沿用统一的
`registry -> ExperimentHookManager -> round_metrics.extra` 链路接入。

默认情况下它不会启用，因此训练行为保持不变。即使启用后，它也只做
检测和记录，不会修改聚合输入、loss 或参数更新。

## 第一个 FSHA-inspired attack-like 模块：ClientPreferenceLeakageProbe

当前新增了第一个参考 FSHA 思想的 attack-like 模块：
`ClientPreferenceLeakageProbe`。

它的定位不是主动攻击器，而是一个只读的隐私泄露风险探针：

- 只在联邦训练的聚合前阶段观察 `participant_params`
- 不训练额外网络，不做重建器
- 只基于客户端更新的可观测统计量估计潜在偏好泄露风险

它参考的是 FSHA 一类方法的核心思想：

- 在 split learning 中，服务器可观测到的中间量可能泄露客户端隐私
- 在当前联邦推荐场景中，我们先不做真实重建，而是从客户端更新出发，
  做一个简化版的偏好泄露风险探针

当前采用的启发式风险分数结合了：

- 更新范数
- 非零程度
- top-k 更新强度集中度

当前主要输出：

- `leakage_scores`
- `high_risk_clients`
- `high_risk_client_count`
- `risk_rule`
- `num_clients`
- `avg_leakage_score`
- `max_leakage_score`

当前它还会与 `malicious_clients` 占位配置联动，补充记录：

- `all_clients_count`
- `malicious_target_clients`
- `malicious_target_count`
- `malicious_target_scores`
- `malicious_target_avg_score`

该 probe 通过 `enabled_attacks` 配置启用，并沿用统一的
`registry -> ExperimentHookManager -> round_metrics.extra` 链路接入。

默认情况下它不会启用，因此训练行为保持不变。即使启用后，它也只做
观察和记录，不会修改客户端训练、聚合输入、loss 或参数更新。

## 第一个轻量主动攻击模块：ClientUpdateScaleAttack

当前新增了第一个真正会作用于 `participant_params` 的轻量主动攻击模块：
`ClientUpdateScaleAttack`。

它的设计保持极简：

- 只在联邦训练的聚合前阶段运行
- 只对本轮 `malicious_clients` 对应的上传更新生效
- 只做统一比例缩放：`scaled_update = attack_scale * original_update`

当前支持的最小配置项：

- `attack_scale`

当前主要记录：

- `attacked_clients`
- `attacked_client_count`
- `attack_scale`
- `touched_update_count`
- `attacked_client_norms_before`
- `attacked_client_norms_after`

该模块通过 `enabled_attacks` 配置启用，并依赖现有 `malicious_clients`
目标链路选择作用对象。

默认情况下它不会启用，因此训练行为保持不变。即使启用后，它也只改变
目标客户端的上传更新，不修改客户端训练算法、loss、聚合算法或模型结构。

## 第一个轻量主动防御模块：NormClipDefense

当前新增了第一个真正会作用于 `participant_params` 的轻量主动防御模块：
`NormClipDefense`。

它的设计同样保持极简：

- 只在联邦训练的聚合前阶段运行
- 对所有参与客户端上传更新执行统一的全局范数裁剪
- 如果某个客户端更新范数超过阈值，则整体缩放到阈值范围内

当前支持的最小配置项：

- `defense_clip_norm`

当前主要记录：

- `clipped_clients`
- `clipped_client_count`
- `defense_clip_norm`
- `norms_before`
- `norms_after`

该模块通过 `enabled_defenses` 配置启用，并沿用现有
`registry -> ExperimentHookManager -> round_metrics.extra` 链路接入。

默认情况下它不会启用，因此训练行为保持不变。即使启用后，它也只是做
聚合前预处理，不替换聚合算法，不修改客户端训练、loss 或模型结构。

## 最小攻防实验编排与场景表达

当前系统已经支持把已启用的 attack / defense / privacy metric 组织成更清晰
的实验方案表达，用于后续 API、前端和比赛展示。

当前新增的实验级字段包括：

- `active_attacks`
- `active_defenses`
- `active_privacy_metrics`
- `experiment_mode`
- `scenario_tags`
- `malicious_client_summary`

其中 `experiment_mode` 用于给出当前实验的主场景标签，例如：

- `baseline`
- `attack_only`
- `defense_only`
- `attack_and_defense`
- `privacy_observation`

`scenario_tags` 则用于补充场景说明，例如在开启 privacy metric 时追加
`privacy_observation`，在开启 malicious client 机制时追加
`malicious_clients_configured`。

在 round 级结果中，当前也会通过 `round_metrics[*].extra` 稳定记录：

- 本轮启用的 attack / defense / privacy metric 名称
- 本轮 `experiment_mode`
- 本轮 `scenario_tags`
- 本轮 `malicious_clients`

这些增强只作用于结果表达层，不改变训练算法、loss、聚合或模型结构。

## 详细版 + 摘要版双结果输出

当前系统已经同时支持两类实验结果视图：

- 详细版结果：保留完整的 `experiment_result_dict`
- 摘要版结果：提供轻量的 `experiment_summary_dict`

两者职责区分如下：

- 详细版用于调试、研究复盘、查看每轮明细和大字段字典
- 摘要版用于未来 API、前端和比赛答辩展示

当前摘要版重点保留：

- `experiment_id`
- `model`
- `dataset`
- `experiment_mode`
- `scenario_tags`
- `active_attacks`
- `active_defenses`
- `active_privacy_metrics`
- `malicious_client_summary`
- `final_eval`
- `round_summaries`

其中 `round_summaries` 只保留每轮关键字段，例如：

- `round_id`
- `num_participants`
- `avg_train_loss`
- `valid_score`
- `test_score`
- `malicious_client_count`
- `attacked_client_count`
- `clipped_client_count`
- `pipeline_info`

像 `participant_clients` 全量列表、`leakage_scores` 全量字典、`norms_before`
和 `norms_after` 全量字典等大字段，仍然只保留在详细版中。
## 结果自动写盘

当前在训练结束后的统一结果出口中，已经同时支持两类 JSON 自动写盘：

- 详细版：`trainer.experiment_result_dict`
- 摘要版：`trainer.experiment_summary_dict`

写盘位置沿用现有结果目录：

- `outputs/results/{model}/{dataset}/{type}/`

命名方式复用当前实验的 `result_file_name` 主名，再追加：

- `*.experiment_result.json`
- `*.experiment_summary.json`

`result_file_name` 的主名现在包含每次运行唯一的 `output_run_id`，例如：

- `[FedRAP]-[KU]-[Contrast.attack_only]-[20260416_201933_123456].csv`
- `[FedRAP]-[KU]-[Contrast.attack_only]-[20260416_201933_123456].experiment_result.json`
- `[FedRAP]-[KU]-[Contrast.attack_only]-[20260416_201933_123456].experiment_summary.json`

这样同一配置重复运行时会生成新的 CSV、详细版 JSON 和摘要版 JSON，不再覆盖旧实验文件；API 仍然按既有后缀扫描结果目录。

这样可以保持：

- 详细版继续服务调试、研究和复盘
- 摘要版直接服务后续 API、前端和比赛展示

这一步只增强结果导出链路，不修改训练算法、loss、聚合或模型结构。
