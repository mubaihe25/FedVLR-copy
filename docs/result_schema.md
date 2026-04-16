# FedVLR 结果结构建议（面向后续 API / 前端）

当前建议把实验结果组织为一个稳定、可序列化的结构，便于：

- 训练完成后统一落盘
- 后续由 API 返回给前端
- 前端展示单次实验结果、对比结果和历史实验

## 顶层字段建议

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `experiment_id` | `string` | 实验唯一标识，建议可直接用于前端和 API 查询 |
| `model` | `string` | 当前模型名，例如 `MMFedRAP`、`FedRAP` |
| `dataset` | `string` | 当前数据集 |
| `attack_type` | `string \| null` | 当前攻击类型，未启用时可为空 |
| `defense_type` | `string \| null` | 当前防御类型，未启用时可为空 |
| `malicious_clients` | `string[]` | 被标记为恶意的客户端列表，当前由占位配置生成，不代表真实攻击已启用 |
| `round_metrics` | `list` | 每轮训练的结构化指标 |
| `attack_success_rate` | `number \| null` | 攻击成功率或攻击效果指标 |
| `privacy_risk_score` | `number \| null` | 隐私风险评分 |
| `robustness_score` | `number \| null` | 鲁棒性评分 |
| `final_eval` | `object` | 最终推荐性能与补充指标 |
| `metadata` | `object` | 额外实验配置、说明信息、扩展字段 |

当前 `metadata` 中建议同时记录：

- `enabled_attacks`
- `enabled_defenses`
- `enabled_privacy_metrics`
- `loaded_attacks`
- `loaded_defenses`
- `loaded_privacy_metrics`

## `round_metrics` 子结构建议

每轮建议至少包含以下字段：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `round_index` | `number` | 当前轮次，兼容旧字段 |
| `round_id` | `number` | 当前轮次的稳定标识，建议前后端优先使用 |
| `participant_clients` | `string[]` | 本轮实际参与的客户端列表 |
| `num_participants` | `number` | 本轮参与客户端数，适合直接用于接口和前端 |
| `avg_train_loss` | `number \| null` | 本轮平均训练损失 |
| `valid_score` | `number \| null` | 本轮验证分数，通常对应当前 `valid_metric` |
| `test_score` | `number \| null` | 本轮测试分数，建议与 `valid_score` 使用同一指标口径 |
| `hooks_enabled` | `boolean` | 本轮是否启用了实验 hooks |
| `malicious_clients` | `string[]` | 本轮恶意客户端列表，当前由 fixed ids 或 ratio 占位生成，默认可为空 |
| `participant_count` | `number` | 兼容旧字段，含义与 `num_participants` 一致 |
| `malicious_client_count` | `number` | 本轮恶意客户端数 |
| `train_loss` | `number \| null` | 兼容旧字段，含义与 `avg_train_loss` 一致 |
| `attack_success_rate` | `number \| null` | 本轮攻击效果 |
| `privacy_risk_score` | `number \| null` | 本轮隐私风险 |
| `robustness_score` | `number \| null` | 本轮鲁棒性分数 |
| `extra` | `object` | 预留扩展字段 |

当前 `round_metrics.extra` 可继续用于记录：

- 本轮已加载的 attack / defense / privacy metric 名称
- NoOp 模块返回的占位输出
- 未来真实模块的 round-level 指标输出

## `final_eval` 子结构建议

建议至少包含：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| `recall20` | `number \| null` | 推荐性能指标 |
| `ndcg20` | `number \| null` | 推荐性能指标 |
| `loss` | `number \| null` | 最终损失或聚合损失 |
| `extra` | `object` | 其他结果指标 |

## 建议字段补充说明

### 1. `experiment_id`

建议由训练入口在实验开始时生成，保持唯一且稳定，便于后续：

- 前端查询任务状态
- API 查询实验详情
- 历史实验列表关联

### 2. `attack_type` / `defense_type`

建议统一使用稳定字符串，而不是直接暴露内部类名。这样后续：

- API 更容易保持兼容
- 前端展示更稳定
- 实验对比更容易做筛选

### 3. `malicious_clients`

建议记录客户端标识列表，而不是只记录数量。这样后续：

- 可以支持更细的回放和分析
- 可以在前端展示恶意客户端分布
- 可以为防御效果评估提供依据

当前代码已支持以下占位配置项：

- `enable_malicious_clients`
- `malicious_client_mode`
- `malicious_client_ratio`
- `malicious_client_ids`
- `enabled_attacks`
- `enabled_defenses`
- `enabled_privacy_metrics`

但这些配置当前只用于生成“记录层面的恶意客户端列表”，不会改变客户端训练、聚合或参数更新行为。后续真实 attack 接入时，可直接基于这些列表选择目标客户端。

目前 attack / defense / privacy metric 也只支持 NoOp/占位模块：

- `noop`
- `noopattack`
- `noopdefense`
- `noopprivacymetric`

它们当前通过注册表和 runtime 正常加载，但不会改变训练行为。未来真实模块可以沿用同一注册、加载和结果记录链路接入。

### 4. `round_metrics`

这是未来最关键的结果区块之一。建议每轮都记录统一结构，便于：

- 训练过程可视化
- 攻击 / 防御效果曲线绘制
- API 按轮返回

### 5. `attack_success_rate` / `privacy_risk_score` / `robustness_score`

这三个字段适合作为顶层摘要指标，方便前端直接展示，也方便后续做实验横向对比。

## 当前代码中的对应草案

本轮在 `privacy_eval/result_schema.py` 中提供了一个最小 dataclass 草案：

- `RoundMetric`
- `FinalEval`
- `ExperimentResult`

它们当前不接入训练逻辑，只作为后续后端与前端协议设计的结构基础。

## 当前已接入的第一个真实 privacy metric

目前第一个真实 privacy metric 已接入：

- `client_update_norm`
- `clientupdatenormmetric`

它只做观察型统计，不参与训练或聚合修改。

当前建议在 `round_metrics.extra.privacy_metric_outputs` 中记录：

- `avg_update_norm`
- `max_update_norm`
- `min_update_norm`
- `num_clients`

同时建议在 `metadata.privacy_metric_summaries` 中记录整场实验的 summary，
例如跨轮平均值和全局最大/最小更新范数。

## 当前已接入的第一个真实 defense 检测模块

目前第一个真实但只读的 defense 模块已接入：

- `client_update_anomaly`
- `clientupdateanomalydetector`

它只在聚合前观察客户端更新，不参与客户端过滤，也不替换聚合算法。

当前建议在 `round_metrics.extra.defense_metrics` 中记录：

- `suspicious_clients`
- `suspicious_client_count`
- `anomaly_threshold`
- `detection_rule`
- `client_scores`

同时建议在 `metadata.defense_summaries` 中记录整场实验的 detector summary，
例如检测轮次数和可疑客户端计数统计。

## 当前已接入的第一个 FSHA-inspired attack-like 模块

目前第一个参考 FSHA 思想、但保持只读的 attack-like 模块已接入：

- `client_preference_leakage_probe`
- `clientpreferenceleakageprobe`

它只在聚合前观察客户端更新，不参与客户端训练修改，也不修改聚合输入。

当前建议在 `round_metrics.extra.attack_metrics` 中记录：

- `attack_family`
- `attack_category`
- `attack_strategy`
- `attack_display_category`
- `mutates_participant_params`
- `is_read_only`
- `leakage_scores`
- `high_risk_clients`
- `high_risk_client_count`
- `risk_rule`
- `num_clients`
- `avg_leakage_score`
- `max_leakage_score`
- `all_clients_count`
- `malicious_target_clients`
- `malicious_target_count`
- `malicious_target_scores`
- `malicious_target_avg_score`

同时建议在 `metadata.attack_summaries` 中记录整场实验的 probe summary，
例如高风险客户端轮次数、累计高风险计数统计，以及 malicious target
相关的观测轮次数与平均风险分数。

## 当前已接入的第一个轻量主动攻击模块

目前第一个真正会作用于 `participant_params` 的轻量主动攻击模块已接入：

- `client_update_scale`
- `clientupdatescaleattack`

它只在聚合前对 `malicious_clients` 对应的上传更新做统一比例缩放。

当前建议在 `round_metrics.extra.attack_metrics` 中记录：

- `attack_family`
- `attack_category`
- `attack_strategy`
- `attack_display_category`
- `mutates_participant_params`
- `is_read_only`
- `attacked_clients`
- `attacked_client_count`
- `attack_scale`
- `touched_update_count`
- `attacked_client_norms_before`
- `attacked_client_norms_after`

同时建议在 `metadata.attack_summaries` 中记录整场实验的缩放攻击摘要，
例如攻击轮次数、累计命中客户端数量和前后范数统计。

## 当前已接入的第一个轻量主动防御模块

目前第一个真正会作用于 `participant_params` 的轻量主动防御模块已接入：

- `norm_clip`
- `normclipdefense`

它只在聚合前对客户端上传更新做统一的全局范数裁剪，不替换聚合算法。

当前建议在 `round_metrics.extra.defense_metrics` 中记录：

- `clipped_clients`
- `clipped_client_count`
- `defense_clip_norm`
- `norms_before`
- `norms_after`

同时建议在 `metadata.defense_summaries` 中记录整场实验的裁剪防御摘要，
例如发生裁剪的轮次数、累计被裁剪客户端数和裁剪前后平均范数。

## 当前已增强的实验场景表达字段

目前 `ExperimentResult` 顶层建议稳定包含：

- `active_attacks`
- `active_defenses`
- `active_privacy_metrics`
- `experiment_mode`
- `scenario_tags`

其中：

- `active_attacks` 表示当前实验实际启用并成功加载的攻击模块名列表
- `active_defenses` 表示当前实验实际启用并成功加载的防御模块名列表
- `active_privacy_metrics` 表示当前实验实际启用并成功加载的隐私观测模块名列表
- `experiment_mode` 表示实验主场景，例如 `baseline / attack_only / defense_only / attack_and_defense / privacy_observation`
- `scenario_tags` 表示补充性的场景标签列表，适合前端直接渲染胶囊标签

当前建议在 `metadata` 中进一步记录：

- `active_attacks`
- `active_defenses`
- `active_privacy_metrics`
- `experiment_mode`
- `scenario_tags`
- `malicious_client_summary`

当前建议在 `round_metrics[*].extra` 中进一步记录：

- `active_attacks`
- `active_defenses`
- `active_privacy_metrics`
- `experiment_mode`
- `scenario_tags`
- `pipeline_info`

这些字段只增强实验结果表达能力，不改变现有训练逻辑。

## 详细版与摘要版结果结构

当前建议同时保留两类结果结构：

### 1. 详细版结果

对应当前完整的 `experiment_result_dict`，适合：

- 调试
- 研究复盘
- 查看每轮大字段明细
- 分析攻击、防御、隐私观测模块的完整输出

详细版继续保留原有字段，不删除任何已有信息。

### 2. 摘要版结果

对应新的 `experiment_summary_dict`，适合：

- API 返回轻量结果
- 前端直接展示
- 比赛答辩材料中的场景对照说明

当前摘要版建议保留：

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

其中 `round_summaries` 每轮只保留关键字段：

- `round_id`
- `num_participants`
- `avg_train_loss`
- `valid_score`
- `test_score`
- `malicious_client_count`
- `attacked_client_count`
- `clipped_client_count`
- `pipeline_info`

当前刻意不放入摘要版的大字段包括：

- `participant_clients` 全量列表
- `leakage_scores` 全量字典
- `malicious_target_scores` 全量字典
- `norms_before / norms_after` 全量字典
- `client_scores` 全量字典

这些字段仍然保留在详细版，用于调试和复盘。
## 自动写盘约定

当前训练结束后会自动把两类结果写入现有 results 目录：

- 详细版：`*.experiment_result.json`
- 摘要版：`*.experiment_summary.json`

输出目录沿用现有结果路径体系：

- `outputs/results/{model}/{dataset}/{type}/`

文件名默认复用当前 `result_file_name` 的主名，再追加后缀。例如：

- `[FedRAP]-[KU]-[Contrast.attack_only].experiment_result.json`
- `[FedRAP]-[KU]-[Contrast.attack_only].experiment_summary.json`

其中：

- 详细版适合调试、研究和复盘，保留完整字段
- 摘要版适合 API、前端和答辩展示，只保留轻量关键字段

## 统一非定向投毒结果字段

新增 `poisoning_attack` 后，详细版结果会在 `round_metrics[*].extra.attack_metrics.poisoning_attack` 中记录统一投毒语义。该模块是非定向投毒入口，不包含 target item、backdoor 或定向曝光目标。

轮级字段建议：

- `attack_family`: 固定为 `poisoning`
- `attack_category`: 固定为 `poisoning`
- `attack_strategy`: 固定为 `unified_nondirected_poisoning`
- `poisoning_mode`: 固定为 `nondirected`
- `poisoning_mix_rule`: 当前默认 `round_robin`
- `poisoning_enabled_substrategies`: 本轮启用的投毒子策略列表
- `poisoned_clients`: 本轮被投毒处理的恶意客户端
- `poisoned_client_count`: 本轮投毒客户端数
- `strategy_client_counts`: 各子策略分到的客户端数量
- `strategy_attacked_clients`: 各子策略实际处理的客户端列表
- `strategy_metrics`: 各旧攻击模块返回的子策略指标
- `avg_poisoned_norm_before`: 投毒前恶意更新平均范数
- `avg_poisoned_norm_after`: 投毒后恶意更新平均范数

实验级字段建议写入 `experiment_result.metadata.attack_summaries.poisoning_attack`：

- `num_rounds`
- `rounds_with_attacks`
- `total_poisoned_clients`
- `max_poisoned_client_count`
- `strategy_client_counts_total`
- `strategy_round_coverage`
- `poisoning_mode`
- `poisoning_mix_rule`

旧的 `client_update_scale`、`sign_flip`、`model_replacement` 结果字段继续保留，用于消融实验和历史结果兼容；推荐展示口径优先使用 `poisoning_attack` 的统一字段。
