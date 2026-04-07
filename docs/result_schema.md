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
