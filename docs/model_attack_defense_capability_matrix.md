# 模型 / 攻防能力矩阵

## 1. 文档目的

本文档用于整理当前 `FedVLR` 子项目中已经实现、已经验证和暂不建议开放的模型与攻防能力关系。它面向三类后续工作：

- 前端实验配置页：明确哪些模型、攻击、防御、观测模块可以作为可选项展示。
- API / launcher：为统一实验启动入口提供可读依据。
- 答辩与团队协作：说明当前能力边界，避免把“理论可配”误认为“已系统验证”。

本轮只整理能力矩阵与配置约束，不新增 attack / defense / metric，不改变训练算法、loss、聚合逻辑或模型结构。

## 2. 模型清单与兼容状态

| 模型 | 分类 | 推荐数据集 | 强攻防兼容验证 | 当前说明 |
| --- | --- | --- | --- | --- |
| `FedAvg` | 单模态联邦推荐 | `KU` | 已验证 | 通过 `baseline / model_replacement / model_replacement + trimmed_mean` 兼容验证。 |
| `FedNCF` | 单模态联邦推荐 | `KU` | 已验证 | 通过三组强攻防兼容验证；默认列表型超参建议在脚本中显式标量化。 |
| `FedRAP` | 单模态联邦推荐 | `KU` | 已验证 / 推荐展示 | 已完成标准矩阵与强攻防展示主线，是当前最稳定展示基线之一。 |
| `FedVBPR` | 视觉特征联邦推荐 | `KU` | 已验证 | 依赖 KU 多模态特征文件，已通过三组强攻防兼容验证。 |
| `PFedRec` | 单模态个性化联邦推荐 | `KU` | 已验证 | 工程类名为 `PFedRec`，已通过三组强攻防兼容验证。 |
| `MMFedAvg` | 多模态联邦推荐 | `KU` | 已验证 | 已通过三组强攻防兼容验证。 |
| `MMFedNCF` | 多模态联邦推荐 | `KU` | 已验证 | 已通过三组强攻防兼容验证。 |
| `MMFedRAP` | 多模态联邦推荐 | `KU` | 已验证 / 推荐展示 | 已完成兼容验证与正式强攻防展示实验，是当前多模态展示主线。 |
| `MMPFedRec` | 多模态个性化联邦推荐 | `KU` | 已验证 | 已通过三组强攻防兼容验证。 |
| `MMGCN` | 图结构多模态推荐 | `KU` | 未通过 | 当前环境缺少 `torch_geometric`，且需进一步确认是否存在可用的 `MMGCNTrainer` 与联邦 hook 链路。 |

## 3. 攻击模块能力表

| 模块名 | 类型 | 关键配置项 | 默认值 | 是否修改 `participant_params` | 当前验证情况 |
| --- | --- | --- | --- | --- | --- |
| `client_update_scale` | 轻量主动攻击 | `attack_scale` | `2.0` | 是 | 已在标准矩阵中验证，可与 `norm_clip` 组成基础攻防对照。 |
| `sign_flip` | 经典联邦更新攻击变体 | `sign_flip_scale` | `1.0` | 是 | 已在标准矩阵和 showcase_v1 中验证，可与 `update_filter` 或 `norm_clip` 做对照。 |
| `model_replacement` | 更强主动攻击，minimal replacement-like | `replacement_scale`, `replacement_rule` | `5.0`, `aligned_mean` | 是 | 已在 FedRAP / MMFedRAP 正式强攻防主线和 9 个模型兼容验证中使用。 |
| `client_preference_leakage_probe` | FSHA-inspired 只读隐私泄露探针 | `attack_probe_topk_ratio`, `attack_probe_std_factor` | `0.1`, `1.5` | 否 | 已实现并接入 hook，适合作为隐私风险观测，不等同于主动攻击。 |

## 4. 防御模块能力表

| 模块名 | 类型 | 关键配置项 | 默认值 | 是否修改 `participant_params` | 当前验证情况 |
| --- | --- | --- | --- | --- | --- |
| `client_update_anomaly` | 只读异常检测 | `defense_anomaly_std_factor` | `2.0` | 否 | 已实现，适合作为检测指标或辅助解释字段。 |
| `norm_clip` | 聚合前范数裁剪 | `defense_clip_norm` | `5.0` | 是 | 已在基础攻防矩阵中验证；展示版中建议结合实际范数调阈值。 |
| `update_filter` | 聚合前可疑更新过滤 | `filter_rule`, `filter_std_factor`, `max_filtered_ratio` | `update_norm > mean + filter_std_factor * std`, `2.0`, `0.5` | 是 | 已实现并在标准矩阵中验证，带有避免过滤全部客户端的保护。 |
| `trimmed_mean` | minimal trimmed-mean-like 鲁棒聚合防御 | `trim_ratio`, `min_clients_for_trim`, `trim_rule` | `0.2`, `5`, `coordinate_trimmed_mean` | 是 | 已在 FedRAP / MMFedRAP 正式强攻防主线和 9 个模型兼容验证中使用。 |

## 5. 隐私观测 / 评估模块能力表

| 模块名 | 类型 | 关键配置项 | 默认值 | 当前验证情况 |
| --- | --- | --- | --- | --- |
| `client_update_norm` | 只读客户端更新范数统计 | 无必需配置 | 无 | 已实现，可输出每轮客户端更新范数统计。 |

## 6. 已验证组合

| 组合 | 场景 | 已验证模型 | 说明 |
| --- | --- | --- | --- |
| 无 attack / 无 defense | `baseline` | 9 个模型 | 基线场景，已在批量兼容验证中覆盖除 `MMGCN` 外的 9 个模型。 |
| `model_replacement` | `attack_only` | 9 个模型 | 当前最稳定的强攻击主线，已覆盖单模态与多模态代表模型。 |
| `model_replacement + trimmed_mean` | `attack_and_defense` | 9 个模型 | 当前推荐的强攻防主展示组合，已覆盖 FedRAP、MMFedRAP 以及模型族兼容验证。 |
| `sign_flip` | `attack_only` | FedRAP | 已在 showcase_v1 / standard matrix 中验证，适合作为经典更新攻击示例。 |
| `client_update_scale + norm_clip` | `attack_and_defense` | FedRAP | 已在 standard matrix 与 showcase_v1 中验证，适合作为基础攻防闭环示例。 |
| `sign_flip + update_filter` | `attack_and_defense` | FedRAP | 已在 standard matrix 中验证，适合作为过滤型防御示例。 |
| `client_update_scale + norm_clip + client_update_norm` | `attack_and_defense + privacy_observation` | FedRAP | 文档矩阵中建议使用；如需正式展示，建议再跑一次固定结果。 |

## 7. 推荐主展示组合

当前最建议对外展示的两条主线是：

- FedRAP + KU：`baseline -> model_replacement -> model_replacement + trimmed_mean`
- MMFedRAP + KU：`baseline -> model_replacement -> model_replacement + trimmed_mean`

原因：

- `model_replacement` 比简单 scale / sign flip 更强，能体现更高层级的主动攻击。
- `trimmed_mean` 更接近鲁棒聚合防御，不只是范数裁剪或单纯过滤。
- 两条主线均已完成正式版结果输出，且可被后续 API / 前端直接消费。

## 8. 多模块组合规则

当前 hook 执行顺序为：

1. `attack chain`：按 `enabled_attacks` 列表顺序执行。
2. `defense chain`：按 `enabled_defenses` 列表顺序执行。
3. `metrics / observation`：收集 `enabled_privacy_metrics` 结果并写入 `round_metrics.extra`。

第一版建议限制：

- `enabled_attacks` 最多 2 个。
- `enabled_defenses` 最多 2 个。
- `enabled_privacy_metrics` 最多 3 个。
- 允许单模块或一攻一防组合优先开放。
- 多攻击叠加、多防御叠加虽然工程上可表达，但大多数尚未系统验证，建议在 UI 中标记为“实验性 / 未验证”。

## 9. 当前建议开放与限制

建议开放：

- `baseline`
- `model_replacement`
- `model_replacement + trimmed_mean`
- `sign_flip`
- `client_update_scale + norm_clip`
- `client_update_norm`
- `client_preference_leakage_probe` 作为只读观测项

建议标记为未验证或谨慎开放：

- 两个主动攻击同时启用，例如 `model_replacement + sign_flip`
- 两个主动防御同时启用，例如 `norm_clip + trimmed_mean`
- `update_filter + trimmed_mean` 这类双鲁棒处理组合
- `MMGCN` 相关强攻防实验，需先解决 `torch_geometric` 与 trainer 链路问题

## 10. 与后续前端配置页的关系

后续前端实验配置页可以基于机器可读能力矩阵提供：

- 模型下拉选项与兼容状态提示。
- 攻击 / 防御 / 观测模块多选。
- 已验证组合的推荐标签。
- 未验证组合的风险提示。
- 统一配置 schema 中的 `malicious_client_config / training_params / attack_params / defense_params / privacy_params` 分组表单。

当前本文件只提供说明依据；真正可供程序读取的结构见 `configs/model_attack_defense_capabilities.json` 与 `configs/experiment_config_schema.json`。
