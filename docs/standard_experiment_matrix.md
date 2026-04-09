# FedVLR 标准实验矩阵

## 1. 文档目的

本文档用于固定当前阶段建议采用的最小标准实验矩阵，作为：

- 后端跑实验的统一模板
- API / 前端字段对齐依据
- 比赛展示与答辩时的标准对照口径

当前矩阵优先围绕：

- 模型：`FedRAP`
- 数据集：`KU`

后续可在验证稳定后扩展到 `MMFedRAP`。

## 2. 当前建议固定的标准实验场景

当前建议至少固定以下 5 组实验：

1. `baseline`
2. `attack_only`
3. `defense_only`
4. `attack_and_defense`
5. `attack_and_defense_with_privacy_observation`

这些实验足以支撑当前最小攻防闭环的展示、比对和答辩说明。

## 3. 通用实验建议

### 通用模型与数据集

- 建议模型：`FedRAP`
- 建议数据集：`KU`

### 通用基础配置建议

建议在最小实验里统一以下基础设置：

- `use_gpu: false` 或按本地环境稳定选择
- `seed: 42`
- `hyper_parameters: []`
- `alpha: 1e-1`
- `beta: 1e-1`
- `epochs: 3`
- `local_epochs: 1`
- `clients_sample_ratio: 1.0`
- `eval_step: 1`
- `collect_round_metrics: true`

### 通用输出关注点

建议所有标准实验至少保留并检查：

- `experiment_mode`
- `scenario_tags`
- `active_attacks`
- `active_defenses`
- `active_privacy_metrics`
- `malicious_client_summary`
- `final_eval`
- `round_summaries`

## 4. 标准实验一：baseline

### 目标

建立没有攻击、没有主动防御、没有隐私观测干预的基线结果，作为所有后续实验的参考起点。

### 建议配置

- 建议模型：`FedRAP`
- 建议数据集：`KU`
- `enabled_attacks: []`
- `enabled_defenses: []`
- `enabled_privacy_metrics: []`
- `enable_malicious_clients: false`

### 关键配置项

```yaml
type: Contrast
comment: baseline
enabled_attacks: []
enabled_defenses: []
enabled_privacy_metrics: []
enable_malicious_clients: false
```

### 预期输出字段

- `experiment_mode: baseline`
- `scenario_tags: ["baseline"]`
- `active_attacks: []`
- `active_defenses: []`
- `active_privacy_metrics: []`
- `malicious_client_summary.enabled: false`
- `final_eval.recall20`
- `final_eval.ndcg20`
- `round_summaries[*].avg_train_loss`

### 适合展示的重点

- 基线推荐性能
- 每轮收敛趋势
- 后续所有实验的参照组

## 5. 标准实验二：attack_only

### 目标

展示在存在恶意客户端时，轻量主动攻击模块如何影响实验场景与输出结果。

### 建议配置

- 建议模型：`FedRAP`
- 建议数据集：`KU`
- `enabled_attacks: ["client_update_scale"]`
- `enabled_defenses: []`
- `enabled_privacy_metrics: []`
- `enable_malicious_clients: true`
- `malicious_client_mode: "ratio"`
- `malicious_client_ratio: 0.2`
- `attack_scale: 3.0`

当前阶段推荐把 `client_update_scale` 作为主展示攻击模块；如果需要展示第二种更接近经典联邦更新攻击的变体，也可以替换为：

- `enabled_attacks: ["sign_flip"]`
- `sign_flip_scale: 3.0`

### 关键配置项

```yaml
type: Contrast
comment: attack_only
enabled_attacks:
  - client_update_scale
enabled_defenses: []
enabled_privacy_metrics: []
enable_malicious_clients: true
malicious_client_mode: ratio
malicious_client_ratio: 0.2
attack_scale: 3.0
```

### 预期输出字段

- `experiment_mode: attack_only`
- `scenario_tags` 至少包含：
  - `attack_only`
  - `malicious_clients_configured`
- `active_attacks: ["client_update_scale"]`
- `malicious_client_summary`
- `round_summaries[*].malicious_client_count`
- 详细版中的：
  - `round_metrics[*].extra.attack_metrics.client_update_scale.attacked_client_count`
  - 或 `round_metrics[*].extra.attack_metrics.sign_flip.attacked_client_count`

### 适合展示的重点

- 攻击场景标签是否清晰
- 恶意客户端链路是否生效
- 攻击前后性能下降趋势
- 可作为 `client_update_scale` 与 `sign_flip` 的轻量攻击对照入口

## 6. 标准实验三：defense_only

### 目标

展示防御模块单独启用时，对上传更新的预处理和结果结构化输出能力。

### 建议配置

- 建议模型：`FedRAP`
- 建议数据集：`KU`
- `enabled_attacks: []`
- `enabled_defenses: ["norm_clip"]`
- `enabled_privacy_metrics: []`
- `enable_malicious_clients: false`
- `defense_clip_norm: 5.0`

如果需要展示第二种更像“主动过滤”的防御变体，也可以替换为：

- `enabled_defenses: ["update_filter"]`
- `filter_std_factor: 2.0`
- `max_filtered_ratio: 0.5`

### 关键配置项

```yaml
type: Contrast
comment: defense_only
enabled_attacks: []
enabled_defenses:
  - norm_clip
enabled_privacy_metrics: []
enable_malicious_clients: false
defense_clip_norm: 5.0
```

### 预期输出字段

- `experiment_mode: defense_only`
- `scenario_tags` 至少包含：
  - `defense_only`
- `active_defenses: ["norm_clip"]`
- `round_summaries[*].clipped_client_count`
- 详细版中的：
  - `round_metrics[*].extra.defense_metrics.norm_clip.clipped_client_count`
  - 或 `round_metrics[*].extra.defense_metrics.update_filter.filtered_client_count`

### 适合展示的重点

- 防御模块已启用但没有攻击时的表现
- 裁剪型防御的结构化记录能力
- 为后续攻防对照提供单独防御组
- 可作为 `norm_clip` 与 `update_filter` 的轻量防御对照入口

## 7. 标准实验四：attack_and_defense

### 目标

形成当前阶段最核心的最小攻防对照实验。

### 建议配置

- 建议模型：`FedRAP`
- 建议数据集：`KU`
- `enabled_attacks: ["client_update_scale"]`
- `enabled_defenses: ["norm_clip"]`
- `enabled_privacy_metrics: []`
- `enable_malicious_clients: true`
- `malicious_client_mode: "ratio"`
- `malicious_client_ratio: 0.2`
- `attack_scale: 3.0`
- `defense_clip_norm: 5.0`

如果需要使用第二个主动攻击模块形成变体实验，也可以将攻击项替换为：

- `enabled_attacks: ["sign_flip"]`
- `sign_flip_scale: 3.0`

如果需要使用第二个主动防御模块形成变体实验，也可以将防御项替换为：

- `enabled_defenses: ["update_filter"]`
- `filter_std_factor: 2.0`
- `max_filtered_ratio: 0.5`

### 关键配置项

```yaml
type: Contrast
comment: attack_and_defense
enabled_attacks:
  - client_update_scale
enabled_defenses:
  - norm_clip
enabled_privacy_metrics: []
enable_malicious_clients: true
malicious_client_mode: ratio
malicious_client_ratio: 0.2
attack_scale: 3.0
defense_clip_norm: 5.0
```

### 预期输出字段

- `experiment_mode: attack_and_defense`
- `scenario_tags` 至少包含：
  - `attack_and_defense`
  - `malicious_clients_configured`
- `active_attacks: ["client_update_scale"]`
- `active_defenses: ["norm_clip"]`
- `malicious_client_summary`
- `round_summaries[*].attacked_client_count`
- `round_summaries[*].clipped_client_count`

### 适合展示的重点

- 从攻击到防御的最小闭环
- 场景标签、结果字段与实验模块的一一对应
- 答辩中最适合讲清楚的主实验
- 后续也适合扩展成 `sign_flip + norm_clip` 或 `sign_flip + update_filter` 的对照变体

## 8. 标准实验五：attack_and_defense_with_privacy_observation

### 目标

在最小攻防对照实验基础上，同时加入只读隐私观测，形成“性能 + 攻防 + 隐私”三重视角。

### 建议配置

- 建议模型：`FedRAP`
- 建议数据集：`KU`
- `enabled_attacks: ["client_update_scale"]`
- `enabled_defenses: ["norm_clip"]`
- `enabled_privacy_metrics: ["client_update_norm"]`
- `enable_malicious_clients: true`
- `malicious_client_mode: "ratio"`
- `malicious_client_ratio: 0.2`
- `attack_scale: 3.0`
- `defense_clip_norm: 5.0`

如果需要保留隐私观测同时切换到第二攻击模块，也可以使用：

- `enabled_attacks: ["sign_flip"]`
- `sign_flip_scale: 3.0`

### 关键配置项

```yaml
type: Contrast
comment: attack_and_defense_with_privacy_observation
enabled_attacks:
  - client_update_scale
enabled_defenses:
  - norm_clip
enabled_privacy_metrics:
  - client_update_norm
enable_malicious_clients: true
malicious_client_mode: ratio
malicious_client_ratio: 0.2
attack_scale: 3.0
defense_clip_norm: 5.0
```

### 预期输出字段

- `experiment_mode: attack_and_defense`
- `scenario_tags` 至少包含：
  - `attack_and_defense`
  - `privacy_observation`
  - `malicious_clients_configured`
- `active_attacks: ["client_update_scale"]`
- `active_defenses: ["norm_clip"]`
- `active_privacy_metrics: ["client_update_norm"]`
- `round_summaries[*].attacked_client_count`
- `round_summaries[*].clipped_client_count`
- 详细版中的：
  - `round_metrics[*].extra.privacy_metric_outputs.client_update_norm`

### 适合展示的重点

- 攻防实验之外，平台还能输出隐私观测信息
- 结果结构已经适合 API / 前端 / 答辩统一复用

## 9. 标准实验矩阵速览表

| 场景 | 攻击 | 防御 | 隐私观测 | malicious clients | 核心展示点 |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 无 | 无 | 无 | 关闭 | 基线性能 |
| `attack_only` | `client_update_scale` / `sign_flip` | 无 | 无 | 开启 | 攻击影响 |
| `defense_only` | 无 | `norm_clip` / `update_filter` | 无 | 关闭 | 防御预处理能力 |
| `attack_and_defense` | `client_update_scale` / `sign_flip` | `norm_clip` / `update_filter` | 无 | 开启 | 最小攻防闭环 |
| `attack_and_defense_with_privacy_observation` | `client_update_scale` / `sign_flip` | `norm_clip` / `update_filter` | `client_update_norm` | 开启 | 攻防 + 隐私三重视角 |

## 10. 当前最适合答辩展示的重点组合

如果只展示最小但完整的一条主线，建议优先展示这 3 组：

1. `baseline`
2. `attack_only`
3. `attack_and_defense`

理由：

- 场景最清楚
- 对比关系最直观
- 能完整讲清当前项目已经做成的攻防闭环

如果还需要再补一组增强展示，可补：

4. `attack_and_defense_with_privacy_observation`

## 11. 建议重点检查的输出字段

所有标准实验，建议统一检查：

- 详细版结果：
  - `*.experiment_result.json`
- 摘要版结果：
  - `*.experiment_summary.json`

重点字段建议为：

- `experiment_mode`
- `scenario_tags`
- `active_attacks`
- `active_defenses`
- `active_privacy_metrics`
- `malicious_client_summary`
- `final_eval`
- `round_summaries`

在需要深入分析时，再进一步查看详细版中的：

- `round_metrics[*].extra.attack_metrics`
- `round_metrics[*].extra.defense_metrics`
- `round_metrics[*].extra.privacy_metric_outputs`

## 12. 后续增强实验矩阵（规划，不代表已实现）

以下内容属于下一阶段规划，当前尚未实现：

### 12.1 `AdaptiveClipDefense`

- 目标：在固定阈值裁剪基础上加入更自适应的裁剪策略
- 预期场景：
  - `defense_only`
  - `attack_and_defense`

### 12.2 `MMFedRAP` 兼容实验

- 目标：将当前标准实验矩阵迁移到多模态主线
- 预期价值：
  - 更贴合项目“多模态推荐安全实验平台”的最终方向

## 13. 已完成内容与下一阶段规划的边界

### 当前已完成

- 标准实验所需的最小模块链路已经具备
- `SignFlipAttack` 已可作为第二个主动攻击模块接入现有标准实验场景
- `UpdateFilterDefense` 已可作为第二个主动防御模块接入现有标准实验场景
- 结果结构已支持场景标签表达
- 详细版 / 摘要版结果均可自动写盘

### 当前仍属规划

- 更复杂攻击
- 更复杂防御
- 更完整隐私评估矩阵
- `MMFedRAP` 标准化实验跑通

因此当前标准实验矩阵应以“已经可以运行和展示的最小闭环”为核心，不应把规划项写成既成事实。
