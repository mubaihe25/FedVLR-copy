# FedVLR 标准实验矩阵运行方案

## 1. 文档目的

本文档补充当前标准实验矩阵的“实际运行方案”，目标是让当前已经完成的最小攻防实验闭环可以直接批量运行，并稳定产出：

- `*.experiment_result.json`
- `*.experiment_summary.json`
- `*.csv`

当前优先固定：

- 模型：`FedRAP`
- 数据集：`KU`

## 2. 当前推荐立即执行的 5 组标准实验

### 2.1 baseline

- 目标：建立无攻击、无主动防御的基线结果。
- 建议模型：`FedRAP`
- 建议数据集：`KU`
- `enabled_attacks: []`
- `enabled_defenses: []`
- `enabled_privacy_metrics: []`
- `enable_malicious_clients: false`
- 关键配置：
  - `epochs: 3`
  - `local_epochs: 1`
  - `clients_sample_ratio: 1.0`
- 预期输出：
  - `experiment_mode = "baseline"`
  - `scenario_tags = ["baseline"]`
- 重点观察字段：
  - `final_eval`
  - `round_summaries[*].avg_train_loss`

### 2.2 attack_only_scale

- 目标：观察更新缩放攻击对结果指标和场景标签的影响。
- 建议模型：`FedRAP`
- 建议数据集：`KU`
- `enabled_attacks: ["client_update_scale"]`
- `enabled_defenses: []`
- `enabled_privacy_metrics: []`
- `enable_malicious_clients: true`
- malicious client 配置：
  - `malicious_client_mode: "ratio"`
  - `malicious_client_ratio: 0.2`
- 关键配置：
  - `attack_scale: 3.0`
- 预期输出：
  - `experiment_mode = "attack_only"`
  - `scenario_tags` 至少包含：
    - `attack_only`
    - `malicious_clients_configured`
- 重点观察字段：
  - `malicious_client_summary`
  - `round_summaries[*].malicious_client_count`
  - `round_metrics[*].extra.attack_metrics.client_update_scale`

### 2.3 attack_only_sign_flip

- 目标：观察更接近经典联邦更新攻击的符号翻转效果。
- 建议模型：`FedRAP`
- 建议数据集：`KU`
- `enabled_attacks: ["sign_flip"]`
- `enabled_defenses: []`
- `enabled_privacy_metrics: []`
- `enable_malicious_clients: true`
- malicious client 配置：
  - `malicious_client_mode: "ratio"`
  - `malicious_client_ratio: 0.2`
- 关键配置：
  - `sign_flip_scale: 1.0`
- 预期输出：
  - `experiment_mode = "attack_only"`
  - `scenario_tags` 至少包含：
    - `attack_only`
    - `malicious_clients_configured`
- 重点观察字段：
  - `malicious_client_summary`
  - `round_metrics[*].extra.attack_metrics.sign_flip`

### 2.4 attack_and_defense_clip

- 目标：形成 `ClientUpdateScaleAttack + NormClipDefense` 的最小攻防对照。
- 建议模型：`FedRAP`
- 建议数据集：`KU`
- `enabled_attacks: ["client_update_scale"]`
- `enabled_defenses: ["norm_clip"]`
- `enabled_privacy_metrics: []`
- `enable_malicious_clients: true`
- malicious client 配置：
  - `malicious_client_mode: "ratio"`
  - `malicious_client_ratio: 0.2`
- 关键配置：
  - `attack_scale: 3.0`
  - `defense_clip_norm: 5.0`
- 预期输出：
  - `experiment_mode = "attack_and_defense"`
  - `scenario_tags` 至少包含：
    - `attack_and_defense`
    - `malicious_clients_configured`
- 重点观察字段：
  - `round_summaries[*].attacked_client_count`
  - `round_summaries[*].clipped_client_count`
  - `round_metrics[*].extra.attack_metrics.client_update_scale`
  - `round_metrics[*].extra.defense_metrics.norm_clip`

### 2.5 attack_and_defense_filter

- 目标：形成 `SignFlipAttack + UpdateFilterDefense` 的最小攻防对照。
- 建议模型：`FedRAP`
- 建议数据集：`KU`
- `enabled_attacks: ["sign_flip"]`
- `enabled_defenses: ["update_filter"]`
- `enabled_privacy_metrics: []`
- `enable_malicious_clients: true`
- malicious client 配置：
  - `malicious_client_mode: "ratio"`
  - `malicious_client_ratio: 0.2`
- 关键配置：
  - `sign_flip_scale: 3.0`
  - `filter_std_factor: 2.0`
  - `max_filtered_ratio: 0.5`
- 预期输出：
  - `experiment_mode = "attack_and_defense"`
  - `scenario_tags` 至少包含：
    - `attack_and_defense`
    - `malicious_clients_configured`
- 重点观察字段：
  - `round_summaries[*].attacked_client_count`
  - `round_metrics[*].extra.defense_metrics.update_filter`
  - `round_metrics[*].extra.attack_metrics.sign_flip`

## 3. 最小运行脚本

当前仓库已补充：

- `scripts/run_standard_matrix.py`

该脚本采用当前已经验证通过的最小链路：

- `Config`
- `_prepare_data`
- `get_model`
- `get_trainer`
- `trainer.fit`

不走复杂 CLI 超参网格。

## 4. 推荐运行方式

### 4.1 查看可用场景

```powershell
.\.venv\Scripts\python.exe scripts\run_standard_matrix.py --list-scenarios
```

### 4.2 仅打印解析后的配置，不启动训练

```powershell
.\.venv\Scripts\python.exe scripts\run_standard_matrix.py --dry-run
```

### 4.3 一次运行全部 5 组实验

```powershell
.\.venv\Scripts\python.exe scripts\run_standard_matrix.py
```

### 4.4 只运行其中几组

```powershell
.\.venv\Scripts\python.exe scripts\run_standard_matrix.py `
  --scenario baseline `
  --scenario attack_only_scale `
  --scenario attack_and_defense_clip
```

### 4.5 指定新的输出标识，避免覆盖旧结果

```powershell
.\.venv\Scripts\python.exe scripts\run_standard_matrix.py --comment-prefix matrix_v2
```

## 5. 输出目录与命名

输出目录沿用当前已有体系：

- `outputs/results/{model}/{dataset}/{type}/`

默认 `type` 为：

- `StandardMatrix`

默认 `comment_prefix` 为：

- `matrix`

因此类似 `baseline` 场景的输出文件会是：

- `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix.baseline].csv`
- `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix.baseline].experiment_result.json`
- `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix.baseline].experiment_summary.json`

## 6. 建议重点检查的结果字段

建议统一优先检查：

- `experiment_mode`
- `scenario_tags`
- `active_attacks`
- `active_defenses`
- `active_privacy_metrics`
- `malicious_client_summary`
- `final_eval`
- `round_summaries`

在需要做更深分析时，再查看详细版中的：

- `round_metrics[*].extra.attack_metrics`
- `round_metrics[*].extra.defense_metrics`
- `round_metrics[*].extra.privacy_metric_outputs`

## 7. 当前最可能遇到的阻塞点

1. 必须从 `FedVLR` 根目录运行脚本。
   - `Config` 依赖当前工作目录定位 `configs/`

2. 必须使用正确的注册表名称。
   - 攻击：
     - `client_update_scale`
     - `sign_flip`
   - 防御：
     - `norm_clip`
     - `update_filter`
   - 隐私观测：
     - `client_update_norm`

3. 必须显式关闭超参网格。
   - 当前脚本已经统一覆盖：
     - `hyper_parameters: []`

4. 攻击场景必须同时开启 malicious client 配置。
   - 否则攻击模块会被加载，但不会命中目标客户端

5. `UpdateFilterDefense` 需要防止过滤过多客户端。
   - 当前推荐：
     - `filter_std_factor: 2.0`
     - `max_filtered_ratio: 0.5`

6. 纯 `sign_flip_scale = 1.0` 的攻击更接近经典 sign flip，但未必足够触发基于范数的过滤。
   - 因此在 `attack_and_defense_filter` 场景中，推荐：
     - `sign_flip_scale: 3.0`

## 8. 跑完后下一步最适合做什么

建议优先完成两件事：

1. 先稳定产出 5 组 `experiment_result.json / experiment_summary.json`
2. 再从中挑出最适合比赛展示的 3 组做前端对照展示：
   - `baseline`
   - `attack_only_scale` 或 `attack_only_sign_flip`
   - `attack_and_defense_clip` 或 `attack_and_defense_filter`

当前阶段最重要的是先把“标准实验矩阵能稳定复跑、结果命名清晰、场景标签清楚”这件事固定下来。
