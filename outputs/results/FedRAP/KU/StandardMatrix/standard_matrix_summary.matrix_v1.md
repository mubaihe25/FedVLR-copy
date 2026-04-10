# 标准实验矩阵结果汇总

- 生成时间：`2026-04-09T22:29:01`
- 模型：`FedRAP`
- 数据集：`KU`
- 实验类型：`StandardMatrix`
- 注释筛选：`matrix_v1`

| 场景 | 模式 | 攻击 | 防御 | 隐私观测 | Recall@20 | NDCG@20 | Loss | malicious | attacked | clipped | filtered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | - | - | - | 0.1225 | 0.0365 | 0.5772 | 0 | 0 | 0 | 0 |
| attack_only_scale | attack_only | client_update_scale | - | - | 0.1225 | 0.0356 | 0.5345 | 41 | 41 | 0 | 0 |
| attack_only_sign_flip | attack_only | sign_flip | - | - | 0.0980 | 0.0316 | 0.6276 | 41 | 41 | 0 | 0 |
| attack_and_defense_clip | attack_and_defense | client_update_scale | norm_clip | - | 0.0784 | 0.0285 | 0.6594 | 41 | 41 | 204 | 0 |
| attack_and_defense_filter | attack_and_defense | sign_flip | update_filter | - | 0.0833 | 0.0241 | 0.6235 | 41 | 41 | 0 | 24 |

## 明细文件路径

- `baseline`
  - summary: `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix_v1.baseline].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix_v1.baseline].experiment_result.json`
- `attack_only_scale`
  - summary: `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix_v1.attack_only_scale].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix_v1.attack_only_scale].experiment_result.json`
- `attack_only_sign_flip`
  - summary: `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix_v1.attack_only_sign_flip].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix_v1.attack_only_sign_flip].experiment_result.json`
- `attack_and_defense_clip`
  - summary: `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix_v1.attack_and_defense_clip].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix_v1.attack_and_defense_clip].experiment_result.json`
- `attack_and_defense_filter`
  - summary: `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix_v1.attack_and_defense_filter].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StandardMatrix/[FedRAP]-[KU]-[StandardMatrix.matrix_v1.attack_and_defense_filter].experiment_result.json`
