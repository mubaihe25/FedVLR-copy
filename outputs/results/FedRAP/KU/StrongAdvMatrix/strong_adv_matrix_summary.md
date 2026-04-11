# strong_adv_matrix 强攻防结果汇总

- 生成时间：`2026-04-11T22:29:08`
- 模型：`FedRAP`
- 数据集：`KU`
- 实验类型：`StrongAdvMatrix`
- 注释前缀：`strong_adv_matrix`

| 场景 | 模式 | 攻击 | 防御 | Recall@20 | NDCG@20 | Loss | malicious | attacked | clipped | filtered | trimmed_mean | trim_count | 展示说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | - | - | 0.1225 | 0.0365 | 0.5772 | 0 | 0 | 0 | 0 | no | 0 | 正常基线：无攻击、无主动防御，用于提供强攻防实验的推荐性能参照。 |
| attack_only_model_replacement | attack_only | model_replacement | - | 0.1029 | 0.0304 | 0.5092 | 41 | 41 | 0 | 0 | no | 0 | 强攻击组：恶意客户端执行 minimal model-replacement-like 更新，观察性能退化与攻击计数字段。 |
| attack_and_defense_trimmed_mean | attack_and_defense | model_replacement | trimmed_mean | 0.1225 | 0.0375 | 0.6212 | 41 | 41 | 0 | 0 | yes | 40 | 强攻防组：在 model_replacement 攻击后启用 trimmed_mean，观察鲁棒聚合式防御对极端更新的约束。 |

## 明细文件

- `baseline`
  - summary: `outputs/results/FedRAP/KU/StrongAdvMatrix/[FedRAP]-[KU]-[StrongAdvMatrix.strong_adv_matrix.baseline].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StrongAdvMatrix/[FedRAP]-[KU]-[StrongAdvMatrix.strong_adv_matrix.baseline].experiment_result.json`
- `attack_only_model_replacement`
  - summary: `outputs/results/FedRAP/KU/StrongAdvMatrix/[FedRAP]-[KU]-[StrongAdvMatrix.strong_adv_matrix.attack_only_model_replacement].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StrongAdvMatrix/[FedRAP]-[KU]-[StrongAdvMatrix.strong_adv_matrix.attack_only_model_replacement].experiment_result.json`
- `attack_and_defense_trimmed_mean`
  - summary: `outputs/results/FedRAP/KU/StrongAdvMatrix/[FedRAP]-[KU]-[StrongAdvMatrix.strong_adv_matrix.attack_and_defense_trimmed_mean].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StrongAdvMatrix/[FedRAP]-[KU]-[StrongAdvMatrix.strong_adv_matrix.attack_and_defense_trimmed_mean].experiment_result.json`
