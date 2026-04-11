# strong_adv_showcase 正式强攻防展示结果

- 生成时间：`2026-04-11T22:40:52`
- 模型：`FedRAP`
- 数据集：`KU`
- 实验类型：`StrongAdvShowcase`
- 注释前缀：`strong_adv_showcase`

| 场景 | 模式 | 攻击 | 防御 | Recall@20 | NDCG@20 | Loss | malicious | attacked | clipped | filtered | trimmed_mean | effective_trim_count | 展示说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | - | - | 0.1225 | 0.0365 | 0.5470 | 0 | 0 | 0 | 0 | no | 0 | 正常基线：无攻击、无主动防御，用于提供强攻防展示实验的推荐性能参照。 |
| attack_only_model_replacement | attack_only | model_replacement | - | 0.0637 | 0.0221 | 0.5894 | 41 | 41 | 0 | 0 | no | 0 | 强攻击组：恶意客户端执行 minimal model-replacement-like 更新，观察推荐性能退化与攻击计数字段。 |
| attack_and_defense_trimmed_mean | attack_and_defense | model_replacement | trimmed_mean | 0.1275 | 0.0377 | 0.5913 | 41 | 41 | 0 | 0 | yes | 40 | 强攻防组：在 model_replacement 攻击后启用 trimmed_mean，观察鲁棒聚合式防御对极端更新的约束与性能恢复。 |

## 明细文件

- `baseline`
  - summary: `outputs/results/FedRAP/KU/StrongAdvShowcase/[FedRAP]-[KU]-[StrongAdvShowcase.strong_adv_showcase.baseline].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StrongAdvShowcase/[FedRAP]-[KU]-[StrongAdvShowcase.strong_adv_showcase.baseline].experiment_result.json`
- `attack_only_model_replacement`
  - summary: `outputs/results/FedRAP/KU/StrongAdvShowcase/[FedRAP]-[KU]-[StrongAdvShowcase.strong_adv_showcase.attack_only_model_replacement].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StrongAdvShowcase/[FedRAP]-[KU]-[StrongAdvShowcase.strong_adv_showcase.attack_only_model_replacement].experiment_result.json`
- `attack_and_defense_trimmed_mean`
  - summary: `outputs/results/FedRAP/KU/StrongAdvShowcase/[FedRAP]-[KU]-[StrongAdvShowcase.strong_adv_showcase.attack_and_defense_trimmed_mean].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/StrongAdvShowcase/[FedRAP]-[KU]-[StrongAdvShowcase.strong_adv_showcase.attack_and_defense_trimmed_mean].experiment_result.json`
