# MMFedRAP strong_adv_showcase 正式展示结果

- 生成时间：`2026-04-11T23:10:10`
- 模型：`MMFedRAP`
- 数据集：`KU`
- 实验类型：`MMFedRAPStrongAdvShowcase`
- 注释前缀：`mmfedrap_strong_adv_showcase`

| 场景 | 模式 | 攻击 | 防御 | Recall@20 | NDCG@20 | Loss | malicious | attacked | trimmed_mean | effective_trim_count | 展示说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | - | - | 0.0441 | 0.0266 | 0.6918 | 0 | 0 | no | 0 | MMFedRAP 正式基线：无攻击、无主动防御，用于提供多模态强攻防展示参照。 |
| attack_only_model_replacement | attack_only | model_replacement | - | 0.0392 | 0.0203 | 0.6914 | 41 | 41 | no | 0 | MMFedRAP 强攻击组：恶意客户端执行 minimal model-replacement-like 更新，观察多模态推荐性能退化。 |
| attack_and_defense_trimmed_mean | attack_and_defense | model_replacement | trimmed_mean | 0.0588 | 0.0241 | 0.6915 | 41 | 41 | yes | 40 | MMFedRAP 强攻防组：在 model_replacement 攻击后启用 trimmed_mean，观察鲁棒聚合式防御的恢复效果。 |

## 明细文件

- `baseline`
  - summary: `outputs/results/MMFedRAP/KU/MMFedRAPStrongAdvShowcase/[MMFedRAP]-[KU]-[MMFedRAPStrongAdvShowcase.mmfedrap_strong_adv_showcase.baseline].experiment_summary.json`
  - result: `outputs/results/MMFedRAP/KU/MMFedRAPStrongAdvShowcase/[MMFedRAP]-[KU]-[MMFedRAPStrongAdvShowcase.mmfedrap_strong_adv_showcase.baseline].experiment_result.json`
- `attack_only_model_replacement`
  - summary: `outputs/results/MMFedRAP/KU/MMFedRAPStrongAdvShowcase/[MMFedRAP]-[KU]-[MMFedRAPStrongAdvShowcase.mmfedrap_strong_adv_showcase.attack_only_model_replacement].experiment_summary.json`
  - result: `outputs/results/MMFedRAP/KU/MMFedRAPStrongAdvShowcase/[MMFedRAP]-[KU]-[MMFedRAPStrongAdvShowcase.mmfedrap_strong_adv_showcase.attack_only_model_replacement].experiment_result.json`
- `attack_and_defense_trimmed_mean`
  - summary: `outputs/results/MMFedRAP/KU/MMFedRAPStrongAdvShowcase/[MMFedRAP]-[KU]-[MMFedRAPStrongAdvShowcase.mmfedrap_strong_adv_showcase.attack_and_defense_trimmed_mean].experiment_summary.json`
  - result: `outputs/results/MMFedRAP/KU/MMFedRAPStrongAdvShowcase/[MMFedRAP]-[KU]-[MMFedRAPStrongAdvShowcase.mmfedrap_strong_adv_showcase.attack_and_defense_trimmed_mean].experiment_result.json`
