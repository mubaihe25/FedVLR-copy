# showcase_v1 对比摘要

- 生成时间：`2026-04-11T21:05:50`
- 模型：`FedRAP`
- 数据集：`KU`
- 输出目录：`outputs/results/FedRAP/KU/ShowcaseV1`

| 场景 | 模式 | 攻击 | 防御 | Recall@20 | NDCG@20 | Loss | malicious | attacked | clipped | filtered | 展示说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | - | - | 0.1225 | 0.0365 | 0.5772 | 0 | 0 | 0 | 0 | 正常基线：无攻击、无主动防御，用于提供推荐性能参照。 |
| attack_only_sign_flip | attack_only | sign_flip | - | 0.0980 | 0.0316 | 0.6276 | 41 | 41 | 0 | 0 | 攻击场景：恶意客户端执行符号翻转，观察推荐性能退化。 |
| attack_and_defense_clip | attack_and_defense | client_update_scale | norm_clip | 0.1176 | 0.0338 | 0.5626 | 41 | 41 | 41 | 0 | 攻防场景：攻击后启用范数裁剪，对异常更新进行约束。 |

## 明细文件

- `baseline`
  - summary: `outputs/results/FedRAP/KU/ShowcaseV1/[FedRAP]-[KU]-[ShowcaseV1.showcase_v1.baseline].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/ShowcaseV1/[FedRAP]-[KU]-[ShowcaseV1.showcase_v1.baseline].experiment_result.json`
- `attack_only_sign_flip`
  - summary: `outputs/results/FedRAP/KU/ShowcaseV1/[FedRAP]-[KU]-[ShowcaseV1.showcase_v1.attack_only_sign_flip].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/ShowcaseV1/[FedRAP]-[KU]-[ShowcaseV1.showcase_v1.attack_only_sign_flip].experiment_result.json`
- `attack_and_defense_clip`
  - summary: `outputs/results/FedRAP/KU/ShowcaseV1/[FedRAP]-[KU]-[ShowcaseV1.showcase_v1.attack_and_defense_clip].experiment_summary.json`
  - result: `outputs/results/FedRAP/KU/ShowcaseV1/[FedRAP]-[KU]-[ShowcaseV1.showcase_v1.attack_and_defense_clip].experiment_result.json`
