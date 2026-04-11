# model_family_strong_adv_compat 批量兼容验证摘要

- 生成时间：`2026-04-11T23:36:47`
- 数据集：`KU`
- 实验类型：`ModelFamilyStrongAdvCompat`
- 场景：`baseline, attack_only_model_replacement, attack_and_defense_trimmed_mean`

## 模型风险点

- `FedAvg`：单模态 FederatedTrainer 路径；默认 lr/l2_reg 为列表，需要脚本显式标量化。；上传更新通常是模型参数字典，现有递归 hook 可处理。
- `FedNCF`：单模态 FederatedTrainer 路径；默认 lr/l2_reg 为列表，需要脚本显式标量化。；上传更新通常是嵌套参数字典，需确认 model_replacement/trimmed_mean 不破坏聚合输入。
- `FedRAP`：已完成正式展示主线；本批次用于横向基准复核。；上传更新主要包含 item_commonality.weight，结构最接近已验证路径。
- `FedVBPR`：标记为多模态联邦模型，依赖 KU 的 image/text 特征文件。；默认 lr/reg_weight 为列表，需要脚本显式标量化。
- `PFedRec`：用户称 pFedRec，工程实际类名为 PFedRec。；默认 lr/l2_reg 为列表，需要脚本显式标量化。
- `MMFedAvg`：多模态 FederatedTrainer 路径，依赖 fusion 层与 item_commonality 上传更新。；默认 lr/l2_reg 为列表，需要脚本显式标量化。
- `MMFedNCF`：多模态 FederatedTrainer 路径，依赖 fusion 层与 item_commonality 上传更新。；默认 lr/l2_reg 为列表，需要脚本显式标量化。
- `MMFedRAP`：已完成 MMFedRAP 正式展示验证；本批次用于横向复核。；上传更新包含 fusion.* 梯度和 item_commonality.weight。
- `MMPFedRec`：多模态个性化联邦模型，依赖 fusion 层与个性化参数存储。；默认 lr/l2_reg 为列表，需要脚本显式标量化。
- `MMGCN`：工程中存在 MMGCN 模型类，但没有 MMGCNTrainer。；当前不在 FederatedTrainer hook 链路中，不适合强行验证 model_replacement + trimmed_mean。

## 运行结果

| 模型 | 场景 | 状态 | 失败阶段 | Recall@20 | NDCG@20 | Loss | malicious | attacked | trimmed_mean | trim_count | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg | baseline | success | - | 0.0245 | 0.0138 | 0.6906 | 0 | 0 | no | 0 | FedAvg / baseline 已跑通，hook 指标和结果文件已生成。 |
| FedAvg | attack_only_model_replacement | success | - | 0.0245 | 0.0138 | 0.6906 | 41 | 41 | no | 0 | FedAvg / attack_only_model_replacement 已跑通，hook 指标和结果文件已生成。 |
| FedAvg | attack_and_defense_trimmed_mean | success | - | 0.0245 | 0.0138 | 0.6906 | 41 | 41 | yes | 40 | FedAvg / attack_and_defense_trimmed_mean 已跑通，hook 指标和结果文件已生成。 |
| FedNCF | baseline | success | - | 0.0735 | 0.0254 | 0.6933 | 0 | 0 | no | 0 | FedNCF / baseline 已跑通，hook 指标和结果文件已生成。 |
| FedNCF | attack_only_model_replacement | success | - | 0.0735 | 0.0254 | 0.6933 | 41 | 41 | no | 0 | FedNCF / attack_only_model_replacement 已跑通，hook 指标和结果文件已生成。 |
| FedNCF | attack_and_defense_trimmed_mean | success | - | 0.0735 | 0.0254 | 0.6933 | 41 | 41 | yes | 40 | FedNCF / attack_and_defense_trimmed_mean 已跑通，hook 指标和结果文件已生成。 |
| FedRAP | baseline | success | - | 0.1225 | 0.0376 | 0.6940 | 0 | 0 | no | 0 | FedRAP / baseline 已跑通，hook 指标和结果文件已生成。 |
| FedRAP | attack_only_model_replacement | success | - | 0.0686 | 0.0234 | 0.6940 | 41 | 41 | no | 0 | FedRAP / attack_only_model_replacement 已跑通，hook 指标和结果文件已生成。 |
| FedRAP | attack_and_defense_trimmed_mean | success | - | 0.1912 | 0.0564 | 0.6940 | 41 | 41 | yes | 40 | FedRAP / attack_and_defense_trimmed_mean 已跑通，hook 指标和结果文件已生成。 |
| FedVBPR | baseline | success | - | 0.0294 | 0.0084 | 0.6950 | 0 | 0 | no | 0 | FedVBPR / baseline 已跑通，hook 指标和结果文件已生成。 |
| FedVBPR | attack_only_model_replacement | success | - | 0.0294 | 0.0082 | 0.6950 | 41 | 41 | no | 0 | FedVBPR / attack_only_model_replacement 已跑通，hook 指标和结果文件已生成。 |
| FedVBPR | attack_and_defense_trimmed_mean | success | - | 0.0294 | 0.0083 | 0.6950 | 41 | 41 | yes | 40 | FedVBPR / attack_and_defense_trimmed_mean 已跑通，hook 指标和结果文件已生成。 |
| PFedRec | baseline | success | - | 0.0245 | 0.0099 | 0.6938 | 0 | 0 | no | 0 | PFedRec / baseline 已跑通，hook 指标和结果文件已生成。 |
| PFedRec | attack_only_model_replacement | success | - | 0.0245 | 0.0099 | 0.6938 | 41 | 41 | no | 0 | PFedRec / attack_only_model_replacement 已跑通，hook 指标和结果文件已生成。 |
| PFedRec | attack_and_defense_trimmed_mean | success | - | 0.0245 | 0.0099 | 0.6938 | 41 | 41 | yes | 40 | PFedRec / attack_and_defense_trimmed_mean 已跑通，hook 指标和结果文件已生成。 |
| MMFedAvg | baseline | success | - | 0.0196 | 0.0092 | 0.7241 | 0 | 0 | no | 0 | MMFedAvg / baseline 已跑通，hook 指标和结果文件已生成。 |
| MMFedAvg | attack_only_model_replacement | success | - | 0.0196 | 0.0091 | 0.7241 | 41 | 41 | no | 0 | MMFedAvg / attack_only_model_replacement 已跑通，hook 指标和结果文件已生成。 |
| MMFedAvg | attack_and_defense_trimmed_mean | success | - | 0.0196 | 0.0091 | 0.7241 | 41 | 41 | yes | 40 | MMFedAvg / attack_and_defense_trimmed_mean 已跑通，hook 指标和结果文件已生成。 |
| MMFedNCF | baseline | success | - | 0.0245 | 0.0068 | 0.6928 | 0 | 0 | no | 0 | MMFedNCF / baseline 已跑通，hook 指标和结果文件已生成。 |
| MMFedNCF | attack_only_model_replacement | success | - | 0.0245 | 0.0070 | 0.6928 | 41 | 41 | no | 0 | MMFedNCF / attack_only_model_replacement 已跑通，hook 指标和结果文件已生成。 |
| MMFedNCF | attack_and_defense_trimmed_mean | success | - | 0.0245 | 0.0069 | 0.6928 | 41 | 41 | yes | 40 | MMFedNCF / attack_and_defense_trimmed_mean 已跑通，hook 指标和结果文件已生成。 |
| MMFedRAP | baseline | success | - | 0.0490 | 0.0283 | 0.6936 | 0 | 0 | no | 0 | MMFedRAP / baseline 已跑通，hook 指标和结果文件已生成。 |
| MMFedRAP | attack_only_model_replacement | success | - | 0.0392 | 0.0206 | 0.6936 | 41 | 41 | no | 0 | MMFedRAP / attack_only_model_replacement 已跑通，hook 指标和结果文件已生成。 |
| MMFedRAP | attack_and_defense_trimmed_mean | success | - | 0.0490 | 0.0279 | 0.6936 | 41 | 41 | yes | 40 | MMFedRAP / attack_and_defense_trimmed_mean 已跑通，hook 指标和结果文件已生成。 |
| MMPFedRec | baseline | success | - | 0.0245 | 0.0112 | 0.6931 | 0 | 0 | no | 0 | MMPFedRec / baseline 已跑通，hook 指标和结果文件已生成。 |
| MMPFedRec | attack_only_model_replacement | success | - | 0.0196 | 0.0101 | 0.6931 | 41 | 41 | no | 0 | MMPFedRec / attack_only_model_replacement 已跑通，hook 指标和结果文件已生成。 |
| MMPFedRec | attack_and_defense_trimmed_mean | success | - | 0.0196 | 0.0097 | 0.6931 | 41 | 41 | yes | 40 | MMPFedRec / attack_and_defense_trimmed_mean 已跑通，hook 指标和结果文件已生成。 |
| MMGCN | baseline | failed | precheck | - | - | - | 0 | 0 | no | 0 | MMGCN / baseline 未跑通：trainer_import_failed:missing_dependency:torch_geometric。 |
| MMGCN | attack_only_model_replacement | failed | precheck | - | - | - | 0 | 0 | no | 0 | MMGCN / attack_only_model_replacement 未跑通：trainer_import_failed:missing_dependency:torch_geometric。 |
| MMGCN | attack_and_defense_trimmed_mean | failed | precheck | - | - | - | 0 | 0 | no | 0 | MMGCN / attack_and_defense_trimmed_mean 未跑通：trainer_import_failed:missing_dependency:torch_geometric。 |
