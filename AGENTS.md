# FedVLR 协作说明

## 仓库定位

`FedVLR` 是算法核心仓库，负责多模态推荐模型、训练流程、联邦学习模拟、攻击模块、防御模块、风险观测、实验配置以及 CSV/JSON/summary 结果输出。

## 重点目录

- `models`：推荐模型和多模态融合模型。
- `common`：通用训练器、损失函数和基础抽象。
- `utils`：配置、数据加载、训练入口、评估、结果输出和联邦训练支持。
- `attacks`：投毒攻击、更新缩放、符号翻转、模型替换、风险探针等模块。
- `defenses`：范数裁剪、更新过滤、截尾均值、Median、Krum、central DP-style noise、鲁棒防御入口和异常检测模块。
- `privacy_eval`：风险观测、成员推断 probe、梯度泄露 demo probe 和隐私相关指标输出。
- `configs`：模型、数据集、能力矩阵、统一实验 schema 和 batch 配置。
- `scripts`：实验启动、batch 运行、结果汇总、showcase artifact 导出和验证脚本。
- `outputs`：实验输出目录；普通输出不要提交，已有 curated 展示结果不要自动删除。

## 开发约束

- 不要随意改变训练主链路、hook 顺序、恶意客户端选择逻辑、聚合参数结构。
- 不要随意修改 CSV/JSON/summary 字段名；API 和前端依赖这些结果字段。
- 不要把差分隐私、同态加密、安全聚合写成已正式实现。
- `dp_noise` 只能表述为聚合前裁剪加 Gaussian noise 的 central DP-style noise defense；当前没有 formal privacy accountant，不能写成完整差分隐私。
- `membership_inference_probe` 只能表述为 score-based privacy probe；`gradient_leakage_probe` 只能表述为 demo/probe，不能写成完整 DLG/InvertingGrad 或原始图像重建能力。
- 当前安全主线应表述为投毒攻击、鲁棒防御和风险观测。
- 不要提交普通 `outputs` 实验结果、`.venv`、`__pycache__`、`.pyc`、日志或临时文件。
- `outputs` 中已有被 Git 跟踪的 curated 展示结果不能自动删除。
- 不要默认运行耗时训练；需要训练时先明确轮数、模型、数据集和输出用途。
- `scripts/export_showcase_artifacts.py` 只能汇总已有结果文件并生成展示 artifact，不应修改训练主链路、TopK 字段、CSV/JSON 既有字段或 API 协议。
- 后续实验如需真实推荐展示数据，优先通过 launcher 的 `training_params.save_recommended_topk: true` 或兼容别名启用测试集 TopK 导出；导出目录应落在本次结果目录下的 `recommend_topk/`。不要改 legacy TopK 宽表字段；联邦逐用户导出必须通过文件名 user 范围、微秒时间戳、counter 或等价机制避免覆盖，并保留 `recommend_topk_manifest.json` 兼容目录聚合。
- `privacy_eval.run_membership_probe_from_recommendations` 只能基于已有推荐文件中的真实 membership label 与 score/rank 生成 summary；legacy TopK 宽表必须配合 `membership_labels.json` 才能用 rank proxy score，输入不足时输出 `not_available`，不要编造成员推断结果。
- `secure_aggregation_sim` 只能表述为 simulation-only 成对 mask 抵消摘要，不是生产级密码学安全聚合协议。
- `targeted_poisoning` / `preference_poisoning` 只能表述为 update-space proxy，不是完整 target-item/backdoor 投毒链路。
- `privacy_eval.run_gradient_leakage_demo` 只能生成 synthetic tensor / image-like tensor 风险摘要，不代表 FedVLR 原始图像恢复能力。

## 指标口径

- 比赛展示优先使用 `Recall@50`、`NDCG@50`。
- 历史和对比摘要优先使用 tail mean 口径。
- `best_summary` 可作为辅助参考，但不要把主要展示口径回退成单轮最大值。

## 验证建议

修改 Python 代码后优先运行：

```powershell
python -m compileall -q scripts utils models common attacks defenses privacy_eval
```

涉及统一配置、能力矩阵或启动链路时，优先使用 validate-only 或 dry-run。不要默认运行耗时训练。

## 后端安全能力边界

- `targeted_poisoning` / `preference_poisoning` 只能表述为 update-space proxy 或最小实现；即使配置了 `target_item_ids`，也不保证完整 target-item/backdoor 投毒。
- `membership_inference_probe` 只基于真实 member/non-member label 与 score/rank；legacy TopK 宽表没有 `membership_labels.json` 时必须输出 `not_available`。
- `preference_inference_probe` 只基于推荐列表和 item metadata；缺少 metadata 时只能使用 item_id group proxy，不能伪造语义偏好。
- `gradient_leakage_probe` 与 `run_gradient_inversion_toy.py` 是 demo/probe；toy gradient inversion 只能说明梯度泄露风险，不代表完整 DLG/InvertingGrad 或 FedVLR 原始图像恢复。
- `dp_noise` 是聚合前裁剪加 Gaussian noise 的 central DP-style update noise；当前没有 Opacus PrivacyEngine 和 formal accountant。
- `privacy_eval/opacus_feasibility_check.py` 只是 future-work 可行性检测，不应接入主训练循环。
- `secure_aggregation_sim` 是 simulation-only 成对 mask 抵消摘要；真实 secure aggregation 与 Krum/median/update_filter 这类需要逐客户端更新的鲁棒过滤应作为不同运行模式描述。
- 新增或修改安全能力后，至少运行 compileall、相关 JSON 解析、模块 synthetic smoke，并汇报是否改动训练主链路、API/前端、依赖和 `git status`。

## Realized Security Sidecars

- `scripts/convert_amazon2023_to_fedvlr.py` is an offline converter for locally downloaded Amazon Reviews 2023 category files. It must not auto-download Amazon data or images; generated `image_features.npy` is a URL-hash placeholder, not visual embeddings. Ordinary converted `datasets/AMAZON_BEAUTY_POC/` files are ignored and should not be committed.
- `configs/experiment_smoke/amazon_beauty_poc_baseline_smoke.json` is a 1-epoch FedAvg smoke for verifying the converted Amazon Beauty PoC dataset and TopK export; do not treat it as an effect-quality experiment.
- `privacy_eval/generate_membership_labels.py` may generate `membership_labels.json` from real train/test splits. For KU, `split_label=0` is member and `split_label=2` is non-member.
- `privacy_eval.run_membership_probe_from_recommendations` must only report available membership inference when labels plus score/rank evidence exist. Legacy TopK rank scores are proxy-only and must keep `score_source=rank_based_proxy`.
- `privacy_eval/recommendation_manipulation_metrics.py` reports TopK overlap/Jaccard/list-shift and optional target exposure. It must accept both single TopK files and `recommend_topk/` directories, aggregate all readable users, and keep injected-user subset metrics honest when a `target_interaction_plan.json` is provided. Without `target_items.json`, target-hit fields must remain unavailable/null rather than inferred.
- `privacy_eval/update_leakage_risk_probe.py` can run as a default-off privacy metric over real `participant_params`; it must not persist raw updates and must not be described as image reconstruction.
- `scripts/build_security_sidecars.py` writes ignored output under `outputs/security_sidecars/<dataset>/`; do not commit ordinary generated sidecars unless explicitly requested.
- `scripts/export_showcase_artifacts.py` may join recommendation rows with local `datasets/<dataset>/item_metadata.json` to fill `title`, `category`, and `image_url`. This must remain a local metadata join only: do not fetch images, do not change TopK fields, and keep missing metadata as null rather than fabricating values. If no defense directory is provided for baseline/attack artifacts, defense recommendations and recovery fields must remain unavailable/null. If `target_rank_summary.json` exists, surface it as diagnostic target rank/score context and do not treat it as proof of successful target manipulation by itself.
- `privacy_eval/export_membership_pair_scores.py` can build `membership_pair_scores.csv` from real score files, supported FedAvg/FedRAP-style saved parameter checkpoints, or TopK rank evidence. Rank-only evidence is a proxy and must keep `score_source=rank_proxy`; unsupported checkpoints must emit a feasibility summary instead of guessed scores.
- `attacks/target_interaction_injection.py` is planner-only by default. When `target_interaction_injection.enabled=true`, it may inject target positive interactions only into malicious clients' in-memory local dataloaders for the current run; it must not rewrite dataset files or be described as a full target-item/backdoor attack.
- `privacy_eval/interaction_reconstruction_probe.py` estimates candidate item ids from item-like embedding updates and can compute `hit_at_k` against available train interactions. It is not complete interaction-history recovery, not image reconstruction, and must not save raw participant updates.
- `save_experiment_json_outputs` may emit security sidecars such as `membership_inference_summary.json`, `update_leakage_risk_summary.json`, `interaction_reconstruction_summary.json`, and `target_interaction_plan.json` into a result directory. Federated test evaluation may additionally emit `target_rank_summary.json` with unmasked and masked target scores/ranks when `target_item_ids` are configured; this must remain diagnostic and must not change Recall/NDCG. Ordinary generated result files under `outputs` remain non-commit artifacts unless explicitly requested.

## V2 Security Observation Boundaries

- Score-based MIA and rank-based MIA must be separated clearly. Prefer `checkpoint_score` when a supported FedAvg/FedRAP-style checkpoint can be scored, then `unmasked_rank` from `target_rank_summary.json`, then `rank_proxy` from TopK. Rank-only results are proxy evidence and must keep `proxy_only=true` or an equivalent warning.
- `privacy_eval/export_membership_pair_scores.py` writes `membership_pair_scores.csv` with `user_id,item_id,label,score,rank,score_source,available` plus `membership_score_summary.json`. Unsupported checkpoints must produce feasibility output, not guessed scores.
- `privacy_eval/interaction_reconstruction_probe.py` is now per-client candidate reconstruction with `hit_at_10`, `hit_at_20`, `hit_at_50`, `per_client_candidates`, `modality_breakdown`, and `highest_risk_modality` when evidence exists. It remains candidate reconstruction only, not full user-history recovery.
- Target promotion should be judged by `target_entered_top50`, `target_rank_gain` / positive rank shift toward rank 1, and `target_exposure_gain`. Do not present score/rank movement alone as successful recommendation manipulation.
- `attacks/target_interaction_injection.py` can use `target_user_strategy`, `repeat_target_interactions`, `rating_value`, `only_non_train_positive_targets`, and `max_injections_per_client`. It must remain default-off and in-memory-only for malicious clients.
- `target_promotion_loss` is currently feasibility-only unless a model-specific target-score local loss hook is implemented and validated.
- `dp_noise` is central DP-style clipping plus Gaussian update noise with no formal accountant. `privacy_eval/run_opacus_toy_demo.py` is only a standalone formal-DP route demo and is not FedVLR full-chain DP.
- `privacy_eval/run_secure_aggregation_demo.py` is a synthetic mask-cancellation demo, not a production cryptographic protocol.
- `privacy_eval/select_target_items_for_promotion.py` may select near-TopK candidates from baseline rank summaries, but selected targets are only smoke-test candidates.
- `privacy_eval/build_defense_matrix_summary.py` can summarize smoke config/result cases for exporter consumption. Config-only matrix cases do not imply defense effectiveness.
- `scripts/export_showcase_artifacts.py` should keep reading new optional sidecars (`membership_score_summary.json`, `target_items_promotion.json`, `target_rank_comparison.json`, `opacus_toy_summary.json`, `secure_aggregation_demo_summary.json`, `defense_matrix_summary.json`) with warnings on missing files and no fabricated success.
- `datasets/AMAZON_BEAUTY_POC/image_features.npy` is a URL-hash placeholder, not real visual embedding data.

## V2.5 Amazon Image Cache And Security Notes

- `scripts/cache_amazon_item_images.py` may download up to the requested bounded limit of local display-only product images into `datasets/AMAZON_BEAUTY_POC/item_images/`, generate optional thumbnails under `item_images/thumbs/`, and write `item_image_manifest.json`. Prioritize target items, V2.5 recommendation rows, showcase recommendation frequency, then high-interaction items. The directories are ignored by Git and ordinary cached images must not be committed.
- The image cache is for showcase/UI resources only. Do not modify `datasets/AMAZON_BEAUTY_POC/image_features.npy`, and do not describe cached images or URL-hash placeholders as trained visual embeddings.
- `scripts/extract_amazon_image_features.py` writes optional sidecar features only when local dependencies allow it. Missing `torchvision` or model support should produce `feature_extraction_available=false`, not a failure or fabricated feature file.
- Target promotion V2.5 configs must stay bounded: no more than 10 epochs for Amazon smoke runs, in-memory target interaction injection only, and `target_promotion_loss` remains feasibility-only unless a validated model-specific local loss hook is added.
- Membership pair score export should prefer result-dir auto-discovered `checkpoint_score`, then `unmasked_rank`, then TopK `rank_proxy`; unsupported checkpoints must remain explicit feasibility output.

## Security Artifact V3 Panels

- `scripts/export_security_artifact_v3.py` creates frontend-ready V3 panels from existing artifacts only. It must not run training, change the API/frontend, or infer missing attack/defense success.
- The default V3 output path is `outputs/showcase_artifacts/amazon_beauty_poc_security_v3/` and should contain exactly the core panel files: `scenario_profile.json`, `runtime_timeline.json`, `training_curves.json`, `target_manipulation_metrics.json`, `membership_inference_panel.json`, `update_leakage_panel.json`, `aggregation_defense_panel.json`, `privacy_defense_panel.json`, `model_support_panel.json`, and `frontend_summary.json`.
- Keep V3 path fields repository-relative. Do not expose local absolute `D:\...` or `C:\...` paths in frontend-facing artifacts.
- In `target_manipulation_metrics.json`, keep `attack_topk_hit=false` when masked TopK exposure is missing, even if unmasked target rank improved or entered Top50.
- In `membership_inference_panel.json`, preserve `evidence_type` and `proxy_only`; never relabel rank/unmasked-rank evidence as checkpoint score.
- In `update_leakage_panel.json`, describe results as candidate reconstruction only, not full user history.
- In `aggregation_defense_panel.json`, config-only or validate-only cases must use a status such as `configured_only` and must not fabricate `recall_after`, `ndcg_after`, or recovery metrics.
- `scripts/cache_amazon_item_images.py` defaults to a bounded 1000-item cache and records `cached_count`, `thumbnail_count`, `failed_count`, and `source_priority`. It must continue to leave `image_features.npy` unchanged.

## Model Security Capability Matrix

- `scripts/build_model_security_capability_matrix.py` is the source of the backend model-security support matrix. It must write repository-relative evidence paths and conservative statuses: `supported`, `partial`, `unsupported`, `future_adapter`, or `not_tested`.
- `scripts/build_model_security_capability_matrix.py --matrix-version v2` writes `outputs/model_security_capability_matrix/model_security_capability_matrix_v2.json` with per-capability `reason` and `evidence_file`. V2 statuses are `supported`, `partial`, `unsupported`, `adapter_required`, and `not_tested`; do not introduce success-like aliases for missing adapters.
- `privacy_eval/model_security_adapter.py` is the lightweight model-security adapter layer. It may report TopK, participant update capture, item embedding access, pair scoring, target-rank diagnostics, privacy probe, and robust-defense capability, but it must not change the training path or fabricate unavailable hooks.
- `scripts/export_model_security_capability_artifacts.py` may copy the matrix into `outputs/showcase_artifacts/model_security_capability_matrix/` and split supported demos, unsupported reasons, frontend labels, and manifest files for API/frontend consumption. Missing or partial support must remain explicit and must not be promoted to success.
- Treat TopK export, TopK manifest aggregation, target-rank diagnostics, rank/proxy MIA export, update leakage risk summaries, and candidate interaction reconstruction as shared observation infrastructure when the model uses the common federated trainer path.
- Treat target promotion effectiveness, target-promotion loss, checkpoint scoring, and parameter-name-sensitive reconstruction as model-specific. FedAvg Amazon V2.5 target-rank gain is evidence for FedAvg on `AMAZON_BEAUTY_POC`, not for every model.
- FedAvg on `AMAZON_BEAUTY_POC` is the strongest current attack/defense validation base; MMFedRAP on `KU` is the multimodal showcase model. Do not rewrite docs, configs, or artifacts to imply that FedRAP, FedNCF, FCF, MGCN, MMFedAvg, MMFedNCF, MMFCF, MMGCN, `MMMGCN`, or `MultiModalMGCN` has inherited FedAvg target-promotion success without matching evidence.
- Treat FCF/MMFCF as `partial` until security effects are validated. Treat MGCN as `adapter_required` when no trainer exists. Treat MMGCN as `adapter_required` when `torch_geometric` is missing. Treat `MMMGCN` / `MultiModalMGCN` aliases as `adapter_required` unless a real implementation module exists.
- New V2 model-matrix smoke configs belong under `configs/experiment_smoke/model_matrix_v2/`. Validate them with `.\.venv\Scripts\python.exe scripts\launch_experiment.py --config <config> --validate-only`; do not generate fake training configs for models that cannot import or lack a trainer.
