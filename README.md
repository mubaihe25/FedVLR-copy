# FedVLR

`FedVLR` 是当前比赛项目的算法核心仓库，负责多模态推荐模型、联邦训练模拟、攻防实验模块、风险观测、实验配置和结果输出。

本仓库基于 FedVLR / MMFedRAP 相关研究代码继续工程化整理，用于支撑“面向多模态推荐场景的联邦隐私攻防一体化靶场平台”。对外材料中的论文出处、作者归属和引用方式由团队统一确认；不要在工程文档中自动追加外部论文出处。

## 项目定位

在总项目中，`FedVLR` 提供训练和实验后端能力：

- 多模态推荐模型和融合模块；
- 单机模拟多客户端联邦训练；
- 投毒攻击、恶意更新模拟、鲁棒防御和风险观测；
- 统一实验配置、批量实验和结果汇总；
- CSV、`.experiment_result.json`、`.experiment_summary.json` 输出，供 `FedVLR-API` 和 `FedVLR-Frontend` 读取。

## 资格赛阶段工作总结

资格赛阶段的核心目标是完成项目原型搭建、实验链路跑通和初步展示材料准备。当前系统已经围绕 KU 数据集跑通基线实验、攻击实验和攻防对比实验，并能输出训练日志、逐轮指标、实验结果文件和可视化分析结果。

已覆盖的实验能力包括：

- FedAvg、FedNCF、FedRAP 等基础联邦推荐模型；
- MMFedAvg、MMFedNCF、MMFedRAP 等多模态联邦推荐配置；
- 基线训练、投毒攻击、鲁棒防御和攻防对比模式；
- Recall@50、NDCG@50 等推荐性能指标；
- 逐轮 CSV 输出、尾部均值统计、历史实验记录和对比分析。

已接入的攻击机制包括更新缩放、符号翻转、模型替换、恶意客户端比例控制和统一非定向投毒入口。已接入的防御机制包括更新裁剪、异常过滤、截尾均值鲁棒聚合和组合式鲁棒防御配置。

当前系统已经具备基本的“正常基线训练 -> 投毒攻击导致性能下降 -> 鲁棒防御带来一定恢复”的攻防演练链路。

## 目录结构

```text
attacks/       投毒攻击、更新缩放、符号翻转、模型替换、风险探针
common/        通用 trainer、loss、模型抽象
configs/       模型/数据集配置、能力矩阵、统一实验 schema、batch 配置
defenses/      范数裁剪、更新过滤、截尾均值、Median、Krum、central DP-style noise、鲁棒防御入口
models/        推荐模型和多模态推荐模型
privacy_eval/  风险观测、成员推断 probe、梯度泄露 demo probe 和隐私相关指标
scripts/       启动、batch、验证、结果汇总和 showcase artifact 导出脚本
utils/         配置、数据加载、评估、联邦训练和结果输出
outputs/       实验输出目录；普通输出不要提交
```

`outputs/` 对普通实验结果保持忽略。仓库中已有少量 curated 展示结果被 Git 跟踪，可能被 API 或前端 demo 引用，不要自动删除。

## 环境建议

原始代码曾记录 `Python ~= 3.8` 和 `torch~=2.4.0+cu118`。当前项目建议使用与目标 PyTorch 版本兼容的 Python 环境，通常优先选择 Python 3.10 或 3.11，除非团队已经验证了其他版本。

不建议使用 Python 3.14 作为 FedVLR 训练环境，因为 PyTorch 及相关多模态依赖可能尚未稳定支持。

安装基础依赖：

```powershell
pip install -r requirements.txt
```

`torch`、`torchvision`、CUDA 相关包应根据目标机器和 PyTorch 官方兼容矩阵单独安装。

## 快速启动

当前主入口使用 `--model`，不是旧的 `--alias` 参数。

直接运行示例：

```powershell
python main.py --model MMFedRAP --dataset KU --type LocalRun --comment smoke
```

API/前端集成流程优先使用统一 launcher：

```powershell
python scripts/launch_experiment.py --config path\to\experiment.json --validate-only
```

涉及配置映射时先使用 validate-only 或 dry-run。不要在未明确模型、数据集、轮数和输出用途时默认运行耗时训练。

已有实验结果可用轻量导出脚本整理为展示 artifact；该脚本只读取现有 `experiment_result` / `experiment_summary` / round CSV / TopK 文件，不运行训练：

```powershell
python scripts/export_showcase_artifacts.py --result-dir path\to\result --output-dir path\to\artifacts
```

通过 `scripts/launch_experiment.py` 启动的后续实验，可以在 `training_params` 中显式设置 `save_recommended_topk: true`（或兼容别名 `output_topk` / `save_recommend_topk` / `topk_export`），训练结束后会额外做一次测试集 TopK 推荐导出并写入本次结果目录下的 `recommend_topk/`。该导出不改变已有 TopK 宽表字段。

如果已有推荐文件包含真实 membership label 和 score/rank，可用轻量 runner 生成成员推断 probe summary；legacy `top_0 ... top_k` 宽表必须配合 `membership_labels.json` 才能用 rank proxy score 生成结果，没有 label 时会输出 `not_available`，不会伪造隐私攻击数值：

```powershell
python -m privacy_eval.run_membership_probe_from_recommendations --recommendation-dir path\to\recommend_topk --output-json path\to\membership_inference_summary.json
```

梯度泄露演示可以用独立 synthetic runner 生成 `gradient_leakage_summary.json` 结构；该输出只用于风险展示，不代表真实 FedVLR 原图恢复：

```powershell
python -m privacy_eval.run_gradient_leakage_demo --demo-kind image --output-json path\to\gradient_leakage_summary.json
```

## 指标口径

当前比赛展示优先使用：

- `Recall@50`
- `NDCG@50`
- 训练后段 tail mean summary

CSV 输出可包含逐轮记录、`best_summary` 和 `tail_mean_summary`。`best_summary` 可作为辅助检查，但主要展示和对比不应退回到单轮最大值口径。

## 安全能力边界

当前已实现和可展示的安全相关能力包括：

- 投毒攻击和恶意更新模拟；
- 更新缩放、符号翻转、模型替换、统一非定向投毒，以及 experimental targeted/preference poisoning proxy；
- 范数裁剪、更新过滤、截尾均值、逐元素 Median、Krum 风格客户端选择、central DP-style noise 防御，以及 secure_aggregation_sim 成对 mask 抵消模拟；
- 客户端更新范数、score-based membership_inference_probe、gradient_leakage_probe 等风险观测。

当前训练链路没有正式实现差分隐私、同态加密或安全聚合。`dp_noise` 只是聚合前裁剪加 Gaussian noise 的 central DP-style noise defense，没有 formal privacy accountant；相关内容不能写成完整差分隐私能力。
`secure_aggregation_sim` 只是成对 mask 抵消的 simulation-only summary，不是生产级密码学安全聚合协议。`targeted_poisoning` / `preference_poisoning` 只是 update-space proxy，不是完整 target-item/backdoor 投毒。`membership_inference_probe` 只是基于 score/rank/loss 差异的轻量成员推断观测；`gradient_leakage_probe` 是图像/多模态梯度泄露风险 demo，不代表完整 DLG/InvertingGrad 或 FedVLR 原始图像恢复能力。

## 轻量验证

修改 Python 代码后优先运行：

```powershell
python -m compileall -q scripts utils models common attacks defenses privacy_eval
```

修改配置、能力矩阵或 launcher 时，优先使用 validate-only 或 dry-run，再考虑真实训练。

## 后端安全能力分层

当前后端安全能力按三层维护：

- 攻击：`poisoning_attack` 统一非定向投毒入口，包含 update scale / sign flip / model replacement；`targeted_poisoning` 和 `preference_poisoning` 是 update-space proxy，支持 target item id 参数和 summary，但不保证真实 item-level backdoor。
- 隐私攻击 / 观测：`membership_inference_probe` 与推荐 TopK runner 只基于真实 label + score/rank；`preference_inference_probe` 基于推荐列表和 item metadata，缺少 metadata 时只能输出 item_id group proxy；`gradient_leakage_probe` 是梯度风险 probe；`run_gradient_inversion_toy.py` 是 synthetic DLG-style toy demo，不是 FedVLR 原图恢复。
- 防御：鲁棒聚合包含 `trimmed_mean`、`median`、`krum`、`multi_krum`、`bulyan`；`dp_noise` 是 central DP-style update noise，没有 formal privacy accountant；`secure_aggregation_sim` 是 simulation-only mask cancellation summary，不是生产级密码学安全聚合协议。

Opacus 当前只通过 `privacy_eval/opacus_feasibility_check.py` 做可行性检测。正式 DP-SGD 需要 per-sample gradient、PrivacyEngine/optimizer 接入和训练循环改造，不能描述为当前已完成能力。

## Realized Security Sidecars

Recent backend security work adds three real-data-oriented sidecar paths:

- `scripts/convert_amazon2023_to_fedvlr.py` converts locally downloaded Amazon Reviews 2023 category shards, such as All_Beauty, into FedVLR PoC files under `datasets/AMAZON_BEAUTY_POC/`. It does not download Amazon data or images; `image_features.npy` is a deterministic URL-hash placeholder and must not be described as visual embeddings.
- `privacy_eval/generate_membership_labels.py` builds `membership_labels.json` from dataset train/test splits. For KU it reads `datasets/KU/inter.csv`, uses `split_label=0` as members and `split_label=2` as non-members, and can filter labels by an exported TopK file.
- `privacy_eval/run_membership_probe_from_recommendations.py` can consume `membership_labels.json` plus recommendation score/rank fields. Legacy `top_0 ... top_k` files use `score = 1 / (rank + 1)` and must be reported as `rank_based_proxy`.
- `privacy_eval/recommendation_manipulation_metrics.py` compares baseline/attack/defense TopK lists and optionally reads `target_items.json`. Without target items it only reports overlap/Jaccard/list-shift metrics.
- `privacy_eval/update_leakage_risk_probe.py` is a read-only privacy metric over real `participant_params`. It reports update norms, sparsity, energy, diversity, and modality-risk buckets, but it does not save raw updates and is not image reconstruction.
- `scripts/build_security_sidecars.py` writes `membership_labels.json`, `target_items.json`, `item_metadata_stub.json`, and `security_sidecar_manifest.json` under `outputs/security_sidecars/<dataset>/`.

Keep these boundaries explicit: item metadata stubs have no real semantic title/tag; membership inference with rank-only TopK is a proxy; recommendation manipulation without target items is list-change analysis only; update leakage risk is a summary probe, not DLG/InvertingGrad reconstruction.

## Teacher-Guided Security Adapters

This backend now includes three additional real-data-oriented adapters:

- `privacy_eval/export_membership_pair_scores.py` exports `membership_pair_scores.csv` and `membership_score_summary.json` from `membership_labels.json` plus supplied score files, supported FedAvg/FedRAP-style checkpoints, or exported TopK rank evidence. TopK-only evidence is marked as `score_source=rank_proxy`; unsupported checkpoints produce a feasibility summary.
- `attacks/target_interaction_injection.py` builds `malicious_interaction_plan.json` for target item promotion. By default it remains planner-only; with `target_interaction_injection.enabled=true` it injects target positive interactions only into malicious clients' in-memory local dataloaders and never rewrites dataset files.
- `privacy_eval/interaction_reconstruction_probe.py` estimates candidate item ids from item-like embedding updates in `participant_params` and can compute `hit_at_k` against dataset train interactions when available. It is interaction candidate reconstruction only, not full user-history recovery and not DLG/image reconstruction.

`privacy_eval/run_membership_probe_from_recommendations.py` should prefer `membership_pair_scores.csv` when available, then fall back to recommendation score/rank rows. Missing labels or missing score/rank evidence must produce `not_available` instead of fabricated attack results.
