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
