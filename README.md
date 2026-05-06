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
defenses/      范数裁剪、更新过滤、截尾均值、鲁棒防御入口
models/        推荐模型和多模态推荐模型
privacy_eval/  风险观测和隐私相关指标
scripts/       启动、batch、验证和汇总脚本
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

## 指标口径

当前比赛展示优先使用：

- `Recall@50`
- `NDCG@50`
- 训练后段 tail mean summary

CSV 输出可包含逐轮记录、`best_summary` 和 `tail_mean_summary`。`best_summary` 可作为辅助检查，但主要展示和对比不应退回到单轮最大值口径。

## 安全能力边界

当前已实现和可展示的安全相关能力包括：

- 投毒攻击和恶意更新模拟；
- 更新缩放、符号翻转、模型替换和统一非定向投毒；
- 范数裁剪、更新过滤、截尾均值和鲁棒防御入口；
- 客户端更新范数、只读泄露探针等风险观测。

当前训练链路没有正式实现差分隐私、同态加密或安全聚合。相关内容只能写成后续扩展方向、规划能力或未生效占位。

## 轻量验证

修改 Python 代码后优先运行：

```powershell
python -m compileall -q scripts utils models common attacks defenses privacy_eval
```

修改配置、能力矩阵或 launcher 时，优先使用 validate-only 或 dry-run，再考虑真实训练。
