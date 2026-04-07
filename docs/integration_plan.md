# FedVLR 后续集成规划（仅骨架阶段）

## 为什么当前只做骨架

当前阶段的目标不是立即实现完整 attack / defense / privacy_eval，而是先在 `FedVLR` 内部建立一套清晰、稳定、可扩展的工程边界。这样做有几个原因：

1. 当前真实训练主链路已经明确，核心运行路径是 `main.py -> quick_start -> FederatedTrainer -> 具体 Trainer`。在没有完全确认每一种攻击和防御方案之前，先补齐独立目录和统一接口，更有利于后续平滑接入。
2. 未来结果需要通过 API 提供给前端，因此新增输出结构必须稳定、可序列化、字段清晰。先做结果结构草案，可以避免后面边开发边改协议。
3. 当前训练逻辑已经可运行，本阶段不应直接改动 `main.py`、`quick_start.py`、`FederatedTrainer` 和 `models/*.py`，否则容易把“功能探索”和“工程铺底”混在一起。

## 为什么不直接迁移 TensorFlow attack 参考仓

`research/SplitNN_FSHA-attack-Fork-` 当前只是参考仓，不适合直接迁移到 `FedVLR` 中，原因包括：

1. 该参考仓以 TensorFlow 2 为主，而 `FedVLR` 当前训练主仓是 PyTorch 体系，训练范式、模型组织方式、张量对象和调试方式都不一致。
2. 参考仓面向的是分裂学习攻击，不等价于当前 `FedVLR` 的联邦推荐训练骨架。即使攻击思想可参考，实现位置和数据结构也不能直接照搬。
3. 当前更需要的是“插桩接口”和“结构化结果”，而不是直接复制一套外部实现。先把 FedVLR 的扩展骨架搭好，后续才能有序地把参考思想转换成适合本仓库的 PyTorch 实现。

## 为什么优先以 MMFedRAP / FedRAP 为落点

当前仓库里没有 `FedVLRTrainer`，所以后续联邦安全扩展最适合优先落在以下两条主线：

1. `MMFedRAPTrainer`
   - 更接近当前项目“多模态联邦推荐安全实验平台”的语义。
   - 同时覆盖客户端本地训练、融合层更新、共享表示聚合等关键位置。
   - 后续更适合接入模态攻击、融合层防御、聚合阶段鲁棒处理。

2. `FedRAPTrainer`
   - 结构更简洁，便于先验证攻击、防御和隐私评估的工程插入链路。
   - 如果需要先做轻量的联邦推荐安全基线，再过渡到多模态版本，FedRAP 是更稳妥的中间落点。

## 未来准备插入 attack / defense / privacy_eval 的训练位置

### 1. `_train_epoch`

位置：`utils/federated/trainer.py`

适合插入：

- round 开始前的实验状态记录
- 恶意客户端选择
- 客户端上传前后的全局调度逻辑
- round 级攻击 / 防御 / 隐私指标汇总

### 2. `_train_client`

位置：`utils/federated/trainer.py`

适合插入：

- 客户端本地训练阶段的标签翻转、模型投毒等攻击
- 本地梯度裁剪、本地差分隐私噪声等处理
- 客户端训练后输出更新的额外统计

### 3. `_aggregate_params`

位置：各具体 Trainer，例如 `models/fedrap.py`、`models/mmfedrap.py`

适合插入：

- 聚合前异常更新过滤
- 鲁棒聚合
- 安全聚合包装
- 聚合后防御恢复效果记录

### 4. `fit` 每轮结束后的出口

位置：`common/trainer.py`

适合插入：

- 每轮评估后的统一结果记录
- round-level 的 attack / defense / privacy_eval 汇总
- 形成后续 API / 前端可消费的结构化结果

## 当前阶段边界

本轮只补工程骨架，不接训练逻辑，不修改现有训练主链路。后续真正实现 attack / defense / privacy_eval 时，应优先围绕现有联邦训练骨架做增量接入，而不是重写训练流程。

## 本轮新增的最小非侵入式插桩准备

当前新增了一层极简运行时容器：`utils/experiment_hooks.py` 中的 `ExperimentHookManager`。

它当前只承担以下职责：

1. 维护统一的实验上下文，例如：
   - `enabled`
   - `attacks`
   - `defenses`
   - `privacy_metrics`
   - `round_states`
2. 为未来 attack / defense / privacy_eval 预留统一调用入口。
3. 以 `privacy_eval/result_schema.py` 中的结果结构为基础，收集最小的 round 级占位结果。

### 默认开关与行为

- `enable_experiment_hooks`：默认 `False`
- `collect_round_metrics`：默认 `True`
- `enable_malicious_clients`：默认 `False`
- `malicious_client_mode`：默认 `none`
- `malicious_client_ratio`：默认 `0.0`
- `malicious_client_ids`：默认空列表

默认情况下：

- 不执行任何真实攻击逻辑
- 不执行任何真实防御逻辑
- 不执行任何真实隐私评估逻辑
- 不修改现有 loss
- 不修改现有聚合算法
- 不修改现有模型参数更新行为

当前只会在内存中记录最基础的 round 信息，供未来 API / 前端接入时复用。
当前的 `malicious_clients` 也只是占位配置与占位记录，不会改变任何客户端的训练行为。

### 本轮已预留的调用点

#### 1. `FederatedTrainer._train_epoch`

已接入：

- round 开始时初始化 round state
- round 开始时按配置生成本轮 `malicious_clients` 占位列表
- 本地训练完成后预留 `after_local_train`
- 聚合前预留 `before_aggregation`
- round 结束时记录最基础的 round loss 和 participant 数量

#### 2. `FederatedTrainer._train_client`

已接入：

- 记录单个客户端本地训练的 loss 轨迹占位信息

#### 3. `Trainer.fit` 每轮结束出口

已接入：

- 每轮训练结束后的 `valid_result / test_result / stop_flag` 占位记录
- 训练结束后生成与 `ExperimentResult` 兼容的顶层结果结构

### 未来如何接 attack / defense / privacy_eval

后续如果要真正实现扩展，可沿用当前 manager：

- attack：
  在 `after_local_train` 或 `before_aggregation` 中接入客户端更新篡改、恶意客户端策略等逻辑。当前生成的 `malicious_clients` 列表可直接作为攻击目标集合。
- defense：
  在 `before_aggregation` 或具体 Trainer 的 `_aggregate_params` 周围接入异常检测、鲁棒聚合、安全聚合等逻辑。
- privacy_eval：
  在 round 结束和 `fit` 末尾汇总输出隐私风险评分、隐私预算和 round-level 指标。

也就是说，本轮不是在训练链路里硬编码安全逻辑，而是先把“可插入的位置”和“可输出的结构”准备好，并确保默认情况下训练行为不变。
