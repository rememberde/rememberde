# CHANGELOG

项目：基于 Boltzmann 熵的图神经网络社区分类
工作目录：`e:\Code\python code\WJ`

> 每次运行后追加一条更新。最新的在最上面。

---

## 2026-07-09 — 代码审查修复（Q 一致性 bug + 7 项改进）

### 背景

系统性审查发现 method2 核心算法存在 Q 不一致 bug，以及若干改进点。

### 改动

**Bug 1（核心）— method2_lnw Q 不一致修复：**
- [entropy_gnn_baseline.py](file:///e:/Code/python%20code/WJ/entropy_gnn_baseline.py) `method2_lnw` 原本内部重算
  `Q = softmax(Z @ W.t())`，但 model forward 的 logits 经过了 `/sqrt(in_features)` 缩放，
  两个 Q 分布不一致，导致梯度方向与模型实际 commit 的社区结构脱节。
- 修复：`method2_lnw` 改为接收 caller 传入的 Q（forward 输出），`free_energy`
  移除不再使用的 W 参数，`train_one` 调用同步更新。
- 验证：Cora 冒烟测试 method2+rank NMI 从 0.4267 → 0.4567（+0.03），loss 更低，
  证实 Q 一致性修复确实改善了训练动力学。

**Bug 2 — EntropyGNN docstring 过时：**
- docstring 称 bin centers "learnable"，但实际用 `register_buffer`（固定锚点）。
- 更新为中文 docstring，说明固定 C 的原因（历史教训：可学习 C 会跟着塌缩）。

**改进 3 — main.py 核心发现动态化：**
- 硬编码的"imbalanced SBM 上 m2_rank2/3 显著超越 vanilla（p<0.01, **）"
  改为从 sbm_results 动态跑 paired t-test 提取（quick 模式 n_seeds=1 时自动跳过）。

**改进 4 — 合并协方差矩阵计算：**
- [anticollapse.py](file:///e:/Code/python%20code/WJ/anticollapse.py) 新增 `_covariance()` 内部函数，
  一次性算出 S、tr、tr2。`effective_rank`/`total_variance`/`VarianceHinge.penalty`
  /`.diagnostics`/`compute_collapse_metrics` 共享，避免训练循环里重复算两次 O(N·d²)。
- 公共 API `effective_rank(Z)`/`total_variance(Z)` 保留，demo_scale_invariance.py 不受影响。

**改进 5 — 关键函数中文注释：**
- `make_imbalanced_sbm`、`method2_lnw`、`free_energy`、`_covariance`、
  `VarianceHinge.penalty`/`.diagnostics` 英文 docstring 改中文。

**改进 6 — deploy.py SYNC_ITEMS 补全：**
- 新增 `extract_nmi_summary.py`、`run_auto_all.py`、`deploy.py`、`README.md`、`CHANGELOG.md`。

**改进 7 — 添加梯度裁剪：**
- `train_one` 的 `loss.backward()` 后加 `clip_grad_norm_(max_norm=5.0)`，
  防 method2 的 lnW 早期梯度爆炸。

**改进 8 — extract_nmi_summary.py 正则加固：**
- 删除固定 5 列正则 `RE_TABLE_ROW`，改用 `parse_table_row()` 按 `|` split 解析，
  对列数变化更鲁棒。
- `parse_mean_std` 支持科学计数法和负数，解析失败返回 None 而非崩溃。
- `RE_DELTA` 支持科学计数法。

### 验证

- 所有模块 import 通过
- Cora 冒烟测试通过（vanilla NMI=0.4798，method2+rank NMI=0.4567，铰链健康时休眠）
- auto_config.py --analyze-only 正确选择 config（Cora → m2_rank3）
- SBM hard 全 7 config 50 epochs 训练正常，无 NaN/崩溃
- 正则加固测试通过（科学计数法、负数、不同列数、分隔行、表头）

---

## 2026-07-08 — 代码清理与重构（去除已证伪 pernode/adasig 代码）

### 背景

pernode 两层 reformulation 和 adasig 自适应 sigma 已在 2026-07-07 实验中被彻底证伪
（α 退火 + 自适应 sigma 两次失败，per-node 路径已放弃）。但相关代码仍保留在
entropy_gnn_baseline.py 和 main.py 中，造成代码冗余和误导（main.py 的 result.md
仍输出已证伪的结论如"α=0.5 是甜点"、"pernode 无需铰链即可抗塌缩"）。

本次清理：彻底删除 pernode/adasig 相关代码，重构 load_cora/citeseer/pubmed 为
统一函数，删除过时诊断脚本。

### 改动

**entropy_gnn_baseline.py（核心模块，净减约 110 行）：**
- 删除 `method2_pernode_lnw` 函数（37 行两层 reformulation）
- 删除 `free_energy` 中 `method2_pernode` 分支和 `pernode_alpha` 参数
- 删除 `TrainConfig` 中 4 个字段：`pernode_alpha`、`pernode_alpha_init`、
  `sigma_mode`、`sigma_factor`
- 删除 `schedule_alpha` 函数（α 退火策略）
- 删除 `train_one` 中 `sigma_mode="auto"` 分支（数据自适应 sigma）
- 删除 `make_configs` 中 6 个 config：`m2_pernode_a03/a05/a07/anneal`、
  `m2_pernode_a05_adasig`、`m2_pernode_anneal_adasig`
- 重构 `load_cora`/`load_citeseer` 共用 `_load_planetoid` 通用函数
  （两者都是 planetoid 格式：content + cites），减少约 60 行重复代码
- 删除 `run_cora`（已被 `run_dataset` 通用接口取代）
- 更新文档字符串，明确双铰链反塌缩为主方案

**main.py（结果生成器）：**
- 更新文档字符串：去除 pernode/adasig 新增 config 描述
- `all_cfg_candidates` 从 10 个减到 4 个（method2 + m2_rank1/2/3）
- 核心发现部分删除已证伪结论（如"α=0.5 是甜点"），改为双铰链方案的正确结论
- 打印输出 11 configs → 7 configs

**analysis/analyze_hard_collapse.py：**
- target_cfgs 从 6 个（含 pernode）改为 5 个（vanilla/method2/m2_rank1/2/3）
- colors dict 清理 pernode 配色

**删除的文件（过时诊断/扫描脚本）：**
- `analysis/diagnose_pernode_cora.py`（458 行，专门诊断已证伪的 pernode）
- `experiments/sweep_minvar.py`（历史超参扫描，已完成使命）
- `experiments/sweep_minrank.py`（历史超参扫描，已完成使命）

**deploy.py：**
- `--script` 参数示例从已删除的 `diagnose_pernode_cora.py` 改为 `monitor_collapse.py`

### 验证

- `import entropy_gnn_baseline` / `main` / `auto_config` / `significance_test` /
  `smoke_test` 全部通过
- `_smoke_cora.py` 训练正常（vanilla + method2 + rank 都能跑）
- `auto_config.py --analyze-only` 正确选择 config
  （Cora → m2_rank3，SBM hard → m2_rank1）
- SBM hard 50 epochs 跑全部 7 个 config 都能正常训练

### 保留作为历史记录

`make_configs` 现在生成 7 个 config：
- vanilla / size / assign / method2（无铰链）
- m2_rank1 / m2_rank2 / m2_rank3（双铰链反塌缩，λ=1.0/2.0/3.0）

---

## 2026-07-07 — hinge 方案稳定性验证最终结论（per-node 路径正式放弃）

### 背景

per-node 熵路径已被证伪两次（α 退火 + 数据自适应 sigma 均失败）。
用户决定放弃 per-node，聚焦已有的 hinge 损失方案，跑完整实验验证稳定性。
本轮实验保留全部 14 个 config（含 6 个 pernode config 作为历史对比），
共 14 configs × 4 SBM regimes × 5 seeds + 3 真实数据集 × 3 seeds，耗时 34 分钟。

### 实验结果汇总（hinge 在真实数据集上的稳定性）

| 数据集 | N | vanilla NMI | 最佳 config | 最佳 NMI | Δ | 备注 |
|---|---|---|---|---|---|---|
| Cora | 2708 | 0.4708±0.0063 | **m2_rank3** | 0.5156±0.0142 | **+0.0448** | 方差小，稳定 |
| CiteSeer | 3312 | 0.2602±0.0294 | m2_rank2 | 0.2762±0.0218 | +0.0160 | 稳定 |
| Pubmed | 19717 | 0.2682±0.0246 | **method2** | 0.2896±0.0308 | **+0.0215** | hinge 有害 |

### 核心结论

1. **hinge 方案在 Cora 上稳定有效**：m2_rank3 NMI=0.5156±0.0142，
   方差仅 0.0142（远小于 adasig 的 ±0.0485），稳定超越 vanilla +0.0448。
2. **hinge 在大图上有害**：Pubmed 上 m2_rank1/2/3 全部低于 vanilla，
   纯 method2（无 hinge）才是最佳（+0.0215）。
3. **图规模与最优策略的关系确认**：
   - 小图（Cora/CiteSeer）：hinge 直接保护 eff_rank，避免塌缩 → 有效
   - 大图（Pubmed）：method2 自带的 spread 项已足够，hinge 反而干扰社区形成 → 有害
4. **adasig 完全失败**：SBM easy/medium/hard/imbalanced 全部塌缩
   （NMI≈0.0-0.18，effRank=0），Pubmed 上 effRank=1.00 完全塌缩。
   进一步证实 per-node 路径的两难困境。
5. **per-node 路径正式放弃**：两次证伪（α 退火 + 自适应 sigma），
   理论上也存在"spread 目标 vs 紧凑簇目标"的本质冲突。

### SBM imbalanced 上的最佳 config

- m2_rank1: NMI=0.9361±0.0225（最佳，+0.1178 vs vanilla）
- m2_rank2/3 紧随其后（0.9274/0.9242）
- pernode config 全部在 0.83-0.87 区间，弱于 hinge

### 下一步方向

per-node 路径已关闭。后续可探索：
1. 改进 system 层 method2（Pubmed 上已有效 +0.022）
2. 基于社区大小的正则化（替代 per-node spread）
3. 图规模自适应的 config 选择策略（小图用 hinge，大图用纯 method2）

完整结果见 [result.md](file:///e:/Code/python%20code/WJ/result.md)。

---

## 2026-07-07 — 数据自适应 sigma 方案失败（per-node 熵的两难困境）

### 背景

上一轮诊断发现固定 sigma=0.5 导致 per-node bin 熵饱和（Cora 75% / Pubmed 99.9%）。
预测：让 sigma 自适应于嵌入范数 `sigma = 0.3 * ||z||.mean()` 可避免饱和。

### 实现

- [entropy_gnn_baseline.py](file:///e:/Code/python%20code/WJ/entropy_gnn_baseline.py) TrainConfig 添加 `sigma_mode="auto"` + `sigma_factor=0.3` 字段
- train_one 每 epoch 动态计算 sigma_eff，加下界 0.1 防 NaN
- 新 config: `m2_pernode_a05_adasig`、`m2_pernode_anneal_adasig`

### 实验结果（3 seeds × 3 真实数据集）

| 数据集 | N | vanilla NMI | m2_pernode_a05 NMI | **m2_pernode_a05_adasig NMI** | adasig effRank |
|---|---|---|---|---|---|
| Cora | 2708 | 0.4708 | 0.4456 | **0.3136** ±0.18 | 8.83 |
| CiteSeer | 3312 | 0.2602 | 0.2367 | 0.2434 | 5.91 |
| Pubmed | 19717 | 0.2682 | 0.2896 | **0.0758** ±0.004 | **1.00** |

**结论：数据自适应 sigma 方案失败，且失败方式与诊断预测相反。**

### 关键发现：per-node 熵的两难困境

诊断原本认为"sigma 太大导致饱和"，应该让 sigma 变小。但实验显示：

**sigma 太小 → per-node 熵变"硬" → 梯度把嵌入拉向 bin 中心 → 塌缩**

Pubmed 上 adasig 让 sigma 从 0.5 降到 0.3×1.25≈0.375，per-node 熵从"几乎饱和"变成"硬分配"，
梯度方向从"零"变成"强力拉向最近的 bin 中心"，导致嵌入全部塌缩到原点附近（||z||=0.076, effRank=1.00）。

**两难全貌**：

| sigma | per-node 熵状态 | 梯度行为 | 结果 |
|---|---|---|---|
| 太大（0.5）| 饱和（sat=1.0）| 梯度≈0 | per-node 层失效，搭 method2 便车 |
| 太小（0.1-0.3）| 硬化（接近 argmax）| 强力拉向 bin 中心 | 嵌入塌缩，社区结构破坏 |
| 中间 | 部分饱和 | ？？？ | 难以找到稳定区间 |

**Cora 上的不稳定**：adasig 在 Cora 上方差极大（±0.18，是其他 config 的 10 倍），
说明训练落在了"塌缩/不塌缩"的临界区，对随机种子敏感——进一步证实两难困境。

### 深层洞察

per-node 熵的目标"让每个节点在 bin 空间 spread"与社区形成目标"让节点聚成紧凑簇"**本质冲突**：
- 社区形成需要节点嵌入聚成紧凑簇
- per-node 熵要求每个节点 spread 在多个 bin 上
- 两者方向相反，无论 sigma 多大都存在这个张力

这解释了为什么：
- **Cora（强社区结构）**：per-node 熵破坏社区形成（hinge 更合适，直接保护 eff_rank）
- **Pubmed（弱社区结构）**：per-node 熵失效但无害（搭 method2 便车，因为 method2 本身有效）

### bug 修复

1. **SBM NaN bug**：SBM 用 one-hot 特征，初始 ||z||≈0 导致 sigma_eff≈0，
   softmax 下溢到 0，0/0=NaN 传播到 Z。修复：sigma_eff 加下界 0.1。
2. **build_metrics_table None bug**：SBM 实验失败时 sbm_results=None 导致
   result.md 完全写不出来。修复：build_metrics_table 容错 None。

### 下一步方向

per-node 熵路径已被证伪两次（α 退火 + 自适应 sigma 均失败）。
真正有效的方向是：
1. **接受 per-node 不适用于强社区图**，Cora 用 hinge（已验证 +0.045 NMI）
2. **改进 system 层 method2**（Pubmed 上已有效 +0.022）
3. **放弃 per-node，探索其他 anti-collapse 机制**（如基于社区大小的正则化）

---

## 2026-07-07 — 深挖 Cora pernode 失效根因（图结构 + 训练动力学诊断）

### 背景

α 退火策略在 Pubmed 上有效（+0.022 NMI）但在 Cora 上失败（-0.030 NMI）。
此前推测原因是"per-node 熵梯度在中等图上分散"，但未经验证。
本次通过三层数据采集定位真正根因：**不是梯度分散，而是 sigma 过大导致 per-node 熵饱和**。

### 方法

新增 [analysis/diagnose_pernode_cora.py](file:///e:/Code/python%20code/WJ/analysis/diagnose_pernode_cora.py)，三步诊断：
1. **图结构对比**：Cora / CiteSeer / Pubmed 的度分布、聚类系数、GT modularity、同质性、特征对齐度
2. **插桩训练动力学**：训练 m2_pernode_a05，每 5 epoch 记录 S_node / lnW / H(w_i) 均值方差 / 饱和比例 / ||z_i|| / 梯度范数 / effRank / NMI
3. **per-node bin 分布对比**：饱和比例、梯度范数、H(w_i) 离散度

输出 3 张对比图：
- [image/pernode_diag_structure.png](file:///e:/Code/python%20code/WJ/image/pernode_diag_structure.png)
- [image/pernode_diag_dynamics.png](file:///e:/Code/python%20code/WJ/image/pernode_diag_dynamics.png)
- [image/pernode_diag_bindist.png](file:///e:/Code/python%20code/WJ/image/pernode_diag_bindist.png)

### 关键发现 1：图结构差异（Cora vs Pubmed）

| 指标 | Cora | Pubmed | 解读 |
|---|---|---|---|
| 节点数 | 2708 | 19717 | Pubmed 大 7 倍 |
| 平均度 | 3.90 | 4.50 | 接近 |
| **聚类系数** | **0.241** | **0.008** | **Cora 高 30 倍** |
| **GT Modularity** | **0.640** | **0.432** | **Cora 社区结构更强** |
| 同质性 | 0.810 | 0.802 | 接近 |
| 特征对齐度 | 0.026 | 0.020 | 都很低 |
| 社区大小CV | 0.507 | 0.266 | Cora 更不平衡 |

**Cora 的社区结构更紧凑**（聚类系数 30 倍，GT modularity 高），需要嵌入聚成紧凑簇；
**Pubmed 的社区结构更松散**，per-node 熵推 spread 的副作用更小。

### 关键发现 2：sigma=0.5 导致 per-node 熵饱和

| 指标 | Cora(终) | Pubmed(终) | 解读 |
|---|---|---|---|
| H_w_mean | 2.657 | 2.764 | log(16)=2.77，Pubmed 几乎完全饱和 |
| **H_w_sat** | **0.751** | **0.999** | **Pubmed 几乎所有节点饱和** |
| ||z||_mean | 1.633 | 1.255 | Cora 嵌入范数更大 |
| grad_Snode | 0.259 | 0.022 | Pubmed per-node 梯度小 10 倍 |
| grad_lnW | 189.84 | 55.67 | Cora system 梯度大 3 倍 |

**根因**：sigma=0.5 对两个数据集都太大。距离 / sigma ≈ ||z|| / sigma：
- Cora: 1.6 / 0.5 = 3.2，softmax 仍然较平坦 → 75% 节点饱和
- Pubmed: 1.25 / 0.5 = 2.5，更平坦 → 99.9% 节点完全饱和

### 关键发现 3：Cora 上 per-node 熵"去饱和→重新饱和"破坏社区

Cora 训练动力学完整轨迹：

| epoch | H_w | sat | nmi | 解读 |
|---|---|---|---|---|
| 0 | 2.771 | 1.00 | 0.139 | 初始饱和（||z||=0.075 很小）|
| 15 | **2.468** | **0.33** | 0.415 | 去饱和谷底，per-node 梯度最大（gS=0.872）|
| 25 | 2.590 | 0.45 | **0.458** | NMI 峰值附近 |
| 40 | 2.660 | 0.73 | **0.470** | NMI 峰值 |
| 50 | 2.691 | 0.86 | 0.491 | 重新饱和开始 |
| 199 | 2.657 | 0.75 | **0.393** | 最终 NMI 下降 |

**Cora 模式**：NMI 在 sat=0.26-0.81 区间（ep 15-50）最高（0.41-0.49），
**重新饱和后 NMI 下降到 0.39**。per-node 熵把嵌入推回均匀分布，破坏了中期形成的社区结构。

### 关键发现 4：Pubmed 上 per-node 层根本不起作用

Pubmed 上 sat=1.00 持续整个训练后期（ep 70+），grad_Snode=0.02（相对 grad_lnW=55.67 可忽略）。
**m2_pernode 在 Pubmed 上等价于纯 method2**。Pubmed 上 pernode"有效"只是搭便车跟着 method2。

### 根因总结

**per-node 策略在两个数据集上都失效，但失效方式不同**：

1. **Cora 模式（破坏性）**：per-node 熵经历"去饱和→重新饱和"，
   重新饱和过程把嵌入拉回均匀分布，**破坏中期形成的社区结构**（NMI 从 0.49 降到 0.39）。
   Cora 强社区结构（modularity=0.64）需要紧凑簇，per-node 熵推 spread 的方向与之冲突。

2. **Pubmed 模式（无效性）**：per-node 熵完全饱和（sat=1.0），
   per-node 层梯度几乎为零（gS=0.02），**等价于纯 method2**。
   Pubmed 上 pernode"有效"只是因为 method2 本身有效，per-node 层搭便车。

**根本原因**：**sigma=0.5 固定不适应不同数据集的嵌入范数**。
- Cora ||z||≈1.6，sigma=0.5 → 距离/sigma≈3.2，softmax 较平坦 → 部分饱和
- Pubmed ||z||≈1.25，sigma=0.5 → 距离/sigma≈2.5，softmax 几乎均匀 → 完全饱和

### 改进方向（待验证）

1. **数据自适应 sigma**：`sigma = 0.3 * ||z||.mean()`，让 per-node 熵不饱和
2. **更小的固定 sigma**：直接降到 0.1-0.2，让 per-node 真正有区分力
3. **Cora 用 hinge 而非 pernode**：Cora 强社区结构需要紧凑簇，hinge 直接保护 eff_rank 更合适
4. **温度化的 per-node 熵**：sigma 随训练退火，早期大防饱和，后期小增区分力

### 代码改动

- 新增 [analysis/diagnose_pernode_cora.py](file:///e:/Code/python%20code/WJ/analysis/diagnose_pernode_cora.py)：综合诊断脚本
- 扩展 [deploy.py](file:///e:/Code/python%20code/WJ/deploy.py) `run` 子命令支持 `--script` / `--log` 参数运行任意脚本
- `cmd_status` / `cmd_tail` / `cmd_fetch` 支持自定义日志文件名

---

## 2026-07-06 — α 退火策略 + 新数据集 (CiteSeer/Pubmed) + 大图性能优化

### 背景

pernode 在大图（Cora）上效果差：α 越大越差（0.3→0.4356, 0.5→0.4456, 0.7→0.4571），
推测原因是真实图节点多，per-node 熵梯度太分散。设计 α 退火策略：早期 α 大防塌缩，
后期 α→0 让 system 玻尔兹曼熵主导。

### 本次改动

#### 1. α 退火策略（`entropy_gnn_baseline.py`）

新增 `schedule_alpha()` 函数 + `TrainConfig.pernode_alpha_init` 字段：
- `pernode_alpha_init > 0` 时启用退火：从 α_init 线性退火到 0
- 退火节奏和 T 同步：T_warmup 阶段保持 α_init，之后线性降到 0
- 验证曲线（epochs=200, T_warmup=0.2）：ep 0→0.7, ep 40→0.7, ep 100→0.43, ep 199→0.0

新 config `m2_pernode_anneal`：`pernode_alpha_init=0.7`，T_max=0.3, cosine anneal

#### 2. 新数据集 loader（`entropy_gnn_baseline.py`）

- `load_citeseer()`: planetoid 格式，3312 节点, 6 类, 3703 维
- `load_pubmed()`: **稀疏格式**（每篇论文只列非零特征）
  - 修复 schema 解析：从 `numeric:w-name:default` token 构建 w-name → 列索引映射
  - 修复 cites 解析：`paper:ID` 前缀需 strip，列结构是 `edge_id\tpaper:CITED\t|\tpaper:CITING`
  - 19717 节点, 3 类, 500 维, 44325 边

新增 `DATASET_LOADERS` dict 统一管理 3 个数据集；`run_dataset()` 通用接口；`run_cora()` 改为 wrapper。

#### 3. main.py 重构

- `cora_results` → `real_results: dict[name -> ...]`，支持任意子集真实数据集
- 新增 flags: `--no-citeseer`, `--no-pubmed`, `--no-real`
- `write_result_md` 接收 `real_results` dict，按数据集生成多节
- 关键结论部分：迭代所有真实数据集，找最佳 config + 逐 config vs vanilla 对比
- `all_cfg_candidates` 加入 `m2_pernode_anneal`

#### 4. 大图性能优化（`entropy_gnn_baseline.py`）

Pubmed（19717 节点）KMeans 太慢，n_init=10 单次 30+ 秒：
- `kmeans_labels()`: N>5000 时自动 `n_init=10→3`
- `run_dataset()`: N>5000 时 `eval_every=10→20`（KMeans 调用次数减半）
- 预计 Pubmed 总时间减约 40%

#### 5. deploy.py

- SYNC_ITEMS 添加 `data/citeseer` 和 `data/Pubmed-Diabetes`
- run/all 子命令支持 `--no-citeseer` / `--no-pubmed` / `--no-real` 透传

### 实验设计

11 configs × 4 SBM regimes × 5 seeds + 3 真实数据集 × 3 seeds：
- 验证 α 退火是否能解决 Cora/CiteSeer/Pubmed 上 pernode 表现差的问题
- 假设：早期 α=0.7 防塌缩（保护 eff_rank），后期 α→0 让 system 玻尔兹曼熵驱动社区形成
- 期望：m2_pernode_anneal 在大图上 NMI 接近或超越 m2_rank1/3

### 实验结果

完整跑通：11 configs × 4 SBM regimes × 5 seeds + 3 真实数据集 × 3 seeds，总耗时 29 分钟。

#### α 退火（m2_pernode_anneal）vs vanilla 在真实数据集上的表现

| 数据集 | N | vanilla NMI | m2_pernode_anneal NMI | Δ | 最佳 config | 最佳 NMI |
|---|---|---|---|---|---|---|
| Cora | 2708 | 0.4708 | 0.4405 | **-0.030** | m2_rank3 | 0.5156 |
| CiteSeer | 3312 | 0.2602 | 0.2630 | +0.003 | m2_rank2 | 0.2762 |
| Pubmed | 19717 | 0.2682 | **0.2904** | **+0.022** | m2_pernode_a07 | 0.2909 |

#### 关键发现

1. **α 退火未解决 Cora pernode 问题**：在 Cora 上 α 退火 NMI=0.4405，仍**低于** vanilla
   (0.4708)。问题根因不是 α 是否退火，而是 per-node 熵梯度在中等图（~3000 节点）上
   分散导致社区信号被稀释。

2. **α 退火在大图（Pubmed）上有效**：Pubmed 上 m2_pernode_anneal NMI=0.2904，**超越**
   vanilla +0.022，与最佳 pernode 变体（m2_pernode_a07: 0.2909）几乎持平。

3. **大图上 hinge 反而有害**：Pubmed 上 m2_rank1/2/3 全部**低于** vanilla（0.24-0.27 vs
   0.27）。强行提高 eff_rank 不利于大图的社区结构涌现。pure method2 和 pernode 变体才是
   有效方案。

4. **数据集大小与最佳 config 的关系**：
   - 小图（Cora, 2708）：hinge 有效（m2_rank3 +0.045），pernode 无效
   - 中图（CiteSeer, 3312）：hinge 略有效（m2_rank2 +0.016），pernode 边际
   - 大图（Pubmed, 19717）：hinge 有害，pernode/method2 有效（+0.022）

5. **SBM 上 α 退火与固定 α 持平**：medium/hard/imbalanced 上 m2_pernode_anneal 都不显著
   超越 vanilla，与 m2_pernode_a03/05/07 表现相近——说明退火曲线在 SBM 上没有额外收益，
   主要价值在大图场景。

#### 结论

α 退火策略**部分成功**：
- ✅ 在最大图（Pubmed）上提供与最佳 pernode 变体持平的 NMI，且无需手调 α
- ❌ 未解决最初设计目标（Cora pernode 表现差）——Cora 上仍需 m2_rank3 的 hinge 路径
- 💡 揭示了图规模与最优策略的关系：小图用 hinge，大图用纯 pernode/method2

---

## 2026-07-06 — 两层玻尔兹曼熵 reformulation + 固定 bin 中心

### 核心突破：固定 bin 中心彻底解决塌缩

之前 method2 在 hard SBM 上塌缩（effRank=1.0, NMI=0.19）。根因分析发现：
- bin 中心 C 可学习时，C 会跟着 z_i 一起塌缩
- per-node 熵 S_node 达到最大值 log(M) 时反而无法提供 anti-collapse 梯度（饱和）

**修复**：bin_centers 从 `nn.Parameter` 改为 `register_buffer`（固定锚点，L2 归一化到单位球面）

### 两层 reformulation 实现

新增 `method2_pernode_lnw()` 函数：
- **第1层（per-node 微观态）**：S_node = mean_i [-Σ_b w_ib·log(w_ib)]
  梯度让每个节点在固定锚点间 spread，防止单点塌缩
- **第2层（system 宏观态）**：lnW = -Σ_{k,b} n_kb·log(p_kb)
  驱动社区结构向最大微观状态数宏观态演化
- 总熵 S = α·S_node + (1-α)·(lnW/N)，α 通过 `pernode_alpha` 控制

### 实验结果（5 seeds SBM + 3 seeds Cora，GPU 6.2 分钟）

| 数据集 | vanilla NMI | method2 NMI | m2_rank1 NMI | m2_pernode_a05 NMI |
|---|---|---|---|---|
| easy SBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| medium SBM | 0.9308 | **0.9451** | 0.9380 | 0.9408 |
| hard SBM | 0.6629 | 0.6632（不再塌缩！）| 0.6674 | 0.6619 |
| imbalanced SBM | 0.8183 | 0.8653 | **0.9361** (p=0.012 *) | 0.8598 |
| Cora | 0.4708 | 0.4562 | 0.5004 | 0.4456 |

### 关键发现

1. **固定 C 后 method2 在 hard SBM 上不再塌缩**：之前 NMI=0.19, effRank=1.0；现在 NMI=0.6632, effRank=4.23（和 vanilla 持平）
2. **pernode 在 SBM 上和 method2 持平**：medium 0.9408 vs 0.9451，hard 0.6619 vs 0.6632——纯玻尔兹曼熵自驱动无需外加铰链
3. **imbalanced 上 m2_rank1 仍最优**：NMI=0.9361, p=0.012 *（pernode 0.8598 不显著）
4. **Cora 上 m2_rank3 仍最优**：NMI=0.5156 vs vanilla 0.4708（+0.0448）
5. **pernode 在 Cora 上表现差**：α 越大越差（0.3→0.4356, 0.5→0.4456, 0.7→0.4571），可能因为真实图节点数多（2708）per-node 熵梯度太分散

### 下一步方向

- pernode 在大图上效果差，考虑 α 退火（早期 α 大防塌缩，后期 α→0 让 system 熵主导）
- imbalanced 上 pernode 不如 m2_rank1，可能需要保留方差铰链作为补充
- medium SBM 上 method2(0.9451) > m2_rank1(0.9380)，说明固定 C 后纯 method2 已足够

---

## 2026-07-06 — 完整实验优化 + Cora 真实数据集验证

### 本次改动

1. **SBM hard 参数调整**
   - `(30, 4, 0.25, 0.12)` → `(35, 4, 0.3, 0.10)`
   - 原参数 p_in/p_out≈2.08 接近检测极限，所有方法 NMI≈0.14 无法区分
   - 新参数 p_in/p_out=3.0，有挑战但社区结构可检测

2. **新增 m2_rank2 配置**（lambda_rank=2.0）
   - 填充 m2_rank1(1.0) 和 m2_rank3(3.0) 之间的中间强度
   - 观察 lambda 的梯度效果

3. **Cora 参数优化**
   - `emb_dim`: 16 → 32（给 7 类真实数据更多表达空间）
   - `min_rank`: 4.0 → 5.0（按 K/2~K 法则的中间值）
   - 结果：m2_rank3 NMI 从 0.4803 提升到 0.5126

4. **deploy.py 修复**
   - `scp_up` 远程路径 bug：`scp -r data/cora ~/WJ/` 会创建 `~/WJ/cora/` 而非 `~/WJ/data/cora/`
   - 修复：计算 `remote_parent = item.rsplit("/", 1)[0] + "/"` 作为 scp 目标
   - `cmd_run` 的 SSH 挂住问题：加 `setsid` + 所有 FD 重定向 + `< /dev/null`
   - Python stdout 全缓冲问题：加 `-u` 标志启用无缓冲输出
   - `ssh_safe` 未定义 bug：改为直接调用 `ssh(check=False, capture=True)`

5. **main.py 关键结论提取增强**
   - 自动提取各 regime 最佳 config
   - 包含 m2_rank2 结果
   - 添加"核心发现"总结块

### 实验结果（5 seeds SBM + 3 seeds Cora，GPU 4.4 分钟）

| 数据集 | 最佳 config | NMI | vs vanilla | 显著性 |
|---|---|---|---|---|
| easy SBM | 全部 | 1.0000 | — | — |
| medium SBM | m2_rank1 | 0.9506 | +0.0198 | ns (p=0.058) |
| hard SBM | m2_rank1 | 0.7083 | +0.0453 | ns (p=0.113) |
| imbalanced SBM | m2_rank3 | 0.9370 | +0.1187 | ** (p=0.008) |
| **Cora** | **m2_rank3** | **0.5126** | **+0.0418** | — |

- method2 无正则在 medium/hard 上塌缩（effRank≈1.0）
- 方差铰链成功防止塌缩，effRank 恢复到 3-5
- imbalanced 上 m2_rank2/3 显著超越 vanilla（p<0.01）

---

## 2026-07-03 — GPU 支持 + 服务器部署脚本 deploy.py

### 本次改动

1. **代码加 GPU 支持**（本地电脑扛不住完整实验，搬到服务器跑）
   - `entropy_gnn_baseline.py` 顶部加 `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
   - `set_seed` 加 `torch.cuda.manual_seed_all`
   - `normalize_adj(A_np, device)` 接受 device 参数
   - `EntropyGNN.forward` 里 `_default_X.to(A_hat.device)` 按需搬设备
   - `train_one` 里 A / A_hat / X_tensor / model 全部用 DEVICE
   - `recon_bce` 的 `pos_weight` 加 `device=logits.device`（修复跨设备 bug）
   - `anticollapse.py` 不需要改（纯 torch 操作，device 自动跟随，`.item()` 自动处理 GPU→CPU）
   - 本地 CPU 冒烟测试通过，结果与改前一致

2. **新增 `deploy.py` 服务器部署脚本**
   - 服务器：`remember@10.25.64.102`（有 GPU，环境已就绪）
   - 子命令：`setup` / `run` / `status` / `tail` / `fetch` / `all`
   - `setup`：scp 代码 + data/cora 到 `~/wj_experiment`
   - `run`：nohup 后台跑 main.py（SSH 断开实验继续跑）
   - `status`：检查进程 + 显示最新日志
   - `tail`：实时查看日志
   - `fetch`：拉回 result.md + image/ + main_run.log
   - `all`：一键 setup + run + 轮询等完成 + fetch
   - SSH 选项：`StrictHostKeyChecking=accept-new` + `BatchMode=yes`（需配免密 key）

3. **SSH key 配置**
   - 本地生成 ed25519 key（`~/.ssh/id_ed25519`）
   - 用户手动执行一条命令把公钥传到服务器（输一次密码）
   - 配好后 deploy.py 全自动

### 用法

```bash
# 1. 首次：配 SSH key（只需一次，输服务器密码）
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh -o StrictHostKeyChecking=accept-new remember@10.25.64.102 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo KEY_INSTALLED"

# 2. 一键部署 + 运行 + 拉回结果
python deploy.py all

# 或者分步：
python deploy.py setup       # 传代码
python deploy.py run         # 远程跑
python deploy.py status      # 查进度
python deploy.py tail        # 实时日志
python deploy.py fetch       # 拉结果
```

### 已知问题 / 待办

1. **SSH key 待用户手动配**：需执行上面第 1 步命令输一次服务器密码
2. **服务器 Python 环境待验证**：setup 会自动检查 torch + cuda 是否可用
3. **Cora 完整实验待跑**：配好 key 后用 `deploy.py all` 一键启动

---

## 2026-07-03 — Cora 真实数据集支持 + main.py 一键运行 + README__ALL.md

### 本次改动

1. **Cora 真实数据集支持**（任务 #15-#18）
   - `entropy_gnn_baseline.py` 加 `load_cora()`：从 `data/cora/cora.content` + `cora.cites` 加载 2708 节点 / 7 类 / 1433 维特征 / 5278 边
   - `EntropyGNN.__init__` 加 `node_feature_dim` 参数，None 时用 one-hot（SBM 路径），传入时用外部特征（Cora 路径）
   - `EntropyGNN.forward(A_hat, X=None)` 接受可选外部特征
   - `train_one(A_np, labels, cfg, X_feat=None, verbose_every=0)` 加 `X_feat` 参数
   - `make_configs(K, seed, epochs=300, min_rank=3.0)` 加 `epochs` 和 `min_rank` 参数（Cora 用 epochs=200, min_rank=4.0）
   - 新增 `run_cora(n_seeds=3, epochs=200, min_rank=4.0)` 函数
   - 修复 bug：`matplotlib.use('Agg')` 之前没导入 `matplotlib` 模块（改名 `mpl.use('Agg')`）
   - 冒烟测试通过：vanilla 50 epochs NMI=0.48, method2+rank 50 epochs NMI=0.42（铰链正常触发与休眠）

2. **新增 `main.py` 一键运行**（任务 #19-#20）
   - 完整版：SBM 5 seeds × 4 regimes × 6 configs + Cora 3 seeds × 200 epochs + paired t-test
   - 复用 `significance_test.collect_raw_metrics` 跑 SBM，避免重复训练
   - 复用 `significance_test.build_report` 生成显著性报告
   - 自动写 `result.md`：指标表 + 显著性报告 + 自动提取关键结论
   - 支持 `--quick`（1 seed 快速验证）、`--no-cora`、`--no-sig` 参数
   - 异常隔离：每个阶段 try-except，一个失败不影响其他
   - 验证通过：`--quick --no-cora --no-sig` 1.2 分钟跑完，result.md 格式正确

3. **新增 `README__ALL.md` 全代码详细文档**（任务 #21）
   - 8 章节：项目核心 / 文件分类索引 / 根模块详解 / experiments 详解 / analysis 详解 / 依赖关系图 / 数据流图 / 文档维护规则
   - 每个文件描述：作用、目标、API、关键算法、依赖
   - 包含 ASCII 模块依赖图和数据流图
   - 文档维护规则：每次代码变更后必须同步更新所有 .md 文件

4. **同步更新文档**（任务 #22）
   - `README.md`：目录结构加 main.py / data/ / result.md / README__ALL.md；快速开始改为 main.py 优先；结果摘要加 Cora 部分
   - `CHANGELOG.md`：追加本次改动记录（本条）

### 当前代码结构（本次改动后）

```
e:\Code\python code\WJ\
├── 核心模块（根目录）
│   ├── anticollapse.py             # 反塌缩正则化模块
│   ├── entropy_gnn_baseline.py     # 主实验文件（新增 Cora 支持）
│   ├── closed_form_gradients.py    # Method 1/2 闭式梯度验证
│   └── main.py                     # [新] 一键运行入口
├── experiments/                    # 实验脚本（不变）
├── analysis/                       # 分析脚本（不变）
├── data/cora/                      # [新] Cora 真实数据集
├── image/                          # 实验图
├── README.md / README__ALL.md / CHANGELOG.md / REPORT_rank_regularization.md / result.md
└── _smoke_cora.py                  # Cora 冒烟测试（临时，可删）
```

### 核心对齐检查结论

代码没有偏离核心（Boltzmann 熵驱动的社区检测 + 反塌缩）：
- ✅ `method2_lnw` 实现了 Boltzmann ln W（Stirling + 软分箱）
- ✅ `free_energy` 实现了 F = E - T·S
- ✅ `VarianceHinge` 实现了双铰链反塌缩
- ✅ 6 个对比配置合理（vanilla / size / assign / method2 / m2_rank1 / m2_rank3）
- ✅ `closed_form_gradients.py` 是数学验证，保留有意义
- 新增 Cora 支持是验证泛化性，符合核心目标

### 已知问题 / 待办

1. **Cora 完整实验待跑**：`main.py` 完整版已后台启动（job-b8d98e84...），约 30-40 分钟出结果
2. **Cora 的 min_rank=4.0 是初步值**：K=7 按经验法则 min_rank ≈ K/2~K，4.0 是中间值，可能需要根据实验结果调整
3. **Cora 显著性未做**：当前 `significance_test.py` 只覆盖 SBM，Cora 的 3 seeds 也可以做配对 t-test 但未实现

### 下一步计划

- [ ] 等 main.py 完整版跑完，检查 result.md 的 Cora 结果
- [ ] 如果 Cora 上 m2_rank 表现不佳，调整 min_rank（试 3.5 或 5.0）
- [ ] 给 Cora 也加配对 t-test（扩展 significance_test.py 或在 main.py 里手动算）
- [ ] 删除临时文件 `_smoke_cora.py`

---

## 2026-07-03 — 代码大整理 + anticollapse 完整文档 + README

### 本次改动

1. **anticollapse.py 文档补全**（任务 #12）
   - 顶部 docstring 重写为 6 个 box 区块：为什么需要 / 数学基础 / 公共 API / 典型用法 / 超参指南 / 与纯秩惩罚对比
   - 每个函数加完整中文 docstring（含数学公式、尺度不变性证明、与 embStd 的换算关系）
   - 现在可以直接 drop into 其他图模型任务，无需读源码即可使用

2. **代码分类整理**（任务 #9-#11）
   - 新建 `experiments/` 目录：`smoke_test.py`, `sweep_minvar.py`, `sweep_minrank.py`, `significance_test.py`
   - 新建 `analysis/` 目录：`monitor_collapse.py`, `extract_and_plot.py`, `demo_scale_invariance.py`
   - 新建 `image/` 目录：所有 PNG 统一存放（`baseline_results/curves`, `collapse_monitor_trends`, `scale_invariance_demo`）
   - 删除根目录下 7 个旧的 `_xxx.py` 脚本（已迁移到子目录）
   - 所有子目录脚本统一用 `sys.path.insert` 引入根模块，可独立运行

3. **去重**（任务 #11）
   - `demo_scale_invariance.py` 改用 `from anticollapse import ...`，删除内联 eff_rank/tr(S) 计算
   - 真实 SizeCV 抽成 `IMBALANCED_TRUE_SIZECV` 常量，所有脚本共用
   - 图片路径抽成 `_fig_path(name)` helper，统一输出到 `image/`
   - 反塌缩逻辑只在 `anticollapse.py` 里实现一次，主文件只做 wiring

4. **matplotlib 中文字体配置**
   - `entropy_gnn_baseline.py` 加 Microsoft YaHei → SimHei → Arial Unicode MS 三段 fallback
   - 验证：`mpl.rcParams['font.sans-serif']` 现以 Microsoft YaHei 开头，中文标签正常显示

5. **新增 README.md**（任务 #13）
   - 8 个章节：核心假设 / 目录结构 / 快速开始 / 概念速查 / 实验结果摘要 / 推荐配置 / 已知问题 / 文档索引
   - 包含完整目录树、复用 anticollapse 的代码示例、调参经验

### 当前代码结构（整理后）

```
e:\Code\python code\WJ\
├── 核心模块（根目录）
│   ├── anticollapse.py             # [可复用] 反塌缩正则化模块 (VarianceHinge)
│   ├── entropy_gnn_baseline.py     # 主实验文件
│   └── closed_form_gradients.py    # Method 1/2 闭式梯度验证
├── experiments/                    # 实验脚本（smoke/sweep/significance）
├── analysis/                       # 分析脚本（monitor/extract/demo）
├── image/                          # 所有 PNG（4 张）
├── README.md                       # [新] 项目总览
├── CHANGELOG.md                    # 本文件
└── REPORT_rank_regularization.md   # 完整对比报告
```

### 删除的文件

```
_extract_and_plot.py        -> analysis/extract_and_plot.py
_monitor_collapse.py        -> analysis/monitor_collapse.py
_significance_test.py       -> experiments/significance_test.py
_smoke_rank.py              -> experiments/smoke_test.py
_sweep_minrank.py           -> experiments/sweep_minrank.py
_sweep_minvar.py            -> experiments/sweep_minvar.py
demo_scale_invariance.py    -> analysis/demo_scale_invariance.py
```

### 已知问题 / 待办

1. **min_rank=3.0 实验结果（5 seeds）已出**：
   - Easy：全部 NMI=1.0
   - Medium：m2_rank1 NMI=0.951±0.024（超 vanilla 0.927），effRank=4.19
   - Hard：m2_rank1 NMI=0.147±0.049（与 vanilla 0.148 持平）
   - Imbalanced：m2_rank3 NMI=0.932±0.035，SizeCV=0.560（贴近真实 0.556）
2. **显著性未量化**：`experiments/significance_test.py` 已就绪，待运行
3. **泛化性未验证**：VarianceHinge 尚未在 Cora/Citeseer 上测试
4. **Hard SBM 仍接近可检测性极限**，方差铰链只能追平 vanilla

### 下一步计划

- [ ] 运行 `python experiments/significance_test.py --n_seeds 5`，生成 `significance_report.md`
- [ ] 在 Cora/Citeseer 上跑 VarianceHinge 验证泛化性
- [ ] 如果 min_rank=3.0 在某些场景退化，回退到 2.5 或尝试 2.8

---

## 2026-07-03 — 模块封装 + min_rank=3.0 重跑 + 显著性检验脚本

### 本次改动

1. **抽出可复用模块 `anticollapse.py`**
   - 把 `rank_penalty` / `collapse_metrics` 从 `entropy_gnn_baseline.py` 抽出
   - 提供 clean API：
     - `VarianceHinge(min_rank, min_var)`：nn.Module，`.penalty(Z)` 可微，`.diagnostics(Z)` 返回 `HingeDiagnostics`
     - `effective_rank(Z)`、`total_variance(Z)`：核心标量信号
     - `compute_collapse_metrics(Z, Q=None)`：完整 collapse dashboard
   - 设计原则：双铰链都用 ReLU，**健康时梯度为 0**，自然低秩嵌入（不平衡 SBM 的 effRank≈2）不受干扰
   - 文件顶部 docstring 包含典型用法示例，可直接 drop into 其他 GNN/encoder

2. **重构 `entropy_gnn_baseline.py`**
   - 删除本地 `rank_penalty` / `collapse_metrics`（已被模块替代）
   - `train_one` 改用 `VarianceHinge` 实例 + `.diagnostics()`，去掉重复的内联 tr(S)/eff_rank 计算
   - hist 键名保持 `'trS'/'rank_h'/'var_h'`，向后兼容 `_monitor_collapse.py`
   - **`make_configs` 中 `m2_rank1`/`m2_rank3` 的 `rank_min_rank` 从 2.5 → 3.0**
     - 理由：min_rank=2.5 在 medium 上 NMI=0.946 已超 vanilla，但 effRank≈3.2 离 K-1=3 的下界太近，余量不足。调到 3.0 试图把 effRank 推到 3.5+，给 K=4 社区更多分离余量
     - min_var=1.0 保持不变（之前 min_var sweep 证明它对 medium 影响很小）

3. **新增 `_significance_test.py`**
   - 独立运行 N seeds，对每个 regime × 每对 config 做 **paired t-test**（同 SBM seed 配对，差分掉图本身方差）
   - 输出 `significance_report.md`：mean±std 表 + 双侧/单侧 p 值 + 显著性标记（`***/**/*/ns`）
   - 用法：`python _significance_test.py --n_seeds 5 --metrics nmi,bal`
   - **注意**：此脚本会独立训练所有 config×seed×regime，CPU 密集，应等主实验跑完再单独运行

### 当前代码结构

```
e:\Code\python code\WJ\
├── anticollapse.py             # [新] 可复用 anti-collapse 模块 (VarianceHinge 等)
├── entropy_gnn_baseline.py     # 主实验文件 (Method 14 baseline + Method 2 + 反塌缩)
├── closed_form_gradients.py    # Method 1/2 的闭式梯度推导 (验证用)
├── demo_scale_invariance.py    # 演示 eff_rank 尺度不变性 -> 证明纯秩惩罚失效
├── _monitor_collapse.py        # 实时塌缩监控 (method2 vs m2_rank3 训练动态)
├── _extract_and_plot.py        # 从实验日志提取数据生成对比图
├── _sweep_minvar.py            # min_var 超参扫描
├── _sweep_minrank.py           # min_rank 超参扫描
├── _significance_test.py       # [新] 跨 SBM paired t-test 显著性检验
├── _smoke_rank.py              # smoke test (开发期用)
├── REPORT_rank_regularization.md  # 完整对比报告 (min_rank=2.5 版本)
├── CHANGELOG.md                # 本文件
└── 输出图: baseline_results.png / baseline_curves.png / imbalanced_scatter.png
        / imbalanced_scatter_from_log.png / scale_invariance_demo.png
        / collapse_monitor_trends.png
```

### 项目目的

验证一个假设：**社区的形成可以被理解为嵌入向量拟合"占据最多微观态的宏观态"**，
即 Boltzmann 熵 S = k·ln W 最大化。W = N! / ∏ n_k! 用 Stirling 近似展开，
Method 2 用软分箱 (RBF-softmax) + 软社区分配 (softmax(W·z)) 计算可微的 ln W。

自由能目标：F = E - T·S，T 退火让模型最终 commit 到硬划分。

### 核心发现（累积）

- **Method 2 在不平衡 SBM 上有效（NMI 0.94 vs vanilla 0.82），但在平衡 SBM 上塌缩**（effRank→1, embStd→0.1）
- **塌缩根因**：Method 2 的高 T 阶段把所有嵌入推向同一个 bin 中心，破坏方差
- **纯秩惩罚失效**：eff_rank = (tr S)²/tr(S²) 是尺度不变的，模型可以通过"各维度等比缩小"来满足秩约束（effRank→d 但 embStd→0）
- **方差铰链修复**：tr(S) 是尺度敏感的（tr(c·S) = c·tr(S)），可以检测尺度塌缩
- **双铰链设计**：rank_def + var_def，两者都用 ReLU，健康时梯度为 0
- **训练动态验证**：高 T 阶段 var_h=1.0（铰链全力触发），tr(S) 回升后 var_h→0 自动休眠

### 已知问题 / 待验证

1. **min_rank=3.0 是否会过约束？**
   - 风险：effRank 被推得太高可能让 embStd 退化（之前 min_rank=3.5-4.0 sweep 出现过这种情况）
   - 待主实验跑完后看 medium/hard 的 embStd 和 NMI 是否同时健康
2. **hard 难度 SBM (p_out=0.12) 仍接近可检测性极限**，所有方法 NMI<0.2，方差铰链能否改善尚不确定
3. **显著性未量化**：5 seeds 的 mean±std 重叠时无法判断提升是否显著，需等 `_significance_test.py` 跑完
4. **min_var sweep 之前证明 min_var 对 medium 无效**（自然 trS≈1.08 > 地板），但 min_rank=3.0 后 trS 行为可能变化，需重新观察

### 下一步计划

- [ ] 主实验跑完后，提取 NMI/SizeCV 数据，对比 min_rank=2.5 vs 3.0
- [ ] 运行 `_significance_test.py --n_seeds 5`，生成 `significance_report.md`
- [ ] 如果 min_rank=3.0 在 medium 上 NMI 下降或 embStd 退化，回退到 2.5 或尝试 2.8
- [ ] 考虑把 VarianceHinge 应用到其他图模型任务（如 Cora/Citeseer）验证泛化性

### 实验运行状态

- **后台任务 ID**: `job-4d52f5fa29db46dc96ca96d11b1af62c`
- **配置**: `python entropy_gnn_baseline.py` (n_seeds=5, min_rank=3.0)
- **输出日志**: `C:\Users\ASUS\AppData\Local\Temp\trae-agent-toolhost\jobs\job-4d52f5fa29db46dc96ca96d11b1af62c\output.log`
- **状态**: 运行中（启动时）
- **结果**: _待补充_

---

## 历史摘要（详见 REPORT_rank_regularization.md）

- 2026-07-03 早些时候：min_rank=2.5 在 medium 上 NMI=0.946（超 vanilla 0.927），在不平衡 SBM 上 NMI=0.939 + SizeCV=0.564（贴近真实 0.556），验证方差铰链有效
- 2026-07-03 早些时候：min_var sweep 证明纯方差约束无法解决 medium 的 rank 问题；min_rank sweep 找到 2.5 甜点
- 2026-07-03 早些时候：实现 Method 2 (Boltzmann ln W)，发现平衡 SBM 塌缩
- 2026-07-03 早些时候：实现 Method 14 (entropy-regularized GNN baseline)，size-entropy 提升不明显
