# 基于 Boltzmann 熵的图神经网络社区分类

> 验证一个物理直觉：**社区的形成可以被理解为嵌入向量拟合「占据最多微观态的宏观态」**，
> 即 Boltzmann 熵 `S = k·ln W` 最大化。本项目用可微的 Stirling 近似 + 软分箱，
> 把微观态计数 `W = N! / ∏ n_k!` 直接嵌入 GNN 训练损失，并配合双铰链反塌缩正则化。

---

## 1. 项目核心

### 1.1 Boltzmann 熵视角的社区检测

把节点嵌入 `Z ∈ R^{N×d}` 视作统计力学中的微观态，社区结构 `Q ∈ R^{N×K}` 视作宏观态。
Boltzmann 假设：**最可能被观测到的宏观态，是占据最多微观态数的那个**（`W` 最大）。

```
W = N! / ∏_k n_k!              # 社区大小 n_k 的多重组合数
S = k · ln W                   # Boltzmann 熵
```

### 1.2 Method 2：Stirling 近似 + 软分箱

直接最大化 `ln W` 不可微。本项目用两步近似让它可微：

1. **Stirling 展开**：`ln W ≈ -Σ_{k,b} n_kb · log(p_kb)`，把社区大小计数推广到 (社区 k, bin b) 联合计数 `n_kb`
2. **软分箱**：`w_ib = softmax_b(-‖z_i - c_b‖² / 2σ²)`（RBF-softmax，bin 中心 `c_b` 可学习），让 `n_kb = Σ_i q_ik · w_ib` 可微

最终自由能目标：

```
F(Z, Q) = E(Z, A) - T(t) · S(Q)
  E(Z, A) = BCE_with_logits(Z Z^T, A)          # 图重建
  S(Q)    = ln W / N                            # 每节点熵
  T(t)    = T_max · 0.5(1 + cos(π·s))          # 退火，T→0 时 commit 到硬划分
```

### 1.3 塌缩问题与双铰链修复

**Method 2 在平衡 SBM 上发生严重塌缩**：高 T 阶段所有嵌入被推向同一个 bin 中心，
`effRank → 1`、`embStd → 0`、NMI 从 0.93 跌到 0.40。

朴素修复是加秩惩罚，但参与率 `eff_rank = (tr S)²/tr(S²)` 是**尺度不变**的：
模型可以通过「各维度等比缩小」来满足秩约束（`effRank→d` 但 `embStd→0`），让纯秩惩罚完全失效。

本项目用**双铰链**修复（详见 [anticollapse.py](file:///e:/Code/python%20code/WJ/anticollapse.py)）：

```
penalty = ReLU(min_rank - eff_rank) / min_rank    # 秩铰链：阻止 rank-1 退化
        + ReLU(min_var  - tr(S))    / min_var    # 方差铰链：尺度敏感，阻止塌缩到点
```

**关键性质**：ReLU 保证健康时梯度为 0，自然低秩结构（如不平衡 SBM 的 `effRank≈2`）不受干扰。

### 1.4 关键概念速查

| 概念 | 公式 / 含义 |
|---|---|
| **W（微观态数）** | `W = N! / ∏ n_k!`，社区多重组合数 |
| **ln W（Stirling 近似）** | `ln W ≈ -Σ_{k,b} n_kb · log(p_kb)`，可微 |
| **软分箱** | `w_ib = softmax_b(-‖z_i - c_b‖² / 2σ²)`，RBF-softmax |
| **自由能** | `F = E - T·S`，T 退火让模型最终 commit 到硬划分 |
| **参与率 / 有效秩** | `eff_rank = (tr S)² / tr(S²) ∈ [1, d]`，**尺度不变** |
| **总方差** | `tr(S) = Σ_d Var(Z[:,d])`，**尺度敏感**（互补信号） |
| **双铰链** | `ReLU(min_rank - eff_rank) + ReLU(min_var - tr S)` |
| **NMI** | 标准化互信息，社区预测 vs 真实 SBM 标签 |
| **SizeCV** | 社区大小的变异系数，不平衡 SBM 真实值 ≈ 0.556 |

### 1.5 推荐配置与调参

```python
TrainConfig(
    entropy="method2", T_max=0.3, T_warmup=0.2, anneal="cosine",
    n_bins=16, sigma=0.5, lambda_rank=3.0,
    rank_min_rank=3.0,   # 秩铰链地板 ≈ K/2~K（K 为社区数）
    rank_min_var=1.0,    # 方差铰链地板 ≈ 0.5~1.0，低于健康 trS≈1.3~2.4
)
```

`lambda_rank` 先试 1.0 看 `eff_rank` 是否被推到 `min_rank` 附近，不够再升到 3.0。

---

## 2. 快速开始

### 2.1 环境依赖

```
python ≥ 3.8
torch, networkx, numpy, scikit-learn, matplotlib, scipy
```

Windows 上 matplotlib 中文字体自动尝试 Microsoft YaHei → SimHei → Arial Unicode MS（见 [entropy_gnn_baseline.py](file:///e:/Code/python%20code/WJ/entropy_gnn_baseline.py) 第 52-59 行）。

### 2.2 一键跑完整实验（推荐，服务器）

封装 `deploy.py all`，跑 3 seeds × 3 真实数据集 + 4 SBM regimes，约 30-45 分钟：

```bash
python run_auto_all.py                 # 默认 3 seeds
python run_auto_all.py --seeds 5       # 更稳定但更慢
python run_auto_all.py --seeds 1       # 快速验证，约 5 分钟
python run_auto_all.py --resume        # 从断点恢复（跳过已完成 task）
python run_auto_all.py --fresh          # 删除旧断点，从头开始
```

流程：setup（传代码+数据）→ run（后台跑）→ 等待 → fetch（拉回结果）。

### 2.3 跑单数据集 / 单 SBM regime

```bash
python auto_config.py --all --seeds 3              # 跑所有数据集 + SBM regimes
python auto_config.py --dataset cora --seeds 3     # 只跑单个真实数据集
python auto_config.py --sbm hard --seeds 5         # 只跑单个 SBM regime
python auto_config.py --analyze-only --dataset cora  # 只分析图属性和决策（不训练，秒级）
```

### 2.4 本地直接跑（不用服务器）

```bash
cd "e:\Code\python code\WJ"
python entropy_gnn_baseline.py          # 5 seeds × 3 SBM × 6 configs
python main.py                           # 完整版：SBM + Cora + 显著性检验（30-40 分钟）
python main.py --quick                   # 快速版（1 seed，5 分钟）
python experiments/smoke_test.py         # 冒烟测试（约 30 秒）
```

输出：`image/baseline_results.png`、`image/baseline_curves.png`、`result.md`。

### 2.5 实时塌缩监控

```bash
python analysis/monitor_collapse.py
```

每 10 epoch 打印 `tr(S) / eff_rank / rank_h / var_h`，肉眼确认铰链触发与休眠。输出 `image/collapse_monitor_trends.png`。

### 2.6 在服务器上分步操作（GPU 加速）

本地跑不动时，用 [deploy.py](file:///e:/Code/python%20code/WJ/deploy.py) 分步部署（首次需配 SSH key 免密登录）：

```bash
python deploy.py setup       # 传代码 + data/cora 到服务器
python deploy.py run         # nohup 后台跑 main.py（SSH 断开不影响）
python deploy.py status      # 查进度
python deploy.py tail        # 实时看日志
python deploy.py fetch       # 拉回 result.md + image/
python deploy.py all         # 一键到底
```

代码自动检测 GPU：有 cuda 用 cuda，无则回退 cpu。Cora 2708 节点在 GPU 上比 CPU 快 5-10 倍。

### 2.7 复用反塌缩模块到其他图模型

```python
from anticollapse import VarianceHinge
hinge = VarianceHinge(min_rank=3.0, min_var=1.0)
loss = recon_loss(Z, A) + lambda_rank * hinge.penalty(Z)  # 健康时梯度为 0
d = hinge.diagnostics(Z)  # 无梯度快照：d.tr_S / d.eff_rank / d.rank_def / d.var_def
```

完整 API 与调参指南见 [anticollapse.py](file:///e:/Code/python%20code/WJ/anticollapse.py) 顶部 docstring。

---

## 3. 代码结构

### 3.1 文件分类索引

| 类别 | 文件 | 作用 |
|---|---|---|
| **核心模块** | [anticollapse.py](file:///e:/Code/python%20code/WJ/anticollapse.py) | 反塌缩正则化模块（VarianceHinge），可复用，无内部依赖 |
| | [entropy_gnn_baseline.py](file:///e:/Code/python%20code/WJ/entropy_gnn_baseline.py) | 主实验文件：EntropyGNN + Method 2 + 反塌缩 + Cora 支持 |
| | [closed_form_gradients.py](file:///e:/Code/python%20code/WJ/closed_form_gradients.py) | Method 1/2 闭式梯度推导 + autograd 验证（独立脚本） |
| **入口** | [main.py](file:///e:/Code/python%20code/WJ/main.py) | 一键运行完整实验，结果写入 result.md |
| | [auto_config.py](file:///e:/Code/python%20code/WJ/auto_config.py) | 图属性分析 → config 自动选择 → 训练 → 报告 |
| | [run_auto_all.py](file:///e:/Code/python%20code/WJ/run_auto_all.py) | 一键脚本：封装 deploy.py 在服务器上跑完整实验 |
| | [deploy.py](file:///e:/Code/python%20code/WJ/deploy.py) | 部署工具：代码同步、远程运行、结果拉回 |
| | [extract_nmi_summary.py](file:///e:/Code/python%20code/WJ/extract_nmi_summary.py) | 从 auto_config_result.md 提取 NMI 生成对比表 |
| **实验** | [experiments/smoke_test.py](file:///e:/Code/python%20code/WJ/experiments/smoke_test.py) | 快速冒烟测试（1 seed，验证代码可用） |
| | [experiments/sweep_minvar.py](file:///e:/Code/python%20code/WJ/experiments/sweep_minvar.py) | min_var 超参扫描 |
| | [experiments/sweep_minrank.py](file:///e:/Code/python%20code/WJ/experiments/sweep_minrank.py) | min_rank 超参扫描 |
| | [experiments/significance_test.py](file:///e:/Code/python%20code/WJ/experiments/significance_test.py) | 跨 SBM paired t-test 显著性检验 |
| **分析** | [analysis/monitor_collapse.py](file:///e:/Code/python%20code/WJ/analysis/monitor_collapse.py) | 实时塌缩监控（method2 vs m2_rank3 训练动态） |
| | [analysis/extract_and_plot.py](file:///e:/Code/python%20code/WJ/analysis/extract_and_plot.py) | 从实验日志提取 NMI/SizeCV 生成对比图 |
| | [analysis/demo_scale_invariance.py](file:///e:/Code/python%20code/WJ/analysis/demo_scale_invariance.py) | 演示 eff_rank 尺度不变性 → 证明纯秩惩罚失效 |
| **数据** | data/cora/cora.content + cora.cites | Cora 引用网络（2708 节点 × 1433 维特征 + 7 类） |
| **输出** | image/*.png | 所有实验图（baseline_results / baseline_curves / imbalanced_scatter / collapse_monitor_trends / scale_invariance_demo） |
| | [result.md](file:///e:/Code/python%20code/WJ/result.md) | main.py 自动生成的实验结果汇总 |

### 3.2 目录约定与依赖

- 子目录脚本统一用 `sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))` 引入根模块
- 图片统一通过 `entropy_gnn_baseline._fig_path(name)` 输出到 `image/`；真实 SizeCV 常量用 `entropy_gnn_baseline.IMBALANCED_TRUE_SIZECV`
- 反塌缩逻辑统一从 `anticollapse` 模块导入，**禁止在主文件里重复实现**
- 依赖链：`anticollapse.py`（叶子，无内部依赖）← `entropy_gnn_baseline.py`（主实验）← `experiments/` + `analysis/` + `main.py`（额外 import `significance_test`）

---

## 4. auto_config 工具用法

### 4.1 自动 config 决策规则

基于已验证实验结果，根据图规模和结构属性自动选择最佳 GNN config：

| 条件 | 选择 config | 理由 |
|---|---|---|
| 大图 (N>5000) | method2（纯） | hinge 在大图上干扰社区形成 |
| 密集图 (avg_deg>15，SBM 类) | m2_rank1（弱 hinge） | SBM 上弱 hinge 最佳 |
| 稀疏小图 + 强社区 (CC≥0.20) | m2_rank3（强 hinge） | 如 Cora |
| 稀疏小图 + 中等社区 (CC≥0.05) | m2_rank2（中 hinge） | 如 CiteSeer |
| 弱社区 (CC<0.05) | method2（纯） | hinge 无益 |

### 4.2 断点恢复系统

长时间实验可能因 SSH 断连/服务器重启中断。断点系统记录已完成任务，支持恢复。

**断点粒度**：一个 task = 一个数据集/SBM regime 的全部 seeds。中断时最多丢失一个 task 的工作（约 3-5 分钟），已完成的 task 结果不丢。

| 场景 | 命令 | 行为 |
|---|---|---|
| 正常首次跑 | `python auto_config.py --all --seeds 3` | 从头开始；检测到旧断点时提示用 --resume 或 --fresh |
| 断后恢复 | `python auto_config.py --all --seeds 3 --resume` | 加载断点，跳过已完成 task，从断点继续 |
| 强制重跑 | `python auto_config.py --all --seeds 3 --fresh` | 删除旧断点，从头开始 |

断点文件：`auto_config_checkpoint.json`（JSON，每个 task 完成后原子写入，防写到一半崩溃）和 `auto_config_progress.log`（人类可读的进度日志，含时间戳 + task 进度）。实验全部完成后断点文件自动清理。

### 4.3 决策日志

每次 config 决策都记录到 `auto_config_decisions.jsonl`（JSONL 格式，每行一条）：

```json
{"timestamp": "2026-07-07 19:36:14", "dataset": "sbm_easy", "N": 240, "E": 4146,
 "avg_deg": 34.55, "density": 0.144561, "cc": 0.364079, "K": 4,
 "selected_config": "m2_rank1", "reason": "密集图(avg_deg=34.5>15, SBM 类) → m2_rank1"}
```

字段：timestamp, dataset, N, E, avg_deg, density, cc, K, selected_config, reason。

### 4.4 提取 NMI 对比表与完整工作流

```bash
# 1. 一键跑完整实验（服务器，30-45 分钟）
python run_auto_all.py

# 2. 跑完后提取 NMI 对比表（默认读 auto_config_result.md，输出 nmi_comparison.md + .csv）
python extract_nmi_summary.py
python extract_nmi_summary.py --input other_result.md      # 指定输入
python extract_nmi_summary.py --csv-only                   # 只输出 CSV

# 3. 查看结果与决策日志
type nmi_comparison.md
type auto_config_decisions.jsonl

# 4. 如果中途中断了，恢复
python run_auto_all.py --yes --resume
```

输出文件清单：`auto_config_result.md`（完整结果报告）、`auto_config_decisions.jsonl`（决策日志）、`auto_config_checkpoint.json`（断点，完成后自动删除）、`auto_config_progress.log`（进度日志）、`auto_config_run.log`（远程运行日志）、`nmi_comparison.md` + `nmi_comparison.csv`（NMI 对比表）。

---

## 5. 实验结果

实验结果不在此重复，请直接查阅以下文件：

- [result.md](file:///e:/Code/python%20code/WJ/result.md) — main.py 自动生成的全量 14 config 实验结果汇总（5 seeds × 3 SBM 难度 × 6 configs + Cora 3 seeds + paired t-test 显著性检验）
- [nmi_comparison.md](file:///e:/Code/python%20code/WJ/nmi_comparison.md) — auto_config 实验的 NMI 对比表（Vanilla vs Auto config，按数据集逐项对比）

**核心结论摘要**：
1. 方差铰链成功修复平衡 SBM 的塌缩：NMI 0.40 → 0.95，embStd 0.16 → 0.27
2. 不平衡 SBM 上铰链休眠不干扰原有优势，反而把 NMI 从 0.83 推到 0.93
3. SizeCV 从 0.29 → 0.56，**正确恢复不平衡的社区结构**（而非强制平衡）
4. auto_config 在 Cora 上把 NMI 从 0.47 推到 0.53（+0.06），在 CiteSeer 上从 0.19 推到 0.27（+0.08）

---

## 6. 文档索引

| 文档 | 用途 |
|---|---|
| [README.md](file:///e:/Code/python%20code/WJ/README.md) | 项目总览（本文件） |
| [CHANGELOG.md](file:///e:/Code/python%20code/WJ/CHANGELOG.md) | 每次运行的更新说明，最新在上 |
| [result.md](file:///e:/Code/python%20code/WJ/result.md) | main.py 自动生成的实验结果汇总（全量 14 config） |
| [nmi_comparison.md](file:///e:/Code/python%20code/WJ/nmi_comparison.md) | auto_config 实验的 NMI 对比表 |
| [anticollapse.py](file:///e:/Code/python%20code/WJ/anticollapse.py) 顶部 docstring | 反塌缩模块的完整 API 文档与用法示例 |

---

## 7. 已知问题与待办

1. **Hard 难度 SBM 接近可检测性极限**（`p_out=0.12`），所有方法 NMI<0.2，方差铰链只能追平 vanilla 不能超越
2. **min_rank=3.0 vs 2.5 的权衡**：3.0 在 medium 上 NMI 略高（0.951 vs 0.920），但需观察 embStd 是否退化
3. **统计显著性**：5 seeds 的 mean±std 重叠时无法下结论，需运行 `experiments/significance_test.py`
