"""
anticollapse.py — 图嵌入反塌缩正则化模块（可复用）
=====================================================

本模块提供一个可直接 drop into 任意图嵌入模型（GCN/GAT/GraphSAGE 等）的
反塌缩正则化器。核心是「双铰链」设计：同时约束有效秩和总方差，覆盖两类塌缩模式。

┌──────────────────────────────────────────────────────────────────────┐
│  为什么需要这个模块                                                  │
├──────────────────────────────────────────────────────────────────────┤
│  熵正则化 GNN（Method 2: Boltzmann ln W 最大化）在高温度 T 阶段会把  │
│  所有节点嵌入推向同一个 bin 中心，导致嵌入塌缩：                      │
│    - effRank → 1（所有节点挤到一个点）                                │
│    - embStd → 0（嵌入方差消失）                                       │
│    - NMI 崩盘（KMeans 无法分离社区）                                  │
│                                                                       │
│  朴素的修复是加秩惩罚，但参与率 eff_rank = (tr S)²/tr(S²) 是尺度不  │
│  变的：模型可以通过「各维度等比缩小」来满足秩约束（effRank→d 但       │
│  embStd→0），让纯秩惩罚完全失效。                                    │
│                                                                       │
│  本模块的方差铰链用 tr(S) = Σ Var(Z[:,d])，它在 Z→c·Z 下缩放 c²，    │
│  是尺度敏感的，补上了纯秩惩罚的盲点。                                │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  数学基础                                                            │
├──────────────────────────────────────────────────────────────────────┤
│  设嵌入矩阵 Z ∈ R^{N×d}，中心化后 Z_c = Z - mean(Z)。协方差          │
│  S = Z_c^T Z_c / N ∈ R^{d×d}。                                       │
│                                                                       │
│  两个核心信号：                                                       │
│    1. 有效秩（参与率）                                                │
│         eff_rank(S) = (tr S)² / tr(S²) ∈ [1, d]                      │
│       尺度不变：eff_rank(c·S) = (c·tr S)²/(c²·tr S²) = eff_rank(S)   │
│       → 检测「秩塌缩」（多少维度有信号），但检测不了「尺度塌缩」     │
│                                                                       │
│    2. 总方差（迹）                                                    │
│         tr(S) = Σ_d Var(Z[:,d])                                       │
│       尺度敏感：tr(c·S) = c·tr(S)                                    │
│       → 检测「尺度塌缩」（信号强度），补上 eff_rank 的盲点            │
│                                                                       │
│  双铰链惩罚（ReLU 保证健康时梯度为 0）：                              │
│    penalty = relu(min_rank - eff_rank) / min_rank   # 秩铰链          │
│            + relu(min_var  - tr(S))    / min_var    # 方差铰链        │
│                                                                       │
│  关键性质：                                                           │
│    - 健康嵌入（eff_rank ≥ min_rank 且 tr(S) ≥ min_var）时惩罚 = 0    │
│    - 自然低秩嵌入（如不平衡 SBM 的 effRank≈2）不受干扰                │
│    - 塌缩时铰链自动触发，嵌入恢复后自动休眠                          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  公共 API                                                            │
├──────────────────────────────────────────────────────────────────────┤
│  effective_rank(Z, eps=1e-8) -> torch.Tensor                         │
│      参与率 (tr S)²/tr(S²)，可微，尺度不变                           │
│                                                                       │
│  total_variance(Z, eps=1e-8) -> torch.Tensor                         │
│      tr(S) = Σ Var(Z[:,d])，可微，尺度敏感                           │
│                                                                       │
│  VarianceHinge(min_rank=2.5, min_var=1.0, eps=1e-8)                  │
│      .penalty(Z)        -> torch.Tensor   可微惩罚（0 当健康）       │
│      .diagnostics(Z)    -> HingeDiagnostics  无梯度快照              │
│      .forward(Z)        -> torch.Tensor   等同 .penalty(Z)           │
│                                                                       │
│  HingeDiagnostics  (dataclass)                                       │
│      eff_rank : float   参与率 ∈ [1, d]                               │
│      tr_S     : float   总方差                                        │
│      rank_def : float   秩铰链分量 ∈ [0, 1]                           │
│      var_def  : float   方差铰链分量 ∈ [0, 1]                         │
│      penalty  : float   rank_def + var_def                            │
│                                                                       │
│  compute_collapse_metrics(Z, Q=None, eps=1e-8) -> dict               │
│      完整塌缩诊断面板：embed_std, eff_rank, tr_S,                    │
│      （可选）q_entropy, q_commit                                     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  典型用法                                                            │
├──────────────────────────────────────────────────────────────────────┤
│  from anticollapse import VarianceHinge                              │
│                                                                       │
│  hinge = VarianceHinge(min_rank=3.0, min_var=1.0)                    │
│  for epoch in range(epochs):                                         │
│      Z = model(graph)                                                │
│      loss = recon_loss(Z, A) + lambda_rank * hinge.penalty(Z)        │
│      loss.backward()                                                 │
│      opt.step()                                                      │
│      if epoch % 10 == 0:                                             │
│          d = hinge.diagnostics(Z)                                    │
│          print(f"tr(S)={d.tr_S:.3f} eff_rank={d.eff_rank:.2f} "      │
│                f"rank_h={d.rank_def:.3f} var_h={d.var_def:.3f}")     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  超参选择指南                                                        │
├──────────────────────────────────────────────────────────────────────┤
│  min_rank：                                                          │
│    - 物理意义：要求嵌入至少使用多少个有效维度                         │
│    - 经验法则：min_rank ≈ K/2 ~ K（K = 社区数）                      │
│    - K=4 社区：min_rank=2.5~3.0 是甜点（实验验证）                   │
│    - 太低（<2）：铰链休眠，等同无惩罚                                │
│    - 太高（>K）：过约束，方差增大但 NMI 下降                          │
│                                                                       │
│  min_var：                                                           │
│    - 物理意义：要求嵌入总方差不低于多少                               │
│    - 经验法则：min_var ≈ 0.5 ~ 1.0，应低于健康 trS                   │
│    - 健康嵌入的 trS 通常在 1.3~2.4（emb_dim=16, embStd~0.3）         │
│    - 设 min_var=1.0 可保证铰链在健康时休眠，塌缩时触发               │
│    - 太高（>2）：强制 embStd 到 vanilla 水平，可能伤 NMI              │
│                                                                       │
│  lambda_rank（惩罚权重，在训练循环里设）：                           │
│    - 1.0：温和约束，适合 K 小的场景                                   │
│    - 3.0：强约束，适合 K 大或塌缩严重的场景                           │
│    - 建议先试 1.0，看 eff_rank 是否被推到 min_rank 附近               │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  与纯秩惩罚的对比（详见 analysis/demo_scale_invariance.py）          │
├──────────────────────────────────────────────────────────────────────┤
│  场景：Z → α·Z（等比缩小，特征值比率不变）                           │
│                                                                       │
│  纯秩惩罚 -eff_rank/d：  对 α 完全平坦（尺度不变）→ 看不到塌缩      │
│  方差铰链 relu(1-trS)：  α→0 时 trS→0，铰链从 0 升到 1 → 捕获塌缩   │
│                                                                       │
│  实验数据（alpha=0.01 vs alpha=1.0）：                               │
│    eff_rank:  0.79 → 3.39  （纯秩惩罚看不到这个变化）                │
│    tr(S):     0.0001 → 0.889  （方差铰链捕获）                       │
│    pure_rank: -0.05 → -0.21  （对 alpha 盲）                         │
│    var_hinge: 1.00 → 0.11  （随 alpha 平滑变化）                     │
└──────────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


# ----------------------------- 核心信号 -----------------------------
def _covariance(Z: torch.Tensor, eps: float = 1e-8):
    """计算中心化协方差矩阵及其迹（内部复用，避免重复算两次 S）。

    S = Z_c^T Z_c / N，其中 Z_c = Z - mean(Z)。
    一次性算出 tr S 和 tr(S²)，供 effective_rank / total_variance /
    VarianceHinge 共享，避免在每个调用点重复 O(N·d²) 的协方差矩阵乘法。

    Returns:
        (S, tr, tr2): 协方差矩阵 (d,d)、tr S（带 eps 下界）、tr(S²)（带 eps 下界）
    """
    N, d = Z.shape
    Zc = Z - Z.mean(dim=0, keepdim=True)        # 中心化
    S = (Zc.t() @ Zc) / N                        # 协方差矩阵 (d, d)
    tr = S.trace().clamp(min=eps)                # tr S = Σλᵢ
    tr2 = (S * S).sum().clamp(min=eps)           # tr(S²) = Σλᵢ²
    return S, tr, tr2


def effective_rank(Z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """有效秩（参与率）= (tr S)² / tr(S²)，取值 [1, d]，尺度不变。

    数学：S = Z_c^T Z_c / N 是协方差矩阵，tr S = Σλᵢ，tr(S²) = Σλᵢ²，
    参与率 = (Σλᵢ)²/(Σλᵢ²) 衡量「有效维度数」。

    尺度不变性证明：S → c·S 时，(c·tr S)²/(c²·tr S²) = (tr S)²/tr(S²)，
    c² 在分子分母间抵消。这意味着纯秩惩罚无法检测「各维度等比缩小」型塌缩。

    Args:
        Z: 嵌入矩阵 (N, d)
        eps: 数值稳定地板
    Returns:
        标量 tensor，可微，值域 [1, d]
    """
    _, tr, tr2 = _covariance(Z, eps)
    return tr * tr / tr2                          # 参与率 (tr S)²/tr(S²)


def total_variance(Z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """总方差 tr(S) = Σ_d Var(Z[:,d])，尺度敏感。

    这是 effective_rank 的互补信号：tr(c·S) = c·tr(S)，所以能检测
    effective_rank 看不到的「尺度塌缩」。

    与 embStd 的关系：tr(S) = d · embStd²（当各维度方差均匀时），
    所以 tr(S) ≈ 1.0 对应 embStd ≈ sqrt(1.0/16) ≈ 0.25（d=16 时）。

    Args:
        Z: 嵌入矩阵 (N, d)
        eps: 数值稳定地板
    Returns:
        标量 tensor，可微
    """
    _, tr, _ = _covariance(Z, eps)
    return tr


# ----------------------------- 铰链正则化器 -----------------------------
@dataclass
class HingeDiagnostics:
    """铰链状态快照（无梯度），用于日志/绘图。

    所有字段都是 python float，可在 print / logging 中直接使用。
    """
    eff_rank: float       # 参与率 ∈ [1, d]；低 => 秩塌缩
    tr_S: float           # 总方差；低 => 尺度塌缩
    rank_def: float       # 秩铰链分量 ∈ [0, 1]；>0 => 秩不足
    var_def: float        # 方差铰链分量 ∈ [0, 1]；>0 => 方差不足
    penalty: float        # rank_def + var_def；>0 => 嵌入不健康


class VarianceHinge(nn.Module):
    """双铰链反塌缩正则化器（nn.Module，可直接 drop into 任意训练循环）。

    penalty = relu(min_rank - eff_rank) / min_rank    # 秩铰链：检测秩塌缩
            + relu(min_var  - tr(S))    / min_var     # 方差铰链：检测尺度塌缩

    两个铰链都用 ReLU，所以健康嵌入（eff_rank ≥ min_rank 且 tr(S) ≥ min_var）
    时梯度为 0，不会干扰自然低秩结构（如不平衡 SBM 的 effRank≈2 是正确的）。

    设计要点：
      - 方差铰链是核心：eff_rank 尺度不变，纯秩惩罚会被「等比缩小」绕过；
        tr(S) 尺度敏感，能捕获这个盲点。
      - 秩铰链是辅助：防止极端 rank-1 退化（所有节点挤到一个点）。
      - 两者都只在「病态」时激活，健康时自动休眠。

    Args:
        min_rank: 有效秩地板。低于此值时秩铰链触发。
                  典型值：K=4 社区用 2.5~3.0（需要 ≥ K-1 维线性分离）。
        min_var:  总方差地板。低于此值时方差铰链触发。
                  典型值：0.5~1.0；应低于健康 trS（~1.3-2.4）以保持休眠。
        eps:      除法数值稳定地板。
    """

    def __init__(self, min_rank: float = 2.5, min_var: float = 1.0,
                 eps: float = 1e-8):
        super().__init__()
        self.min_rank = float(min_rank)
        self.min_var = float(min_var)
        self.eps = float(eps)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """等同 penalty(Z)，让本类可被当作 nn.Module 调用。"""
        return self.penalty(Z)

    def penalty(self, Z: torch.Tensor) -> torch.Tensor:
        """可微反塌缩惩罚（健康时为 0）。

        单次计算协方差矩阵后同时取 effective_rank 和 tr(S)，避免重复算 S。

        Args:
            Z: 嵌入矩阵 (N, d)
        Returns:
            标量 tensor，可加入训练 loss：loss = recon + lambda * hinge.penalty(Z)
        """
        _, tr, tr2 = _covariance(Z, self.eps)
        er = tr * tr / tr2
        # ReLU 保证健康时梯度为 0；除以地板值让惩罚归一化到 [0, 1]
        rank_def = torch.relu(self.min_rank - er) / max(self.min_rank, 1.0)
        var_def = torch.relu(self.min_var - tr) / max(self.min_var, self.eps)
        return rank_def + var_def

    @torch.no_grad()
    def diagnostics(self, Z: torch.Tensor) -> HingeDiagnostics:
        """无梯度快照，用于日志/绘图（不影响反向传播）。

        单次计算协方差矩阵后同时取 effective_rank 和 tr(S)，避免重复算 S。

        Args:
            Z: 嵌入矩阵 (N, d)
        Returns:
            HingeDiagnostics dataclass，含 eff_rank, tr_S, rank_def, var_def, penalty
        """
        _, tr, tr2 = _covariance(Z, self.eps)
        er = (tr * tr / tr2).item()
        tr_val = tr.item()
        rank_def = max(0.0, self.min_rank - er) / max(self.min_rank, 1.0)
        var_def = max(0.0, self.min_var - tr_val) / max(self.min_var, self.eps)
        return HingeDiagnostics(er, tr_val, rank_def, var_def, rank_def + var_def)

    def extra_repr(self) -> str:
        """nn.Module 打印时的额外信息。"""
        return f"min_rank={self.min_rank}, min_var={self.min_var}"


# ----------------------------- 完整塌缩诊断面板 -----------------------------
def compute_collapse_metrics(Z: torch.Tensor, Q: Optional[torch.Tensor] = None,
                             eps: float = 1e-8) -> Dict[str, float]:
    """完整塌缩诊断面板，返回多个指标的 dict。

    前三个指标只依赖 Z（嵌入本身），后两个依赖 Q（社区分配头）。
    Q 为 None 时只返回前三个。

    单次计算协方差矩阵后同时取 effective_rank 和 tr(S)，避免重复算 S。

    Args:
        Z: 嵌入矩阵 (N, d)
        Q: 社区分配矩阵 (N, K)，soft assignment（每行和为 1）。可选。
        eps: 数值稳定地板
    Returns:
        dict，包含：
          embed_std : 所有嵌入元素的标准差（点塌缩检测，低 => 塌缩到一个点）
          eff_rank  : 参与率 ∈ [1, d]（秩塌缩检测）
          tr_S      : 总方差（尺度塌缩检测）
          q_entropy : 节点社区熵均值（高 => Q 均匀/退化，仅当 Q 非 None）
          q_commit  : 节点最大 q_ik 均值（低 => 无社区承诺，仅当 Q 非 None）
    """
    _, tr, tr2 = _covariance(Z, eps)
    out: Dict[str, float] = {
        'embed_std': Z.std().item(),
        'eff_rank': (tr * tr / tr2).item(),
        'tr_S': tr.item(),
    }
    if Q is not None:
        # 社区头诊断：q_entropy 高 + q_commit 低 => Q 退化到均匀分布
        out['q_entropy'] = (-(Q * torch.log(Q + eps)).sum(dim=-1)).mean().item()
        out['q_commit'] = Q.max(dim=-1)[0].mean().item()
    return out
