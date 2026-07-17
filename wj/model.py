"""model.py — 熵正则 GNN 模型定义 + 损失函数 + 自由能。

职责：模型结构和前向计算的唯一定义点。
  - GCNLayer / EntropyGNN：两层 GCN → 嵌入 Z → 社区头 Q
  - 损失函数：recon_bce（图重建）、size_entropy / assign_entropy（社区熵变体）
  - method2_lnw：玻尔兹曼微观态计数 ln W（Stirling + soft binning）
  - free_energy：自由能 F = E - T·S（统一入口）

无内部依赖（EntropyGNN.forward 通过 A_hat.device 动态获取设备，不依赖全局 DEVICE）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Model -----------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X, A_hat):
        return self.lin(A_hat @ X)


class EntropyGNN(nn.Module):
    """两层 GCN → 嵌入 Z → 社区头 Q（可选双路编码器）。

    bin 中心 C 是固定锚点（register_buffer，不可学习），用正态初始化后 L2
    归一化到单位球面。固定 C 让 method2 的 ln W 梯度真正指向"z_i 在多个
    固定锚点间 spread"，而不是通过移动 C 偷懒到饱和状态（历史教训：可学习
    C 会跟着 z_i 一起塌缩）。

    节点特征支持两种模式：
      - 默认（node_feature_dim=None）：用 one-hot I_N，SBM 实验用
      - 外部特征（Cora 等）：传入 node_feature_dim，forward 时再传 X

    双路编码器（dual_encoder=True，仅在有外部特征时生效）：
      - 图路径（GCN）：A_hat @ X → H_g → Z_g，编码图结构社区
      - 特征路径（MLP）：X → H_f → Z_f，编码特征语义（不经过 GCN）
      - 融合：Z = Z_g + Z_f，让 Z 同时编码图结构和特征语义
      动机：CiteSeer 弱社区图（CC=0.14）图结构不可靠，mod=0.75 但 ARI=0.22
      （图结构社区≠标签社区）。特征路径不依赖图结构，弥补 GCN 过度依赖
      弱社区结构的缺陷。DCRN 的双自编码器已证明此架构有效。
    """
    def __init__(self, n_nodes: int, hidden_dim: int, emb_dim: int,
                 n_communities: int, n_bins: int = 16,
                 node_feature_dim: int = None, feat_recon: bool = False,
                 dual_encoder: bool = True):
        super().__init__()
        # 输入特征维度：外部特征优先，否则退化为 one-hot（SBM 无节点特征）
        feat_dim = node_feature_dim if node_feature_dim is not None else n_nodes
        # 图路径（现有）：两层 GCN
        self.gcn1 = GCNLayer(feat_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, emb_dim)
        # 特征路径（新增）：两层 MLP，不经过 GCN
        # 仅在有外部特征时启用（one-hot 无语义信息，双路无意义）
        self.dual_encoder = dual_encoder and node_feature_dim is not None
        if self.dual_encoder:
            self.fc1_feat = nn.Linear(feat_dim, hidden_dim)
            self.fc2_feat = nn.Linear(hidden_dim, emb_dim)
        self.comm_head = nn.Linear(emb_dim, n_communities)
        # 特征重建解码器（可选）：从 Z 重建 X，强制 Z 编码特征信息
        # 适用于弱社区图（CiteSeer），特征比图结构更可靠
        self.feat_decoder = nn.Linear(emb_dim, feat_dim) if feat_recon else None
        # 逐维 bin 中心：固定 1D 锚点（不可学习）。
        # 形状 (d, M)：第 j 行是第 j 维的 M 个 1D bin 中心。
        # 每维独立 L2 归一化到单位球面，保证每维的 M 个 bin 中心 spread 开。
        # 固定 C 让 method2 的 ln W 梯度真正指向"z_i 在多个固定锚点间 spread"，
        # 而不是通过移动 C 偷懒到饱和状态（历史教训：可学习 C 会跟着 z_i 一起塌缩）。
        # 逐维 1D binning 相比 d 维整体 binning：
        #   - 粒度更细（d×M 个 1D bin vs M 个 d 维 bin）
        #   - 每维独立优化，单维塌缩时该维 S_j→0，更敏感的塌缩检测
        #   - 每维 L2 归一化匹配 Z 的单维典型尺度（原版 d 维 L2 归一化的逐维推广）
        _anchors = torch.randn(emb_dim, n_bins)
        # 每维独立 L2 归一化：dim=1 对应 M 个 bin 中心
        _anchors = _anchors / _anchors.norm(dim=1, keepdim=True)
        self.register_buffer("bin_centers", _anchors)
        # 默认特征缓存：SBM 用 one-hot；Cora 等真实图由 forward 接收 X_feat
        # 注意：_default_X 不注册为 parameter/buffer，避免被 .to(device) 重复移动；
        # 在 train_one 里手动 .to(DEVICE)
        self._default_X = torch.eye(n_nodes) if node_feature_dim is None else None

    @property
    def W(self) -> torch.Tensor:
        """Community prototype matrix (K, d)."""
        return self.comm_head.weight

    def forward(self, A_hat, X=None):
        # 外部特征优先；否则用 one-hot（SBM 路径，向后兼容）
        # _default_X 不是 nn.Parameter，model.to(device) 不会移动它，
        # 这里按需搬到 A_hat 所在设备，避免跨设备报错
        if X is None:
            X = self._default_X.to(A_hat.device)
        # 图路径（GCN）：A_hat @ X → H_g → Z_g，编码图结构社区
        H_g = F.relu(self.gcn1(X, A_hat))
        Z_g = self.gcn2(H_g, A_hat)
        # 特征路径（MLP）：X → H_f → Z_f，编码特征语义（不经过 GCN）
        # 仅在 dual_encoder 启用时计算（SBM 无外部特征时自动跳过）
        if self.dual_encoder:
            H_f = F.relu(self.fc1_feat(X))
            Z_f = self.fc2_feat(H_f)
            # 加法融合：让 Z 同时编码图结构和特征语义
            Z = Z_g + Z_f
        else:
            Z = Z_g
        logits = self.comm_head(Z) / math.sqrt(self.comm_head.in_features)
        Q = F.softmax(logits, dim=-1)
        # 特征重建（可选）：从 Z 重建 X，强制 Z 编码特征信息
        X_rec = self.feat_decoder(Z) if self.feat_decoder is not None else None
        return Z, Q, logits, X_rec


# ----------------------------- Losses -----------------------------
def recon_bce(Z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Inner-product decoder, pos-weighted BCE for sparse graphs."""
    logits = Z @ Z.t()
    n_pos = A.sum().clamp(min=1.0)
    n_neg = A.numel() - n_pos
    # pos_weight 跟 logits 同设备，避免 GPU/CPU 跨设备报错
    pos_weight = torch.full((), (n_neg / n_pos).item(),
                            dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, A, pos_weight=pos_weight)


def size_entropy(Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """S = -sum_k p_k log p_k,  p_k = mean_i q_ik.
    Leading-order Stirling of ln W = N! / prod_k n_k!."""
    p = Q.mean(dim=0)                       # (K,)
    return -(p * torch.log(p + eps)).sum()


def assign_entropy(Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """S = mean_i [ -sum_k q_ik log q_ik ]. Free-energy variant."""
    return -(Q * torch.log(Q + eps)).sum(dim=-1).mean()


def method2_lnw(Z: torch.Tensor, Q: torch.Tensor, C: torch.Tensor,
                sigma: float, eps: float = 1e-8) -> torch.Tensor:
    """逐维度玻尔兹曼微观态计数（每维独立 1D soft binning + Stirling 近似）。

    对每个维度 j 独立计算 1D 玻尔兹曼熵，总 lnW = 各维 lnW 之平均：
      对维度 j：
        w_ib_j = softmax_b(-|z_i[j] - c_b[j]|^2 / 2 sigma^2)   (1D 距离)
        n_kb_j = Σ_i q_ik * w_ib_j                              (社区 k 在 bin b 的计数)
        p_kb_j = n_kb_j / n_k_j
        lnW_j = -Σ_{k,b} n_kb_j * log p_kb_j
      总 lnW = (1/d) Σ_j lnW_j    ← 除以 d 归一化，消除尺度膨胀

    关键设计：
      1. 除以 d 归一化：逐维求和会让 lnW 膨胀 d 倍，压倒重建项 E。
         除以 d 后 lnW 量级与原 d 维整体计算一致，T 和 sigma 无需大幅调整。
      2. sigma 是 1D 语义：1D 距离 (z[j]-c[j])² 的典型量级是 σ_z²，
         而 d 维距离 ||z-c||² 的量级是 d·σ_z²。要达到相同的 softmax 柔和度，
         σ_1d = σ_d / sqrt(d)。TrainConfig.sigma 默认 0.125（=0.5/√16）。
      3. bin 中心每维 L2 归一化：匹配 Z 的单维典型尺度。

    Q 仍由 EntropyGNN.forward 的 comm_head(Z) 整体计算（不逐维分解），
    保证社区分配与嵌入整体结构一致。

    Args:
        Z: 嵌入矩阵 (N, d)
        Q: 社区分配矩阵 (N, K)，由 EntropyGNN.forward 产生（每行和为 1）
        C: 逐维 bin 中心 (d, M)，固定 1D 锚点（每维 L2 归一化）
        sigma: 1D RBF 带宽（注意：1D 语义，非 d 维语义）
        eps: 数值稳定地板

    Returns:
        lnW（标量 tensor，未归一化）；caller 除以 N 得 per-node 尺度
    """
    d = Z.shape[1]
    # 逐维 1D 距离: (N, d, M)
    # Z[:,j] 是第 j 维坐标 (N,)，C[j,:] 是第 j 维的 M 个 bin 中心 (M,)
    Z_exp = Z.unsqueeze(2)              # (N, d, 1)
    C_exp = C.unsqueeze(0)              # (1, d, M)
    dist2 = (Z_exp - C_exp) ** 2        # (N, d, M) 1D 平方距离

    # 逐维 soft binning: (N, d, M)
    Wb = F.softmax(-dist2 / (2.0 * sigma ** 2), dim=-1)

    # 社区计数: Nk[k, j, b] = Σ_i q_ik * w_ib_j  → (K, d, M)
    Nk = torch.einsum('ik,ijb->kjb', Q, Wb)

    # 归一化: p_kb = n_kb / n_k（每社区每维独立归一化）
    n_k = Nk.sum(dim=2, keepdim=True)   # (K, d, 1)
    p_kb = Nk / (n_k + eps)             # (K, d, M)

    # 玻尔兹曼熵: 对所有维度、社区、bin 求和，然后除以 d 归一化
    # 除以 d 让 lnW 量级与原 d 维整体计算一致，避免熵项压倒重建项
    lnW = -(Nk * torch.log(p_kb + eps)).sum() / d
    return lnW


def free_energy(Z, A, Q, T, entropy: str,
                C: torch.Tensor = None, sigma: float = 1.0):
    """计算自由能 F = E - T·S。

    支持的 entropy 类型：
      - "none"    : 纯重建（vanilla baseline）
      - "size"    : 社区大小熵（Stirling 首项）
      - "assign"  : 节点分配熵
      - "method2" : 玻尔兹曼微观态计数 ln W（Stirling + soft binning），
                    在嵌入空间上计算，驱动社区向最大微观状态数宏观态演化。
                    使用 caller 传入的 Q（模型 forward 输出），保证与社区头一致。

    Args:
        Z: 嵌入矩阵 (N, d)
        A: 邻接矩阵 (N, N)
        Q: 社区分配矩阵 (N, K)（model forward 输出）
        T: 温度
        entropy: 熵类型（见上）
        C: bin 中心（method2 用）
        sigma: RBF 带宽（method2 用）
    """
    E = recon_bce(Z, A)
    if entropy == "none" or T == 0.0:
        return E, E, torch.zeros((), device=Q.device)
    if entropy == "size":
        S = size_entropy(Q)
    elif entropy == "assign":
        S = assign_entropy(Q)
    elif entropy == "method2":
        lnW = method2_lnw(Z, Q, C, sigma)
        S = lnW / Z.shape[0]          # per-node entropy, scale-stable
    else:
        raise ValueError(entropy)
    return E - T * S, E, S
