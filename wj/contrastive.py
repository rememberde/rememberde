"""contrastive.py — 对比学习损失（结构 CL + 特征 CL + 混合 CL）。

三种对比模式：
  1. 结构 CL（graph_contrastive_loss）：用邻接矩阵定义正/负样本
     - 适用于强社区图（CC≥0.20，如 Cora），图结构可靠
  2. 特征 CL（feature_contrastive_loss）：用特征 KNN 定义正/负样本
     - 适用于弱社区图（CC<0.20，如 CiteSeer），特征比图结构更可靠
     - 解决了结构 CL 在弱社区图上放大噪声的问题
     - 注意：高维稀疏特征（如 CiteSeer 3703 维 BoW）会因维度灾难
       导致 KNN 不可靠，建议先用 PCA 降维（见 training.py 的 pca_dim 参数）
  3. 混合 CL（combined_contrastive_loss）：结构 CL + 特征 CL 加权平均
     - 适用于弱社区图但不希望完全丢弃图结构信息的场景
     - 实验表明：纯特征 CL 在 CiteSeer 上让 NMI↑但 ARI↓（聚类边界与标签
       不对齐），混合 CL 保留部分图结构信号可缓解此问题

大图优化：N>n_anchors 时采样锚节点子集，避免 O(N²) 显存爆炸。
"""
import torch
import torch.nn.functional as F


def graph_contrastive_loss(Z: torch.Tensor, A: torch.Tensor,
                           tau: float = 0.5,
                           n_anchors: int = 2000) -> torch.Tensor:
    """基于图结构的对比学习损失（InfoNCE）。

    正样本 = 邻居节点，负样本 = 非邻居节点。
    适用于强社区图（CC≥0.20）。

    Args:
        Z: 节点嵌入 (N, d)
        A: 邻接矩阵 (N, N)
        tau: 温度参数
        n_anchors: 大图采样锚节点数
    """
    N = Z.shape[0]
    Z_norm = F.normalize(Z, dim=1)

    if N > n_anchors:
        anchor_idx = torch.randperm(N, device=Z.device)[:n_anchors]
    else:
        anchor_idx = torch.arange(N, device=Z.device)

    sim = Z_norm[anchor_idx] @ Z_norm.t() / tau  # (n_anchors, N)

    # 正样本掩码：邻居，排除自环（向量化，不用 for 循环）
    pos_mask = A[anchor_idx].bool().clone()
    row_idx = torch.arange(len(anchor_idx), device=Z.device)
    pos_mask[row_idx, anchor_idx] = False

    # InfoNCE（数值稳定）
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - sim_max)

    pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
    self_contrib = exp_sim[row_idx, anchor_idx]
    all_sum = exp_sim.sum(dim=1) - self_contrib

    has_pos = pos_mask.any(dim=1)
    if has_pos.any():
        loss = -torch.log(pos_sum[has_pos] / (all_sum[has_pos] + 1e-8) + 1e-8)
        return loss.mean()
    return torch.zeros((), device=Z.device)


def feature_contrastive_loss(Z: torch.Tensor, X: torch.Tensor,
                             tau: float = 0.5, k_neighbors: int = 10,
                             n_anchors: int = 2000) -> torch.Tensor:
    """基于特征 KNN 的对比学习损失（InfoNCE）。

    正样本 = 特征空间中的 K 近邻，负样本 = 特征远距节点。
    适用于弱社区图（CC<0.20，如 CiteSeer）——特征比图结构更可靠，
    避免了结构 CL 放大图结构噪声的问题。

    Args:
        Z: 节点嵌入 (N, d)
        X: 节点特征 (N, F)，用于定义正/负样本（不参与梯度）
        tau: 温度参数
        k_neighbors: 每个锚节点的正样本数（特征 KNN 的 K）
        n_anchors: 大图采样锚节点数
    """
    N = Z.shape[0]
    Z_norm = F.normalize(Z, dim=1)
    # 特征归一化（不参与梯度，纯用于定义正/负样本）
    with torch.no_grad():
        X_norm = F.normalize(X, dim=1)

    if N > n_anchors:
        anchor_idx = torch.randperm(N, device=Z.device)[:n_anchors]
    else:
        anchor_idx = torch.arange(N, device=Z.device)

    # 在特征空间找 K 近邻作为正样本（不参与梯度）
    with torch.no_grad():
        feat_sim = X_norm[anchor_idx] @ X_norm.t()  # (n_anchors, N)
        row_idx = torch.arange(len(anchor_idx), device=Z.device)
        feat_sim[row_idx, anchor_idx] = -2.0  # 排除自身
        # Top-K 近邻
        _, topk_idx = feat_sim.topk(min(k_neighbors, N - 1), dim=1)
        pos_mask = torch.zeros(len(anchor_idx), N, device=Z.device)
        pos_mask.scatter_(1, topk_idx, 1.0)

    # 嵌入空间相似度（参与梯度）
    sim = Z_norm[anchor_idx] @ Z_norm.t() / tau  # (n_anchors, N)

    # InfoNCE（数值稳定）
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - sim_max)

    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    self_contrib = exp_sim[row_idx, anchor_idx]
    all_sum = exp_sim.sum(dim=1) - self_contrib

    has_pos = pos_mask.any(dim=1)
    if has_pos.any():
        loss = -torch.log(pos_sum[has_pos] / (all_sum[has_pos] + 1e-8) + 1e-8)
        return loss.mean()
    return torch.zeros((), device=Z.device)


def combined_contrastive_loss(Z: torch.Tensor, A: torch.Tensor, X: torch.Tensor,
                              tau: float = 0.5, k_neighbors: int = 10,
                              n_anchors: int = 2000,
                              feat_weight: float = 0.5) -> torch.Tensor:
    """混合对比学习损失（结构 CL + 特征 CL 加权平均）。

    L = (1 - w) * L_graph + w * L_feature

    设计动机：纯特征 CL 在弱社区图（如 CiteSeer）上会让 NMI↑但 ARI↓
    （特征 KNN 聚类边界与真实标签不对齐）。混合 CL 保留部分图结构信号，
    既利用特征信息弥补弱社区结构，又保留图结构的社区指向性。

    Args:
        Z: 节点嵌入 (N, d)
        A: 邻接矩阵 (N, N)
        X: 节点特征 (N, F)，用于特征 CL（建议先 PCA 降维）
        tau: 温度参数（两种 CL 共用）
        k_neighbors: 特征 CL 的 K 近邻数
        n_anchors: 大图采样锚节点数
        feat_weight: 特征 CL 权重 w ∈ [0, 1]
            - w=0：纯结构 CL（等同 graph_contrastive_loss）
            - w=1：纯特征 CL（等同 feature_contrastive_loss）
            - w=0.5（默认）：两种各半
    """
    loss_graph = graph_contrastive_loss(Z, A, tau=tau, n_anchors=n_anchors)
    loss_feat = feature_contrastive_loss(Z, X, tau=tau,
                                         k_neighbors=k_neighbors,
                                         n_anchors=n_anchors)
    return (1.0 - feat_weight) * loss_graph + feat_weight * loss_feat
