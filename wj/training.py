"""training.py — 训练配置 + 温度调度 + 单次训练循环。

职责：训练流程的唯一定义点。
  - TrainConfig：训练超参 dataclass（社区数、维度、epochs、lr、温度退火、反塌缩参数）
  - schedule_T：温度退火（warmup → cosine/linear/constant 衰减）
  - train_one：单次训练循环（前向 → 自由能 → 反塌缩铰链 → 反向 → KMeans 评估）
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score

from .runtime import DEVICE, set_seed
from .data import normalize_adj
from .model import EntropyGNN, free_energy
from .evaluate_metrics import modularity, kmeans_labels, size_balance
from .anticollapse import VarianceHinge, compute_collapse_metrics
from .contrastive import (
    graph_contrastive_loss, feature_contrastive_loss, combined_contrastive_loss,
)


# ----------------------------- Training -----------------------------
@dataclass
class TrainConfig:
    n_communities: int = 4
    hidden_dim: int = 64
    emb_dim: int = 16
    epochs: int = 300
    lr: float = 0.01
    T_max: float = 0.5
    T_warmup: float = 0.2          # fraction of epochs to ramp up
    anneal: str = "cosine"         # "constant" | "linear" | "cosine"
    entropy: str = "size"          # "none" | "size" | "assign" | "method2"
    n_bins: int = 16               # Method-2 bin count
    sigma: float = 0.5             # Method-2 RBF bandwidth
    lambda_rank: float = 0.0       # anti-collapse penalty weight (0=off)
    rank_min_rank: float = 2.0     # hinge: only penalise eff_rank below this
    rank_min_var: float = 0.5      # hinge: only penalise tr(S) below this
    lambda_contrast: float = 0.0   # 对比学习损失权重（0=关闭，参考 DCRN）
    contrast_tau: float = 0.5      # 对比学习温度参数
    contrast_mode: str = "structure"  # "structure"|"feature"|"combined"
    lambda_feat_recon: float = 0.0 # 特征重建损失权重（0=关闭，弱社区图用）
    pca_dim: int = 0               # 特征 CL 前的 PCA 降维目标维度（0=不降维）
                                   # 高维稀疏特征（如 CiteSeer 3703 维 BoW）
                                   # 因维度灾难导致 KNN 不可靠，建议降到 ~200
    cl_feat_weight: float = 0.5    # combined CL 中特征 CL 权重 w ∈ [0,1]
                                   # 仅 contrast_mode="combined" 时生效
    eval_every: int = 10           # run KMeans metrics every N epochs (speed)
    seed: int = 42


def schedule_T(epoch: int, epochs: int, cfg: TrainConfig) -> float:
    if cfg.entropy == "none":
        return 0.0
    t = epoch / max(epochs - 1, 1)
    if t < cfg.T_warmup:
        return cfg.T_max * (t / max(cfg.T_warmup, 1e-6))
    s = (t - cfg.T_warmup) / max(1 - cfg.T_warmup, 1e-6)
    if cfg.anneal == "constant":
        return cfg.T_max
    if cfg.anneal == "linear":
        return cfg.T_max * (1 - s)
    # cosine decay to 0
    return cfg.T_max * 0.5 * (1 + math.cos(math.pi * s))


def train_one(A_np: np.ndarray, labels: np.ndarray, cfg: TrainConfig,
              X_feat: np.ndarray = None, verbose_every: int = 0):
    """训练单次实验。

    Args:
        A_np: 邻接矩阵 (N, N) numpy
        labels: 真实社区标签 (N,) numpy
        cfg: TrainConfig 训练配置
        X_feat: 节点特征 (N, F) numpy，None 时用 one-hot I_N（SBM 路径）
        verbose_every: 每 N epoch 打印 hinge 状态，0 = 静默
    """
    set_seed(cfg.seed)
    n = A_np.shape[0]
    A = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)
    A_hat = normalize_adj(A_np)
    # 外部特征（Cora 等）优先；SBM 默认 None，模型内部用 one-hot
    # 注意：不做 L1 行归一化。实测 L1 归一化会让 GCN 第一层输出尺度从 0/1
    # 降到 ~1/avg_doc_len，导致嵌入塌缩（eff_rank→1, vanilla NMI 0.48→0.14）。
    # GCN 的 D^{-1/2}AD^{-1/2} 已处理度的问题，无需额外行归一化。
    if X_feat is not None:
        X_tensor = torch.tensor(X_feat, dtype=torch.float32, device=DEVICE)
        node_feat_dim = X_feat.shape[1]
    else:
        X_tensor = None
        node_feat_dim = None
    # PCA 降维：高维稀疏特征（如 CiteSeer 3703 维 BoW）因维度灾难导致
    # 特征 KNN 不可靠，在特征 CL 前降到低维（默认 ~200）让 KNN 更稳定。
    # 仅用于 CL 的正/负样本定义，不影响模型输入（模型仍用原始 X）。
    # 用 torch.pca_lowrank 一次性预计算，每个 epoch 复用结果。
    X_for_cl = X_tensor
    if (cfg.pca_dim > 0 and X_tensor is not None
            and X_tensor.shape[1] > cfg.pca_dim):
        with torch.no_grad():
            # pca_lowrank: X = U @ diag(S) @ V^T，取前 pca_dim 列投影
            U, S_pca, Vh = torch.pca_lowrank(X_tensor, q=cfg.pca_dim)
            X_for_cl = U @ torch.diag(S_pca)  # (N, pca_dim) 降维后特征
    model = EntropyGNN(n, cfg.hidden_dim, cfg.emb_dim, cfg.n_communities,
                       n_bins=cfg.n_bins, node_feature_dim=node_feat_dim,
                       feat_recon=cfg.lambda_feat_recon > 0.0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Build the anti-collapse hinge from cfg; reuse the same instance every
    # epoch so diagnostics stay consistent. When lambda_rank == 0 the hinge
    # is still constructed (cheap) but never applied to the loss.
    hinge = VarianceHinge(min_rank=cfg.rank_min_rank, min_var=cfg.rank_min_var)

    keys = ('loss', 'E', 'S', 'T', 'mod', 'nmi', 'bal',
            'embed_std', 'eff_rank', 'q_entropy', 'q_commit', 'rank_pen',
            'trS', 'rank_h', 'var_h')
    hist = {k: [] for k in keys}
    for epoch in range(cfg.epochs):
        T = schedule_T(epoch, cfg.epochs, cfg)
        opt.zero_grad()
        Z, Q, _, X_rec = model(A_hat, X_tensor)
        loss, E, S = free_energy(Z, A, Q, T, cfg.entropy,
                                 C=model.bin_centers,
                                 sigma=cfg.sigma)
        rpen = torch.zeros((), device=Z.device)
        if cfg.lambda_rank > 0.0:
            rpen = hinge.penalty(Z)
            loss = loss + cfg.lambda_rank * rpen
        # 对比学习损失：自适应选择结构 CL / 特征 CL / 混合 CL
        # - 结构 CL（强社区图）：用邻接矩阵定义正/负样本
        # - 特征 CL（弱社区图）：用特征 KNN 定义正/负样本，避免放大图结构噪声
        #   ※ X_for_cl 是经过 PCA 降维的特征（若 cfg.pca_dim > 0）
        # - 混合 CL（弱社区图 + 保留图结构信号）：结构 CL + 特征 CL 加权平均
        #   解决纯特征 CL 在 CiteSeer 上 NMI↑但 ARI↓（聚类边界与标签不对齐）问题
        if cfg.lambda_contrast > 0.0:
            if cfg.contrast_mode == "feature" and X_for_cl is not None:
                cl_loss = feature_contrastive_loss(Z, X_for_cl, tau=cfg.contrast_tau)
            elif cfg.contrast_mode == "combined" and X_for_cl is not None:
                cl_loss = combined_contrastive_loss(
                    Z, A, X_for_cl, tau=cfg.contrast_tau,
                    feat_weight=cfg.cl_feat_weight,
                )
            else:
                cl_loss = graph_contrastive_loss(Z, A, tau=cfg.contrast_tau)
            loss = loss + cfg.lambda_contrast * cl_loss
        # 特征重建损失：从 Z 重建 X，强制嵌入编码特征信息（弱社区图关键）
        if cfg.lambda_feat_recon > 0.0 and X_rec is not None:
            feat_loss = F.mse_loss(X_rec, X_tensor if X_tensor is not None
                                   else model._default_X.to(A_hat.device))
            loss = loss + cfg.lambda_feat_recon * feat_loss
        loss.backward()
        # 梯度裁剪：method2 的 lnW 在早期（bin 距离大、softmax 接近 argmax）
        # 梯度可能很大，导致训练不稳定。裁剪到 max_norm=5.0 防止爆炸。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        # 评估与诊断（KMeans 是昂贵步骤，只在 eval_every 间隔 + 最终 epoch 做）
        # 合并 verbose 和 eval 的诊断调用：只调一次 compute_collapse_metrics，
        # 从返回的 eff_rank/tr_S 手动算铰链分量，避免再调 hinge.diagnostics
        # （hinge.diagnostics 内部会重复计算 _covariance(Z)，合并后从 3 次降到 1 次）
        is_verbose = verbose_every > 0 and (epoch % verbose_every == 0 or epoch == cfg.epochs - 1)
        do_eval = (epoch % cfg.eval_every == 0) or (epoch == cfg.epochs - 1)

        # 1) 只需 verbose 打印、不需 eval（不跑 KMeans）：单独算 hinge 诊断（成本低）
        if is_verbose and not do_eval:
            d = hinge.diagnostics(Z)
            print(f"  ep {epoch:3d}  T={T:.4f}  tr(S)={d.tr_S:.4f}  "
                  f"eff_rank={d.eff_rank:.2f}  rank_h={d.rank_def:.3f}  "
                  f"var_h={d.var_def:.3f}  loss={float(loss):.4f}")
            continue

        # 2) 只需 eval、不需 verbose：算完整诊断但不打印
        # 3) verbose + eval 同时发生：算一次完整诊断，复用于打印和 hist
        if not do_eval:
            continue

        with torch.no_grad():
            Z_np = Z.cpu().numpy()
            pred = kmeans_labels(Z_np, cfg.n_communities, seed=cfg.seed)
            mod = modularity(A_np, pred)
            nmi = normalized_mutual_info_score(labels, pred)
            bal = size_balance(pred)
            # 一次 compute_collapse_metrics 获取所有塌缩指标（内部 1 次 _covariance）
            cm = compute_collapse_metrics(Z, Q)
            # 从 cm 的 eff_rank/tr_S 手动算铰链分量，复用已算好的协方差
            # （不再调 hinge.diagnostics，省掉 1 次 _covariance）
            rank_def = max(0.0, hinge.min_rank - cm['eff_rank']) / max(hinge.min_rank, 1.0)
            var_def = max(0.0, hinge.min_var - cm['tr_S']) / max(hinge.min_var, hinge.eps)

            if is_verbose:
                print(f"  ep {epoch:3d}  T={T:.4f}  tr(S)={cm['tr_S']:.4f}  "
                      f"eff_rank={cm['eff_rank']:.2f}  rank_h={rank_def:.3f}  "
                      f"var_h={var_def:.3f}  loss={float(loss):.4f}")

            hist['loss'].append(float(loss)); hist['E'].append(float(E))
            hist['S'].append(float(S)); hist['T'].append(float(T))
            hist['mod'].append(mod); hist['nmi'].append(nmi); hist['bal'].append(bal)
            hist['rank_pen'].append(float(rpen))
            hist['trS'].append(cm['tr_S'])
            hist['rank_h'].append(rank_def)
            hist['var_h'].append(var_def)
            hist['embed_std'].append(cm['embed_std'])
            hist['eff_rank'].append(cm['eff_rank'])
            hist['q_entropy'].append(cm['q_entropy'])
            hist['q_commit'].append(cm['q_commit'])
    return model, hist
