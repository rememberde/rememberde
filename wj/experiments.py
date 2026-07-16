"""experiments.py — 实验配置生成 + 运行器 + 绘图。

职责：实验编排和可视化的唯一定义点。
  - make_configs：生成 7 个对比配置（vanilla/size/assign/method2/m2_rank1/2/3）
  - 实验运行器：run_multi_seed / run_imbalanced / run_dataset / run_difficulty_sweep / run_single_trace
  - 绘图：plot_sweep / plot_curves / plot_imbalanced_scatter
  - 图片输出路径：IMAGE_DIR + _fig_path
"""
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

# matplotlib 配置（Agg 后端 + 中文字体）统一走 plot_config
from . import plot_config

from .training import TrainConfig, train_one
from .data import (
    make_sbm, make_imbalanced_sbm, DIFFICULTIES,
    IMBALANCED_SIZES, IMBALANCED_P, IMBALANCED_TRUE_SIZECV,
    DATASET_LOADERS,
)


# ----------------------------- 图片输出路径 -----------------------------
# 所有实验图片统一输出到 image/ 目录，避免散落在根目录
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image")
os.makedirs(IMAGE_DIR, exist_ok=True)


def _fig_path(name: str) -> str:
    """统一图片输出路径，所有 savefig 都走这里。"""
    return os.path.join(IMAGE_DIR, name)


# ----------------------------- 配置生成 -----------------------------
def make_configs(K: int, seed: int, epochs: int = 300,
                 min_rank: float = 3.0, emb_dim: int = 16) -> Dict[str, TrainConfig]:
    """生成 7 个对比配置。K=社区数，min_rank=秩铰链地板，emb_dim=嵌入维度。

    经验法则：min_rank ≈ K/2 ~ K。K=4 SBM 用 3.0，K=7 Cora 用 5.0。
    emb_dim：SBM 用 16 足够；Cora 等真实图用 32 给更大表达空间。

    7 个 config：
      - vanilla          : 纯重建（baseline）
      - size / assign    : 社区熵变体（Stirling 首项 / per-node 分配熵）
      - method2          : 玻尔兹曼 ln W 最大化（无反塌缩保护，大图适用）
      - m2_rank1/2/3     : method2 + 双铰链反塌缩（λ=1.0/2.0/3.0，小图适用）
    """
    common = dict(n_communities=K, hidden_dim=64, emb_dim=emb_dim,
                  epochs=epochs, lr=0.01, seed=seed)
    # min_var 按 emb_dim 缩放：tr(S)=Σ Var(Z[:,d]) ≈ d·embStd²，
    # 所以固定 min_var=1.0 对 d=16→embStd 阈 0.25，对 d=64→0.125，语义漂移。
    # 用 min_var = 0.0625 * emb_dim 等价 embStd 地板 ~0.25 恒定，三 config 一致。
    min_var = 0.0625 * emb_dim
    return {
        "vanilla": TrainConfig(**common, entropy="none"),
        "size":    TrainConfig(**common, entropy="size",
                               T_max=0.5, anneal="cosine", T_warmup=0.2),
        "assign":  TrainConfig(**common, entropy="assign",
                               T_max=0.3, anneal="cosine", T_warmup=0.2),
        "method2": TrainConfig(**common, entropy="method2",
                               T_max=0.3, anneal="cosine", T_warmup=0.2,
                               n_bins=16, sigma=0.5),
        # Method 2 + anti-collapse regularisation: hinge on eff_rank AND tr(S).
        # min_rank 按经验法则 ≈ K/2~K 设置；min_var 按 emb_dim 缩放（见上）
        # 健康时休眠，塌缩时触发。三个 lambda 强度看梯度效果。
        "m2_rank1": TrainConfig(**common, entropy="method2",
                                T_max=0.3, anneal="cosine", T_warmup=0.2,
                                n_bins=16, sigma=0.5, lambda_rank=1.0,
                                rank_min_rank=min_rank, rank_min_var=min_var),
        "m2_rank2": TrainConfig(**common, entropy="method2",
                                T_max=0.3, anneal="cosine", T_warmup=0.2,
                                n_bins=16, sigma=0.5, lambda_rank=2.0,
                                rank_min_rank=min_rank, rank_min_var=min_var),
        "m2_rank3": TrainConfig(**common, entropy="method2",
                                T_max=0.3, anneal="cosine", T_warmup=0.2,
                                n_bins=16, sigma=0.5, lambda_rank=3.0,
                                rank_min_rank=min_rank, rank_min_var=min_var),
        # Method 2 + 对比学习 + 特征重建（无 hinge）：
        # - 默认用特征 CL（特征 KNN 定义正/负样本），适用于弱社区图
        # - run_wj.py 会根据 CC + 特征维度自适应切换：
        #     CC≥0.20 → 结构 CL
        #     CC<0.20 且 F>500 → 混合 CL + PCA 降维（高维稀疏特征，如 CiteSeer）
        #     CC<0.20 且 F≤500 → 纯特征 CL（低维特征，如 PubMed）
        # - 特征重建强制 Z 编码特征信息，弥补弱社区图结构不足
        "m2_cl":   TrainConfig(**common, entropy="method2",
                               T_max=0.3, anneal="cosine", T_warmup=0.2,
                               n_bins=16, sigma=0.5,
                               lambda_contrast=0.1, contrast_tau=0.5,
                               contrast_mode="feature",
                               lambda_feat_recon=0.1),
        # Method 2 + hinge + 结构对比学习（强社区图专用，合并 m2_rank3 和 m2_cl）：
        # - 在 m2_rank3 基础上加结构 CL，同时享受 hinge 防塌缩 + CL 增强社区边界
        # - 动机：Cora 上 m2_rank3 的 ACC/NMI 已接近 SOTA，但 ARI 落后 0.06
        #   原因是聚类边界不够精确，结构 CL 可强化同社区节点的嵌入聚合
        # - run_wj.py 会自适应：强社区图(CC≥0.20) 用 hinge+结构CL，
        #   弱社区图关闭 hinge（λ=0），CL 模式同 m2_cl
        "m2_rank3_cl": TrainConfig(**common, entropy="method2",
                                   T_max=0.3, anneal="cosine", T_warmup=0.2,
                                   n_bins=16, sigma=0.5, lambda_rank=3.0,
                                   rank_min_rank=min_rank, rank_min_var=min_var,
                                   lambda_contrast=0.1, contrast_tau=0.5,
                                   contrast_mode="structure"),
    }


METRIC_KEYS = ('mod', 'nmi', 'bal', 'embed_std', 'eff_rank', 'q_commit')


# ----------------------------- 实验运行器 -----------------------------
def run_multi_seed(difficulty: str, n_seeds: int = 5):
    n_per, n_blk, p_in, p_out = DIFFICULTIES[difficulty]
    print(f"\n##### Difficulty: {difficulty}  "
          f"(n_per_block={n_per}, blocks={n_blk}, p_in={p_in}, p_out={p_out}) #####")
    K = n_blk
    all_metrics = {name: {k: [] for k in METRIC_KEYS}
                   for name in make_configs(K, 0)}
    for s in range(n_seeds):
        A_np, labels, _ = make_sbm(n_per_block=n_per, n_blocks=n_blk,
                                   p_in=p_in, p_out=p_out, seed=s)
        for name, cfg in make_configs(K, s).items():
            _, h = train_one(A_np, labels, cfg)
            for k in METRIC_KEYS:
                all_metrics[name][k].append(h[k][-1])
    _print_metrics(all_metrics)
    return all_metrics


def run_imbalanced(n_seeds: int = 5):
    sizes = IMBALANCED_SIZES
    p_in, p_out = IMBALANCED_P
    print(f"\n##### Imbalanced SBM  (sizes={sizes}, p_in={p_in}, p_out={p_out}) #####")
    K = len(sizes)
    all_metrics = {name: {k: [] for k in METRIC_KEYS}
                   for name in make_configs(K, 0)}
    for s in range(n_seeds):
        A_np, labels, _ = make_imbalanced_sbm(sizes, p_in=p_in, p_out=p_out, seed=s)
        for name, cfg in make_configs(K, s).items():
            _, h = train_one(A_np, labels, cfg)
            for k in METRIC_KEYS:
                all_metrics[name][k].append(h[k][-1])
    _print_metrics(all_metrics)
    return all_metrics


def run_dataset(name: str, n_seeds: int = 3, epochs: int = 200,
                min_rank: float = None, emb_dim: int = None):
    """在真实数据集上跑实验（通用接口）。

    Args:
        name: 'cora' | 'citeseer' | 'pubmed'
        n_seeds: 随机种子数
        epochs: 训练轮数
        min_rank: 秩铰链地板，None 时用数据集默认值
        emb_dim: 嵌入维度，None 时用数据集默认值
    """
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}")
    loader, K, default_emb, default_rank = DATASET_LOADERS[name]
    if min_rank is None:
        min_rank = default_rank
    if emb_dim is None:
        emb_dim = default_emb
    print(f"\n##### {name.upper()} dataset  (K={K}, {n_seeds} seeds, "
          f"epochs={epochs}, min_rank={min_rank}, emb_dim={emb_dim}) #####")
    A_np, labels, features = loader()
    n_nodes = A_np.shape[0]
    # 大图（N>5000）降低 KMeans 频率：每次 KMeans 在 19717 节点的 Pubmed 上很慢，
    # eval_every 从 10 → 20 让 KMeans 调用次数减半，总训练时间约减 40%
    adaptive_eval_every = 20 if n_nodes > 5000 else 10
    print(f"  N={n_nodes}, eval_every={adaptive_eval_every} (大图降频以加速)")
    all_metrics = {name: {k: [] for k in METRIC_KEYS}
                   for name in make_configs(K, 0, emb_dim=emb_dim)}
    for s in range(n_seeds):
        print(f"  --- seed {s} ---")
        for cfg_name, cfg in make_configs(K, s, epochs=epochs,
                                           min_rank=min_rank,
                                           emb_dim=emb_dim).items():
            cfg.eval_every = adaptive_eval_every  # 覆盖默认值
            _, h = train_one(A_np, labels, cfg, X_feat=features)
            for k in METRIC_KEYS:
                all_metrics[cfg_name][k].append(h[k][-1])
            print(f"    {cfg_name:<10} NMI={h['nmi'][-1]:.4f}  "
                  f"Mod={h['mod'][-1]:.4f}  SizeCV={h['bal'][-1]:.4f}  "
                  f"effRank={h['eff_rank'][-1]:.2f}  embStd={h['embed_std'][-1]:.4f}")
    _print_metrics(all_metrics)
    return all_metrics


def _print_metrics(all_metrics):
    print(f"\n{'config':<10} {'Modularity':>16} {'NMI':>16} {'SizeCV':>16} "
          f"{'embStd':>14} {'effRank':>14} {'qCommit':>14}")
    for name, m in all_metrics.items():
        def fmt(k):
            v = np.array(m[k]); return f"{v.mean():>8.4f}+/-{v.std():.4f}"
        print(f"{name:<10} {fmt('mod')} {fmt('nmi')} {fmt('bal')} "
              f"{fmt('embed_std')} {fmt('eff_rank')} {fmt('q_commit')}")


def run_difficulty_sweep(n_seeds: int = 5):
    results = {}
    for diff in DIFFICULTIES:
        results[diff] = run_multi_seed(diff, n_seeds=n_seeds)
    results["imbalanced"] = run_imbalanced(n_seeds=n_seeds)
    plot_sweep(results)
    return results


def run_single_trace(seed: int = 0, difficulty: str = "hard"):
    """One detailed run with full training curves, for visualization."""
    n_per, n_blk, p_in, p_out = DIFFICULTIES[difficulty]
    A_np, labels, _ = make_sbm(n_per_block=n_per, n_blocks=n_blk,
                               p_in=p_in, p_out=p_out, seed=seed)
    K = n_blk
    hists = {}
    for name, cfg in make_configs(K, seed).items():
        print(f"[{name}] training ...")
        _, hists[name] = train_one(A_np, labels, cfg)
        print(f"    final  Mod={hists[name]['mod'][-1]:.4f}  "
              f"NMI={hists[name]['nmi'][-1]:.4f}  "
              f"SizeCV={hists[name]['bal'][-1]:.4f}  "
              f"effRank={hists[name]['eff_rank'][-1]:.2f}  "
              f"qCommit={hists[name]['q_commit'][-1]:.3f}")
    plot_curves(hists)
    return hists


# ----------------------------- 绘图 -----------------------------
def plot_sweep(results):
    diffs = list(results.keys())
    configs = list(next(iter(results.values())).keys())
    # plot the core community-quality + collapse metrics
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [('nmi', 'NMI vs GT (higher=better)', True),
              ('mod', 'Modularity (higher=better)', True),
              ('bal', 'Size CV (lower=more balanced)', False),
              ('eff_rank', 'Effective rank of Z (low=collapse)', True),
              ('embed_std', 'Embedding std (low=collapse)', True),
              ('q_commit', 'Mean max q_ik (low=no commit)', True)]
    for ax, (metric, title, higher_better) in zip(axes.flat, panels):
        x = np.arange(len(diffs))
        for cfg_name in configs:
            means = [np.mean(results[d][cfg_name][metric]) for d in diffs]
            stds = [np.std(results[d][cfg_name][metric]) for d in diffs]
            ax.errorbar(x, means, yerr=stds, marker='o', label=cfg_name, capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(diffs, rotation=15)
        ax.set_xlabel('regime'); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    out = _fig_path("baseline_results.png")
    plt.savefig(out, dpi=120)
    print(f"\nSaved figure -> {out}")


def plot_curves(hists: Dict[str, Dict]):
    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    panels = [('loss', 'Free Energy F = E - T*S'),
              ('mod', 'Modularity (KMeans on Z)'),
              ('nmi', 'NMI vs ground truth'),
              ('T', 'Temperature T(t)'),
              ('rank_pen', 'Rank penalty (lower=more collapse)'),
              ('E', 'Reconstruction E'),
              ('S', 'Community Entropy S'),
              ('eff_rank', 'Effective rank (collapse check)'),
              ('q_commit', 'Community commitment (collapse check)'),
              ('embed_std', 'Embedding std (collapse check)')]
    for ax, (key, title) in zip(axes.flat, panels):
        for name, h in hists.items():
            ax.plot(h[key], label=name, alpha=0.85)
        ax.set_title(title); ax.set_xlabel('epoch')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    out = _fig_path("baseline_curves.png")
    plt.savefig(out, dpi=120)
    print(f"\nSaved figure -> {out}")


def plot_imbalanced_scatter(results):
    """Intuitive visualisation of imbalanced SBM: SizeCV vs NMI scatter.

    Each config is a point; the vertical dashed line marks the TRUE SizeCV
    of the SBM ([80,50,30,15] -> 0.596).  Configs that recover the true
    imbalance land near the line; higher NMI is better.  This makes it
    visually obvious that m2_rank lands closest to the true structure AND
    achieves the highest NMI, while vanilla/size/assign over-balance
    (SizeCV ~ 0.2, wrong) and pay an NMI price.
    """
    imb = results["imbalanced"]
    configs = list(imb.keys())
    true_sizecv = IMBALANCED_TRUE_SIZECV   # 复用模块常量，避免重复计算

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left panel: scatter SizeCV vs NMI ---
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(configs)))
    for color, name in zip(colors, configs):
        nmis = np.array(imb[name]['nmi'])
        scvs = np.array(imb[name]['bal'])
        ax1.errorbar(scvs.mean(), nmis.mean(),
                     xerr=scvs.std(), yerr=nmis.std(),
                     fmt='o', color=color, markersize=11, capsize=5,
                     label=name, alpha=0.85, markeredgecolor='k',
                     markeredgewidth=0.5)
        ax1.annotate(name, (scvs.mean(), nmis.mean()),
                     textcoords="offset points", xytext=(10, 6), fontsize=9,
                     color=color, fontweight='bold')
    ax1.axvline(true_sizecv, color='red', linestyle='--', alpha=0.7, linewidth=2,
                label=f'true SizeCV = {true_sizecv:.3f}')
    ax1.axvspan(true_sizecv - 0.05, true_sizecv + 0.05,
                color='red', alpha=0.08, label='true structure band')
    ax1.set_xlabel('SizeCV  (lower = more balanced communities)', fontsize=11)
    ax1.set_ylabel('NMI  (higher = better community recovery)', fontsize=11)
    ax1.set_title('Imbalanced SBM: size recovery vs community quality', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)

    # --- Right panel: bar chart NMI with error bars, coloured by config ---
    x = np.arange(len(configs))
    nmi_means = [np.mean(imb[c]['nmi']) for c in configs]
    nmi_stds = [np.std(imb[c]['nmi']) for c in configs]
    scv_means = [np.mean(imb[c]['bal']) for c in configs]
    bars = ax2.bar(x, nmi_means, yerr=nmi_stds, color=colors, capsize=5,
                   edgecolor='k', linewidth=0.5, alpha=0.85)
    for bar, scv in zip(bars, scv_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'SCV\n{scv:.2f}', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=20, ha='right')
    ax2.set_ylabel('NMI', fontsize=11)
    ax2.set_title('NMI ranking (SizeCV annotated on bars)', fontsize=12)
    ax2.set_ylim(0, 1.08)
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    out = _fig_path("imbalanced_scatter.png")
    plt.savefig(out, dpi=120)
    print(f"\nSaved figure -> {out}")


# ----------------------------- 入口 -----------------------------
if __name__ == "__main__":
    # 1) Multi-seed comparison: easy/medium/hard SBM + imbalanced SBM
    results = run_difficulty_sweep(n_seeds=5)
    # 2) Intuitive scatter for imbalanced SBM (SizeCV vs NMI)
    plot_imbalanced_scatter(results)
    # 3) Detailed single-run training curves on medium (clearest collapse->recovery)
    print("\n=== Detailed single-run trace (medium) ===")
    run_single_trace(seed=0, difficulty="medium")
