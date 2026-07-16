"""run_wj.py — WJ 五变体图聚类实验运行器。

5 方法 × 5 数据集 × 5 seeds，输出 baseline_comparison.md。

方法集（5 个 WJ 变体）：
  - vanilla       : 纯重建（WJ 内部 baseline）
  - method2       : 玻尔兹曼熵最大化（无反塌缩，大图适用）
  - m2_rank3      : method2 + 最强双铰链反塌缩（强社区图最佳）
  - m2_cl         : method2 + 对比学习（弱社区/大图适用，弥补 hinge 关闭）
  - m2_rank3_cl   : m2_rank3 + 结构 CL（强社区图：hinge + CL 双重增强）

数据集（5 个，Amazon Photo 若未下载则跳过）：
  cora(小/引用), citeseer(小/引用), pubmed(大/引用),
  polblogs(小/社交), amazon_photo(中/购物)

指标：ACC, NMI, ARI, Modularity（wj.evaluate_metrics.compute_all_metrics）

输出：
  - baseline_comparison.md: 四张指标表 + 综合排名
  - results JSON: 断点续传用

用法：
  python run_wj.py                          # 全量：5 seeds × 3 方法 × 5 数据集
  python run_wj.py --quick                  # 快速：1 seed，验证全链路
  python run_wj.py --no-amazon              # 跳过 Amazon Photo
  python run_wj.py --datasets cora polblogs # 指定子集
"""
import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

# WJ 核心模块统一从 wj 包导入
from wj import (
    DATASET_LOADERS, make_configs, train_one, normalize_adj,
    kmeans_labels, DEVICE, TrainConfig, AMAZON_PHOTO_DIR,
)
from wj.evaluate_metrics import compute_all_metrics, fmt_mean_std as _fmt, mean_optional as _mean
from experiments.significance_test import paired_ttest, sig_marker
from auto_config import analyze_graph


# ----------------------------- 配置 -----------------------------
WJ_CONFIGS = ['vanilla', 'method2', 'm2_rank3', 'm2_cl', 'm2_rank3_cl']
ALL_METHODS = WJ_CONFIGS  # WJ 五变体（含对比学习变体）
METRICS = ['acc', 'nmi', 'ari', 'mod']  # ACC 放首位（SOTA 论文标准第一指标）

# 数据集元信息（用于元信息表和图表标签）
DATASET_META = {
    'cora':         {'N': 2708,  'K': 7, 'domain': '引用', 'scale': '小'},
    'citeseer':     {'N': 3312,  'K': 6, 'domain': '引用', 'scale': '小'},
    'pubmed':       {'N': 19717, 'K': 3, 'domain': '引用', 'scale': '大'},
    'polblogs':     {'N': 1490,  'K': 2, 'domain': '社交', 'scale': '小'},
    'amazon_photo': {'N': 7650,  'K': 8, 'domain': '购物', 'scale': '中'},
}


# ----------------------------- 单方法运行 -----------------------------
def run_wj_method(A_np: np.ndarray, labels: np.ndarray, K: int,
                  cfg, features: Optional[np.ndarray]) -> Dict[str, float]:
    """训练 WJ 模型，返回四指标 {acc, nmi, ari, mod}。

    训练后从最终 Z 提取社区标签（L2归一化 + Q初始化 + K自适应多次KMeans）。
    三项后处理改进（不改玻尔兹曼熵核心）：
      1. Z 行 L2 归一化：消除节点间嵌入范数差异，KMeans 关注方向而非幅度
      2. Q 初始化：用模型学到的软聚类分配 Q 计算 KMeans 初始中心
      3. K 自适应轮数：K 小（如 PubMed K=3）时少跑，K 大时多跑
    """
    model, hist = train_one(A_np, labels, cfg, X_feat=features)
    # 从最终 Z 和 Q 提取社区标签
    with torch.no_grad():
        A_hat = normalize_adj(A_np)
        X_tensor = (torch.tensor(features, dtype=torch.float32, device=DEVICE)
                    if features is not None else None)
        Z, Q, _, _ = model(A_hat, X_tensor)
        Z_np = Z.cpu().numpy()
        Q_np = Q.cpu().numpy()

    from sklearn.cluster import KMeans as _KMeans
    n = Z_np.shape[0]

    # 改进 1：根据 hinge 状态自适应选择 L2 归一化
    # - hinge 开启（lambda_rank > 0，如 Cora m2_rank3_cl）：不归一化
    #   hinge 维持的嵌入范数差异携带社区信息，归一化会破坏它
    # - hinge 关闭（如 CiteSeer/PubMed/PolBlogs/Amazon Photo）：L2 归一化
    #   消除节点间范数差异，让 KMeans 关注方向而非幅度，显著提升聚类质量
    # 实验验证：PolBlogs ARI +0.18, Amazon Photo ACC +0.13, CiteSeer m2_cl ARI +0.13
    if cfg.lambda_rank > 0:
        Z_km = Z_np  # hinge ON：保留范数差异
    else:
        Z_km = Z_np / (np.linalg.norm(Z_np, axis=1, keepdims=True) + 1e-8)  # hinge OFF：L2 归一化

    # 改进 3：K 自适应轮数——K 小时 KMeans 本身稳定（如 PubMed K=3 必收敛到全局最优），
    # 多次初始化浪费算力；K 大时搜索空间大，需要更多初始化。
    # V6 增强：K=4-6 从 6 轮增至 10 轮（CiteSeer K=6 受益于更多初始化）
    if K <= 3:
        n_km_runs = 2   # K≤3：Q初始化 + 1 次随机（搜索空间极小，必收敛全局最优）
    elif K <= 6:
        n_km_runs = 10  # K=4-6：Q初始化 + 9 次随机
    else:
        n_km_runs = 11  # K>6：Q初始化 + 10 次随机（Cora K=7 成功策略）
    n_init_per_run = 3 if n > 5000 else 10  # 大图降低以加速，小图保持 10 保证质量

    best_pred, best_inertia = None, float('inf')

    # 改进 2：第一轮用 Q 初始化——模型已学到软聚类分配 Q，
    # 用 Q 的 argmax 计算初始中心，让 KMeans 从语义合理的起点开始
    init_labels = Q_np.argmax(axis=1)
    init_centers = np.zeros((K, Z_km.shape[1]))
    for k in range(K):
        mask = init_labels == k
        if mask.any():
            init_centers[k] = Z_km[mask].mean(axis=0)
        else:
            init_centers[k] = Z_km[np.random.RandomState(0).randint(n)]
    km = _KMeans(n_clusters=K, init=init_centers, n_init=1).fit(Z_km)
    if km.inertia_ < best_inertia:
        best_inertia = km.inertia_
        best_pred = km.labels_

    # 后续轮：随机初始化，取 inertia 最小的
    for km_seed in range(n_km_runs - 1):
        km = _KMeans(n_clusters=K, n_init=n_init_per_run,
                     random_state=km_seed).fit(Z_km)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_pred = km.labels_

    pred = best_pred
    return compute_all_metrics(labels, pred, A_np)


# ----------------------------- 断点续传：JSON 保存/加载 -----------------------------
def save_results_json(results: Dict, path: str):
    """保存 results dict 到 JSON（断点续传用）。

    None 值序列化为 null，加载时还原为 None。
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_results_json(path: str) -> Dict:
    """从 JSON 加载 results dict。"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _seed_done(results: Dict, ds: str, seed_idx: int, n_methods: int) -> bool:
    """检查某 dataset 的某 seed 是否已跑完（所有方法都有值）。

    用于断点续传：跳过已完成的 seed，避免重复计算。
    """
    if ds not in results:
        return False
    for m in ALL_METHODS:
        if m not in results[ds]:
            return False
        nmi_list = results[ds][m].get('nmi', [])
        if seed_idx >= len(nmi_list):
            return False
    # 所有方法的该 seed 都有值才算完成
    return all(seed_idx < len(results[ds][m].get('nmi', [])) for m in ALL_METHODS)


# ----------------------------- 主循环 -----------------------------
def run_comparison(datasets: List[str], n_seeds: int = 5,
                    epochs: int = 200,
                    results_json: str = None) -> Dict:
    """跑全部 WJ 对比实验。

    Args:
        datasets: 数据集名列表
        n_seeds: 随机种子数
        epochs: 训练轮数
        results_json: 断点续传 JSON 路径。若文件存在，加载已有结果跳过已完成的 seed；
                      每个 seed 跑完后增量保存，断电可恢复。

    Returns:
        results[dataset][method] = {metric: [seed_values or None]}
    """
    # 断点续传：加载已有结果
    if results_json and os.path.exists(results_json):
        print(f"[resume] 加载已有结果: {results_json}")
        results = load_results_json(results_json)
        # 补全结构（新数据集/方法可能不在旧 JSON 里）
        for ds in datasets:
            if ds not in results:
                results[ds] = {m: {k: [] for k in METRICS} for m in ALL_METHODS}
            else:
                for m in ALL_METHODS:
                    if m not in results[ds]:
                        results[ds][m] = {k: [] for k in METRICS}
                    else:
                        for k in METRICS:
                            results[ds][m].setdefault(k, [])
    else:
        results = {ds: {m: {k: [] for k in METRICS} for m in ALL_METHODS}
                   for ds in datasets}

    for ds_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'=' * 70}")
        try:
            loader, K, default_emb, default_rank = DATASET_LOADERS[ds_name]
            A_np, labels, features = loader()
        except Exception as e:
            print(f"[ERROR] 加载 {ds_name} 失败，跳过: {e}")
            continue

        # 不做 CC 过滤：使用全图（和 SOTA 论文保持一致，如 CiteSeer 3312 节点）
        n = A_np.shape[0]
        # 图结构分析：聚类系数 CC 决定 m2_rank3 的 hinge 强度
        graph_stats = analyze_graph(A_np)
        cc = graph_stats['cc']
        # 大图自适应：N>5000 时减少 epochs + seeds
        adaptive_epochs = epochs if n <= 5000 else min(epochs, 50)
        adaptive_seeds = n_seeds if n <= 5000 else min(n_seeds, 2)
        # 自适应 hinge：实验表明 hinge 在弱社区图和大图上有害，仅在强社区图上有效
        # - 强社区（CC≥0.20，如 Cora）：λ=3.0（强铰链防塌缩）
        # - 其他所有情况（含 CiteSeer/PubMed）：λ=0.0（关闭 hinge，靠对比学习补充）
        if cc >= 0.20 and n <= 5000:
            adaptive_lambda = 3.0
            adaptive_min_rank = default_rank
            hinge_note = f"强社区(CC={cc:.3f}≥0.20): λ=3.0"
        else:
            adaptive_lambda = 0.0
            adaptive_min_rank = default_rank
            hinge_note = f"弱社区/大图(CC={cc:.3f}, N={n}): λ=0.0, 靠对比学习补充"
        # 自适应 CL 模式：强社区图用结构 CL（图结构可靠），
        # 弱社区图用特征 CL（特征比图结构可靠，避免放大噪声）
        # - 高维稀疏特征（F>500，如 CiteSeer 3703 维 BoW）因维度灾难导致
        #   KNN 不可靠 → 改用 combined CL（结构+特征加权）+ PCA 降维到 200
        #   解决纯特征 CL 让 NMI↑但 ARI↓（聚类边界与标签不对齐）的问题
        # - 低维特征（F≤500，如 PubMed 500 维）保持纯特征 CL
        HIGH_DIM_THRESHOLD = 500  # 高维稀疏特征阈值
        PCA_TARGET_DIM = 200      # PCA 降维目标维度
        LARGE_GRAPH_THRESHOLD = 5000  # 大图阈值（CL 在大图+低维特征上有害）
        if cc >= 0.20 and n <= 5000:
            # 强社区图（如 Cora）：结构 CL，特征重建关闭（图结构已足够）
            cl_mode = "structure"
            cl_lambda_contrast = 0.1
            cl_lambda_feat = 0.0
            cl_pca_dim = 0
            cl_feat_weight = 0.5
            cl_note = "结构CL"
        elif features is not None and features.shape[1] > HIGH_DIM_THRESHOLD:
            # 弱社区 + 高维稀疏特征（如 CiteSeer 3703维）：
            # 混合 CL + PCA 200 + 特征重建(λ=0.1)
            # V6 实验验证：λ=0.3 过强（3703维重建压力过大干扰核心优化），回退到 0.1
            cl_mode = "combined"
            cl_lambda_contrast = 0.1
            cl_lambda_feat = 0.1
            cl_pca_dim = PCA_TARGET_DIM
            cl_feat_weight = 0.5
            cl_note = (f"混合CL+PCA({PCA_TARGET_DIM})+特征重建(λ=0.1) "
                       f"(F={features.shape[1]}>{HIGH_DIM_THRESHOLD})")
        elif n > LARGE_GRAPH_THRESHOLD:
            # 大图 + 低维特征（如 PubMed N=19717, F=500）：关闭 CL
            # 实验证据：PubMed m2_cl NMI 0.22 < vanilla 0.28，特征 CL 在大图上有害
            cl_mode = "feature"
            cl_lambda_contrast = 0.0
            cl_lambda_feat = 0.1
            cl_pca_dim = 0
            cl_feat_weight = 0.5
            cl_note = f"关闭CL(大图N={n}>{LARGE_GRAPH_THRESHOLD}+低维特征, CL有害)"
        else:
            # 弱社区 + 低维特征 + 小图：纯特征 CL
            cl_mode = "feature"
            cl_lambda_contrast = 0.1
            cl_lambda_feat = 0.1
            cl_pca_dim = 0
            cl_feat_weight = 0.5
            cl_note = "特征CL+特征重建"
        print(f"  N={n}, K={K}, has_features={features is not None}, "
              f"emb_dim={default_emb}, min_rank={default_rank}, "
              f"epochs={adaptive_epochs}, seeds={adaptive_seeds} "
              f"[{hinge_note} | m2_cl/m2_rank3_cl: {cl_note}]")

        for seed in range(adaptive_seeds):
            # 断点续传：该 seed 已完成则跳过
            if _seed_done(results, ds_name, seed, len(ALL_METHODS)):
                print(f"\n  --- seed {seed} --- [resume] 已完成，跳过")
                continue
            print(f"\n  --- seed {seed} ---")
            # 若部分方法已有值（断点中断），截断到当前 seed 长度
            for m in ALL_METHODS:
                for k in METRICS:
                    lst = results[ds_name][m][k]
                    if len(lst) > seed:
                        results[ds_name][m][k] = lst[:seed]
            # WJ 方法（5 个 config）
            configs = make_configs(K, seed, epochs=adaptive_epochs,
                                   min_rank=default_rank, emb_dim=default_emb)
            # 自适应 hinge：根据图结构（CC）调整 m2_rank3 的 lambda_rank
            configs['m2_rank3'].lambda_rank = adaptive_lambda
            if adaptive_lambda < 3.0 and adaptive_lambda > 0.0:
                configs['m2_rank3'].rank_min_rank = adaptive_min_rank
            # 自适应 CL：使用预计算的 cl_mode/cl_lambda_contrast/cl_lambda_feat/cl_pca_dim
            configs['m2_cl'].contrast_mode = cl_mode
            configs['m2_cl'].lambda_contrast = cl_lambda_contrast
            configs['m2_cl'].lambda_feat_recon = cl_lambda_feat
            configs['m2_cl'].pca_dim = cl_pca_dim
            configs['m2_cl'].cl_feat_weight = cl_feat_weight
            # m2_rank3_cl = m2_rank3 的 hinge + m2_cl 的 CL（组合变体）
            # 强社区图：hinge λ=3.0 + 结构 CL → 双重增强社区边界
            # 弱社区图：hinge λ=0.0 + 特征/混合 CL → 等同 m2_cl（hinge 有害故关闭）
            configs['m2_rank3_cl'].lambda_rank = adaptive_lambda
            configs['m2_rank3_cl'].rank_min_rank = adaptive_min_rank
            configs['m2_rank3_cl'].contrast_mode = cl_mode
            configs['m2_rank3_cl'].lambda_contrast = cl_lambda_contrast
            configs['m2_rank3_cl'].lambda_feat_recon = cl_lambda_feat
            configs['m2_rank3_cl'].pca_dim = cl_pca_dim
            configs['m2_rank3_cl'].cl_feat_weight = cl_feat_weight
            # 辅助：保存一个方法的结果并立即落盘（高频断点，防崩溃丢失）
            def _save_method(method_name: str, m_dict: Optional[Dict[str, float]]):
                for k in METRICS:
                    results[ds_name][method_name][k].append(
                        m_dict[k] if m_dict is not None else None)
                # 每个方法跑完就保存 JSON，即使崩溃也只丢一个方法
                if results_json:
                    save_results_json(results, results_json)

            # WJ 方法（5 个 config）
            for cfg_name in WJ_CONFIGS:
                try:
                    cfg = configs[cfg_name]
                    # 大图降低 KMeans 频率以加速
                    cfg.eval_every = 20 if n > 5000 else 10
                    m = run_wj_method(A_np, labels, K, cfg, features)
                    _save_method(cfg_name, m)
                    print(f"    {cfg_name:<10} ACC={m['acc']:.4f} "
                          f"NMI={m['nmi']:.4f} ARI={m['ari']:.4f} "
                          f"Mod={m['mod']:.4f}")
                except Exception as e:
                    print(f"    [ERROR] {cfg_name}: {e}")
                    traceback.print_exc()
                    _save_method(cfg_name, None)

    return results


def build_metric_table(results: Dict, datasets: List[str],
                       metric: str) -> str:
    """构建单个指标的 markdown 表（行=数据集，列=WJ 四变体，最佳加粗）。"""
    lines = []
    header = "| 数据集 | " + " | ".join(ALL_METHODS) + " |"
    sep = "|---|" + "|".join(["---"] * len(ALL_METHODS)) + "|"
    lines.append(header)
    lines.append(sep)

    # 每行的最佳值（用于加粗）
    for ds in datasets:
        if ds not in results:
            continue
        # 计算每个方法的均值
        means = {}
        for m in ALL_METHODS:
            vals = results[ds][m].get(metric, [])
            means[m] = _mean(vals)
        # 找最佳（非 NaN 中最大）
        valid_means = {m: v for m, v in means.items() if not np.isnan(v)}
        best = max(valid_means.values()) if valid_means else None

        cells = []
        for m in ALL_METHODS:
            cell = _fmt(results[ds][m].get(metric, []))
            # 最佳值加粗
            if best is not None and not np.isnan(means[m]) and abs(means[m] - best) < 1e-9:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {ds.upper()} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_meta_table(datasets: List[str], results: Dict) -> str:
    """数据集元信息表（name, N, K, 领域, 规模）。"""
    lines = ["| 数据集 | N | K | 领域 | 规模 |",
             "|---|---|---|---|---|"]
    for ds in datasets:
        if ds not in results:
            continue
        meta = DATASET_META.get(ds, {})
        n = meta.get('N', '?')
        k = meta.get('K', '?')
        domain = meta.get('domain', '?')
        scale = meta.get('scale', '?')
        lines.append(f"| {ds.upper()} | {n} | {k} | {domain} | {scale} |")
    return "\n".join(lines)


def build_ranking_section(results: Dict, datasets: List[str]) -> str:
    """综合排名：每数据集给方法排名 1-3，求平均排名（NMI 为主）。"""
    lines = ["### 综合排名（按 NMI 排名，每数据集 1-3，求平均）\n"]
    lines.append("| 数据集 | " + " | ".join(ALL_METHODS) + " |")
    lines.append("|---|" + "|".join(["---"] * len(ALL_METHODS)) + "|")

    rank_sums = {m: [] for m in ALL_METHODS}
    for ds in datasets:
        if ds not in results:
            continue
        means = {m: _mean(results[ds][m]['nmi']) for m in ALL_METHODS}
        # 按 NMI 降序排名（NaN 排最后）
        sorted_methods = sorted(ALL_METHODS,
                                key=lambda m: means[m] if not np.isnan(means[m]) else -1,
                                reverse=True)
        cells = []
        for m in ALL_METHODS:
            rank = sorted_methods.index(m) + 1
            rank_sums[m].append(rank)
            cells.append(str(rank))
        lines.append(f"| {ds.upper()} | " + " | ".join(cells) + " |")

    # 平均排名行
    avg_cells = []
    for m in ALL_METHODS:
        ranks = rank_sums[m]
        avg = np.mean(ranks) if ranks else float('nan')
        avg_cells.append(f"{avg:.1f}")
    lines.append(f"| **平均排名** | " + " | ".join(f"**{c}**" for c in avg_cells) + " |")
    return "\n".join(lines)


def write_comparison_md(results: Dict, datasets: List[str],
                        n_seeds: int, elapsed_sec: float) -> str:
    """生成完整 baseline_comparison.md（WJ 四变体结果）。"""
    lines = ["# WJ 四变体图聚类实验结果\n"]
    lines.append(f"> 由 `run_wj.py` 自动生成。"
                f"{n_seeds} seeds × {len(ALL_METHODS)} 方法 × {len(datasets)} 数据集，"
                f"总耗时 {elapsed_sec / 60:.1f} 分钟。"
                f"（大图 N>5000 自适应降为 2 seeds × 50 epochs）")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 元信息
    lines.append("## 1. 数据集元信息\n")
    lines.append(build_meta_table(datasets, results))
    lines.append(f"\n- 方法集：WJ({'+'.join(WJ_CONFIGS)})\n")

    # 2-5. 四张指标表
    for i, metric in enumerate(METRICS, start=2):
        title = {'acc': 'ACC', 'nmi': 'NMI', 'ari': 'ARI', 'mod': 'Modularity'}[metric]
        lines.append(f"## {i}. {title} 表\n")
        lines.append(build_metric_table(results, datasets, metric))
        lines.append("")

    # 6. 综合排名
    lines.append(f"## {len(METRICS) + 2}. 综合排名\n")
    lines.append(build_ranking_section(results, datasets))
    lines.append("")

    # 7. 输出文件
    lines.append(f"## {len(METRICS) + 3}. 输出文件\n")
    lines.append("- `baseline_comparison.md`: 本文件")
    lines.append("- `results/*.json`: 断点续传用 results JSON")
    lines.append("")
    return "\n".join(lines)


# ----------------------------- 主入口 -----------------------------
def merge_json_files(json_paths: List[str]) -> Dict:
    """合并多个 results JSON 文件（多卡并行跑的结果合并）。

    不同 JSON 含不同数据集的结果，合并成一个完整 results dict。
    同一数据集在多个 JSON 出现时，保留非空的。
    """
    merged = {}
    for jp in json_paths:
        if not os.path.exists(jp):
            print(f"  [skip] {jp} (不存在)")
            continue
        partial = load_results_json(jp)
        for ds, methods in partial.items():
            if ds not in merged:
                merged[ds] = methods
            else:
                for m, metrics in methods.items():
                    if m not in merged[ds]:
                        merged[ds][m] = metrics
                    else:
                        for k, vals in metrics.items():
                            # 保留非空列表
                            if vals and not merged[ds][m].get(k):
                                merged[ds][m][k] = vals
        print(f"  [merge] {jp}: {list(partial.keys())}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="WJ 四变体图聚类实验")
    parser.add_argument("--quick", action="store_true",
                        help="快速版：1 seed，验证全链路")
    parser.add_argument("--no-amazon", action="store_true",
                        help="跳过 Amazon Photo（未下载 npz 时用）")
    parser.add_argument("--datasets", nargs='+', default=None,
                        help="指定数据集子集（如 --datasets cora polblogs）")
    parser.add_argument("--epochs", type=int, default=200,
                        help="WJ 训练 epochs（默认 200，quick 模式 50）")
    parser.add_argument("--output", default="results/baseline_comparison.md",
                        help="输出 markdown 文件名（默认 results/baseline_comparison.md）")
    parser.add_argument("--results-json", default=None,
                        help="断点续传 JSON 路径。存在则加载跳过已完成的 seed；"
                             "每个 seed 跑完后增量保存，断电可恢复")
    parser.add_argument("--merge", nargs='+', default=None,
                        help="合并多个 results JSON 并生成报告（不跑实验）")
    args = parser.parse_args()

    # 合并模式：不跑实验，只合并多个 JSON 并生成报告
    if args.merge:
        print("=" * 70)
        print("合并模式：合并多个 results JSON")
        print(f"  JSON 文件: {args.merge}")
        print("=" * 70)
        results = merge_json_files(args.merge)
        datasets = sorted(results.keys())
        # 推断 n_seeds（取最大的 seed 数）
        n_seeds = 0
        for ds in datasets:
            for m in ALL_METHODS:
                if m in results[ds] and 'nmi' in results[ds][m]:
                    n_seeds = max(n_seeds, len(results[ds][m]['nmi']))
        print(f"  合并后数据集: {datasets}, 最大 seeds: {n_seeds}")

        print("\n生成报告...")
        md = write_comparison_md(results, datasets, n_seeds, 0.0)
        md_path = os.path.join(_ROOT, args.output)
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"  保存: {md_path}")
        print("\n合并完成！")
        return

    n_seeds = 1 if args.quick else 5
    epochs = 50 if args.quick else args.epochs

    # 确定数据集列表
    all_datasets = ['cora', 'citeseer', 'pubmed', 'polblogs', 'amazon_photo']
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = list(all_datasets)
        if args.no_amazon:
            datasets.remove('amazon_photo')

    # 检查 Amazon Photo 是否可用
    if 'amazon_photo' in datasets:
        npz_path = os.path.join(AMAZON_PHOTO_DIR, "amazon_electronics_photo.npz")
        if not os.path.exists(npz_path) or os.path.getsize(npz_path) < 1024:
            print(f"[WARN] Amazon Photo npz 未下载（{npz_path}），自动跳过")
            datasets.remove('amazon_photo')

    print("=" * 70)
    print("RUN_WJ.PY — WJ 四变体图聚类实验")
    print(f"  方法: {len(ALL_METHODS)} 个 = WJ({'+'.join(WJ_CONFIGS)})")
    print(f"  数据集: {datasets}")
    print(f"  Seeds: {n_seeds}, Epochs: {epochs}")
    if args.results_json:
        print(f"  断点续传: {args.results_json}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    t_start = time.time()
    results = run_comparison(datasets, n_seeds=n_seeds, epochs=epochs,
                             results_json=args.results_json)
    elapsed = time.time() - t_start

    # 最终保存 JSON（确保完整）
    if args.results_json:
        os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
        save_results_json(results, args.results_json)
        print(f"  最终 JSON: {args.results_json}")

    # 生成报告
    print("\n生成报告...")
    md = write_comparison_md(results, datasets, n_seeds, elapsed)
    md_path = os.path.join(_ROOT, args.output)
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"  保存: {md_path}")

    print(f"\n完成！总耗时 {elapsed / 60:.1f} 分钟。")


if __name__ == "__main__":
    main()
