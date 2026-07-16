"""auto_config.py — 根据图规模和结构自动选择最佳 GNN config

核心思路（基于已验证实验结果）
================================

实验表明最优 config 取决于图结构：
  - 小图 + 强社区结构（高聚类系数）→ hinge 损失（m2_rank3）
    理由：hinge 直接保护 eff_rank，避免小图上 method2 塌缩
  - 大图 或 弱社区结构（低聚类系数）→ 纯 method2
    理由：大图上 method2 自带 spread 已足够，hinge 反而干扰社区形成

判别指标（无监督，不需要 GT 标签）
--------------------------------
  - N：节点数
  - CC：聚类系数（小图用 networkx 精确计算，大图采样 2000 节点近似）
  - avg_deg：平均度数（区分 SBM 密集图 vs 真实稀疏图）
  - density：图密度

决策规则（阈值基于已验证实验）
------------------------------
  三层决策：图规模 → 图密度 → 社区强度

  1. 大图 (N > 5000)                     → method2 (纯，无 hinge)
  2. 密集图 (avg_deg > 15，即 SBM 类)     → m2_rank1 (弱 hinge, λ=1.0)
  3. 稀疏小图 + 强社区 (CC >= 0.20)       → m2_rank3 (强 hinge, λ=3.0)
  4. 稀疏小图 + 中等社区 (CC >= 0.05)     → m2_rank2 (中 hinge, λ=2.0)
  5. 弱社区 (CC < 0.05)                   → method2 (纯，无 hinge)

阈值依据（基于已验证实验结果）：
  | 数据集       | N     | CC    | avg_deg | 最佳 config | NMI Δ     |
  |-------------|-------|-------|---------|-------------|-----------|
  | Cora        | 2708  | 0.241 | 3.90    | m2_rank3    | +0.0448   |
  | CiteSeer    | 3312  | 0.143 | 2.78    | m2_rank2    | +0.0160   |
  | Pubmed      | 19717 | 0.008 | 4.50    | method2     | +0.0215   |
  | SBM medium  | 160   | 0.159 | 20.62   | m2_rank1    | +0.0072   |
  | SBM hard    | 140   | 0.150 | 19.94   | m2_rank1    | +0.0045   |
  | SBM imbal.  | 175   | 0.184 | 22.80   | m2_rank1    | +0.1178   |

  关键区分：SBM 是密集图（avg_deg≈20-35），真实图是稀疏图（avg_deg≈3-4）。
  SBM 上弱 hinge (m2_rank1) 最佳，真实图上需要根据社区强度选择 hinge 强度。

用法
----
  python auto_config.py --dataset cora              # 跑 Cora，自动选 config
  python auto_config.py --dataset pubmed --seeds 5  # 跑 Pubmed，5 seeds
  python auto_config.py --sbm hard                   # 跑 hard SBM
  python auto_config.py --sbm imbalanced             # 跑 imbalanced SBM
  python auto_config.py --analyze-only --dataset cora # 只分析图属性，不训练
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

# 让脚本能在根目录或子目录运行
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import wj as m
from wj import (
    TrainConfig, make_configs, train_one,
    DIFFICULTIES, IMBALANCED_SIZES, IMBALANCED_P,
    DATASET_LOADERS, METRIC_KEYS,
)


# ----------------------------- 图结构分析 -----------------------------
def analyze_graph(A_np: np.ndarray, max_cc_sample: int = 2000,
                  seed: int = 42) -> Dict[str, float]:
    """无监督分析图属性，返回判别所需的关键指标。

    指标说明：
      - N: 节点数
      - E: 边数（无向，A.sum()/2）
      - avg_deg: 平均度数 = 2E/N
      - density: 图密度 = 2E / (N*(N-1))
      - cc: 聚类系数（小图精确，大图采样近似）

    Args:
        A_np: 邻接矩阵 (N, N)
        max_cc_sample: 大图聚类系数采样的最大节点数
        seed: 采样随机种子

    Returns:
        dict: {N, E, avg_deg, density, cc}
    """
    n = A_np.shape[0]
    n_edges = int(A_np.sum() / 2.0)
    avg_deg = 2.0 * n_edges / max(n, 1)
    density = 2.0 * n_edges / max(n * (n - 1), 1)

    # 聚类系数：小图用 networkx 精确计算，大图采样近似以加速
    # Pubmed (19717 节点) 上精确计算聚类系数需要几十秒，采样 2000 节点 <1 秒
    if n <= 5000:
        G = nx.from_numpy_array(A_np)
        cc = nx.average_clustering(G)
    else:
        rng = np.random.default_rng(seed)
        sample = rng.choice(n, min(max_cc_sample, n), replace=False)
        sub = A_np[np.ix_(sample, sample)]
        Gs = nx.from_numpy_array(sub)
        cc = nx.average_clustering(Gs)

    return {
        'N': n,
        'E': n_edges,
        'avg_deg': avg_deg,
        'density': density,
        'cc': cc,
    }


# ----------------------------- 决策日志埋点 -----------------------------
# 每个 数据集/SBM 的图属性 + 最终选择的 config 记录到 JSONL 日志，
# 方便后续排查决策是否符合预期、调阈值时有据可依。
# 日志和结果统一放 logs/ 和 results/ 子目录，保持项目结构整洁
os.makedirs(os.path.join(_ROOT, "logs"), exist_ok=True)
os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
DECISION_LOG = os.path.join(_ROOT, "results", "auto_config_decisions.jsonl")


def log_decision(dataset_name: str, graph_stats: Dict, K: int,
                 cfg_name: str, reason: str):
    """把 config 决策记录到 JSONL 日志文件（追加模式）。

    每行一个 JSON 对象，字段：
      timestamp, dataset, N, E, avg_deg, density, cc, K, selected_config, reason

    用 JSONL 而非 CSV：reason 含逗号/中文，JSON 更安全；
    每行独立可流式追加，不怕中断丢失全部记录。
    """
    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': dataset_name,
        'N': graph_stats['N'],
        'E': graph_stats['E'],
        'avg_deg': round(graph_stats['avg_deg'], 4),
        'density': round(graph_stats['density'], 6),
        'cc': round(graph_stats['cc'], 6),
        'K': K,
        'selected_config': cfg_name,
        'reason': reason,
    }
    with open(DECISION_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    # 同步打印一行摘要，方便运行时实时观察
    print(f"  [decision] {dataset_name}: N={entry['N']}, "
          f"avg_deg={entry['avg_deg']:.2f}, CC={entry['cc']:.4f}, "
          f"K={K} → {cfg_name}")


# ----------------------------- 断点恢复 -----------------------------
# 长时间实验可能因 SSH 断连/服务器重启中断。断点文件记录已完成任务，
# 下次用 --resume 恢复时跳过已完成部分，从断点继续。
# 粒度：一个 task = 一个数据集或 SBM regime 的全部 seeds（独立完成）。
CHECKPOINT_FILE = os.path.join(_ROOT, "auto_config_checkpoint.json")
PROGRESS_LOG = os.path.join(_ROOT, "auto_config_progress.log")


def log_progress(msg: str):
    """写入人类可读的进度日志（追加模式，带时间戳）。

    用于排查实验跑到哪儿了、在哪一步断的。每次开始/完成一个 task 都写一行。
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(PROGRESS_LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{ts}] {msg}\n")
    # 同步打印到控制台，运行时实时可见
    print(f"  [progress] {msg}")


def _to_jsonable(obj):
    """递归把 numpy 类型转成原生 Python 类型，方便 JSON 序列化。"""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class Checkpoint:
    """断点恢复管理器。

    记录已完成的 task（数据集/SBM regime），支持中断后用 --resume 恢复。
    每个 task 独立完成（全 seeds 跑完才存），存完即 flush 到磁盘，
    保证即使后续 task 崩溃，已完成的 task 结果也不丢。

    用法：
        ckpt = Checkpoint()
        if args.resume:
            ckpt.load()  # 加载已有断点
        if not ckpt.is_done(task_name):
            result = run_task(...)
            ckpt.save_task(task_name, result)
    """

    def __init__(self, path: str = CHECKPOINT_FILE):
        self.path = path
        self.data = {
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tasks_done': [],
            'results': {},
        }

    def load(self) -> bool:
        """加载断点文件。返回 True 如果存在且加载成功。"""
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return True
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [checkpoint] 断点文件损坏，忽略: {e}")
            return False

    def is_done(self, task_name: str) -> bool:
        """检查某个 task 是否已完成。"""
        return task_name in self.data['tasks_done']

    def get_result(self, task_name: str):
        """获取已完成 task 的结果（用于恢复时填入 all_results）。"""
        return self.data['results'].get(task_name)

    def save_task(self, task_name: str, result):
        """保存一个 task 的完成结果，立即 flush 到磁盘。

        结果转成 JSON-safe 格式（numpy → python），保证文件可读。
        """
        self.data['tasks_done'].append(task_name)
        self.data['results'][task_name] = _to_jsonable(result)
        self._flush()

    def _flush(self):
        """把断点数据原子写入磁盘（先写临时文件再 rename，防写到一半崩溃）。"""
        tmp_path = self.path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    def clear(self):
        """实验全部完成后清理断点文件。"""
        if os.path.exists(self.path):
            os.remove(self.path)

    def summary(self) -> str:
        """返回人类可读的断点摘要。"""
        done = self.data.get('tasks_done', [])
        return f"已完成 {len(done)} 个 task: {', '.join(done) if done else '(无)'}"


# ----------------------------- Config 自动选择 -----------------------------
# 决策阈值（基于已验证实验结果，可调整）
N_SMALL_THRESHOLD = 5000          # 小图/大图分界
AVG_DEG_DENSE_THRESHOLD = 15.0    # 密集图（SBM 类）/ 稀疏图分界
CC_VERY_STRONG_THRESHOLD = 0.20   # 很强社区（用更强 hinge λ=3.0）
CC_STRONG_THRESHOLD = 0.05         # 中等社区分界（用中 hinge λ=2.0）


def _auto_hyperparams(N: int, K: int) -> Tuple[int, float]:
    """根据图规模和社区数自适应选择 emb_dim 和 min_rank。

    经验法则（已通过实验验证）：
      - emb_dim: 小图 32, 大图 64（大图需要更高维表达复杂结构）
      - min_rank: ≈ 0.7*K，下界 2.0（防 K 太小时 hinge 过强导致训练崩溃）

    将此逻辑集中在一处，避免 select_config/run_auto_experiment/_run_sbm_multi_seed
    三处重复计算导致不一致。
    """
    emb_dim = 32 if N <= N_SMALL_THRESHOLD else 64
    min_rank = max(0.7 * K, 2.0)
    return emb_dim, min_rank


def select_config(graph_stats: Dict[str, float], K: int,
                  seed: int = 42, epochs: int = 200) -> Tuple[str, TrainConfig, str]:
    """根据图属性自动选择最佳 config。

    三层决策（按优先级）：
      1. 图规模：大图 (N > 5000) → method2
      2. 图密度：密集图 (avg_deg > 15，即 SBM 类) → m2_rank1
      3. 社区强度（仅稀疏小图）：
         - CC >= 0.20 → m2_rank3（强 hinge）
         - 0.05 <= CC < 0.20 → m2_rank2（中 hinge）
         - CC < 0.05 → method2（纯）

    emb_dim 经验法则：
      - 小图（N <= 5000）：32
      - 大图（N > 5000）：64

    min_rank 经验法则：≈ 0.7 * K，下界 2.0
      - K=4 SBM → 2.8 ≈ 3.0
      - K=7 Cora → 4.9 ≈ 5.0
      - K=6 CiteSeer → 4.2 ≈ 4.0
      - K=3 Pubmed → 2.1 ≈ 2.5

    Args:
        graph_stats: analyze_graph() 返回的图属性 dict
        K: 社区数
        seed: 随机种子
        epochs: 训练轮数

    Returns:
        (config_name, config, reason): 选中的 config 名、TrainConfig、决策理由
    """
    N = graph_stats['N']
    cc = graph_stats['cc']
    avg_deg = graph_stats['avg_deg']

    # emb_dim 和 min_rank 自适应（集中在一处，避免三处重复计算不一致）
    emb_dim, min_rank = _auto_hyperparams(N, K)

    # 生成全部 config，从中挑选（复用 make_configs，不重复造轮子）
    all_configs = make_configs(K, seed, epochs=epochs,
                               min_rank=min_rank, emb_dim=emb_dim)

    # 三层决策逻辑
    if N > N_SMALL_THRESHOLD:
        # 1. 大图 → 纯 method2（hinge 在大图上有害）
        cfg_name = "method2"
        reason = (f"大图(N={N}>{N_SMALL_THRESHOLD}) → method2 "
                  f"(纯，无 hinge；大图上 hinge 干扰社区形成)")
    elif avg_deg > AVG_DEG_DENSE_THRESHOLD:
        # 2. 密集图（SBM 类）→ 弱 hinge
        # SBM 上弱 hinge (λ=1.0) 最佳：结构清晰，弱保护足够
        cfg_name = "m2_rank1"
        reason = (f"密集图(avg_deg={avg_deg:.1f}>{AVG_DEG_DENSE_THRESHOLD:.0f}, "
                  f"SBM 类) → m2_rank1 (弱 hinge, λ=1.0)")
    elif cc >= CC_VERY_STRONG_THRESHOLD:
        # 3a. 稀疏小图 + 强社区 → 强 hinge（如 Cora）
        cfg_name = "m2_rank3"
        reason = (f"稀疏小图(N={N}≤{N_SMALL_THRESHOLD}, avg_deg={avg_deg:.1f}) + "
                  f"强社区(CC={cc:.3f}≥{CC_VERY_STRONG_THRESHOLD}) → "
                  f"m2_rank3 (强 hinge, λ=3.0)")
    elif cc >= CC_STRONG_THRESHOLD:
        # 3b. 稀疏小图 + 中等社区 → 中 hinge（如 CiteSeer）
        cfg_name = "m2_rank2"
        reason = (f"稀疏小图(N={N}≤{N_SMALL_THRESHOLD}, avg_deg={avg_deg:.1f}) + "
                  f"中等社区(CC={cc:.3f}≥{CC_STRONG_THRESHOLD}) → "
                  f"m2_rank2 (中 hinge, λ=2.0)")
    else:
        # 3c. 弱社区 → 纯 method2
        cfg_name = "method2"
        reason = (f"稀疏小图 + 弱社区(CC={cc:.3f}<{CC_STRONG_THRESHOLD}) → "
                  f"method2 (纯，无 hinge)")

    config = all_configs[cfg_name]
    return cfg_name, config, reason


# ----------------------------- 训练与评估 -----------------------------
def _run_paired_train(graph_stats: Dict, K: int, cfg_name: str, reason: str,
                     get_graph_fn, n_seeds: int, epochs: int,
                     verbose: bool = True) -> Dict:
    """统一的 vanilla vs auto_config 配对训练。

    把 run_auto_experiment（真实数据集，固定图）和 _run_sbm_multi_seed（SBM，
    每 seed 重新生成图）的共同逻辑抽出来：config 生成、训练循环、结果汇总。

    Args:
        graph_stats: analyze_graph() 返回的图属性
        K: 社区数
        cfg_name: 自动选择的 config 名
        reason: 决策理由（用于返回结果）
        get_graph_fn: 回调，get_graph_fn(seed) -> (A_np, labels, features)
                     真实数据集返回固定三元组；SBM 每 seed 重新生成
        n_seeds: 种子数
        epochs: 训练轮数
        verbose: 是否打印详细日志

    Returns:
        dict: {graph_stats, config_name, reason, results}
    """
    # 大图降频 KMeans 以加速
    eval_every = 20 if graph_stats['N'] > 5000 else 10
    configs_to_run = ['vanilla', cfg_name]
    results = {name: {k: [] for k in METRIC_KEYS} for name in configs_to_run}

    if verbose:
        print(f"\n{'='*60}")
        print(f"训练对比: {configs_to_run} × {n_seeds} seeds × {epochs} epochs")
        print(f"{'='*60}")

    # emb_dim/min_rank 只算一次（vanilla 和 auto 共用，保证公平对比）
    emb_dim, min_rank = _auto_hyperparams(graph_stats['N'], K)

    for s in range(n_seeds):
        if verbose:
            print(f"\n  --- seed {s} ---")
        A_s, y_s, feat_s = get_graph_fn(s)

        # 每个 seed 重新生成 config（seed 不同）
        _, selected_cfg, _ = select_config(graph_stats, K, seed=s, epochs=epochs)
        selected_cfg.eval_every = eval_every
        vanilla_cfg = make_configs(K, s, epochs=epochs,
                                   min_rank=min_rank, emb_dim=emb_dim)['vanilla']
        vanilla_cfg.eval_every = eval_every

        for name, cfg in [('vanilla', vanilla_cfg), (cfg_name, selected_cfg)]:
            _, h = train_one(A_s, y_s, cfg, X_feat=feat_s)
            for k in METRIC_KEYS:
                results[name][k].append(h[k][-1])
            if verbose:
                print(f"    {name:<12} NMI={h['nmi'][-1]:.4f}  "
                      f"Mod={h['mod'][-1]:.4f}  SizeCV={h['bal'][-1]:.4f}  "
                      f"effRank={h['eff_rank'][-1]:.2f}  embStd={h['embed_std'][-1]:.4f}")

    # 结果汇总
    if verbose:
        print(f"\n{'='*60}")
        print(f"结果汇总（{n_seeds} seeds mean±std）")
        print(f"{'='*60}")
        print(f"  {'config':<12} {'NMI':<18} {'Modularity':<18} {'SizeCV':<18} {'effRank':<14}")
        for name in configs_to_run:
            nmi = f"{np.mean(results[name]['nmi']):.4f}±{np.std(results[name]['nmi']):.4f}"
            mod = f"{np.mean(results[name]['mod']):.4f}±{np.std(results[name]['mod']):.4f}"
            scv = f"{np.mean(results[name]['bal']):.4f}±{np.std(results[name]['bal']):.4f}"
            er = f"{np.mean(results[name]['eff_rank']):.2f}±{np.std(results[name]['eff_rank']):.2f}"
            print(f"  {name:<12} {nmi:<18} {mod:<18} {scv:<18} {er:<14}")

        v_nmi = np.mean(results['vanilla']['nmi'])
        a_nmi = np.mean(results[cfg_name]['nmi'])
        delta = a_nmi - v_nmi
        print(f"\n  Δ NMI (auto - vanilla): {delta:+.4f}")
        if delta > 0:
            print(f"  ✅ auto_config 超越 vanilla")
        else:
            print(f"  ⚠️  auto_config 未超越 vanilla（可能需要调整阈值）")

    return {
        'graph_stats': graph_stats,
        'config_name': cfg_name,
        'reason': reason,
        'results': results,
    }


def run_auto_experiment(A_np: np.ndarray, labels: np.ndarray, K: int,
                       features: np.ndarray = None,
                       n_seeds: int = 3, epochs: int = 200,
                       verbose: bool = True,
                       dataset_name: str = "unknown") -> Dict:
    """自动选择 config 并训练，对比 vanilla baseline。

    流程：
      1. 分析图属性
      2. 自动选择 config
      3. 在 n_seeds 个种子上跑 {vanilla, auto_config} 对比
      4. 输出结果表格

    Args:
        A_np: 邻接矩阵
        labels: GT 标签（仅用于评估，不参与 config 选择）
        K: 社区数
        features: 节点特征（None 时用 one-hot）
        n_seeds: 随机种子数
        epochs: 训练轮数
        verbose: 是否打印详细日志

    Returns:
        dict: {graph_stats, config_name, reason, results}
    """
    # 1. 分析图属性
    graph_stats = analyze_graph(A_np)
    if verbose:
        print(f"\n{'='*60}")
        print(f"图属性分析")
        print(f"{'='*60}")
        print(f"  节点数 N      : {graph_stats['N']}")
        print(f"  边数 E        : {graph_stats['E']}")
        print(f"  平均度数      : {graph_stats['avg_deg']:.2f}")
        print(f"  图密度        : {graph_stats['density']:.4f}")
        print(f"  聚类系数 CC   : {graph_stats['cc']:.4f}")

    # 2. 自动选择 config
    cfg_name, auto_cfg, reason = select_config(graph_stats, K, epochs=epochs)
    # 日志埋点：记录图属性 + 最终选择的 config，方便后续排查
    log_decision(dataset_name, graph_stats, K, cfg_name, reason)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Config 自动选择")
        print(f"{'='*60}")
        print(f"  决策: {cfg_name}")
        print(f"  理由: {reason}")
        print(f"  参数: emb_dim={auto_cfg.emb_dim}, min_rank={auto_cfg.rank_min_rank:.2f}, "
              f"lambda_rank={auto_cfg.lambda_rank}, T_max={auto_cfg.T_max}")

    # 3. 配对训练（真实数据集：固定图，所有 seed 用同一张图）
    return _run_paired_train(graph_stats, K, cfg_name, reason,
                             lambda s: (A_np, labels, features),
                             n_seeds, epochs, verbose)


# ----------------------------- 数据集入口 -----------------------------
def run_dataset_auto(name: str, n_seeds: int = 3, epochs: int = 200,
                     verbose: bool = True) -> Dict:
    """在真实数据集上跑自动选择实验。

    Args:
        name: 'cora' | 'citeseer' | 'pubmed'
        n_seeds: 随机种子数
        epochs: 训练轮数
    """
    if name not in DATASET_LOADERS:
        raise ValueError(f"未知数据集: {name}，可选: {list(DATASET_LOADERS.keys())}")
    loader, K, _, _ = DATASET_LOADERS[name]
    A_np, labels, features = loader()
    print(f"\n{'#'*60}")
    print(f"# 数据集: {name.upper()}  (K={K}, {n_seeds} seeds)")
    print(f"{'#'*60}")
    return run_auto_experiment(A_np, labels, K, features=features,
                               n_seeds=n_seeds, epochs=epochs, verbose=verbose,
                               dataset_name=name)


def run_sbm_auto(regime: str, n_seeds: int = 5, epochs: int = 300,
                 verbose: bool = True) -> Dict:
    """在 SBM 合成数据集上跑自动选择实验。

    Args:
        regime: 'easy' | 'medium' | 'hard' | 'imbalanced'
        n_seeds: 随机种子数
        epochs: 训练轮数
    """
    print(f"\n{'#'*60}")
    print(f"# SBM regime: {regime}  ({n_seeds} seeds)")
    print(f"{'#'*60}")

    if regime == 'imbalanced':
        sizes = IMBALANCED_SIZES
        p_in, p_out = IMBALANCED_P
        K = len(sizes)
        # 用第一个 seed 生成图来分析结构
        A_np, labels, _ = m.make_imbalanced_sbm(sizes, p_in=p_in, p_out=p_out, seed=0)
        # 多 seed 训练
        return _run_sbm_multi_seed(regime, sizes, p_in, p_out, K, n_seeds, epochs, verbose)
    elif regime in DIFFICULTIES:
        n_per, n_blk, p_in, p_out = DIFFICULTIES[regime]
        K = n_blk
        return _run_sbm_multi_seed(regime, [n_per] * n_blk, p_in, p_out, K,
                                   n_seeds, epochs, verbose)
    else:
        raise ValueError(f"未知 regime: {regime}，可选: {list(DIFFICULTIES.keys()) + ['imbalanced']}")


def _run_sbm_multi_seed(regime: str, sizes, p_in, p_out, K,
                        n_seeds: int, epochs: int, verbose: bool) -> Dict:
    """SBM 多 seed 训练（共享图结构分析）。

    根据 regime 选对的 SBM 生成函数：
      - imbalanced: make_imbalanced_sbm(sizes, ...)
      - easy/medium/hard: make_sbm(n_per_block=sizes[0], n_blocks=K, ...)
    """
    # 生成图结构的辅助函数：根据 regime 选对的生成器
    def _gen_graph(seed):
        if regime == 'imbalanced':
            return m.make_imbalanced_sbm(sizes, p_in=p_in, p_out=p_out, seed=seed)
        else:
            return m.make_sbm(n_per_block=sizes[0], n_blocks=K,
                              p_in=p_in, p_out=p_out, seed=seed)

    # 用 seed=0 生成图做结构分析（代表该 regime 的典型结构）
    A_np, labels, _ = _gen_graph(0)

    # 分析图属性（用 seed=0 的图，代表该 regime 的典型结构）
    graph_stats = analyze_graph(A_np)
    if verbose:
        print(f"\n图属性分析（{regime} SBM, seed=0 代表）")
        print(f"  N={graph_stats['N']}, E={graph_stats['E']}, "
              f"avg_deg={graph_stats['avg_deg']:.2f}, CC={graph_stats['cc']:.4f}")

    # 自动选择 config
    cfg_name, _, reason = select_config(graph_stats, K, epochs=epochs)
    # 日志埋点：记录 SBM regime 的图属性 + 选择的 config
    log_decision(f"sbm_{regime}", graph_stats, K, cfg_name, reason)
    if verbose:
        print(f"  决策: {cfg_name}")
        print(f"  理由: {reason}")

    # 配对训练（SBM：每个 seed 重新生成图；features=None 用 one-hot）
    def _get_graph(seed):
        A_s, y_s, _ = _gen_graph(seed)
        return A_s, y_s, None

    return _run_paired_train(graph_stats, K, cfg_name, reason,
                             _get_graph, n_seeds, epochs, verbose)


# ----------------------------- 报告生成 -----------------------------
def write_report(all_results: Dict, output_path: str = "results/auto_config_result.md"):
    """把自动选择实验结果写入 markdown 报告。"""
    lines = []
    lines.append("# auto_config.py 实验结果\n")
    lines.append(f"> 自动根据图规模和结构选择 config，对比 vanilla baseline。\n")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 决策规则\n")
    lines.append(f"- 小图(N≤{N_SMALL_THRESHOLD}) + 强社区(CC≥{CC_VERY_STRONG_THRESHOLD}) → m2_rank3")
    lines.append(f"- 小图(N≤{N_SMALL_THRESHOLD}) + 中等社区(CC≥{CC_STRONG_THRESHOLD}) → m2_rank2")
    lines.append("- 大图 或 弱社区 → method2（纯）\n")

    lines.append("## 各数据集结果\n")
    for name, res in all_results.items():
        gs = res['graph_stats']
        cfg = res['config_name']
        reason = res['reason']
        results = res['results']

        lines.append(f"### {name.upper()}\n")
        lines.append(f"- **图属性**: N={gs['N']}, E={gs['E']}, "
                    f"avg_deg={gs['avg_deg']:.2f}, CC={gs['cc']:.4f}")
        lines.append(f"- **选择 config**: `{cfg}`")
        lines.append(f"- **决策理由**: {reason}\n")

        lines.append("| config | NMI | Modularity | SizeCV | effRank |")
        lines.append("|---|---|---|---|---|")
        for c_name in ['vanilla', cfg]:
            if c_name not in results:
                continue
            r = results[c_name]
            nmi = f"{np.mean(r['nmi']):.4f}±{np.std(r['nmi']):.4f}"
            mod = f"{np.mean(r['mod']):.4f}±{np.std(r['mod']):.4f}"
            scv = f"{np.mean(r['bal']):.4f}±{np.std(r['bal']):.4f}"
            er = f"{np.mean(r['eff_rank']):.2f}±{np.std(r['eff_rank']):.2f}"
            lines.append(f"| {c_name} | {nmi} | {mod} | {scv} | {er} |")
        v_nmi = np.mean(results['vanilla']['nmi'])
        a_nmi = np.mean(results[cfg]['nmi'])
        lines.append(f"\n**Δ NMI (auto - vanilla): {a_nmi - v_nmi:+.4f}**\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n报告已写入: {output_path}")


def _get_graph_for_analysis(args):
    """从 args 解析出用于 analyze_only 的图和元信息。

    统一 dataset/sbm 两个分支的图生成逻辑，避免 analyze_only 里重复。
    """
    if args.dataset:
        loader, K, _, _ = DATASET_LOADERS[args.dataset]
        A_np, _, _ = loader()
        return A_np, K, args.dataset, f"数据集: {args.dataset.upper()}"
    elif args.sbm:
        if args.sbm == 'imbalanced':
            sizes = IMBALANCED_SIZES
            p_in, p_out = IMBALANCED_P
            K = len(sizes)
            A_np, _, _ = m.make_imbalanced_sbm(sizes, p_in=p_in, p_out=p_out, seed=0)
        else:
            n_per, n_blk, p_in, p_out = DIFFICULTIES[args.sbm]
            K = n_blk
            A_np, _, _ = m.make_sbm(n_per_block=n_per, n_blocks=K,
                                    p_in=p_in, p_out=p_out, seed=0)
        return A_np, K, f"sbm_{args.sbm}", f"SBM regime: {args.sbm}"
    return None, None, None, None


# ----------------------------- CLI -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="根据图规模和结构自动选择最佳 GNN config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 数据集选择（互斥）
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()),
                   help="真实数据集名（cora/citeseer/pubmed）")
    g.add_argument("--sbm", choices=list(DIFFICULTIES.keys()) + ['imbalanced'],
                   help="SBM regime（easy/medium/hard/imbalanced）")
    g.add_argument("--all", action="store_true",
                   help="跑所有真实数据集 + 所有 SBM regimes")

    parser.add_argument("--seeds", type=int, default=3,
                        help="随机种子数（默认 3，SBM 默认 5）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（默认：真实 200, SBM 300）")
    parser.add_argument("--analyze-only", action="store_true",
                        help="只分析图属性和决策，不训练")
    parser.add_argument("--output", default="results/auto_config_result.md",
                        help="结果报告输出路径")
    # 断点恢复
    parser.add_argument("--resume", action="store_true",
                        help="从断点恢复（跳过已完成的 task，仅配合 --all 使用）")
    parser.add_argument("--fresh", action="store_true",
                        help="忽略断点文件，从头开始（删除旧断点）")
    parser.add_argument("--clear-logs", action="store_true",
                        help="清空旧日志文件（决策日志+进度日志）。"
                             "默认只在 --fresh 时清进度日志；加此选项同时清决策日志，"
                             "避免多次重跑后 auto_config_decisions.jsonl 膨胀")
    args = parser.parse_args()

    # 默认 epochs
    if args.epochs is None:
        args.epochs = 200 if args.dataset or args.all else 300

    all_results = {}

    if args.analyze_only:
        # 只分析模式（也记录决策日志，方便排查）
        A_np, K, log_name, display = _get_graph_for_analysis(args)
        if A_np is not None:
            stats = analyze_graph(A_np)
            cfg_name, _, reason = select_config(stats, K)
            log_decision(log_name, stats, K, cfg_name, reason)
            print(f"\n{display}")
            print(f"图属性: N={stats['N']}, E={stats['E']}, "
                  f"avg_deg={stats['avg_deg']:.2f}, CC={stats['cc']:.4f}")
            print(f"选择: {cfg_name}")
            print(f"理由: {reason}")
        return

    if args.all:
        # 跑所有数据集和 SBM regimes
        print("\n" + "=" * 60)
        print("运行所有真实数据集 + SBM regimes")
        print("=" * 60)

        # 构建完整任务列表（顺序固定，方便断点恢复对齐）
        # 真实数据集 + SBM regimes，每个 task 含类型/名称/参数
        task_list = []
        for ds_name in DATASET_LOADERS.keys():
            task_list.append(('real', ds_name, args.seeds, args.epochs))
        for regime in list(DIFFICULTIES.keys()) + ['imbalanced']:
            n_seeds_sbm = 5 if args.seeds == 3 else args.seeds
            task_list.append(('sbm', regime, n_seeds_sbm, 300))

        total_tasks = len(task_list)

        # 断点恢复初始化
        ckpt = Checkpoint()
        if args.fresh:
            ckpt.clear()
            # 清进度日志（--fresh 默认行为：进度日志是单次运行的，重跑应清空）
            if os.path.exists(PROGRESS_LOG):
                os.remove(PROGRESS_LOG)
            # 清决策日志（需要 --clear-logs 显式开启，因为决策日志可能跨多次实验保留用于对比）
            if args.clear_logs and os.path.exists(DECISION_LOG):
                os.remove(DECISION_LOG)
                log_progress(f"实验开始（--fresh --clear-logs，从头跑 + 清空决策日志，"
                             f"{total_tasks} 个 task）")
            else:
                log_progress(f"实验开始（--fresh，从头跑，{total_tasks} 个 task）")
        elif args.resume:
            if ckpt.load():
                log_progress(f"实验恢复（--resume，{ckpt.summary()}，"
                             f"共 {total_tasks} 个 task）")
            else:
                log_progress(f"实验恢复（无断点文件，从头开始，{total_tasks} 个 task）")
        else:
            # 默认：检查是否有残留断点（上次未正常结束），提示用户
            if ckpt.load():
                print(f"\n  [checkpoint] 检测到未完成的断点：{ckpt.summary()}")
                print(f"  [checkpoint] 如需恢复，请用 --resume；如需从头跑，请用 --fresh")
                print(f"  [checkpoint] 本次默认从头开始（旧断点会被覆盖）")
                ckpt = Checkpoint()  # 重新初始化
                log_progress(f"实验开始（覆盖旧断点，{total_tasks} 个 task）")
            else:
                log_progress(f"实验开始（{total_tasks} 个 task）")

        # 逐个执行 task，每个完成后立即存断点
        for idx, (task_type, name, n_seeds, epochs) in enumerate(task_list):
            task_key = f"{task_type}_{name}"
            # 断点恢复：跳过已完成的 task
            if ckpt.is_done(task_key):
                all_results[task_key] = ckpt.get_result(task_key)
                log_progress(f"[{idx+1}/{total_tasks}] 跳过已完成: {task_key}")
                continue

            log_progress(f"[{idx+1}/{total_tasks}] 开始 {task_key} "
                        f"(seeds={n_seeds}, epochs={epochs})...")

            # 执行 task
            if task_type == 'real':
                res = run_dataset_auto(name, n_seeds=n_seeds, epochs=epochs)
            else:
                res = run_sbm_auto(name, n_seeds=n_seeds, epochs=epochs)
            all_results[task_key] = res

            # 立即存断点（即使后续崩溃，这个 task 的结果也不丢）
            ckpt.save_task(task_key, res)
            log_progress(f"[{idx+1}/{total_tasks}] 完成 {task_key}")

        # 全部完成，清理断点
        ckpt.clear()
        log_progress(f"全部 {total_tasks} 个 task 完成，断点文件已清理")
    elif args.dataset:
        res = run_dataset_auto(args.dataset, n_seeds=args.seeds, epochs=args.epochs)
        all_results[f"real_{args.dataset}"] = res
    elif args.sbm:
        n_seeds_sbm = 5 if args.seeds == 3 else args.seeds
        res = run_sbm_auto(args.sbm, n_seeds=n_seeds_sbm, epochs=args.epochs)
        all_results[f"sbm_{args.sbm}"] = res

    # 写报告
    if all_results:
        write_report(all_results, args.output)


if __name__ == "__main__":
    main()
