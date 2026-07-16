"""evaluate_metrics.py — 统一计算社区检测评估指标。

提供 NMI / ARI / Modularity 三指标的统一接口，供 WJ 方法和传统算法公平对比。

- NMI (Normalized Mutual Information)：信息论指标，衡量预测社区与真实社区的信息
  共享程度。范围 [0, 1]，1=完全一致。对社区数偏多的预测有偏好。
- ARI (Adjusted Rand Index)：调整兰德指数，衡量样本对在两个划分中是否被归为同类。
  范围 [-1, 1]，1=完全一致，0=随机，<0=负相关。抗随机基线，更严格。
- Modularity (Newman Q)：无监督质量指标，衡量社区内边密度 vs 随机期望。
  范围 [-0.5, 1]，越高越好。不依赖 ground truth，可跨算法公平比较。

本模块是指标计算的唯一定义点：
  - modularity() 向量化实现（原 entropy_gnn_baseline.py 迁入）
  - kmeans_labels() / size_balance() 也迁入此模块
  - fmt_mean_std() / mean_optional() 统一格式化辅助（合并 3 处重复）
  - baselines_classic.py / main.py / compare_with_baselines.py 都从此导入

注意：传统算法预测社区数 ≠ GT K 时，NMI/ARI 仍可计算（自动处理），
modularity 也不依赖 GT（基于图结构）。
"""
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans


# ----------------------------- Modularity（向量化 Newman Q） -----------------------------
def modularity(A_np: np.ndarray, labels: np.ndarray) -> float:
    """计算 Newman 模块度 Q = Σ_c [in_deg_c/(2m) - (tot_deg_c/(2m))²]。

    向量化实现：用 one-hot 矩阵 H (N,K) 一次性算出所有社区的 in_deg 和 tot_deg，
    避免 Python for 循环切片。Pubmed(N=19717) 上训练循环提速 20-30%。
    """
    m = A_np.sum() / 2.0
    if m == 0:
        return 0.0
    deg = A_np.sum(axis=1)
    labels = labels.astype(np.int64)
    K = int(labels.max()) + 1
    # one-hot (N, K)
    H = np.zeros((len(labels), K), dtype=A_np.dtype)
    H[np.arange(len(labels)), labels] = 1.0
    # in_deg[c] = Σ_{i,j in c} A[i,j]（含对角自环和双重计数，与原 for 循环一致）
    in_deg = np.einsum('ic,ij,jc->c', H, A_np, H)
    tot_deg = H.T @ deg
    Q = np.sum(in_deg / (2 * m) - (tot_deg / (2 * m)) ** 2)
    return float(Q)


def kmeans_labels(Z: np.ndarray, k: int, seed: int = 0,
                  n_init: int = None) -> np.ndarray:
    """KMeans 聚类提取社区标签。

    对大图（N>5000）自动减少 n_init 以加速：n_init=10 在 19717 节点的 Pubmed 上
    单次 KMeans 要 30+ 秒，训练循环里会调用几百次，必须优化。
    小图保持 n_init=10 保证聚类质量；大图降到 3 已足够稳定。
    """
    if n_init is None:
        n_init = 3 if Z.shape[0] > 5000 else 10
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    return km.fit_predict(Z)


def size_balance(labels: np.ndarray) -> float:
    """Coefficient of variation of community sizes (lower = more balanced).
    This is the 'microstate balance' the size-entropy term encourages."""
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.std() / (counts.mean() + 1e-8))


# ----------------------------- ACC（聚类准确率，Hungarian 匹配） -----------------------------
def cluster_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """聚类准确率 ACC，用 Hungarian 匹配解决标签排列问题。

    聚类标签编号与真实标签无对应关系，直接比 accuracy 无意义。
    用 scipy.optimize.linear_sum_assignment 在 K×K 混淆矩阵上做
    最优指派（最大化匹配样本数），再除以 N 得 ACC。

    当 pred 的类别数 ≠ true 的类别数时，D = max(两者)+1，
    混淆矩阵自动补零行/列，Hungarian 仍可求解。

    SOTA 论文（DAEGC/SDCN/DCRN/SCAGC 等）标准报告指标，与 NMI/ARI 并列。
    """
    true_labels = np.asarray(true_labels).astype(np.int64).ravel()
    pred_labels = np.asarray(pred_labels).astype(np.int64).ravel()
    D = max(pred_labels.max(), true_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(pred_labels.size):
        w[pred_labels[i], true_labels[i]] += 1
    # linear_sum_assignment 求最小，取负求最大匹配
    row, col = linear_sum_assignment(w.max() - w)
    return float(w[row, col].sum() / pred_labels.size)


# ----------------------------- 统一指标接口 -----------------------------
def compute_all_metrics(true_labels: Optional[np.ndarray],
                        pred_labels: np.ndarray,
                        A_np: np.ndarray) -> Dict[str, Optional[float]]:
    """统一计算 ACC / NMI / ARI / Modularity 四指标。

    Args:
        true_labels: 真实社区标签 (N,) int。None 时 acc/nmi/ari 返回 None
                     （如无 ground truth 的场景），mod 仍可计算。
        pred_labels: 预测社区标签 (N,) int。社区编号无需连续。
        A_np: 邻接矩阵 (N, N) float，用于计算 modularity。

    Returns:
        {'acc': float or None, 'nmi': float or None, 'ari': float or None, 'mod': float}
    """
    pred_labels = np.asarray(pred_labels).astype(np.int64).ravel()
    if true_labels is not None:
        true_labels = np.asarray(true_labels).astype(np.int64).ravel()
        acc = cluster_accuracy(true_labels, pred_labels)
        nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
        ari = float(adjusted_rand_score(true_labels, pred_labels))
    else:
        acc = None
        nmi = None
        ari = None
    mod = float(modularity(A_np, pred_labels))
    return {'acc': acc, 'nmi': nmi, 'ari': ari, 'mod': mod}


def compute_acc(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """单独计算 ACC（便捷接口，复用 Hungarian 匹配实现）。"""
    return cluster_accuracy(true_labels, pred_labels)


def compute_nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """单独计算 NMI（便捷接口）。"""
    return float(normalized_mutual_info_score(true_labels, pred_labels))


def compute_ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """单独计算 ARI（便捷接口）。"""
    return float(adjusted_rand_score(true_labels, pred_labels))


def compute_modularity(A_np: np.ndarray, pred_labels: np.ndarray) -> float:
    """单独计算 Modularity（便捷接口，复用向量化实现）。"""
    return float(modularity(A_np, pred_labels))


# ----------------------------- 格式化辅助函数（统一取代 3 处重复） -----------------------------
def fmt_mean_std(values, na_str: str = "N/A") -> str:
    """格式化 mean±std，跳过 None。

    取代：
      - main.py 的 fmt_ms（简单 mean±std，不跳 None）
      - compare_with_baselines.py 的 _fmt（跳 None 的 mean±std）

    Args:
        values: 数值列表，可含 None
        na_str: 全部为 None 时的返回字符串
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return na_str
    a = np.array(valid, dtype=float)
    return f"{a.mean():.4f}±{a.std():.4f}"


def mean_optional(values) -> float:
    """取均值，跳过 None。

    取代 compare_with_baselines.py 的 _mean。
    """
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else float('nan')


# 命令行自检：跑一个小例子确认指标计算正确
if __name__ == "__main__":
    # 构造 4 节点图，2 个社区
    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32)
    true = np.array([0, 0, 1, 1])
    # 完美预测
    perfect = compute_all_metrics(true, true, A)
    print("完美预测（true==pred）:")
    print(f"  ACC={perfect['acc']:.4f} (期望 1.0)")
    print(f"  NMI={perfect['nmi']:.4f} (期望 1.0)")
    print(f"  ARI={perfect['ari']:.4f} (期望 1.0)")
    print(f"  Mod={perfect['mod']:.4f} (期望 >0)")

    # 错误预测
    wrong_pred = np.array([0, 1, 0, 1])
    wrong = compute_all_metrics(true, wrong_pred, A)
    print("\n错误预测（全部错分）:")
    print(f"  ACC={wrong['acc']:.4f} (期望 0.5, K=2 随机基线)")
    print(f"  NMI={wrong['nmi']:.4f} (期望 0.0)")
    print(f"  ARI={wrong['ari']:.4f} (期望 <0)")
    print(f"  Mod={wrong['mod']:.4f}")

    # 无 GT 场景
    no_gt = compute_all_metrics(None, true, A)
    print("\n无 GT 场景:")
    print(f"  ACC={no_gt['acc']} (期望 None)")
    print(f"  NMI={no_gt['nmi']} (期望 None)")
    print(f"  ARI={no_gt['ari']} (期望 None)")
    print(f"  Mod={no_gt['mod']:.4f} (期望 >0)")

    # ACC 排列不变性：pred 标签整体 +1 取模 K，ACC 应不变
    pred_perm = (true + 1) % 2  # [0,0,1,1] → [1,1,0,0]，语义相同
    acc_orig = cluster_accuracy(true, true)
    acc_perm = cluster_accuracy(true, pred_perm)
    print(f"\nACC 排列不变性: orig={acc_orig:.4f}, perm={acc_perm:.4f} (期望相等)")

    # 格式化函数验证
    print("\n格式化函数验证:")
    print(f"  fmt_mean_std([0.5, 0.6, None, 0.4]) = {fmt_mean_std([0.5, 0.6, None, 0.4])}")
    print(f"  fmt_mean_std([None, None]) = {fmt_mean_std([None, None])}")
    print(f"  mean_optional([0.5, 0.6, None, 0.4]) = {mean_optional([0.5, 0.6, None, 0.4]):.4f}")

    print("\n所有指标计算正确。")
