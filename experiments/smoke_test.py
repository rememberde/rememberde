"""快速冒烟测试：验证 WJ 训练链路在 SBM 和 Cora 上的可用性。

默认跑 SBM 冒烟测试（medium/hard/imbalanced，1 seed × 全 config，快速检查
秩惩罚是否修复了塌缩、是否伤了不平衡优势）。
加 --cora 切换到 Cora 数据集冒烟测试（1 seed × vanilla/method2+rank × 50 epochs，
验证 load_cora 和 train_one(X_feat=...) 路径通畅）。

正式实验请用 run_wj.py（多 seed）。

用法：
  python experiments/smoke_test.py            # SBM 冒烟测试
  python experiments/smoke_test.py --cora     # Cora 冒烟测试
"""
import argparse
import os
import sys
import time

# 让子目录脚本能 import 根目录的模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import wj as m


# ============================== SBM 冒烟测试 ==============================
def run_one(A, labels, K, seed=0):
    """对一个 SBM 实例跑全部 config，打印关键指标。"""
    print(f"{'config':<10} {'Mod':>8} {'NMI':>8} {'SizeCV':>8} "
          f"{'effRank':>8} {'embStd':>8} {'qCommit':>8} {'rankPen':>10}")
    rows = []
    for name, cfg in m.make_configs(K, seed).items():
        _, h = m.train_one(A, labels, cfg)
        row = (name, h['mod'][-1], h['nmi'][-1], h['bal'][-1],
               h['eff_rank'][-1], h['embed_std'][-1], h['q_commit'][-1],
               h['rank_pen'][-1])
        rows.append(row)
        print(f"{name:<10} {row[1]:>8.4f} {row[2]:>8.4f} {row[3]:>8.4f} "
              f"{row[4]:>8.2f} {row[5]:>8.3f} {row[6]:>8.3f} {row[7]:>10.4f}")
    return rows


def smoke_sbm():
    """SBM 冒烟测试：medium/hard/imbalanced 各跑 1 seed。"""
    for diff in ['medium', 'hard']:
        print(f"=== SMOKE TEST: {diff} difficulty (1 seed) ===")
        n_per, n_blk, p_in, p_out = m.DIFFICULTIES[diff]
        A, labels, _ = m.make_sbm(n_per_block=n_per, n_blocks=n_blk,
                                  p_in=p_in, p_out=p_out, seed=0)
        K = n_blk
        print(f"N={len(labels)}, K={K}")
        run_one(A, labels, K)
        print()

    print("=== SMOKE TEST: imbalanced SBM (1 seed) ===")
    A2, labels2, _ = m.make_imbalanced_sbm(
        m.IMBALANCED_SIZES, p_in=m.IMBALANCED_P[0], p_out=m.IMBALANCED_P[1], seed=0)
    K2 = len(m.IMBALANCED_SIZES)
    print(f"N={len(labels2)}, K={K2}, sizes={np.bincount(labels2)}")
    run_one(A2, labels2, K2)


# ============================== Cora 冒烟测试 ==============================
def smoke_cora():
    """Cora 冒烟测试：1 seed × (vanilla + method2+rank) × 50 epochs。"""
    print("=== 测试 1: load_cora ===")
    A, labels, features = m.load_cora()
    print(f"  A shape: {A.shape}, labels shape: {labels.shape}, features shape: {features.shape}")
    print(f"  节点数: {A.shape[0]}, 边数: {int(A.sum() / 2)}, 类别数: {len(set(labels))}")
    print(f"  特征维度: {features.shape[1]}, 特征稀疏度: {features.mean():.4f}")

    print("\n=== 测试 2: vanilla 50 epochs ===")
    cfg = m.TrainConfig(n_communities=7, hidden_dim=64, emb_dim=16,
                        epochs=50, lr=0.01, seed=0, entropy="none")
    t0 = time.time()
    _, h = m.train_one(A, labels, cfg, X_feat=features)
    t1 = time.time()
    print(f"  耗时: {t1 - t0:.1f}s")
    print(f"  final  NMI={h['nmi'][-1]:.4f}  Mod={h['mod'][-1]:.4f}  "
          f"SizeCV={h['bal'][-1]:.4f}  effRank={h['eff_rank'][-1]:.2f}  "
          f"embStd={h['embed_std'][-1]:.4f}")

    print("\n=== 测试 3: method2 + rank 正则 50 epochs ===")
    cfg2 = m.TrainConfig(n_communities=7, hidden_dim=64, emb_dim=16,
                         epochs=50, lr=0.01, seed=0, entropy="method2",
                         T_max=0.3, anneal="cosine", T_warmup=0.2,
                         n_bins=16, sigma=0.5, lambda_rank=3.0,
                         rank_min_rank=4.0, rank_min_var=1.0)
    t0 = time.time()
    _, h2 = m.train_one(A, labels, cfg2, X_feat=features, verbose_every=10)
    t1 = time.time()
    print(f"  耗时: {t1 - t0:.1f}s")
    print(f"  final  NMI={h2['nmi'][-1]:.4f}  Mod={h2['mod'][-1]:.4f}  "
          f"SizeCV={h2['bal'][-1]:.4f}  effRank={h2['eff_rank'][-1]:.2f}  "
          f"embStd={h2['embed_std'][-1]:.4f}  rankPen={h2['rank_pen'][-1]:.4f}")

    print("\n冒烟测试通过。")


# ============================== 主入口 ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WJ 冒烟测试（SBM / Cora 二选一）")
    parser.add_argument("--cora", action="store_true",
                        help="跑 Cora 数据集冒烟测试（默认跑 SBM）")
    args = parser.parse_args()

    if args.cora:
        smoke_cora()
    else:
        smoke_sbm()
