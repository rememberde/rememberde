"""演示为什么纯秩惩罚在尺度不变性下失效，以及方差铰链如何修复。

构造一个健康嵌入，然后逐步缩小（Z -> alpha*Z），保持特征值比率不变。展示：
  - eff_rank 是平坦的（尺度不变）-> 纯秩惩罚无法检测塌缩
  - tr(S) 和 embStd 降到 0 -> 方差铰链激活并捕获塌缩

这是 anticollapse 模块设计的核心动机证明：参与率 (tr S)²/tr(S²) 在 Z->c·Z
下不变（c² 在分子分母间抵消），所以纯秩惩罚可以让模型通过"各维度等比缩小"
来满足秩约束。tr(S) 是尺度敏感的，补上了这个盲点。

用法：python analysis/demo_scale_invariance.py
输出：image/scale_invariance_demo.png
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

# 直接用 anticollapse 模块的核心函数，避免内联重复实现
from wj.anticollapse import effective_rank, total_variance, VarianceHinge
import wj as m   # 复用 _fig_path 统一图片输出路径


def demo():
    torch.manual_seed(42)
    N, d = 120, 16

    # 健康嵌入：4 个簇在 4 维子空间里，其余维度接近 0
    Z_base = torch.zeros(N, d)
    for i in range(N):
        block = i // 30                  # 4 个块，每块 30 个节点
        Z_base[i, block] = 1.0 + 0.3 * torch.randn(1).item()
        Z_base[i, 4] = 0.2 * torch.randn(1).item()   # 第 5 维加小噪声

    # 扫描尺度：alpha 从 0.01（塌缩）到 1.0（健康）
    alphas = np.linspace(0.01, 1.0, 50)
    eff_ranks, trSs, emb_stds = [], [], []
    pure_rank_pen, var_hinge_pen, combined_pen = [], [], []

    # 用 VarianceHinge 实例计算铰链分量（与训练时使用的代码路径一致）
    hinge = VarianceHinge(min_rank=2.5, min_var=1.0)

    for a in alphas:
        Z = a * Z_base
        # 核心信号全部来自 anticollapse 模块，不内联重复
        eff_rank = effective_rank(Z).item()
        trS = total_variance(Z).item()
        emb_std = Z.std().item()
        d_hinge = hinge.diagnostics(Z)

        # 纯秩惩罚（旧方案）：-eff_rank / d，尺度不变 -> 对 alpha 盲
        pure = -eff_rank / d
        # 方差铰链（新方案）：d_hinge.var_def，尺度敏感 -> alpha 小时激活
        var_h = d_hinge.var_def
        # 组合铰链：rank_def + var_def
        combined = d_hinge.penalty

        eff_ranks.append(eff_rank)
        trSs.append(trS)
        emb_stds.append(emb_std)
        pure_rank_pen.append(pure)
        var_hinge_pen.append(var_h)
        combined_pen.append(combined)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: eff_rank 和 tr(S) 随尺度的变化
    ax = axes[0]
    ax.plot(alphas, eff_ranks, 'b-o', label='eff_rank  (尺度不变)', markersize=4)
    ax.plot(alphas, [t / max(trSs) * max(eff_ranks) for t in trSs],
            'r-s', label='tr(S)  [归一化, 尺度敏感]', markersize=4)
    ax.axhline(2.5, color='gray', linestyle=':', alpha=0.5, label='min_rank=2.5')
    ax.set_xlabel(r'尺度因子 $\alpha$  (Z $\rightarrow$ $\alpha$·Z)')
    ax.set_ylabel('值')
    ax.set_title('参与率的尺度不变性\n(eff_rank 平坦, tr(S) 下降)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: 惩罚值随尺度的变化
    ax = axes[1]
    ax.plot(alphas, pure_rank_pen, 'b--', label='纯秩惩罚  (旧, 平坦)', linewidth=2)
    ax.plot(alphas, var_hinge_pen, 'r-', label='方差铰链  (新, 激活)', linewidth=2)
    ax.plot(alphas, combined_pen, 'g-', label='组合铰链  (新)', linewidth=2)
    ax.set_xlabel(r'尺度因子 $\alpha$')
    ax.set_ylabel('惩罚值')
    ax.set_title('惩罚对尺度塌缩的响应\n(纯秩=盲, 方差铰链=捕获)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: 两个 eff_rank 相同但 tr(S) 不同的具体嵌入
    ax = axes[2]
    Z_healthy = Z_base.clone()
    Z_collapsed = 0.05 * Z_base.clone()  # 缩小 20x，特征值比率不变
    for Z, label, color, ypos in [(Z_healthy, '健康\n(alpha=1.0)', 'green', 0.8),
                                   (Z_collapsed, '塌缩\n(alpha=0.05)', 'red', 0.3)]:
        # 同样用模块函数，避免重复
        er = effective_rank(Z).item()
        tr = total_variance(Z).item()
        std = Z.std().item()
        pure = -er / d
        d_h = hinge.diagnostics(Z)
        vh = d_h.var_def
        ax.barh(ypos, er, height=0.08, color=color, alpha=0.6, label=f'eff_rank={er:.2f}')
        ax.barh(ypos - 0.12, tr, height=0.08, color=color, alpha=0.9, hatch='//',
                label=f'tr(S)={tr:.3f}')
        ax.text(max(er, tr) + 0.1, ypos,
                f'{label}\neff_rank={er:.2f}  tr(S)={tr:.3f}\n'
                f'pure_rank_pen={pure:.4f}  var_hinge={vh:.3f}',
                fontsize=8, va='center', color=color, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, 6)
    ax.set_xlabel('值')
    ax.set_title('两个 eff_rank 相同的嵌入\n但其中一个已塌缩 (tr(S) -> 0)')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    out = m._fig_path("scale_invariance_demo.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved figure -> {out}")

    # 打印汇总表
    print("\n=== 尺度不变性演示 ===")
    print(f"{'alpha':>7} {'eff_rank':>10} {'tr(S)':>10} {'embStd':>8} "
          f"{'pure_rank':>12} {'var_hinge':>12} {'combined':>12}")
    for i in [0, 10, 25, 40, 49]:
        a = alphas[i]
        print(f"{a:>7.3f} {eff_ranks[i]:>10.4f} {trSs[i]:>10.4f} {emb_stds[i]:>8.4f} "
              f"{pure_rank_pen[i]:>12.4f} {var_hinge_pen[i]:>12.4f} {combined_pen[i]:>12.4f}")


if __name__ == "__main__":
    demo()
