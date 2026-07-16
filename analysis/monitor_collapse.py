"""实时塌缩监控：在 medium SBM 上对比 method2（无惩罚，会塌缩）vs m2_rank3
（带方差铰链，应恢复），每 10 epoch 打印 tr(S)、eff_rank 和铰链分量。

训练完后并排绘制 tr(S) 和 eff_rank 趋势图，可以肉眼确认：
  - 方差铰链在塌缩发生时确实触发（var_h > 0）
  - 嵌入恢复后铰链自动休眠（var_h → 0）

用法：python analysis/monitor_collapse.py
输出：image/collapse_monitor_trends.png
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
import wj as m


def run_with_verbose(name, cfg, A, labels, verbose_every=10):
    """跑一次训练，每 verbose_every epoch 打印 hinge 分量。"""
    print(f"\n--- [{name}]  lambda_rank={cfg.lambda_rank}  "
          f"min_rank={cfg.rank_min_rank}  min_var={cfg.rank_min_var} ---")
    print(f"  {'ep':>4} {'T':>7} {'tr(S)':>9} {'eff_rank':>9} "
          f"{'rank_h':>8} {'var_h':>8} {'loss':>10}")
    _, h = m.train_one(A, labels, cfg, verbose_every=verbose_every)
    final = (f"  FINAL: NMI={h['nmi'][-1]:.4f}  tr(S)={h['trS'][-1]:.4f}  "
             f"eff_rank={h['eff_rank'][-1]:.2f}  embStd={h['embed_std'][-1]:.4f}")
    print(final)
    return h


def plot_trends(hists):
    """绘制 4 联趋势图：tr(S), eff_rank, var_h, NMI。"""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    panels = [
        ('trS', 'tr(S)  (总方差, 尺度敏感)', True),
        ('eff_rank', 'eff_rank  (参与率)', True),
        ('var_h', '方差铰链  (尺度塌缩时触发)', False),
        ('nmi', 'NMI  (社区质量)', True),
    ]
    for ax, (key, title, _) in zip(axes, panels):
        for name, h in hists.items():
            ax.plot(h[key], label=name, alpha=0.85, marker='o', markersize=3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('eval step (每 10 epoch)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        if key == 'var_h':
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax.annotate('铰链休眠\n(健康)', xy=(0.7, 0.02),
                        xycoords='axes fraction', fontsize=8, color='green')
    plt.tight_layout()
    out = m._fig_path("collapse_monitor_trends.png")
    plt.savefig(out, dpi=120)
    print(f"\nSaved trend figure -> {out}")


if __name__ == "__main__":
    n_per, n_blk, p_in, p_out = m.DIFFICULTIES['medium']
    A, labels, _ = m.make_sbm(n_per_block=n_per, n_blocks=n_blk,
                              p_in=p_in, p_out=p_out, seed=0)
    K = n_blk
    cfgs = m.make_configs(K, 0)

    hists = {}
    # 1) method2：无反塌缩惩罚 -> 应该塌缩
    hists['method2'] = run_with_verbose(
        'method2 (no penalty)', cfgs['method2'], A, labels, verbose_every=10)
    # 2) m2_rank3：带方差铰链 -> 应该捕获塌缩
    hists['m2_rank3'] = run_with_verbose(
        'm2_rank3 (var hinge)', cfgs['m2_rank3'], A, labels, verbose_every=10)

    print("\n" + "=" * 70)
    print("TREND COMPARISON: 方差铰链是否阻止了塌缩?")
    print("=" * 70)
    for name, h in hists.items():
        trs = h['trS']
        print(f"\n[{name}]")
        print(f"  tr(S):    start={trs[0]:.4f}  min={min(trs):.4f}  "
              f"final={trs[-1]:.4f}")
        ers = h['eff_rank']
        print(f"  eff_rank: start={ers[0]:.2f}  min={min(ers):.2f}  "
              f"final={ers[-1]:.2f}")
        vhs = h['var_h']
        print(f"  var_h:    max={max(vhs):.3f}  final={vhs[-1]:.3f}  "
              f"{'(塌缩时触发)' if max(vhs) > 0.1 else '(休眠)'}")
        print(f"  NMI:      final={h['nmi'][-1]:.4f}")

    plot_trends(hists)
