"""analyze_hard_collapse.py — 分析 hard SBM 上 method2 无正则的塌缩过程。

跑 3 个 config（vanilla / method2 / m2_rank1）× hard SBM，每 10 epoch 记录：
- loss / E / S（损失分量）
- eff_rank / tr(S) / embed_std（塌缩指标）
- nmi / mod（社区质量）

输出：
- image/hard_collapse_curves.png：6 面板曲线对比
- 打印关键 epoch 的指标表格，定位塌缩发生的 epoch
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 让子目录脚本能 import 根目录模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# matplotlib 配置（Agg 后端 + 中文字体）统一走 wj.plot_config
from wj import plot_config

import wj as m
from wj import (
    make_sbm, make_configs, train_one, DIFFICULTIES, _fig_path,
)


def run_config_capture_history(cfg_name, cfg, A_np, labels):
    """跑一个 config，返回完整的 history dict。"""
    print(f"  [{cfg_name}] 训练 {cfg.epochs} epochs ...")
    model, hist = train_one(A_np, labels, cfg, verbose_every=50)
    # 计算每 10 epoch 的 epoch 编号
    eval_epochs = list(range(0, cfg.epochs, cfg.eval_every))
    if (cfg.epochs - 1) not in eval_epochs:
        eval_epochs.append(cfg.epochs - 1)
    hist['epochs'] = eval_epochs
    print(f"  [{cfg_name}] 完成。最终: NMI={hist['nmi'][-1]:.4f}  "
          f"effRank={hist['eff_rank'][-1]:.4f}  tr(S)={hist['trS'][-1]:.4f}")
    return hist


def plot_collapse_curves(hists, save_path):
    """画 6 面板曲线对比图。

    面板：loss / eff_rank / tr(S) / nmi / embed_std / T
    每条线一个 config，x 轴是 epoch。
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        ('loss', 'Loss（总损失）', True),
        ('eff_rank', 'Effective Rank（低=塌缩）', True),
        ('trS', 'tr(S) 总方差（低=塌缩）', True),
        ('nmi', 'NMI vs GT（社区质量）', True),
        ('embed_std', 'Embedding Std（低=塌缩）', True),
        ('T', 'Temperature T（熵项权重）', True),
    ]
    colors = {
        'vanilla': '#1f77b4', 'method2': '#d62728', 'm2_rank1': '#2ca02c',
        'm2_rank2': '#ff7f0e', 'm2_rank3': '#9467bd',
    }

    for ax, (key, title, _) in zip(axes.flat, panels):
        for cfg_name, hist in hists.items():
            if key not in hist:
                continue
            epochs = hist['epochs']
            vals = hist[key]
            ax.plot(epochs, vals, label=cfg_name, color=colors.get(cfg_name, None),
                    linewidth=2, marker='o', markersize=3)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Hard SBM 塌缩过程分析（p_in=0.3, p_out=0.10, K=4）',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"\n保存图片 -> {save_path}")
    plt.close()


def print_collapse_table(hists):
    """打印关键 epoch 的指标表格，定位塌缩发生的 epoch。"""
    print("\n" + "=" * 90)
    print("Hard SBM 塌缩过程指标表（每 10 epoch）")
    print("=" * 90)
    # 表头
    cfgs = list(hists.keys())
    header = f"{'epoch':>6} | "
    for cfg in cfgs:
        header += f"{cfg+' loss':>12} {cfg+' effR':>10} {cfg+' trS':>8} {cfg+' nmi':>8} | "
    print(header)
    print("-" * len(header))

    # 找最长的 epoch 列表
    max_len = max(len(h['epochs']) for h in hists.values())
    for i in range(max_len):
        row = f"{list(hists.values())[0]['epochs'][i]:>6} | "
        for cfg in cfgs:
            h = hists[cfg]
            if i < len(h['epochs']):
                row += f"{h['loss'][i]:>12.4f} {h['eff_rank'][i]:>10.4f} {h['trS'][i]:>8.4f} {h['nmi'][i]:>8.4f} | "
            else:
                row += " " * 44 + "| "
        print(row)

    print("=" * 90)
    # 塌缩诊断
    if 'method2' in hists:
        h = hists['method2']
        # 找 eff_rank 第一次跌破 1.5 的 epoch
        collapse_ep = None
        for i, (ep, er) in enumerate(zip(h['epochs'], h['eff_rank'])):
            if er < 1.5 and i > 0:
                collapse_ep = ep
                break
        if collapse_ep is not None:
            print(f"\n[诊断] method2 eff_rank 在 epoch {collapse_ep} 跌破 1.5（开始塌缩）")
            idx = h['epochs'].index(collapse_ep)
            print(f"  当时: loss={h['loss'][idx]:.4f}  eff_rank={h['eff_rank'][idx]:.4f}  "
                  f"tr(S)={h['trS'][idx]:.4f}  NMI={h['nmi'][idx]:.4f}  T={h['T'][idx]:.4f}")
            # 对比前一个 epoch
            if idx > 0:
                print(f"  前:   loss={h['loss'][idx-1]:.4f}  eff_rank={h['eff_rank'][idx-1]:.4f}  "
                      f"tr(S)={h['trS'][idx-1]:.4f}  NMI={h['nmi'][idx-1]:.4f}  T={h['T'][idx-1]:.4f}")
        else:
            print("\n[诊断] method2 eff_rank 未跌破 1.5（未塌缩或一直很低）")
        # 最终状态
        print(f"\n[最终] method2: eff_rank={h['eff_rank'][-1]:.4f}  "
              f"tr(S)={h['trS'][-1]:.4f}  NMI={h['nmi'][-1]:.4f}")


def main():
    print("=" * 70)
    print("分析 Hard SBM 上 method2 无正则的塌缩过程")
    print("=" * 70)

    # Hard SBM 参数
    n_per, n_blk, p_in, p_out = DIFFICULTIES['hard']
    print(f"\nHard SBM: n_per_block={n_per}, K={n_blk}, p_in={p_in}, p_out={p_out}")

    # 生成一张固定的 hard SBM 图
    A_np, labels, _ = make_sbm(n_per_block=n_per, n_blocks=n_blk,
                                p_in=p_in, p_out=p_out, seed=0)
    print(f"图: {A_np.shape[0]} 节点, {A_np.sum()/2:.0f} 条边")

    # 跑 5 个核心 config，用同一个 seed 保证可比
    K = n_blk
    all_cfgs = make_configs(K, seed=0, epochs=300, min_rank=3.0)
    target_cfgs = ['vanilla', 'method2', 'm2_rank1', 'm2_rank2', 'm2_rank3']
    hists = {}
    for name in target_cfgs:
        hists[name] = run_config_capture_history(name, all_cfgs[name], A_np, labels)

    # 画曲线
    save_path = _fig_path('hard_collapse_curves.png')
    plot_collapse_curves(hists, save_path)

    # 打印表格 + 塌缩诊断
    print_collapse_table(hists)

    print(f"\n图片已保存: {save_path}")
    print("=" * 70)
    print("分析完成。")
    print("=" * 70)


if __name__ == "__main__":
    main()
