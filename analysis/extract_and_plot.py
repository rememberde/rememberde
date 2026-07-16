"""从实验输出日志中提取 NMI 和 SizeCV 数据，生成对比可视化。

解析 run_difficulty_sweep 生成的汇总表（格式：
"config  mod+/-std  nmi+/-std  sizecv+/-std  embstd+/-std  effrank+/-std
qcommit+/-std"），生成：
  1. image/imbalanced_scatter_from_log.png -- SizeCV vs NMI 散点图 + 柱状图
  2. 控制台打印结构化对比表

用法：python analysis/extract_and_plot.py [output_log_path]
  （不传路径则自动找最新的实验日志）
"""
import re
import sys
import glob
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
import wj as m

# 实验日志的默认搜索路径（trae-agent-toolhost 后台任务的输出位置）
DEFAULT_LOG_GLOB = (r"C:\Users\ASUS\AppData\Local\Temp\trae-agent-toolhost"
                    r"\jobs\*\output.log")


def find_latest_log():
    """找最新的实验日志文件。"""
    logs = glob.glob(DEFAULT_LOG_GLOB)
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def parse_log(log_path):
    """解析实验日志，返回嵌套 dict。

    Returns: {regime_name: {config_name: {metric: (mean, std)}}}
    """
    with open(log_path, encoding='utf-8') as f:
        text = f.read()

    # 匹配 regime 标题：##### Difficulty: easy ... ##### 或 ##### Imbalanced SBM ... #####
    regime_re = re.compile(r'#####\s*(.+?)\s*#####', re.IGNORECASE)
    # 匹配数据行：config_name  num+/-num  num+/-num ...
    # config 名是小写字母 + 数字 + 下划线
    row_re = re.compile(
        r'^(\w+)\s+'
        r'([\d.]+)\+/-([\d.]+)\s+'   # modularity
        r'([\d.]+)\+/-([\d.]+)\s+'   # nmi
        r'([\d.]+)\+/-([\d.]+)\s+'   # sizecv
        r'([\d.]+)\+/-([\d.]+)\s+'   # embstd
        r'([\d.]+)\+/-([\d.]+)\s+'   # effrank
        r'([\d.]+)\+/-([\d.]+)',     # qcommit
        re.MULTILINE)

    results = {}
    current_regime = None
    for line in text.split('\n'):
        m_regime = regime_re.search(line)
        if m_regime:
            # 提取短 regime 名
            raw = m_regime.group(1)
            if 'easy' in raw.lower():
                current_regime = 'easy'
            elif 'medium' in raw.lower():
                current_regime = 'medium'
            elif 'hard' in raw.lower():
                current_regime = 'hard'
            elif 'imbal' in raw.lower():
                current_regime = 'imbalanced'
            else:
                current_regime = raw.lower().split()[0]
            results[current_regime] = {}
            continue
        m_row = row_re.match(line)
        if m_row and current_regime:
            cfg = m_row.group(1)
            vals = [float(m_row.group(i)) for i in range(2, 14)]
            # 配对：(mean, std) for mod, nmi, sizecv, embstd, effrank, qcommit
            metrics = {}
            names = ['mod', 'nmi', 'bal', 'embstd', 'effrank', 'qcommit']
            for i, name in enumerate(names):
                metrics[name] = (vals[2 * i], vals[2 * i + 1])
            results[current_regime][cfg] = metrics
    return results


def plot_imbalanced(results):
    """从不平衡 SBM 的解析结果生成 SizeCV vs NMI 散点图 + 柱状图。"""
    if 'imbalanced' not in results:
        print("WARNING: imbalanced regime not found in log")
        return
    imb = results['imbalanced']
    configs = list(imb.keys())
    true_sizecv = m.IMBALANCED_TRUE_SIZECV   # 复用模块常量，避免重复计算

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(configs)))

    # --- 左：SizeCV vs NMI 散点 ---
    for color, name in zip(colors, configs):
        nmi_mean, nmi_std = imb[name]['nmi']
        scv_mean, scv_std = imb[name]['bal']
        ax1.errorbar(scv_mean, nmi_mean, xerr=scv_std, yerr=nmi_std,
                     fmt='o', color=color, markersize=12, capsize=5,
                     label=name, alpha=0.85, markeredgecolor='k',
                     markeredgewidth=0.5)
        ax1.annotate(name, (scv_mean, nmi_mean),
                     textcoords="offset points", xytext=(10, 6), fontsize=9,
                     color=color, fontweight='bold')
    ax1.axvline(true_sizecv, color='red', linestyle='--', alpha=0.7, linewidth=2,
                label=f'true SizeCV = {true_sizecv:.3f}')
    ax1.axvspan(true_sizecv - 0.05, true_sizecv + 0.05,
                color='red', alpha=0.08, label='true structure band')
    ax1.set_xlabel('SizeCV  (lower = more balanced)', fontsize=11)
    ax1.set_ylabel('NMI  (higher = better)', fontsize=11)
    ax1.set_title('Imbalanced SBM: size recovery vs community quality', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)

    # --- 右：NMI 柱状图，柱上标注 SizeCV ---
    x = np.arange(len(configs))
    nmi_means = [imb[c]['nmi'][0] for c in configs]
    nmi_stds = [imb[c]['nmi'][1] for c in configs]
    scv_means = [imb[c]['bal'][0] for c in configs]
    bars = ax2.bar(x, nmi_means, yerr=nmi_stds, color=colors, capsize=5,
                   edgecolor='k', linewidth=0.5, alpha=0.85)
    for bar, scv in zip(bars, scv_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'SCV\n{scv:.2f}', ha='center', va='bottom', fontsize=8)
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=20, ha='right')
    ax2.set_ylabel('NMI', fontsize=11)
    ax2.set_title('NMI ranking (SizeCV annotated on bars)', fontsize=12)
    ax2.set_ylim(0, 1.08)
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    out = m._fig_path("imbalanced_scatter_from_log.png")
    plt.savefig(out, dpi=120)
    print(f"Saved figure -> {out}")


def print_comparison(results):
    """打印所有 regime 的结构化对比表。"""
    print("\n" + "=" * 90)
    print("EXTRACTED EXPERIMENT RESULTS")
    print("=" * 90)
    for regime in ['easy', 'medium', 'hard', 'imbalanced']:
        if regime not in results:
            continue
        print(f"\n--- {regime} ---")
        print(f"  {'config':<12} {'NMI':>16} {'SizeCV':>16} "
              f"{'effRank':>16} {'embStd':>16}")
        for cfg, metrics in results[regime].items():
            nmi_m, nmi_s = metrics['nmi']
            bal_m, bal_s = metrics['bal']
            er_m, er_s = metrics['effrank']
            std_m, std_s = metrics['embstd']
            print(f"  {cfg:<12} {nmi_m:>8.4f}+/-{nmi_s:.4f} "
                  f"{bal_m:>8.4f}+/-{bal_s:.4f} "
                  f"{er_m:>8.2f}+/-{er_s:.2f} "
                  f"{std_m:>8.4f}+/-{std_s:.4f}")

    # 高亮不平衡对比
    if 'imbalanced' in results:
        imb = results['imbalanced']
        true_sizecv = m.IMBALANCED_TRUE_SIZECV
        print(f"\n--- Imbalanced SBM: SizeCV recovery (true={true_sizecv:.3f}) ---")
        for cfg in imb:
            scv = imb[cfg]['bal'][0]
            dist = abs(scv - true_sizecv)
            tag = "*** CLOSEST" if dist < 0.1 else ""
            print(f"  {cfg:<12}  SizeCV={scv:.4f}  |true|={dist:.4f}  {tag}")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_log()
    if not log_path or not os.path.exists(log_path):
        print("ERROR: no experiment log found. Run entropy_gnn_baseline.py first.")
        sys.exit(1)
    print(f"Parsing log: {log_path}")
    results = parse_log(log_path)
    if not results:
        print("ERROR: could not parse any data from log.")
        sys.exit(1)
    print(f"Found regimes: {list(results.keys())}")
    print_comparison(results)
    plot_imbalanced(results)
