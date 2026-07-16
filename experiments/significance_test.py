"""跨 SBM paired t-test 显著性检验。

对每个 regime（easy/medium/hard/imbalanced）和每个指标（NMI, SizeCV）：
  1. 生成 n_seeds 个 SBM 实例（同一 seed 跨所有 config -> 配对）
  2. 每个 config 在每个 seed 上训练
  3. 对每对 config 做 two-sided paired t-test（同 SBM 差分掉图本身方差）
  4. 报告 mean+/-std, t 统计量, p 值, 显著性标记（***/**/*/ns）

配对设计很重要：每个 config 看到同一张 SBM 图，所以主要方差源（随机图）
被差分掉，比非配对检验给出更紧的 p 值。

同时报告单侧 p 值（"X > vanilla" / "X < vanilla"），因为我们有方向性假设
（m2_rank 应在 NMI 上超 vanilla；应在 imbalanced 上更贴近真实 SizeCV）。

输出：控制台打印 + 写 markdown 表到 significance_report.md

用法：python experiments/significance_test.py [--n_seeds 5] [--metrics nmi,bal]
注意：CPU 密集（n_seeds × n_regimes × n_configs 次完整训练），应单独运行。
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from scipy import stats

from wj import (
    make_configs, train_one, make_sbm, make_imbalanced_sbm,
    DIFFICULTIES, IMBALANCED_SIZES, IMBALANCED_P, METRIC_KEYS,
)


# ----------------------------- 实验运行器 -----------------------------
def collect_raw_metrics(n_seeds: int, metrics: List[str]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """运行所有 config × seed，收集每个 seed 的原始指标值。

    Returns:
        results[regime][config][metric] = [v_seed0, v_seed1, ...]
    """
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    # 平衡 regime
    for diff_name, (n_per, n_blk, p_in, p_out) in DIFFICULTIES.items():
        K = n_blk
        cfgs = make_configs(K, 0)
        results[diff_name] = {name: {m: [] for m in metrics} for name in cfgs}
        print(f"\n=== {diff_name}  (n_per_block={n_per}, K={K}, "
              f"p_in={p_in}, p_out={p_out}) ===")
        for s in range(n_seeds):
            A_np, labels, _ = make_sbm(n_per_block=n_per, n_blocks=n_blk,
                                       p_in=p_in, p_out=p_out, seed=s)
            for name, cfg in make_configs(K, s).items():
                _, h = train_one(A_np, labels, cfg)
                for m in metrics:
                    results[diff_name][name][m].append(h[m][-1])
            print(f"  seed {s} done")

    # 不平衡 regime
    p_in, p_out = IMBALANCED_P
    K = len(IMBALANCED_SIZES)
    cfgs = make_configs(K, 0)
    results["imbalanced"] = {name: {m: [] for m in metrics} for name in cfgs}
    print(f"\n=== imbalanced  (sizes={IMBALANCED_SIZES}, p_in={p_in}, p_out={p_out}) ===")
    for s in range(n_seeds):
        A_np, labels, _ = make_imbalanced_sbm(IMBALANCED_SIZES,
                                              p_in=p_in, p_out=p_out, seed=s)
        for name, cfg in make_configs(K, s).items():
            _, h = train_one(A_np, labels, cfg)
            for m in metrics:
                results["imbalanced"][name][m].append(h[m][-1])
        print(f"  seed {s} done")

    return results


# ----------------------------- 显著性检验 -----------------------------
def sig_marker(p: float) -> str:
    """p 值转显著性标记。"""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """双侧 paired t-test。返回 (t_stat, p_value)。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b), f"paired test needs equal lengths, got {len(a)} vs {len(b)}"
    diff = a - b
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_rel(a, b)
    return float(t_stat), float(p_val)


def one_sided_p(t_stat: float, p_two_sided: float, alternative: str) -> float:
    """把双侧 p 值转成单侧（方向性假设）。"""
    if alternative == "greater":   # H1: a > b
        return p_two_sided / 2.0 if t_stat > 0 else 1.0 - p_two_sided / 2.0
    if alternative == "less":      # H1: a < b
        return p_two_sided / 2.0 if t_stat < 0 else 1.0 - p_two_sided / 2.0
    return p_two_sided


# ----------------------------- 报告生成 -----------------------------
def format_mean_std(xs: List[float]) -> str:
    a = np.asarray(xs)
    return f"{a.mean():.4f}+/-{a.std():.4f}"


def build_report(results: Dict, metrics: List[str], n_seeds: int) -> str:
    """构建 markdown 显著性报告。"""
    lines: List[str] = []
    lines.append(f"# Cross-SBM Significance Report\n")
    lines.append(f"- Seeds per regime: **{n_seeds}** (paired across configs)")
    lines.append(f"- Metrics: {', '.join(metrics)}")
    lines.append(f"- Test: two-sided **paired** t-test (`scipy.stats.ttest_rel`)")
    lines.append(f"- Baselines for comparison: `vanilla` (no entropy) and `method2` (entropy, no anti-collapse)")
    lines.append(f"- Significance: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` otherwise\n")

    for regime in results:
        configs = list(results[regime].keys())
        lines.append(f"## Regime: `{regime}`\n")
        for metric in metrics:
            lines.append(f"### Metric: `{metric}`\n")
            # 表头：每个 config 的 mean+/-std
            lines.append("| config | mean+/-std |")
            lines.append("|---|---|")
            for c in configs:
                vals = results[regime][c][metric]
                lines.append(f"| {c} | {format_mean_std(vals)} |")
            lines.append("")

            # 与 vanilla 和 method2 两个基线的配对检验
            lines.append(f"#### Pairwise paired t-test (`{metric}`, regime=`{regime}`)\n")
            lines.append("| comparison | t_stat | p (two-sided) | sig | "
                         "p (X>baseline, one-sided) | p (X<baseline, one-sided) |")
            lines.append("|---|---|---|---|---|---|")
            for baseline in ["vanilla", "method2"]:
                if baseline not in results[regime]:
                    continue
                base_vals = results[regime][baseline][metric]
                for c in configs:
                    if c == baseline:
                        continue
                    c_vals = results[regime][c][metric]
                    t_stat, p_two = paired_ttest(c_vals, base_vals)
                    p_gt = one_sided_p(t_stat, p_two, "greater")
                    p_lt = one_sided_p(t_stat, p_two, "less")
                    lines.append(
                        f"| {c} vs {baseline} | {t_stat:+.3f} | {p_two:.4f} | "
                        f"{sig_marker(p_two)} | {p_gt:.4f} | {p_lt:.4f} |"
                    )
            lines.append("")
        lines.append("---\n")

    # 汇总：哪些 config 在 NMI 上显著超过 vanilla
    lines.append("## Summary: configs that significantly beat `vanilla` on NMI\n")
    lines.append("| regime | config | NMI mean | vs vanilla NMI mean | "
                 "p (X>vanilla) | sig |")
    lines.append("|---|---|---|---|---|---|")
    for regime in results:
        if "vanilla" not in results[regime]:
            continue
        v_vals = results[regime]["vanilla"]["nmi"]
        for c in results[regime]:
            if c == "vanilla":
                continue
            c_vals = results[regime][c]["nmi"]
            t_stat, p_two = paired_ttest(c_vals, v_vals)
            p_gt = one_sided_p(t_stat, p_two, "greater")
            lines.append(
                f"| {regime} | {c} | {np.mean(c_vals):.4f} | "
                f"{np.mean(v_vals):.4f} | {p_gt:.4f} | {sig_marker(p_gt)} |"
            )
    lines.append("")
    return "\n".join(lines)


def print_console_summary(results: Dict, metrics: List[str]):
    """紧凑控制台汇总（完整 markdown 写文件）。"""
    print("\n" + "=" * 70)
    print("SIGNIFICANCE TEST SUMMARY (paired t-test vs vanilla, NMI)")
    print("=" * 70)
    for regime in results:
        if "vanilla" not in results[regime]:
            continue
        v_vals = results[regime]["vanilla"]["nmi"]
        print(f"\n[{regime}]  vanilla NMI = {np.mean(v_vals):.4f}")
        for c in results[regime]:
            if c == "vanilla":
                continue
            c_vals = results[regime][c]["nmi"]
            t_stat, p_two = paired_ttest(c_vals, v_vals)
            p_gt = one_sided_p(t_stat, p_two, "greater")
            delta = np.mean(c_vals) - np.mean(v_vals)
            print(f"  {c:10s}  NMI={np.mean(c_vals):.4f}  "
                  f"delta={delta:+.4f}  t={t_stat:+.2f}  "
                  f"p(X>vanilla)={p_gt:.4f}  {sig_marker(p_gt)}")


# ----------------------------- 主入口 -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="number of SBM seeds per regime (paired across configs)")
    parser.add_argument("--metrics", type=str, default="nmi,bal",
                        help="comma-separated metrics to test (nmi, bal, mod, eff_rank, ...)")
    parser.add_argument("--out", type=str, default="significance_report.md",
                        help="output markdown report path")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",")]
    print(f"Running significance test: n_seeds={args.n_seeds}, "
          f"metrics={metrics}, out={args.out}")

    results = collect_raw_metrics(args.n_seeds, metrics)

    print_console_summary(results, metrics)

    report = build_report(results, metrics, args.n_seeds)
    # 报告写到项目根目录（脚本在 experiments/ 子目录）
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
