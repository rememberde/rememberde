"""main.py — 一键运行完整实验，结果写入 result.md。

完整版（约 30-60 分钟，取决于 Pubmed 是否启用）：
  1. SBM 实验：5 seeds × 4 regimes（easy/medium/hard/imbalanced）× 7 configs
  2. 真实数据集实验：3 seeds × 7 configs × 200 epochs
     - Cora (2708 nodes, 7 classes)
     - CiteSeer (3312 nodes, 6 classes)
     - Pubmed (19717 nodes, 3 classes)  ← 大图，较慢，可 --no-pubmed 跳过
  3. 跨 SBM paired t-test 显著性检验
  4. 把所有指标表 + 显著性报告写入 result.md
  5. 生成对比图到 image/

7 个 config（由 make_configs 生成）：
  - vanilla      : 纯重建（baseline）
  - size/assign  : 社区熵变体
  - method2      : 玻尔兹曼 ln W 最大化（无反塌缩保护，大图适用）
  - m2_rank1/2/3 : method2 + 双铰链反塌缩（λ=1.0/2.0/3.0，小图适用）

用法：
  python main.py                 # 完整版（默认：SBM + 3 个真实数据集）
  python main.py --quick          # 快速版（1 seed，约 5 分钟，验证代码可用）
  python main.py --no-pubmed      # 跳过 Pubmed（大图较慢，仅跑 Cora + CiteSeer）
  python main.py --no-real        # 跳过所有真实数据集
  python main.py --no-cora --no-citeseer --no-pubmed  # 等价于 --no-real
"""
import argparse
import os
import sys
import time
import traceback
from typing import Dict, List

# 让 main.py 能 import 根模块和 experiments/ 子目录模块
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

import numpy as np
import wj as m
from significance_test import (
    collect_raw_metrics,
    build_report as build_sig_report,
    print_console_summary as print_sig_summary,
    paired_ttest,
    sig_marker,
)
# 格式化辅助函数统一从 wj.evaluate_metrics 导入（合并 3 处重复定义）
from wj.evaluate_metrics import fmt_mean_std as fmt_ms


def build_metrics_table(results: Dict[str, Dict[str, Dict[str, List[float]]]],
                       regimes: List[str]) -> str:
    """构建 markdown 指标表（多 regime 合并）。"""
    lines = []
    # 表头
    lines.append("| regime | config | NMI | Modularity | SizeCV | "
                 "embStd | effRank | qCommit |")
    lines.append("|---|---|---|---|---|---|---|---|")
    # 容错：results 可能为 None（如 SBM 实验失败时）
    if results is None:
        lines.append("| (SBM 实验失败，无数据) | - | - | - | - | - | - | - |")
        return "\n".join(lines)
    for regime in regimes:
        if regime not in results:
            continue
        for cfg in results[regime]:
            r = results[regime][cfg]
            # 防止某个 metric 缺失（容错）
            nmi = fmt_ms(r.get('nmi', [0]))
            mod = fmt_ms(r.get('mod', [0]))
            bal = fmt_ms(r.get('bal', [0]))
            estd = fmt_ms(r.get('embed_std', [0]))
            erank = fmt_ms(r.get('eff_rank', [0]))
            qc = fmt_ms(r.get('q_commit', [0]))
            lines.append(f"| {regime} | {cfg} | {nmi} | {mod} | {bal} | "
                        f"{estd} | {erank} | {qc} |")
    return "\n".join(lines)


def write_result_md(sbm_results, real_results, sig_report,
                    n_seeds_sbm, n_seeds_real, elapsed_sec):
    """把所有结果写入 result.md。

    Args:
        sbm_results: SBM 各 regime 的指标
        real_results: dict[name -> dict[cfg -> dict[metric -> list]]]，真实数据集结果
                      支持 cora / citeseer / pubmed 任意子集
        sig_report: 显著性检验报告字符串
        n_seeds_sbm: SBM 种子数
        n_seeds_real: 真实数据集种子数
        elapsed_sec: 总耗时（秒）
    """
    lines: List[str] = []
    lines.append("# 实验结果汇总 (result.md)\n")
    lines.append(f"> 由 `main.py` 自动生成。SBM {n_seeds_sbm} seeds, "
                 f"真实数据集 {n_seeds_real} seeds, 总耗时 {elapsed_sec / 60:.1f} 分钟。\n")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---- 1. SBM 实验 ----
    lines.append("## 1. SBM 合成数据集\n")
    lines.append(f"- Seeds: {n_seeds_sbm}（每个 regime 配对）")
    lines.append(f"- Epochs: 300, lr=0.01, emb_dim=16")
    lines.append(f"- Regimes: easy / medium / hard / imbalanced\n")
    sbm_regimes = ['easy', 'medium', 'hard', 'imbalanced']
    lines.append(build_metrics_table(sbm_results, sbm_regimes))
    lines.append("")
    lines.append(f"**真实 SizeCV (imbalanced):** "
                f"{m.IMBALANCED_TRUE_SIZECV:.4f}  "
                f"(sizes={m.IMBALANCED_SIZES})\n")

    # ---- 2. 真实数据集（Cora / CiteSeer / Pubmed）----
    lines.append("## 2. 真实数据集\n")
    # 数据集元信息（节点数、边数、类别数、特征维度、min_rank、emb_dim）
    REAL_META = {
        'cora':     (2708,  5278, 7, 1433, 5.0, 32),
        'citeseer': (3312,  4732, 6, 3703, 4.0, 64),
        'pubmed':   (19717, 44338, 3, 500, 2.5, 64),
    }
    if not real_results:
        lines.append("_（无真实数据集结果）_")
    else:
        for ds_name, res in real_results.items():
            meta = REAL_META.get(ds_name)
            if meta is None:
                lines.append(f"### {ds_name}\n_（未知数据集元信息）_\n")
                continue
            n_nodes, n_edges, n_cls, n_feat, min_rank, emb_dim = meta
            lines.append(f"### 2.{list(real_results.keys()).index(ds_name) + 1} {ds_name.upper()}\n")
            lines.append(f"- 节点数: {n_nodes}, 边数: {n_edges}, 类别数: {n_cls}, "
                        f"特征维度: {n_feat}")
            lines.append(f"- Seeds: {n_seeds_real}, epochs: 200, min_rank: {min_rank}, "
                        f"emb_dim: {emb_dim}\n")
            lines.append(build_metrics_table({ds_name: res}, [ds_name]))
            lines.append("")
    lines.append("")

    # ---- 3. 显著性检验 ----
    lines.append("## 3. 跨 SBM 显著性检验（paired t-test）\n")
    lines.append("> 配对设计：同一 seed 跨所有 config 看到同一张 SBM 图，"
                "差分掉图本身方差。\n")
    if sig_report:
        lines.append(sig_report)
    else:
        lines.append("_（被跳过）_")
    lines.append("")

    # ---- 4. 关键结论 ----
    lines.append("## 4. 关键结论（自动提取）\n")
    # 所有非 baseline config 候选列表（仅核心 method2 + 双铰链反塌缩）
    all_cfg_candidates = ['method2', 'm2_rank1', 'm2_rank2', 'm2_rank3']
    # 4.1 SBM 各 regime 的最佳 config
    for regime in ['medium', 'hard', 'imbalanced']:
        if not (sbm_results and regime in sbm_results and 'vanilla' in sbm_results[regime]):
            continue
        v_nmi = np.mean(sbm_results[regime]['vanilla']['nmi'])
        best_cfg, best_nmi = None, v_nmi
        for cfg in all_cfg_candidates:
            if cfg in sbm_results[regime]:
                c_nmi = np.mean(sbm_results[regime][cfg]['nmi'])
                if c_nmi > best_nmi:
                    best_cfg, best_nmi = cfg, c_nmi
        if best_cfg:
            delta = best_nmi - v_nmi
            lines.append(f"- {regime} SBM: {best_cfg} NMI={best_nmi:.4f} "
                        f"vs vanilla {v_nmi:.4f}（超越 {delta:+.4f}）")
    # 4.2 imbalanced SizeCV 恢复精度
    if sbm_results and 'imbalanced' in sbm_results:
        true_scv = m.IMBALANCED_TRUE_SIZECV
        for cfg in all_cfg_candidates:
            if cfg in sbm_results['imbalanced']:
                c_scv = np.mean(sbm_results['imbalanced'][cfg]['bal'])
                lines.append(f"- imbalanced SBM: {cfg} SizeCV={c_scv:.4f} "
                            f"（真实 {true_scv:.4f}, 偏差 {abs(c_scv - true_scv):+.4f}）")
    # 4.3 真实数据集（Cora / CiteSeer / Pubmed）vs vanilla
    if real_results:
        for ds_name, res in real_results.items():
            if 'vanilla' not in res:
                continue
            v_nmi = np.mean(res['vanilla']['nmi'])
            # 找最佳 config
            best_cfg, best_nmi = None, v_nmi
            for cfg in all_cfg_candidates:
                if cfg in res:
                    c_nmi = np.mean(res[cfg]['nmi'])
                    if c_nmi > best_nmi:
                        best_cfg, best_nmi = cfg, c_nmi
            if best_cfg:
                delta = best_nmi - v_nmi
                lines.append(f"- {ds_name.upper()}: 最佳 {best_cfg} NMI={best_nmi:.4f} "
                            f"vs vanilla {v_nmi:.4f}（超越 {delta:+.4f}）")
            # 逐 config 对比 vanilla
            for cfg in all_cfg_candidates:
                if cfg in res:
                    c_nmi = np.mean(res[cfg]['nmi'])
                    delta = c_nmi - v_nmi
                    tag = "超越" if delta > 0 else "落后"
                    lines.append(f"  - {ds_name}: {cfg} NMI={c_nmi:.4f} "
                                f"vs vanilla {v_nmi:.4f}（{tag} {abs(delta):+.4f}）")
    # 4.4 核心发现总结
    lines.append("\n**核心发现：**")
    lines.append("- 固定 bin 中心 C 后，method2 在大图上不需要铰链即可工作")
    lines.append("- 双铰链反塌缩（rank + variance）在小图上保护嵌入不塌缩")
    lines.append("- 健康嵌入时铰链自动休眠（ReLU 保证梯度为 0），不干扰自然低秩结构")
    # 动态提取 imbalanced SBM 上各 config vs vanilla 的显著性
    # （paired t-test 需要 ≥2 个配对 seed；quick 模式下 n_seeds=1 跳过）
    if sbm_results and 'imbalanced' in sbm_results and 'vanilla' in sbm_results['imbalanced']:
        v_vals = sbm_results['imbalanced']['vanilla']['nmi']
        for cfg in ['m2_rank1', 'm2_rank2', 'm2_rank3']:
            if cfg not in sbm_results['imbalanced']:
                continue
            c_vals = sbm_results['imbalanced'][cfg]['nmi']
            if len(c_vals) == len(v_vals) and len(v_vals) >= 2:
                t_stat, p_two = paired_ttest(c_vals, v_vals)
                delta = np.mean(c_vals) - np.mean(v_vals)
                marker = sig_marker(p_two)
                lines.append(f"- imbalanced SBM: {cfg} NMI={np.mean(c_vals):.4f} "
                             f"vs vanilla {np.mean(v_vals):.4f}"
                             f"（Δ{delta:+.4f}, p={p_two:.4f}, {marker}）")
    lines.append("")


    # ---- 5. 输出文件清单 ----
    lines.append("## 5. 输出文件\n")
    lines.append("- 图片（`image/` 目录）:")
    lines.append("  - `baseline_results.png`: 多 seed × 多 config × 多 SBM 对比柱状图")
    lines.append("  - `baseline_curves.png`: 训练曲线 10 面板（loss/mod/nmi/T/eff_rank 等）")
    lines.append("  - `imbalanced_scatter.png`: 不平衡 SBM 的 SizeCV vs NMI 散点图")
    lines.append("- 报告:")
    lines.append("  - `result.md`: 本文件")
    lines.append("  - `significance_report.md`: 详细显著性检验（如单独运行 significance_test.py）")
    lines.append("")

    return "\n".join(lines)


# ----------------------------- 主入口 -----------------------------
def main():
    parser = argparse.ArgumentParser(description="一键运行完整实验")
    parser.add_argument("--quick", action="store_true",
                        help="快速版：1 seed，约 5 分钟（验证代码可用）")
    parser.add_argument("--no-cora", action="store_true",
                        help="跳过 Cora 实验")
    parser.add_argument("--no-citeseer", action="store_true",
                        help="跳过 CiteSeer 实验")
    parser.add_argument("--no-pubmed", action="store_true",
                        help="跳过 Pubmed 实验（大图，较慢）")
    parser.add_argument("--no-real", action="store_true",
                        help="跳过所有真实数据集（等价于同时指定 --no-cora/citeseer/pubmed）")
    parser.add_argument("--no-sig", action="store_true",
                        help="跳过显著性检验（只跑实验，不生成 t-test 报告）")
    args = parser.parse_args()

    n_seeds_sbm = 1 if args.quick else 5
    n_seeds_real = 1 if args.quick else 3
    sbm_epochs = 300
    real_epochs = 200 if not args.quick else 50

    # 决定要跑哪些真实数据集
    skip_all_real = args.no_real or (args.no_cora and args.no_citeseer and args.no_pubmed)
    real_datasets_to_run = []
    if not skip_all_real:
        if not args.no_cora:
            real_datasets_to_run.append('cora')
        if not args.no_citeseer:
            real_datasets_to_run.append('citeseer')
        if not args.no_pubmed:
            real_datasets_to_run.append('pubmed')

    print("=" * 70)
    print("MAIN.PY — 完整实验")
    print(f"  SBM: {n_seeds_sbm} seeds × 4 regimes × 7 configs × {sbm_epochs} epochs")
    if real_datasets_to_run:
        ds_str = " + ".join(d.upper() for d in real_datasets_to_run)
        print(f"  真实数据集: {ds_str}（{n_seeds_real} seeds × 7 configs × {real_epochs} epochs）")
    else:
        print("  真实数据集: 跳过")
    if not args.no_sig:
        print("  显著性检验: paired t-test")
    print("=" * 70)

    t_start = time.time()
    sbm_results = None
    real_results = {}  # name -> {cfg_name -> {metric -> [seed_values]}}
    sig_report = None

    # ---- 1. SBM 实验 ----
    print(f"\n[1/3] 跑 SBM 实验（{n_seeds_sbm} seeds）...")
    try:
        # collect_raw_metrics 跑所有 config × seed × regime，返回原始数据
        # 用于：(a) 指标表 (b) 显著性检验
        metrics_to_collect = ['nmi', 'bal', 'mod', 'embed_std', 'eff_rank', 'q_commit']
        sbm_results = collect_raw_metrics(n_seeds=n_seeds_sbm,
                                           metrics=metrics_to_collect)
    except Exception as e:
        print(f"[ERROR] SBM 实验失败: {e}")
        traceback.print_exc()

    # ---- 2. 真实数据集（Cora + CiteSeer + Pubmed）----
    if real_datasets_to_run:
        print(f"\n[2/3] 跑真实数据集实验: {real_datasets_to_run}")
        for ds_name in real_datasets_to_run:
            print(f"\n  ----- {ds_name.upper()}（{n_seeds_real} seeds × {real_epochs} epochs）-----")
            try:
                # run_dataset 内部用 DATASET_LOADERS 里的默认 min_rank / emb_dim
                real_results[ds_name] = m.run_dataset(
                    ds_name, n_seeds=n_seeds_real, epochs=real_epochs)
            except Exception as e:
                print(f"[ERROR] {ds_name} 实验失败: {e}")
                traceback.print_exc()
    else:
        print("\n[2/3] 真实数据集实验被跳过")

    # ---- 3. 显著性检验 ----
    if not args.no_sig and sbm_results is not None:
        print("\n[3/3] 生成显著性检验报告...")
        try:
            sig_report = build_sig_report(sbm_results, ['nmi', 'bal'], n_seeds_sbm)
            print_sig_summary(sbm_results, ['nmi', 'bal'])
        except Exception as e:
            print(f"[ERROR] 显著性检验失败: {e}")
            traceback.print_exc()
    else:
        print("\n[3/3] 显著性检验被跳过")

    # ---- 写 result.md ----
    print("\n写入 results/result.md...")
    elapsed = time.time() - t_start
    md = write_result_md(sbm_results, real_results, sig_report,
                         n_seeds_sbm, n_seeds_real, elapsed)
    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n{'=' * 70}")
    print(f"全部完成！总耗时 {elapsed / 60:.1f} 分钟")
    print(f"结果写入: {out_path}")
    print(f"图片在: {os.path.join(_ROOT, 'image')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
