"""sota_comparison.py — WJ 方法 vs SOTA 深度图聚类对比报告生成。

读取我们的 results JSON（含 ACC/NMI/ARI）+ sota_results.json（论文引用数字），
生成 sota_comparison.md。纯报告生成，不跑任何实验。

对比范围：
  - WJ 四变体：vanilla, method2, m2_rank3, m2_cl
  - SOTA：深度图聚类（DAEGC/SDCN/AGC/MVGRL/DCRN/SCAGC）
         + 嵌入聚类（GAE/VGAE/DeepWalk/node2vec + KMeans）
  - 数据集：Cora/CiteSeer/PubMed（SOTA 论文标准数据集）
  - 指标：ACC/NMI/ARI（SOTA 论文标准指标）

用法：
  python sota/sota_comparison.py
  python sota/sota_comparison.py --our-json results/results_small.json --sota-json sota/sota_results.json
  python sota/sota_comparison.py --datasets cora citeseer pubmed
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

# sota/ 是子目录，_ROOT 上跳一级到项目根目录
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from run_wj import load_results_json
from wj.evaluate_metrics import fmt_mean_std as _fmt, mean_optional as _mean


def merge_results_jsons(paths: List[str]) -> Dict:
    """合并多个 results JSON（按数据集键合并，后加载的覆盖同名数据集）。

    用于 sota_comparison.py 读取分散在 results_small.json / results_pubmed.json
    等多个文件中的 WJ 结果。
    """
    merged = {}
    for p in paths:
        data = load_results_json(p)
        for ds, methods in data.items():
            merged[ds] = methods
    return merged

# ============================== 配置 ==============================
OUR_METHODS = ['vanilla', 'method2', 'm2_rank3', 'm2_cl', 'm2_rank3_cl']  # WJ 五变体
SOTA_DATASETS = ['cora', 'citeseer', 'pubmed']     # SOTA 论文标准数据集
METRICS = ['acc', 'nmi', 'ari']                    # SOTA 论文标准三指标


# ============================== SOTA 数据加载 ==============================
def load_sota_results(path: str) -> Dict:
    """加载 sota_results.json。

    结构：
      {
        "metadata": {...},
        "methods": {
          "方法名": {
            "source": "论文标题",
            "type": "deep_graph_clustering" | "embedding_clustering",
            "cora": {"acc": 0.0, "nmi": 0.0, "ari": 0.0},
            "citeseer": {...},
            "pubmed": {...}
          }
        }
      }
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_sota_value(sota_results: Dict, method: str, dataset: str, metric: str) -> Optional[float]:
    """从 sota_results 取某方法在某数据集某指标的值。缺失返回 None。"""
    methods = sota_results.get('methods', {})
    if method not in methods:
        return None
    ds_data = methods[method].get(dataset, {})
    return ds_data.get(metric)


def get_sota_methods(sota_results: Dict) -> List[str]:
    """获取所有 SOTA 方法名（按 methods dict 顺序）。"""
    return list(sota_results.get('methods', {}).keys())


def get_all_methods(our_results: Dict, sota_results: Dict) -> List[str]:
    """合并 WJ 方法 + SOTA 方法的完整列表。"""
    return OUR_METHODS + get_sota_methods(sota_results)


# ============================== 取值辅助 ==============================
def get_our_value(our_results: Dict, method: str, dataset: str, metric: str) -> Optional[float]:
    """从 our_results 取某方法在某数据集某指标的均值。缺失返回 None。"""
    if dataset not in our_results or method not in our_results[dataset]:
        return None
    vals = our_results[dataset][method].get(metric, [])
    valid = [v for v in vals if v is not None]
    return float(np.mean(valid)) if valid else None


def get_method_value(our_results: Dict, sota_results: Dict,
                     method: str, dataset: str, metric: str) -> Optional[float]:
    """统一取值：WJ 方法从 our_results（多 seed 均值），SOTA 从 sota_results（单值）。"""
    if method in OUR_METHODS:
        return get_our_value(our_results, method, dataset, metric)
    return get_sota_value(sota_results, method, dataset, metric)


def format_cell(our_results: Dict, sota_results: Dict,
                method: str, dataset: str, metric: str) -> str:
    """格式化单元格：WJ 显示 mean±std，SOTA 显示单值。"""
    if method in OUR_METHODS:
        # WJ 方法：多 seed，显示 mean±std
        if dataset not in our_results or method not in our_results[dataset]:
            return "N/A"
        vals = our_results[dataset][method].get(metric, [])
        return _fmt(vals) if vals else "N/A"
    # SOTA 方法：单值
    v = get_sota_value(sota_results, method, dataset, metric)
    return f"{v:.4f}" if v is not None else "N/A"


# ============================== 对比表生成 ==============================
def build_sota_metric_table(our_results: Dict, sota_results: Dict,
                            datasets: List[str], metric: str) -> str:
    """构建单个指标的 SOTA 对比表（行=方法，列=数据集，最佳加粗）。

    行 = WJ 三变体 + 所有 SOTA 方法
    列 = 数据集（Cora/CiteSeer/PubMed）
    WJ 显示 mean±std，SOTA 显示单值
    每列最佳值加粗
    """
    all_methods = get_all_methods(our_results, sota_results)
    lines = []
    header = "| 方法 | " + " | ".join(ds.upper() for ds in datasets) + " |"
    sep = "|---|" + "|".join(["---"] * len(datasets)) + "|"
    lines.append(header)
    lines.append(sep)

    # 每列最佳值
    for m in all_methods:
        # 方法名标注类型
        if m in OUR_METHODS:
            label = f"{m} (WJ)"
        else:
            mtype = sota_results.get('methods', {}).get(m, {}).get('type', '')
            tag = "深度" if mtype == 'deep_graph_clustering' else "嵌入"
            label = f"{m} ({tag})"
        cells = []
        for ds in datasets:
            cells.append(format_cell(our_results, sota_results, m, ds, metric))
        # 找每列最佳
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # 加粗每列最佳
    result_lines = lines[:2]  # header + sep
    for line in lines[2:]:
        result_lines.append(line)

    # 后处理：找每列最佳并加粗
    return _bold_best_columns(result_lines, datasets)


def _bold_best_columns(lines: List[str], datasets: List[str]) -> str:
    """对每列最佳值加粗。"""
    if len(lines) <= 2:
        return "\n".join(lines)
    n_cols = len(datasets)
    # 收集每列值
    col_vals = [[] for _ in range(n_cols)]
    for line in lines[2:]:
        parts = line.split("|")
        # parts[0]=空, parts[1]=方法名, parts[2..n_cols+1]=数据, parts[-1]=空
        for j in range(n_cols):
            cell = parts[j + 2].strip()
            # 解析数值（WJ 的 mean±std 取 mean，SOTA 的单值）
            try:
                val = float(cell.split("±")[0].replace("**", ""))
                col_vals[j].append((val, len(col_vals[j])))
            except ValueError:
                col_vals[j].append((None, len(col_vals[j])))
    # 找每列最佳（最大）
    best_idx = []
    for j in range(n_cols):
        valid = [(v, i) for v, i in col_vals[j] if v is not None]
        if valid:
            best_val = max(v for v, _ in valid)
            best_idx.append([i for v, i in valid if abs(v - best_val) < 1e-9])
        else:
            best_idx.append([])
    # 加粗
    new_lines = lines[:2]
    for row, line in enumerate(lines[2:]):
        parts = line.split("|")
        for j in range(n_cols):
            if row in best_idx[j]:
                cell = parts[j + 2].strip()
                if not cell.startswith("**"):
                    parts[j + 2] = f" **{cell}** "
        new_lines.append("|".join(parts))
    return "\n".join(new_lines)


# ============================== 差距分析 ==============================
def build_gap_analysis(our_results: Dict, sota_results: Dict,
                       datasets: List[str]) -> str:
    """以 WJ 最佳变体为基准，列出与最强 SOTA 的差距。

    对每个数据集/指标，取 WJ 四变体中表现最好的来对比 SOTA 最佳。
    状态分：领先 / 接近(差距<0.03) / 落后(0.03~0.10) / 崩溃(>0.10)
    """
    lines = ["### 差距分析（WJ 最佳变体 vs 最强 SOTA）\n"]
    lines.append("> 对每个数据集/指标，取 WJ 四变体中表现最好的来对比 SOTA 最佳。")
    lines.append("> 状态：领先 / 接近(差距<0.03) / 落后(0.03~0.10) / 崩溃(>0.10)\n")
    lines.append("| 数据集 | 指标 | WJ最佳 | WJ变体 | SOTA最佳 | SOTA方法 | 差距 | 状态 |")
    lines.append("|---|---|---|---|---|---|---|---|")

    improvements = []
    for ds in datasets:
        for metric in METRICS:
            # 找 WJ 最佳变体
            best_wj_val = None
            best_wj_method = None
            for wm in OUR_METHODS:
                v = get_our_value(our_results, wm, ds, metric)
                if v is not None and (best_wj_val is None or v > best_wj_val):
                    best_wj_val = v
                    best_wj_method = wm
            # 找最强 SOTA
            sota_methods = get_sota_methods(sota_results)
            best_sota_val = None
            best_sota_method = None
            for sm in sota_methods:
                v = get_sota_value(sota_results, sm, ds, metric)
                if v is not None and (best_sota_val is None or v > best_sota_val):
                    best_sota_val = v
                    best_sota_method = sm
            if best_wj_val is None or best_sota_val is None:
                lines.append(f"| {ds.upper()} | {metric.upper()} | N/A | - | N/A | N/A | - | - |")
                continue
            gap = best_wj_val - best_sota_val
            if gap > 0.001:
                status = "领先"
            elif gap > -0.03:
                status = "接近"
            elif gap > -0.10:
                status = "落后"
            else:
                status = "崩溃"
            lines.append(f"| {ds.upper()} | {metric.upper()} | {best_wj_val:.4f} | "
                        f"{best_wj_method} | {best_sota_val:.4f} | {best_sota_method} | "
                        f"{gap:+.4f} | {status} |")
            if status in ("落后", "崩溃"):
                improvements.append((ds, metric, gap, status))

    # 改进优先级
    if improvements:
        lines.append("\n**改进优先级（按差距从小到大）：**\n")
        improvements.sort(key=lambda x: x[2])  # 差距最小的在前（最容易改进）
        for i, (ds, metric, gap, status) in enumerate(improvements, 1):
            lines.append(f"{i}. {ds.upper()} {metric.upper()}: 差距 {gap:+.4f} ({status})")
    else:
        lines.append("\n**无落后项，WJ 在所有数据集/指标上均领先或接近 SOTA。**")

    return "\n".join(lines)


# ============================== 排名 ==============================
def build_sota_ranking(our_results: Dict, sota_results: Dict,
                       datasets: List[str]) -> str:
    """跨 WJ+SOTA 的平均排名（按 ACC 为主指标）。"""
    all_methods = get_all_methods(our_results, sota_results)
    lines = ["### 综合排名（按 ACC 排名，每数据集 1~N，求平均）\n"]
    lines.append("| 方法 | " + " | ".join(ds.upper() for ds in datasets) + " | 平均排名 |")
    lines.append("|---|" + "|".join(["---"] * (len(datasets) + 1)) + "|")

    rank_sums = {m: [] for m in all_methods}
    for ds in datasets:
        # 每方法的 ACC 值
        means = {}
        for m in all_methods:
            means[m] = get_method_value(our_results, sota_results, m, ds, 'acc')
        # 按 ACC 降序排名
        valid = {m: v for m, v in means.items() if v is not None}
        sorted_methods = sorted(valid.keys(), key=lambda m: valid[m], reverse=True)
        for rank, m in enumerate(sorted_methods, 1):
            rank_sums[m].append(rank)
        # 缺失的方法排最后
        for m in all_methods:
            if m not in valid:
                rank_sums[m].append(len(all_methods))

    for m in all_methods:
        # 方法名标注
        if m in OUR_METHODS:
            label = f"{m} (WJ)"
        else:
            mtype = sota_results.get('methods', {}).get(m, {}).get('type', '')
            tag = "深度" if mtype == 'deep_graph_clustering' else "嵌入"
            label = f"{m} ({tag})"
        ranks = rank_sums[m]
        cells = [str(r) for r in ranks]
        avg = np.mean(ranks) if ranks else float('nan')
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {avg:.1f} |")

    return "\n".join(lines)


# ============================== 完整报告 ==============================
def write_sota_comparison_md(our_results: Dict, sota_results: Dict,
                             datasets: List[str]) -> str:
    """生成完整 sota_comparison.md。"""
    n_sota = len(get_sota_methods(sota_results))
    lines = ["# WJ vs SOTA 深度图聚类对比结果\n"]
    lines.append(f"> 由 `sota_comparison.py` 自动生成。WJ 三变体 vs {n_sota} 个 SOTA 方法，"
                f"数据集：{', '.join(ds.upper() for ds in datasets)}。")
    lines.append(f"> SOTA 数字均引用自原论文（未自行复现），WJ 数字为多 seed 均值±标准差。")
    lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # SOTA 方法来源说明
    lines.append("## 1. SOTA 方法来源\n")
    lines.append("| 方法 | 类型 | 论文 |")
    lines.append("|---|---|---|")
    for m in get_sota_methods(sota_results):
        info = sota_results.get('methods', {}).get(m, {})
        mtype = info.get('type', '')
        tag = "深度图聚类" if mtype == 'deep_graph_clustering' else "嵌入+聚类"
        source = info.get('source', '?')
        lines.append(f"| {m} | {tag} | {source} |")
    lines.append("")

    # 三张指标表
    for i, metric in enumerate(METRICS, start=2):
        title = metric.upper()
        lines.append(f"## {i}. {title} 对比表\n")
        lines.append(build_sota_metric_table(our_results, sota_results, datasets, metric))
        lines.append("")

    # 差距分析
    lines.append(f"## {len(METRICS) + 2}. 差距分析\n")
    lines.append(build_gap_analysis(our_results, sota_results, datasets))
    lines.append("")

    # 排名
    lines.append(f"## {len(METRICS) + 3}. 综合排名\n")
    lines.append(build_sota_ranking(our_results, sota_results, datasets))
    lines.append("")

    # 注意事项
    lines.append(f"## {len(METRICS) + 4}. 注意事项\n")
    lines.append("- SOTA 数字均引用自原论文 Table，未自行复现")
    lines.append("- WJ 方法使用全图（不做 filter_largest_cc），和 SOTA 论文保持一致，"
                "如 CiteSeer 用全部 3312 节点")
    lines.append("- m2_rank3 的 hinge 策略：仅强社区图（CC≥0.20 且 N≤5000，如 Cora）用 λ=3.0，"
                "其余数据集（含 CiteSeer/PubMed）λ=0.0 关闭 hinge")
    lines.append("- m2_cl = method2 + 对比学习 + 特征重建（无 hinge）："
                "强社区图（CC≥0.20）用结构 CL（邻接定义正/负样本），"
                "弱社区图（CC<0.20）+高维稀疏特征（F>500，如 CiteSeer 3703 维）"
                "用混合 CL（结构+特征加权）+ PCA 降维(200) + 特征重建，"
                "弱社区图+低维特征（F≤500，如 PubMed）用纯特征 CL + 特征重建")
    lines.append("- m2_rank3_cl = m2_rank3 的 hinge + m2_cl 的 CL（组合变体）："
                "强社区图用 hinge λ=3.0 + 结构 CL（双重增强社区边界），"
                "弱社区图关闭 hinge（λ=0），CL 模式同 m2_cl")
    lines.append("- DCRN 论文不测 Cora，SDCN 论文不测 Cora/PubMed，对应单元格标 N/A")
    lines.append("- MVGRL 不报告聚类 ACC/NMI/ARI（只做节点/图分类线性评估），已从对比列表移除")
    lines.append("")

    return "\n".join(lines)


# ============================== 主入口 ==============================
def main():
    parser = argparse.ArgumentParser(description="WJ vs SOTA 深度图聚类对比报告")
    parser.add_argument("--our-json", nargs='+', default=["results/results_small.json"],
                        help="我们的 results JSON 路径（可多个，自动合并，含 ACC/NMI/ARI）")
    parser.add_argument("--sota-json", default="sota/sota_results.json",
                        help="SOTA 论文引用数据 JSON 路径")
    parser.add_argument("--datasets", nargs='+', default=SOTA_DATASETS,
                        help=f"数据集列表（默认 {SOTA_DATASETS}）")
    parser.add_argument("--output", default="results/sota_comparison.md",
                        help="输出 markdown 文件名")
    args = parser.parse_args()

    # 支持多个 JSON 文件（results_small.json + results_pubmed.json 等）
    our_json_paths = [os.path.join(_ROOT, p) for p in args.our_json]
    sota_json_path = os.path.join(_ROOT, args.sota_json)

    for p in our_json_paths:
        if not os.path.exists(p):
            print(f"[ERROR] 我们的 results JSON 不存在: {p}")
            print(f"        先跑 run_wj.py 生成结果")
            sys.exit(1)
    if not os.path.exists(sota_json_path):
        print(f"[ERROR] SOTA JSON 不存在: {sota_json_path}")
        print(f"        先创建 sota/sota_results.json（收集 SOTA 论文数据）")
        sys.exit(1)

    print("=" * 70)
    print("SOTA_COMPARISON.PY")
    print(f"  我们的 JSON: {our_json_paths}")
    print(f"  SOTA JSON:   {sota_json_path}")
    print(f"  数据集:      {args.datasets}")
    print("=" * 70)

    # 合并多个 JSON（results_small.json + results_pubmed.json 等）
    our_results = merge_results_jsons(our_json_paths)
    sota_results = load_sota_results(sota_json_path)
    n_sota = len(get_sota_methods(sota_results))
    print(f"  WJ 方法: {len(OUR_METHODS)} 个 ({', '.join(OUR_METHODS)})")
    print(f"  SOTA 方法: {n_sota} 个 ({', '.join(get_sota_methods(sota_results))})")

    md = write_sota_comparison_md(our_results, sota_results, args.datasets)
    md_path = os.path.join(_ROOT, args.output)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"\n  保存: {md_path}")
    print("完成！")


if __name__ == "__main__":
    main()
