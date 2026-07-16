"""extract_nmi_summary.py — 从 auto_config_result.md 提取所有数据集 NMI 生成对比表

功能
====
  1. 解析 auto_config_result.md（或通过 --input 指定其他文件）
  2. 提取每个数据集的：图属性、选择的 config、vanilla NMI、auto NMI、Δ NMI
  3. 生成汇总对比表（markdown + CSV）
  4. 按数据集类型（真实 / SBM）分组，统计胜率

用法
====
  python extract_nmi_summary.py                          # 默认读 auto_config_result.md
  python extract_nmi_summary.py --input other.md         # 指定输入
  python extract_nmi_summary.py --output my_summary.md   # 指定输出
  python extract_nmi_summary.py --csv-only               # 只输出 CSV

输出文件
========
  - nmi_comparison.md  : markdown 对比表（人读）
  - nmi_comparison.csv : CSV 格式（程序读、导入 Excel）
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np


# ----------------------------- Markdown 解析 -----------------------------
# 匹配数据集段头：### REAL_CORA 或 ### SBM_EASY
RE_SECTION = re.compile(r'^###\s+(.+)$')

# 匹配图属性行：N=2708, E=5278, avg_deg=3.90, CC=0.2407
RE_GRAPH = re.compile(r'N=(\d+),\s*E=(\d+),\s*avg_deg=([\d.]+),\s*CC=([\d.]+)')

# 匹配选择 config 行：**选择 config**: `m2_rank3`
RE_CONFIG = re.compile(r'\*\*选择 config\*\*:\s*`(\w+)`')

# 匹配 Δ NMI 行：**Δ NMI (auto - vanilla): +0.0588**（支持科学计数法）
RE_DELTA = re.compile(r'Δ NMI.*:\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)')


def parse_mean_std(s: str):
    """解析 '0.4708±0.0063' 或 '1.2e-1±2e-3' 格式，返回 (mean, std) 元组。

    支持科学计数法和负数；解析失败返回 None（让 caller 跳过该行而非崩溃）。
    """
    s = s.strip()
    if '±' in s:
        mean_str, std_str = s.split('±', 1)
        try:
            return float(mean_str), float(std_str)
        except ValueError:
            return None
    try:
        return float(s), 0.0
    except ValueError:
        return None


def parse_table_row(line: str):
    """解析 markdown 表格行，返回 (config_name, [cell_strs...]) 或 None。

    用 `|` 分割而非固定列数正则，对列数变化更鲁棒（即使 write_report
    增减列也不会崩溃）。跳过分隔行(|---|)和表头(非数据行)。
    """
    parts = [p.strip() for p in line.strip().strip('|').split('|')]
    if len(parts) < 2:
        return None
    # 跳过分隔行 |---|---|
    if all(set(p) <= set('-: ') for p in parts):
        return None
    config = parts[0]
    # config 必须是合法标识符（vanilla/m2_rank3 等）
    if not re.match(r'^\w+$', config):
        return None
    # 至少有一个数据 cell 能解析成数字才算数据行（跳过表头 config/NMI/...）
    if not parse_mean_std(parts[1]):
        return None
    return config, parts[1:]


def parse_result_md(filepath: str):
    """解析 auto_config_result.md，返回数据集列表。

    每个数据集是 dict：
      {
        'name': 'REAL_CORA',
        'N': 2708, 'E': 5278, 'avg_deg': 3.90, 'cc': 0.2407,
        'config': 'm2_rank3',
        'vanilla_nmi': (0.4708, 0.0063),
        'auto_nmi': (0.5296, 0.0259),
        'delta': 0.0588,
      }
    """
    datasets = []
    current = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 段头
            m = RE_SECTION.match(line)
            if m and ('REAL_' in m.group(1) or 'SBM_' in m.group(1)):
                if current:
                    datasets.append(current)
                current = {'name': m.group(1)}
                continue

            if current is None:
                continue

            # 图属性
            m = RE_GRAPH.search(line)
            if m:
                current['N'] = int(m.group(1))
                current['E'] = int(m.group(2))
                current['avg_deg'] = float(m.group(3))
                current['cc'] = float(m.group(4))
                continue

            # 选择 config
            m = RE_CONFIG.search(line)
            if m:
                current['config'] = m.group(1)
                continue

            # 表格行（用 split 解析，列数变化也鲁棒）
            parsed = parse_table_row(line)
            if parsed:
                cfg_name, cells = parsed
                # NMI 是第一个数据列（cells[0]）
                nmi = parse_mean_std(cells[0]) if cells else None
                if nmi is None:
                    continue
                if cfg_name == 'vanilla':
                    current['vanilla_nmi'] = nmi
                else:
                    current['auto_nmi'] = nmi
                    current['auto_config'] = cfg_name
                continue

            # Δ NMI
            m = RE_DELTA.search(line)
            if m:
                current['delta'] = float(m.group(1))
                continue

    # 最后一个
    if current:
        datasets.append(current)

    return datasets


# ----------------------------- 表格生成 -----------------------------
def generate_markdown_table(datasets):
    """生成 markdown 对比表。"""
    lines = []
    lines.append("# NMI 对比汇总表\n")
    lines.append(f"> 自动提取自 auto_config_result.md，{len(datasets)} 个数据集。\n")

    # 按类型分组
    real_ds = [d for d in datasets if d['name'].startswith('REAL_')]
    sbm_ds = [d for d in datasets if d['name'].startswith('SBM_')]

    for group_name, group in [("真实数据集", real_ds), ("SBM 合成数据", sbm_ds)]:
        if not group:
            continue
        lines.append(f"## {group_name}\n")
        lines.append("| 数据集 | N | avg_deg | CC | 自动 config | "
                     "Vanilla NMI | Auto NMI | Δ NMI | 结论 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for d in group:
            name = d['name'].replace('REAL_', '').replace('SBM_', '')
            v_nmi = f"{d['vanilla_nmi'][0]:.4f}±{d['vanilla_nmi'][1]:.4f}"
            a_nmi = f"{d['auto_nmi'][0]:.4f}±{d['auto_nmi'][1]:.4f}"
            delta = d.get('delta', 0.0)
            verdict = "✅ 提升" if delta > 0.001 else ("≈ 持平" if abs(delta) <= 0.001 else "⚠️ 下降")
            lines.append(f"| {name} | {d.get('N', '-')} | {d.get('avg_deg', '-'):.2f} | "
                        f"{d.get('cc', '-'):.4f} | `{d.get('config', '-')}` | "
                        f"{v_nmi} | {a_nmi} | {delta:+.4f} | {verdict} |")
        lines.append("")

    # 总体统计
    lines.append("## 总体统计\n")
    wins = sum(1 for d in datasets if d.get('delta', 0) > 0.001)
    ties = sum(1 for d in datasets if abs(d.get('delta', 0)) <= 0.001)
    losses = len(datasets) - wins - ties
    total_delta = sum(d.get('delta', 0) for d in datasets)
    avg_delta = total_delta / len(datasets) if datasets else 0

    lines.append(f"- 数据集总数: {len(datasets)}")
    lines.append(f"- 提升 / 持平 / 下降: {wins} / {ties} / {losses}")
    lines.append(f"- 平均 Δ NMI: {avg_delta:+.4f}")
    lines.append(f"- 总 Δ NMI: {total_delta:+.4f}")
    lines.append(f"- 最大提升: {max((d.get('delta', 0) for d in datasets), default=0):+.4f}"
                f"（{max(datasets, key=lambda x: x.get('delta', 0))['name'] if datasets else '-'}）")
    lines.append("")

    return "\n".join(lines)


def generate_csv(datasets, filepath):
    """生成 CSV 文件。"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'type', 'N', 'E', 'avg_deg', 'cc',
                         'selected_config', 'vanilla_nmi_mean', 'vanilla_nmi_std',
                         'auto_nmi_mean', 'auto_nmi_std', 'delta_nmi'])
        for d in datasets:
            ds_type = 'real' if d['name'].startswith('REAL_') else 'sbm'
            writer.writerow([
                d['name'],
                ds_type,
                d.get('N', ''),
                d.get('E', ''),
                d.get('avg_deg', ''),
                d.get('cc', ''),
                d.get('config', ''),
                d['vanilla_nmi'][0], d['vanilla_nmi'][1],
                d['auto_nmi'][0], d['auto_nmi'][1],
                d.get('delta', 0.0),
            ])


# ----------------------------- 主入口 -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="从 auto_config_result.md 提取 NMI 结果生成对比表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", default="auto_config_result.md",
                        help="输入文件（默认 auto_config_result.md）")
    parser.add_argument("--output", default="nmi_comparison.md",
                        help="markdown 输出文件（默认 nmi_comparison.md）")
    parser.add_argument("--csv", default="nmi_comparison.csv",
                        help="CSV 输出文件（默认 nmi_comparison.csv）")
    parser.add_argument("--csv-only", action="store_true",
                        help="只输出 CSV，不生成 markdown")
    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"[错误] 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    # 解析
    print(f"[解析] 读取 {args.input}...")
    datasets = parse_result_md(args.input)
    print(f"[解析] 提取到 {len(datasets)} 个数据集")

    if not datasets:
        print("[错误] 未提取到任何数据集，请检查文件格式", file=sys.stderr)
        sys.exit(1)

    # 打印摘要
    print(f"\n{'数据集':<20} {'config':<12} {'vanilla':<16} {'auto':<16} {'Δ':<8}")
    print("-" * 75)
    for d in datasets:
        name = d['name'].replace('REAL_', '').replace('SBM_', '')
        v = f"{d['vanilla_nmi'][0]:.4f}"
        a = f"{d['auto_nmi'][0]:.4f}"
        delta = f"{d.get('delta', 0):+.4f}"
        print(f"{name:<20} {d.get('config', '-'):<12} {v:<16} {a:<16} {delta:<8}")

    # 生成 CSV
    generate_csv(datasets, args.csv)
    print(f"\n[CSV] 已写入 {args.csv}")

    # 生成 markdown
    if not args.csv_only:
        md_content = generate_markdown_table(datasets)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"[markdown] 已写入 {args.output}")


if __name__ == "__main__":
    main()
