"""plot_config.py — matplotlib 全局配置（Agg 后端 + 中文字体）。

统一取代 3 处重复的字体配置代码：
  - entropy_gnn_baseline.py（原 L61-75，现由 experiments.py import 此模块）
  - compare_with_baselines.py（原 L42-49）
  - analysis/analyze_hard_collapse.py（原 L23-30）

用法：在任何 import matplotlib.pyplot 之前 `import plot_config`，
import 的副作用即应用 Agg 后端 + 中文字体配置。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# 用 Agg 后端避免无显示环境报错（Windows 也兼容）
mpl.use('Agg')

# matplotlib 中文字体配置：Windows 上尝试 Microsoft YaHei / SimHei，
# 让图表里的中文标签正常显示。找不到则回退到默认字体（中文会显示为方块）。
for _font in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    try:
        mpl.font_manager.findfont(_font, fallback_to_default=False)
        mpl.rcParams["font.sans-serif"] = [_font] + mpl.rcParams["font.sans-serif"]
        mpl.rcParams["axes.unicode_minus"] = False  # 负号显示
        break
    except Exception:
        continue
