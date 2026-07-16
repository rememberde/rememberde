"""V5 最终对比：hinge自适应归一化 + TF-IDF仅CiteSeer + Q初始化 + K自适应。"""
import json
import numpy as np

small = json.load(open('results/results_small.json', 'r', encoding='utf-8'))
amazon = json.load(open('results/results_amazon.json', 'r', encoding='utf-8'))
pubmed = json.load(open('results/results_pubmed.json', 'r', encoding='utf-8'))
new = {**small, **amazon, **pubmed}

# V2: 多次KMeans（Cora超越SOTA的版本，基线）
v2 = {
    'cora': {'vanilla': (0.6137,0.4544,0.3713), 'm2_rank3': (0.6524,0.5035,0.4074), 'm2_rank3_cl': (0.7151,0.5395,0.4932), 'm2_cl': (0.6044,0.4690,0.3635), 'method2': (0.5999,0.4485,0.3557)},
    'citeseer': {'vanilla': (0.4526,0.2407,0.1790), 'm2_rank3': (0.4462,0.2401,0.1821), 'm2_cl': (0.4402,0.2659,0.1025), 'method2': (0.4462,0.2401,0.1821), 'm2_rank3_cl': (0.4402,0.2659,0.1025)},
    'pubmed': {'vanilla': (0.6525,0.2823,0.2663), 'm2_rank3': (0.6323,0.2642,0.2356), 'm2_cl': (0.6666,0.2595,0.2698), 'method2': (0.6323,0.2642,0.2356), 'm2_rank3_cl': (0.6666,0.2595,0.2698)},
    'polblogs': {'vanilla': (0.8016,0.3837,0.3635), 'm2_rank3': (0.7980,0.3844,0.3618), 'm2_cl': (0.7819,0.3474,0.3176), 'method2': (0.8003,0.3799,0.3603), 'm2_rank3_cl': (0.8408,0.4470,0.4655)},
    'amazon_photo': {'vanilla': (0.4114,0.3504,0.1941), 'm2_rank3': (0.3645,0.3165,0.1541), 'm2_cl': (0.5586,0.5236,0.3563), 'method2': (0.3645,0.3165,0.1541), 'm2_rank3_cl': (0.5586,0.5236,0.3563)},
}

METHODS = ['vanilla', 'method2', 'm2_rank3', 'm2_cl', 'm2_rank3_cl']
DATASETS = ['cora', 'citeseer', 'pubmed', 'polblogs', 'amazon_photo']

print("=" * 120)
print("V5(最终) vs V2(基线-Cora超越SOTA)")
print("  V5: hinge自适应归一化 + TF-IDF仅CiteSeer + Q初始化 + K自适应 + n_init=10")
print("=" * 120)

total_imp = 0
total_cmp = 0
for ds in DATASETS:
    if ds not in new:
        continue
    print(f"\n--- {ds.upper()} ---")
    print(f"{'method':<16} {'指标':<6} {'V2基线':>10} {'V5最终':>10} {'delta':>8} {'提升?':<6}")
    print("-" * 65)
    for m in METHODS:
        if m not in new[ds]:
            continue
        new_vals = new[ds][m]
        for i, metric in enumerate(['acc', 'nmi', 'ari']):
            v2_v = v2.get(ds, {}).get(m, (None,None,None))[i]
            new_v = np.mean(new_vals.get(metric, [0]))
            if v2_v is None:
                continue
            delta = new_v - v2_v
            imp = "YES" if delta > 0.005 else ("=" if abs(delta) < 0.005 else "no")
            if delta > 0.005:
                total_imp += 1
            total_cmp += 1
            print(f"{m:<16} {metric:<6} {v2_v:>10.4f} {new_v:>10.4f} {delta:>+8.4f} {imp:<6}")

print(f"\n总结: {total_imp}/{total_cmp} 项提升 ({total_imp/total_cmp*100:.0f}%)")

# SOTA 差距
print("\n" + "=" * 120)
print("SOTA 差距（V5 最终）")
print("=" * 120)
sota = json.load(open('sota/sota_results.json', 'r', encoding='utf-8'))['methods']
print(f"\n{'数据集':<12} {'指标':<6} {'WJ最佳':>10} {'WJ变体':<14} {'SOTA':>10} {'SOTA方法':<14} {'差距':>8} {'状态':<6}")
print("-" * 85)
for ds in ['cora', 'citeseer', 'pubmed']:
    if ds not in new:
        continue
    for metric in ['acc', 'nmi', 'ari']:
        wj_best_val, wj_best_m = -1, '-'
        for m in METHODS:
            if m in new[ds] and new[ds][m].get(metric):
                v = np.mean(new[ds][m][metric])
                if v > wj_best_val:
                    wj_best_val, wj_best_m = v, m
        sota_best_val, sota_best_m = -1, '-'
        for sm, sd in sota.items():
            if ds in sd and sd[ds].get(metric) is not None:
                v = sd[ds][metric]
                if v > sota_best_val:
                    sota_best_val, sota_best_m = v, sm
        if wj_best_val < 0 or sota_best_val < 0:
            continue
        gap = wj_best_val - sota_best_val
        status = "领先" if gap > 0 else ("接近" if abs(gap) < 0.03 else ("落后" if abs(gap) < 0.10 else "崩溃"))
        print(f"{ds:<12} {metric:<6} {wj_best_val:>10.4f} {wj_best_m:<14} {sota_best_val:>10.4f} {sota_best_m:<14} {gap:>+8.4f} {status:<6}")
