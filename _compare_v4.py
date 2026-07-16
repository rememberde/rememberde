"""最终对比 V4：TF-IDF仅CiteSeer + n_init=10 + 双路KMeans + Q初始化。"""
import json
import numpy as np

small = json.load(open('results/results_small.json', 'r', encoding='utf-8'))
amazon = json.load(open('results/results_amazon.json', 'r', encoding='utf-8'))
pubmed = json.load(open('results/results_pubmed.json', 'r', encoding='utf-8'))
new = {**small, **amazon, **pubmed}

# V2: 多次KMeans（Cora超越SOTA的版本，二值BoW + n_init=10）
v2 = {
    'cora': {'vanilla': (0.6137,0.4544,0.3713), 'm2_rank3': (0.6524,0.5035,0.4074), 'm2_rank3_cl': (0.7151,0.5395,0.4932), 'm2_cl': (0.6044,0.4690,0.3635), 'method2': (0.5999,0.4485,0.3557)},
    'citeseer': {'vanilla': (0.4526,0.2407,0.1790), 'm2_rank3': (0.4462,0.2401,0.1821), 'm2_cl': (0.4402,0.2659,0.1025), 'method2': (0.4462,0.2401,0.1821), 'm2_rank3_cl': (0.4402,0.2659,0.1025)},
    'pubmed': {'vanilla': (0.6525,0.2823,0.2663), 'm2_rank3': (0.6323,0.2642,0.2356), 'm2_cl': (0.6666,0.2595,0.2698), 'method2': (0.6323,0.2642,0.2356), 'm2_rank3_cl': (0.6666,0.2595,0.2698)},
    'polblogs': {'vanilla': (0.8016,0.3837,0.3635), 'm2_rank3': (0.7980,0.3844,0.3618), 'm2_cl': (0.7819,0.3474,0.3176), 'method2': (0.8003,0.3799,0.3603), 'm2_rank3_cl': (0.8408,0.4470,0.4655)},
    'amazon_photo': {'vanilla': (0.4114,0.3504,0.1941), 'm2_rank3': (0.3645,0.3165,0.1541), 'm2_cl': (0.5586,0.5236,0.3563), 'method2': (0.3645,0.3165,0.1541), 'm2_rank3_cl': (0.5586,0.5236,0.3563)},
}

# V3（上一轮，TF-IDF全部 + n_init=5）
v3 = {
    'cora': {'vanilla': (0.6493,0.4892,0.3997), 'm2_rank3': (0.6621,0.4964,0.4257), 'm2_rank3_cl': (0.6642,0.5065,0.4338), 'm2_cl': (0.6214,0.4791,0.3793), 'method2': (0.6529,0.4928,0.4107)},
    'citeseer': {'vanilla': (0.4374,0.2415,0.1436), 'm2_rank3': (0.4298,0.2362,0.1381), 'm2_cl': (0.5146,0.3189,0.2191), 'method2': (0.4298,0.2362,0.1381), 'm2_rank3_cl': (0.5146,0.3189,0.2191)},
    'pubmed': {'vanilla': (0.6684,0.2829,0.2883), 'm2_rank3': (0.6448,0.2586,0.2517), 'm2_cl': (0.6477,0.2187,0.2333), 'method2': (0.6448,0.2586,0.2517), 'm2_rank3_cl': (0.6477,0.2187,0.2333)},
    'polblogs': {'vanilla': (0.8694,0.4776,0.5455), 'm2_rank3': (0.7983,0.3783,0.4214), 'm2_cl': (0.8687,0.4858,0.5440), 'method2': (0.8675,0.4731,0.5400), 'm2_rank3_cl': (0.8821,0.5138,0.5865)},
    'amazon_photo': {'vanilla': (0.5031,0.4387,0.3006), 'm2_rank3': (0.4262,0.4101,0.2538), 'm2_cl': (0.6863,0.5859,0.4760), 'method2': (0.4262,0.4101,0.2538), 'm2_rank3_cl': (0.6863,0.5859,0.4760)},
}

METHODS = ['vanilla', 'method2', 'm2_rank3', 'm2_cl', 'm2_rank3_cl']
DATASETS = ['cora', 'citeseer', 'pubmed', 'polblogs', 'amazon_photo']

print("=" * 120)
print("V4(最终) vs V2(Cora超越SOTA) vs V3(TF-IDF全部)")
print("  V4: TF-IDF仅CiteSeer + n_init=10 + 双路KMeans + Q初始化 + K自适应")
print("=" * 120)

for ds in DATASETS:
    if ds not in new:
        continue
    print(f"\n--- {ds.upper()} ---")
    print(f"{'method':<16} {'指标':<6} {'V2基线':>10} {'V3全TFIDF':>10} {'V4最终':>10} {'V4-V2':>8} {'V4-V3':>8}")
    print("-" * 75)
    for m in METHODS:
        if m not in new[ds]:
            continue
        new_vals = new[ds][m]
        for i, metric in enumerate(['acc', 'nmi', 'ari']):
            v2_v = v2.get(ds, {}).get(m, (None,None,None))[i]
            v3_v = v3.get(ds, {}).get(m, (None,None,None))[i]
            new_v = np.mean(new_vals.get(metric, [0]))
            if v2_v is None:
                continue
            d42 = new_v - v2_v
            d43 = new_v - v3_v
            print(f"{m:<16} {metric:<6} {v2_v:>10.4f} {v3_v:>10.4f} {new_v:>10.4f} {d42:>+8.4f} {d43:>+8.4f}")

# SOTA 差距
print("\n" + "=" * 120)
print("SOTA 差距（V4 最终）")
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
