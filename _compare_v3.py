"""三版对比：原始 → 多次KMeans → 双路KMeans+TF-IDF+Q初始化。"""
import json
import numpy as np

small = json.load(open('results/results_small.json', 'r', encoding='utf-8'))
amazon = json.load(open('results/results_amazon.json', 'r', encoding='utf-8'))
pubmed = json.load(open('results/results_pubmed.json', 'r', encoding='utf-8'))
new = {**small, **amazon, **pubmed}

# V1: 原始（单次KMeans + 二值BoW）
v1 = {
    'cora': {'vanilla': (0.6152,0.4549,0.3732), 'm2_rank3': (0.6775,0.5123,0.4344), 'm2_rank3_cl': (0.7021,0.5213,0.4710), 'm2_cl': (0.6003,0.4431,0.3433), 'method2': (0.5783,0.4431,0.3466)},
    'citeseer': {'vanilla': (0.4289,0.2078,0.1480), 'm2_rank3': (0.4380,0.2331,0.1707), 'm2_cl': (0.4382,0.2515,0.1218), 'method2': (0.4380,0.2331,0.1707), 'm2_rank3_cl': (0.4382,0.2515,0.1218)},
    'pubmed': {'vanilla': (0.6525,0.2823,0.2663), 'm2_rank3': (0.6323,0.2642,0.2356), 'm2_cl': (0.6665,0.2594,0.2697), 'method2': (0.6322,0.2642,0.2355), 'm2_rank3_cl': (0.6665,0.2594,0.2697)},
    'polblogs': {'vanilla': (0.8016,0.3837,0.3635), 'm2_rank3': (0.7979,0.3841,0.3615), 'm2_cl': (0.7831,0.3498,0.3203), 'method2': (0.8004,0.3802,0.3606), 'm2_rank3_cl': (0.8381,0.4426,0.4581)},
    'amazon_photo': {'vanilla': (0.4113,0.3500,0.1940), 'm2_rank3': (0.3646,0.3166,0.1548), 'm2_cl': (0.5554,0.5228,0.3549), 'method2': (0.3646,0.3166,0.1548), 'm2_rank3_cl': (0.5554,0.5228,0.3549)},
}

# V2: 多次KMeans取最佳（上一轮，Cora超越SOTA的版本）
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
print("三版对比：V1(原始) → V2(多次KMeans) → V3(双路KMeans+TF-IDF+Q初始化+K自适应)")
print("=" * 120)

for ds in DATASETS:
    if ds not in new:
        continue
    print(f"\n--- {ds.upper()} ---")
    print(f"{'method':<16} {'指标':<6} {'V1原始':>10} {'V2多次KM':>10} {'V3双路+TFIDF':>12} {'V3-V1':>8} {'V3-V2':>8}")
    print("-" * 80)
    for m in METHODS:
        if m not in new[ds]:
            continue
        new_vals = new[ds][m]
        for i, metric in enumerate(['acc', 'nmi', 'ari']):
            v1_v = v1.get(ds, {}).get(m, (None,None,None))[i]
            v2_v = v2.get(ds, {}).get(m, (None,None,None))[i]
            new_v = np.mean(new_vals.get(metric, [0]))
            if v1_v is None:
                continue
            d31 = new_v - v1_v
            d32 = new_v - v2_v
            print(f"{m:<16} {metric:<6} {v1_v:>10.4f} {v2_v:>10.4f} {new_v:>12.4f} {d31:>+8.4f} {d32:>+8.4f}")

# SOTA 差距
print("\n" + "=" * 120)
print("SOTA 差距（V3）")
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
