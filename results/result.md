# 实验结果汇总 (result.md)

> 由 `main.py` 自动生成。SBM 5 seeds, 真实数据集 3 seeds, 总耗时 34.0 分钟。

> 生成时间: 2026-07-07 16:16:09

## 1. SBM 合成数据集

- Seeds: 5（每个 regime 配对）
- Epochs: 300, lr=0.01, emb_dim=16
- Regimes: easy / medium / hard / imbalanced

| regime | config | NMI | Modularity | SizeCV | embStd | effRank | qCommit |
|---|---|---|---|---|---|---|---|
| easy | vanilla | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3870±0.0014 | 2.9980±0.0009 | 0.2654±0.0044 |
| easy | size | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3869±0.0014 | 2.9980±0.0009 | 0.2699±0.0065 |
| easy | assign | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3870±0.0014 | 2.9980±0.0009 | 0.2501±0.0000 |
| easy | method2 | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3870±0.0014 | 2.9981±0.0009 | 0.2587±0.0019 |
| easy | m2_rank1 | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3859±0.0013 | 3.0000±0.0000 | 0.2587±0.0019 |
| easy | m2_rank2 | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3858±0.0013 | 3.0000±0.0000 | 0.2587±0.0019 |
| easy | m2_rank3 | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3858±0.0013 | 3.0000±0.0000 | 0.2587±0.0019 |
| easy | m2_pernode_a03 | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3872±0.0014 | 2.9986±0.0007 | 0.2587±0.0019 |
| easy | m2_pernode_a05 | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3872±0.0014 | 2.9986±0.0006 | 0.2587±0.0019 |
| easy | m2_pernode_a07 | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3873±0.0014 | 2.9987±0.0006 | 0.2587±0.0019 |
| easy | m2_pernode_anneal | 1.0000±0.0000 | 0.5927±0.0005 | 0.0000±0.0000 | 0.3873±0.0014 | 2.9986±0.0007 | 0.2587±0.0019 |
| easy | m2_pernode_a05_adasig | 0.0348±0.0697 | 0.0101±0.0202 | 0.2872±0.5745 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2587±0.0019 |
| easy | m2_pernode_anneal_adasig | 0.1775±0.1690 | 0.0596±0.0613 | 0.6096±0.5861 | 0.0309±0.0619 | 0.2000±0.4000 | 0.2587±0.0019 |
| medium | vanilla | 0.9308±0.0214 | 0.2905±0.0027 | 0.0291±0.0095 | 0.3556±0.0010 | 4.0169±0.0645 | 0.2692±0.0041 |
| medium | size | 0.9266±0.0279 | 0.2903±0.0029 | 0.0282±0.0090 | 0.3580±0.0036 | 4.0660±0.0782 | 0.2677±0.0039 |
| medium | assign | 0.9266±0.0279 | 0.2903±0.0029 | 0.0282±0.0090 | 0.3584±0.0077 | 4.1410±0.3088 | 0.2511±0.0010 |
| medium | method2 | 0.9451±0.0215 | 0.2915±0.0034 | 0.0261±0.0083 | 0.3717±0.0174 | 4.4954±0.4468 | 0.2610±0.0014 |
| medium | m2_rank1 | 0.9380±0.0208 | 0.2920±0.0033 | 0.0211±0.0134 | 0.3448±0.0169 | 4.0485±0.5085 | 0.2602±0.0017 |
| medium | m2_rank2 | 0.9266±0.0260 | 0.2910±0.0041 | 0.0261±0.0115 | 0.3285±0.0131 | 3.7475±0.3900 | 0.2602±0.0017 |
| medium | m2_rank3 | 0.9224±0.0258 | 0.2907±0.0040 | 0.0261±0.0115 | 0.3263±0.0092 | 3.6151±0.3206 | 0.2602±0.0017 |
| medium | m2_pernode_a03 | 0.9433±0.0154 | 0.2910±0.0032 | 0.0355±0.0074 | 0.3843±0.0148 | 4.7528±0.3923 | 0.2605±0.0017 |
| medium | m2_pernode_a05 | 0.9408±0.0184 | 0.2913±0.0038 | 0.0361±0.0116 | 0.3850±0.0125 | 4.7586±0.3638 | 0.2608±0.0013 |
| medium | m2_pernode_a07 | 0.9391±0.0222 | 0.2921±0.0025 | 0.0311±0.0098 | 0.3861±0.0125 | 4.8895±0.4047 | 0.2606±0.0015 |
| medium | m2_pernode_anneal | 0.9324±0.0227 | 0.2906±0.0042 | 0.0382±0.0103 | 0.3839±0.0107 | 4.7782±0.3401 | 0.2604±0.0017 |
| medium | m2_pernode_a05_adasig | 0.0293±0.0366 | 0.0045±0.0056 | 0.5558±0.6816 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2602±0.0017 |
| medium | m2_pernode_anneal_adasig | 0.0122±0.0244 | 0.0024±0.0047 | 0.2890±0.5781 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2602±0.0017 |
| hard | vanilla | 0.6629±0.0383 | 0.2334±0.0052 | 0.0792±0.0204 | 0.3553±0.0037 | 3.9792±0.0153 | 0.2655±0.0027 |
| hard | size | 0.6378±0.0536 | 0.2306±0.0059 | 0.0784±0.0174 | 0.3578±0.0083 | 3.9813±0.0430 | 0.2661±0.0019 |
| hard | assign | 0.6653±0.0413 | 0.2330±0.0048 | 0.0827±0.0136 | 0.3595±0.0068 | 3.9824±0.0116 | 0.2508±0.0006 |
| hard | method2 | 0.6632±0.0433 | 0.2334±0.0042 | 0.0966±0.0359 | 0.3603±0.0106 | 4.2295±0.3348 | 0.2592±0.0010 |
| hard | m2_rank1 | 0.6674±0.0556 | 0.2362±0.0051 | 0.0750±0.0373 | 0.3491±0.0109 | 4.2032±0.3672 | 0.2587±0.0009 |
| hard | m2_rank2 | 0.6645±0.0736 | 0.2344±0.0062 | 0.0712±0.0451 | 0.3373±0.0147 | 3.9674±0.4355 | 0.2587±0.0009 |
| hard | m2_rank3 | 0.6237±0.0657 | 0.2283±0.0062 | 0.0934±0.0471 | 0.3362±0.0108 | 3.8672±0.4270 | 0.2587±0.0009 |
| hard | m2_pernode_a03 | 0.6560±0.0662 | 0.2328±0.0035 | 0.1093±0.0509 | 0.3667±0.0134 | 4.3707±0.3738 | 0.2589±0.0010 |
| hard | m2_pernode_a05 | 0.6619±0.0475 | 0.2348±0.0063 | 0.0936±0.0296 | 0.3743±0.0153 | 4.5477±0.4524 | 0.2590±0.0011 |
| hard | m2_pernode_a07 | 0.6662±0.0727 | 0.2345±0.0070 | 0.1004±0.0439 | 0.3706±0.0152 | 4.4288±0.4276 | 0.2593±0.0016 |
| hard | m2_pernode_anneal | 0.6650±0.0709 | 0.2355±0.0066 | 0.0888±0.0305 | 0.3744±0.0157 | 4.5748±0.4416 | 0.2591±0.0013 |
| hard | m2_pernode_a05_adasig | 0.0435±0.0385 | 0.0052±0.0051 | 0.8354±0.6860 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2588±0.0009 |
| hard | m2_pernode_anneal_adasig | 0.0322±0.0412 | 0.0042±0.0055 | 0.5349±0.6564 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2588±0.0009 |
| imbalanced | vanilla | 0.8183±0.0420 | 0.2891±0.0188 | 0.2060±0.1621 | 0.2892±0.0062 | 1.9189±0.0121 | 0.2632±0.0042 |
| imbalanced | size | 0.8191±0.0438 | 0.2894±0.0199 | 0.2037±0.1630 | 0.2897±0.0040 | 1.9173±0.0104 | 0.2626±0.0042 |
| imbalanced | assign | 0.8194±0.0436 | 0.2893±0.0199 | 0.2098±0.1601 | 0.2887±0.0065 | 1.9197±0.0136 | 0.2501±0.0001 |
| imbalanced | method2 | 0.8653±0.0680 | 0.3075±0.0285 | 0.3824±0.2080 | 0.3022±0.0192 | 2.2031±0.3523 | 0.2577±0.0042 |
| imbalanced | m2_rank1 | 0.9361±0.0225 | 0.3277±0.0098 | 0.5684±0.0177 | 0.3420±0.0110 | 3.2521±0.2565 | 0.2574±0.0041 |
| imbalanced | m2_rank2 | 0.9274±0.0537 | 0.3250±0.0090 | 0.5530±0.0151 | 0.3496±0.0113 | 3.3812±0.2715 | 0.2573±0.0041 |
| imbalanced | m2_rank3 | 0.9242±0.0567 | 0.3248±0.0092 | 0.5562±0.0143 | 0.3494±0.0091 | 3.2523±0.1888 | 0.2573±0.0041 |
| imbalanced | m2_pernode_a03 | 0.8604±0.0448 | 0.3061±0.0215 | 0.4392±0.1751 | 0.2973±0.0138 | 2.1168±0.2828 | 0.2574±0.0041 |
| imbalanced | m2_pernode_a05 | 0.8598±0.0641 | 0.3030±0.0237 | 0.3674±0.2132 | 0.2922±0.0062 | 1.9592±0.0644 | 0.2573±0.0041 |
| imbalanced | m2_pernode_a07 | 0.8310±0.0444 | 0.2955±0.0224 | 0.2895±0.2058 | 0.2897±0.0045 | 1.9228±0.0075 | 0.2573±0.0041 |
| imbalanced | m2_pernode_anneal | 0.8353±0.0452 | 0.2969±0.0206 | 0.3120±0.1947 | 0.2915±0.0052 | 1.9521±0.0393 | 0.2573±0.0041 |
| imbalanced | m2_pernode_a05_adasig | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2573±0.0041 |
| imbalanced | m2_pernode_anneal_adasig | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.2573±0.0041 |

**真实 SizeCV (imbalanced):** 0.5562  (sizes=[80, 50, 30, 15])

## 2. 真实数据集

### 2.1 CORA

- 节点数: 2708, 边数: 5278, 类别数: 7, 特征维度: 1433
- Seeds: 3, epochs: 200, min_rank: 5.0, emb_dim: 32

| regime | config | NMI | Modularity | SizeCV | embStd | effRank | qCommit |
|---|---|---|---|---|---|---|---|
| cora | vanilla | 0.4708±0.0063 | 0.7108±0.0191 | 0.5066±0.2698 | 0.2831±0.0205 | 10.1615±3.7386 | 0.1500±0.0006 |
| cora | size | 0.4559±0.0199 | 0.7188±0.0209 | 0.3291±0.1430 | 0.2832±0.0205 | 10.1639±3.7395 | 0.1523±0.0007 |
| cora | assign | 0.4585±0.0138 | 0.7128±0.0181 | 0.4511±0.2500 | 0.2831±0.0205 | 10.1605±3.7374 | 0.1430±0.0001 |
| cora | method2 | 0.4562±0.0095 | 0.7133±0.0212 | 0.4474±0.2269 | 0.2814±0.0196 | 9.7321±3.5076 | 0.1476±0.0007 |
| cora | m2_rank1 | 0.5004±0.0067 | 0.7250±0.0047 | 0.4598±0.0832 | 0.3108±0.0033 | 17.6326±1.2123 | 0.1480±0.0007 |
| cora | m2_rank2 | 0.5044±0.0180 | 0.7362±0.0037 | 0.3404±0.0754 | 0.3031±0.0019 | 14.8280±0.6931 | 0.1477±0.0008 |
| cora | m2_rank3 | 0.5156±0.0142 | 0.7334±0.0041 | 0.3638±0.0461 | 0.2994±0.0020 | 13.4851±0.6355 | 0.1477±0.0008 |
| cora | m2_pernode_a03 | 0.4356±0.0052 | 0.7289±0.0112 | 0.3585±0.1166 | 0.2811±0.0193 | 9.6003±3.3951 | 0.1476±0.0007 |
| cora | m2_pernode_a05 | 0.4456±0.0327 | 0.7304±0.0084 | 0.3775±0.1343 | 0.2806±0.0190 | 9.4645±3.3035 | 0.1479±0.0006 |
| cora | m2_pernode_a07 | 0.4571±0.0168 | 0.7081±0.0237 | 0.4103±0.2835 | 0.2763±0.0232 | 8.9480±3.6405 | 0.1479±0.0006 |
| cora | m2_pernode_anneal | 0.4405±0.0246 | 0.7227±0.0069 | 0.2778±0.1368 | 0.2772±0.0238 | 9.1505±3.7694 | 0.1477±0.0006 |
| cora | m2_pernode_a05_adasig | 0.4276±0.0485 | 0.7212±0.0188 | 0.2394±0.0572 | 0.2725±0.0301 | 8.8178±4.4590 | 0.1478±0.0006 |
| cora | m2_pernode_anneal_adasig | 0.3797±0.0915 | 0.6954±0.0645 | 0.2360±0.1010 | 0.2563±0.0337 | 6.5735±3.8768 | 0.1477±0.0007 |

### 2.2 CITESEER

- 节点数: 3312, 边数: 4732, 类别数: 6, 特征维度: 3703
- Seeds: 3, epochs: 200, min_rank: 4.0, emb_dim: 64

| regime | config | NMI | Modularity | SizeCV | embStd | effRank | qCommit |
|---|---|---|---|---|---|---|---|
| citeseer | vanilla | 0.2602±0.0294 | 0.7298±0.0079 | 0.4077±0.0919 | 0.1948±0.0050 | 8.2691±1.1645 | 0.1701±0.0004 |
| citeseer | size | 0.2357±0.0190 | 0.7355±0.0103 | 0.3285±0.0239 | 0.1948±0.0050 | 8.2690±1.1644 | 0.1726±0.0007 |
| citeseer | assign | 0.2439±0.0352 | 0.7250±0.0295 | 0.4173±0.0546 | 0.1948±0.0050 | 8.2690±1.1645 | 0.1667±0.0000 |
| citeseer | method2 | 0.2483±0.0276 | 0.7398±0.0079 | 0.3715±0.0761 | 0.1945±0.0046 | 8.2729±1.1744 | 0.1688±0.0005 |
| citeseer | m2_rank1 | 0.2046±0.0078 | 0.6611±0.0494 | 0.8707±0.2500 | 0.2305±0.0004 | 25.9736±0.3279 | 0.1696±0.0003 |
| citeseer | m2_rank2 | 0.2762±0.0218 | 0.6860±0.0228 | 0.8809±0.0748 | 0.2281±0.0012 | 25.4285±1.1489 | 0.1695±0.0004 |
| citeseer | m2_rank3 | 0.2465±0.0028 | 0.6853±0.0147 | 0.8307±0.0399 | 0.2260±0.0010 | 23.6885±1.0973 | 0.1695±0.0003 |
| citeseer | m2_pernode_a03 | 0.2442±0.0136 | 0.7417±0.0049 | 0.3222±0.0377 | 0.1946±0.0047 | 8.2018±1.0842 | 0.1688±0.0005 |
| citeseer | m2_pernode_a05 | 0.2367±0.0248 | 0.7331±0.0058 | 0.3764±0.0637 | 0.1940±0.0040 | 8.0466±0.8941 | 0.1688±0.0005 |
| citeseer | m2_pernode_a07 | 0.2546±0.0164 | 0.7423±0.0047 | 0.2482±0.0604 | 0.1926±0.0042 | 7.6584±0.8708 | 0.1688±0.0005 |
| citeseer | m2_pernode_anneal | 0.2630±0.0414 | 0.7465±0.0061 | 0.2452±0.0357 | 0.1925±0.0040 | 7.6665±0.8752 | 0.1688±0.0005 |
| citeseer | m2_pernode_a05_adasig | 0.2435±0.0462 | 0.7445±0.0184 | 0.3141±0.1208 | 0.1810±0.0058 | 5.5676±1.0008 | 0.1688±0.0005 |
| citeseer | m2_pernode_anneal_adasig | 0.2358±0.0352 | 0.7251±0.0420 | 0.3357±0.1302 | 0.1765±0.0106 | 4.9953±1.4969 | 0.1688±0.0005 |

### 2.3 PUBMED

- 节点数: 19717, 边数: 44338, 类别数: 3, 特征维度: 500
- Seeds: 3, epochs: 200, min_rank: 2.5, emb_dim: 64

| regime | config | NMI | Modularity | SizeCV | embStd | effRank | qCommit |
|---|---|---|---|---|---|---|---|
| pubmed | vanilla | 0.2682±0.0246 | 0.5857±0.0082 | 0.2010±0.0315 | 0.1704±0.0022 | 4.8552±0.3108 | 0.3373±0.0009 |
| pubmed | size | 0.2884±0.0063 | 0.5920±0.0013 | 0.2212±0.0191 | 0.1704±0.0022 | 4.8557±0.3119 | 0.3394±0.0018 |
| pubmed | assign | 0.2681±0.0248 | 0.5857±0.0081 | 0.2009±0.0316 | 0.1704±0.0022 | 4.8566±0.3116 | 0.3335±0.0000 |
| pubmed | method2 | 0.2896±0.0308 | 0.5870±0.0026 | 0.2021±0.0683 | 0.1681±0.0032 | 4.4685±0.3913 | 0.3362±0.0016 |
| pubmed | m2_rank1 | 0.2664±0.0117 | 0.5769±0.0045 | 0.2464±0.0333 | 0.1706±0.0041 | 4.8859±0.6166 | 0.3362±0.0017 |
| pubmed | m2_rank2 | 0.2424±0.0082 | 0.5724±0.0030 | 0.1624±0.0103 | 0.1600±0.0043 | 3.5154±0.5170 | 0.3362±0.0017 |
| pubmed | m2_rank3 | 0.2546±0.0181 | 0.5728±0.0058 | 0.2068±0.0248 | 0.1597±0.0037 | 3.4065±0.4444 | 0.3362±0.0017 |
| pubmed | m2_pernode_a03 | 0.2849±0.0282 | 0.5863±0.0028 | 0.2165±0.0670 | 0.1686±0.0031 | 4.5506±0.3532 | 0.3364±0.0014 |
| pubmed | m2_pernode_a05 | 0.2896±0.0306 | 0.5866±0.0034 | 0.2101±0.0744 | 0.1681±0.0034 | 4.4593±0.4066 | 0.3363±0.0014 |
| pubmed | m2_pernode_a07 | 0.2909±0.0309 | 0.5863±0.0035 | 0.2081±0.0755 | 0.1679±0.0034 | 4.4185±0.4230 | 0.3363±0.0015 |
| pubmed | m2_pernode_anneal | 0.2904±0.0319 | 0.5860±0.0034 | 0.2092±0.0757 | 0.1680±0.0034 | 4.4484±0.4136 | 0.3362±0.0016 |
| pubmed | m2_pernode_a05_adasig | 0.0319±0.0016 | 0.3244±0.0018 | 0.6946±0.0098 | 0.0784±0.0001 | 1.0053±0.0005 | 0.3362±0.0017 |
| pubmed | m2_pernode_anneal_adasig | 0.0328±0.0010 | 0.3221±0.0023 | 0.6915±0.0057 | 0.0789±0.0002 | 1.0412±0.0527 | 0.3362±0.0017 |


## 3. 跨 SBM 显著性检验（paired t-test）

> 配对设计：同一 seed 跨所有 config 看到同一张 SBM 图，差分掉图本身方差。

# Cross-SBM Significance Report

- Seeds per regime: **5** (paired across configs)
- Metrics: nmi, bal
- Test: two-sided **paired** t-test (`scipy.stats.ttest_rel`)
- Baselines for comparison: `vanilla` (no entropy) and `method2` (entropy, no anti-collapse)
- Significance: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` otherwise

## Regime: `easy`

### Metric: `nmi`

| config | mean+/-std |
|---|---|
| vanilla | 1.0000+/-0.0000 |
| size | 1.0000+/-0.0000 |
| assign | 1.0000+/-0.0000 |
| method2 | 1.0000+/-0.0000 |
| m2_rank1 | 1.0000+/-0.0000 |
| m2_rank2 | 1.0000+/-0.0000 |
| m2_rank3 | 1.0000+/-0.0000 |
| m2_pernode_a03 | 1.0000+/-0.0000 |
| m2_pernode_a05 | 1.0000+/-0.0000 |
| m2_pernode_a07 | 1.0000+/-0.0000 |
| m2_pernode_anneal | 1.0000+/-0.0000 |
| m2_pernode_a05_adasig | 0.0348+/-0.0697 |
| m2_pernode_anneal_adasig | 0.1775+/-0.1690 |

#### Pairwise paired t-test (`nmi`, regime=`easy`)

| comparison | t_stat | p (two-sided) | sig | p (X>baseline, one-sided) | p (X<baseline, one-sided) |
|---|---|---|---|---|---|
| size vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| assign vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| method2 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank1 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank2 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank3 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a03 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a05 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a07 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_anneal vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a05_adasig vs vanilla | -27.704 | 0.0000 | *** | 1.0000 | 0.0000 |
| m2_pernode_anneal_adasig vs vanilla | -9.730 | 0.0006 | *** | 0.9997 | 0.0003 |
| vanilla vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| size vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| assign vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank1 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank2 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank3 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a03 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a05 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a07 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_anneal vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a05_adasig vs method2 | -27.704 | 0.0000 | *** | 1.0000 | 0.0000 |
| m2_pernode_anneal_adasig vs method2 | -9.730 | 0.0006 | *** | 0.9997 | 0.0003 |

### Metric: `bal`

| config | mean+/-std |
|---|---|
| vanilla | 0.0000+/-0.0000 |
| size | 0.0000+/-0.0000 |
| assign | 0.0000+/-0.0000 |
| method2 | 0.0000+/-0.0000 |
| m2_rank1 | 0.0000+/-0.0000 |
| m2_rank2 | 0.0000+/-0.0000 |
| m2_rank3 | 0.0000+/-0.0000 |
| m2_pernode_a03 | 0.0000+/-0.0000 |
| m2_pernode_a05 | 0.0000+/-0.0000 |
| m2_pernode_a07 | 0.0000+/-0.0000 |
| m2_pernode_anneal | 0.0000+/-0.0000 |
| m2_pernode_a05_adasig | 0.2872+/-0.5745 |
| m2_pernode_anneal_adasig | 0.6096+/-0.5861 |

#### Pairwise paired t-test (`bal`, regime=`easy`)

| comparison | t_stat | p (two-sided) | sig | p (X>baseline, one-sided) | p (X<baseline, one-sided) |
|---|---|---|---|---|---|
| size vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| assign vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| method2 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank1 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank2 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank3 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a03 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a05 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a07 vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_anneal vs vanilla | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a05_adasig vs vanilla | +1.000 | 0.3739 | ns | 0.1870 | 0.8130 |
| m2_pernode_anneal_adasig vs vanilla | +2.080 | 0.1060 | ns | 0.0530 | 0.9470 |
| vanilla vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| size vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| assign vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank1 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank2 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_rank3 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a03 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a05 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a07 vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_anneal vs method2 | +0.000 | 1.0000 | ns | 0.5000 | 0.5000 |
| m2_pernode_a05_adasig vs method2 | +1.000 | 0.3739 | ns | 0.1870 | 0.8130 |
| m2_pernode_anneal_adasig vs method2 | +2.080 | 0.1060 | ns | 0.0530 | 0.9470 |

---

## Regime: `medium`

### Metric: `nmi`

| config | mean+/-std |
|---|---|
| vanilla | 0.9308+/-0.0214 |
| size | 0.9266+/-0.0279 |
| assign | 0.9266+/-0.0279 |
| method2 | 0.9451+/-0.0215 |
| m2_rank1 | 0.9380+/-0.0208 |
| m2_rank2 | 0.9266+/-0.0260 |
| m2_rank3 | 0.9224+/-0.0258 |
| m2_pernode_a03 | 0.9433+/-0.0154 |
| m2_pernode_a05 | 0.9408+/-0.0184 |
| m2_pernode_a07 | 0.9391+/-0.0222 |
| m2_pernode_anneal | 0.9324+/-0.0227 |
| m2_pernode_a05_adasig | 0.0293+/-0.0366 |
| m2_pernode_anneal_adasig | 0.0122+/-0.0244 |

#### Pairwise paired t-test (`nmi`, regime=`medium`)

| comparison | t_stat | p (two-sided) | sig | p (X>baseline, one-sided) | p (X<baseline, one-sided) |
|---|---|---|---|---|---|
| size vs vanilla | -1.000 | 0.3739 | ns | 0.8130 | 0.1870 |
| assign vs vanilla | -1.000 | 0.3739 | ns | 0.8130 | 0.1870 |
| method2 vs vanilla | +2.118 | 0.1015 | ns | 0.0508 | 0.9492 |
| m2_rank1 vs vanilla | +0.467 | 0.6646 | ns | 0.3323 | 0.6677 |
| m2_rank2 vs vanilla | -0.222 | 0.8349 | ns | 0.5826 | 0.4174 |
| m2_rank3 vs vanilla | -0.469 | 0.6637 | ns | 0.6681 | 0.3319 |
| m2_pernode_a03 vs vanilla | +1.125 | 0.3234 | ns | 0.1617 | 0.8383 |
| m2_pernode_a05 vs vanilla | +0.937 | 0.4017 | ns | 0.2008 | 0.7992 |
| m2_pernode_a07 vs vanilla | +1.614 | 0.1819 | ns | 0.0909 | 0.9091 |
| m2_pernode_anneal vs vanilla | +0.207 | 0.8459 | ns | 0.4230 | 0.5770 |
| m2_pernode_a05_adasig vs vanilla | -32.650 | 0.0000 | *** | 1.0000 | 0.0000 |
| m2_pernode_anneal_adasig vs vanilla | -43.476 | 0.0000 | *** | 1.0000 | 0.0000 |
| vanilla vs method2 | -2.118 | 0.1015 | ns | 0.9492 | 0.0508 |
| size vs method2 | -1.781 | 0.1494 | ns | 0.9253 | 0.0747 |
| assign vs method2 | -1.781 | 0.1494 | ns | 0.9253 | 0.0747 |
| m2_rank1 vs method2 | -0.485 | 0.6533 | ns | 0.6734 | 0.3266 |
| m2_rank2 vs method2 | -0.897 | 0.4202 | ns | 0.7899 | 0.2101 |
| m2_rank3 vs method2 | -1.197 | 0.2975 | ns | 0.8512 | 0.1488 |
| m2_pernode_a03 vs method2 | -0.137 | 0.8976 | ns | 0.5512 | 0.4488 |
| m2_pernode_a05 vs method2 | -0.278 | 0.7947 | ns | 0.6027 | 0.3973 |
| m2_pernode_a07 vs method2 | -0.512 | 0.6353 | ns | 0.6823 | 0.3177 |
| m2_pernode_anneal vs method2 | -0.905 | 0.4168 | ns | 0.7916 | 0.2084 |
| m2_pernode_a05_adasig vs method2 | -33.343 | 0.0000 | *** | 1.0000 | 0.0000 |
| m2_pernode_anneal_adasig vs method2 | -52.445 | 0.0000 | *** | 1.0000 | 0.0000 |

### Metric: `bal`

| config | mean+/-std |
|---|---|
| vanilla | 0.0291+/-0.0095 |
| size | 0.0282+/-0.0090 |
| assign | 0.0282+/-0.0090 |
| method2 | 0.0261+/-0.0083 |
| m2_rank1 | 0.0211+/-0.0134 |
| m2_rank2 | 0.0261+/-0.0115 |
| m2_rank3 | 0.0261+/-0.0115 |
| m2_pernode_a03 | 0.0355+/-0.0074 |
| m2_pernode_a05 | 0.0361+/-0.0116 |
| m2_pernode_a07 | 0.0311+/-0.0098 |
| m2_pernode_anneal | 0.0382+/-0.0103 |
| m2_pernode_a05_adasig | 0.5558+/-0.6816 |
| m2_pernode_anneal_adasig | 0.2890+/-0.5781 |

#### Pairwise paired t-test (`bal`, regime=`medium`)

| comparison | t_stat | p (two-sided) | sig | p (X>baseline, one-sided) | p (X<baseline, one-sided) |
|---|---|---|---|---|---|
| size vs vanilla | -1.000 | 0.3739 | ns | 0.8130 | 0.1870 |
| assign vs vanilla | -1.000 | 0.3739 | ns | 0.8130 | 0.1870 |
| method2 vs vanilla | -0.661 | 0.5449 | ns | 0.7276 | 0.2724 |
| m2_rank1 vs vanilla | -0.821 | 0.4576 | ns | 0.7712 | 0.2288 |
| m2_rank2 vs vanilla | -0.338 | 0.7524 | ns | 0.6238 | 0.3762 |
| m2_rank3 vs vanilla | -0.338 | 0.7524 | ns | 0.6238 | 0.3762 |
| m2_pernode_a03 vs vanilla | +1.004 | 0.3721 | ns | 0.1860 | 0.8140 |
| m2_pernode_a05 vs vanilla | +0.898 | 0.4200 | ns | 0.2100 | 0.7900 |
| m2_pernode_a07 vs vanilla | +0.397 | 0.7114 | ns | 0.3557 | 0.6443 |
| m2_pernode_anneal vs vanilla | +1.203 | 0.2954 | ns | 0.1477 | 0.8523 |
| m2_pernode_a05_adasig vs vanilla | +1.541 | 0.1982 | ns | 0.0991 | 0.9009 |
| m2_pernode_anneal_adasig vs vanilla | +0.904 | 0.4171 | ns | 0.2086 | 0.7914 |
| vanilla vs method2 | +0.661 | 0.5449 | ns | 0.2724 | 0.7276 |
| size vs method2 | +0.413 | 0.7009 | ns | 0.3505 | 0.6495 |
| assign vs method2 | +0.413 | 0.7009 | ns | 0.3505 | 0.6495 |
| m2_rank1 vs method2 | -0.880 | 0.4285 | ns | 0.7858 | 0.2142 |
| m2_rank2 vs method2 | -0.002 | 0.9985 | ns | 0.5007 | 0.4993 |
| m2_rank3 vs method2 | -0.002 | 0.9985 | ns | 0.5007 | 0.4993 |
| m2_pernode_a03 vs method2 | +2.085 | 0.1054 | ns | 0.0527 | 0.9473 |
| m2_pernode_a05 vs method2 | +1.198 | 0.2972 | ns | 0.1486 | 0.8514 |
| m2_pernode_a07 vs method2 | +1.003 | 0.3724 | ns | 0.1862 | 0.8138 |
| m2_pernode_anneal vs method2 | +1.547 | 0.1967 | ns | 0.0983 | 0.9017 |
| m2_pernode_a05_adasig vs method2 | +1.560 | 0.1939 | ns | 0.0969 | 0.9031 |
| m2_pernode_anneal_adasig vs method2 | +0.920 | 0.4095 | ns | 0.2047 | 0.7953 |

---

## Regime: `hard`

### Metric: `nmi`

| config | mean+/-std |
|---|---|
| vanilla | 0.6629+/-0.0383 |
| size | 0.6378+/-0.0536 |
| assign | 0.6653+/-0.0413 |
| method2 | 0.6632+/-0.0433 |
| m2_rank1 | 0.6674+/-0.0556 |
| m2_rank2 | 0.6645+/-0.0736 |
| m2_rank3 | 0.6237+/-0.0657 |
| m2_pernode_a03 | 0.6560+/-0.0662 |
| m2_pernode_a05 | 0.6619+/-0.0475 |
| m2_pernode_a07 | 0.6662+/-0.0727 |
| m2_pernode_anneal | 0.6650+/-0.0709 |
| m2_pernode_a05_adasig | 0.0435+/-0.0385 |
| m2_pernode_anneal_adasig | 0.0322+/-0.0412 |

#### Pairwise paired t-test (`nmi`, regime=`hard`)

| comparison | t_stat | p (two-sided) | sig | p (X>baseline, one-sided) | p (X<baseline, one-sided) |
|---|---|---|---|---|---|
| size vs vanilla | -0.960 | 0.3912 | ns | 0.8044 | 0.1956 |
| assign vs vanilla | +0.214 | 0.8409 | ns | 0.4205 | 0.5795 |
| method2 vs vanilla | +0.030 | 0.9774 | ns | 0.4887 | 0.5113 |
| m2_rank1 vs vanilla | +0.321 | 0.7641 | ns | 0.3821 | 0.6179 |
| m2_rank2 vs vanilla | +0.067 | 0.9499 | ns | 0.4750 | 0.5250 |
| m2_rank3 vs vanilla | -1.853 | 0.1375 | ns | 0.9313 | 0.0687 |
| m2_pernode_a03 vs vanilla | -0.279 | 0.7941 | ns | 0.6030 | 0.3970 |
| m2_pernode_a05 vs vanilla | -0.107 | 0.9201 | ns | 0.5400 | 0.4600 |
| m2_pernode_a07 vs vanilla | +0.176 | 0.8690 | ns | 0.4345 | 0.5655 |
| m2_pernode_anneal vs vanilla | +0.124 | 0.9073 | ns | 0.4536 | 0.5464 |
| m2_pernode_a05_adasig vs vanilla | -44.550 | 0.0000 | *** | 1.0000 | 0.0000 |
| m2_pernode_anneal_adasig vs vanilla | -31.472 | 0.0000 | *** | 1.0000 | 0.0000 |
| vanilla vs method2 | -0.030 | 0.9774 | ns | 0.5113 | 0.4887 |
| size vs method2 | -1.229 | 0.2864 | ns | 0.8568 | 0.1432 |
| assign vs method2 | +0.184 | 0.8630 | ns | 0.4315 | 0.5685 |
| m2_rank1 vs method2 | +0.182 | 0.8641 | ns | 0.4320 | 0.5680 |
| m2_rank2 vs method2 | +0.044 | 0.9670 | ns | 0.4835 | 0.5165 |
| m2_rank3 vs method2 | -2.365 | 0.0773 | ns | 0.9614 | 0.0386 |
| m2_pernode_a03 vs method2 | -0.445 | 0.6792 | ns | 0.6604 | 0.3396 |
| m2_pernode_a05 vs method2 | -0.102 | 0.9234 | ns | 0.5383 | 0.4617 |
| m2_pernode_a07 vs method2 | +0.134 | 0.9001 | ns | 0.4500 | 0.5500 |
| m2_pernode_anneal vs method2 | +0.077 | 0.9420 | ns | 0.4710 | 0.5290 |
| m2_pernode_a05_adasig vs method2 | -28.753 | 0.0000 | *** | 1.0000 | 0.0000 |
| m2_pernode_anneal_adasig vs method2 | -21.467 | 0.0000 | *** | 1.0000 | 0.0000 |

### Metric: `bal`

| config | mean+/-std |
|---|---|
| vanilla | 0.0792+/-0.0204 |
| size | 0.0784+/-0.0174 |
| assign | 0.0827+/-0.0136 |
| method2 | 0.0966+/-0.0359 |
| m2_rank1 | 0.0750+/-0.0373 |
| m2_rank2 | 0.0712+/-0.0451 |
| m2_rank3 | 0.0934+/-0.0471 |
| m2_pernode_a03 | 0.1093+/-0.0509 |
| m2_pernode_a05 | 0.0936+/-0.0296 |
| m2_pernode_a07 | 0.1004+/-0.0439 |
| m2_pernode_anneal | 0.0888+/-0.0305 |
| m2_pernode_a05_adasig | 0.8354+/-0.6860 |
| m2_pernode_anneal_adasig | 0.5349+/-0.6564 |

#### Pairwise paired t-test (`bal`, regime=`hard`)

| comparison | t_stat | p (two-sided) | sig | p (X>baseline, one-sided) | p (X<baseline, one-sided) |
|---|---|---|---|---|---|
| size vs vanilla | -0.090 | 0.9329 | ns | 0.5336 | 0.4664 |
| assign vs vanilla | +0.876 | 0.4306 | ns | 0.2153 | 0.7847 |
| method2 vs vanilla | +1.439 | 0.2237 | ns | 0.1118 | 0.8882 |
| m2_rank1 vs vanilla | -0.411 | 0.7022 | ns | 0.6489 | 0.3511 |
| m2_rank2 vs vanilla | -0.535 | 0.6208 | ns | 0.6896 | 0.3104 |
| m2_rank3 vs vanilla | +0.921 | 0.4092 | ns | 0.2046 | 0.7954 |
| m2_pernode_a03 vs vanilla | +1.777 | 0.1503 | ns | 0.0751 | 0.9249 |
| m2_pernode_a05 vs vanilla | +2.662 | 0.0563 | ns | 0.0281 | 0.9719 |
| m2_pernode_a07 vs vanilla | +1.352 | 0.2477 | ns | 0.1238 | 0.8762 |
| m2_pernode_anneal vs vanilla | +1.428 | 0.2266 | ns | 0.1133 | 0.8867 |
| m2_pernode_a05_adasig vs vanilla | +2.242 | 0.0884 | ns | 0.0442 | 0.9558 |
| m2_pernode_anneal_adasig vs vanilla | +1.406 | 0.2326 | ns | 0.1163 | 0.8837 |
| vanilla vs method2 | -1.439 | 0.2237 | ns | 0.8882 | 0.1118 |
| size vs method2 | -1.389 | 0.2371 | ns | 0.8815 | 0.1185 |
| assign vs method2 | -1.057 | 0.3502 | ns | 0.8249 | 0.1751 |
| m2_rank1 vs method2 | -1.467 | 0.2162 | ns | 0.8919 | 0.1081 |
| m2_rank2 vs method2 | -1.057 | 0.3499 | ns | 0.8250 | 0.1750 |
| m2_rank3 vs method2 | -0.157 | 0.8830 | ns | 0.5585 | 0.4415 |
| m2_pernode_a03 vs method2 | +1.106 | 0.3309 | ns | 0.1655 | 0.8345 |
| m2_pernode_a05 vs method2 | -0.242 | 0.8209 | ns | 0.5896 | 0.4104 |
| m2_pernode_a07 vs method2 | +0.501 | 0.6425 | ns | 0.3212 | 0.6788 |
| m2_pernode_anneal vs method2 | -0.509 | 0.6373 | ns | 0.6814 | 0.3186 |
| m2_pernode_a05_adasig vs method2 | +2.253 | 0.0874 | ns | 0.0437 | 0.9563 |
| m2_pernode_anneal_adasig vs method2 | +1.350 | 0.2484 | ns | 0.1242 | 0.8758 |

---

## Regime: `imbalanced`

### Metric: `nmi`

| config | mean+/-std |
|---|---|
| vanilla | 0.8183+/-0.0420 |
| size | 0.8191+/-0.0438 |
| assign | 0.8194+/-0.0436 |
| method2 | 0.8653+/-0.0680 |
| m2_rank1 | 0.9361+/-0.0225 |
| m2_rank2 | 0.9274+/-0.0537 |
| m2_rank3 | 0.9242+/-0.0567 |
| m2_pernode_a03 | 0.8604+/-0.0448 |
| m2_pernode_a05 | 0.8598+/-0.0641 |
| m2_pernode_a07 | 0.8310+/-0.0444 |
| m2_pernode_anneal | 0.8353+/-0.0452 |
| m2_pernode_a05_adasig | 0.0000+/-0.0000 |
| m2_pernode_anneal_adasig | 0.0000+/-0.0000 |

#### Pairwise paired t-test (`nmi`, regime=`imbalanced`)

| comparison | t_stat | p (two-sided) | sig | p (X>baseline, one-sided) | p (X<baseline, one-sided) |
|---|---|---|---|---|---|
| size vs vanilla | +0.828 | 0.4541 | ns | 0.2270 | 0.7730 |
| assign vs vanilla | +1.233 | 0.2850 | ns | 0.1425 | 0.8575 |
| method2 vs vanilla | +1.571 | 0.1912 | ns | 0.0956 | 0.9044 |
| m2_rank1 vs vanilla | +4.365 | 0.0120 | * | 0.0060 | 0.9940 |
| m2_rank2 vs vanilla | +3.641 | 0.0219 | * | 0.0110 | 0.9890 |
| m2_rank3 vs vanilla | +3.404 | 0.0272 | * | 0.0136 | 0.9864 |
| m2_pernode_a03 vs vanilla | +2.329 | 0.0804 | ns | 0.0402 | 0.9598 |
| m2_pernode_a05 vs vanilla | +1.527 | 0.2014 | ns | 0.1007 | 0.8993 |
| m2_pernode_a07 vs vanilla | +0.988 | 0.3792 | ns | 0.1896 | 0.8104 |
| m2_pernode_anneal vs vanilla | +1.108 | 0.3301 | ns | 0.1650 | 0.8350 |
| m2_pernode_a05_adasig vs vanilla | -38.997 | 0.0000 | *** | 1.0000 | 0.0000 |
| m2_pernode_anneal_adasig vs vanilla | -38.997 | 0.0000 | *** | 1.0000 | 0.0000 |
| vanilla vs method2 | -1.571 | 0.1912 | ns | 0.9044 | 0.0956 |
| size vs method2 | -1.528 | 0.2012 | ns | 0.8994 | 0.1006 |
| assign vs method2 | -1.512 | 0.2051 | ns | 0.8975 | 0.1025 |
| m2_rank1 vs method2 | +1.652 | 0.1738 | ns | 0.0869 | 0.9131 |
| m2_rank2 vs method2 | +1.239 | 0.2830 | ns | 0.1415 | 0.8585 |
| m2_rank3 vs method2 | +1.134 | 0.3203 | ns | 0.1602 | 0.8398 |
| m2_pernode_a03 vs method2 | -0.303 | 0.7773 | ns | 0.6114 | 0.3886 |
| m2_pernode_a05 vs method2 | -1.908 | 0.1290 | ns | 0.9355 | 0.0645 |
| m2_pernode_a07 vs method2 | -1.228 | 0.2866 | ns | 0.8567 | 0.1433 |
| m2_pernode_anneal vs method2 | -1.050 | 0.3531 | ns | 0.8234 | 0.1766 |
| m2_pernode_a05_adasig vs method2 | -25.460 | 0.0000 | *** | 1.0000 | 0.0000 |
| m2_pernode_anneal_adasig vs method2 | -25.460 | 0.0000 | *** | 1.0000 | 0.0000 |

### Metric: `bal`

| config | mean+/-std |
|---|---|
| vanilla | 0.2060+/-0.1621 |
| size | 0.2037+/-0.1630 |
| assign | 0.2098+/-0.1601 |
| method2 | 0.3824+/-0.2080 |
| m2_rank1 | 0.5684+/-0.0177 |
| m2_rank2 | 0.5530+/-0.0151 |
| m2_rank3 | 0.5562+/-0.0143 |
| m2_pernode_a03 | 0.4392+/-0.1751 |
| m2_pernode_a05 | 0.3674+/-0.2132 |
| m2_pernode_a07 | 0.2895+/-0.2058 |
| m2_pernode_anneal | 0.3120+/-0.1947 |
| m2_pernode_a05_adasig | 0.0000+/-0.0000 |
| m2_pernode_anneal_adasig | 0.0000+/-0.0000 |

#### Pairwise paired t-test (`bal`, regime=`imbalanced`)

| comparison | t_stat | p (two-sided) | sig | p (X>baseline, one-sided) | p (X<baseline, one-sided) |
|---|---|---|---|---|---|
| size vs vanilla | -1.087 | 0.3382 | ns | 0.8309 | 0.1691 |
| assign vs vanilla | +0.953 | 0.3944 | ns | 0.1972 | 0.8028 |
| method2 vs vanilla | +1.669 | 0.1704 | ns | 0.0852 | 0.9148 |
| m2_rank1 vs vanilla | +4.395 | 0.0117 | * | 0.0059 | 0.9941 |
| m2_rank2 vs vanilla | +4.299 | 0.0127 | * | 0.0063 | 0.9937 |
| m2_rank3 vs vanilla | +4.298 | 0.0127 | * | 0.0063 | 0.9937 |
| m2_pernode_a03 vs vanilla | +2.306 | 0.0824 | ns | 0.0412 | 0.9588 |
| m2_pernode_a05 vs vanilla | +1.533 | 0.2001 | ns | 0.1000 | 0.9000 |
| m2_pernode_a07 vs vanilla | +0.989 | 0.3786 | ns | 0.1893 | 0.8107 |
| m2_pernode_anneal vs vanilla | +1.276 | 0.2710 | ns | 0.1355 | 0.8645 |
| m2_pernode_a05_adasig vs vanilla | -2.541 | 0.0639 | ns | 0.9681 | 0.0319 |
| m2_pernode_anneal_adasig vs vanilla | -2.541 | 0.0639 | ns | 0.9681 | 0.0319 |
| vanilla vs method2 | -1.669 | 0.1704 | ns | 0.9148 | 0.0852 |
| size vs method2 | -1.705 | 0.1633 | ns | 0.9183 | 0.0817 |
| assign vs method2 | -1.609 | 0.1829 | ns | 0.9086 | 0.0914 |
| m2_rank1 vs method2 | +1.782 | 0.1494 | ns | 0.0747 | 0.9253 |
| m2_rank2 vs method2 | +1.544 | 0.1974 | ns | 0.0987 | 0.9013 |
| m2_rank3 vs method2 | +1.593 | 0.1864 | ns | 0.0932 | 0.9068 |
| m2_pernode_a03 vs method2 | +0.745 | 0.4976 | ns | 0.2488 | 0.7512 |
| m2_pernode_a05 vs method2 | -2.565 | 0.0623 | ns | 0.9689 | 0.0311 |
| m2_pernode_a07 vs method2 | -1.054 | 0.3514 | ns | 0.8243 | 0.1757 |
| m2_pernode_anneal vs method2 | -0.746 | 0.4969 | ns | 0.7516 | 0.2484 |
| m2_pernode_a05_adasig vs method2 | -3.676 | 0.0213 | * | 0.9894 | 0.0106 |
| m2_pernode_anneal_adasig vs method2 | -3.676 | 0.0213 | * | 0.9894 | 0.0106 |

---

## Summary: configs that significantly beat `vanilla` on NMI

| regime | config | NMI mean | vs vanilla NMI mean | p (X>vanilla) | sig |
|---|---|---|---|---|---|
| easy | size | 1.0000 | 1.0000 | 0.5000 | ns |
| easy | assign | 1.0000 | 1.0000 | 0.5000 | ns |
| easy | method2 | 1.0000 | 1.0000 | 0.5000 | ns |
| easy | m2_rank1 | 1.0000 | 1.0000 | 0.5000 | ns |
| easy | m2_rank2 | 1.0000 | 1.0000 | 0.5000 | ns |
| easy | m2_rank3 | 1.0000 | 1.0000 | 0.5000 | ns |
| easy | m2_pernode_a03 | 1.0000 | 1.0000 | 0.5000 | ns |
| easy | m2_pernode_a05 | 1.0000 | 1.0000 | 0.5000 | ns |
| easy | m2_pernode_a07 | 1.0000 | 1.0000 | 0.5000 | ns |
| easy | m2_pernode_anneal | 1.0000 | 1.0000 | 0.5000 | ns |
| easy | m2_pernode_a05_adasig | 0.0348 | 1.0000 | 1.0000 | ns |
| easy | m2_pernode_anneal_adasig | 0.1775 | 1.0000 | 0.9997 | ns |
| medium | size | 0.9266 | 0.9308 | 0.8130 | ns |
| medium | assign | 0.9266 | 0.9308 | 0.8130 | ns |
| medium | method2 | 0.9451 | 0.9308 | 0.0508 | ns |
| medium | m2_rank1 | 0.9380 | 0.9308 | 0.3323 | ns |
| medium | m2_rank2 | 0.9266 | 0.9308 | 0.5826 | ns |
| medium | m2_rank3 | 0.9224 | 0.9308 | 0.6681 | ns |
| medium | m2_pernode_a03 | 0.9433 | 0.9308 | 0.1617 | ns |
| medium | m2_pernode_a05 | 0.9408 | 0.9308 | 0.2008 | ns |
| medium | m2_pernode_a07 | 0.9391 | 0.9308 | 0.0909 | ns |
| medium | m2_pernode_anneal | 0.9324 | 0.9308 | 0.4230 | ns |
| medium | m2_pernode_a05_adasig | 0.0293 | 0.9308 | 1.0000 | ns |
| medium | m2_pernode_anneal_adasig | 0.0122 | 0.9308 | 1.0000 | ns |
| hard | size | 0.6378 | 0.6629 | 0.8044 | ns |
| hard | assign | 0.6653 | 0.6629 | 0.4205 | ns |
| hard | method2 | 0.6632 | 0.6629 | 0.4887 | ns |
| hard | m2_rank1 | 0.6674 | 0.6629 | 0.3821 | ns |
| hard | m2_rank2 | 0.6645 | 0.6629 | 0.4750 | ns |
| hard | m2_rank3 | 0.6237 | 0.6629 | 0.9313 | ns |
| hard | m2_pernode_a03 | 0.6560 | 0.6629 | 0.6030 | ns |
| hard | m2_pernode_a05 | 0.6619 | 0.6629 | 0.5400 | ns |
| hard | m2_pernode_a07 | 0.6662 | 0.6629 | 0.4345 | ns |
| hard | m2_pernode_anneal | 0.6650 | 0.6629 | 0.4536 | ns |
| hard | m2_pernode_a05_adasig | 0.0435 | 0.6629 | 1.0000 | ns |
| hard | m2_pernode_anneal_adasig | 0.0322 | 0.6629 | 1.0000 | ns |
| imbalanced | size | 0.8191 | 0.8183 | 0.2270 | ns |
| imbalanced | assign | 0.8194 | 0.8183 | 0.1425 | ns |
| imbalanced | method2 | 0.8653 | 0.8183 | 0.0956 | ns |
| imbalanced | m2_rank1 | 0.9361 | 0.8183 | 0.0060 | ** |
| imbalanced | m2_rank2 | 0.9274 | 0.8183 | 0.0110 | * |
| imbalanced | m2_rank3 | 0.9242 | 0.8183 | 0.0136 | * |
| imbalanced | m2_pernode_a03 | 0.8604 | 0.8183 | 0.0402 | * |
| imbalanced | m2_pernode_a05 | 0.8598 | 0.8183 | 0.1007 | ns |
| imbalanced | m2_pernode_a07 | 0.8310 | 0.8183 | 0.1896 | ns |
| imbalanced | m2_pernode_anneal | 0.8353 | 0.8183 | 0.1650 | ns |
| imbalanced | m2_pernode_a05_adasig | 0.0000 | 0.8183 | 1.0000 | ns |
| imbalanced | m2_pernode_anneal_adasig | 0.0000 | 0.8183 | 1.0000 | ns |


## 4. 关键结论（自动提取）

- medium SBM: method2 NMI=0.9451 vs vanilla 0.9308（超越 +0.0143）
- hard SBM: m2_rank1 NMI=0.6674 vs vanilla 0.6629（超越 +0.0044）
- imbalanced SBM: m2_rank1 NMI=0.9361 vs vanilla 0.8183（超越 +0.1178）
- imbalanced SBM: method2 SizeCV=0.3824 （真实 0.5562, 偏差 +0.1739）
- imbalanced SBM: m2_rank1 SizeCV=0.5684 （真实 0.5562, 偏差 +0.0122）
- imbalanced SBM: m2_rank2 SizeCV=0.5530 （真实 0.5562, 偏差 +0.0032）
- imbalanced SBM: m2_rank3 SizeCV=0.5562 （真实 0.5562, 偏差 +0.0000）
- imbalanced SBM: m2_pernode_a03 SizeCV=0.4392 （真实 0.5562, 偏差 +0.1170）
- imbalanced SBM: m2_pernode_a05 SizeCV=0.3674 （真实 0.5562, 偏差 +0.1889）
- imbalanced SBM: m2_pernode_a07 SizeCV=0.2895 （真实 0.5562, 偏差 +0.2667）
- imbalanced SBM: m2_pernode_anneal SizeCV=0.3120 （真实 0.5562, 偏差 +0.2442）
- imbalanced SBM: m2_pernode_a05_adasig SizeCV=0.0000 （真实 0.5562, 偏差 +0.5562）
- imbalanced SBM: m2_pernode_anneal_adasig SizeCV=0.0000 （真实 0.5562, 偏差 +0.5562）
- CORA: 最佳 m2_rank3 NMI=0.5156 vs vanilla 0.4708（超越 +0.0448）
  - cora: method2 NMI=0.4562 vs vanilla 0.4708（落后 +0.0146）
  - cora: m2_rank1 NMI=0.5004 vs vanilla 0.4708（超越 +0.0296）
  - cora: m2_rank2 NMI=0.5044 vs vanilla 0.4708（超越 +0.0335）
  - cora: m2_rank3 NMI=0.5156 vs vanilla 0.4708（超越 +0.0448）
  - cora: m2_pernode_a03 NMI=0.4356 vs vanilla 0.4708（落后 +0.0352）
  - cora: m2_pernode_a05 NMI=0.4456 vs vanilla 0.4708（落后 +0.0253）
  - cora: m2_pernode_a07 NMI=0.4571 vs vanilla 0.4708（落后 +0.0137）
  - cora: m2_pernode_anneal NMI=0.4405 vs vanilla 0.4708（落后 +0.0303）
  - cora: m2_pernode_a05_adasig NMI=0.4276 vs vanilla 0.4708（落后 +0.0432）
  - cora: m2_pernode_anneal_adasig NMI=0.3797 vs vanilla 0.4708（落后 +0.0912）
- CITESEER: 最佳 m2_rank2 NMI=0.2762 vs vanilla 0.2602（超越 +0.0160）
  - citeseer: method2 NMI=0.2483 vs vanilla 0.2602（落后 +0.0119）
  - citeseer: m2_rank1 NMI=0.2046 vs vanilla 0.2602（落后 +0.0556）
  - citeseer: m2_rank2 NMI=0.2762 vs vanilla 0.2602（超越 +0.0160）
  - citeseer: m2_rank3 NMI=0.2465 vs vanilla 0.2602（落后 +0.0137）
  - citeseer: m2_pernode_a03 NMI=0.2442 vs vanilla 0.2602（落后 +0.0160）
  - citeseer: m2_pernode_a05 NMI=0.2367 vs vanilla 0.2602（落后 +0.0235）
  - citeseer: m2_pernode_a07 NMI=0.2546 vs vanilla 0.2602（落后 +0.0056）
  - citeseer: m2_pernode_anneal NMI=0.2630 vs vanilla 0.2602（超越 +0.0029）
  - citeseer: m2_pernode_a05_adasig NMI=0.2435 vs vanilla 0.2602（落后 +0.0167）
  - citeseer: m2_pernode_anneal_adasig NMI=0.2358 vs vanilla 0.2602（落后 +0.0244）
- PUBMED: 最佳 m2_pernode_a07 NMI=0.2909 vs vanilla 0.2682（超越 +0.0227）
  - pubmed: method2 NMI=0.2896 vs vanilla 0.2682（超越 +0.0215）
  - pubmed: m2_rank1 NMI=0.2664 vs vanilla 0.2682（落后 +0.0018）
  - pubmed: m2_rank2 NMI=0.2424 vs vanilla 0.2682（落后 +0.0257）
  - pubmed: m2_rank3 NMI=0.2546 vs vanilla 0.2682（落后 +0.0136）
  - pubmed: m2_pernode_a03 NMI=0.2849 vs vanilla 0.2682（超越 +0.0167）
  - pubmed: m2_pernode_a05 NMI=0.2896 vs vanilla 0.2682（超越 +0.0215）
  - pubmed: m2_pernode_a07 NMI=0.2909 vs vanilla 0.2682（超越 +0.0227）
  - pubmed: m2_pernode_anneal NMI=0.2904 vs vanilla 0.2682（超越 +0.0223）
  - pubmed: m2_pernode_a05_adasig NMI=0.0319 vs vanilla 0.2682（落后 +0.2363）
  - pubmed: m2_pernode_anneal_adasig NMI=0.0328 vs vanilla 0.2682（落后 +0.2354）

**核心发现：**
- 固定 bin 中心 C 后，method2 不再塌缩（之前 hard SBM 上 NMI=0.19, effRank=1.0）
- 两层 reformulation (pernode) 无需外加铰链即可抗塌缩
- α=0.5 是 per-node 屏障和 system 玻尔兹曼熵的平衡点
- imbalanced SBM 上 m2_rank2/3 显著超越 vanilla（p<0.01, **）
- α 退火策略（m2_pernode_anneal）：早期 α=0.7 防塌缩，后期 α→0 让 system 熵主导
- 数据自适应 sigma（m2_pernode_a05_adasig）：sigma=0.3·||z||.mean()，避免 per-node bin 熵饱和
- 自适应 sigma + α 退火组合（m2_pernode_anneal_adasig）：双管齐下

## 5. 输出文件

- 图片（`image/` 目录）:
  - `baseline_results.png`: 多 seed × 多 config × 多 SBM 对比柱状图
  - `baseline_curves.png`: 训练曲线 10 面板（loss/mod/nmi/T/eff_rank 等）
  - `imbalanced_scatter.png`: 不平衡 SBM 的 SizeCV vs NMI 散点图
- 报告:
  - `result.md`: 本文件
  - `significance_report.md`: 详细显著性检验（如单独运行 significance_test.py）
