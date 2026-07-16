# WJ vs SOTA 深度图聚类对比结果

> 由 `sota_comparison.py` 自动生成。WJ 三变体 vs 9 个 SOTA 方法，数据集：CORA, CITESEER, PUBMED。
> SOTA 数字均引用自原论文（未自行复现），WJ 数字为多 seed 均值±标准差。
> 生成时间: 2026-07-16 19:42:57

## 1. SOTA 方法来源

| 方法 | 类型 | 论文 |
|---|---|---|
| DAEGC | 深度图聚类 | Wang et al., IJCAI 2019 (arXiv:1904.08372) Table 2/3/4 |
| AGC | 深度图聚类 | Zhang et al., IJCAI 2019 (arXiv:1906.01210) Table 2; ARI 不报告 |
| DCRN | 深度图聚类 | Liu et al., AAAI 2022 (arXiv:2112.14731) Table 2; 论文不测 Cora |
| SDCN | 深度图聚类 | Bo et al., WWW 2020 (arXiv:2002.01633) Table 2; 论文不测 Cora/PubMed |
| SCAGC | 深度图聚类 | Xia et al., IEEE TMM 2023 (arXiv:2110.08264v1); arxiv v1 不测 Cora/CiteSeer/PubMed |
| GAE+KMeans | 嵌入+聚类 | Kipf & Welling, NIPS 2016 ws; 聚类结果由 DAEGC 论文 Table 2/3/4 复现报告 |
| VGAE+KMeans | 嵌入+聚类 | Kipf & Welling, NIPS 2016 ws; 聚类结果由 DAEGC 论文 Table 2/3/4 复现报告 |
| DeepWalk+KMeans | 嵌入+聚类 | Perozzi et al., KDD 2014; 聚类结果由 AGC 论文 Table 2 复现报告; ARI 不报告 |
| node2vec+KMeans | 嵌入+聚类 | Grover & Leskovec, KDD 2016; DAEGC/AGC 均未作为 baseline 报告 |

## 2. ACC 对比表

| 方法 | CORA | CITESEER | PUBMED |
|---|---|---|---|
| vanilla (WJ) | 0.6116±0.0438 | 0.4314±0.0245 | 0.6684±0.0038 |
| method2 (WJ) | 0.6062±0.0248 | 0.4447±0.0274 | 0.6684±0.0037 |
| m2_rank3 (WJ) | 0.6618±0.0241 | 0.4447±0.0274 | 0.6684±0.0037 |
| m2_cl (WJ) | 0.5776±0.0454 | 0.5289±0.0397 | 0.6628±0.0015 |
| m2_rank3_cl (WJ) | **0.7333±0.0212** | 0.5289±0.0397 | 0.6628±0.0015 |
| DAEGC (深度) | 0.7040 | 0.6720 | 0.6710 |
| AGC (深度) | 0.6892 | 0.6700 | 0.6978 |
| DCRN (深度) | N/A | **0.7086** | **0.6987** |
| SDCN (深度) | N/A | 0.6596 | N/A |
| SCAGC (深度) | N/A | N/A | N/A |
| GAE+KMeans (嵌入) | 0.5300 | 0.3800 | 0.6320 |
| VGAE+KMeans (嵌入) | 0.5920 | 0.3920 | 0.6190 |
| DeepWalk+KMeans (嵌入) | 0.4674 | 0.3615 | 0.6186 |
| node2vec+KMeans (嵌入) | N/A | N/A | N/A |

## 3. NMI 对比表

| 方法 | CORA | CITESEER | PUBMED |
|---|---|---|---|
| vanilla (WJ) | 0.4618±0.0340 | 0.2350±0.0076 | 0.2829±0.0089 |
| method2 (WJ) | 0.4551±0.0180 | 0.2335±0.0118 | 0.2836±0.0098 |
| m2_rank3 (WJ) | 0.5026±0.0169 | 0.2335±0.0118 | 0.2836±0.0098 |
| m2_cl (WJ) | 0.4646±0.0234 | 0.3199±0.0284 | 0.2795±0.0083 |
| m2_rank3_cl (WJ) | **0.5529±0.0201** | 0.3199±0.0284 | 0.2795±0.0083 |
| DAEGC (深度) | 0.5280 | 0.3970 | 0.2660 |
| AGC (深度) | 0.5368 | 0.4113 | 0.3159 |
| DCRN (深度) | N/A | **0.4586** | **0.3220** |
| SDCN (深度) | N/A | 0.3871 | N/A |
| SCAGC (深度) | N/A | N/A | N/A |
| GAE+KMeans (嵌入) | 0.3970 | 0.1740 | 0.2490 |
| VGAE+KMeans (嵌入) | 0.4080 | 0.1630 | 0.2160 |
| DeepWalk+KMeans (嵌入) | 0.3175 | 0.0966 | 0.1671 |
| node2vec+KMeans (嵌入) | N/A | N/A | N/A |

## 4. ARI 对比表

| 方法 | CORA | CITESEER | PUBMED |
|---|---|---|---|
| vanilla (WJ) | 0.3818±0.0501 | 0.1375±0.0139 | 0.2883±0.0051 |
| method2 (WJ) | 0.3708±0.0337 | 0.1307±0.0128 | 0.2890±0.0055 |
| m2_rank3 (WJ) | 0.4192±0.0267 | 0.1307±0.0128 | 0.2890±0.0055 |
| m2_cl (WJ) | 0.3408±0.0512 | 0.2288±0.0598 | 0.2804±0.0009 |
| m2_rank3_cl (WJ) | **0.5140±0.0317** | 0.2288±0.0598 | 0.2804±0.0009 |
| DAEGC (深度) | 0.4960 | 0.4100 | 0.2780 |
| AGC (深度) | N/A | N/A | N/A |
| DCRN (深度) | N/A | **0.4764** | **0.3141** |
| SDCN (深度) | N/A | 0.4017 | N/A |
| SCAGC (深度) | N/A | N/A | N/A |
| GAE+KMeans (嵌入) | 0.2930 | 0.1410 | 0.2460 |
| VGAE+KMeans (嵌入) | 0.3470 | 0.1010 | 0.2010 |
| DeepWalk+KMeans (嵌入) | N/A | N/A | N/A |
| node2vec+KMeans (嵌入) | N/A | N/A | N/A |

## 5. 差距分析

### 差距分析（WJ 最佳变体 vs 最强 SOTA）

> 对每个数据集/指标，取 WJ 四变体中表现最好的来对比 SOTA 最佳。
> 状态：领先 / 接近(差距<0.03) / 落后(0.03~0.10) / 崩溃(>0.10)

| 数据集 | 指标 | WJ最佳 | WJ变体 | SOTA最佳 | SOTA方法 | 差距 | 状态 |
|---|---|---|---|---|---|---|---|
| CORA | ACC | 0.7333 | m2_rank3_cl | 0.7040 | DAEGC | +0.0293 | 领先 |
| CORA | NMI | 0.5529 | m2_rank3_cl | 0.5368 | AGC | +0.0161 | 领先 |
| CORA | ARI | 0.5140 | m2_rank3_cl | 0.4960 | DAEGC | +0.0180 | 领先 |
| CITESEER | ACC | 0.5289 | m2_cl | 0.7086 | DCRN | -0.1797 | 崩溃 |
| CITESEER | NMI | 0.3199 | m2_cl | 0.4586 | DCRN | -0.1387 | 崩溃 |
| CITESEER | ARI | 0.2288 | m2_cl | 0.4764 | DCRN | -0.2476 | 崩溃 |
| PUBMED | ACC | 0.6684 | vanilla | 0.6987 | DCRN | -0.0303 | 落后 |
| PUBMED | NMI | 0.2836 | m2_rank3 | 0.3220 | DCRN | -0.0384 | 落后 |
| PUBMED | ARI | 0.2890 | method2 | 0.3141 | DCRN | -0.0251 | 接近 |

**改进优先级（按差距从小到大）：**

1. CITESEER ARI: 差距 -0.2476 (崩溃)
2. CITESEER ACC: 差距 -0.1797 (崩溃)
3. CITESEER NMI: 差距 -0.1387 (崩溃)
4. PUBMED NMI: 差距 -0.0384 (落后)
5. PUBMED ACC: 差距 -0.0303 (落后)

## 6. 综合排名

### 综合排名（按 ACC 排名，每数据集 1~N，求平均）

| 方法 | CORA | CITESEER | PUBMED | 平均排名 |
|---|---|---|---|---|
| vanilla (WJ) | 5 | 9 | 4 | 6.0 |
| method2 (WJ) | 6 | 7 | 5 | 6.0 |
| m2_rank3 (WJ) | 4 | 8 | 6 | 6.0 |
| m2_cl (WJ) | 8 | 5 | 7 | 6.7 |
| m2_rank3_cl (WJ) | 1 | 6 | 8 | 5.0 |
| DAEGC (深度) | 2 | 2 | 3 | 2.3 |
| AGC (深度) | 3 | 3 | 2 | 2.7 |
| DCRN (深度) | 14 | 1 | 1 | 5.3 |
| SDCN (深度) | 14 | 4 | 14 | 10.7 |
| SCAGC (深度) | 14 | 14 | 14 | 14.0 |
| GAE+KMeans (嵌入) | 9 | 11 | 9 | 9.7 |
| VGAE+KMeans (嵌入) | 7 | 10 | 10 | 9.0 |
| DeepWalk+KMeans (嵌入) | 10 | 12 | 11 | 11.0 |
| node2vec+KMeans (嵌入) | 14 | 14 | 14 | 14.0 |

## 7. 注意事项

- SOTA 数字均引用自原论文 Table，未自行复现
- WJ 方法使用全图（不做 filter_largest_cc），和 SOTA 论文保持一致，如 CiteSeer 用全部 3312 节点
- m2_rank3 的 hinge 策略：仅强社区图（CC≥0.20 且 N≤5000，如 Cora）用 λ=3.0，其余数据集（含 CiteSeer/PubMed）λ=0.0 关闭 hinge
- m2_cl = method2 + 对比学习 + 特征重建（无 hinge）：强社区图（CC≥0.20）用结构 CL（邻接定义正/负样本），弱社区图（CC<0.20）+高维稀疏特征（F>500，如 CiteSeer 3703 维）用混合 CL（结构+特征加权）+ PCA 降维(200) + 特征重建，弱社区图+低维特征（F≤500，如 PubMed）用纯特征 CL + 特征重建
- m2_rank3_cl = m2_rank3 的 hinge + m2_cl 的 CL（组合变体）：强社区图用 hinge λ=3.0 + 结构 CL（双重增强社区边界），弱社区图关闭 hinge（λ=0），CL 模式同 m2_cl
- DCRN 论文不测 Cora，SDCN 论文不测 Cora/PubMed，对应单元格标 N/A
- MVGRL 不报告聚类 ACC/NMI/ARI（只做节点/图分类线性评估），已从对比列表移除
