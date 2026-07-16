"""wj — WJ 图聚类模型核心包。

包含子模块：
  - runtime：DEVICE（GPU 自动检测）, set_seed（随机种子）
  - data：SBM 生成器, 真实数据集加载器, normalize_adj, DATASET_LOADERS
  - model：GCNLayer, EntropyGNN, 损失函数, free_energy
  - training：TrainConfig, schedule_T, train_one
  - anticollapse：VarianceHinge, compute_collapse_metrics, effective_rank, total_variance
  - evaluate_metrics：modularity, NMI, ARI, ACC, kmeans_labels, fmt_mean_std
  - experiments：make_configs, 实验运行器, 绘图
  - plot_config：matplotlib Agg + 中文字体配置（import 副作用）
  - closed_form_gradients：闭式梯度数学验证

用法：
  import wj as m                    # 兼容旧 entropy_gnn_baseline 用法
  from wj import EntropyGNN, train_one, make_configs
  from wj.data import filter_largest_cc, load_cora
"""
# matplotlib 配置（Agg 后端 + 中文字体）—— import 副作用即应用
from . import plot_config

from .runtime import DEVICE, set_seed, _pick_best_device
from .data import (
    make_sbm, make_imbalanced_sbm, normalize_adj, DIFFICULTIES,
    IMBALANCED_SIZES, IMBALANCED_P, IMBALANCED_TRUE_SIZECV, _load_planetoid,
    load_cora, load_citeseer, load_pubmed, load_polblogs, load_amazon_photo,
    CORA_DIR, CORA_N_CLASSES, CORA_N_FEATURES,
    CITESEER_DIR, CITESEER_N_CLASSES, CITESEER_N_FEATURES,
    PUBMED_DIR, PUBMED_N_CLASSES, PUBMED_N_FEATURES,
    POLBLOGS_DIR, POLBLOGS_N_CLASSES,
    AMAZON_PHOTO_DIR, AMAZON_PHOTO_N_CLASSES, AMAZON_PHOTO_N_FEATURES,
    DATASET_LOADERS, filter_largest_cc,
)
from .model import (
    GCNLayer, EntropyGNN, recon_bce, size_entropy,
    assign_entropy, method2_lnw, free_energy,
)
from .evaluate_metrics import (
    modularity, kmeans_labels, size_balance,
    cluster_accuracy, compute_all_metrics,
    compute_acc, compute_nmi, compute_ari, compute_modularity,
    fmt_mean_std, mean_optional,
)
from .training import TrainConfig, schedule_T, train_one
from .anticollapse import (
    VarianceHinge, compute_collapse_metrics,
    effective_rank, total_variance,
)
from .contrastive import (
    graph_contrastive_loss, feature_contrastive_loss, combined_contrastive_loss,
)
from .experiments import (
    make_configs, METRIC_KEYS, run_multi_seed, run_imbalanced,
    run_dataset, _print_metrics, run_difficulty_sweep, run_single_trace,
    IMAGE_DIR, _fig_path, plot_sweep, plot_curves, plot_imbalanced_scatter,
)
