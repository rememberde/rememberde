"""data.py — SBM 合成图生成 + 真实数据集加载 + 图归一化。

职责：数据生成与加载的唯一入口。
  - SBM 生成器：make_sbm / make_imbalanced_sbm + 难度常量
  - 图归一化：normalize_adj（D^{-1/2}(A+I)D^{-1/2}）
  - 真实数据集加载器：Cora / CiteSeer / Pubmed / PolBlogs / Amazon Photo
  - DATASET_LOADERS 注册表（供 auto_config / run_dataset / compare_with_baselines 查询）
"""
import os

import numpy as np
import torch
import networkx as nx

from .runtime import DEVICE

# 项目根目录（wj/ 的上一级），数据集放在 <项目根>/data/ 下
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------- SBM 生成器 -----------------------------
def make_sbm(n_per_block: int = 60,
             n_blocks: int = 4,
             p_in: float = 0.5,
             p_out: float = 0.03,
             seed: int = 0):
    sizes = [n_per_block] * n_blocks
    p = [[p_in if i == j else p_out for j in range(n_blocks)]
         for i in range(n_blocks)]
    G = nx.stochastic_block_model(sizes, p, seed=seed)
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    labels = np.array([G.nodes[i]['block'] for i in nodes])
    return A, labels, G


def make_imbalanced_sbm(sizes, p_in: float = 0.3, p_out: float = 0.05,
                        seed: int = 0):
    """不等大小 SBM。size-entropy（强制均衡）在此应受惩罚；
    method2（最大化社区内 spread 而非均衡）不应受影响。"""
    n_blocks = len(sizes)
    p = [[p_in if i == j else p_out for j in range(n_blocks)]
         for i in range(n_blocks)]
    G = nx.stochastic_block_model(sizes, p, seed=seed)
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    labels = np.array([G.nodes[i]['block'] for i in nodes])
    return A, labels, G


def normalize_adj(A_np: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """对称归一化邻接矩阵 Â = D^{-1/2} (A+I) D^{-1/2}，返回 tensor。

    用行/列广播代替稠密对角阵 D_inv_sqrt，避免构造 (N,N) 对角矩阵。
    Pubmed(N=19717) 上单此一项可省 ~1.5GB 显存。
    """
    if device is None:
        device = DEVICE
    A = torch.tensor(A_np, dtype=torch.float32, device=device)
    A_tilde = A + torch.eye(A.shape[0], device=device)
    deg = A_tilde.sum(dim=1)
    # deg_inv = 1/sqrt(deg)，用 rsqrt() 一步到位；广播做行/列缩放
    deg_inv = deg.rsqrt()
    return deg_inv[:, None] * A_tilde * deg_inv[None, :]


# ----------------------------- SBM 难度常量 -----------------------------
DIFFICULTIES = {
    # name:    (n_per_block, n_blocks, p_in, p_out)  -- p_out rising toward threshold
    "easy":   (60, 4, 0.5, 0.03),
    "medium": (40, 4, 0.3, 0.08),
    # hard: p_in/p_out=3.0，有挑战但社区结构可检测（原 0.25/0.12≈2.08 接近检测极限）
    "hard":   (35, 4, 0.3, 0.10),
}

# Imbalanced SBM: blocks of very different sizes.
# size-entropy should hurt (it forces balance); Method-2 should be robust.
IMBALANCED_SIZES = [80, 50, 30, 15]
IMBALANCED_P = (0.3, 0.05)
# 真实 SizeCV = std([80,50,30,15]) / mean([80,50,30,15])，作为不平衡 SBM 的恢复目标
IMBALANCED_TRUE_SIZECV = float(np.std(IMBALANCED_SIZES) / np.mean(IMBALANCED_SIZES))


# ----------------------------- 真实数据集常量 -----------------------------
# Cora 是引用网络（2708 节点 / 7 类 / 1433 维词特征 / 5429 边），
# 用来验证 VarianceHinge + Method 2 在真实图上的泛化性。
CORA_DIR = os.path.join(_PROJECT_ROOT, "data", "cora")
CORA_N_CLASSES = 7
CORA_N_FEATURES = 1433

CITESEER_DIR = os.path.join(_PROJECT_ROOT, "data", "citeseer")
CITESEER_N_CLASSES = 6
CITESEER_N_FEATURES = 3703

PUBMED_DIR = os.path.join(_PROJECT_ROOT, "data", "Pubmed-Diabetes")
PUBMED_N_CLASSES = 3
PUBMED_N_FEATURES = 500

# PolBlogs 是政治博客网络（1490 节点 / 2 类 / 无节点特征），社交领域，小规模。
# 无节点特征 → train_one 走 one-hot I_N 路径（与传统算法公平：传统算法也不用特征）。
POLBLOGS_DIR = os.path.join(_PROJECT_ROOT, "data", "polblogs_pyg")
POLBLOGS_NPZ = os.path.join(_PROJECT_ROOT, "data", "polblogs_npz", "polblogs.npz")
POLBLOGS_N_CLASSES = 2

# Amazon Photo 是购物网络（7650 节点 / 8 类 / 745 维词特征），购物领域，中等规模。
# npz 需用户手动下载（Shchur gnn-benchmark 格式）。
AMAZON_PHOTO_DIR = os.path.join(_PROJECT_ROOT, "data", "amazon_photo")
AMAZON_PHOTO_N_CLASSES = 8
AMAZON_PHOTO_N_FEATURES = 745


# ----------------------------- Planetoid 格式加载器（Cora/CiteSeer 共用） -----------------------------
def _load_planetoid(content_path: str, cites_path: str, n_features: int,
                    dataset_name: str):
    """加载 planetoid 格式数据集（Cora / CiteSeer 共用）。

    格式：
      <name>.content 每行: <paper_id> <n_features 个 0/1 词特征> <class_label>
      <name>.cites   每行: <cited paper> <citing paper>，构建无向图

    Returns:
        A: 邻接矩阵 (N, N) float32，无向
        labels: 类别标签 (N,) int64，从 0 开始编号
        features: 词特征 (N, n_features) float32，二值
    """
    # 1) 读 content：每行 = paper_id + n_features 维特征 + 类别
    paper_ids, features_list, labels_str = [], [], []
    with open(content_path, 'r', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < n_features + 2:
                continue
            paper_ids.append(parts[0])
            features_list.append([int(x) for x in parts[1:1 + n_features]])
            labels_str.append(parts[-1])

    features = np.array(features_list, dtype=np.float32)
    # TF-IDF 加权：仅在特征维度很高（>2000，如 CiteSeer 3703 维）时应用。
    # 动机：高维稀疏二值 BoW 所有词同等权重，TF-IDF 能提升区分度高的词权重。
    # 但 Cora（1433 维）用二值 BoW 已超越 SOTA，TF-IDF 反而破坏其特征结构，故不应用。
    if features.shape[1] > 2000:
        from sklearn.feature_extraction.text import TfidfTransformer
        features = TfidfTransformer().fit_transform(features).toarray().astype(np.float32)
    # 类别字符串 -> 0..K-1（排序保证可复现）
    label_names = sorted(set(labels_str))
    label_map = {name: i for i, name in enumerate(label_names)}
    labels = np.array([label_map[l] for l in labels_str], dtype=np.int64)

    # paper_id -> 行号
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    n = len(paper_ids)

    # 2) 读 cites：构建无向邻接矩阵
    A = np.zeros((n, n), dtype=np.float32)
    with open(cites_path, 'r', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            cited, citing = parts[0], parts[1]
            if cited in id_to_idx and citing in id_to_idx:
                i, j = id_to_idx[cited], id_to_idx[citing]
                A[i, j] = 1.0
                A[j, i] = 1.0  # 无向

    print(f"  {dataset_name} loaded: {n} nodes, {int(A.sum() / 2)} edges, "
          f"{features.shape[1]}-dim features, {len(label_names)} classes")
    return A, labels, features


def load_cora(data_dir: str = None):
    """加载 Cora 数据集（planetoid 格式）。"""
    if data_dir is None:
        data_dir = CORA_DIR
    return _load_planetoid(
        content_path=os.path.join(data_dir, "cora.content"),
        cites_path=os.path.join(data_dir, "cora.cites"),
        n_features=CORA_N_FEATURES,
        dataset_name="Cora",
    )


def load_citeseer(data_dir: str = None):
    """加载 CiteSeer 数据集（planetoid 格式）。"""
    if data_dir is None:
        data_dir = CITESEER_DIR
    return _load_planetoid(
        content_path=os.path.join(data_dir, "citeseer.content"),
        cites_path=os.path.join(data_dir, "citeseer.cites"),
        n_features=CITESEER_N_FEATURES,
        dataset_name="CiteSeer",
    )


def load_pubmed(data_dir: str = None):
    """加载 Pubmed-Diabetes 数据集（.tab 稀疏格式）。

    文件格式：
      Line 0: header  "NODE\\tpaper"
      Line 1: schema  "cat=1,2,3:label\\tnumeric:w-rat:0.0\\t...\\tstring:summary"
                       (1 label + 500 numeric features + 1 summary = 502 列)
      Line 2+: data    "paper_id\\tlabel=N\\tw-name=value\\t...\\tsummary=..."
                       每篇论文只列非零特征（稀疏格式）

    需要从 schema 解析 w-name 到列索引的映射，再把稀疏值展开成 500 维向量。

    Returns: A (19717, 19717), labels (19717,), features (19717, 500)
    """
    if data_dir is None:
        data_dir = PUBMED_DIR
    node_path = os.path.join(data_dir, "data", "Pubmed-Diabetes.NODE.paper.tab")
    cites_path = os.path.join(data_dir, "data", "Pubmed-Diabetes.DIRECTED.cites.tab")

    # ---- 第1遍：读 schema 构建 w-name -> 列索引映射 ----
    feat_name_to_idx = {}
    with open(node_path, 'r', errors='ignore') as f:
        next(f)  # skip header "NODE\tpaper"
        schema = next(f).strip().split('\t')
        col_idx = 0
        for tok in schema:
            # token 形如 "numeric:w-rat:0.0" 或 "cat=1,2,3:label" 或 "string:summary"
            if tok.startswith("numeric:"):
                # "numeric:w-rat:0.0" -> feature name = "w-rat"
                name = tok.split(":")[1]
                feat_name_to_idx[name] = col_idx
                col_idx += 1
            else:
                # cat=...:label 或 string:summary：不占 feature 列
                pass
        n_features = len(feat_name_to_idx)
        print(f"  [schema] 解析到 {n_features} 个 numeric features（期望 {PUBMED_N_FEATURES}）")

    # ---- 第2遍：读每篇论文，展开稀疏特征到稠密向量 ----
    paper_ids, features_list, labels_str = [], [], []
    with open(node_path, 'r', errors='ignore') as f:
        next(f); next(f)  # skip header + schema
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            pid = parts[0]
            feat_vec = np.zeros(n_features, dtype=np.float32)
            label = None
            for tok in parts[1:]:
                if tok.startswith("label="):
                    label = tok[6:]
                elif tok.startswith("w-"):
                    # "w-rat=0.0939..." -> name="w-rat", val=0.0939...
                    if '=' in tok:
                        name, val_str = tok.split('=', 1)
                        if name in feat_name_to_idx:
                            try:
                                feat_vec[feat_name_to_idx[name]] = float(val_str)
                            except ValueError:
                                pass
                # summary= 忽略
            if label is None:
                continue
            paper_ids.append(pid)
            features_list.append(feat_vec)
            labels_str.append(label)

    features = np.array(features_list, dtype=np.float32)
    label_names = sorted(set(labels_str))
    label_map = {name: i for i, name in enumerate(label_names)}
    labels = np.array([label_map[l] for l in labels_str], dtype=np.int64)
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    n = len(paper_ids)
    A = np.zeros((n, n), dtype=np.float32)
    with open(cites_path, 'r', errors='ignore') as f:
        next(f); next(f)  # skip header + schema
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            # DIRECTED.cites.tab 格式：edge_id\tpaper:CITED\t|\tpaper:CITING
            # 去掉 "paper:" 前缀，留下纯数字 ID（与 NODE.paper.tab 的 paper_id 一致）
            cited = parts[1].split(':')[-1]
            citing = parts[3].split(':')[-1]
            if cited in id_to_idx and citing in id_to_idx:
                i, j = id_to_idx[cited], id_to_idx[citing]
                A[i, j] = 1.0; A[j, i] = 1.0
    print(f"  Pubmed loaded: {n} nodes, {int(A.sum() / 2)} edges, "
          f"{features.shape[1]}-dim features, {len(label_names)} classes")
    return A, labels, features


def load_polblogs(data_dir: str = None):
    """加载 PolBlogs 数据集（政治博客网络，1490 节点，2 类）。

    优先用 npz 格式（data/polblogs_npz/polblogs.npz，无需 PyG 依赖），
    若 npz 不存在则降级到 PyG 格式（data/polblogs_pyg/）。
    无节点特征（返回 None），train_one 会自动用 one-hot I_N。

    Returns: A (1490, 1490) float32, labels (1490,) int64, features=None
    """
    # 优先用 npz 格式（无需 PyG 依赖，服务器部署更方便）
    npz_path = POLBLOGS_NPZ if data_dir is None else os.path.join(data_dir, "polblogs.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        A = d['adj_data'].astype(np.float32)
        labels = d['label'].astype(np.int64)
        n = A.shape[0]
        print(f"  PolBlogs loaded (npz): {n} nodes, {int(A.sum() / 2)} edges, "
              f"{len(np.unique(labels))} classes, no node features (use one-hot)")
        return A, labels, None
    # 降级：PyG 格式
    from torch_geometric.utils import to_scipy_sparse_matrix
    if data_dir is None:
        data_dir = POLBLOGS_DIR
    from torch_geometric.datasets import PolBlogs
    ds = PolBlogs(root=data_dir)
    data = ds[0]
    n = data.num_nodes
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).toarray().astype(np.float32)
    A = np.maximum(A, A.T)  # 保证对称（PyG 可能存有向边）
    labels = data.y.cpu().numpy().astype(np.int64)
    print(f"  PolBlogs loaded (pyg): {n} nodes, {int(A.sum() / 2)} edges, "
          f"{len(np.unique(labels))} classes, no node features (use one-hot)")
    return A, labels, None


def load_amazon_photo(data_dir: str = None):
    """加载 Amazon Photo 数据集（购物网络，7650 节点，8 类）。

    数据来源：Shchur gnn-benchmark npz 格式（用户手动下载）。
    npz 采用 Shchur 标准格式：adj_*/attr_*/label 等字段存储稀疏矩阵。

    Returns: A (7650, 7650) float32, labels (7650,) int64, features (7650, 745) float32
    """
    import scipy.sparse as sp
    if data_dir is None:
        data_dir = AMAZON_PHOTO_DIR
    npz_path = os.path.join(data_dir, "amazon_electronics_photo.npz")
    # 检查文件存在且非空（PyG 自动下载失败会留 0 字节文件）
    if not os.path.exists(npz_path) or os.path.getsize(npz_path) < 1024:
        raise FileNotFoundError(
            f"Amazon Photo npz 未找到或下载不完整: {npz_path}\n"
            f"请从 https://github.com/shchur/gnn-benchmark/raw/master/data/npz/"
            f"amazon_electronics_photo.npz 下载并放到 {data_dir}/"
            f"\n（正常文件约 5-10MB，若小于 1MB 说明下载不完整）"
        )
    loader = np.load(npz_path, allow_pickle=True)
    keys = list(loader.keys())
    # Shchur 标准格式：adj_data/adj_indices/adj_indptr/adj_shape + attr_* + label
    if 'adj_data' in keys:
        A_sparse = sp.csr_matrix(
            (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape'])
    else:
        # 备用格式：直接存 'A' 稀疏矩阵
        A_sparse = sp.csr_matrix(loader['A'])
    A = A_sparse.toarray().astype(np.float32)
    A = np.maximum(A, A.T)  # 保证对称
    # Shchur 格式键名可能是 'label' 或 'labels'（复数），兼容两种
    label_key = 'labels' if 'labels' in keys else 'label'
    labels = np.asarray(loader[label_key]).astype(np.int64).ravel()
    # 特征：attr_*（稀疏）或 feat（稠密）
    if 'attr_data' in keys:
        feat_sparse = sp.csr_matrix(
            (loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
            shape=loader['attr_shape'])
        features = np.asarray(feat_sparse.todense(), dtype=np.float32)
    elif 'feat' in keys:
        features = np.asarray(loader['feat'], dtype=np.float32)
    elif 'x' in keys:
        features = np.asarray(loader['x'], dtype=np.float32)
    else:
        features = None
    print(f"  Amazon Photo loaded: {A.shape[0]} nodes, {int(A.sum() / 2)} edges, "
          f"{features.shape[1] if features is not None else 'no'}-dim features, "
          f"{len(np.unique(labels))} classes")
    return A, labels, features


# ----------------------------- 图预处理工具 -----------------------------
def filter_largest_cc(A: np.ndarray, labels: np.ndarray, features=None):
    """过滤到最大连通分量（GNN 标准预处理）。

    碎片化图（如 CiteSeer 438 个 CC）中，GCN 消息传递无法到达孤立节点，
    导致 WJ 系统性落后于不依赖消息传递的传统算法。过滤到最大 CC 后，
    所有方法在同一连通子图上公平对比。

    对已连通的图（Cora/Pubmed/PolBlogs）是 no-op（只丢几个孤立点）。
    """
    G = nx.from_numpy_array(A)
    largest_cc = max(nx.connected_components(G), key=len)
    keep = sorted(largest_cc)
    A_f = A[np.ix_(keep, keep)]
    labels_f = labels[keep]
    features_f = features[keep] if features is not None else None
    n_old = A.shape[0]
    n_new = len(keep)
    n_ccs = nx.number_connected_components(G)
    if n_new < n_old:
        print(f"  [CC filter] {n_old} -> {n_new} nodes "
              f"(removed {n_old - n_new} isolated, {n_ccs} CCs -> 1)")
    return A_f, labels_f, features_f


# ----------------------------- 数据集注册表 -----------------------------
# 格式：name -> (loader_fn, n_classes, default_emb_dim, default_min_rank)
DATASET_LOADERS = {
    'cora': (load_cora, CORA_N_CLASSES, 32, 5.0),
    'citeseer': (load_citeseer, CITESEER_N_CLASSES, 64, 4.0),
    'pubmed': (load_pubmed, PUBMED_N_CLASSES, 64, 2.5),
    'polblogs': (load_polblogs, POLBLOGS_N_CLASSES, 16, 1.5),
    'amazon_photo': (load_amazon_photo, AMAZON_PHOTO_N_CLASSES, 32, 4.0),
}
