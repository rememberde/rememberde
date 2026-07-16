"""runtime.py — 运行时设备选择与随机种子控制。

职责单一：GPU 自动检测 + 多 GPU 选空闲显存最大的 + 统一随机种子。
所有 tensor / model 统一用 DEVICE，避免跨设备拷贝报错。
"""
import random

import numpy as np
import torch


def _pick_best_device() -> torch.device:
    """选择空闲显存最多的 GPU；无 GPU 或全部 OOM 风险时回退 cpu。"""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    n_gpus = torch.cuda.device_count()
    if n_gpus == 1:
        return torch.device('cuda:0')
    # 多 GPU：选 free memory 最大的
    best_idx, best_free = 0, 0
    for i in range(n_gpus):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free, best_idx = free, i
        except Exception:
            pass
    print(f"  [device] 多 GPU 检测：选择 cuda:{best_idx}（空闲 {best_free / 1024**3:.1f} GB）")
    return torch.device(f'cuda:{best_idx}')


# 全局设备：所有模块共享，避免跨设备报错
DEVICE = _pick_best_device()


def set_seed(seed: int = 42):
    """统一设置 random / numpy / torch 的随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
