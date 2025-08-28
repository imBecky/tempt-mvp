import torch
import random
import os
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def l2_loss(model):
    """返回模型所有权重矩阵的 L2 范数平方和（不含 bias 和 BN）"""
    l2 = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2 += torch.sum(param**2)
    return l2


def random_mini_batches(x, y, batch_size, seed):
    np.random.seed(seed)
    m = x.shape[0]
    permutation = np.random.permutation(m)
    num_batches = int(np.ceil(m / batch_size))
    batches = []
    for k in range(num_batches):
        idx = permutation[k*batch_size: (k+1)*batch_size]
        batches.append((x[idx], y[idx]))
    return batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
