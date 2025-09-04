import torch
import random
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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


def plot_label(y):
    plt.figure(figsize=(12, 9))  # 按需要调大或调小
    im = plt.imshow(y, cmap='tab20', vmin=0, vmax=25)  # tab20 能区分 20+ 类
    plt.colorbar(im, ticks=np.unique(y))  # 右侧色条
    plt.title('Label map')
    plt.show()
