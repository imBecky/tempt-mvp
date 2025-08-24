import h5py
import torch
from torch import nn
import numpy as np
import scipy.io as sio
from config import args
import os
import matplotlib
import matplotlib.pyplot as plt
from skimage.morphology import reconstruction, disk
from skimage.filters import rank
matplotlib.use('TkAgg')

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 读取各模态数据为tensor =======================
# HSI_file_path = os.path.join(args.data_dir, 'HSI.mat')
LiDAR_file_path = os.path.join(args.data_dir, 'LiDAR.mat')
# RGB_file_path = './rgb_down.mat'
# with h5py.File(HSI_file_path, 'r') as file:
#     key = list(file.keys())
#     data = file[key[0]][()]
#     data = data.T
#     HSI_tensor = torch.from_numpy(np.transpose(data, (2, 0, 1))).to(CUDA0)
with h5py.File(LiDAR_file_path, 'r') as file:
    key = list(file.keys())
    LiDAR_np = file[key[0]][()]
    LiDAR_np = LiDAR_np.T
    LiDAR_tensor = torch.from_numpy(LiDAR_np).unsqueeze(0).to(CUDA0)
#
# RGB_np = sio.loadmat(RGB_file_path)['rgb_down'].transpose(2, 0, 1)
# RGB_tensor = torch.from_numpy(RGB_np).to(CUDA0)

# ========= 对LiDAR做AP ============
dsm = LiDAR_np[1:, 1:].astype(np.float32)
H, W = dsm.shape

# 阈值列表
lambdas = np.arange(2, 22, 2)   # 2,4,...,20 → 10 个

profile = [dsm]                 # 第 0 层放原图
se = disk(1)                    # 3×3 结构元

for lam in lambdas:
    # 开运算 (opening by reconstruction)
    marker = dsm - lam
    opened = reconstruction(marker, dsm, method='dilation')
    profile.append(opened)

    # 关运算 (closing by reconstruction)
    marker = dsm + lam
    closed = reconstruction(marker, dsm, method='erosion')
    profile.append(closed)

profile = np.stack(profile, axis=0)   # (21, H, W)
profile = np.insert(profile, 0, -1, axis=1)
profile = np.insert(profile, 0, -1, axis=2)
save_dict = {'profile': profile}
sio.savemat('LiDAR_AP.mat', save_dict)
