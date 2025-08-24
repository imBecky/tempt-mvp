import h5py
import torch
from torch import nn
import numpy as np
import scipy.io as sio
from config import args
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 读取各模态数据为tensor =======================
HSI_file_path = os.path.join(args.data_dir, 'HSI.mat')
LiDAR_file_path = './LiDAR_AP.mat'
RGB_file_path = './rgb_down.mat'
# with h5py.File(HSI_file_path, 'r') as file:
#     key = list(file.keys())
#     data = file[key[0]][()]
#     data = data.T
#     HSI_tensor = torch.from_numpy(np.transpose(data, (2, 0, 1))).to(CUDA0)
LiDAR_np = sio.loadmat(LiDAR_file_path)['profile']
LiDAR_tensor = torch.from_numpy(LiDAR_np).to(CUDA0)
RGB_np = sio.loadmat(RGB_file_path)['rgb_down']
RGB_tensor = torch.from_numpy(RGB_np).to(CUDA0)

RGB_flatten = RGB_np.reshape(-1, 3)

save_dict = {'RGB': RGB_flatten}
sio.savemat('RGB_full.mat', save_dict)
