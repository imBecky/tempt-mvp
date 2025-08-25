import h5py
import torch
from torch import nn
import numpy as np
import scipy.io as sio
from config import args
import os
import matplotlib
import matplotlib.pyplot as plt
from torchvision.models import resnet50
matplotlib.use('TkAgg')

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 读取各模态数据为tensor =======================
HSI_file_path = './HSI_full.mat'
LiDAR_file_path = './LiDAR_full.mat'
RGB_file_path = './RGB_full.mat'
HSI_np = sio.loadmat(HSI_file_path)['HSI']
LiDAR_np = sio.loadmat(LiDAR_file_path)['LiDAR']
RGB_np = sio.loadmat(RGB_file_path)['RGB']
HSI_tensor = torch.from_numpy(HSI_np).to(CUDA0)
LiDAR_tensor = torch.from_numpy(LiDAR_np).to(CUDA0)
RGB_tensor = torch.from_numpy(RGB_np).to(CUDA0)

# ===================== 编码各模态数据到维度2048 =======================
# TODO：构建resnet50后编码数据
RGB_encoder = resnet50(pretrained=True).to(CUDA0)
RGB_encoder.fc = nn.Identity()
print(RGB_encoder)
RGB_feature = RGB_encoder(RGB_tensor)
