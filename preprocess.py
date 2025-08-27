import h5py
import torch
from torch import nn
import numpy as np
import scipy.io as sio
from config import args
import os
import matplotlib
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
from torchvision.models import resnet50

matplotlib.use('TkAgg')

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RGB_path = os.path.join(args.data_dir, 'RGB.mat')
LiDAR_path = './LiDAR_AP.mat'
HSI_path = os.path.join(args.data_dir, 'HSI.mat')


def load_higher_mat(path):
    with h5py.File(path, 'r') as file:
        key = list(file.keys())
        data = file[key[0]][()]
        data = data.T
        return data


# RGB_np = sio.loadmat(RGB_path)['data']
# RGB_tensor = torch.from_numpy(RGB_np).to(CUDA0)
LiDAR_np = sio.loadmat(LiDAR_path)['profile']
# HSI_np = load_higher_mat(HSI_path)
# HSI_np = np.transpose(HSI_np, (2, 0, 1))
#
# Resample LiDAR and HSI data to RGB's dimension
LiDAR_tensor = torch.from_numpy(LiDAR_np).unsqueeze(0).float().to(CUDA0)
LiDAR_resampled = F.interpolate(LiDAR_tensor, scale_factor=2, mode='bilinear', align_corners=False)

# HSI_tensor = torch.from_numpy(HSI_np).unsqueeze(0).float().to(CUDA0)
# HSI_resampled = F.interpolate(HSI_tensor, scale_factor=2, mode='bilinear', align_corners=False)
#
LiDAR_resampled_np = LiDAR_resampled.cpu().squeeze(0).detach().numpy()
# HSI_resampled_np = HSI_resampled.cpu().squeeze(0).detach().numpy()


# sio.savemat("LiDAR_interpolated.mat", {"data": LiDAR_resampled_np})
# HSI就不保存了，太大了，后续做了encode之后再保存吧


# 切分patch
def to_patches(x: torch.Tensor, patch_size: int = 224):
    """
        x: (B, C, H, W)   channel-first
        如果 H 或 W 不能被 patch_size 整除，直接裁剪边缘。
        返回: (B, num_patches, C, patch_size, patch_size)
        """
    if x.ndim != 4:
        raise ValueError(f"Expected 4-D tensor, got {x.ndim}-D.")
    c = x.shape[1]
    if c > 200:
        raise ValueError(
            f"Channel dimension ({c}) > 200. "
            f"Are you sure the tensor is in (B, C, H, W) order?"
        )

    p = patch_size
    B, C, H, W = x.shape

    # 裁剪到能被 p 整除
    H_crop = H - (H % p)
    W_crop = W - (W % p)
    x = x[:, :, :H_crop, :W_crop]

    # 切 patch：reshape 成 (B, C, h, p, w, p) 再合并
    x = rearrange(
        x,
        'b c (h p1) (w p2) -> b (h w) c p1 p2',
        p1=p,
        p2=p
    )  # (B, num_patches, C, 224, 224)
    return x


# RGB_patches = to_patches(RGB_tensor.unsqueeze(0), 224).squeeze(0).float()
resnet50 = resnet50(pretrained=True)
resnet50.fc = torch.nn.Identity()     # 输出 2048-d 特征
resnet50 = resnet50.to(CUDA0).eval()

# 前向
# with torch.no_grad():
#     features = resnet50(RGB_patches)      # (567, 2048)
#
# feature_np = features.cpu().numpy()
# sio.savemat('RGB_full.mat', {'data': feature_np})
