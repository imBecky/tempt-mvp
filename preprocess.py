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
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
from skimage.morphology import reconstruction, disk
from torchvision.models import resnet50

matplotlib.use('TkAgg')

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RGB_path = os.path.join(args.data_dir, 'RGB.mat')
# LiDAR_path = os.path.join(args.data_dir, 'LiDAR.mat')
# HSI_path = os.path.join(args.data_dir, 'HSI.mat')


def load_higher_mat(path):
    with h5py.File(path, 'r') as file:
        key = list(file.keys())
        data = file[key[0]][()]
        data = data.T
        return data


def AP_LiDAR(path):
    with h5py.File(path, 'r') as file:
        key = list(file.keys())
        LiDAR_np = file[key[0]][()]
        LiDAR_np = LiDAR_np.T[1:, 1:]
        LiDAR_tensor = torch.from_numpy(LiDAR_np).unsqueeze(0).to(CUDA0)
    dsm = LiDAR_np.astype(np.float32)
    H, W = dsm.shape
    # threshold list
    lambdas = [0.5]

    profile = [dsm]
    se = disk(1)
    for lam in lambdas:
        marker = dsm - lam
        opened = reconstruction(marker, dsm, method='dilation')
        profile.append(opened)

        marker = dsm + lam
        closed = reconstruction(marker, dsm, method='erosion')
        profile.append(closed)

    profile = np.stack(profile, axis=0)
    return profile


# RGB_np = sio.loadmat(RGB_path)['data']
# RGB_tensor = torch.from_numpy(RGB_np).to(CUDA0)
# LiDAR_np = sio.loadmat('./LiDAR_AP.mat')['profile'][:, 1:, 1:]
# HSI_np = load_higher_mat(HSI_path)
# HSI_np = np.transpose(HSI_np, (2, 0, 1))
#
# Resample LiDAR and HSI data to RGB's dimension
# LiDAR_tensor = torch.from_numpy(LiDAR_np).unsqueeze(0).float().to(CUDA0)
# LiDAR_resampled = F.interpolate(LiDAR_tensor, scale_factor=2, mode='bilinear', align_corners=False)
# HSI_tensor = torch.from_numpy(HSI_np).unsqueeze(0).float().to(CUDA0)
# HSI_resampled = F.interpolate(HSI_tensor, scale_factor=2, mode='bilinear', align_corners=False)
#
# LiDAR_resampled_np = LiDAR_resampled.cpu().squeeze(0).detach().numpy()

# HSI_resampled_np = HSI_resampled.cpu().squeeze(0).detach().numpy()


# sio.savemat("LiDAR_full.mat", {"data": LiDAR_resampled_np})
# HSI就不保存了，太大了，后续做了encode之后再保存吧


# 切分patch
def to_patches(x, patch_size: int = 224):
    """
        x: (B, C, H, W)   channel-first
        如果 H 或 W 不能被 patch_size 整除，直接裁剪边缘。
        返回: (B, num_patches, C, patch_size, patch_size)
        """
    x_tensor = torch.from_numpy(x).to(CUDA0)
    if x_tensor.ndim == 3:
        x_tensor = x_tensor.unsqueeze(0)
    if x_tensor.ndim != 4:
        raise ValueError(f"Expected 4-D tensor, got {x_tensor.ndim}-D.")
    c = x_tensor.shape[1]
    if c > 200:
        raise ValueError(
            f"Channel dimension ({c}) > 200. "
            f"Are you sure the tensor is in (B, C, H, W) order?"
        )

    p = patch_size
    B, C, H, W = x_tensor.shape

    # 裁剪到能被 p 整除
    H_crop = H - (H % p)
    W_crop = W - (W % p)
    x_tensor = x_tensor[:, :, :H_crop, :W_crop]

    # 切 patch：reshape 成 (B, C, h, p, w, p) 再合并
    x_tensor = rearrange(
        x_tensor,
        'b c (h p1) (w p2) -> b (h w) c p1 p2',
        p1=p,
        p2=p
    )  # (B, num_patches, C, 224, 224)
    x_tensor = x_tensor.squeeze(0)  # (num_patches, C, 224, 224)
    return x_tensor


# RGB_patches = to_patches(RGB_tensor.unsqueeze(0), 224).squeeze(0).float()
resnet50 = resnet50(pretrained=True)
resnet50.fc = torch.nn.Identity()     # 输出 2048-d 特征
resnet50 = resnet50.to(CUDA0).eval()

# RGB_full = sio.loadmat('./RGB_full.mat')['data']
# train_size = int(0.7 * RGB_full.shape[0])
# test_size = int(0.3 * RGB_full.shape[0])
# train_np = RGB_full[:train_size, :]
# test_np = RGB_full[train_size:, :]
# sio.savemat('RGB_TrSet.mat', {'data': train_np})
# sio.savemat('RGB_TeSet.mat', {'data': test_np})

# label_full = sio.loadmat(os.path.join(args.data_dir, 'label.mat'))['data']
# label_patches = to_patches(torch.from_numpy(label_full).unsqueeze(0).unsqueeze(0), 224).squeeze(0)
# label_patches = label_patches.squeeze(1)
# label_train_np = label_patches[:int(0.7*567), :]
# label_test_np = label_patches[int(0.7*567):, :]
#
# sio.savemat('label_TrSet.mat', {'data': label_train_np})
# sio.savemat('label_TeSet.mat', {'data': label_test_np})

# RGB_full = sio.loadmat(os.path.join(args.data_dir, 'RGB.mat'))['data'][:, 2:, 2:]
# LiDAR_full = sio.loadmat('./LiDAR_full.mat')['data']
# HSI_full = load_higher_mat(os.path.join(args.data_dir, 'HSI.mat'))
# HSI_full = np.transpose(HSI_full, [2, 0, 1])[:, 1:, 1:]
# label = sio.loadmat(os.path.join(args.data_dir, 'label.mat'))['data'][2:, 2:]
#
# RGB_patches = to_patches(RGB_full)
# LiDAR_patches = to_patches(LiDAR_full)
# with torch.no_grad():
#     features = resnet50(LiDAR_patches.float())      # (567, 2048)
# LiDAR_feature_np = features.cpu().numpy()
# sio.savemat('LiDAR.mat', {'data': LiDAR_feature_np})

LiDAR_file_path = os.path.join(args.data_dir, 'LiDAR.mat')
profile = AP_LiDAR(LiDAR_file_path)
sio.savemat('LiDAR_AP.mat', {'data': profile})

