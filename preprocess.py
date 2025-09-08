import h5py
import torch
from torch import nn
import numpy as np
import scipy.io as sio
from config import args
from spectral import envi
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


def mat2hdr(path):
    cube = load_higher_mat(path)
    envi.save_image('HSI.hdr', cube, dtype='float32', interleave='bsq')
    print('hdr file has been saved!')


def hdr2mat(path):
    cube = envi.open(path).load()
    cube = cube.transpose([2, 0, 1])
    print()
    return cube


# 切分patch
def to_patches(x, patch_size: int = 224):
    """
        x: (B, C, H, W)   channel-first
        如果 H 或 W 不能被 patch_size 整除，直接裁剪边缘。
        返回: (B, num_patches, C, patch_size, patch_size)
        """
    if torch.is_tensor(x):
        x_tensor = x
    else:
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


def Encode_3D(data_np):
    patches = to_patches(data_np)
    with torch.no_grad():
        features = resnet50(patches.float())
    feature_np = features.cpu().numpy()
    return feature_np


def Encode_HSI(HSI_np):
    H, W = HSI_np.shape[1:]
    groups = HSI_np.reshape(-1, 3, H, W)    # 把30维光谱的HSI数据按3维为一组分组
    HSI_feature_np = np.zeros((10, 567, 2048), dtype=np.float32)
    for i, spectral in enumerate(groups):
        encoded_feature = Encode_3D(spectral)
        HSI_feature_np[i] = encoded_feature
    sio.savemat('HSI.mat', {'data': HSI_feature_np})
    print()


# RGB_patches = to_patches(RGB_tensor.unsqueeze(0), 224).squeeze(0).float()
resnet50 = resnet50(pretrained=True)
resnet50.fc = torch.nn.Identity()     # 输出 2048-d 特征
resnet50 = resnet50.to(CUDA0).eval()

label_full = sio.loadmat(os.path.join(args.data_dir, 'label.mat'))['data']
label_patches = to_patches(torch.from_numpy(label_full).unsqueeze(0).unsqueeze(0), 224).squeeze(0)
label_patches = label_patches.squeeze(1)
label_train_np = label_patches[:int(0.7*567), :]
label_test_np = label_patches[int(0.7*567):, :]

sio.savemat('label_TrSet.mat', {'data': label_train_np})
sio.savemat('label_TeSet.mat', {'data': label_test_np})
print()
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

"""
对LiDAR做三维的AP
"""
# LiDAR_file_path = os.path.join(args.data_dir, 'LiDAR.mat')
# profile = AP_LiDAR(LiDAR_file_path)
# sio.savemat('LiDAR_AP.mat', {'data': profile})

"""
转换hsi为ENVI格式hdr，再用ENVI做pca降维
"""
# hdr_path = './HSI_PCA.HDR'
# cube = hdr2mat(hdr_path)
# sio.savemat('HSI_PCA.mat', {'data': cube})

"""
对三维AP后的LiDAR, HSI做上采样
"""
# LiDAR_path = './LiDAR_AP.mat'
# LiDAR = sio.loadmat(LiDAR_path)['data']
# LiDAR_tensor = torch.from_numpy(LiDAR).unsqueeze(0).float().to(CUDA0)
# LiDAR_resampled = F.interpolate(LiDAR_tensor, scale_factor=2, mode='bilinear', align_corners=False)
# LiDAR_resampled_np = LiDAR_resampled.cpu().squeeze(0).detach().numpy()
# sio.savemat('LiDAR.mat', {'data': LiDAR_resampled_np})
# HSI_path = './HSI.mat'
# HSI = sio.loadmat(HSI_path)['data']
# HSI_tensor = torch.from_numpy(HSI).unsqueeze(0).float().to(CUDA0)
# HSI_resampled = F.interpolate(HSI_tensor, scale_factor=2, mode='bilinear', align_corners=False)
# HSI_resampled_np = HSI_resampled.cpu().squeeze(0).detach().numpy()
# sio.savemat('HSI.mat', {'data': HSI_resampled_np})

"""
编码各模态数据
"""
# RGB_np = sio.loadmat('./RGB.mat')['data']   # RGB已经被编码好了
# LiDAR_np = sio.loadmat('./LiDAR_interpolated.mat')['data']
# HSI_np = sio.loadmat('./HSI.mat')['data']
# Encoded_LiDAR = Encode_3D(LiDAR_np)
# sio.savemat('LiDAR.mat', {'data': Encoded_LiDAR})
# Encode_HSI(HSI_np)

"""
切割Train Test数据集
"""
# RGB_full = sio.loadmat('./RGB.mat')['data']
# LiDAR_full = sio.loadmat('./LiDAR.mat')['data']
HSI_full = sio.loadmat('./HSI.mat')['data'].transpose(1, 2, 0)
# label_full = sio.loadmat('./label.mat')['data']
#
train_size = int(0.7 * HSI_full.shape[0])
test_size = int(0.3 * HSI_full.shape[0])
# # RGB
# train_np = RGB_full[:train_size, :]
# test_np = RGB_full[train_size:, :]
# sio.savemat('RGB_TrSet.mat', {'data': train_np})
# sio.savemat('RGB_TeSet.mat', {'data': test_np})
# # LiDAR
# train_np = LiDAR_full[:train_size, :]
# test_np = LiDAR_full[train_size:, :]
# sio.savemat('LiDAR_TrSet.mat', {'data': train_np})
# sio.savemat('LiDAR_TeSet.mat', {'data': test_np})
# # HSI
train_np = HSI_full[:train_size, :, :]
test_np = HSI_full[train_size:, :, :]
sio.savemat('HSI_TrSet.mat', {'data': train_np})
sio.savemat('HSI_TeSet.mat', {'data': test_np})
# label

print()
