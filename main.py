import torch
import numpy as np
from config import args
import os
import scipy.io as sio
from Fusion import train_rgb_lidar_hsi
from DiffusionModel import train_with_augmentation
from HSI_model import train_hsi
from RGB_LiDAR_Model import train
import torch_utils as utils

# print(os.getcwd())

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING = 1

if __name__ == "__main__":
    RGB_TrSet = sio.loadmat('./data/RGB_TrSet.mat')['data']
    RGB_TeSet = sio.loadmat('./data/RGB_TeSet.mat')['data']
    LiDAR_TrSet = sio.loadmat('./data/LiDAR_TrSet.mat')['data']
    LiDAR_TeSet = sio.loadmat('./data/LiDAR_TeSet.mat')['data']
    HSI_TrSet = sio.loadmat('./data/HSI_TrSet.mat')['data']
    HSI_TeSet = sio.loadmat('./data/HSI_TeSet.mat')['data']
    label_TrSet = sio.loadmat('./data/label_TrSet.mat')['data']
    label_TeSet = sio.loadmat('./data/label_TeSet.mat')['data']
    # parameters, val_acc = train_rgb_lidar_hsi(RGB_TrSet, LiDAR_TrSet, HSI_TrSet, label_TrSet,
    #                                           RGB_TeSet, LiDAR_TeSet, HSI_TeSet, label_TeSet,
    #                                           args)
    parameters, val_acc = train_with_augmentation(args, RGB_TrSet, LiDAR_TrSet, HSI_TrSet, label_TrSet,
                                                  RGB_TeSet, LiDAR_TeSet, HSI_TeSet, label_TeSet)
