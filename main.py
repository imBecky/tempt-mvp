import torch
import numpy as np
from config import args
import os
import scipy.io as sio
from tempt_HSI import train_hsi
from MyModel import train
import torch_utils as utils
print(os.getcwd())

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING = 1


if __name__ == "__main__":
    RGB_TrSet = sio.loadmat('./RGB_TrSet.mat')['data']
    RGB_TeSet = sio.loadmat('./RGB_TeSet.mat')['data']
    LiDAR_TrSet = sio.loadmat('./LiDAR_TrSet.mat')['data']
    LiDAR_TeSet = sio.loadmat('./LiDAR_TeSet.mat')['data']
    HSI_TrSet = sio.loadmat('./HSI_TrSet.mat')['data']
    HSI_TeSet = sio.loadmat('./HSI_TeSet.mat')['data']
    label_TrSet = sio.loadmat('./label_TrSet.mat')['data']
    label_TeSet = sio.loadmat('./label_TeSet.mat')['data']
    parameters, val_acc = train_hsi(HSI_TrSet, HSI_TeSet, label_TrSet, label_TeSet, args)
    print()
