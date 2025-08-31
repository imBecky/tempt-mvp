import torch
from config import args
import os
import scipy.io as sio
from MyModel import train
print(os.getcwd())

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING = 1


if __name__ == "__main__":
    args = args
    RGB_TrSet = sio.loadmat('./RGB_TrSet.mat')['data']
    RGB_TeSet = sio.loadmat('./RGB_TeSet.mat')['data']
    label = sio.loadmat(os.path.join(args.data_dir, 'label.mat'))['data']

