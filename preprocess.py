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


# ===================== 读取各模态数据 =======================
HSI_file_path = os.path.join(args.data_dir, 'HSI.mat')
LiDAR_file_path = os.path.join(args.data_dir, 'LiDAR.mat')
RGB_file_path = os.path.join(args.data_dir, 'RGB.mat')
with h5py.File(HSI_file_path, 'r') as file:
    key = list(file.keys())
    data = file[key[0]][()]
    data = data.T
    HSI_dataset = torch.from_numpy(np.transpose(data, (2, 0, 1)))
with h5py.File(LiDAR_file_path, 'r') as file:
    key = list(file.keys())
    data = file[key[0]][()]
    data = data.T
    LiDAR_dataset = torch.from_numpy(data).unsqueeze(0)

RGB_np = sio.loadmat(RGB_file_path)['data']

# ===================== 处理RGB =======================
H, W, C = RGB_np.shape
print('原图尺寸：', H, W, C)
# 转成 float32，范围 [0,1]
RGB_dataset = torch.from_numpy(RGB_np).float() / 255.0
RGB_dataset = RGB_dataset.unsqueeze(0)
plt.figure(figsize=(6, 3))
plt.title('Original RGB')
plt.imshow(RGB_np.transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()
plt.show()

down = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)

with torch.no_grad():
    down.weight.zero_()
    # 每个输出通道只取对应输入通道中心像素
    down.weight[0, 0, 1, 1] = 1.0
    down.weight[1, 1, 1, 1] = 1.0
    down.weight[2, 2, 1, 1] = 1.0
    down_tensor = down(RGB_dataset)   # (1, 3,H/2,W/2)
down_np = down_tensor.squeeze(0).permute(1, 2, 0).numpy()

plt.figure(figsize=(6, 3))
plt.title('Downsampled RGB (Conv stride=2)')
plt.imshow(down_np)
plt.axis('off')
plt.tight_layout()
plt.show()
# 5. 保存新的 mat -------------------------------------------------------------
save_dict = {'rgb_down': (down_np*255).astype('uint8')}
sio.savemat('rgb_down.mat', save_dict)
print('已保存为 rgb_down.mat')
