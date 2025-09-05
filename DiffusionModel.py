import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import matplotlib
import matplotlib.pyplot as plt
import torch_utils as utils
matplotlib.use('TkAgg')

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
