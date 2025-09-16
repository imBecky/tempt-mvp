"""
@author: danfeng Hong
implemented by Binqian Huang
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import torch_utils as utils
matplotlib.use('TkAgg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


def initialize_parameters():
    utils.set_seed(1)


class EndNet(nn.Module):
    def __init__(self):
        super().__init__()
        # x1 encoder: 144 → 16 → 32 → 64 → 128
        self.x1_enc = nn.Sequential(
            nn.Linear(144, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        # x2 encoder: 21 → 16 → 32 → 64 → 128
        self.x2_enc = nn.Sequential(
            nn.Linear(21, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        # joint: 256 → 128 → 64 → 15
        self.joint_enc = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 15)  # followed with cross_entropy so no ReLU
        )
        # decoder: 128 → 64 → 32 → 16 → original dim
        self.x1_dec = nn.Sequential(
            nn.Linear(128, 64), nn.Sigmoid(),  # why there's not ReLU anymore, and no BN?
            nn.Linear(64, 32), nn.Sigmoid(),
            nn.Linear(32, 16), nn.Sigmoid(),
            nn.Linear(16, 144), nn.Sigmoid()
        )
        self.x2_dec = nn.Sequential(
            nn.Linear(128, 64), nn.Sigmoid(),
            nn.Linear(64, 32), nn.Sigmoid(),
            nn.Linear(32, 16), nn.Sigmoid(),
            nn.Linear(16, 21), nn.Sigmoid()
        )

    def forward(self, x1, x2):
        h1 = self.x1_enc(x1)
        h2 = self.x2_enc(x2)
        joint = torch.cat([h1, h2], dim=1)
        logits = self.joint_enc(joint)
        x1_rec = self.x1_dec(h1)
        x2_rec = self.x2_dec(h2)
        return logits, x1_rec, x2_rec


def train(x1_train, x2_train, x1_train_full, x2_train_full,
          x1_test, x2_test,
          y_train, y_test,
          lr_base=1e-3, beta_reg=1e-3, num_epochs=150,
          batch_size=64, print_cost=True):
    """
    :param x1_train: HSI data(reduced dimension) for VAE
    :param x2_train: LiDAR data(reduced dimension) for VAE
    :param x1_train_full: full HSI data for reconstruction loss
    :param x2_train_full: full LiDAR data for reconstruction loss
    """
    x1_train = torch.tensor(x1_train).to(device)
    x2_train = torch.tensor(x2_train).to(device)
    x1_train_full = torch.tensor(x1_train_full).to(device)
    x2_train_full = torch.tensor(x2_train_full).to(device)
    x1_test = torch.tensor(x1_test, dtype=torch.float32).to(device)
    x2_test = torch.tensor(x2_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train).to(device)
    y_test = torch.tensor(y_test).to(device)

    model = EndNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_base)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    y_train_cls = torch.argmax(y_train, dim=1)  # one_hot → cls idx
    y_test_cls = torch.argmax(y_test, dim=1)

    # recording
    costs, costs_dev = [], []  # what's the meaning of costs_dev
    train_acc, val_acc = [], []

    seed = 1
    m = x1_train.size(0)  # what does this mean?

    # train
    for epoch in range(num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        seed += 1  # make sure every epoch is shuffled
        minibatches = utils.random_mini_batches(x1_train.cpu().numpy(),
                                                x2_train.cpu().numpy(),
                                                x1_train_full.cpu().numpy(),
                                                x2_train_full.cpu().numpy(),
                                                y_train.cpu().numpy(),
                                                batch_size, seed)
        num_batches = len(minibatches)

        for (mb_x1, mb_x2, mb_x1f, mb_x2f, mb_y) in minibatches:
            mb_x1 = torch.tensor(mb_x1, dtype=torch.float32).to(device)
            mb_x2 = torch.tensor(mb_x2, dtype=torch.float32).to(device)
            mb_x1f = torch.tensor(mb_x1f, dtype=torch.float32).to(device)
            mb_x2f = torch.tensor(mb_x2f, dtype=torch.float32).to(device)
            mb_y = torch.tensor(mb_y, dtype=torch.float32).to(device)
            mb_y_cls = torch.argmax(mb_y, dim=1)

            optimizer.zero_grad()
            logits, x1_re, x2_re = model(mb_x1, mb_x2)

            ce_loss = loss_fn(logits, mb_y_cls)
            mse1 = torch.mean((x1_re - mb_x1f) ** 2)
            mse2 = torch.mean((x2_re - mb_x2f) ** 2)
            l2 = utils.l2_loss(model)
            cost = ce_loss + beta_reg * l2 + 1.0 * mse1 + 1.0 * mse2

            cost.backward()
            optimizer.step()

            epoch_loss += cost.item() / num_batches
            preds = torch.argmax(logits, dim=1)
            epoch_acc += (preds == mb_y_cls).float().mean().item() / num_batches

        scheduler.step()

        # ---------- 验证 ----------
        model.eval()
        with torch.no_grad():
            logits_dev, x1d, x2d = model(x1_test, x2_test)
            ce_dev = loss_fn(logits_dev, y_test_cls)
            mse1_dev = torch.mean((x1d - x1_test) ** 2)
            mse2_dev = torch.mean((x2d - x2_test) ** 2)
            l2_dev = utils.l2_loss(model)
            cost_dev = ce_dev + beta_reg * l2_dev + 1.0 * mse1_dev + 1.0 * mse2_dev

            preds_dev = torch.argmax(logits_dev, dim=1)
            acc_dev = (preds_dev == y_test_cls).float().mean().item()

        if print_cost and epoch % 50 == 0:
            print(f"epoch {epoch}: "
                  f"Train_loss: {epoch_loss:.4f}, Val_loss: {cost_dev.item():.4f}, "
                  f"Train_acc: {epoch_acc:.4f}, Val_acc: {acc_dev:.4f}")

        if epoch % 5 == 0:
            costs.append(epoch_loss)
            costs_dev.append(cost_dev.item())
            train_acc.append(epoch_acc)
            val_acc.append(acc_dev)

        # ---------- 画图 ----------
    plt.plot(costs, label='train')
    plt.plot(costs_dev, label='val')
    plt.ylabel('cost')
    plt.xlabel('epoch (/5)')
    plt.legend()
    plt.show()

    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val')
    plt.ylabel('accuracy')
    plt.xlabel('epoch (/5)')
    plt.legend()
    plt.show()

    # ---------- 返回 ----------
    # 提取参数到 dict（与 TF 版接口一致）
    state_dict = model.state_dict()
    parameters = {k: v.cpu().numpy() for k, v in state_dict.items()}

    # joint 特征 (转置后形状 [15, N]，与 TF 版相同)
    model.eval()
    with torch.no_grad():
        feature = model(x1_test, x2_test)[0].T.cpu().numpy()

    return parameters, val_acc, feature


HSI_TrSet = sio.loadmat('HSI_LiDAR_FC/HSI_TrSet.mat')
LiDAR_TrSet = sio.loadmat('HSI_LiDAR_FC/LiDAR_TrSet.mat')
HSI_TeSet = sio.loadmat('HSI_LiDAR_FC/HSI_TeSet.mat')
LiDAR_TeSet = sio.loadmat('HSI_LiDAR_FC/LiDAR_TeSet.mat')

TrLabel = sio.loadmat('HSI_LiDAR_FC/TrLabel.mat')
TeLabel = sio.loadmat('HSI_LiDAR_FC/TeLabel.mat')

HSI_TrSet = HSI_TrSet['HSI_TrSet']
LiDAR_TrSet = LiDAR_TrSet['LiDAR_TrSet']
HSI_TeSet = HSI_TeSet['HSI_TeSet']
LiDAR_TeSet = LiDAR_TeSet['LiDAR_TeSet']

TrLabel = TrLabel['TrLabel']
TeLabel = TeLabel['TeLabel']

Y_train = utils.convert_to_one_hot(TrLabel - 1, 15)
Y_test = utils.convert_to_one_hot(TeLabel - 1, 15)

Y_train = Y_train.T
Y_test = Y_test.T

parameters, val_acc, feature = train(HSI_TrSet, LiDAR_TrSet, HSI_TrSet, LiDAR_TrSet, HSI_TeSet, LiDAR_TeSet, Y_train,
                                     Y_test)
# data hasn't been pcaed, based on pixel
sio.savemat('feature.mat', {'feature': feature})
print("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
