import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import matplotlib
import matplotlib.pyplot as plt
import torch_utils as utils

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True


class GroupEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2048, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            # nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 32, 1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            # nn.Conv1d(256, 64, 1), nn.ReLU(inplace=True)
        )

    def forward(self, HSI):
        h = self.enc(HSI)
        return h


class BandFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1, groups=32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 1)
        )

    def forward(self, z):
        z = self.fuse(z)  # 靠大小为3的kernel组间融合
        return z


class SegmentModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SegmentModel, self).__init__(*args, **kwargs)
        self.group_encoder = GroupEncoder()
        self.band_fuse = BandFusion()
        self.seg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, 1024), nn.ReLU(),
            nn.Linear(1024, 224 * 224 * 21)  # 与输入图像同分辨率
        )

    def forward(self, HSI):
        B, mid_dim, _ = HSI.shape
        h1 = self.group_encoder(HSI)
        h2 = self.band_fuse(h1)
        output = self.seg_head(h2)
        output = output.reshape(B, 21, 224, 224)
        return output


def train_hsi(HSI_train, HSI_test, y_train, y_test, args,
              beta_reg=1e-3, print_cost=True):
    HSI_train = torch.tensor(HSI_train)
    HSI_test = torch.tensor(HSI_test)
    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()

    model = SegmentModel().to(CUDA0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    # recording
    costs, costs_dev = [], []  # what's the meaning of costs_dev
    train_acc, val_acc = [], []

    seed = 1
    m = HSI_train.size(0)

    # train
    for epoch in range(args.epoch + 1):
        model.train()
        torch.cuda.memory_summary()
        epoch_loss = 0.0
        epoch_acc = 0.0
        seed += 1  # make sure every epoch is shuffled
        minibatches = utils.random_mini_batches(HSI_train.cpu().numpy(),
                                                y_train.cpu().numpy(),
                                                args.batch_size, seed)
        num_batches = len(minibatches)

        for (mb_x, mb_y) in minibatches:
            mb_x = torch.tensor(mb_x, dtype=torch.float32).to(CUDA0)
            mb_y = torch.tensor(mb_y, dtype=torch.long).to(CUDA0)

            optimizer.zero_grad()
            train_output = model(mb_x)

            ce_loss = loss_fn(train_output, mb_y)
            l2 = utils.l2_loss(model)
            cost = ce_loss + beta_reg * l2

            cost.backward()
            optimizer.step()

            epoch_loss += cost.item() / num_batches
            preds = torch.argmax(train_output, dim=1)
            epoch_acc += (preds == mb_y).float().mean().item() / num_batches

        scheduler.step()

        # ---------- 验证 ----------
        model.eval()
        with torch.no_grad():
            rgb_test_output = model(HSI_test)
            ce_dev = loss_fn(rgb_test_output, y_test)
            l2_dev = utils.l2_loss(model)
            cost_dev = ce_dev + beta_reg * l2_dev

            preds_dev = torch.argmax(rgb_test_output, dim=1)
            acc_dev = (preds_dev == y_test).float().mean().item()

        if print_cost:
            print(f"epoch {epoch}: "
                  f"Train_loss: {epoch_loss:.4f}, Val_loss: {cost_dev.item():.4f}, "
                  f"Train_acc: {epoch_acc:.4f}, Val_acc: {acc_dev:.4f}")

        if epoch % 5 == 0:
            costs.append(epoch_loss)
            costs_dev.append(cost_dev.item())
            train_acc.append(epoch_acc)
            val_acc.append(acc_dev)

        # ---------- 画图 ----------
    # plt.plot(costs, label='train')
    # plt.plot(costs_dev, label='val')
    # plt.ylabel('cost')
    # plt.xlabel('epoch (/5)')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(train_acc, label='train')
    # plt.plot(val_acc, label='val')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch (/5)')
    # plt.legend()
    # plt.show()

    # ---------- 返回 ----------
    # 提取参数到 dict（与 TF 版接口一致）
    state_dict = model.state_dict()
    parameters = {k: v.cpu().numpy() for k, v in state_dict.items()}

    return parameters, val_acc
