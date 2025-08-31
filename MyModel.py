import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import matplotlib
import matplotlib.pyplot as plt
import torch_utils as utils
matplotlib.use('TkAgg')

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_enc = nn.Sequential(
            nn.Linear(144, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        self.rgb_dec = nn.Sequential(
            nn.Linear(128, 64), nn.Sigmoid(),
            nn.Linear(64, 32), nn.Sigmoid(),
            nn.Linear(32, 16), nn.Sigmoid(),
            nn.Linear(16, 144), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(144, 224), nn.Sigmoid(),
            nn.Linear(224, 224*224), nn.Sigmoid()
        )

    def forward(self, rgb):
        hidden_rgb = self.rgb_enc(rgb)
        output = self.rgb_dec(hidden_rgb)
        return output


def train(rgb_train, rgb_test, y_train, y_test,
          lr_base=1e-3, epochs=150, beta_reg=1e-3, batch_size=64, print_cost=True):
    rgb_train = torch.tensor(rgb_train).to(CUDA0)
    rgb_test = torch.tensor(rgb_test).to(CUDA0)
    y_train = torch.tensor(y_train).to(CUDA0)
    y_test = torch.tensor(y_test).to(CUDA0)

    model = MyModel().to(CUDA0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_base)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    y_train_cls = torch.argmax(y_train, dim=1)  # one_hot → cls idx
    y_test_cls = torch.argmax(y_test, dim=1)

    # recording
    costs, costs_dev = [], []  # what's the meaning of costs_dev
    train_acc, val_acc = [], []

    seed = 1
    m = rgb_train.size(0)

    # train
    for epoch in range(epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        seed += 1  # make sure every epoch is shuffled
        minibatches = utils.random_mini_batches(rgb_train.cpu().numpy(),
                                                y_train.cpu().numpy(),
                                                batch_size, seed)
        num_batches = len(minibatches)

        for (mb_x, mb_y) in minibatches:
            mb_x = torch.tensor(mb_x, dtype=torch.float32).to(CUDA0)
            mb_y = torch.tensor(mb_y, dtype=torch.float32).to(CUDA0)
            mb_y_cls = torch.argmax(mb_y, dim=1)       # one_hot

            optimizer.zero_grad()
            rgb_train_output = model(mb_x)

            ce_loss = loss_fn(rgb_train_output, mb_y_cls)
            l2 = utils.l2_loss(model)
            cost = ce_loss + beta_reg * l2

            cost.backward()
            optimizer.step()

            epoch_loss += cost.item() / num_batches
            preds = torch.argmax(rgb_train_output, dim=1)
            epoch_acc += (preds == mb_y_cls).float().mean().item() / num_batches

        scheduler.step()

        # ---------- 验证 ----------
        model.eval()
        with torch.no_grad():
            rgb_test_output = model(rgb_test)
            ce_dev = loss_fn(rgb_test_output, y_test)
            l2_dev = utils.l2_loss(model)
            cost_dev = ce_dev + beta_reg * l2_dev

            preds_dev = torch.argmax(rgb_test_output, dim=1)
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

    return parameters, val_acc
