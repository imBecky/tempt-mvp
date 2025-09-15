import torch
import os
from datetime import datetime
from torch import nn
from torch.optim.lr_scheduler import StepLR
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
import torch_utils as utils
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import warnings

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch.cpu.amp.autocast.*",
    module="torch.*"  # 可限定模块名
)


def log_args_and_time(args, epoch, train_loss, train_acc, val_acc,
                      val_correct, val_total, log_file='log.txt'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    record = (f"{timestamp} | "
              f"lr:{args.lr} | "
              f"seed:{args.seed} | "
              f"epoch {epoch}: | "
              f"Train_loss: {train_loss:.4f}, | "
              f"Train_acc: {train_acc * 100:.2f}%, Val_acc: {val_acc * 100:.4f}%"
              f"(correct={val_correct}, total={val_total})\n")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(record)


class HSIDataset(Dataset):
    def __init__(self, hsi, label):
        self.hsi = hsi  # numpy / torch.Tensor 都行，保持 CPU
        self.label = label

    def __len__(self):
        return self.hsi.shape[0]

    def __getitem__(self, idx):
        return torch.as_tensor(self.hsi[idx], dtype=torch.float32), \
               torch.as_tensor(self.label[idx], dtype=torch.long)


class GroupEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2048, 512, 1), nn.GroupNorm(8, 512), nn.ReLU(inplace=True),
            # nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 32, 1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            # nn.Conv1d(256, 64, 1), nn.ReLU(inplace=True)
        )

    def forward(self, HSI):
        h = self.enc(HSI)
        return h


class BandFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1, groups=32), nn.GroupNorm(8, 32), nn.ReLU(),
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
            nn.Linear(320, 512), nn.ReLU(),
            nn.Linear(512, 224 * 224 * 21)  # 与输入图像同分辨率
        )

    def forward(self, HSI):
        B = HSI.shape[0]
        h1 = checkpoint(self.group_encoder, HSI, use_reentrant=False)
        h2 = self.band_fuse(h1)
        output = self.seg_head(h2)
        output = output.reshape(B, 21, 224, 224)
        return output


def train_hsi(HSI_train, HSI_test, y_train, y_test, args, print_cost=True):
    train_set = HSIDataset(HSI_train, y_train)
    test_set = HSIDataset(HSI_test, y_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    model = SegmentModel().to(CUDA0)
    print(args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    # scaler = GradScaler('cuda')  # 自动混合精度训练

    costs, costs_val, train_acc_list, val_acc_list = [], [], [], []

    # ---------------- 训练 ----------------
    for epoch in range(args.epoch + 1):
        model.train()
        epoch_loss = epoch_acc = 0.0
        num_batches = len(train_loader)

        for x, y in train_loader:
            x, y = x.to(CUDA0, non_blocking=True), y.to(CUDA0, non_blocking=True, dtype=torch.long)
            optimizer.zero_grad()
            # with autocast('cuda'):  # 自动混合精度训练
            output = model(x).float()
            train_loss = loss_fn(output, y)
            # l2 = sum(p.pow(2).sum() for name, p in model.named_parameters() if 'weight' in name)
            loss = train_loss
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() / num_batches
            preds = torch.argmax(output, dim=1)
            epoch_acc += (preds == y).float().mean().item() / num_batches
        scheduler.step()

        # ---------------- 验证（mini-batch） ----------------
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(CUDA0, non_blocking=True), y.to(CUDA0, non_blocking=True, dtype=torch.long)
                # with autocast('cuda'):
                output = model(x).float()
                preds = torch.argmax(output, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.numel()
        val_acc = val_correct / val_total

        if print_cost:
            print(f"epoch {epoch}: | "
                  f"Train_loss: {epoch_loss:.4f},  | "
                  f"Train_acc: {epoch_acc * 100:.2f}%,  |  Val_acc: {val_acc * 100:.4f}%  | "
                  f"  (correct={val_correct}, total={val_total})")
            log_args_and_time(args, epoch, epoch_loss, epoch_acc, val_acc,
                              val_correct, val_total)
        torch.cuda.empty_cache()

    # ---------- 返回 ----------
    # 提取参数到 dict（与 TF 版接口一致）
    state_dict = model.state_dict()
    parameters = {k: v.cpu().numpy() for k, v in state_dict.items()}
    return parameters, val_acc_list  # 统一返回列表
