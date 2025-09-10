import hashlib
import sys
import numpy as np
import torch
import os
from torch import nn
from torch.optim.lr_scheduler import StepLR
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
import torch_utils as utils
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- 原 Dataset / Model 定义 ----------------
class HSIDataset(Dataset):
    def __init__(self, hsi, label):
        self.hsi, self.label = hsi, label

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
            nn.Conv1d(512, 32, 1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
        )

    def forward(self, HSI):
        return self.enc(HSI)


class BandFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1, groups=32), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv1d(32, 32, 1)
        )

    def forward(self, z):
        return self.fuse(z)


class SegmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_encoder = GroupEncoder()
        self.band_fuse = BandFusion()
        self.gap = nn.AdaptiveAvgPool1d(1)  # 峰值激活减半
        self.seg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 512), nn.ReLU(),
            nn.Linear(512, 224 * 224 * 21)  # 21 为类别数
        )

    def forward(self, HSI):
        B, _, _ = HSI.shape
        h1 = torch.utils.checkpoint.checkpoint(self.group_encoder, HSI, use_reentrant=False)
        h2 = self.band_fuse(h1)
        h2 = self.gap(h2)  # [B,32,1]
        out = self.seg_head(h2)
        return out.reshape(B, 21, 224, 224)


# ---------------- 监测工具 ----------------
def _cprint(txt, flag=None):
    """彩色醒目打印"""
    if flag == 'warn':
        print("\033[93m" + txt + "\033[00m", file=sys.stderr)
    elif flag == 'err':
        print("\033[91m" + txt + "\033[00m", file=sys.stderr)
    else:
        print(txt)


def _quick_stats(loader, name):
    labels = []
    for _, y in loader:
        labels.append(y.numpy())
    labels = np.concatenate(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    _cprint(f"[{name}] 类别分布: {dict(zip(map(int, uniq), map(int, counts)))}")
    return uniq, counts


def _output_variance_check(model, loader):
    model.eval()
    with torch.no_grad():
        x0, _ = next(iter(loader))
        x0 = x0.to(CUDA0, non_blocking=True)
        out = model(x0)  # [B,21,H,W]
        var = out.var(dim=1).mean().item()
    _cprint(f"[退化检查] 验证 batch 输出通道方差 = {var:.6f} "
            f"{'<0.01 疑常数' if var < 0.01 else '≥0.01 正常'}",
            flag='warn' if var < 0.01 else None)
    return var


def _hash_first_sample(train_loader, test_loader):
    x1, y1 = next(iter(train_loader))
    x2, y2 = next(iter(test_loader))
    h1 = hashlib.md5(x1[0].cpu().numpy().tobytes()).hexdigest()[:8]
    h2 = hashlib.md5(x2[0].cpu().numpy().tobytes()).hexdigest()[:8]
    _cprint(f"[数据泄漏] 训练首样本哈希 = {h1}，测试首样本哈希 = {h2}")
    return h1 == h2


def _channel_match(model_out_ch, loader):
    x, y = next(iter(loader))
    max_label = int(y.max().item())
    _cprint(f"[通道匹配] 模型输出通道 = {model_out_ch}，数据最大标签 = {max_label}")
    if model_out_ch <= max_label:
        _cprint("❌  输出通道数 ≤ 最大标签，CrossEntropy 会静默掩码导致假高 acc！", flag='err')
        return False
    return True


# ---------------- 主训练函数 ----------------
def train_hsi_monitor(HSI_train, HSI_test, y_train, y_test, args,
                      beta_reg=1e-3, print_cost=True):
    train_set = HSIDataset(HSI_train, y_train)
    test_set = HSIDataset(HSI_test, y_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # ============ 首次 epoch 前监测 ============
    print("\n========== 监测报告（epoch -1） ==========")
    train_uniq, train_cnt = _quick_stats(train_loader, "Train")
    test_uniq, test_cnt = _quick_stats(test_loader, "Test")
    if len(test_uniq) == 1:
        _cprint("⚠️  测试集仅 1 个类别，acc 将恒为 1.0！", flag='warn')
    leak = _hash_first_sample(train_loader, test_loader)
    if leak:
        _cprint("⚠️  训练/测试首样本哈希相同，可能存在数据泄漏！", flag='warn')
    ch_ok = _channel_match(21, test_loader)
    var_ok = _output_variance_check(SegmentModel().to(CUDA0), test_loader)
    print("==========================================\n")

    model = SegmentModel().to(CUDA0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()

    costs, costs_val, train_acc_list, val_acc_list = [], [], [], []

    # ---------------- 训练 ----------------
    for epoch in range(args.epoch + 1):
        model.train()
        epoch_loss = epoch_acc = 0.0
        num_batches = len(train_loader)

        for x, y in train_loader:
            x, y = x.to(CUDA0, non_blocking=True), y.to(CUDA0, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                out = model(x)
                ce = loss_fn(out, y)
                l2 = sum(p.pow(2).sum() for p in model.parameters() if 'weight' in p.name)
                loss = ce + beta_reg * l2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() / num_batches
            preds = torch.argmax(out, dim=1)
            epoch_acc += (preds == y).float().mean().item() / num_batches
        scheduler.step()

        # ---------------- 验证（mini-batch） ----------------
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(CUDA0, non_blocking=True), y.to(CUDA0, non_blocking=True)
                out = model(x)
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
        acc_dev = correct / total

        if print_cost:
            print(f"epoch {epoch}: "
                  f"Train_loss: {epoch_loss:.4f}, Val_loss: N/A, "
                  f"Train_acc: {epoch_acc:.4f}, Val_acc: {acc_dev:.4f}  (correct={correct}, total={total})")

        if epoch % 5 == 0:
            costs.append(epoch_loss)
            costs_val.append(0)  # 占位，保持接口
            train_acc_list.append(epoch_acc)
            val_acc_list.append(acc_dev)
        torch.cuda.empty_cache()

    state_dict = model.state_dict()
    parameters = {k: v.cpu().numpy() for k, v in state_dict.items()}
    return parameters, val_acc_list  # 统一返回列表
