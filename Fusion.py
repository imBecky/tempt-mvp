import torch
import os
from datetime import datetime
from torch import nn
import torch.nn.functional as F
import warnings
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

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


# ---------- 1. HSI 通道压缩 ----------
class HSISqueeze(nn.Module):
    def __init__(self, in_ch=10, out_ch=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, 1, 1) / (in_ch ** 0.5))
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):  # (B,2048,10)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (B,2048,10,1,1)
        w = self.weight.unsqueeze(0)  # (1,1,10,1,1)
        output = torch.sum(x * w, dim=2) + self.bias  # (B,2048,1,1) → squeeze→(B,2048)
        output = output.squeeze(-1).squeeze(-1)
        return output


# ---------- 2. 跨模态注意力融合 ----------
class CrossAttnFusion(nn.Module):
    def __init__(self, d=2048):
        super().__init__()
        self.scale = d ** -0.5
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)

    def forward(self, rgb, lidar, hsi):  # 均为 (B,2048)
        q = self.w_q(rgb).unsqueeze(1)  # (B,1,d)
        k = self.w_k(torch.stack([lidar, hsi], dim=1))  # (B,2,d)
        v = self.w_v(torch.stack([lidar, hsi], dim=1))
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)  # (B,1,2)
        fused = (attn @ v).squeeze(1) + rgb  # 残差
        return fused  # (B,2048)


# ---------- 3. 轻量跳跃解码器 ----------
class TinyDecoder(nn.Module):
    def __init__(self, in_dim=2048, n_class=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, 512, 1)        # 1×1
        self.up1   = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 2×
        self.up2   = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 4×
        self.up3   = nn.ConvTranspose2d(128,  64, 4, 2, 1)  # 8×
        self.up4   = nn.ConvTranspose2d(64 ,  32, 4, 2, 1)  # 16×
        self.up5   = nn.ConvTranspose2d(32 , n_class, 4, 2, 1)  # 32×32
        # 新增：一次性放大 7 倍到 224×224
        self.final = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):               # (B,2048)
        f = x.view(x.size(0), -1, 1, 1)
        f = self.conv1(f)
        f = self.up1(f)
        f = self.up2(f)
        f = self.up3(f)
        f = self.up4(f)
        f = self.up5(f)                 # (B,K,32,32)
        out = self.final(f)             # (B,K,224,224)
        return out


# ---------- 4. 整体网络 ----------
class MultiModalSeg(nn.Module):
    def __init__(self, n_class=8):
        super().__init__()
        self.hsi_sq = HSISqueeze()
        self.fusion = CrossAttnFusion()
        self.decoder = TinyDecoder(n_class=n_class)

    def forward(self, rgb, lidar, hsi):  # 均为 (B,2048) 或 (B,2048,10)
        hsi = self.hsi_sq(hsi)  # (B,2048)
        fused = self.fusion(rgb, lidar, hsi)  # (B,2048)
        return self.decoder(fused)  # (B,K,224,224)


class MultiModalDataset(Dataset):
    def __init__(self, rgb, lidar, hsi, mask):
        self.rgb = torch.as_tensor(rgb)  # (N,2048)
        self.lidar = torch.as_tensor(lidar)  # (N,2048)
        self.hsi = torch.as_tensor(hsi)  # (N,2048,10)
        self.mask = torch.as_tensor(mask, dtype=torch.long)  # (N,224,224)

    def __len__(self): return self.rgb.shape[0]

    def __getitem__(self, idx):
        return (self.rgb[idx], self.lidar[idx], self.hsi[idx]), self.mask[idx]


def train_rgb_lidar_hsi(RGB_train, LiDAR_train, HSI_train, y_train,
                        RGB_test, LiDAR_test, HSI_test, y_test,
                        args, print_cost=True):
    """
    args 必须包含：batch_size, lr, epoch, grad_accum=1, use_amp=True
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 数据 ----
    train_set = MultiModalDataset(RGB_train, LiDAR_train, HSI_train, y_train)
    test_set = MultiModalDataset(RGB_test, LiDAR_test, HSI_test, y_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ---- 模型 ----
    model = MultiModalSeg(n_class=args.n_class).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=args.use_amp)

    # ---- 日志 ----
    costs, train_acc_list, val_acc_list = [], [], []

    # ---------------- 训练 ----------------
    for epoch in range(args.epoch + 1):
        model.train()
        epoch_loss = epoch_acc = 0.0
        num_batches = len(train_loader)
        optimizer.zero_grad()
        for batch_idx, (x, y) in enumerate(train_loader):
            rgb, lidar, hsi = [t.to(device, non_blocking=True) for t in x]
            y = y.to(device, non_blocking=True)

            with autocast('cuda', enabled=args.use_amp):
                out = model(rgb, lidar, hsi)  # (B,K,224,224)
                loss = loss_fn(out, y) / args.grad_accum
            scaler.scale(loss).backward()
            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum / num_batches
            preds = torch.argmax(out, dim=1)
            epoch_acc += (preds == y).float().mean().item() / num_batches
        scheduler.step()

        # ---------------- 验证 ----------------
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                rgb, lidar, hsi = [t.to(device, non_blocking=True) for t in x]
                y = y.to(device, non_blocking=True)
                with autocast('cuda', enabled=args.use_amp):
                    out = model(rgb, lidar, hsi)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.numel()
        val_acc = val_correct / val_total
        train_acc_list.append(epoch_acc)
        val_acc_list.append(val_acc)
        costs.append(epoch_loss)

        if print_cost:
            print(f"Epoch {epoch:03d} | "
                  f"loss: {epoch_loss:.4f} | "
                  f"train acc: {epoch_acc * 100:.2f}% | "
                  f"val acc: {val_acc * 100:.2f}%")
            log_args_and_time(args, epoch, epoch_loss, epoch_acc, val_acc,
                              val_correct, val_total, log_file='./Trial1.txt')
        torch.cuda.empty_cache()

    # ---------- 返回 ----------
    return {k: v.cpu().numpy() for k, v in model.state_dict().items()}, val_acc_list
