import torch, os, warnings
from datetime import datetime
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cpu.amp.autocast.*")


# ---------------- 日志 ----------------
def log_args_and_time(args, epoch, train_loss, train_acc, val_acc,
                      val_correct, val_total, log_file='Trial2.txt'):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rec = (f"{ts} | lr:{args.lr} | seed:{args.seed} | epoch {epoch}: | "
           f"Train_loss:{train_loss:.4f} | Train_acc:{train_acc*100:.2f}% | "
           f"Val_acc:{val_acc*100:.2f}% (correct={val_correct},total={val_total})\n")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(rec)


# ---------------- 1. HSI 压缩 ----------------
class HSISqueeze(nn.Module):
    """
    144 → 64 的 1×1 映射，可当作通道降维
    """
    def __init__(self, in_ch=144, out_ch=64):
        super().__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):          # (B, 144)
        return self.fc(x)          # (B, 64)


# ---------------- 2. 跨模态注意力 ----------------
class CrossAttnFusion(nn.Module):
    """
    把 AP(21) 与 HSI(64) 做 cross-attn，得到 64 维融合向量
    """
    def __init__(self, d=64):
        super().__init__()
        self.scale = d ** -0.5
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)

    def forward(self, hsi, ap):
        # hsi:(B,64)  ap:(B,21)→先升到 64
        ap = F.pad(ap, (0, 64-21))              # 简单 zero-pad 到 64
        q = self.w_q(hsi).unsqueeze(1)          # (B,1,64)
        k = self.w_k(ap).unsqueeze(1)           # (B,1,64)
        v = self.w_v(ap).unsqueeze(1)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)  # (B,1,1)
        fused = (attn @ v).squeeze(1) + hsi     # 残差
        return fused                            # (B,64)


# ---------------- 3. 分类头 ----------------
class PixelClassifier(nn.Module):
    def __init__(self, in_dim=64, n_class=15):  # ← 改成15
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_class)  # ← 输出15维
        )

    def forward(self, x):
        return self.net(x)


# ---------------- 4. 整体网络 ----------------
class MultiModalPixel(nn.Module):
    def __init__(self, n_class=15):
        super().__init__()
        self.hsi_sq = HSISqueeze(144, 64)
        self.fusion = CrossAttnFusion(64)
        self.head = PixelClassifier(64, n_class)

    def forward(self, hsi, ap):
        hsi = self.hsi_sq(hsi)      # (B,64)
        fused = self.fusion(hsi, ap)
        return self.head(fused)     # (B,2)


# ---------------- 5. 数据集 ----------------
class PixelDataset(Dataset):
    def __init__(self, hsi, ap, mask):
        self.hsi = torch.as_tensor(hsi, dtype=torch.float32)   # (N,144)
        self.ap = torch.as_tensor(ap, dtype=torch.float32)     # (N,21)
        self.mask = torch.as_tensor(mask, dtype=torch.long)    # (N,)

    def __len__(self): return self.mask.shape[0]

    def __getitem__(self, idx):
        return (self.hsi[idx], self.ap[idx]), (self.mask[idx] - 1).long()


# ---------------- 6. 训练函数 ----------------
def train_hsi_lidar(hsi_train, ap_train, y_train,
                    hsi_test, ap_test, y_test, args, print_cost=True):
    """
    args 需含：batch_size, lr, epoch, grad_accum=1, use_amp=True, n_class=2, log_file
    """
    train_set = PixelDataset(hsi_train, ap_train, y_train)
    test_set = PixelDataset(hsi_test, ap_test, y_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    model = MultiModalPixel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=args.use_amp)

    costs, train_acc_list, val_acc_list = [], [], []

    for epoch in range(args.epoch + 1):
        # ---- train ----
        model.train()
        epoch_loss = epoch_acc = 0.
        num_batches = len(train_loader)
        optimizer.zero_grad()
        for batch_idx, (x, y) in enumerate(train_loader):
            hsi, ap = [t.to(device, non_blocking=True) for t in x]
            y = y.long().squeeze()
            y = y.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=args.use_amp):
                out = model(hsi, ap)                # (B,2)
                loss = loss_fn(out, y) / args.grad_accum
            scaler.scale(loss).backward()
            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum / num_batches
            preds = torch.argmax(out, dim=1)
            epoch_acc += (preds == y).float().mean().item() / num_batches
        scheduler.step()

        # ---- eval ----
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                hsi, ap = [t.to(device, non_blocking=True) for t in x]
                y = y.long().squeeze()
                y = y.to(device, non_blocking=True)
                with autocast(device_type='cuda', enabled=args.use_amp):
                    out = model(hsi, ap)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.numel()
        val_acc = val_correct / val_total
        train_acc_list.append(epoch_acc)
        val_acc_list.append(val_acc)
        costs.append(epoch_loss)

        if print_cost:
            print(f"Epoch {epoch:03d} | loss:{epoch_loss:.4f} | "
                  f"train acc:{epoch_acc*100:.2f}% | val acc:{val_acc*100:.2f}%")
            log_args_and_time(args, epoch, epoch_loss, epoch_acc, val_acc,
                              val_correct, val_total, log_file=args.log_file)
        torch.cuda.empty_cache()

    return {k: v.cpu().numpy() for k, v in model.state_dict().items()}, val_acc_list