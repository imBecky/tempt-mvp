import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Fusion import MultiModalSeg, MultiModalDataset, train_rgb_lidar_hsi
from math import sqrt
import copy
CUDA0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# 扩散模型核心组件 (适配现有模型)
# =====================


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, cond_dim=2048, embed_dim=512):
        super().__init__()
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 条件处理
        self.cond_proj = nn.Linear(cond_dim, embed_dim)

        # 主干网络
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.res_blocks = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, input_dim)
        )

    def forward(self, x, time, cond):
        # 时间嵌入 (B,1) -> (B,embed_dim)
        t_emb = self.time_embed(time.unsqueeze(1))

        # 条件投影 (B,cond_dim) -> (B,embed_dim)
        cond_emb = self.cond_proj(cond)

        # 输入投影
        h = self.input_proj(x)

        # 融合时间+条件
        h = h + t_emb + cond_emb

        # 残差处理
        return self.res_blocks(h)


# =====================
# 多模态扩散模型 (适配HSI结构)
# =====================
class MultimodalDiffusionAugmenter:
    def __init__(self, seg_model, timesteps=500):
        self.timesteps = timesteps
        self.seg_model = seg_model.eval().to(CUDA0)

        # 初始化扩散模型
        self.rgb_diff = ConditionalDiffusionModel(2048).to(CUDA0)
        self.lidar_diff = ConditionalDiffusionModel(2048).to(CUDA0)
        self.hsi_diff = ConditionalDiffusionModel(2048).to(CUDA0)  # 每组光谱独立处理

        # 优化器
        params = list(self.rgb_diff.parameters()) + \
                 list(self.lidar_diff.parameters()) + \
                 list(self.hsi_diff.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=1e-4)

        # 噪声调度
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(CUDA0)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        """前向扩散过程"""
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].view(-1, 1)
        noisy_x = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return noisy_x, noise

    def diffusion_loss(self, pred_noise, true_noise):
        """基础扩散损失"""
        return F.mse_loss(pred_noise, true_noise)

    def semantic_consistency_loss(self, rgb, lidar, hsi, real_mask):
        """语义一致性损失 (使用预训练分割模型)"""
        with torch.no_grad():
            # 真实分割结果
            real_pred = self.seg_model(rgb, lidar, hsi)
            real_seg = torch.argmax(real_pred, dim=1)

        # 生成特征的分割结果
        gen_pred = self.seg_model(rgb, lidar, hsi)

        # KL散度损失
        kl_loss = F.kl_div(
            F.log_softmax(gen_pred, dim=1),
            F.softmax(real_pred, dim=1),
            reduction='batchmean'
        )

        # 分割图一致性损失
        seg_loss = F.cross_entropy(gen_pred, real_seg)

        return kl_loss + seg_loss

    def spectral_correlation_loss(self, hsi):
        """HSI光谱相关性损失"""
        # 计算光谱通道间的相关性
        channels = hsi.shape[1]
        corr_loss = 0
        for i in range(channels):
            for j in range(i + 1, channels):
                # 余弦相似度
                cos_sim = F.cosine_similarity(hsi[:, i], hsi[:, j], dim=-1)
                # 鼓励相似度接近1 (保持光谱相关性)
                corr_loss += torch.mean((1 - cos_sim) ** 2)
        return corr_loss / (channels * (channels - 1) / 2)

    def train_step(self, batch):
        # 解包数据
        (rgb, lidar, hsi), mask = batch
        rgb, lidar, hsi, mask = [t.to(CUDA0) for t in [rgb, lidar, hsi, mask]]
        batch_size = rgb.shape[0]

        # 采样时间步
        t = torch.randint(0, self.timesteps, (batch_size,)).to(CUDA0)

        # === 提取融合特征作为条件 ===
        with torch.no_grad():
            # 压缩HSI
            hsi_compressed = self.seg_model.hsi_sq(hsi)
            # 获取融合特征
            cond = self.seg_model.fusion(rgb, lidar, hsi_compressed)

        # === RGB扩散 ===
        rgb_noisy, rgb_noise = self.add_noise(rgb, t)
        rgb_pred_noise = self.rgb_diff(rgb_noisy, t, cond)
        rgb_loss = self.diffusion_loss(rgb_pred_noise, rgb_noise)

        # === LiDAR扩散 ===
        lidar_noisy, lidar_noise = self.add_noise(lidar, t)
        lidar_pred_noise = self.lidar_diff(lidar_noisy, t, cond)
        lidar_loss = self.diffusion_loss(lidar_pred_noise, lidar_noise)

        # === HSI扩散 (分组处理) ===
        hsi_loss = 0
        spectral_loss = 0
        for i in range(hsi.shape[2]):  # 遍历10个光谱组
            hsi_group = hsi[:, :, i]
            hsi_noisy, hsi_noise = self.add_noise(hsi_group, t)
            hsi_pred_noise = self.hsi_diff(hsi_noisy, t, cond)
            hsi_loss += self.diffusion_loss(hsi_pred_noise, hsi_noise)

            # 计算光谱相关性损失
            spectral_loss += self.spectral_correlation_loss(
                hsi_noisy.unsqueeze(1)  # 添加通道维度
            )
        hsi_loss /= hsi.shape[2]
        spectral_loss /= hsi.shape[2]

        # === 语义一致性损失 ===
        semantic_loss = self.semantic_consistency_loss(
            rgb_noisy, lidar_noisy, hsi, mask
        )

        # 总损失
        total_loss = (
                rgb_loss + lidar_loss + hsi_loss +
                0.5 * semantic_loss + 0.1 * spectral_loss
        )

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def generate_samples(self, cond, n_samples=1):
        """生成新样本"""
        self.rgb_diff.eval()
        self.lidar_diff.eval()
        self.hsi_diff.eval()

        with torch.no_grad():
            # 生成RGB
            rgb_noise = torch.randn(n_samples, 2048).to(CUDA0)
            rgb_gen = self._sample_loop(self.rgb_diff, rgb_noise, cond)

            # 生成LiDAR
            lidar_noise = torch.randn(n_samples, 2048).to(CUDA0)
            lidar_gen = self._sample_loop(self.lidar_diff, lidar_noise, cond)

            # 生成HSI (每组独立生成)
            hsi_gen = []
            for _ in range(10):  # 10个光谱组
                hsi_noise = torch.randn(n_samples, 2048).to(CUDA0)
                hsi_group = self._sample_loop(self.hsi_diff, hsi_noise, cond)
                hsi_gen.append(hsi_group.unsqueeze(-1))
            hsi_gen = torch.cat(hsi_gen, dim=-1)

        return rgb_gen, lidar_gen, hsi_gen

    def _sample_loop(self, model, x, cond):
        """DDIM采样"""
        x_t = x
        for t in reversed(range(self.timesteps)):
            # 创建时间张量
            t_tensor = torch.full((x.shape[0],), t, dtype=torch.float).to(CUDA0)

            # 预测噪声
            pred_noise = model(x_t, t_tensor, cond)

            # 计算alpha参数
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            alpha_bar_prev = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0)

            # 计算预测的原始样本
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

            # 计算方向
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise

            # 更新x
            x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        return x_t


# =====================
# 增强数据生成流程
# =====================
def generate_augmented_data(seg_model, train_loader, num_augment):
    """生成增强数据"""
    # 初始化扩散模型
    diffuser = MultimodalDiffusionAugmenter(seg_model)

    # 收集真实融合特征作为条件
    all_cond = []
    with torch.no_grad():
        for batch in train_loader:
            (rgb, lidar, hsi), _ = batch
            rgb, lidar, hsi = [t.to(CUDA0) for t in [rgb, lidar, hsi]]

            # 提取融合特征
            hsi_compressed = seg_model.hsi_sq(hsi)
            cond = seg_model.fusion(rgb, lidar, hsi_compressed)
            all_cond.append(cond.cpu())

    all_cond = torch.cat(all_cond, dim=0)

    # 生成新样本
    augmented_rgb, augmented_lidar, augmented_hsi = [], [], []

    for _ in range(num_augment):
        # 随机选择条件
        idx = torch.randint(0, len(all_cond), (1,))
        cond = all_cond[idx].to(CUDA0)

        # 生成新样本
        rgb_gen, lidar_gen, hsi_gen = diffuser.generate_samples(cond)

        augmented_rgb.append(rgb_gen.cpu())
        augmented_lidar.append(lidar_gen.cpu())
        augmented_hsi.append(hsi_gen.cpu())

    # 合并结果
    augmented_rgb = torch.cat(augmented_rgb, dim=0).numpy()
    augmented_lidar = torch.cat(augmented_lidar, dim=0).numpy()
    augmented_hsi = torch.cat(augmented_hsi, dim=0).numpy()

    return augmented_rgb, augmented_lidar, augmented_hsi


# =====================
# 集成到训练流程
# =====================
def train_with_augmentation(args, RGB_train, LiDAR_train, HSI_train, y_train,
                            RGB_test, LiDAR_test, HSI_test, y_test):

    # 初始训练分割模型
    print("Training initial segmentation model...")
    seg_model = MultiModalSeg(n_class=args.n_class).to(CUDA0)
    train_set = MultiModalDataset(RGB_train, LiDAR_train, HSI_train, y_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # 初始训练 (少量epoch)
    optimizer = torch.optim.AdamW(seg_model.parameters(), lr=args.lr)
    for epoch in range(3):  # 初始训练3个epoch
        for batch in train_loader:
            (rgb, lidar, hsi), mask = batch
            rgb, lidar, hsi, mask = [t.to(CUDA0) for t in [rgb, lidar, hsi, mask]]

            optimizer.zero_grad()
            out = seg_model(rgb, lidar, hsi)
            loss = F.cross_entropy(out, mask)
            loss.backward()
            optimizer.step()

    # 生成增强数据
    print("Generating augmented data...")
    aug_rgb, aug_lidar, aug_hsi = generate_augmented_data(
        seg_model, train_loader, num_augment=len(RGB_train)
    )

    # 合并原始数据和增强数据
    combined_rgb = np.vstack([RGB_train, aug_rgb])
    combined_lidar = np.vstack([LiDAR_train, aug_lidar])
    combined_hsi = np.concatenate([HSI_train, aug_hsi], axis=0)
    combined_y = np.concatenate([y_train, y_train], axis=0)  # 使用相同标签

    # 使用增强数据重新训练
    print("Training with augmented data...")
    final_model, val_acc = train_rgb_lidar_hsi(
        combined_rgb, combined_lidar, combined_hsi, combined_y,
        RGB_test, LiDAR_test, HSI_test, y_test,
        args
    )

    return final_model, val_acc
