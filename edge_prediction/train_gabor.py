# -*- coding: utf-8 -*-
"""
Multi-scale Feature Frontend + U-Net Head + BCE
用于像素级结构/边缘概率图预测。

模型 (V5):
    image [B,1,H,W]
      -> 特征前端（三选一）
           - 'gabor_fixed'      : 固定 Gabor
           - 'gabor_learnable'  : Gabor 核可学习
           - 'cnn'              : 可学习 CNN
      -> [B, S*C, H, W] 全分辨率拼接特征
      -> U-Net Head
           stem (3x3)
           encoder stage 0 (H,   base)
             ↓ maxpool
           encoder stage 1 (H/2, base*2)
             ↓ maxpool
           encoder stage 2 (H/4, base*4)
             ↓ maxpool
           encoder stage 3 (H/8, base*8)
           decoder（3 次上采样 + skip concat）
      -> 1x1 head -> 1ch
      -> sigmoid
      -> probability

支持 resume 训练：
    - 每个 epoch 保存 last.pth（含 model / optimizer / epoch / best_loss / global_counter）
    - train_model(..., resume_from="runs_cnn/xxxx/last.pth")
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # 只使用第 0 块 GPU
import math
from pathlib import Path
from datetime import datetime
from typing import Sequence, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edge_prediction.dataloader.data_loader import BSDS_RCFLoader
from tqdm import tqdm

# ============================================================
# 1. 基础工具
# ============================================================

EPS = 1e-8


def build_ignore_mask(label: torch.Tensor, ignore_value: int = 2) -> torch.Tensor:
    if label.ndim == 3:
        label = label.unsqueeze(1)
    return (label != ignore_value).float()


def compute_rcf_weight(label: torch.Tensor,
                       ignore_value: int = 2,
                       neg_boost: float = 1.0
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    if label.ndim == 3:
        label = label.unsqueeze(1)
    label = label.float()
    valid_mask = (label != ignore_value).float()
    pos_mask = (label == 1).float() * valid_mask
    neg_mask = (label == 0).float() * valid_mask
    num_pos = pos_mask.sum() + EPS
    num_neg = neg_mask.sum() + EPS
    total = num_pos + num_neg
    w_pos = num_neg / total
    w_neg = (num_pos / total) * neg_boost
    weight = torch.zeros_like(label)
    weight[label == 1] = w_pos
    weight[label == 0] = w_neg
    weight = weight * valid_mask
    return weight, valid_mask


def bce_loss_probability(
    prediction: torch.Tensor,
    label: torch.Tensor,
    ignore_value: int = 2,
    balance: bool = True,
    neg_boost: float = 1.0,
    reduction: str = "sum",
) -> torch.Tensor:
    if label.ndim == 3:
        label = label.unsqueeze(1)
    label = label.float()
    if balance:
        weight, valid_mask = compute_rcf_weight(label, ignore_value, neg_boost)
    else:
        valid_mask = (label != ignore_value).float()
        weight = valid_mask
    target = torch.clamp(label, 0.0, 1.0)
    bce = F.binary_cross_entropy(
        prediction.clamp(EPS, 1.0 - EPS),
        target,
        reduction="none",
    )
    loss = bce * weight
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / (valid_mask.sum() + EPS)
    raise ValueError(f"Unknown reduction: {reduction}")


# ============================================================
# 2. Gabor Kernel 生成
# ============================================================

def make_gabor_quadrature_kernels(
    sigma: float,
    wavelength: float,
    orientations: Sequence[float],
    gamma: float = 0.5,
    kernel_size: Optional[int] = None,
) -> torch.Tensor:
    if kernel_size is None:
        kernel_size = int(math.ceil(6.0 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 7)
    half = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    kernels = []
    for theta in orientations:
        ct = math.cos(theta)
        st = math.sin(theta)
        x_theta = x * ct + y * st
        y_theta = -x * st + y * ct
        gaussian = torch.exp(
            -(x_theta ** 2 + (gamma ** 2) * y_theta ** 2) / (2.0 * sigma ** 2)
        )
        carrier = 2.0 * math.pi * x_theta / wavelength
        even = gaussian * torch.cos(carrier)
        odd = gaussian * torch.sin(carrier)
        even = even - even.mean()
        odd = odd - odd.mean()
        even = even / (torch.sqrt((even ** 2).sum()) + EPS)
        odd = odd / (torch.sqrt((odd ** 2).sum()) + EPS)
        kernels.append(even.unsqueeze(0).unsqueeze(0))
        kernels.append(odd.unsqueeze(0).unsqueeze(0))
    return torch.cat(kernels, dim=0)


# ============================================================
# 3. 三种前端（输出 [B, S*C, H, W]）
# ============================================================

class FixedGaborFrontend(nn.Module):
    def __init__(
        self,
        sigmas: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        wavelengths: Optional[Sequence[float]] = None,
        orientations: Sequence[float] = (
            math.radians(11.25), math.radians(33.75),
            math.radians(56.25), math.radians(78.75),
            math.radians(101.25), math.radians(123.75),
            math.radians(146.25), math.radians(168.75),
        ),
        gamma: float = 0.5,
    ):
        super().__init__()
        self.sigmas = tuple(sigmas)
        self.orientations = tuple(orientations)
        if wavelengths is None:
            wavelengths = tuple(2.5 * s for s in sigmas)
        if len(wavelengths) != len(sigmas):
            raise ValueError("wavelengths 必须与 sigmas 长度一致")
        self.wavelengths = tuple(wavelengths)
        self.gamma = gamma
        self.num_scales = len(self.sigmas)
        self.num_orientations = len(self.orientations)
        self.out_channels_per_scale = self.num_orientations

        for i, (sigma, wavelength) in enumerate(zip(self.sigmas, self.wavelengths)):
            kernel = make_gabor_quadrature_kernels(
                sigma=sigma, wavelength=wavelength,
                orientations=self.orientations, gamma=gamma,
            )
            self.register_buffer(f"kernel_{i}", kernel, persistent=True)

    @staticmethod
    def _to_4d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.ndim == 3:
            return x.unsqueeze(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_4d(x)
        features = []
        O = self.num_orientations
        for i in range(self.num_scales):
            kernel = getattr(self, f"kernel_{i}")
            padding = kernel.shape[-1] // 2
            response = F.conv2d(x, kernel, padding=padding)
            B, C, H, W = response.shape
            response = response.view(B, O, 2, H, W)
            even = response[:, :, 0]
            odd = response[:, :, 1]
            magnitude = torch.sqrt(even * even + odd * odd + EPS)
            features.append(magnitude)
        return torch.cat(features, dim=1)


class LearnableGaborFrontend(FixedGaborFrontend):
    """kernel_i 从 buffer 改为 Parameter，用 Gabor 公式初始化后可学习。"""

    def __init__(
        self,
        sigmas: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        wavelengths: Optional[Sequence[float]] = None,
        orientations: Sequence[float] = (
            math.radians(11.25), math.radians(33.75),
            math.radians(56.25), math.radians(78.75),
            math.radians(101.25), math.radians(123.75),
            math.radians(146.25), math.radians(168.75),
        ),
        gamma: float = 0.5,
    ):
        nn.Module.__init__(self)
        self.sigmas = tuple(sigmas)
        self.orientations = tuple(orientations)
        if wavelengths is None:
            wavelengths = tuple(2.5 * s for s in sigmas)
        if len(wavelengths) != len(sigmas):
            raise ValueError("wavelengths 必须与 sigmas 长度一致")
        self.wavelengths = tuple(wavelengths)
        self.gamma = gamma
        self.num_scales = len(self.sigmas)
        self.num_orientations = len(self.orientations)
        self.out_channels_per_scale = self.num_orientations

        for i, (sigma, wavelength) in enumerate(zip(self.sigmas, self.wavelengths)):
            kernel = make_gabor_quadrature_kernels(
                sigma=sigma, wavelength=wavelength,
                orientations=self.orientations, gamma=gamma,
            )
            self.register_parameter(f"kernel_{i}", nn.Parameter(kernel))


class LearnableCNNFrontend(nn.Module):
    """4 条并行 3x3 卷积分支，dilation 不同模拟多尺度。"""

    def __init__(
        self,
        num_scales: int = 4,
        channels_per_scale: int = 8,
        kernel_size: int = 3,
        dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32][:num_scales]
        if len(dilations) != num_scales:
            raise ValueError("dilations 长度需等于 num_scales")

        self.num_scales = num_scales
        self.channels_per_scale = channels_per_scale
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.num_orientations = channels_per_scale
        self.out_channels_per_scale = channels_per_scale

        pad = kernel_size // 2
        self.convs = nn.ModuleList([
            nn.Conv2d(1, channels_per_scale, kernel_size,
                      padding=d * pad, dilation=d, bias=False)
            for d in self.dilations
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(channels_per_scale) for _ in range(num_scales)
        ])
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _to_4d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.ndim == 3:
            return x.unsqueeze(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_4d(x)
        outs = [self.act(bn(conv(x))) for conv, bn in zip(self.convs, self.bns)]
        return torch.cat(outs, dim=1)


# ============================================================
# 4. U-Net Head（带池化）
# ============================================================

class ConvBlock(nn.Module):
    """Conv3x3 + BN + ReLU（可选 Dropout）"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetHead(nn.Module):
    """
    U-Net 风格深头：多 stage，池化下采样，逐层上采样 + skip concat。

    channel_mult=(1,2,4,8) 时：
        stage 0 : H,   base
        stage 1 : H/2, base*2
        stage 2 : H/4, base*4
        stage 3 : H/8, base*8
    然后 3 次上采样回 H，每次与对应的 skip 拼接。
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        blocks_per_stage: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        ch = [base_channels * m for m in channel_mult]
        self.channels = ch
        n = len(ch)

        self.stem = ConvBlock(in_channels, ch[0], dropout)

        # ---- 编码器 ----
        self.encoders = nn.ModuleList()
        prev_ch = ch[0]
        for i in range(n):
            blocks = [ConvBlock(prev_ch, ch[i], dropout)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(ConvBlock(ch[i], ch[i], dropout))
            self.encoders.append(nn.Sequential(*blocks))
            prev_ch = ch[i]

        # 池化：除最后一个 stage 外，每个 stage 后做一次 maxpool
        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(n - 1)])

        # ---- 解码器 ----
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(n - 1, 0, -1):
            self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            blocks = [ConvBlock(ch[i] + ch[i - 1], ch[i - 1], dropout)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(ConvBlock(ch[i - 1], ch[i - 1], dropout))
            self.decoders.append(nn.Sequential(*blocks))

        # ---- 输出头 ----
        self.head = nn.Conv2d(ch[0], 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = self.stem(x)
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            h = F.interpolate(
                h, size=skip.shape[-2:],
                mode='bilinear', align_corners=False,
            )
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return self.head(h)


# ============================================================
# 5. GaborLogistic（前端 + U-Net 深头）
# ============================================================

class GaborLogistic(nn.Module):
    """
    feature_type:
        'gabor_fixed' | 'gabor_learnable' | 'cnn'

    深头超参：
        base_channels    : 第一个 stage 的通道数
        channel_mult     : 各 stage 通道倍率（长度 = stage 数）
        blocks_per_stage : 每个 stage 堆叠的 Conv3x3 数量
    """

    def __init__(
        self,
        sigmas: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        wavelengths: Optional[Sequence[float]] = None,
        orientations: Sequence[float] = (
            math.pi / 16,
            3 * math.pi / 16,
            5 * math.pi / 16,
            7 * math.pi / 16,
            9 * math.pi / 16,
            11 * math.pi / 16,
            13 * math.pi / 16,
            15 * math.pi / 16,
        ),
        gamma: float = 0.5,
        feature_type: str = 'gabor_fixed',
        cnn_channels_per_scale: int = 8,
        # ---- U-Net 深头 ----
        base_channels: int = 32,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        blocks_per_stage: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_type = feature_type
        self.cnn_channels_per_scale = cnn_channels_per_scale

        # ---- 前端 ----
        if feature_type == 'gabor_fixed':
            self.gabor = FixedGaborFrontend(
                sigmas=sigmas, wavelengths=wavelengths,
                orientations=orientations, gamma=gamma,
            )
        elif feature_type == 'gabor_learnable':
            self.gabor = LearnableGaborFrontend(
                sigmas=sigmas, wavelengths=wavelengths,
                orientations=orientations, gamma=gamma,
            )
        elif feature_type == 'cnn':
            self.gabor = LearnableCNNFrontend(
                num_scales=len(sigmas),
                channels_per_scale=cnn_channels_per_scale,
            )
        else:
            raise ValueError(
                f"Unknown feature_type='{feature_type}'. "
                f"可选：'gabor_fixed' | 'gabor_learnable' | 'cnn'"
            )

        self.sigmas = getattr(self.gabor, "sigmas", tuple(sigmas))
        self.wavelengths = getattr(self.gabor, "wavelengths", None)
        self.orientations = getattr(self.gabor, "orientations", None)
        self.gamma = getattr(self.gabor, "gamma", None)

        self.num_scales = self.gabor.num_scales
        self.num_orientations = self.gabor.out_channels_per_scale

        # ---- U-Net 深头 ----
        in_channels = self.num_scales * self.gabor.out_channels_per_scale
        self.base_channels = base_channels
        self.channel_mult = tuple(channel_mult)
        self.blocks_per_stage = blocks_per_stage
        self.dropout = dropout

        self.unet = UNetHead(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mult=self.channel_mult,
            blocks_per_stage=blocks_per_stage,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.unet.head.weight, std=0.01)
        nn.init.zeros_(self.unet.head.bias)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.gabor(x)
        logits = self.unet(feats)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.sigmoid(self.forward_logits(x))
        return y

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)


class GaborLogistic_logit(GaborLogistic):
    """同结构，forward 不套 sigmoid。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(x)


# ============================================================
# 6. 可视化保存
# ============================================================

def save_visualization(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: Optional[torch.Tensor],
    step: int,
    save_dir: str,
    max_save: int = 4,
):
    if images.ndim == 3:
        images = images.unsqueeze(1)
    elif images.ndim == 4 and images.shape[1] != 1:
        images = images[:, 0:1, :, :]

    if predictions.ndim == 3:
        predictions = predictions.unsqueeze(1)

    if labels is not None and labels.ndim == 3:
        labels = labels.unsqueeze(1)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    B = images.shape[0]
    num_save = min(B, max_save)
    for i in range(num_save):
        img = images[i, 0].detach().cpu().numpy()
        pred = predictions[i, 0].detach().cpu().numpy()

        img_min, img_max = img.min(), img.max()
        img_norm = (img - img_min) / (img_max - img_min) if img_max > img_min else img
        Image.fromarray((img_norm * 255).astype(np.uint8), mode='L').save(
            save_path / f"img_{step}.png")

        pred_min, pred_max = pred.min(), pred.max()
        pred_norm = (pred - pred_min) / (pred_max - pred_min) if pred_max > pred_min else pred
        Image.fromarray((pred_norm * 255).astype(np.uint8), mode='L').save(
            save_path / f"pred_{step}.png")

        if labels is not None:
            lbl = labels[i, 0].detach().cpu().numpy()
            lbl_vis = np.where(lbl == 2, 128, lbl * 255).astype(np.uint8)
            Image.fromarray(lbl_vis, mode='L').save(save_path / f"label_{step}.png")


# ============================================================
# 7. 单 epoch / 验证
# ============================================================

def train_one_epoch(
    model, loader, optimizer, device,
    ignore_value: int = 2, balance: bool = True, neg_boost: float = 1.0,
    epoch: int = 0, total_epochs: int = 0,
    save_callback=None, save_interval: int = 500, global_counter=None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    desc = f"Train Epoch {epoch}/{total_epochs}" if total_epochs else "Training"
    pbar = tqdm(loader, desc=desc, leave=False)

    counter, last_saved = (global_counter if global_counter is not None else (0, 0))

    for data in pbar:
        if isinstance(data, (tuple, list)):
            image, label = (data if len(data) == 2 else (data[0], None))
        elif isinstance(data, dict):
            image = data.get('image') or data.get('img')
            label = data.get('label') or data.get('gt')
        elif isinstance(data, torch.Tensor):
            image, label = data, None
        else:
            raise TypeError(f"Unexpected data type: {type(data)}")

        image = image.to(device, non_blocking=True)
        if label is not None:
            label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(image)

        if label is not None:
            loss = bce_loss_probability(
                prediction, label,
                ignore_value=ignore_value, balance=balance,
                neg_boost=neg_boost, reduction="mean",
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        else:
            pbar.set_postfix({"info": "no label"})

        counter += image.size(0)
        if global_counter is not None:
            global_counter[0] = counter

        if save_callback is not None and label is not None:
            if counter - last_saved >= save_interval:
                save_callback(image, prediction, label, counter)
                last_saved = counter
                if global_counter is not None:
                    global_counter[1] = last_saved

    return total_loss / max(num_batches, 1) if num_batches else 0.0


@torch.no_grad()
def evaluate(
    model, loader, device,
    ignore_value: int = 2, balance: bool = True, neg_boost: float = 1.0,
    epoch: int = 0, total_epochs: int = 0,
) -> Tuple[float, bool]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    has_label = True
    desc = f"Val Epoch {epoch}/{total_epochs}" if total_epochs else "Validation"
    pbar = tqdm(loader, desc=desc, leave=False)

    for data in pbar:
        if isinstance(data, (tuple, list)):
            if len(data) == 1:
                image, label, has_label = data[0], None, False
            else:
                image, label = data[0], data[1]
        elif isinstance(data, dict):
            image = data.get('image') or data.get('img')
            label = data.get('label') or data.get('gt')
            if label is None:
                has_label = False
        elif isinstance(data, torch.Tensor):
            image, label, has_label = data, None, False
        else:
            raise TypeError(f"Unexpected data type: {type(data)}")

        image = image.to(device, non_blocking=True)
        prediction = model(image)

        if label is not None:
            label = label.to(device, non_blocking=True)
            loss = bce_loss_probability(
                prediction, label,
                ignore_value=ignore_value, balance=balance,
                neg_boost=neg_boost, reduction="mean",
            )
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        else:
            pbar.set_postfix({"info": "no label"})

    if num_batches == 0:
        print("Warning: Validation set has no labels. Returning loss=0.0.")
        return 0.0, has_label
    return total_loss / max(num_batches, 1), has_label


# ============================================================
# 8. 训练主函数（支持 resume）
# ============================================================

def train_model(
    train_root: str,
    val_root: Optional[str] = None,
    save_path: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    device: Optional[str] = None,
    balance: bool = True,
    neg_boost: float = 1.0,
    result_dir: Optional[str] = None,
    save_interval: int = 2000,
    base_dir: str = "runs_gabor",
    # ---- 前端 ----
    feature_type: str = 'gabor_fixed',
    cnn_channels_per_scale: int = 8,
    # ---- U-Net 深头 ----
    base_channels: int = 32,
    channel_mult: Sequence[int] = (1, 2, 4, 8),
    blocks_per_stage: int = 2,
    dropout: float = 0.0,
    # ---- resume ----
    resume_from: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ============================================================
    # 0. Resume 预读取：从 checkpoint 拿 run_dir 和模型超参
    # ============================================================
    resume_ckpt = None
    if resume_from is not None:
        print(f"Resuming from: {resume_from}")
        resume_ckpt = torch.load(resume_from, map_location="cpu")

        feature_type = resume_ckpt.get("feature_type", feature_type)
        cnn_channels_per_scale = resume_ckpt.get(
            "cnn_channels_per_scale", cnn_channels_per_scale)
        base_channels = resume_ckpt.get("base_channels", base_channels)
        channel_mult = resume_ckpt.get("channel_mult", channel_mult)
        blocks_per_stage = resume_ckpt.get("blocks_per_stage", blocks_per_stage)
        dropout = resume_ckpt.get("dropout", dropout)

        run_dir = resume_ckpt.get("run_dir", None)
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = str(Path(base_dir) / timestamp)
        run_dir = Path(run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_dir) / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)

    if save_path is None:
        save_path = str(run_dir / "model_best.pth")
    else:
        save_path = Path(save_path)

    if result_dir is None:
        result_dir = str(run_dir / "vis")
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    print(f"训练运行目录: {run_dir}")
    print(f"可视化结果将保存至: {result_dir}")

    # ============================================================
    # 1. Dataset / DataLoader
    # ============================================================
    train_dataset = BSDS_RCFLoader(
        root=train_root, split="train", resize=None, transform=False, get_pil=True,
        use_twin_augmentation=False, twin_aug_prob=0.5,
        use_speckle_augmentation=False, speckle_aug_prob=0.5, channel=1,
        use_rotation=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = None
    if val_root is not None:
        val_dataset = BSDS_RCFLoader(
            root=val_root, split="test", resize=None, transform=False, get_pil=True,
            use_twin_augmentation=False, twin_aug_prob=0.5,
            use_speckle_augmentation=False, speckle_aug_prob=0.2, channel=1,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    # ============================================================
    # 2. 模型
    # ============================================================
    model = GaborLogistic(
        feature_type=feature_type,
        cnn_channels_per_scale=cnn_channels_per_scale,
        base_channels=base_channels,
        channel_mult=channel_mult,
        blocks_per_stage=blocks_per_stage,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {num_params:,}")

    # ============================================================
    # 3. 恢复训练状态
    # ============================================================
    start_epoch = 1
    best_loss = float("inf")
    global_counter = [0, 0]

    if resume_ckpt is not None:
        # 3.1 模型权重
        if "model_state_dict" in resume_ckpt:
            model.load_state_dict(resume_ckpt["model_state_dict"])
        else:
            print("Warning: checkpoint 中没有 model_state_dict，跳过模型权重加载。")

        # 3.2 优化器状态（含 device 迁移）
        if "optimizer_state_dict" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            except Exception as e:
                print(f"Warning: 优化器状态加载失败 ({e})，将使用全新优化器状态。")

        # 3.3 轮次
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1

        # 3.4 best loss
        best_loss = float(resume_ckpt.get("best_loss", float("inf")))

        # 3.5 全局计数器 [image_counter, last_saved_counter]
        gc = resume_ckpt.get("global_counter", [0, 0])
        if isinstance(gc, (list, tuple)) and len(gc) == 2:
            global_counter = [int(gc[0]), int(gc[1])]

        print(f"恢复训练：start_epoch = {start_epoch}, best_loss = {best_loss:.6f}")
        print(f"           global_counter = {global_counter}")

    # ============================================================
    # 4. 可视化回调
    # ============================================================
    def save_callback(images, predictions, labels, step):
        save_visualization(
            images=images, predictions=predictions, labels=labels,
            step=step, save_dir=result_dir, max_save=4,
        )

    # ============================================================
    # 5. 打印配置
    # ============================================================
    print("=" * 70)
    print("Multi-scale Feature Frontend + U-Net Head + BCE")
    print("=" * 70)
    print(f"device          : {device}")
    print(f"feature_type    : {feature_type}")
    print(f"train size      : {len(train_dataset)}")
    print(f"epochs          : {epochs}")
    print(f"batch size      : {batch_size}")
    print(f"lr              : {lr}")
    print(f"balance         : {balance}")
    print(f"neg_boost       : {neg_boost}")
    print(f"result_dir      : {result_dir}")
    print(f"save_interval   : {save_interval}")
    print(f"base_channels   : {base_channels}")
    print(f"channel_mult    : {tuple(channel_mult)}")
    print(f"blocks_per_stage: {blocks_per_stage}")
    if feature_type == 'cnn':
        print(f"cnn_ch/scale    : {cnn_channels_per_scale}")
    if resume_from:
        print(f"resume_from     : {resume_from}")
    print("=" * 70)

    # ============================================================
    # 6. 训练循环
    # ============================================================
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            balance=balance, neg_boost=neg_boost,
            epoch=epoch, total_epochs=epochs,
            save_callback=save_callback, save_interval=save_interval,
            global_counter=global_counter,
        )

        if val_loader is not None:
            val_loss, val_has_label = evaluate(
                model, val_loader, device,
                balance=balance, neg_boost=neg_boost,
                epoch=epoch, total_epochs=epochs,
            )
            if not val_has_label:
                val_loss = train_loss
                print("Note: Validation set has no labels, using training loss for model selection.")
        else:
            val_loss = train_loss

        print(f"[{epoch:03d}/{epochs:03d}] train={train_loss:.6f} val={val_loss:.6f}")

        # ---- 保存最佳模型 ----
        if val_loss < best_loss:
            best_loss = val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "best_loss": best_loss,
                "feature_type": model.feature_type,
                "cnn_channels_per_scale": model.cnn_channels_per_scale,
                "sigmas": model.sigmas,
                "wavelengths": model.wavelengths,
                "orientations": model.orientations,
                "gamma": model.gamma,
                "base_channels": model.base_channels,
                "channel_mult": model.channel_mult,
                "blocks_per_stage": model.blocks_per_stage,
                "dropout": model.dropout,
            }
            p = run_dir / f"model_loss_{val_loss:.6f}.pth"
            torch.save(ckpt, p)
            print(f"  -> 保存最佳模型: {p}")

        # ---- 每个 epoch 保存 last.pth（用于 resume） ----
        last_ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "global_counter": list(global_counter),
            "run_dir": str(run_dir),
            "feature_type": model.feature_type,
            "cnn_channels_per_scale": model.cnn_channels_per_scale,
            "sigmas": model.sigmas,
            "wavelengths": model.wavelengths,
            "orientations": model.orientations,
            "gamma": model.gamma,
            "base_channels": model.base_channels,
            "channel_mult": model.channel_mult,
            "blocks_per_stage": model.blocks_per_stage,
            "dropout": model.dropout,
        }
        torch.save(last_ckpt, run_dir / "last.pth")

    # ============================================================
    # 7. 最终模型
    # ============================================================
    ckpt = {
        "model_state_dict": model.state_dict(),
        "best_loss": best_loss,
        "feature_type": model.feature_type,
        "cnn_channels_per_scale": model.cnn_channels_per_scale,
        "sigmas": model.sigmas,
        "wavelengths": model.wavelengths,
        "orientations": model.orientations,
        "gamma": model.gamma,
        "base_channels": model.base_channels,
        "channel_mult": model.channel_mult,
        "blocks_per_stage": model.blocks_per_stage,
        "dropout": model.dropout,
    }
    final_path = run_dir / "model_final.pth"
    torch.save(ckpt, final_path)
    print(f"最终模型已保存至: {final_path}")

    return model


# ============================================================
# 9. 加载模型
# ============================================================

_DEFAULT_SIGMAS = (1.0, 2.0, 4.0, 8.0)
_DEFAULT_ORIENTATIONS = (
    math.pi / 16, 3 * math.pi / 16, 5 * math.pi / 16, 7 * math.pi / 16,
    9 * math.pi / 16, 11 * math.pi / 16, 13 * math.pi / 16, 15 * math.pi / 16,
)


def _build_from_ckpt(ckpt, cls, device):
    feature_type = ckpt.get("feature_type", "gabor_fixed")
    orientations = ckpt.get("orientations", None)
    if feature_type in ("gabor_fixed", "gabor_learnable") and orientations is None:
        orientations = _DEFAULT_ORIENTATIONS

    model = cls(
        sigmas=ckpt.get("sigmas", _DEFAULT_SIGMAS),
        wavelengths=ckpt.get("wavelengths", None),
        orientations=orientations,
        gamma=ckpt.get("gamma", 0.5),
        feature_type=feature_type,
        cnn_channels_per_scale=ckpt.get("cnn_channels_per_scale", 8),
        base_channels=ckpt.get("base_channels", 32),
        channel_mult=ckpt.get("channel_mult", (1, 2, 4, 8)),
        blocks_per_stage=ckpt.get("blocks_per_stage", 2),
        dropout=ckpt.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_model(checkpoint_path: str, device: Optional[str] = None) -> nn.Module:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    return _build_from_ckpt(ckpt, GaborLogistic, device)


def load_model_logit(checkpoint_path: str, device: Optional[str] = None) -> nn.Module:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    return _build_from_ckpt(ckpt, GaborLogistic_logit, device)


# ============================================================
# 10. 单张图推理 / 保存 / 堆栈预测
# ============================================================

@torch.no_grad()
def predict_single_image(model, image, device=None) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4:
        raise ValueError("image 必须是 [H,W] / [1,H,W] / [1,1,H,W]")
    image = image.float().to(device)
    return model.predict(image)[0, 0].cpu()


def save_probability_map(probability: torch.Tensor, path: str):
    arr = probability.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


@torch.no_grad()
def predict_reconstruction_stack(
    model, reconstruction_stack, device=None, batch_size: int = 8,
) -> torch.Tensor:
    if reconstruction_stack.ndim != 3:
        raise ValueError("reconstruction_stack 必须是 [K,H,W]")
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.eval()
    outputs = []
    K = reconstruction_stack.shape[0]
    for start in range(0, K, batch_size):
        batch = reconstruction_stack[start:start + batch_size].float().to(device).unsqueeze(1)
        outputs.append(model(batch)[:, 0].cpu())
    return torch.cat(outputs, dim=0)


# ============================================================
# 11. 主程序
# ============================================================

if __name__ == "__main__":
    # ==================== 前端 ====================
    # 'gabor_fixed'     : 固定 Gabor（不可学习）
    # 'gabor_learnable' : Gabor 核可学习
    # 'cnn'             : 可学习 CNN
    FEATURE_TYPE = 'gabor_fixed'

    # ==================== U-Net 深头 ====================
    # base_channels   : stage 0 的通道数
    # channel_mult    : 各 stage 通道倍率，长度 = stage 数（= 池化次数 + 1）
    # blocks_per_stage: 每 stage 堆叠的 Conv3x3 数量
    #
    # 例：
    #   base=32, mult=(1,2,4,8), blocks=2  → ~4M 参数，3 次池化（H → H/8）
    #   base=16, mult=(1,2,4,8), blocks=2  → ~1M 参数
    #   base=32, mult=(1,2,4,8,16), blocks=2 → ~16M 参数，4 次池化（H → H/16）
    BASE_CHANNELS = 32
    CHANNEL_MULT = (1, 2, 4, 8)
    BLOCKS_PER_STAGE = 2
    DROPOUT = 0.0

    # ==================== resume 配置 ====================
    # 全新训练：RESUME_FROM = None
    # 断点续训：RESUME_FROM = r"runs_cnn\20260901_213420\last.pth"
    # RESUME_FROM = r'F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing\edge_prediction\runs_cnn\20260916_095313\model_loss_0.021887.pth'
    RESUME_FROM = None
    model = train_model(
        train_root=r"F:\DongJiayao\Data\PASCAL",
        val_root=r"F:\DongJiayao\Data\PASCAL",
        save_path="gabor_logistic_best.pth",
        epochs=100,
        batch_size=1,
        lr=1e-4,
        balance=True,
        neg_boost=1.0,
        result_dir="vis_results",
        save_interval=5000,
        feature_type=FEATURE_TYPE,
        cnn_channels_per_scale=8,
        base_channels=BASE_CHANNELS,
        channel_mult=CHANNEL_MULT,
        blocks_per_stage=BLOCKS_PER_STAGE,
        dropout=DROPOUT,
        resume_from=RESUME_FROM,
    )