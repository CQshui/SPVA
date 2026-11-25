# -*- coding: utf-8 -*-
"""
基于 PTH 模型参数 + 单张图像特征统计 的联合分析
==================================================

打印内容：
1. 每个尺度每个方向的 权重 (w)
2. 每个尺度每个方向的 特征均值 (mean)
3. 每个尺度每个方向的 特征标准差 (std)

用于诊断：特征统计相同的方向，权重是否对称？
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# 1. 参数
# ============================================================

MODEL_PATH = (
    r"F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing\edge_prediction_gabor\runs_rotate\20260901_001521\model_loss_0.024344.pth"
)

# 随便选一张测试图像（用于提取特征统计）
TEST_IMAGE_PATH = (
    r"F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing\edge_prediction_gabor\plot\input\off_88_0.0002350.jpg"
)

OUTPUT_DIR = r"weight_feature_joint_analysis"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPS = 1e-8


# ============================================================
# 2. 默认方向
# ============================================================

DEFAULT_ORIENTATIONS = (
    0.0,
    math.pi / 8,
    math.pi / 4,
    3.0 * math.pi / 8,
    math.pi / 2,
    5.0 * math.pi / 8,
    3.0 * math.pi / 4,
    7.0 * math.pi / 8,
)


# ============================================================
# 3. Gabor 工具（与训练完全一致）
# ============================================================

def make_odd(value):
    value = int(value)
    if value % 2 == 0:
        value += 1
    return max(value, 3)


def create_coordinate_grid(kernel_size, device="cpu"):
    half = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        indexing="ij"
    )
    return x, y


def make_base_gabor_kernel(
    sigma,
    wavelength,
    gamma=0.5,
    kernel_size=None,
    device="cpu"
):
    if kernel_size is None:
        kernel_size = int(math.ceil(6.0 * sigma))
        kernel_size = make_odd(kernel_size)
        kernel_size = max(kernel_size, 7)

    x, y = create_coordinate_grid(kernel_size, device=device)
    x_theta = x
    y_theta = y

    gaussian = torch.exp(
        -(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2.0 * sigma ** 2)
    )
    carrier = 2.0 * math.pi * x_theta / wavelength
    even = gaussian * torch.cos(carrier)
    odd = gaussian * torch.sin(carrier)

    even = even - even.mean()
    odd = odd - odd.mean()
    even = even / (torch.sqrt(torch.sum(even ** 2)) + EPS)
    odd = odd / (torch.sqrt(torch.sum(odd ** 2)) + EPS)

    return even.unsqueeze(0).unsqueeze(0), odd.unsqueeze(0).unsqueeze(0)


def rotate_kernel(kernel, angle):
    angle_degree = angle * 180.0 / math.pi
    nearest_multiple = round(angle_degree / 90.0)
    if abs(angle_degree - nearest_multiple * 90.0) < 1e-6:
        k = nearest_multiple % 4
        return torch.rot90(kernel, k=k, dims=(-2, -1))

    device = kernel.device
    theta = torch.tensor(
        [
            [math.cos(angle), math.sin(angle), 0.0],
            [-math.sin(angle), math.cos(angle), 0.0]
        ],
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)
    grid = F.affine_grid(theta, kernel.size(), align_corners=True)
    rotated = F.grid_sample(kernel, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return rotated


def make_rotation_equivariant_gabor_kernels(
    sigma,
    wavelength,
    orientations,
    gamma=0.5,
    kernel_size=None,
    device="cpu"
):
    base_even, base_odd = make_base_gabor_kernel(
        sigma=sigma,
        wavelength=wavelength,
        gamma=gamma,
        kernel_size=kernel_size,
        device=device
    )
    kernels = []
    for theta in orientations:
        even = rotate_kernel(base_even, theta)
        odd = rotate_kernel(base_odd, theta)
        even = even - even.mean()
        odd = odd - odd.mean()
        even = even / (torch.sqrt(torch.sum(even ** 2)) + EPS)
        odd = odd / (torch.sqrt(torch.sum(odd ** 2)) + EPS)
        kernels.append(even)
        kernels.append(odd)
    return torch.cat(kernels, dim=0)


# ============================================================
# 4. MultiScaleGabor（与训练一致）
# ============================================================

class MultiScaleGabor(nn.Module):
    def __init__(
        self,
        sigmas=(1.0, 2.0, 4.0, 8.0),
        wavelengths=None,
        orientations=DEFAULT_ORIENTATIONS,
        gamma=0.5,
        normalize_scales=True,
        noise_wavelength_for_sigma1=1.2
    ):
        super().__init__()
        self.sigmas = tuple(sigmas)
        self.orientations = tuple(orientations)
        self.gamma = gamma
        self.normalize_scales = normalize_scales
        self.noise_wavelength_for_sigma1 = noise_wavelength_for_sigma1

        if wavelengths is None:
            custom_wavelengths = []
            for s in sigmas:
                if abs(s - 1.0) < 1e-6:
                    custom_wavelengths.append(noise_wavelength_for_sigma1)
                else:
                    custom_wavelengths.append(2.5 * s)
            wavelengths = tuple(custom_wavelengths)
        if len(wavelengths) != len(sigmas):
            raise ValueError("wavelengths 必须与 sigmas 长度一致")
        self.wavelengths = tuple(wavelengths)

        self.num_scales = len(sigmas)
        self.num_orientations = len(orientations)

        for i, (sigma, wavelength) in enumerate(zip(self.sigmas, self.wavelengths)):
            kernel = make_rotation_equivariant_gabor_kernels(
                sigma=sigma,
                wavelength=wavelength,
                orientations=self.orientations,
                gamma=gamma
            )
            self.register_buffer(f"kernel_{i}", kernel, persistent=True)

    def forward(self, x):
        """完整的特征提取（含尺度归一化）"""
        if x.ndim == 3:
            if x.shape[0] != 1:
                x = x.unsqueeze(1)
        elif x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)

        features = []
        for i in range(self.num_scales):
            kernel = getattr(self, f"kernel_{i}")
            padding = kernel.shape[-1] // 2
            response = F.conv2d(x, kernel, padding=padding)
            B, C, H, W = response.shape
            O = self.num_orientations
            response = response.view(B, O, 2, H, W)
            even = response[:, :, 0, :, :]
            odd = response[:, :, 1, :, :]
            magnitude = torch.sqrt(even * even + odd * odd + EPS)

            # 尺度归一化
            if self.normalize_scales:
                mean = magnitude.mean(dim=(2, 3), keepdim=True)
                var = magnitude.var(dim=(2, 3), keepdim=True, unbiased=False)
                magnitude = (magnitude - mean) / (var + EPS).sqrt()

            features.append(magnitude)
        return torch.cat(features, dim=1)


# ============================================================
# 5. GaborLogistic（与训练一致）
# ============================================================

class GaborLogistic(nn.Module):
    def __init__(
        self,
        sigmas=(1.0, 2.0, 4.0, 8.0),
        wavelengths=None,
        orientations=DEFAULT_ORIENTATIONS,
        gamma=0.5,
        normalize_scales=True,
        noise_wavelength_for_sigma1=1.2
    ):
        super().__init__()
        self.gabor = MultiScaleGabor(
            sigmas=sigmas,
            wavelengths=wavelengths,
            orientations=orientations,
            gamma=gamma,
            normalize_scales=normalize_scales,
            noise_wavelength_for_sigma1=noise_wavelength_for_sigma1
        )
        num_features = len(sigmas) * len(orientations)
        self.classifier = nn.Conv2d(
            in_channels=num_features,
            out_channels=1,
            kernel_size=1,
            bias=True
        )

    def forward_features(self, x):
        return self.gabor(x)

    def forward(self, x):
        features = self.forward_features(x)
        return torch.sigmoid(self.classifier(features))


# ============================================================
# 6. 加载模型
# ============================================================

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    sigmas = checkpoint.get("sigmas", (1.0, 2.0, 4.0, 8.0))
    wavelengths = checkpoint.get("wavelengths", None)
    orientations = checkpoint.get("orientations", DEFAULT_ORIENTATIONS)
    gamma = checkpoint.get("gamma", 0.5)
    normalize_scales = checkpoint.get("normalize_scales", True)
    noise_wavelength_for_sigma1 = checkpoint.get("noise_wavelength_for_sigma1", 1.2)

    if wavelengths is None:
        wavelengths = tuple(
            1.2 if abs(s - 1.0) < 1e-6 else 2.5 * s
            for s in sigmas
        )

    model = GaborLogistic(
        sigmas=sigmas,
        wavelengths=wavelengths,
        orientations=orientations,
        gamma=gamma,
        normalize_scales=normalize_scales,
        noise_wavelength_for_sigma1=noise_wavelength_for_sigma1
    )

    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        clean_state_dict[key] = value

    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    return model


# ============================================================
# 7. 主程序：联合打印权重 + 特征统计
# ============================================================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("权重 + 特征统计 联合分析")
    print("=" * 80)

    # ---------- 加载模型 ----------
    print("\n加载模型...")
    model = load_model().to(DEVICE)

    # 提取权重
    weight = model.classifier.weight.detach().cpu().numpy().reshape(-1)
    bias = float(model.classifier.bias.detach().cpu().numpy().reshape(-1)[0])

    sigmas = model.gabor.sigmas
    orientations = model.gabor.orientations
    wavelengths = model.gabor.wavelengths
    num_scales = len(sigmas)
    num_orient = len(orientations)

    print(f"\n模型配置:")
    print(f"  sigmas: {sigmas}")
    print(f"  wavelengths: {wavelengths}")
    print(f"  normalize_scales: {model.gabor.normalize_scales}")
    print(f"  bias: {bias:.8f}")

    # ---------- 加载测试图像 ----------
    print(f"\n加载测试图像: {TEST_IMAGE_PATH}")
    img_pil = Image.open(TEST_IMAGE_PATH).convert('L')
    img_np = np.array(img_pil, dtype=np.float32)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # ---------- 提取特征 ----------
    print("提取特征统计...")
    with torch.no_grad():
        features = model.forward_features(img_tensor)  # [1, 32, H, W]

    # 重塑为 [num_scales, num_orient, H, W]
    features_reshaped = features.view(1, num_scales, num_orient, features.shape[-2], features.shape[-1])
    features_reshaped = features_reshaped.squeeze(0)  # [num_scales, num_orient, H, W]

    # 计算每个通道的 mean 和 std（在空间维度上）
    channel_means = features_reshaped.mean(dim=(2, 3)).cpu().numpy()  # [num_scales, num_orient]
    channel_stds = features_reshaped.std(dim=(2, 3), unbiased=False).cpu().numpy()  # [num_scales, num_orient]

    # 权重重塑为 [num_scales, num_orient]
    weight_matrix = weight.reshape(num_scales, num_orient)

    # ---------- 打印表格 ----------
    print("\n" + "=" * 80)
    print("按尺度分组打印：权重 | 特征均值 | 特征标准差")
    print("=" * 80)

    for s in range(num_scales):
        print(f"\n【尺度 {s}】 σ={sigmas[s]:.1f}, λ={wavelengths[s]:.2f}")
        print("-" * 80)
        print(f"{'方向':>8} | {'角度':>8} | {'权重 w':>12} | {'特征均值':>14} | {'特征标准差':>14}")
        print("-" * 80)

        for o in range(num_orient):
            angle_deg = np.degrees(orientations[o])
            w_val = weight_matrix[s, o]
            mean_val = channel_means[s, o]
            std_val = channel_stds[s, o]
            print(f"{o:>8} | {angle_deg:>8.1f}° | {w_val:>+12.8f} | {mean_val:>+14.8f} | {std_val:>14.8f}")

        # 本尺度统计
        print("-" * 80)
        print(f"{'本尺度统计':>8} | {'':>8} | mean={np.mean(weight_matrix[s]):>+8.6f} | mean={np.mean(channel_means[s]):>+14.8f} | mean={np.mean(channel_stds[s]):>14.8f}")
        print(f"{'':>8} | {'':>8} | std={np.std(weight_matrix[s]):>+8.6f} | std={np.std(channel_means[s]):>+14.8f} | std={np.std(channel_stds[s]):>14.8f}")

    # ---------- 全局统计 ----------
    print("\n" + "=" * 80)
    print("全局统计")
    print("=" * 80)
    print(f"权重 全局均值: {np.mean(weight):.8f}")
    print(f"权重 全局标准差: {np.std(weight):.8f}")
    print(f"特征均值 全局均值: {np.mean(channel_means):.8f}")
    print(f"特征均值 全局标准差: {np.std(channel_means):.8f}")
    print(f"特征标准差 全局均值: {np.mean(channel_stds):.8f}")
    print(f"特征标准差 全局标准差: {np.std(channel_stds):.8f}")

    # ---------- 关键发现：找特征相近但权重相差大的方向对 ----------
    print("\n" + "=" * 80)
    print("⚠️ 特征相近但权重差异大的方向对（潜在问题）")
    print("=" * 80)

    # 对每个尺度，找特征均值最接近但权重差异最大的两个方向
    for s in range(num_scales):
        means = channel_means[s]
        weights = weight_matrix[s]

        # 计算所有方向对的特征均值差异和权重差异
        pairs = []
        for i in range(num_orient):
            for j in range(i + 1, num_orient):
                mean_diff = abs(means[i] - means[j])
                weight_diff = abs(weights[i] - weights[j])
                pairs.append((i, j, mean_diff, weight_diff))

        # 按特征均值差异从小到大排序，取前3个
        pairs_sorted = sorted(pairs, key=lambda x: x[2])
        print(f"\n尺度 {s} (σ={sigmas[s]:.1f}):")
        print(f"  {'方向对':>12} | {'角度':>20} | {'均值差':>14} | {'权重差':>14}")
        print("  " + "-" * 60)
        for i, j, md, wd in pairs_sorted[:5]:
            ang_i = np.degrees(orientations[i])
            ang_j = np.degrees(orientations[j])
            print(f"  {i:>3} ↔ {j:<3} | {ang_i:>6.1f}° ↔ {ang_j:<6.1f}° | {md:>14.8f} | {wd:>+14.8f}")

    # ---------- 保存结果 ----------
    # 保存为 CSV
    import csv
    csv_path = output_dir / "weight_feature_joint.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['scale', 'sigma', 'orientation_idx', 'angle_deg', 'weight', 'feature_mean', 'feature_std'])
        for s in range(num_scales):
            for o in range(num_orient):
                writer.writerow([
                    s,
                    sigmas[s],
                    o,
                    np.degrees(orientations[o]),
                    weight_matrix[s, o],
                    channel_means[s, o],
                    channel_stds[s, o]
                ])
    print(f"\n结果已保存至: {csv_path}")

    # 保存为文本
    txt_path = output_dir / "weight_feature_joint.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("权重 + 特征统计 联合分析\n")
        f.write("=" * 80 + "\n\n")
        for s in range(num_scales):
            f.write(f"尺度 {s}: σ={sigmas[s]:.1f}, λ={wavelengths[s]:.2f}\n")
            f.write(f"{'方向':>6} {'角度':>8} {'权重':>14} {'特征均值':>16} {'特征标准差':>16}\n")
            f.write("-" * 60 + "\n")
            for o in range(num_orient):
                f.write(f"{o:>6} {np.degrees(orientations[o]):>8.1f}° {weight_matrix[s, o]:>+14.8f} {channel_means[s, o]:>+16.8f} {channel_stds[s, o]:>16.8f}\n")
            f.write("\n")

    print(f"文本结果已保存至: {txt_path}")

    print("\n" + "=" * 80)
    print("分析完成！请检查上述打印中，是否存在特征均值接近但权重符号/大小差异巨大的方向对。")
    print("如果存在，说明训练过程中这些方向受到了不对称的影响（数据增强、优化器随机性等）。")
    print("=" * 80)


if __name__ == "__main__":
    main()