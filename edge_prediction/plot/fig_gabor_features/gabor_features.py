# -*- coding: utf-8 -*-
"""
多尺度 Gabor 特征提取与可视化（论文双栏适配版）
- 白底
- 特征图之间以白色边缘线分隔
- 行/列标签在单元格内居中
- 尺度为行、方向为列（横向为方向，纵向为尺度）
- 子图 128×128，字号 26（文字约占子图高度 20%）
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-8


# ----------------------------- Gabor 核生成 -------------------------
def make_gabor_quadrature_kernels(
        sigma: float,
        wavelength: float,
        orientations: List[float],
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


# ----------------------------- 多尺度 Gabor 提取器 -------------------
class MultiScaleGabor(torch.nn.Module):
    def __init__(
            self,
            sigmas: List[float] = (1.0, 2.0, 4.0, 8.0),
            wavelengths: Optional[List[float]] = None,
            orientations: List[float] = (
                    0.0, math.pi / 8, math.pi / 4, 3.0 * math.pi / 8,
                    math.pi / 2, 5.0 * math.pi / 8, 3.0 * math.pi / 4,
                    7.0 * math.pi / 8,
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
        self.num_scales = len(sigmas)
        self.num_orientations = len(orientations)

        for i, (sigma, wavelength) in enumerate(zip(self.sigmas, self.wavelengths)):
            kernel = make_gabor_quadrature_kernels(
                sigma=sigma,
                wavelength=wavelength,
                orientations=self.orientations,
                gamma=gamma,
            )
            self.register_buffer(f"kernel_{i}", kernel, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[0] != 1:
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
            if C == 2 * O:
                response = response.view(B, O, 2, H, W)
                even = response[:, :, 0, :, :]
                odd = response[:, :, 1, :, :]
            else:
                even = response[:, 0::2]
                odd = response[:, 1::2]
                if even.shape[2:] != odd.shape[2:]:
                    odd = F.interpolate(odd, size=even.shape[2:],
                                        mode='bilinear', align_corners=False)
            magnitude = torch.sqrt(even * even + odd * odd + EPS)
            features.append(magnitude)
        return torch.cat(features, dim=1)


# ----------------------------- 创建提取器 ----------------------------
def create_gabor_extractor(
        sigmas: List[float] = (1.0, 2.0, 4.0, 8.0),
        wavelengths: Optional[List[float]] = None,
        orientations: List[float] = (
                0.0, math.pi / 8, math.pi / 4, 3.0 * math.pi / 8,
                math.pi / 2, 5.0 * math.pi / 8, 3.0 * math.pi / 4,
                7.0 * math.pi / 8,
        ),
        gamma: float = 0.5,
        device: str = "cpu",
) -> MultiScaleGabor:
    extractor = MultiScaleGabor(
        sigmas=sigmas, wavelengths=wavelengths,
        orientations=orientations, gamma=gamma,
    )
    extractor.to(device)
    extractor.eval()
    return extractor


# ----------------------------- 特征提取封装 -------------------------
def extract_gabor_features(
        extractor: MultiScaleGabor,
        image: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    with torch.no_grad():
        combined = extractor(image)
    O = extractor.num_orientations
    per_scale = []
    start = 0
    for _ in range(extractor.num_scales):
        end = start + O
        per_scale.append(combined[:, start:end, :, :])
        start = end
    return combined, per_scale


# ----------------------------- 保存单独特征图 -----------------------
def save_individual_maps(
        per_scale: List[torch.Tensor],
        save_dir: str = "individual_maps",
        prefix: str = "gabor",
        resize_to: Optional[Tuple[int, int]] = (256, 256),
):
    os.makedirs(save_dir, exist_ok=True)
    for s, scale_feat in enumerate(per_scale):
        feat = scale_feat[0]
        for o in range(feat.shape[0]):
            fmap = feat[o].cpu().numpy()
            if resize_to is not None:
                t = torch.from_numpy(fmap).float().unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=resize_to, mode='bilinear',
                                  align_corners=False)
                fmap = t.squeeze().numpy()
            fmin, fmax = fmap.min(), fmap.max()
            if fmax > fmin:
                fmap = (fmap - fmin) / (fmax - fmin) * 255
            else:
                fmap = np.zeros_like(fmap)
            Image.fromarray(fmap.astype(np.uint8), mode='L').save(
                os.path.join(save_dir, f"{prefix}_scale{s:02d}_ori{o:02d}.png")
            )
    print(f"单独特征图已保存至: {save_dir}")


# ----------------------------- 字体加载 -----------------------------
def _load_font(font_size: int, font_path: Optional[str] = None):
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates += [
        "arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


# ----------------------------- 拼接图（论文双栏适配） ----------------
def save_combined_grid(
        per_scale: List[torch.Tensor],
        sigmas: List[float],
        orientations: List[float],
        save_path: str = "gabor_grid.png",
        mode: str = "scale_major",
        normalize_per_patch: bool = True,
        border: int = 4,                    # 白色边缘线宽度
        show_labels: bool = True,
        font_size: int = 26,                # 相对子图放大
        resize_to: Optional[Tuple[int, int]] = (128, 128),  # 子图缩小
        font_path: Optional[str] = None,
):
    feat_list = [f[0].cpu().numpy() for f in per_scale]
    num_scales = len(feat_list)
    num_orient = len(orientations)

    # 1. 统一缩放到较小尺寸
    if resize_to is not None:
        new_H, new_W = resize_to
        for s in range(num_scales):
            t = torch.from_numpy(feat_list[s]).float().unsqueeze(0)
            t = F.interpolate(t, size=(new_H, new_W), mode='bilinear',
                              align_corners=False)
            feat_list[s] = t.squeeze(0).numpy()
        H, W = new_H, new_W
    else:
        H, W = feat_list[0].shape[1], feat_list[0].shape[2]

    # 2. 归一化
    def normalize(patch):
        if normalize_per_patch:
            pmin, pmax = patch.min(), patch.max()
            if pmax > pmin:
                return ((patch - pmin) / (pmax - pmin) * 255).astype(np.uint8)
            return np.zeros_like(patch, dtype=np.uint8)
        return patch.astype(np.uint8)

    # 3. 组织 patch 矩阵与标签（标签更紧凑）
    if mode == "scale_major":
        patches = [[normalize(feat_list[s][o]) for o in range(num_orient)]
                   for s in range(num_scales)]
        row_labels = [f"σ={sigmas[s]:.0f}" for s in range(num_scales)]
        col_labels = [f"{int(round(np.degrees(orientations[o])))}°"
                      for o in range(num_orient)]
    else:
        patches = [[normalize(feat_list[s][o]) for s in range(num_scales)]
                   for o in range(num_orient)]
        row_labels = [f"{int(round(np.degrees(orientations[o])))}°"
                      for o in range(num_orient)]
        col_labels = [f"σ={sigmas[s]:.0f}" for s in range(num_scales)]

    num_rows = len(patches)
    num_cols = len(patches[0])

    # 4. 字体与标签尺寸
    font = _load_font(font_size, font_path)
    tmp_img = Image.new('L', (1, 1), 255)
    tmp_draw = ImageDraw.Draw(tmp_img)

    def measure(text):
        bbox = tmp_draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    row_label_w = 0
    col_label_h = 0
    if show_labels:
        for lbl in row_labels:
            w, _ = measure(lbl)
            row_label_w = max(row_label_w, w)
        row_label_w += 10

        for lbl in col_labels:
            _, h = measure(lbl)
            col_label_h = max(col_label_h, h)
        col_label_h += 10

    # 5. 整图尺寸
    total_W = row_label_w + num_cols * W + (num_cols - 1) * border
    total_H = col_label_h + num_rows * H + (num_rows - 1) * border

    # 6. 白色背景
    grid_img = Image.new('L', (total_W, total_H), color=255)
    draw = ImageDraw.Draw(grid_img)

    # 7. 标签居中
    if show_labels:
        # 列标签：水平居中
        for c, lbl in enumerate(col_labels):
            x0 = row_label_w + c * (W + border)
            tw, th = measure(lbl)
            draw.text((x0 + (W - tw) // 2,
                       (col_label_h - th) // 2 - 2),
                      lbl, fill=0, font=font)
        # 行标签：垂直居中，右对齐贴近子图
        for r, lbl in enumerate(row_labels):
            y0 = col_label_h + r * (H + border)
            tw, th = measure(lbl)
            draw.text((row_label_w - tw - 5,
                       y0 + (H - th) // 2 - 2),
                      lbl, fill=0, font=font)

    # 8. 粘贴子图（白底 + border 间隙 = 白色边缘线）
    for r, row in enumerate(patches):
        for c, patch in enumerate(row):
            patch_img = Image.fromarray(patch, mode='L')
            x0 = row_label_w + c * (W + border)
            y0 = col_label_h + r * (H + border)
            grid_img.paste(patch_img, (x0, y0))

    grid_img.save(save_path)
    print(f"组合拼接图已保存: {save_path}  (尺寸 {total_W}×{total_H})")


# ----------------------------- 主程序 -------------------------------
if __name__ == "__main__":
    image_path = (
        r"F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing"
        r"\edge_prediction_gabor\plot\images\input\off_88_0.0002350.jpg"
    )
    save_dir_indiv = "../individual_maps"
    save_grid_scale = "gabor_grid_scale.png"
    save_grid_orient = "gabor_grid_orient.png"

    img_pil = Image.open(image_path).convert('L')
    img_np = np.array(img_pil, dtype=np.float32)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    sigmas = (1.0, 2.0, 4.0, 8.0)
    orientations = (0.0, math.pi / 8, math.pi / 4, 3 * math.pi / 8,
                    math.pi / 2, 5 * math.pi / 8, 3 * math.pi / 4,
                    7 * math.pi / 8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    extractor = create_gabor_extractor(
        sigmas=sigmas, orientations=orientations, device=device,
    )
    combined, per_scale = extract_gabor_features(extractor,
                                                 img_tensor.to(device))

    print(f"组合特征形状: {combined.shape}")
    print(f"尺度数: {len(per_scale)}, 每个尺度方向数: {per_scale[0].shape[1]}")

    # 单独特征图仍保持 256，便于单独查看
    save_individual_maps(per_scale, save_dir_indiv, resize_to=(256, 256))

    # 拼接图：尺度为行、方向为列（双栏推荐）
    save_combined_grid(
        per_scale,
        sigmas=sigmas,
        orientations=orientations,
        save_path=save_grid_scale,
        mode="scale_major",
        normalize_per_patch=True,
        border=4,
        show_labels=True,
        font_size=26,              # 文字相对子图放大
        resize_to=(128, 128),      # 子图缩小
    )

    # 拼接图：方向为行、尺度为列
    save_combined_grid(
        per_scale,
        sigmas=sigmas,
        orientations=orientations,
        save_path=save_grid_orient,
        mode="orientation_major",
        normalize_per_patch=True,
        border=4,
        show_labels=True,
        font_size=26,
        resize_to=(128, 128),
    )

    print("所有处理完成！")