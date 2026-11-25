#!/usr/bin/env python
# coding=utf-8
import os
import argparse
from pathlib import Path  # 新增导入

# ── 必须在所有 torch/cuda import 之前设置 ──────────────────────
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--gpu', default='3', type=str)   # ← 改这里指定 GPU
_pre_args, _ = _pre_parser.parse_known_args()
os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu
# ────────────────────────────────────────────────────────────────

import csv
import math
import re
import time
import random
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from natsort import natsorted
from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from edge_prediction.model.DLEP_complex import ComplexDLEPV1
# 替换 RCF 导入为 GaborLogistic
from main_rotate_deep_Gabor import GaborLogistic, load_model   # 确保 main.py 在同一目录
from peak_finder.peakfinder import PeakFinder
from peak_finder.brent import brent_method, multi_start_brent

from edge_prediction.predict import calculate_brightness_concentration
from edge_prediction.dataloader.data_loader import convert_to_rgb, prepare_image_PIL, prepare_image_cv2, prepare_image_torch_gpu
from edge_prediction.model.DLEP import DLEP, DLEPLite, DLEPLite1, DLEPLite2
from peak_finder.traverse import traverse

from propagate.angular_torch import AngularBatch


def set_deterministic():
    """设置确定性运行环境"""
    import random
    import numpy as np
    import torch

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"设置确定性环境，随机种子: {seed}")

set_deterministic()


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行耗时: {end - start:.4f} 秒")
        return result
    return wrapper


# ==================== 完全采用 Experiment 的数据读取方式 ====================
def initial(root, fx_ref=None, fy_ref=None):
    hologram = Image.open(root).convert('L')
    filtered = AngularBatch(lam=532e-9,
                            pix=0.098e-6,
                            z_initial=0.00030,
                            input_img=hologram,
                            inline=False,
                            get_filtered=True,
                            fx_ref=fx_ref,
                            fy_ref=fy_ref,
                            cut_size=[2108, 2163, 300, 150],
                            ).start()  # filtered处于空间域，获取频谱滤波结果，位于torch_gpu上
    return filtered
# ==========================================================================


# ==================== GPU 版 Gabor 能量计算 ====================
def generate_gabor_kernel(frequencies=None, thetas=None, kernel_size=55, device='cuda'):
    """
    生成一组 Gabor 卷积核，返回形状为 (num_filters, 1, kernel_size, kernel_size) 的张量。
    """
    if frequencies is None:
        frequencies = [0.08, 0.12, 0.18, 0.25]
    if thetas is None:
        thetas = [0.0, np.pi / 4, np.pi / 2, 3.0 * np.pi / 4]

    kernels = []
    for freq in frequencies:
        wavelength = 1.0 / max(freq, 1e-6)
        sigma = 0.56 * wavelength
        for theta in thetas:
            kernel_cv = cv2.getGaborKernel(
                ksize=(kernel_size, kernel_size),
                sigma=sigma,
                theta=float(theta),
                lambd=wavelength,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            kernel_torch = torch.from_numpy(kernel_cv).float().unsqueeze(0).unsqueeze(0)
            kernels.append(kernel_torch)
    return torch.cat(kernels, dim=0).to(device)


def compute_gabor_energy_gpu(gray_tensor, gabor_kernels):
    """
    在 GPU 上计算 Gabor 能量。
    """
    if gray_tensor.dim() == 2:
        gray_tensor = gray_tensor.unsqueeze(0).unsqueeze(0)
    elif gray_tensor.dim() == 3:
        gray_tensor = gray_tensor.unsqueeze(1)

    padding = gabor_kernels.shape[-1] // 2
    responses = torch.nn.functional.conv2d(
        gray_tensor, gabor_kernels, padding=padding, groups=1
    )

    energy = (responses ** 2).mean(dim=1).squeeze(0)
    return energy
# ================================================================


# ==================== 完全采用 Experiment 的 focus_and_z ====================
class focus_and_z():
    def __init__(self, filtered_tensor, dep_model):
        self.filtered_tensor = filtered_tensor
        self.dep_model = dep_model
        self.device = next(dep_model.parameters()).device

        # 预生成 Gabor 卷积核（初始化一次，避免每个 z 重复创建）
        self.gabor_kernels = generate_gabor_kernel(
            frequencies=[0.08, 0.12, 0.18, 0.25],
            thetas=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            kernel_size=55,
            device=self.device
        )

    def func(self, z, **kwargs):
        # 与 Experiment 一致：出图时 compress=1，用于 brent 优化时 compress=8
        if "get_img" in kwargs:
            compress = 12
        else:
            compress = 12

        propped = AngularBatch(lam=532e-9,
                               pix=0.098e-6,
                               z_initial=z,
                               input_img=self.filtered_tensor,
                               inline=True,
                               get_filtered=False,
                               compress=compress,
                               ).start()

        propped = torch.abs(propped)
        propped = (propped / propped.max() * 255).to(torch.float32)
        propped_3d = propped.unsqueeze(2).repeat(1, 1, 3)   # (H, W, 3)

        if "get_img" in kwargs:
            # ★ 拿到 score 的同时拿回 prob 概率图
            score, prob = focus_value(self.dep_model, propped_3d, scale=1,
                                      gabor_kernels=self.gabor_kernels,
                                      return_prob=True)

            # 重建图（灰度 -> RGB，与原来一致）
            propped_np = propped.cpu().numpy()
            propped_np = np.clip(propped_np, 0, 255).astype(np.uint8)
            propped_pil = Image.fromarray(propped_np, mode="L").convert('RGB')

            # ★ prob 概率图：按当前图自身 min-max 归一化到 [0,255]
            prob_np = prob.detach().cpu().float().numpy()
            p_min = prob_np.min()
            p_max = prob_np.max()
            if p_max - p_min > 1e-12:
                prob_np = (prob_np - p_min) / (p_max - p_min)
            else:
                prob_np = np.zeros_like(prob_np)
            prob_np = (prob_np * 255.0).astype(np.uint8)
            prob_pil = Image.fromarray(prob_np, mode="L")

            return propped_pil, prob_pil
        else:
            score = focus_value(self.dep_model, propped_3d, scale=1,
                                gabor_kernels=self.gabor_kernels)
            return score
# ==========================================================================


def safe_crop(image, left, top, right, bottom):
    width, height = image.size
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    if right <= left or bottom <= top:
        return image
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def compute_entropy_torch(values, bins=256):
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    min_val = values.min()
    max_val = values.max()
    if max_val <= min_val:
        return torch.tensor(0.0, device=values.device)
    hist = torch.histc(values, bins=bins, min=min_val, max=max_val)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -torch.sum(hist * torch.log2(hist))


def compute_kurtosis_torch(values):
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    mean = values.mean()
    var = values.var(unbiased=False)
    if var < 1e-12:
        return torch.tensor(0.0, device=values.device)
    fourth = ((values - mean) ** 4).mean()
    return fourth / (var ** 2) - 3.0


def compute_skewness_torch(values):
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    mean = values.mean()
    var = values.var(unbiased=False)
    if var < 1e-12:
        return torch.tensor(0.0, device=values.device)
    third = ((values - mean) ** 3).mean()
    return third / (var ** 1.5)


def focus_value(model, image, scale=1, gabor_kernels=None, score_type='std',
                keep_ratio=0.95, gradient_weight=0.3, return_prob=False):
    model.eval()
    with torch.no_grad():
        # RGB -> 灰度
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        # ★ 不再 /255.0 —— 与 BSDS_RCFLoader channel=1 的输出范围 [0, 255] 保持一致

        # 下采样
        h, w = gray.shape
        new_h, new_w = h // scale, w // scale
        if new_h < 1 or new_w < 1:
            new_h, new_w = 1, 1
        gray = F.interpolate(
            gray.view(1, 1, h, w),
            size=(new_h, new_w), mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)

        # 模型推理
        input_tensor = gray.unsqueeze(0).unsqueeze(0)
        prob = model(input_tensor).squeeze()   # (H, W) 概率图

        BORDER = 20
        # ★ 边缘置零：排除 zero-padding 边界效应
        if BORDER > 0:
            prob = prob.clone()
            prob[:BORDER, :] = 0
            prob[-BORDER:, :] = 0
            prob[:, :BORDER] = 0
            prob[:, -BORDER:] = 0

        # 统一返回包装：需要 prob 时返回 (score, prob)
        def _ret(score):
            return (score, prob) if return_prob else score

        if score_type != 'gabor_weighted':
            non_zero_mask = prob > 0
            non_zero_vals = prob[non_zero_mask]
            if non_zero_vals.numel() == 0:
                return _ret(0.0)

        # ---- 处理不同的分数类型 ----
        if score_type == 'gini':
            return _ret(prob.std().item() / prob.mean().item())

        if score_type == 'variance':
            return _ret(non_zero_vals.var().item())

        elif score_type == 'std':
            return _ret(non_zero_vals.std().item())

        elif score_type == 'mean':
            return _ret(non_zero_vals.mean().item())

        elif score_type == 'dirichlet_uncertainty':
            uncertainty = 2.0 * torch.min(non_zero_vals, 1.0 - non_zero_vals)
            return _ret(uncertainty.mean().item())

        elif score_type == 'entropy':
            ent = compute_entropy_torch(non_zero_vals, bins=256)
            return _ret(-ent.item())

        elif score_type == 'kurtosis':
            return _ret(compute_kurtosis_torch(non_zero_vals).item())

        elif score_type == 'skewness':
            return _ret(compute_skewness_torch(non_zero_vals).item())

        elif score_type == 'gradient_magnitude':
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                                   device=prob.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
                                   device=prob.device).view(1, 1, 3, 3)
            grad_x = F.conv2d(prob.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
            grad_y = F.conv2d(prob.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze()
            grad_vals = grad_mag[non_zero_mask]
            if grad_vals.numel() == 0:
                return _ret(0.0)
            flat = grad_vals.flatten()
            k = max(1, int(flat.numel() * keep_ratio))
            sorted_vals, _ = torch.sort(flat)
            return _ret(sorted_vals[:k].mean().item())

        else:  # 'gabor_weighted' 或默认
            if gabor_kernels is None:
                raise ValueError("gabor_kernels must be provided for gabor_weighted")
            gabor_energy = compute_gabor_energy_gpu(gray, gabor_kernels)
            weighted = prob * gabor_energy

            w_flat = weighted.flatten()
            e_flat = gabor_energy.flatten()
            k = max(1, int(w_flat.numel() * keep_ratio))
            w_sorted, _ = torch.sort(w_flat)
            e_sorted, _ = torch.sort(e_flat)
            w_keep = w_sorted[:k]
            e_keep = e_sorted[:k]
            weighted_mean = w_keep.mean()
            energy_mean = e_keep.mean() + 1e-8
            return _ret((weighted_mean / energy_mean).item())


def main(root, save_dir, lb, ub, model=None, fx_ref=None, fy_ref=None,
         save_curve=False, curve_dir=None, save_prob=True):
    """
    单张全息图聚焦距离预测。
    save_prob=True 时会把 prob 概率图保存到 {save_dir}/prob/ 下。
    """
    dlep_model = model

    filtered = initial(root=root, fx_ref=fx_ref, fy_ref=fy_ref)
    focus_z = focus_and_z(filtered, dlep_model)

    # ---- 可选：保存聚焦分数曲线 ----
    if save_curve:
        zs = np.linspace(lb, ub, num=50)
        scores = []
        for z_val in zs:
            score = focus_z.func(z_val)
            if isinstance(score, torch.Tensor):
                score = score.item()
            scores.append(score)

        plt.figure(figsize=(10, 5))
        plt.plot(zs, scores, marker='o', markersize=3, linestyle='-')
        plt.xlabel('Reconstruction distance z (m)')
        plt.ylabel('Focus score')
        plt.title(f'Focus curve: {Path(root).stem}')
        plt.grid(True)

        if curve_dir:
            os.makedirs(curve_dir, exist_ok=True)
            curve_path = os.path.join(curve_dir, f"{Path(root).stem}_curve.png")
        else:
            curve_path = os.path.join(save_dir, f"{Path(root).stem}_curve.png")

        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        # print(f"  已保存曲线: {curve_path}")

    def max_func(z, scale=None):
        score = focus_z.func(z, in_brent=True, scale=scale)
        return -score.item() if isinstance(score, torch.Tensor) else -score

    finder = PeakFinder(max_func, lb=lb, ub=ub, max_evals=200, target='min',
                        precision_factor=5, initial_points_factor=5)
    z_result = finder.find_optimum()

    # ---- ★ 用最优 z 重算一次，保存重建图 + prob 概率图 ----
    img_result, prob_result = focus_z.func(z_result, in_brent=True, get_img=True)
    stem = Path(root).stem

    os.makedirs(save_dir, exist_ok=True)
    img_result.save(os.path.join(save_dir, f"{stem}.png"))

    if save_prob:
        prob_dir = os.path.join(save_dir, "prob")
        os.makedirs(prob_dir, exist_ok=True)
        prob_path = os.path.join(prob_dir, f"{stem}_prob.png")
        prob_result.save(prob_path)
        # print(f"  已保存 prob: {prob_path}")

    return z_result


def batch_main(root_dir, save_dir, test_csv=None, save_curves=False, curves_subdir='curves'):
    # ★ 与 Experiment 完全一致的 z 范围
    lb = 0
    ub = 0.00060

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [
        f for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f)) and
           os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS
    ]
    image_files = natsorted(image_files)

    os.makedirs(save_dir, exist_ok=True)

    curve_dir = None
    if save_curves:
        curve_dir = os.path.join(save_dir, curves_subdir)
        os.makedirs(curve_dir, exist_ok=True)

    ground_truth = {}
    fx = {}
    fy = {}
    if test_csv and os.path.exists(test_csv):
        print(f"加载测试数据: {test_csv}")
        with open(test_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hologram_name = row['hologram_name']
                z_gt = float(row['z'])
                ground_truth[hologram_name] = z_gt
                fx_ref = float(row['fx_ref']) if 'fx_ref' in row else None
                fy_ref = float(row['fy_ref']) if 'fy_ref' in row else None
                fx[hologram_name] = fx_ref
                fy[hologram_name] = fy_ref

    results = []
    results_csv = os.path.join(save_dir, "focusing_results.csv")
    fieldnames = ['hologram_name', 'z_gt', 'z_pred', 'abs_error', 'rel_error']

    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # ========== 加载 GaborLogistic 模型 ==========
    # model_path = r'F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing\edge_prediction\runs_gabor\20260916_150908\model_loss_0.024620.pth'
    model_path = r'F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing\edge_prediction_gabor\runs_gabor\20260916_150908\model_loss_0.022564.pth'
    model = load_model(model_path, device=str(device))
    model.to(device)
    model.eval()
    # ========== 加载结束 ==========

    pbar = tqdm(image_files, desc="Processing holograms", unit="image")

    total_abs_error = 0.0
    total_rel_error = 0.0
    valid_rel_count = 0
    processed_count = 0

    for image_file in pbar:
        image_path = os.path.join(root_dir, image_file)
        pbar.set_description(f"Processing: {image_file}")

        fx_ref = None
        fy_ref = None
        if image_file in fx:
            fx_ref = fx[image_file]
            fy_ref = fy[image_file]

        z_pred = main(lb=lb, ub=ub, root=image_path, save_dir=save_dir,
                      model=model, fx_ref=fx_ref, fy_ref=fy_ref,
                      save_curve=save_curves, curve_dir=curve_dir,
                      save_prob=True)

        result_row = {
            'hologram_name': image_file,
            'z_gt': '',
            'z_pred': z_pred,
            'abs_error': '',
            'rel_error': ''
        }

        if image_file in ground_truth:
            z_gt = ground_truth[image_file]
            abs_error = abs(z_pred - z_gt)
            rel_error = abs_error / abs(ub - lb) if abs(ub - lb) > 1e-9 else float('inf')

            processed_count += 1
            total_abs_error += abs_error
            if not math.isinf(rel_error):
                total_rel_error += rel_error
                valid_rel_count += 1

            avg_abs_error = total_abs_error / processed_count
            avg_rel_error = (total_rel_error / valid_rel_count) * 100 if valid_rel_count > 0 else float('nan')

            result_row.update({
                'z_gt': z_gt,
                'abs_error': abs_error,
                'rel_error': rel_error
            })

            results.append(result_row.copy())

            pbar.set_postfix({
                'z_gt': f"{z_gt:.7f}",
                'z_pred': f"{z_pred:.7f}",
                'abs_err': f"{abs_error:.7f}",
                'rel_err(%)': f"{rel_error * 100:.2f}" if not math.isinf(rel_error) else "Inf",
                'avg_abs': f"{avg_abs_error:.7f}",
                'avg_rel(%)': f"{avg_rel_error:.2f}" if not math.isnan(avg_rel_error) else "NaN"
            })
        else:
            results.append(result_row.copy())
            pbar.set_postfix({
                'z_pred': f"{z_pred:.7f}"
            })

        with open(results_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result_row)

    if results:
        print("\n" + "=" * 50)
        print("聚焦距离预测误差统计")
        print("=" * 50)

        abs_errors = [r['abs_error'] for r in results if r['abs_error'] != '']
        rel_errors = [r['rel_error'] for r in results
                      if r['rel_error'] != '' and not math.isinf(r['rel_error'])]

        if abs_errors:
            avg_abs_error = sum(abs_errors) / len(abs_errors)
            min_abs_error = min(abs_errors)
            max_abs_error = max(abs_errors)
        else:
            avg_abs_error = min_abs_error = max_abs_error = 0

        if rel_errors:
            avg_rel_error = sum(rel_errors) / len(rel_errors) * 100
            min_rel_error = min(rel_errors) * 100
            max_rel_error = max(rel_errors) * 100
        else:
            avg_rel_error = min_rel_error = max_rel_error = float('nan')

        print(f"处理图像数量: {len(results)}")
        print(f"平均绝对误差: {avg_abs_error:.7f} m")
        print(f"最小绝对误差: {min_abs_error:.7f} m")
        print(f"最大绝对误差: {max_abs_error:.7f} m")

        if not math.isnan(avg_rel_error):
            print(f"平均相对误差: {avg_rel_error:.2f}%")
            print(f"最小相对误差: {min_rel_error:.2f}%")
            print(f"最大相对误差: {max_rel_error:.2f}%")

        results_csv = os.path.join(save_dir, "focusing_results.csv")
        with open(results_csv, 'w', newline='') as f:
            fieldnames = ['hologram_name', 'z_gt', 'z_pred', 'abs_error', 'rel_error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n详细结果已保存至: {results_csv}")


def test_focusing_accuracy(root_dir, test_csv, save_dir):
    with open(test_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            print("测试CSV文件为空")
            return

    print(f"开始测试聚焦算法准确度")
    print(f"测试数据集: {test_csv}")
    print(f"图像目录: {root_dir}")
    print(f"保存目录: {save_dir}")

    batch_main(root_dir, save_dir, test_csv=test_csv,
               save_curves=False, curves_subdir='curves')


if __name__ == "__main__":
    # ★ 数据集路径完全替换为 Experiment 的数据集
    test_focusing_accuracy(
        root_dir=r'F:\dongjiayao\Data\AutoFocusDatabase\hologram_amplitude\hologram',
        save_dir=r'F:\dongjiayao\Data\COCO\val\article\IEEE TIM\revise1\exp_test\ours_gabor_22564',
        test_csv=r'F:\dongjiayao\Data\AutoFocusDatabase\AutoFocusDatabase_amplitude.csv'
    )