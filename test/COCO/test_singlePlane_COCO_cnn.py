#!/usr/bin/env python
# coding=utf-8
import os
import argparse
from pathlib import Path

# ★ 显存优化：必须在 import torch 之前设置分配器，缓解多尺寸碎片。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── 必须在所有 torch/cuda import 之前设置 ──────────────────────
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--gpu', default='3', type=str)
_pre_parser.add_argument('--device', default='cuda', type=str,
                         choices=['auto', 'cuda', 'cpu'])
_pre_args, _ = _pre_parser.parse_known_args()

if _pre_args.device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _FORCE_DEVICE = 'cpu'
elif _pre_args.device == 'cuda':
    os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu
    _FORCE_DEVICE = 'cuda'
else:
    os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu
    _FORCE_DEVICE = 'auto'
# ────────────────────────────────────────────────────────────────

import csv
import math
import re
import time
import random
import cv2
import numpy as np
import torch
from contextlib import nullcontext
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
from main_rotate_deep_CNN import GaborLogistic, load_model
from peak_finder.peakfinder import PeakFinder
from peak_finder.brent import brent_method, multi_start_brent

from edge_prediction.predict import calculate_brightness_concentration
from edge_prediction.dataloader.data_loader import convert_to_rgb, prepare_image_PIL, prepare_image_cv2, prepare_image_torch_gpu
from edge_prediction.model.DLEP import DLEP, DLEPLite, DLEPLite1, DLEPLite2
from peak_finder.traverse import traverse

from propagate.angular_torch import AngularBatch

# 是否打印单步细分耗时（默认关闭，需要时设 PROFILE=1）
_PROFILE = os.environ.get("PROFILE", "0") == "1"

# ★ 缓存优化：同一时刻最多保留 FOCUS_CACHE_MAX 个 focus_and_z 实例。
_FOCUS_OBJECT_CACHE = {}
_FOCUS_CACHE_MAX = max(1, int(os.environ.get("FOCUS_CACHE_MAX", "1")))


# ── 设备选择辅助 ────────────────────────────────────────────────
def _pick_device():
    if _FORCE_DEVICE == 'cpu':
        return torch.device('cpu')
    if _FORCE_DEVICE == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("指定 --device cuda，但当前环境无可用 GPU")
        return torch.device('cuda:0')
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


DEVICE = _pick_device()
print(f"[device] 使用 {DEVICE}")


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _autocast():
    if DEVICE.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()
# ────────────────────────────────────────────────────────────────


def set_deterministic():
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

# ============================================================
# ★ 加速开关：cudnn autotune + TF32
# ============================================================
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
# ============================================================


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行耗时: {end - start:.4f} 秒")
        return result
    return wrapper


def initial(root, fx_ref=None, fy_ref=None):
    hologram = Image.open(root).convert('L')
    filtered = AngularBatch(lam=532e-9,
                            pix=1e-6,
                            z_initial=0.00030,
                            input_img=hologram,
                            inline=False,
                            get_filtered=True,
                            fx_ref=fx_ref,
                            fy_ref=fy_ref
                            ).start()
    if isinstance(filtered, torch.Tensor):
        filtered = filtered.cpu()
    return filtered


# ==================== GPU 版 Gabor 能量计算 ====================
def generate_gabor_kernel(frequencies=None, thetas=None, kernel_size=55, device='cuda'):
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


def gini_coefficient(prob, eps=1e-8):
    if prob.dim() == 1:
        prob = prob.unsqueeze(0)
    x = prob.flatten(1)
    x = torch.clamp(x, min=0)
    x_sorted, _ = torch.sort(x, dim=1)
    n = x.size(1)
    index = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    gini = (2 * (index * x_sorted).sum(dim=1) /
            (n * x_sorted.sum(dim=1) + eps)) - (n + 1) / n
    return gini


def tamura_contrast(prob, eps=1e-8):
    if prob.dim() == 1:
        prob = prob.unsqueeze(0)
    x = prob.flatten(1)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    std = torch.sqrt(var + eps)
    mu4 = ((x - mean) ** 4).mean(dim=1, keepdim=True)
    alpha4 = mu4 / (var ** 2 + eps)
    contrast = std / (alpha4 ** 0.25 + eps)
    return contrast.squeeze(1)


def compute_entropy_torch(values, bins=256):
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    min_val = values.min(); max_val = values.max()
    if max_val <= min_val:
        return torch.tensor(0.0, device=values.device)
    hist = torch.histc(values, bins=bins, min=min_val, max=max_val)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -torch.sum(hist * torch.log2(hist))


# ============================================================================
# ★ CUDA Graph 版 focus_and_z（B=1，结构与你原来验证过的一致）
#   - 传播 + CNN + score 全部 capture 进 graph
#   - 每张图只更新输入频谱，保留 u1_cached 存储地址
#   - release()：释放 graph / 静态张量 / 网格，归还显存
# ============================================================================
class focus_and_z():
    _gabor_cache = {}

    def __init__(self, filtered_tensor, dep_model, use_cuda_graph=True):
        self.device = DEVICE
        self.filtered_tensor = filtered_tensor.to(self.device)
        self.dep_model = dep_model

        h, w = self.filtered_tensor.shape
        self.height, self.width = h, w
        pix = 1e-6
        lam = 532e-9
        self.lam = lam
        self.k0 = 2.0 * torch.pi / lam

        fx = torch.linspace(-1 / (2 * pix), 1 / (2 * pix), w, device=self.device)
        fy = torch.linspace(-1 / (2 * pix), 1 / (2 * pix), h, device=self.device)
        self.FX, self.FY = torch.meshgrid(fx, fy, indexing='xy')

        temp = 1 - ((lam * self.FX) ** 2 + (lam * self.FY) ** 2)
        temp[temp < 0] = 0
        self.sqrt_temp = torch.sqrt(temp)

        self.sqrt_temp_shifted = torch.fft.fftshift(self.sqrt_temp)
        self.mask_keep = (self.sqrt_temp_shifted != 0).to(self.sqrt_temp_shifted.dtype)

        self.u1_cached = torch.fft.fft2(torch.fft.fftshift(self.filtered_tensor))

        gkey = (55, str(self.device))
        if gkey not in focus_and_z._gabor_cache:
            focus_and_z._gabor_cache[gkey] = generate_gabor_kernel(
                frequencies=[0.08, 0.12, 0.18, 0.25],
                thetas=[0.0, np.pi/4, np.pi/2, 3*np.pi/4],
                kernel_size=55,
                device=self.device
            )
        self.gabor_kernels = focus_and_z._gabor_cache[gkey]

        self._graph = None
        self._z_static = None
        self._prob_static = None
        self._propped_abs_static = None
        self._score_static = None
        self._timing = {
            'setup_ms': 0.0,
            'warmup_ms': 0.0,
            'capture_ms': 0.0,
            'capture_total_ms': 0.0,
        }
        self.use_cuda_graph = (
            use_cuda_graph
            and self.device.type == 'cuda'
            and torch.cuda.is_available()
        )

        if self.use_cuda_graph:
            self._try_capture()

    # ------------------------------------------------------------------
    def release(self):
        self._prob_static = None
        self._propped_abs_static = None
        self._score_static = None
        self._z_static = None
        self._graph = None

        self.u1_cached = None
        self.filtered_tensor = None
        self.FX = None
        self.FY = None
        self.sqrt_temp = None
        self.sqrt_temp_shifted = None
        self.mask_keep = None
        self.gabor_kernels = None
        self.dep_model = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @torch.no_grad()
    def update_input(self, filtered_tensor):
        if tuple(filtered_tensor.shape) != (self.height, self.width):
            raise ValueError(
                f'复用 focus_and_z 时图像尺寸必须一致：期望 '
                f'{(self.height, self.width)}，实际 {tuple(filtered_tensor.shape)}'
            )
        filtered_device = filtered_tensor.to(
            device=self.device, dtype=self.filtered_tensor.dtype
        )
        self.filtered_tensor.copy_(filtered_device)

        shifted = torch.fft.fftshift(self.filtered_tensor)
        try:
            torch.fft.fft2(shifted, out=self.u1_cached)
        except Exception:
            new_spectrum = torch.fft.fft2(shifted)
            if self.u1_cached.shape != new_spectrum.shape:
                raise RuntimeError('更新输入时频谱尺寸发生变化，不能复用 CUDA Graph。')
            self.u1_cached.copy_(new_spectrum)
            del new_spectrum
        del shifted

    # ------------------------------------------------------------------
    def _forward_capture(self, z_t):
        g_shifted = torch.exp(1j * self.k0 * z_t * self.sqrt_temp_shifted)
        g_shifted = g_shifted * self.mask_keep
        u2 = self.u1_cached * g_shifted
        propped = torch.fft.ifftshift(torch.fft.ifft2(u2))

        propped_abs = torch.abs(propped)
        m = torch.clamp_min(propped_abs.max(), 1e-12)
        propped_abs = propped_abs / m * 255.0
        propped_abs = propped_abs.to(torch.float32)

        input_tensor = propped_abs.unsqueeze(0).unsqueeze(0)
        prob = self.dep_model(input_tensor).squeeze()
        prob = prob.float()

        BORDER = 20
        prob[:BORDER, :] = 0
        prob[-BORDER:, :] = 0
        prob[:, :BORDER] = 0
        prob[:, -BORDER:] = 0

        score = prob.var()
        return prob, propped_abs, score

    # ------------------------------------------------------------------
    def _try_capture(self):
        total_start = time.perf_counter()
        try:
            self._z_static = torch.zeros((), device=self.device, dtype=torch.float32)

            warmup_start = time.perf_counter()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        self._forward_capture(self._z_static)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self._timing['warmup_ms'] = (time.perf_counter() - warmup_start) * 1000.0

            capture_start = time.perf_counter()
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.cuda.graph(graph):
                        prob_out, propped_abs_out, score_out = self._forward_capture(self._z_static)
            torch.cuda.synchronize()
            self._timing['capture_ms'] = (time.perf_counter() - capture_start) * 1000.0

            self._graph = graph
            self._prob_static = prob_out
            self._propped_abs_static = propped_abs_out
            self._score_static = score_out
            self._timing['capture_total_ms'] = (time.perf_counter() - total_start) * 1000.0
            tqdm.write(
                '    [focus_and_z] CUDA Graph 捕获成功：'
                f"warmup={self._timing['warmup_ms']:.1f} ms, "
                f"capture={self._timing['capture_ms']:.1f} ms, "
                f"合计={self._timing['capture_total_ms']:.1f} ms"
            )
        except Exception as e:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            self._timing['capture_total_ms'] = (time.perf_counter() - total_start) * 1000.0
            tqdm.write(f"    [focus_and_z] CUDA Graph 捕获失败: {e}")
            tqdm.write("    [focus_and_z] 已回退到 eager 模式")
            self._graph = None
            self.use_cuda_graph = False
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward_eager(self, z_t):
        g_shifted = torch.exp(1j * self.k0 * z_t * self.sqrt_temp_shifted)
        g_shifted = g_shifted * self.mask_keep
        u2 = self.u1_cached * g_shifted
        propped = torch.fft.ifftshift(torch.fft.ifft2(u2))

        propped_abs = torch.abs(propped)
        m = torch.clamp_min(propped_abs.max(), 1e-12)
        propped_abs = propped_abs / m * 255.0
        propped_abs = propped_abs.to(torch.float32)

        input_tensor = propped_abs.unsqueeze(0).unsqueeze(0)
        with _autocast():
            prob = self.dep_model(input_tensor).squeeze()
        prob = prob.float()

        BORDER = 20
        prob[:BORDER, :] = 0
        prob[-BORDER:, :] = 0
        prob[:, :BORDER] = 0
        prob[:, -BORDER:] = 0

        return prob, propped_abs

    # ------------------------------------------------------------------
    def _compute_score(self, prob, score_type='variance', keep_ratio=0.95):
        if score_type == 'variance':
            return prob.var()
        elif score_type == 'std':
            nz = prob[prob > 0]
            return nz.std()
        elif score_type == 'gini':
            return prob.std() / prob.mean()
        elif score_type == 'mean':
            nz = prob[prob > 0]
            return nz.mean()
        elif score_type == 'entropy':
            nz = prob[prob > 0]
            return -compute_entropy_torch(nz, bins=256)
        elif score_type == 'gradient_magnitude':
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=prob.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32, device=prob.device).view(1, 1, 3, 3)
            gx = F.conv2d(prob.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
            gy = F.conv2d(prob.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)
            gm = torch.sqrt(gx**2 + gy**2).squeeze()
            flat = gm.flatten()
            k = max(1, int(flat.numel() * keep_ratio))
            sv, _ = torch.sort(flat)
            return sv[:k].mean()
        else:
            return prob.var()

    # ------------------------------------------------------------------
    def func(self, z, **kwargs):
        if isinstance(z, torch.Tensor):
            z_val = float(z.detach().cpu())
        else:
            z_val = float(z)

        if self._graph is not None:
            self._z_static.fill_(z_val)
            self._graph.replay()

            if "get_img" in kwargs:
                propped_np = self._propped_abs_static.detach().clone().cpu().numpy()
                propped_np = np.clip(propped_np, 0, 255).astype(np.uint8)
                propped_pil = Image.fromarray(propped_np, mode="L").convert('RGB')

                prob_np = self._prob_static.detach().clone().cpu().float().numpy()
                p_min = prob_np.min(); p_max = prob_np.max()
                if p_max - p_min > 1e-12:
                    prob_np = (prob_np - p_min) / (p_max - p_min)
                else:
                    prob_np = np.zeros_like(prob_np)
                prob_np = (prob_np * 255.0).astype(np.uint8)
                prob_pil = Image.fromarray(prob_np, mode="L")
                return propped_pil, prob_pil
            else:
                return self._score_static.clone()
        else:
            z_t = torch.tensor(z_val, device=self.device, dtype=torch.float32)
            prob, propped_abs = self._forward_eager(z_t)

            if "get_img" in kwargs:
                propped_np = propped_abs.detach().cpu().numpy()
                propped_np = np.clip(propped_np, 0, 255).astype(np.uint8)
                propped_pil = Image.fromarray(propped_np, mode="L").convert('RGB')

                prob_np = prob.detach().cpu().float().numpy()
                p_min = prob_np.min(); p_max = prob_np.max()
                if p_max - p_min > 1e-12:
                    prob_np = (prob_np - p_min) / (p_max - p_min)
                else:
                    prob_np = np.zeros_like(prob_np)
                prob_np = (prob_np * 255.0).astype(np.uint8)
                prob_pil = Image.fromarray(prob_np, mode="L")
                return propped_pil, prob_pil
            else:
                return self._compute_score(prob)


# 兼容旧调用
def focus_value(model, image, scale=1, gabor_kernels=None, score_type='variance',
                keep_ratio=0.95, gradient_weight=0.3, return_prob=False):
    model.eval()
    with torch.inference_mode():
        with _autocast():
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            h, w = gray.shape
            if scale != 1:
                new_h, new_w = h // scale, w // scale
                if new_h < 1 or new_w < 1:
                    new_h, new_w = 1, 1
                gray = F.interpolate(gray.view(1, 1, h, w),
                                     size=(new_h, new_w), mode='bilinear', align_corners=False
                                     ).squeeze(0).squeeze(0)
            input_tensor = gray.unsqueeze(0).unsqueeze(0)
            prob = model(input_tensor).squeeze()
        prob = prob.float()

        BORDER = 20
        if BORDER > 0:
            prob = prob.clone()
            prob[:BORDER, :] = 0
            prob[-BORDER:, :] = 0
            prob[:, :BORDER] = 0
            prob[:, -BORDER:] = 0

        if return_prob:
            return prob.var(), prob
        return prob.var()


# ★ 缓存淘汰：切换尺寸/模型时释放旧实例
def _prepare_focus_cache_slot(keep_key=None):
    for k in list(_FOCUS_OBJECT_CACHE.keys()):
        if k == keep_key:
            continue
        obj = _FOCUS_OBJECT_CACHE.pop(k)
        try:
            obj.release()
        except Exception:
            pass
        del obj
    while len(_FOCUS_OBJECT_CACHE) >= _FOCUS_CACHE_MAX:
        k = next(iter(_FOCUS_OBJECT_CACHE))
        if k == keep_key:
            break
        obj = _FOCUS_OBJECT_CACHE.pop(k)
        try:
            obj.release()
        except Exception:
            pass
        del obj


def main(root, save_dir, lb, ub, model=None, fx_ref=None, fy_ref=None,
         save_curve=False, curve_dir=None, save_prob=False, use_cuda_graph=True,
         verbose=True):
    dlep_model = model
    t_main_start = time.perf_counter()

    # 步骤 1：预处理
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    filtered = initial(root=root, fx_ref=fx_ref, fy_ref=fy_ref)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_preprocess_ms = (time.perf_counter() - t0) * 1000.0

    # 步骤 2：复用 focus_and_z / CUDA Graph（同尺寸同模型）
    global _FOCUS_OBJECT_CACHE
    cache_key = (id(dlep_model), str(DEVICE), tuple(filtered.shape), bool(use_cuda_graph))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache_hit = cache_key in _FOCUS_OBJECT_CACHE
    if cache_hit:
        focus_z = _FOCUS_OBJECT_CACHE[cache_key]
        focus_z.update_input(filtered)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_input_update_ms = (time.perf_counter() - t0) * 1000.0
        t_setup_ms = 0.0
        t_warmup_ms = 0.0
        t_capture_ms = 0.0
        t_capture_total_ms = 0.0
    else:
        _prepare_focus_cache_slot(keep_key=None)
        focus_z = focus_and_z(filtered, dlep_model, use_cuda_graph=use_cuda_graph)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_constructor_ms = (time.perf_counter() - t0) * 1000.0
        _FOCUS_OBJECT_CACHE[cache_key] = focus_z
        t_input_update_ms = 0.0
        t_warmup_ms = focus_z._timing.get('warmup_ms', 0.0)
        t_capture_ms = focus_z._timing.get('capture_ms', 0.0)
        t_capture_total_ms = focus_z._timing.get('capture_total_ms', 0.0)
        t_setup_ms = max(0.0, t_constructor_ms - t_capture_total_ms)

    del filtered

    t_profile_ms = 0.0
    if _PROFILE:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        focus_z.profile_once(z=0.0005)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_profile_ms = (time.perf_counter() - t0) * 1000.0

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

    # 步骤 3：寻峰（完全保持原样）
    def max_func(z, scale=None):
        score = focus_z.func(z, in_brent=True, scale=scale)
        if isinstance(score, torch.Tensor):
            return -float(score.item())
        return -float(score)

    n_z = int((ub - lb) / 0.000005) + 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    # finder = PeakFinder(max_func, lb=lb, ub=ub, max_evals=200, target='min',
    #                     precision_factor=5, initial_points_factor=5)
    # z_result = finder.find_optimum()

    z_result = traverse(max_func, lb=lb, ub=ub, step=0.000005)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_traverse_ms = (time.time() - t0) * 1000.0

    # 步骤 4：get_img
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    img_result, prob_result = focus_z.func(z_result, in_brent=True, get_img=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_getimg_ms = (time.time() - t0) * 1000.0

    stem = Path(root).stem
    t_save_start = time.perf_counter()
    os.makedirs(save_dir, exist_ok=True)
    if save_prob:
        prob_dir = os.path.join(save_dir, "prob")
        os.makedirs(prob_dir, exist_ok=True)
        prob_path = os.path.join(prob_dir, f"{stem}_prob.png")
        prob_result.save(prob_path)
    t_save_ms = (time.perf_counter() - t_save_start) * 1000.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_main_ms = (time.perf_counter() - t_main_start) * 1000.0

    if verbose:
        tqdm.write("    " + "=" * 76)
        tqdm.write(f"    [单张耗时细分]   z_pred = {z_result:.7f}   device={DEVICE}")
        tqdm.write("    " + "-" * 76)
        tqdm.write(f"      预处理 initial            : {t_preprocess_ms:8.1f} ms")
        if cache_hit:
            tqdm.write(f"      更新输入 + FFT            : {t_input_update_ms:8.1f} ms")
            tqdm.write("      CUDA Graph 初始化/捕获    : 已复用（0 ms）")
        else:
            tqdm.write(f"      张量/频率网格初始化       : {t_setup_ms:8.1f} ms（估算）")
            tqdm.write(f"      CUDA Graph warmup         : {t_warmup_ms:8.1f} ms")
            tqdm.write(f"      CUDA Graph capture        : {t_capture_ms:8.1f} ms")
            tqdm.write(f"      初始化及捕获合计          : {t_capture_total_ms:8.1f} ms")
        tqdm.write(f"      聚焦搜索 traverse          : {t_traverse_ms:8.1f} ms")
        tqdm.write(f"      最优位置图像重建/取图      : {t_getimg_ms:8.1f} ms")
        tqdm.write(f"      保存结果                  : {t_save_ms:8.1f} ms")
        if t_profile_ms > 0:
            tqdm.write(f"      额外 PROFILE 开销         : {t_profile_ms:8.1f} ms")
        tqdm.write(f"      main 内部总耗时           : {t_main_ms:8.1f} ms")
        tqdm.write("    " + "=" * 76)

    return z_result


def _bucket_by_size(root_dir, image_files):
    """★ 按图像尺寸分桶，同尺寸连续处理，避免每换一种尺寸就重新 capture。"""
    buckets = {}
    for f in image_files:
        path = os.path.join(root_dir, f)
        try:
            with Image.open(path) as im:
                key = im.size  # (w, h)
        except Exception:
            key = None
        buckets.setdefault(key, []).append(f)
    ordered = []
    for k in sorted(buckets.keys(), key=lambda x: (x is None, x)):
        ordered.extend(buckets[k])
    return ordered


def batch_main(root_dir, save_dir, test_csv=None, save_curves=False, curves_subdir='curves',
               use_cuda_graph=True):
    lb = 0.00000
    ub = 0.00100

    device = DEVICE

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [
        f for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f)) and
           os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS
    ]
    image_files = natsorted(image_files)

    # ★ 分桶：同尺寸连续处理，capture 每种尺寸只发生一次
    image_files = _bucket_by_size(root_dir, image_files)

    os.makedirs(save_dir, exist_ok=True)

    curve_dir = None
    if save_curves:
        curve_dir = os.path.join(save_dir, curves_subdir)
        os.makedirs(curve_dir, exist_ok=True)

    ground_truth = {}; fx = {}; fy = {}
    if test_csv and os.path.exists(test_csv):
        print(f"加载测试数据: {test_csv}")
        with open(test_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hologram_name = row['hologram_name']
                ground_truth[hologram_name] = float(row['z'])
                fx[hologram_name] = float(row['fx_ref']) if 'fx_ref' in row else None
                fy[hologram_name] = float(row['fy_ref']) if 'fy_ref' in row else None

    results = []
    results_csv = os.path.join(save_dir, "focusing_results.csv")
    fieldnames = ['hologram_name', 'z_gt', 'z_pred', 'abs_error', 'rel_error', 'time_ms']

    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    model_path = r'F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing\edge_prediction_gabor\runs_cnn\20260916_123348\model_loss_0.021667.pth'
    model = load_model(model_path, device=str(device))
    model.to(device)
    model.eval()

    pbar = tqdm(image_files, desc="Processing holograms", unit="image")

    total_abs_error = 0.0
    total_rel_error = 0.0
    valid_rel_count = 0
    processed_count = 0
    total_time_ms = 0.0
    time_count = 0

    for idx, image_file in enumerate(pbar):
        image_path = os.path.join(root_dir, image_file)
        pbar.set_description(f"Processing: {image_file}")

        fx_ref = fx.get(image_file, None)
        fy_ref = fy.get(image_file, None)

        verbose = True

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()
        z_pred = main(lb=lb, ub=ub, root=image_path, save_dir=save_dir,
                      model=model, fx_ref=fx_ref, fy_ref=fy_ref,
                      save_curve=save_curves, curve_dir=curve_dir,
                      save_prob=False, use_cuda_graph=use_cuda_graph,
                      verbose=verbose)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.time()
        elapsed_ms = (t_end - t_start) * 1000.0

        total_time_ms += elapsed_ms
        time_count += 1
        avg_time_ms = total_time_ms / time_count

        result_row = {
            'hologram_name': image_file,
            'z_gt': '',
            'z_pred': z_pred,
            'abs_error': '',
            'rel_error': '',
            'time_ms': f"{elapsed_ms:.2f}"
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

            result_row.update({'z_gt': z_gt, 'abs_error': abs_error, 'rel_error': rel_error})
            results.append(result_row.copy())

            pbar.set_postfix({
                'z_gt': f"{z_gt:.7f}",
                'z_pred': f"{z_pred:.7f}",
                'abs_err': f"{abs_error:.7f}",
                'rel_err(%)': f"{rel_error * 100:.2f}" if not math.isinf(rel_error) else "Inf",
                'avg_abs': f"{avg_abs_error:.7f}",
                'avg_rel(%)': f"{avg_rel_error:.2f}" if not math.isnan(avg_rel_error) else "NaN",
                't(ms)': f"{elapsed_ms:.0f}",
                'avg_t(ms)': f"{avg_time_ms:.0f}"
            })
        else:
            results.append(result_row.copy())
            pbar.set_postfix({
                'z_pred': f"{z_pred:.7f}",
                't(ms)': f"{elapsed_ms:.0f}",
                'avg_t(ms)': f"{avg_time_ms:.0f}"
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

        if time_count > 0:
            times_ms = [float(r['time_ms']) for r in results if r.get('time_ms', '') != '']
            if times_ms:
                print(f"总计算时间: {total_time_ms:.2f} ms")
                print(f"平均每张图像计算时间: {avg_time_ms:.2f} ms")
                print(f"最小单张耗时: {min(times_ms):.2f} ms")
                print(f"最大单张耗时: {max(times_ms):.2f} ms")

        print(f"\n详细结果已保存至: {results_csv}")


def test_focusing_accuracy(root_dir, test_csv, save_dir):
    with open(test_csv, 'r') as f:
        rows = list(csv.DictReader(f))
        if not rows:
            print("测试CSV文件为空")
            return

    print(f"开始测试聚焦算法准确度")
    print(f"测试数据集: {test_csv}")
    print(f"图像目录: {root_dir}")
    print(f"保存目录: {save_dir}")

    use_cuda_graph = (
        os.environ.get("USE_CUDA_GRAPH", "1") == "1"
        and DEVICE.type == 'cuda'
    )
    print(f"CUDA Graph: {'开启' if use_cuda_graph else '关闭'}")
    print(f"PROFILE:    {'开启' if _PROFILE else '关闭'}")

    batch_main(root_dir, save_dir, test_csv=test_csv,
               save_curves=False, curves_subdir='curves',
               use_cuda_graph=use_cuda_graph)


if __name__ == "__main__":
    test_focusing_accuracy(root_dir=r'F:\dongjiayao\Data\COCO\val\holograms_amp_phase\amplitude',
                           save_dir=r'F:\dongjiayao\Data\COCO\val\article\IEEE TIM\revise1\coco_time_test\cnsa_gpu_traverse',
                           test_csv=r'F:\dongjiayao\Data\COCO\val\holograms_amp_phase\amplitude\AutoFocusDatabase.csv')