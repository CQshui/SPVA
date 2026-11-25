#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化不同自动聚焦方法在 z 扫描重建图像上的聚焦分数曲线差距。

对比的方法:
    - VAR / ToG / GoG / Eigen  [传统]
    - MGSA  : GaborLogistic (Gabor 前端) 概率图的非零值 std
    - CNSA  : GaborLogistic (CNN 前端)   概率图的非零值 std

Ours 计算方法（与 test_singlePlane_COCO_gabor.py 完全一致）:
    prob = model(img).squeeze()             # forward 已带 sigmoid，直接是概率图
    prob[:BORDER,:] = 0; prob[-BORDER:,:] = 0
    prob[:,:BORDER] = 0; prob[:,-BORDER:] = 0
    non_zero_vals = prob[prob > 0]
    score = non_zero_vals.std().item()

每个方法可独立设置缩放倍率 N（new_size = size / N）:
    N > 1     : 缩小  (如 N=4  -> 1/4)
    N = 1     : 不缩放
    0 < N < 1 : 放大  (如 N=0.5 -> 2 倍, N=0.25 -> 4 倍)

文件名规则:
    图像文件名必须形如  off_<int>_<float>.<ext>
    例如:  off_1_0.0000000.jpg  ,  off_3_0.0000203.jpg
    其它文件名（FFT_0_0.jpg 等）会被自动忽略。

输出（每次处理一个文件夹）:
    <output_dir>/focus_curves_double_column.{png,pdf}
    <output_dir>/focus_curves_single_column.{png,pdf}
    <output_dir>/focus_curves.csv
"""

import os
import re
import csv
import time
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- 添加项目根目录与 edge_prediction 目录到 path ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GABOR_DIR = os.path.dirname(_THIS_DIR)                       # edge_prediction
_PROJECT_ROOT = os.path.dirname(_GABOR_DIR)                   # 项目根目录

import sys
sys.path.insert(0, _PROJECT_ROOT)   # for comparison_method / edge_prediction 包
sys.path.insert(0, _GABOR_DIR)      # for main 模块

# ★ 两个 GaborLogistic 变体，各自从对应的训练脚本加载
from edge_prediction.train_cnn import GaborLogistic as CNNModel, load_model as load_model_cnn
from edge_prediction.train_gabor import GaborLogistic as GaborModel, load_model as load_model_gabor

from comparison_method.selected.variance import variance_focus
from comparison_method.selected.GoG_ToG import calculate_tog, calculate_gog
from comparison_method.selected.eig_best import eigenvalue_focus_metric
from comparison_method.selected.wavelet import compute_wavelet_focus_score

try:
    from natsort import natsorted
except ImportError:
    natsorted = None

from tqdm import tqdm
HAS_TQDM = True


# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser(description='绘制各自动聚焦方法的聚焦分数曲线')
parser.add_argument('--dir', type=str,
                    # default=r'F:\dongjiayao\Data\AutoFocusDatabase\hologram_amplitude\reconstruction',
                    default=r'F:\dongjiayao\Data\USAF\PROP\detail\USAF1951_template.png',
                    help='包含不同 z 重建图像的文件夹路径')
parser.add_argument('--out', type=str, default=None,
                    help='输出目录（默认为图像文件夹）')
parser.add_argument('--gt', type=float, default=None,
                    help='真实 z 值 (m)，用于在图中标注')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID (默认 0)')
parser.add_argument('--pattern', type=str, default=r'(-?\d+\.\d+(?:[eE][-+]?\d+)?)',
                    help='从文件名提取 z 值的正则模式 (默认提取第一个浮点数)')
parser.add_argument('--z-scale', type=float, default=1e6,
                    help='z 值换算到显示单位的倍率 (默认 1e6: m -> μm)')

# ---- 文件名白名单规则 ----
# 只接受形如 off_<int>_<float>.<ext> 的文件；
# 从文件名里提取 z 值时，因为 z 部分是带小数点的浮点数，
# 而前面的序号是整数，--pattern 会正确命中 z 部分。
parser.add_argument('--filename-pattern', type=str,
                    default=r'^off_\d+_-?\d+\.\d+(?:[eE][-+]?\d+)?$',
                    help='图像文件名（不含扩展名）必须匹配的白名单正则。'
                         '默认只接受 off_<int>_<float> 格式，如 off_1_0.0000000。')

# ---- 两个 GaborLogistic 的模型权重 ----
parser.add_argument('--cnn-checkpoint', type=str,
                    default=(r'F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing'
                             r'\edge_prediction_gabor\runs_cnn\20260916_123348'
                             r'\model_loss_0.021667.pth'),
                    help='CNN 前端 GaborLogistic 模型权重路径')
parser.add_argument('--gabor-checkpoint', type=str,
                    default=(r'F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing'
                             r'\edge_prediction_gabor\runs_gabor\20260916_150908'
                             r'\model_loss_0.022564.pth'),
                    help='Gabor 前端 GaborLogistic 模型权重路径')

parser.add_argument('--border-width', type=int, default=20,
                    help='CNN / Gabor 方法概率图边缘置零宽度（与测试代码一致，默认 20）')

# ---------------- 每个方法的缩放系数 N ----------------
parser.add_argument('--var-downscale', type=float, default=8,
                    help='VAR 输入缩放倍率 N')
parser.add_argument('--tog-downscale', type=float, default=8,
                    help='ToG 输入缩放倍率 N')
parser.add_argument('--gog-downscale', type=float, default=8,
                    help='GoG 输入缩放倍率 N')
parser.add_argument('--eigen-downscale', type=float, default=8,
                    help='Eigen 输入缩放倍率 N')
parser.add_argument('--wavelet-downscale', type=float, default=8,
                    help='Wavelet 输入缩放倍率 N')
parser.add_argument('--cnn-downscale', type=float, default=8,
                    help='CNN 输入缩放倍率 N')
parser.add_argument('--gabor-downscale', type=float, default=8,
                    help='Gabor 输入缩放倍率 N')

parser.add_argument('--downscale-mode', type=str, default='bilinear',
                    choices=['bilinear', 'area', 'bicubic', 'nearest'],
                    help='传统方法缩放时使用的插值方式（默认 bilinear）')
parser.add_argument('--cnn-downscale-mode', type=str, default='bilinear',
                    choices=['bilinear', 'area', 'bicubic', 'nearest'],
                    help='CNN 缩放时使用的插值方式（默认 bilinear）')
parser.add_argument('--gabor-downscale-mode', type=str, default='bilinear',
                    choices=['bilinear', 'area', 'bicubic', 'nearest'],
                    help='Gabor 缩放时使用的插值方式（默认 bilinear）')

parser.add_argument('--cnn-batch-size', type=int, default=8,
                    help='CNN 推理批大小')
parser.add_argument('--gabor-batch-size', type=int, default=8,
                    help='Gabor 推理批大小')

# ---------------- 保存边缘概率图 ----------------
parser.add_argument('--save-cnn-maps', action='store_true',
                    help='保存 CNN 方法预测的边缘概率图（灰度 PNG）。')
parser.add_argument('--save-gabor-maps', action='store_true',
                    help='保存 Gabor 方法预测的边缘概率图（灰度 PNG）。')
parser.add_argument('--cnn-maps-dir', type=str, default=None,
                    help='CNN 概率图保存目录（默认 <输出目录>/cnn_maps/）。')
parser.add_argument('--gabor-maps-dir', type=str, default=None,
                    help='Gabor 概率图保存目录（默认 <输出目录>/gabor_maps/）。')
parser.add_argument('--maps-upsample', action='store_true',
                    help='保存概率图时上采样回原图分辨率。')

# ---------------- 批量模式 ----------------
parser.add_argument('--batch', default=True,
                    help='批量模式：将 --dir 视为主文件夹，遍历其中每个子文件夹。')
parser.add_argument('--subdirs', type=str, default=None,
                    help='批量模式下要处理的子文件夹名称，逗号分隔（如 "sample1,sample2"）。'
                         '不指定则处理全部子文件夹。')
parser.add_argument('--skip-missing-subdirs', action='store_true',
                    help='批量模式下，若 --subdirs 中某个子文件夹不存在则跳过（默认报错退出）。')

# ---------------- 绘图参数 ----------------
# 双栏（stack）
parser.add_argument('--double-fig-width', type=float, default=7.0,
                    help='双栏图宽度（英寸），默认 7.0')
parser.add_argument('--double-row-height', type=float, default=0.62,
                    help='双栏 stack 视图每行高度（英寸），默认 0.62')
parser.add_argument('--double-font-size', type=float, default=8.0,
                    help='双栏图字号（pt），默认 8.0')

# 单栏（stack）
parser.add_argument('--single-fig-width', type=float, default=3.4,
                    help='单栏图宽度（英寸），默认 3.4')
parser.add_argument('--single-row-height', type=float, default=0.45,
                    help='单栏 stack 视图每行高度（英寸），默认 0.45')
parser.add_argument('--single-font-size', type=float, default=7.0,
                    help='单栏图字号（pt），默认 7.0')

# 通用
parser.add_argument('--line-width', type=float, default=1.0,
                    help='曲线线宽')
parser.add_argument('--dpi', type=int, default=600,
                    help='输出分辨率')
parser.add_argument('--save-format', type=str, default='both',
                    choices=['png', 'pdf', 'svg', 'both'],
                    help="保存格式：'png'、'pdf'、'svg' 或 'both'。")
parser.add_argument('--show-title', action='store_true',
                    help='是否在图上显示标题。')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# rcParams
# ============================================================
def _set_rcparams(font_size: float, dpi: int):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size + 1,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'lines.linewidth': 1.0,
        'lines.solid_capstyle': 'round',
        'axes.grid': False,
        'savefig.dpi': dpi,
        'figure.dpi': 120,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


_set_rcparams(args.double_font_size, args.dpi)


# ============================================================
# 通用工具
# ============================================================
Z_PATTERN = re.compile(args.pattern)

# 文件名白名单（不含扩展名）。
# 默认只接受 off_<int>_<float>，例如 off_1_0.0000000 / off_3_0.0000203。
# 这样 FFT_0_0.jpg、hologram.png 之类的文件会被自动忽略。
_FILENAME_PATTERN = re.compile(args.filename_pattern, re.IGNORECASE)


def extract_z(filename: str):
    name = os.path.splitext(os.path.basename(filename))[0]
    matches = Z_PATTERN.findall(name)
    if matches:
        return float(matches[0])
    return None


def load_image(filepath: str) -> torch.Tensor:
    """加载图像为 GPU 张量 (H, W)，float32，归一化到 [0, 1]"""
    img = Image.open(filepath).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(DEVICE)


def _describe_scale(name, N):
    if N is None or N <= 0:
        return f"{name}: 输入不缩放"
    if abs(N - 1.0) < 1e-9:
        return f"{name}: 输入保持原尺寸"
    if N > 1.0:
        return f"{name}: 输入缩小到 1/{N:g}"
    return f"{name}: 输入放大到 {1.0 / N:g} 倍 (N={N:g})"


# ============================================================
# 通用缩放
# ============================================================
def _resize_2d(img_2d: torch.Tensor,
               downscale: float,
               mode: str = 'bilinear') -> torch.Tensor:
    """
    N > 1     : 缩小，new_size = size / N
    N = 1     : 不缩放
    0 < N < 1 : 放大
    """
    if downscale is None or downscale <= 0:
        return img_2d
    if abs(float(downscale) - 1.0) < 1e-9:
        return img_2d

    h, w = img_2d.shape
    new_h = max(1, int(round(h / float(downscale))))
    new_w = max(1, int(round(w / float(downscale))))
    if (new_h, new_w) == (h, w):
        return img_2d

    inp = img_2d.unsqueeze(0).unsqueeze(0)
    if mode in ('bilinear', 'bicubic'):
        out = F.interpolate(inp, size=(new_h, new_w), mode=mode,
                            align_corners=False)
    else:
        out = F.interpolate(inp, size=(new_h, new_w), mode=mode)
    return out.squeeze(0).squeeze(0)


def make_scaled_scorer(score_fn_2d, downscale, mode='bilinear'):
    def scorer(img_2d: torch.Tensor):
        img_in = _resize_2d(img_2d, downscale, mode)
        return score_fn_2d(img_in)
    return scorer


# ============================================================
# GaborLogistic 评分器（CNN / Gabor 共用）
# ============================================================
class GaborLogisticScorer:
    def __init__(self, checkpoint, border_width,
                 downscale=1.0, downscale_mode='bilinear', device=None,
                 collect_maps=False, loader_fn=None, name='Model'):
        self.device = device if device is not None else DEVICE
        self.name = name
        self.loader_fn = loader_fn
        self.model = loader_fn(checkpoint, device=str(self.device))
        self.model.eval()
        self.border_width = int(border_width)
        self.downscale = float(downscale) if downscale else 1.0
        self.downscale_mode = downscale_mode
        self.collect_maps = bool(collect_maps)
        self.last_maps = []

        print(f"{self.name} 模型加载成功: {checkpoint}")
        print("  " + _describe_scale(self.name, self.downscale)
              + f" (mode={self.downscale_mode})")
        print(f"  BORDER = {self.border_width}")
        if self.collect_maps:
            print(f"  {self.name} 概率图缓存已开启")

    def _resize(self, img_2d):
        return _resize_2d(img_2d, self.downscale, self.downscale_mode)

    def _prepare_input(self, img_2d):
        """compare 脚本内部图像为 [0,1]；模型训练输入是 [0,255]。"""
        img_in = self._resize(img_2d)
        img_in = img_in * 255.0
        return img_in

    def _postprocess(self, prob_2d: torch.Tensor):
        """
        完全复刻 test_singlePlane_COCO_gabor.py 里 focus_value 的计算：
            prob = model(img).squeeze()
            边缘 BORDER 像素置零
            non_zero_vals = prob[prob > 0]
            score = non_zero_vals.std().item()
        """
        prob = prob_2d.detach().float().cpu().numpy()

        bw = self.border_width
        if bw > 0:
            h, w = prob.shape
            if bw < min(h, w) // 2:
                prob[:bw, :] = 0.0
                prob[-bw:, :] = 0.0
                prob[:, :bw] = 0.0
                prob[:, -bw:] = 0.0

        nz = prob[prob > 0]
        if nz.size == 0:
            score = 0.0
        else:
            score = float(nz.std())

        return score, prob

    @torch.inference_mode()
    def score(self, img_2d: torch.Tensor) -> float:
        img_in = self._prepare_input(img_2d)
        prob = self.model(img_in.unsqueeze(0).unsqueeze(0))
        prob = prob.squeeze(0).squeeze(0)
        score, prob_np = self._postprocess(prob)
        if self.collect_maps:
            self.last_maps = [prob_np]
        return score

    @torch.inference_mode()
    def score_batch(self, img_list, batch_size=8, show_progress=True):
        n = len(img_list)
        if n == 0:
            return []

        self.last_maps = []
        prepared = [self._prepare_input(im) for im in img_list]

        shapes = {tuple(t.shape) for t in prepared}
        if len(shapes) > 1:
            print(f"    [warn] {self.name} 缩放后图像尺寸不一致，回退为逐张推理。")
            it = range(n)
            if show_progress:
                it = tqdm(it, total=n, desc=f"  {self.name}", ncols=100)
            out = []
            for i in it:
                t = prepared[i]
                prob = self.model(t.unsqueeze(0).unsqueeze(0))
                prob = prob.squeeze(0).squeeze(0)
                score, prob_np = self._postprocess(prob)
                out.append(score)
                if self.collect_maps:
                    self.last_maps.append(prob_np)
            return out

        scores = []
        indices = list(range(0, n, batch_size))
        it = indices
        if show_progress:
            it = tqdm(indices, total=len(indices),
                      desc=f"  {self.name}", ncols=100)

        for start in it:
            chunk = prepared[start:start + batch_size]
            batch = torch.stack(chunk, dim=0).unsqueeze(1)
            out = self.model(batch)
            if out.dim() == 4:
                out = out.squeeze(1)
            for b in range(out.shape[0]):
                score, prob_np = self._postprocess(out[b])
                scores.append(score)
                if self.collect_maps:
                    self.last_maps.append(prob_np)

        return scores


# ============================================================
# 保存概率图
# ============================================================
def save_maps(maps_dir, z_file_pairs, img_list, prob_list,
              upsample=False, suffix='prob'):
    os.makedirs(maps_dir, exist_ok=True)
    saved_paths = []

    for i, (z, filepath) in enumerate(z_file_pairs):
        prob = prob_list[i]

        if upsample and i < len(img_list):
            h_orig, w_orig = img_list[i].shape
            prob_t = torch.from_numpy(prob).unsqueeze(0).unsqueeze(0).float()
            prob_t = F.interpolate(prob_t, size=(h_orig, w_orig),
                                   mode='bilinear', align_corners=False)
            prob = prob_t.squeeze(0).squeeze(0).numpy()

        img_u8 = np.clip(prob * 255.0, 0.0, 255.0).astype(np.uint8)
        stem = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(maps_dir, f'{stem}_{suffix}.png')
        Image.fromarray(img_u8, mode='L').save(out_path)
        saved_paths.append(out_path)

    return saved_paths


# ============================================================
# 绘图
# ============================================================
def normalize_curve(curve):
    c = np.asarray(curve, dtype=np.float64)
    c_min, c_max = c.min(), c.max()
    if c_max - c_min > 1e-12:
        return (c - c_min) / (c_max - c_min)
    return np.zeros_like(c)


def _save_fig(fig, out_dir, base_name):
    saved = []
    fmt = args.save_format
    if fmt in ('png', 'both'):
        p = os.path.join(out_dir, f'{base_name}.png')
        fig.savefig(p, dpi=args.dpi, bbox_inches='tight', pad_inches=0.02)
        saved.append(p)
    if fmt in ('pdf', 'both'):
        p = os.path.join(out_dir, f'{base_name}.pdf')
        fig.savefig(p, bbox_inches='tight', pad_inches=0.02)
        saved.append(p)
    if fmt == 'svg':
        p = os.path.join(out_dir, f'{base_name}.svg')
        fig.savefig(p, bbox_inches='tight', pad_inches=0.02)
        saved.append(p)
    return saved


def plot_stacked_custom(scores_norm, x, xlabel, colors,
                        fig_width, row_height, font_size, dpi,
                        gt=None, z_scale=1e6,
                        show_title=False, line_width=1.0):
    """
    Stack 视图（每方法一行），宽度/行高/字号可调。
    双栏和单栏都用这个形式，只是参数不同。
    """
    _set_rcparams(font_size, dpi)
    n = len(scores_norm)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(fig_width, row_height * n),
        sharex=True,
        gridspec_kw={'hspace': 0.12},
    )
    if n == 1:
        axes = [axes]

    for i, (ax, (name, curve), color) in enumerate(
            zip(axes, scores_norm.items(), colors)):
        ax.plot(x, curve, '-', color=color,
                linewidth=line_width, alpha=0.9)

        if gt is not None:
            ax.axvline(x=gt * z_scale, color='black',
                       linestyle=':', linewidth=0.6, alpha=0.5)

        ax.text(0.985, 0.82, f'{name}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=font_size - 0.5,
                color=color, fontweight='bold')

        ax.set_ylim(-0.10, 1.15)
        ax.set_yticks([0, 1])
        ax.tick_params(axis='y', direction='out', length=2.0, pad=1.0)
        ax.tick_params(axis='x', direction='out', length=2.0, pad=1.0)
        ax.grid(True, axis='x', linestyle=':', linewidth=0.4, alpha=0.4)
        ax.set_axisbelow(True)

        if i < n - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel(xlabel)
    if show_title:
        axes[0].set_title('Autofocus method comparison')

    fig.tight_layout(pad=0.2)
    return fig


# ============================================================
# 单文件夹处理
# ============================================================
def scan_image_files(image_dir):
    """
    扫描目录下的图像文件。

    只返回文件名（去掉扩展名后）能匹配 _FILENAME_PATTERN 的文件，
    默认即形如 off_<int>_<float>.<ext>（如 off_1_0.0000000.jpg）。
    其它文件（FFT_0_0.jpg、hologram.png 等）一律忽略。
    """
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = []
    for f in os.listdir(image_dir):
        ext = os.path.splitext(f.lower())[1]
        if ext not in IMAGE_EXTS:
            continue
        stem = os.path.splitext(f)[0]
        if not _FILENAME_PATTERN.match(stem):
            continue
        files.append(f)
    return natsorted(files) if natsorted else sorted(files)


def process_folder(image_dir, output_dir, methods,
                   cnn_scorer=None, gabor_scorer=None):
    os.makedirs(output_dir, exist_ok=True)

    image_files = scan_image_files(image_dir)
    if not image_files:
        print(f"  跳过 '{os.path.basename(image_dir)}': "
              f"未找到符合命名规则的图像文件 (需匹配 {args.filename_pattern})")
        return None

    z_file_pairs = []
    for f in image_files:
        z = extract_z(f)
        z_file_pairs.append((z, os.path.join(image_dir, f)))

    has_z = all(z is not None for z, _ in z_file_pairs)
    if has_z:
        z_file_pairs.sort(key=lambda p: p[0])
        z_values = [p[0] for p in z_file_pairs]
        x = [z * args.z_scale for z in z_values]
        xlabel = 'z (μm)' if abs(args.z_scale - 1e6) < 1e-9 else f'z (x{args.z_scale:g})'
    else:
        z_values = list(range(len(z_file_pairs)))
        x = list(range(len(z_file_pairs)))
        xlabel = 'Image index'

    if len(z_file_pairs) < 2:
        print(f"  跳过 '{os.path.basename(image_dir)}': 有效图像数量不足 2")
        return None

    n_imgs = len(z_file_pairs)

    img_list = []
    loader = tqdm(z_file_pairs, desc="  加载图像", ncols=100)
    for _, filepath in loader:
        img_list.append(load_image(filepath))
    if hasattr(loader, 'close'):
        loader.close()

    scores = OrderedDict()

    # ---- 传统方法 ----
    for method_name, score_fn in methods.items():
        t1 = time.time()
        curve = []
        pbar = tqdm(img_list, total=n_imgs, desc=f"  {method_name}", ncols=100)
        for img_t in pbar:
            s = score_fn(img_t)
            if isinstance(s, torch.Tensor):
                s = s.item()
            curve.append(float(s))
        if hasattr(pbar, 'close'):
            pbar.close()
        scores[method_name] = curve
        print(f"  {method_name} 完成，用时 {time.time() - t1:.2f}s")

    # ---- Gabor ----
    if gabor_scorer is not None:
        t1 = time.time()
        gabor_scores = gabor_scorer.score_batch(
            img_list, batch_size=args.gabor_batch_size, show_progress=True
        )
        scores['MGSA'] = gabor_scores
        print(f"  Gabor 完成，用时 {time.time() - t1:.2f}s "
              f"(batch_size={args.gabor_batch_size})")

        if args.save_gabor_maps:
            maps_dir = args.gabor_maps_dir or os.path.join(output_dir, 'gabor_maps')
            t2 = time.time()
            saved_maps = save_maps(
                maps_dir, z_file_pairs, img_list,
                gabor_scorer.last_maps,
                upsample=args.maps_upsample,
                suffix='gabor_prob',
            )
            print(f"  Gabor 概率图已保存 {len(saved_maps)} 张 -> {maps_dir} "
                  f"(用时 {time.time() - t2:.2f}s)")

    # ---- CNN ----
    if cnn_scorer is not None:
        t1 = time.time()
        cnn_scores = cnn_scorer.score_batch(
            img_list, batch_size=args.cnn_batch_size, show_progress=True
        )
        scores['CNSA'] = cnn_scores
        print(f"  CNN 完成，用时 {time.time() - t1:.2f}s "
              f"(batch_size={args.cnn_batch_size})")

        if args.save_cnn_maps:
            maps_dir = args.cnn_maps_dir or os.path.join(output_dir, 'cnn_maps')
            t2 = time.time()
            saved_maps = save_maps(
                maps_dir, z_file_pairs, img_list,
                cnn_scorer.last_maps,
                upsample=args.maps_upsample,
                suffix='cnn_prob',
            )
            print(f"  CNN 概率图已保存 {len(saved_maps)} 张 -> {maps_dir} "
                  f"(用时 {time.time() - t2:.2f}s)")

    # ---- 保存 CSV ----
    csv_path = os.path.join(output_dir, 'focus_curves.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z', 'x'] + list(scores.keys()))
        for i in range(n_imgs):
            writer.writerow([z_values[i], x[i]] + [scores[m][i] for m in scores.keys()])

    # ---- 归一化 ----
    scores_norm = OrderedDict((m, normalize_curve(c)) for m, c in scores.items())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#e377c2', '#17becf']

    # ============================================================
    # 版本 1：论文双栏 —— stack 视图，7.0 in 宽，8 pt
    # ============================================================
    fig_double = plot_stacked_custom(
        scores_norm, x, xlabel, colors,
        fig_width=args.double_fig_width,
        row_height=args.double_row_height,
        font_size=args.double_font_size,
        dpi=args.dpi,
        gt=args.gt,
        z_scale=args.z_scale,
        show_title=args.show_title,
        line_width=args.line_width,
    )
    saved_d = _save_fig(fig_double, output_dir, 'focus_curves_double_column')
    plt.close(fig_double)
    for p in saved_d:
        print(f"  [double-column] 已保存: {p}")

    # ============================================================
    # 版本 2：论文单栏 —— stack 视图，3.4 in 宽，7 pt
    # ============================================================
    fig_single = plot_stacked_custom(
        scores_norm, x, xlabel, colors,
        fig_width=args.single_fig_width,
        row_height=args.single_row_height,
        font_size=args.single_font_size,
        dpi=args.dpi,
        gt=args.gt,
        z_scale=args.z_scale,
        show_title=args.show_title,
        line_width=args.line_width,
    )
    saved_s = _save_fig(fig_single, output_dir, 'focus_curves_single_column')
    plt.close(fig_single)
    for p in saved_s:
        print(f"  [single-column] 已保存: {p}")

    peaks = OrderedDict()
    for method_name, curve in scores.items():
        idx = int(np.argmax(curve))
        peaks[method_name] = (z_values[idx], x[idx])

    return has_z, peaks


def print_summary(has_z, peaks, tag=''):
    print(f"{tag}聚焦曲线峰值:")
    for method_name, (z, xv) in peaks.items():
        if has_z:
            print(f"  {tag}{method_name:8s}: z = {z:.7f} m (= {xv:.2f} x{args.z_scale:g})")
        else:
            print(f"  {tag}{method_name:8s}: index = {z}")
    if args.gt is not None and has_z:
        gt_x = args.gt * args.z_scale
        print(f"  {tag}{'GT':8s}: z = {args.gt:.7f} m (= {gt_x:.2f} x{args.z_scale:g})")
        print(f"{tag}各方法峰值相对 GT 的偏差:")
        for method_name, (z, xv) in peaks.items():
            print(f"  {tag}{method_name:8s}: error = {xv - gt_x:+.2f} x{args.z_scale:g}")


def _resolve_subdirs(root_dir):
    """
    根据 args.subdirs / args.skip_missing_subdirs 决定要处理哪些子文件夹。
    返回 (subdirs, ok)，ok=False 表示应当终止批量流程。
    """
    if args.subdirs:
        requested = [s.strip() for s in args.subdirs.split(',') if s.strip()]
        subdirs, missing = [], []
        for d in requested:
            full = os.path.join(root_dir, d)
            if os.path.isdir(full):
                subdirs.append(d)
            else:
                missing.append(d)
        if missing:
            if args.skip_missing_subdirs:
                print(f"警告: 以下子文件夹不存在，已跳过: {missing}")
            else:
                print(f"错误: 以下子文件夹不存在"
                      f"（如确认跳过请加 --skip-missing-subdirs）: {missing}")
                return [], False
        if not subdirs:
            print("错误: 没有任何有效的子文件夹可处理")
            return [], False
        print(f"批量模式: 用户指定了 {len(requested)} 个子文件夹，"
              f"实际处理 {len(subdirs)} 个: {subdirs}")
        return subdirs, True

    subdirs = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    if not subdirs:
        print(f"错误: 主文件夹 {root_dir} 下未找到子文件夹")
        return [], False
    print(f"批量模式: 找到 {len(subdirs)} 个子文件夹")
    return subdirs, True


def main():
    t_start = time.time()

    scale_cfg = {
        'Var':     args.var_downscale,
        'ToG':     args.tog_downscale,
        'GoG':     args.gog_downscale,
        'Eigen':   args.eigen_downscale,
        'Wavelet': args.wavelet_downscale,
        'MGSA':    args.gabor_downscale,
        'CNSA':    args.cnn_downscale,
    }
    print("各方法的输入缩放设定:")
    for k, v in scale_cfg.items():
        print("  " + _describe_scale(k, v))
    print(f"  传统方法插值方式: {args.downscale_mode}")
    print(f"  CNN     插值方式: {args.cnn_downscale_mode}")
    print(f"  Gabor   插值方式: {args.gabor_downscale_mode}")
    print(f"  图像文件名白名单: {args.filename_pattern}")

    # ---- 两个 GaborLogistic 评分器，跨文件夹复用 ----
    cnn_scorer = GaborLogisticScorer(
        args.cnn_checkpoint, args.border_width,
        downscale=args.cnn_downscale,
        downscale_mode=args.cnn_downscale_mode,
        device=DEVICE,
        collect_maps=args.save_cnn_maps,
        loader_fn=load_model_cnn,
        name='CNSA',
    )
    gabor_scorer = GaborLogisticScorer(
        args.gabor_checkpoint, args.border_width,
        downscale=args.gabor_downscale,
        downscale_mode=args.gabor_downscale_mode,
        device=DEVICE,
        collect_maps=args.save_gabor_maps,
        loader_fn=load_model_gabor,
        name='MGSA',
    )

    # ---- 传统方法 ----
    dm = args.downscale_mode
    methods = OrderedDict([
        ('VAR',     make_scaled_scorer(
            lambda x: variance_focus(x.unsqueeze(0).unsqueeze(0)),
            args.var_downscale, dm)),
        ('ToG',     make_scaled_scorer(
            lambda x: calculate_tog(x.unsqueeze(0).unsqueeze(0)),
            args.tog_downscale, dm)),
        ('GoG',     make_scaled_scorer(
            lambda x: calculate_gog(x.unsqueeze(0).unsqueeze(0)),
            args.gog_downscale, dm)),
        ('Eigen',   make_scaled_scorer(
            lambda x: eigenvalue_focus_metric(x.unsqueeze(0).unsqueeze(0), k_percent=0.02),
            args.eigen_downscale, dm)),
        # ('Wavelet', make_scaled_scorer(
        #     lambda x: compute_wavelet_focus_score(x),
        #     args.wavelet_downscale, dm)),
    ])

    # ============================================================
    # 单文件夹模式
    # ============================================================
    if not args.batch:
        output_dir = args.out if args.out else args.dir
        result = process_folder(args.dir, output_dir, methods,
                                cnn_scorer, gabor_scorer)
        if result is not None:
            has_z, peaks = result
            print_summary(has_z, peaks)
        print(f"\n总用时: {time.time() - t_start:.2f}s")
        return

    # ============================================================
    # 批量模式
    # ============================================================
    root_out = args.out if args.out else args.dir
    os.makedirs(root_out, exist_ok=True)

    subdirs, ok = _resolve_subdirs(args.dir)
    if not ok:
        return

    summary_rows = []
    for i, sub in enumerate(subdirs, 1):
        sub_path = os.path.join(args.dir, sub)
        out_path = os.path.join(root_out, sub)
        print(f"\n[{i}/{len(subdirs)}] 处理子文件夹 '{sub}' ...")
        t_sub = time.time()
        result = process_folder(sub_path, out_path, methods,
                                cnn_scorer, gabor_scorer)
        if result is None:
            continue
        has_z, peaks = result
        print_summary(has_z, peaks, tag='  ')
        print(f"  子文件夹用时: {time.time() - t_sub:.2f}s")
        for method_name, (z, xv) in peaks.items():
            summary_rows.append([sub, method_name, z, xv])

    if summary_rows:
        summary_path = os.path.join(root_out, 'batch_summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['subfolder', 'method', 'peak_z', 'peak_x']
            if args.gt is not None:
                header += ['gt_x', 'error']
            writer.writerow(header)
            gt_x = args.gt * args.z_scale if args.gt is not None else None
            for sub, method_name, z, xv in summary_rows:
                row = [sub, method_name, z, xv]
                if gt_x is not None:
                    row += [gt_x, xv - gt_x]
                writer.writerow(row)
        print(f"\n批量汇总已保存: {summary_path}")

    print(f"\n总用时: {time.time() - t_start:.2f}s")


if __name__ == '__main__':
    main()