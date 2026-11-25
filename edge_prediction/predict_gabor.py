# -*- coding: utf-8 -*-
"""
GaborLogistic (Gabor 前端) 批量预测 + 聚焦曲线

输入模式：
  单序列模式（默认，不加 --batch）：
      --input 传一个或多个"直接含图像"的文件夹，每个作为一个序列处理。
      仅传一个时输出到 <output>；传多个时各自输出到 <output>/<文件夹名>/。

  批量模式（加 --batch）：
      --input 传"总路径"，总路径下每个一级子文件夹（含图像）作为一个序列。
      输出到 <output>/<子文件夹名>/，并生成 <output>/batch_summary.csv。

输出：
    <output_dir>/prob/                每张图的概率图（0-255 灰度 PNG）
    <output_dir>/side_by_side/        原图 + 概率图的并排对比图
    <output_dir>/focus_scores.csv     每张图的 std 聚焦分数
    <output_dir>/focus_curve.png      聚焦分数随 z / 索引变化曲线
    <output_dir>/batch_summary.csv    （仅批量/多输入时）跨序列峰值汇总
"""

import os
import re
import csv
import argparse
from pathlib import Path

# ── 必须在所有 torch/cuda import 之前设置 ──────────────────────
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--gpu', default='0', type=str)
_pre_args, _ = _pre_parser.parse_known_args()
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu
# ────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt

# ★ 从 Gabor 前端训练脚本中导入模型与加载器
from train_gabor import GaborLogistic, load_model


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# ============================================================
# 聚焦分数：std of non-zero prob
# ============================================================
def compute_std_score(prob, border=20):
    """
    prob: torch.Tensor [H, W], float32, 值范围 [0, 1]
    border: 边缘置零像素数（排除 zero-padding 边界效应）
    返回: non_zero_vals.std().item()
    """
    if border > 0:
        prob = prob.clone()
        prob[:border, :]  = 0
        prob[-border:, :] = 0
        prob[:, :border]  = 0
        prob[:, -border:] = 0

    non_zero_vals = prob[prob > 0]
    if non_zero_vals.numel() == 0:
        return 0.0
    return non_zero_vals.std().item()


def parse_z(stem):
    """
    从文件名解析最后一个浮点数作为 z（米）。
    例：'off_88_0.0002350' → 0.0002350
    """
    matches = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', stem)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


# ============================================================
# 单张预测
# ============================================================
@torch.no_grad()
def predict_single(model, gray_tensor, device, compress=1):
    """
    gray_tensor: torch.Tensor [H, W], float32, 值范围 [0, 255]
    compress:    下采样倍数，仅作用于送入模型的图像
    返回:         torch.Tensor [H//c, W//c], float32, 值范围 [0, 1]（概率图）
    """
    x = gray_tensor.unsqueeze(0).unsqueeze(0).to(device)

    if compress > 1:
        h, w = x.shape[-2:]
        new_h = max(1, h // compress)
        new_w = max(1, w // compress)
        x = F.interpolate(x, size=(new_h, new_w),
                          mode='bilinear', align_corners=False)

    prob = model(x)
    return prob[0, 0].cpu()


# ============================================================
# 保存工具
# ============================================================
def save_probability_map(prob, path):
    arr = prob.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode='L').save(path)


def save_side_by_side(img_gray, prob, path, gap=10, compress=1):
    img_t = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0)
    if compress > 1:
        h, w = img_t.shape[-2:]
        new_h = max(1, h // compress)
        new_w = max(1, w // compress)
        img_t = F.interpolate(img_t, size=(new_h, new_w),
                              mode='bilinear', align_corners=False)
    img_arr = img_t.squeeze().numpy().clip(0, 255).astype(np.uint8)

    prob_arr = (np.clip(prob.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)

    h, w = img_arr.shape
    canvas = np.full((h, 2 * w + gap), 255, dtype=np.uint8)
    canvas[:, :w]       = img_arr
    canvas[:, w + gap:] = prob_arr
    Image.fromarray(canvas, mode='L').save(path)


# ============================================================
# 曲线绘制
# ============================================================
def plot_focus_curve(records, save_path, x_axis='auto',
                     title='Focus score (std of non-zero prob)'):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    use_z = False
    if x_axis == 'z':
        use_z = True
    elif x_axis == 'auto':
        if records and all(r['z'] is not None for r in records):
            use_z = True

    if use_z:
        xs = [r['z'] for r in records]
        xlabel = 'z (m)'
    else:
        xs = list(range(len(records)))
        xlabel = 'Frame index'

    ys = [r['score'] for r in records]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(xs, ys, marker='o', markersize=3, linewidth=1.2, color='#1f77b4')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Focus score (std)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if ys:
        idx_best = int(np.argmax(ys))
        ax.scatter([xs[idx_best]], [ys[idx_best]],
                   s=60, facecolors='none', edgecolors='red',
                   linewidths=1.5, zorder=5,
                   label=f'Peak: {xs[idx_best]:.6f}' if use_z
                         else f'Peak idx: {idx_best}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# ============================================================
# 批量预测（单个序列）
# ============================================================
def batch_predict(input_dir, output_dir, model_path, device_str='cuda:0',
                  compress=1, border=20, save_prob=True, save_side=True,
                  save_csv=True, save_curve=True, x_axis='auto'):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prob_dir = output_dir / 'prob'
    side_dir = output_dir / 'side_by_side'
    if save_prob:
        prob_dir.mkdir(parents=True, exist_ok=True)
    if save_side:
        side_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("GaborLogistic (Gabor 前端) 批量预测 + 聚焦曲线")
    print("=" * 70)
    print(f"输入目录   : {input_dir}")
    print(f"输出目录   : {output_dir}")
    print(f"模型权重   : {model_path}")
    print(f"使用设备   : {device}")
    print(f"compress   : {compress}  （只作用于送入模型的图像）")
    print(f"BORDER     : {border}")

    # ---- 加载模型 ----
    model = load_model(model_path, device=str(device))
    model.to(device)
    model.eval()
    print(f"模型前端   : {model.feature_type}")
    print("=" * 70)

    # ---- 枚举图像 ----
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(input_dir / f)
        and os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS
    ]
    image_files = natsorted(image_files)

    if not image_files:
        print(f"警告: 在 {input_dir} 中未找到图像文件")
        return []

    print(f"共 {len(image_files)} 张图像，开始预测...")

    # ---- 逐张处理 ----
    pbar = tqdm(image_files, desc="Predicting", unit="image")
    ok_count = 0
    fail_count = 0
    records = []      # [{'file': ..., 'z': ..., 'score': ...}, ...]

    for fname in pbar:
        pbar.set_description(f"Processing: {fname}")
        img_path = input_dir / fname
        try:
            img_pil = Image.open(img_path).convert('L')
            img_np  = np.array(img_pil, dtype=np.float32)
            gray_tensor = torch.from_numpy(img_np)

            prob = predict_single(model, gray_tensor, device, compress=compress)

            stem = os.path.splitext(fname)[0]
            if save_prob:
                save_probability_map(prob, prob_dir / f"{stem}_prob.png")
            if save_side:
                save_side_by_side(img_np, prob,
                                  side_dir / f"{stem}_side.png",
                                  compress=compress)

            # ★ 顺手算聚焦分数
            score = compute_std_score(prob, border=border)
            z = parse_z(stem)
            records.append({'file': fname, 'z': z, 'score': score})

            ok_count += 1
        except Exception as e:
            fail_count += 1
            print(f"\n处理 {fname} 失败: {e}")

    # ---- 保存 CSV ----
    if save_csv and records:
        csv_path = output_dir / "focus_scores.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=['file', 'z', 'score'])
            writer.writeheader()
            writer.writerows(records)
        print(f"\n聚焦分数 CSV 已保存: {csv_path}")

    # ---- 绘制曲线 ----
    if save_curve and records:
        curve_path = output_dir / "focus_curve.png"
        plot_focus_curve(records, curve_path, x_axis=x_axis)
        print(f"聚焦曲线已保存    : {curve_path}")

    # ---- 打印统计 ----
    if records:
        scores = [r['score'] for r in records]
        best_idx = int(np.argmax(scores))
        print("\n" + "=" * 70)
        print("聚焦分数统计")
        print("=" * 70)
        print(f"count    = {len(scores)}")
        print(f"mean     = {np.mean(scores):.6f}")
        print(f"std      = {np.std(scores):.6f}")
        print(f"min      = {np.min(scores):.6f}")
        print(f"max      = {np.max(scores):.6f}")
        print(f"peak     = {records[best_idx]['file']}")
        if records[best_idx]['z'] is not None:
            print(f"peak z   = {records[best_idx]['z']:.7f} m")
        print("=" * 70)

    print("\n" + "=" * 70)
    print(f"预测完成: 成功 {ok_count} / 失败 {fail_count}")
    if save_prob:
        print(f"概率图已保存至     : {prob_dir}")
    if save_side:
        print(f"并排对比图已保存至 : {side_dir}")
    print("=" * 70)

    return records


# ============================================================
# 输入展开 & 跨序列汇总
# ============================================================
def _images_in(folder):
    """folder(Path) 直接子级中的图像文件列表(自然排序)"""
    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(folder / f)
        and os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS
    ]
    return natsorted(files)


def collect_sequences(paths):
    """
    ★ 批量模式专用：
    把"总路径"展开为若干"序列目录" [(序列名, 目录), ...]
      - 总路径下的一级子文件夹中含图像 → 该子文件夹作为一个序列
      - 不含图像的子文件夹 → 跳过
    序列重名时自动追加 _2, _3 ... 去重。
    """
    sequences = []
    used_names = set()
    for p in paths:
        folder = Path(p)
        if not folder.is_dir():
            print(f"警告: 跳过不存在的目录: {folder}")
            continue

        candidates = [d for d in sorted(folder.iterdir())
                      if d.is_dir() and _images_in(d)]
        if not candidates:
            print(f"警告: {folder} 下没有含图像的一级子文件夹, 跳过")
            continue

        for d in candidates:
            name = d.name
            k = 2
            while name in used_names:          # 序列重名时加序号去重
                name = f"{d.name}_{k}"
                k += 1
            used_names.add(name)
            sequences.append((name, d))
    return sequences


def _write_summary(summary, output_root):
    """写 batch_summary.csv 并打印汇总"""
    if not summary:
        return
    summary_csv = Path(output_root) / "batch_summary.csv"
    with open(summary_csv, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(
            fp, fieldnames=['sequence', 'count', 'peak_file',
                            'peak_z', 'peak_score'])
        writer.writeheader()
        writer.writerows(summary)

    print()
    print("=" * 70)
    print("各序列聚焦峰值汇总")
    print("=" * 70)
    for s in summary:
        z_str = ''
        if s['peak_z'] != '':
            z_str = f", z = {float(s['peak_z']):.7f} m"
        print(f"  {s['sequence']}: peak = {s['peak_file']}{z_str} "
              f"(score = {s['peak_score']})")
    print()
    print(f"汇总 CSV 已保存: {summary_csv}")
    print("=" * 70)


def run_all(inputs, output_root, model_path, batch_mode=False,
            device_str='cuda:0', compress=4, border=20,
            save_prob=True, save_side=True,
            save_csv=True, save_curve=True, x_axis='auto'):
    """
    batch_mode=False：单序列模式。每个 --input 直接当作一个图像文件夹，
                      各自跑一次 batch_predict。
    batch_mode=True ：批量模式。--input 是总路径，总路径下每个一级
                      子文件夹（含图像）作为一个序列，并生成汇总 CSV。
    """
    if batch_mode:
        # ================= 批量模式 =================
        sequences = collect_sequences(inputs)
        if not sequences:
            print("错误: 批量模式下未找到任何含图像的一级子文件夹")
            return

        print()
        print("#" * 70)
        print(f"批量模式: 共 {len(sequences)} 个序列")
        for name, d in sequences:
            print(f"  - {name} <- {d}")
        print("#" * 70)

        summary = []
        for i, (name, d) in enumerate(sequences, 1):
            print()
            print(f"[{i}/{len(sequences)}] 序列: {name}")
            records = batch_predict(
                input_dir=d, output_dir=Path(output_root) / name,
                model_path=model_path, device_str=device_str,
                compress=compress, border=border,
                save_prob=save_prob, save_side=save_side,
                save_csv=save_csv, save_curve=save_curve, x_axis=x_axis)
            if records:
                best = max(records, key=lambda r: r['score'])
                summary.append({
                    'sequence': name,
                    'count': len(records),
                    'peak_file': best['file'],
                    'peak_z': best['z'] if best['z'] is not None else '',
                    'peak_score': f"{best['score']:.6f}",
                })
        _write_summary(summary, output_root)

    else:
        # ================= 单序列模式 =================
        print()
        print("#" * 70)
        print(f"单序列模式: 共 {len(inputs)} 个输入")
        print("#" * 70)

        multi = len(inputs) > 1
        summary = []
        for i, p in enumerate(inputs, 1):
            d = Path(p)
            if not d.is_dir():
                print(f"警告: 跳过不存在的目录: {d}")
                continue
            if not _images_in(d):
                print(f"警告: {d} 中没有图像, 跳过")
                continue

            # 单个输入 → 直接输出到 output_root
            # 多个输入 → 各自输出到 output_root/<输入文件夹名>，避免覆盖
            out_dir = Path(output_root) / d.name if multi else Path(output_root)

            print()
            print(f"[{i}/{len(inputs)}] 序列: {d.name}")
            records = batch_predict(
                input_dir=d, output_dir=out_dir,
                model_path=model_path, device_str=device_str,
                compress=compress, border=border,
                save_prob=save_prob, save_side=save_side,
                save_csv=save_csv, save_curve=save_curve, x_axis=x_axis)
            if records:
                best = max(records, key=lambda r: r['score'])
                summary.append({
                    'sequence': d.name,
                    'count': len(records),
                    'peak_file': best['file'],
                    'peak_z': best['z'] if best['z'] is not None else '',
                    'peak_score': f"{best['score']:.6f}",
                })

        # 多输入时也写一份汇总，单输入不写
        if multi:
            _write_summary(summary, output_root)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GaborLogistic (Gabor 前端) 批量预测 + 聚焦曲线")
    parser.add_argument(
        '--input', '-i', type=str, nargs='+',
        default=[r'F:\dongjiayao\Data\AutoFocusDatabase\edge\input'],
        help=('输入路径(可多个)。'
              '单序列模式(默认)：直接含图像的一个或多个文件夹；'
              '批量模式(--batch)：总路径，其下每个一级子文件夹为一个序列')
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default=r'F:\dongjiayao\Data\AutoFocusDatabase\edge\output_gabor',
        help='输出文件夹路径'
    )
    parser.add_argument(
        '--model', '-m', type=str,
        default=(r'F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing'
                 r'\edge_prediction_gabor\runs_gabor\20260916_150908'
                 r'\model_loss_0.022564.pth'),
        help='Gabor 前端模型权重路径 (.pth)'
    )
    parser.add_argument(
        '--batch', default=True,
        help=('批量模式开关：开启时 --input 为总路径，'
              '总路径下每个一级子文件夹作为一个序列处理，'
              '并输出 batch_summary.csv')
    )
    parser.add_argument('--gpu-device', type=str, default='cuda:0',
                        help='推理使用的设备，如 cuda:0 或 cpu')
    parser.add_argument('--compress', '-c', type=int, default=12,
                        help='送入模型前对图像的下采样倍数（1 表示不压缩）')
    parser.add_argument('--border', '-b', type=int, default=20,
                        help='边缘置零像素数（默认 20）')
    parser.add_argument('--x-axis', type=str, default='auto',
                        choices=['auto', 'z', 'index'],
                        help='曲线横轴类型')
    parser.add_argument('--no-prob',  action='store_true', help='不保存概率图')
    parser.add_argument('--no-side',  action='store_true', help='不保存并排对比图')
    parser.add_argument('--no-csv',   action='store_true', help='不保存聚焦分数 CSV')
    parser.add_argument('--no-curve', action='store_true', help='不绘制聚焦曲线')
    args = parser.parse_args()

    run_all(
        inputs=args.input,
        output_root=args.output,
        model_path=args.model,
        batch_mode=args.batch,
        device_str=args.gpu_device,
        compress=args.compress,
        border=args.border,
        save_prob=not args.no_prob,
        save_side=not args.no_side,
        save_csv=not args.no_csv,
        save_curve=not args.no_curve,
        x_axis=args.x_axis,
    )