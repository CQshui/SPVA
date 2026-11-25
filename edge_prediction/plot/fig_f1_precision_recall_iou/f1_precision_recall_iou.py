import os
import csv
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

# ================= 配置 =================
REF_PATH     = r'F:\dongjiayao\Data\USAF\image\USAF_edge.jpg'      # 参考边缘图
FOLDER_MGSA  = r'F:\dongjiayao\Data\USAF\PROP_EDGE\Gabor\prob'     # MGSA 概率图文件夹
FOLDER_CNSA  = r'F:\dongjiayao\Data\USAF\PROP_EDGE\CNN\prob'       # CNSA 概率图文件夹
OUT_DIR      = r'F:\dongjiayao\Data\USAF\PROP_EDGE\metrics'        # 输出目录
THRESHOLD    = 0.5
Z_SCALE      = 1e6
DPI          = 600
LINE_WIDTH   = 1.0
SAVE_FORMAT  = 'both'

DOUBLE_FIG_WIDTH  = 7.0
DOUBLE_ROW_HEIGHT = 0.62
DOUBLE_FONT_SIZE  = 8.0

SINGLE_FIG_WIDTH  = 3.4
SINGLE_ROW_HEIGHT = 0.45
SINGLE_FONT_SIZE  = 7.0

os.makedirs(OUT_DIR, exist_ok=True)


def _set_rcparams(font_size, dpi):
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


ref = cv2.imread(REF_PATH, cv2.IMREAD_GRAYSCALE)
if ref is None:
    raise FileNotFoundError(f"无法读取参考图: {REF_PATH}")
_, ref_bin = cv2.threshold(ref, 127, 1, cv2.THRESH_BINARY)
ref_bin = ref_bin.astype(np.uint8)


def process_folder(folder):
    files = []
    for f in os.listdir(folder):
        full = os.path.join(folder, f)
        if not os.path.isfile(full):
            continue
        stem = os.path.splitext(f)[0]
        parts = stem.split('_')
        if len(parts) < 2:
            print(f"警告：文件名段数不足，跳过 {f}")
            continue
        try:
            z = float(parts[-2])
        except ValueError:
            print(f"警告：无法从文件名提取 z，跳过 {f}")
            continue
        files.append((z, full))

    files.sort(key=lambda x: x[0])
    if not files:
        raise FileNotFoundError(f"未在 {folder} 中找到有效文件。")
    print(f"[{os.path.basename(folder)}] 共处理 {len(files)} 个文件，z 范围 [{files[0][0]}, {files[-1][0]}]")

    z_values, file_names = [], []
    precision_l, recall_l, iou_l, f1_l = [], [], [], []
    mean_l, var_l, entropy_l = [], [], []

    for z, path in files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"警告：无法读取 {path}，跳过")
            continue
        prob = img.astype(np.float32) / 255.0

        pred = (prob > THRESHOLD).astype(np.uint8)
        if pred.shape != ref_bin.shape:
            pred = cv2.resize(pred, (ref_bin.shape[1], ref_bin.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        TP = int(np.sum((pred == 1) & (ref_bin == 1)))
        FP = int(np.sum((pred == 1) & (ref_bin == 0)))
        FN = int(np.sum((pred == 0) & (ref_bin == 1)))

        p   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        r   = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
        f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        mean_v = float(np.mean(prob))
        var_v  = float(np.var(prob))

        hist, _ = np.histogram(prob, bins=256, range=(0, 1))
        h = hist.astype(np.float64) / np.sum(hist)
        h = h[h > 0]
        entropy_v = float(-np.sum(h * np.log2(h)))

        z_values.append(z)
        file_names.append(os.path.basename(path))
        precision_l.append(p); recall_l.append(r)
        iou_l.append(iou);     f1_l.append(f1)
        mean_l.append(mean_v); var_l.append(var_v)
        entropy_l.append(entropy_v)

    x = np.array(z_values) * Z_SCALE
    return {
        'z': z_values, 'x': x, 'file': file_names,
        'precision': precision_l, 'recall': recall_l,
        'iou': iou_l, 'f1': f1_l,
        'mean': mean_l, 'var': var_l, 'entropy': entropy_l,
    }


data_mgsa = process_folder(FOLDER_MGSA)
data_cnsa = process_folder(FOLDER_CNSA)

x = data_mgsa['x']
xlabel = 'z (μm)' if abs(Z_SCALE - 1e6) < 1e-9 else f'z (x{Z_SCALE:g})'


def save_csv(data, name):
    csv_path = os.path.join(OUT_DIR, f'metrics_{name}_vs_z.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['z', 'x', 'file',
                         'precision', 'recall', 'iou', 'f1',
                         'mean', 'var', 'entropy'])
        for k in range(len(data['z'])):
            writer.writerow([
                data['z'][k], data['x'][k], data['file'][k],
                data['precision'][k], data['recall'][k], data['iou'][k], data['f1'][k],
                data['mean'][k], data['var'][k], data['entropy'][k]
            ])
    print(f"CSV 已保存: {csv_path}")

save_csv(data_mgsa, 'MGSA')
save_csv(data_cnsa, 'CNSA')


def _clean_y_formatter():
    def fmt(v, _pos):
        av = abs(v)
        if av == 0:
            return '0'
        if av >= 1e4 or av < 1e-3:
            s = f'{v:.2e}'
            mant, exp = s.split('e')
            mant = mant.rstrip('0').rstrip('.')
            return f'{mant}e{int(exp):+d}'
        else:
            if av >= 100:
                return f'{v:.0f}'
            elif av >= 1:
                return f'{v:.2f}'.rstrip('0').rstrip('.')
            else:
                return f'{v:.3f}'.rstrip('0').rstrip('.')
    return FuncFormatter(fmt)


def plot_compare(curves_mgsa, curves_cnsa, x, xlabel,
                 fig_width, row_height, font_size, dpi,
                 line_width=1.0, y_pad_ratio=0.10):
    """
    每个子图绘制两条曲线：MGSA（蓝）和 CNSA（红）。
    指标名放左上角，图例放右上角，避免重叠。
    """
    _set_rcparams(font_size, dpi)
    names = list(curves_mgsa.keys())
    n = len(names)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(fig_width, row_height * n),
        sharex=True,
        gridspec_kw={'hspace': 0.12},
    )
    if n == 1:
        axes = [axes]

    color_mgsa = '#1f77b4'
    color_cnsa = '#d62728'

    for i, (ax, name) in enumerate(zip(axes, names)):
        y_mgsa = np.asarray(curves_mgsa[name], dtype=np.float64)
        y_cnsa = np.asarray(curves_cnsa[name], dtype=np.float64)

        ax.plot(x, y_mgsa, '-', color=color_mgsa, linewidth=line_width,
                alpha=0.9, label='MGSA')
        ax.plot(x, y_cnsa, '-', color=color_cnsa, linewidth=line_width,
                alpha=0.9, label='CNSA')

        y_all = np.concatenate([y_mgsa, y_cnsa])
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        span = y_max - y_min
        if span < 1e-12:
            span = max(abs(y_max), 1e-6)
        pad = span * y_pad_ratio
        ax.set_ylim(y_min - pad, y_max + pad)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax.yaxis.set_major_formatter(_clean_y_formatter())
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis='y', direction='out', length=2.0, pad=1.5)

        # 指标名放到左上角
        ax.text(0.015, 0.82, name,
                transform=ax.transAxes, ha='left', va='top',
                fontsize=font_size - 0.5,
                color='black', fontweight='bold')

        # 图例只放第一个子图，并放到右上角
        if i == 0:
            ax.legend(loc='upper right', frameon=False,
                      fontsize=font_size - 1, ncol=2,
                      handlelength=1.5, columnspacing=1.0,
                      borderaxespad=0.2)

        ax.tick_params(axis='x', direction='out', length=2.0, pad=1.0)
        ax.grid(True, axis='x', linestyle=':', linewidth=0.4, alpha=0.4)
        ax.set_axisbelow(True)

        if i < n - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel(xlabel)
    fig.tight_layout(pad=0.2)
    return fig


def save_fig(fig, out_dir, base_name,
             save_format=SAVE_FORMAT, dpi=DPI):
    saved = []
    if save_format in ('png', 'both'):
        p = os.path.join(out_dir, f'{base_name}.png')
        fig.savefig(p, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        saved.append(p)
    if save_format in ('pdf', 'both'):
        p = os.path.join(out_dir, f'{base_name}.pdf')
        fig.savefig(p, bbox_inches='tight', pad_inches=0.02)
        saved.append(p)
    if save_format == 'svg':
        p = os.path.join(out_dir, f'{base_name}.svg')
        fig.savefig(p, bbox_inches='tight', pad_inches=0.02)
        saved.append(p)
    return saved


metrics_mgsa = {
    'Precision': data_mgsa['precision'],
    'Recall':    data_mgsa['recall'],
    'IoU':       data_mgsa['iou'],
    'F1':        data_mgsa['f1'],
}
metrics_cnsa = {
    'Precision': data_cnsa['precision'],
    'Recall':    data_cnsa['recall'],
    'IoU':       data_cnsa['iou'],
    'F1':        data_cnsa['f1'],
}

stats_mgsa = {
    'Variance': data_mgsa['var'],
    'Entropy':  data_mgsa['entropy'],
    'Mean':     data_mgsa['mean'],
}
stats_cnsa = {
    'Variance': data_cnsa['var'],
    'Entropy':  data_cnsa['entropy'],
    'Mean':     data_cnsa['mean'],
}


# ================= 双栏版本 =================
fig = plot_compare(metrics_mgsa, metrics_cnsa, x, xlabel,
                   fig_width=DOUBLE_FIG_WIDTH,
                   row_height=DOUBLE_ROW_HEIGHT,
                   font_size=DOUBLE_FONT_SIZE, dpi=DPI,
                   line_width=LINE_WIDTH)
for p in save_fig(fig, OUT_DIR, 'metrics_MGSA_vs_CNSA_double'):
    print(f"[double] 已保存: {p}")
plt.close(fig)

fig = plot_compare(stats_mgsa, stats_cnsa, x, xlabel,
                   fig_width=DOUBLE_FIG_WIDTH,
                   row_height=DOUBLE_ROW_HEIGHT,
                   font_size=DOUBLE_FONT_SIZE, dpi=DPI,
                   line_width=LINE_WIDTH)
for p in save_fig(fig, OUT_DIR, 'stats_MGSA_vs_CNSA_double'):
    print(f"[double] 已保存: {p}")
plt.close(fig)


# ================= 单栏版本 =================
fig = plot_compare(metrics_mgsa, metrics_cnsa, x, xlabel,
                   fig_width=SINGLE_FIG_WIDTH,
                   row_height=SINGLE_ROW_HEIGHT,
                   font_size=SINGLE_FONT_SIZE, dpi=DPI,
                   line_width=LINE_WIDTH)
for p in save_fig(fig, OUT_DIR, 'metrics_MGSA_vs_CNSA_single'):
    print(f"[single] 已保存: {p}")
plt.close(fig)

fig = plot_compare(stats_mgsa, stats_cnsa, x, xlabel,
                   fig_width=SINGLE_FIG_WIDTH,
                   row_height=SINGLE_ROW_HEIGHT,
                   font_size=SINGLE_FONT_SIZE, dpi=DPI,
                   line_width=LINE_WIDTH)
for p in save_fig(fig, OUT_DIR, 'stats_MGSA_vs_CNSA_single'):
    print(f"[single] 已保存: {p}")
plt.close(fig)

print("完成。")