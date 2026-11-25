# -*- coding: utf-8 -*-
"""
分析根目录下不同方法的 focusing_results.csv（只看 abs_error），
做长尾分布分析并制图。

论文双栏风格 + 18 张分析图（部分输出两份）：
    01_box                    箱线图
    01b_violin                log 小提琴图
    01c_violin_linear         线性小提琴图
    02_log_box                log 箱线图
    03_ecdf                   ECDF
    04_ccdf_loglog            互补 CDF
    05_hist_log               各方法 log 直方图
    06_qq_normal              QQ 图
    07_tail_metrics           长尾指标
    08_mean_median_bar        均值 vs 中位数
    09_quantile_curves        分位数曲线
    10_mean_std_errorbar      均值±标准差 误差棒
    11_success_vs_threshold   成功率 vs 阈值
    12_pareto_median          帕累托（中位数）
    12_pareto_mean            帕累托（均值）
    13_mad_bar                MAD 条形图
    14_cv_bar                 变异系数
    15_cumulative_error       累积误差贡献
    16_heatmap_median_ratio   成对中位数比值热力图
    17_lognormal_fit          对数正态拟合叠加
    18_density_all_in_one     ★ 所有方法密度曲线叠加（log）
    18b_density_all_in_one_linear  ★ 所有方法密度曲线叠加（线性）

同时导出 KDE 结果（供后续分析）：
    18_density_all_in_one_curves.csv
    18_density_all_in_one_summary.csv
    18b_density_all_in_one_linear_curves.csv
    18b_density_all_in_one_linear_summary.csv
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats

# ============================================================
# 配置
# ============================================================
ROOT_DIR = r'F:\dongjiayao\Data\COCO\val\article\IEEE TIM\revise1\exp_test'
# ROOT_DIR = r'F:\dongjiayao\Data\COCO\val\article\IEEE TIM\revise1\coco_simulation_test'
CSV_DIR  = os.path.join(ROOT_DIR, '000CSVS')   # 各方法 CSV 所在目录(方法名 = 文件名去 .csv)
OUT_DIR  = os.path.join(ROOT_DIR, '_analysis_abs_error')
COL      = 'abs_error'

os.makedirs(OUT_DIR, exist_ok=True)

# 论文尺寸（英寸）
FIG_W_SINGLE = 3.5
FIG_W_1_5    = 5.0
FIG_W_DOUBLE = 7.16

SAVE_DPI     = 300
SAVE_FORMATS = ('png', 'pdf')

# ---------------- 论文风全局样式 ----------------
plt.rcParams.update({
    'font.family':        'STIXGeneral',
    'mathtext.fontset':   'stix',
    'axes.unicode_minus': False,
    'font.size':          8,
    'axes.titlesize':     8,
    'axes.labelsize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    7,
    'legend.title_fontsize': 7,
    'figure.titlesize':   8,
    'axes.linewidth':     0.6,
    'axes.edgecolor':     'black',
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   2.5,
    'ytick.major.size':   2.5,
    'xtick.minor.size':   1.3,
    'ytick.minor.size':   1.3,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'lines.linewidth':    1.0,
    'lines.markersize':   3.0,
    'patch.linewidth':    0.5,
    'grid.color':         '#bfbfbf',
    'grid.linewidth':     0.4,
    'grid.alpha':         0.6,
    'savefig.facecolor':  'white',
    'figure.facecolor':   'white',
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})


PALETTE = [
    '#1f4e79', '#c00000', '#548235', '#7030a0', '#bf8f00',
    '#2e75b6', '#a84300', '#404040', '#2f7d7d', '#a03e8e',
]


def get_colors(n):
    if n <= len(PALETTE):
        return PALETTE[:n]
    reps = int(np.ceil(n / len(PALETTE)))
    return (PALETTE * reps)[:n]


def save_fig(fig, out_dir, name):
    for fmt in SAVE_FORMATS:
        p = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(p, dpi=SAVE_DPI, format=fmt)
        print(f"[保存] {p}")


# ============================================================
# 1. 收集每个方法的 abs_error（读取 000CSV 目录下的各方法 CSV）
# ============================================================
def collect_methods(csv_dir, col):
    data = {}
    for path in sorted(Path(csv_dir).glob('*.csv')):
        name = path.stem
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[跳过] {path}: {e}")
            continue

        # 误差列: 优先 abs_error / absolute_error, 否则任一含 'error' 的列
        err_col = None
        for c in df.columns:
            cl = str(c).lower()
            if 'abs_error' in cl or 'absolute_error' in cl:
                err_col = c
                break
        if err_col is None:
            for c in df.columns:
                if 'error' in str(c).lower():
                    err_col = c
                    break
        if err_col is None:
            print(f"[跳过] {name}: 没有误差列")
            continue

        vals = pd.to_numeric(df[err_col], errors='coerce')

        # 误差口径与 fig_error_vs_foreground / xianzhuxing 保持一致:
        # 优先由 真值/预测 两列重算绝对误差(米制) —— 存储的 error 列可能
        # 带符号且单位混乱(米 或 ×1e5 缩放), 直接使用会得到负值/错误统计
        gt_col = next((c for c in ('z_gt', 'y_true', 'true_value_original')
                       if c in df.columns), None)
        pred_col = next((c for c in ('z_pred', 'y_pred', 'predicted_original')
                         if c in df.columns), None)
        note = ''
        if gt_col is not None and pred_col is not None:
            gt = pd.to_numeric(df[gt_col], errors='coerce')
            pr = pd.to_numeric(df[pred_col], errors='coerce')
            ok = gt.notna() & pr.notna()
            if ok.sum() > 0:
                scale = 1e5 if gt[ok].abs().median() > 0.01 else 1.0
                vals = (pr - gt).abs() / scale
                note = (f" (由 {gt_col}/{pred_col} 重算"
                        + (", ×1e5 缩放已换算为米)" if scale != 1.0 else ")"))
        else:
            vals = vals.abs()

        vals = vals.dropna().values.astype(np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            print(f"[跳过] {name}: 误差全为空")
            continue
        data[name] = vals
        print(f"[加载] {name:30s}  N={vals.size}{note}")
    return data


# ============================================================
# 2. 统计量
# ============================================================
def compute_stats(vals):
    vals = np.asarray(vals, dtype=np.float64)
    n = vals.size
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    median = float(np.median(vals))
    q1 = float(np.percentile(vals, 25))
    q3 = float(np.percentile(vals, 75))
    iqr = q3 - q1
    p75 = float(np.percentile(vals, 75))
    p90 = float(np.percentile(vals, 90))
    p95 = float(np.percentile(vals, 95))
    p99 = float(np.percentile(vals, 99))
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    skewness = float(stats.skew(vals, bias=False)) if n > 2 and std > 0 else 0.0
    kurtosis = float(stats.kurtosis(vals, bias=False, fisher=True)) if n > 3 and std > 0 else 0.0

    mad = float(np.median(np.abs(vals - median)))

    p99_over_p50 = p99 / median if median > 0 else float('nan')
    p95_over_p50 = p95 / median if median > 0 else float('nan')
    max_over_median = vmax / median if median > 0 else float('nan')
    if std > 0:
        tail_ratio = float(np.mean(vals > (mean + 3 * std)))
    else:
        tail_ratio = 0.0
    tail_ratio_p90 = float(np.mean(vals > p90)) if n > 0 else 0.0
    cv = std / mean if mean > 0 else float('nan')

    return dict(
        N=n, mean=mean, std=std, min=vmin, max=vmax,
        median=median, Q1=q1, Q3=q3, IQR=iqr, MAD=mad,
        P75=p75, P90=p90, P95=p95, P99=p99,
        skewness=skewness, kurtosis=kurtosis,
        cv=cv,
        P95_over_P50=p95_over_p50,
        P99_over_P50=p99_over_p50,
        max_over_median=max_over_median,
        tail_ratio_3sigma=tail_ratio,
        tail_ratio_above_P90=tail_ratio_p90,
    )


# ============================================================
# 3. 绘图工具
# ============================================================
def ecdf(vals):
    x = np.sort(vals)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle='-', linewidth=0.4,
            color='#bfbfbf', alpha=0.55)
    ax.set_axisbelow(True)


# ---------- 原有图 ----------
def plot_box(data, out_dir):
    names = list(data.keys())
    vals_list = [data[k] for k in names]
    colors = get_colors(len(names))

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.4))
    bp = ax.boxplot(
        vals_list, labels=names, widths=0.55, showfliers=True, patch_artist=True,
        boxprops=dict(linewidth=0.6, edgecolor='black'),
        whiskerprops=dict(linewidth=0.6, color='black'),
        capprops=dict(linewidth=0),
        medianprops=dict(color='#c00000', linewidth=1.0),
        flierprops=dict(marker='o', markersize=1.2,
                        markerfacecolor='#808080',
                        markeredgecolor='none', alpha=0.5),
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_ylabel('abs error')
    ax.set_title('Boxplot of abs error', pad=2)
    ax.tick_params(axis='x', rotation=20)
    _clean_axes(ax)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '01_box')
    plt.close(fig)


def plot_violin(data, out_dir, log_scale=True, filename='01b_violin'):
    names = list(data.keys())
    n = len(names)
    colors = get_colors(n)

    fig_w = FIG_W_SINGLE if n <= 5 else FIG_W_1_5
    fig, ax = plt.subplots(figsize=(fig_w, 2.6))

    can_log = all(np.sum(np.asarray(data[k]) > 0) >= 2 for k in names)
    use_log = log_scale and can_log

    plot_vals_list = []
    for k in names:
        v = np.asarray(data[k], dtype=np.float64)
        if use_log:
            v = v[v > 0]
            v = np.log10(v)
        plot_vals_list.append(v)

    positions = np.arange(1, n + 1)

    vp = ax.violinplot(
        plot_vals_list, positions=positions, widths=0.85,
        showmeans=False, showmedians=False, showextrema=False,
        bw_method=0.35,
    )
    for body, c in zip(vp['bodies'], colors):
        body.set_facecolor(c)
        body.set_edgecolor('black')
        body.set_linewidth(0.5)
        body.set_alpha(0.65)

    ax.boxplot(
        plot_vals_list, positions=positions, widths=0.10,
        showfliers=False, patch_artist=True,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.5),
        whiskerprops=dict(color='black', linewidth=0.5),
        capprops=dict(color='black', linewidth=0.5),
        medianprops=dict(color='#c00000', linewidth=1.0),
    )

    means = [np.mean(v) for v in plot_vals_list]
    ax.scatter(positions, means, marker='D', s=10,
               facecolor='#bf8f00', edgecolor='black', linewidth=0.4,
               zorder=5, label='Mean')
    ax.scatter([], [], marker='_', s=60, color='#c00000', label='Median')

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=20)
    if use_log:
        ax.set_ylabel('abs error (log scale)')
        ax.set_title('Violin plot of abs error (log)', pad=2)
    else:
        ax.set_ylabel('abs error')
        ax.set_title('Violin plot of abs error', pad=2)
    _clean_axes(ax)

    if use_log:
        ymin, ymax = ax.get_ylim()
        ticks = np.arange(np.floor(ymin), np.ceil(ymax) + 1, 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
    else:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
        ax.yaxis.get_offset_text().set_fontsize(7)

    ax.legend(loc='upper right', frameon=False, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, filename)
    plt.close(fig)


def plot_log_box(data, out_dir):
    names = list(data.keys())
    vals_list = [data[k] for k in names]
    colors = get_colors(len(names))

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.4))
    bp = ax.boxplot(
        vals_list, labels=names, widths=0.55, showfliers=True, patch_artist=True,
        boxprops=dict(linewidth=0.6, edgecolor='black'),
        whiskerprops=dict(linewidth=0.6, color='black'),
        capprops=dict(linewidth=0),
        medianprops=dict(color='#c00000', linewidth=1.0),
        flierprops=dict(marker='o', markersize=1.2,
                        markerfacecolor='#808080',
                        markeredgecolor='none', alpha=0.5),
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_yscale('log')
    ax.set_ylabel('abs error (log)')
    ax.set_title('Boxplot of abs error (log)', pad=2)
    ax.tick_params(axis='x', rotation=20)
    _clean_axes(ax)
    ax.grid(True, which='both', alpha=0.4)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '02_log_box')
    plt.close(fig)


def plot_ecdf(data, out_dir):
    colors = get_colors(len(data))
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.4))
    for (name, vals), c in zip(data.items(), colors):
        x, y = ecdf(vals)
        ax.plot(x, y, label=f"{name} (N={vals.size})", linewidth=1.0, color=c)
    ax.set_xlabel('abs error')
    ax.set_ylabel('ECDF')
    ax.set_title('ECDF of abs error', pad=2)
    _clean_axes(ax)
    ax.legend(loc='lower right', frameon=False, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '03_ecdf')
    plt.close(fig)


def plot_ccdf_log(data, out_dir):
    colors = get_colors(len(data))
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.4))
    for (name, vals), c in zip(data.items(), colors):
        x = np.sort(vals)
        n = x.size
        ccdf = 1.0 - np.arange(1, n + 1) / n
        mask = ccdf > 0
        ax.plot(x[mask], ccdf[mask], label=f"{name} (N={n})",
                linewidth=1.0, color=c)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('abs error')
    ax.set_ylabel(r'$P(X > x)$')
    ax.set_title('Complementary CDF (log-log)', pad=2)
    _clean_axes(ax)
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='lower left', frameon=False, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '04_ccdf_loglog')
    plt.close(fig)


def plot_hist_log(data, out_dir):
    names = list(data.keys())
    n = len(names)
    cols = 3
    rows = math.ceil(n / cols)
    colors = get_colors(n)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(FIG_W_DOUBLE, 1.8 * rows), squeeze=False)
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        vals = data[name]
        pos = vals[vals > 0]
        if pos.size > 0:
            bins = np.logspace(np.log10(max(pos.min(), 1e-9)),
                               np.log10(pos.max() + 1e-12), 30)
            ax.hist(pos, bins=bins, color=colors[i], alpha=0.85,
                    edgecolor='white', linewidth=0.3)
            ax.set_xscale('log')
        else:
            ax.hist(vals, bins=30, color=colors[i], alpha=0.85,
                    edgecolor='white', linewidth=0.3)
        ax.set_title(name, fontsize=7, pad=1.5)
        ax.set_xlabel('abs error', labelpad=1)
        ax.set_ylabel('count', labelpad=1)
        _clean_axes(ax)
        ax.grid(True, which='both', alpha=0.35)

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis('off')

    plt.tight_layout(pad=0.25, h_pad=0.6, w_pad=0.5)
    save_fig(fig, out_dir, '05_hist_log')
    plt.close(fig)


def plot_qq(data, out_dir):
    names = list(data.keys())
    n = len(names)
    cols = 3
    rows = math.ceil(n / cols)
    colors = get_colors(n)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(FIG_W_DOUBLE, 1.8 * rows), squeeze=False)
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        vals = data[name]
        if vals.size >= 3 and np.std(vals) > 0:
            z = (vals - vals.mean()) / vals.std(ddof=1)
            (osm, osr), (slope, intercept, _) = stats.probplot(z, dist="norm")
            ax.scatter(osm, osr, s=2, color=colors[i], alpha=0.7,
                       edgecolor='none')
            ax.plot(osm, slope * osm + intercept, color='#c00000', linewidth=0.9)
            ax.set_title(name, fontsize=7, pad=1.5)
            ax.set_xlabel('Theoretical', labelpad=1)
            ax.set_ylabel('Sample', labelpad=1)
            _clean_axes(ax)
            ax.grid(True, alpha=0.4)
        else:
            ax.axis('off')

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis('off')

    plt.tight_layout(pad=0.25, h_pad=0.6, w_pad=0.5)
    save_fig(fig, out_dir, '06_qq_normal')
    plt.close(fig)


def plot_tail_metrics(summary_df, out_dir):
    df = summary_df.copy()
    metrics = [
        ('skewness', 'Skewness'),
        ('kurtosis', 'Excess Kurtosis'),
        ('P95_over_P50', r'$P_{95}/P_{50}$'),
        ('P99_over_P50', r'$P_{99}/P_{50}$'),
        ('max_over_median', r'$\max/\mathrm{median}$'),
    ]
    n = len(metrics)
    cols = 3
    rows = math.ceil(n / cols)
    colors = get_colors(len(df))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(FIG_W_DOUBLE, 1.9 * rows), squeeze=False)
    for i, (key, title) in enumerate(metrics):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        vals = df[key].values
        ax.bar(df.index, vals, color=colors, alpha=0.9,
               edgecolor='black', linewidth=0.4)
        ax.set_title(title, fontsize=7, pad=1.5)
        ax.tick_params(axis='x', rotation=28, labelsize=6)
        _clean_axes(ax)
        if key in ('skewness', 'kurtosis'):
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axhline(3, color='#c00000', linewidth=0.7,
                       linestyle='--', alpha=0.8)

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis('off')

    plt.tight_layout(pad=0.25, h_pad=0.6, w_pad=0.5)
    save_fig(fig, out_dir, '07_tail_metrics')
    plt.close(fig)


def plot_mean_median_bar(summary_df, out_dir):
    df = summary_df.copy()
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.4))
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w / 2, df['median'].values, width=w,
           label='Median', color='#1f4e79',
           edgecolor='black', linewidth=0.4, alpha=0.9)
    ax.bar(x + w / 2, df['mean'].values, width=w,
           label='Mean', color='#bf8f00',
           edgecolor='black', linewidth=0.4, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20)
    ax.set_ylabel('abs error')
    ax.set_title('Mean vs Median', pad=2)
    _clean_axes(ax)
    ax.legend(frameon=False, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '08_mean_median_bar')
    plt.close(fig)


def plot_quantile_curves(summary_df, out_dir):
    df = summary_df.copy()
    qs = ['min', 'Q1', 'median', 'P75', 'P90', 'P95', 'P99', 'max']
    colors = get_colors(len(df))

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.6))
    for (name, row), c in zip(df.iterrows(), colors):
        y = [row[q] for q in qs]
        ax.plot(qs, y, marker='o', markersize=2.5,
                label=name, linewidth=1.0, color=c)
    ax.set_yscale('log')
    ax.set_ylabel('abs error (log)')
    ax.set_xlabel('Quantile')
    ax.set_title('Quantile curves', pad=2)
    _clean_axes(ax)
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='best', frameon=False, ncol=1,
              handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '09_quantile_curves')
    plt.close(fig)


# ============================================================
# 4. 新增图
# ============================================================

def plot_mean_std_errorbar(summary_df, out_dir):
    """均值 ± 标准差 误差棒"""
    df = summary_df.copy()

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.4))
    x = np.arange(len(df))
    mean = df['mean'].values
    std = df['std'].values

    ax.errorbar(
        x, mean, yerr=std,
        fmt='o', markersize=3.5, linewidth=0.8, capsize=2.0,
        color='#1f4e79', ecolor='#1f4e79',
        markerfacecolor='#1f4e79', markeredgecolor='black',
        markeredgewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20)
    ax.set_ylabel('abs error')
    ax.set_title('Mean ± std', pad=2)
    _clean_axes(ax)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '10_mean_std_errorbar')
    plt.close(fig)


def plot_success_vs_threshold(data, out_dir, n_points=200):
    """成功率 vs 阈值曲线"""
    colors = get_colors(len(data))

    all_vals = np.concatenate([np.asarray(v) for v in data.values()])
    vmin = max(float(np.min(all_vals)), 1e-9)
    vmax = float(np.max(all_vals))

    use_log = vmax / vmin > 100
    if use_log:
        thresholds = np.logspace(np.log10(vmin), np.log10(vmax), n_points)
    else:
        thresholds = np.linspace(0, vmax, n_points)

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.6))
    for (name, vals), c in zip(data.items(), colors):
        vals = np.asarray(vals)
        rates = [(vals <= t).mean() for t in thresholds]
        ax.plot(thresholds, rates, linewidth=1.1, color=c, label=name)

    if use_log:
        ax.set_xscale('log')
        ax.set_xlabel('Threshold (abs error, log)')
    else:
        ax.set_xlabel('Threshold (abs error)')
    ax.set_ylabel('Success rate')
    ax.set_ylim(0, 1.02)
    ax.set_title('Success rate vs threshold', pad=2)
    _clean_axes(ax)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(loc='lower right', frameon=False, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '11_success_vs_threshold')
    plt.close(fig)


def plot_pareto(data, out_dir, metric='median'):
    """帕累托图"""
    names = list(data.keys())
    vals = np.array([np.median(data[k]) if metric == 'median'
                     else np.mean(data[k]) for k in names])
    order = np.argsort(vals)
    names_sorted = [names[i] for i in order]
    vals_sorted = vals[order]

    cum = np.cumsum(vals_sorted)
    cum_pct = cum / cum[-1] * 100 if cum[-1] > 0 else cum

    fig, ax1 = plt.subplots(figsize=(FIG_W_SINGLE, 2.6))
    x = np.arange(len(names_sorted))
    colors = get_colors(len(names_sorted))

    ax1.bar(x, vals_sorted, color=colors, alpha=0.85,
            edgecolor='black', linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names_sorted, rotation=25)
    ax1.set_ylabel(f'{metric} abs error')
    ax1.set_title(f'Pareto ({metric})', pad=2)
    _clean_axes(ax1)

    ax2 = ax1.twinx()
    ax2.plot(x, cum_pct, color='#c00000', marker='o', markersize=2.8,
             linewidth=1.0, label='Cumulative %')
    ax2.set_ylabel('Cumulative (%)', color='#c00000')
    ax2.tick_params(axis='y', colors='#c00000')
    ax2.set_ylim(0, 105)
    ax2.spines['top'].set_visible(False)

    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, f'12_pareto_{metric}')
    plt.close(fig)


def plot_mad_bar(summary_df, out_dir):
    """MAD 条形图"""
    df = summary_df.copy()
    colors = get_colors(len(df))

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.4))
    x = np.arange(len(df))
    ax.bar(x, df['MAD'].values, color=colors, alpha=0.9,
           edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20)
    ax.set_ylabel('MAD')
    ax.set_title('Median absolute deviation', pad=2)
    _clean_axes(ax)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '13_mad_bar')
    plt.close(fig)


def plot_cv_bar(summary_df, out_dir):
    """变异系数 CV 条形图"""
    df = summary_df.copy()
    colors = get_colors(len(df))

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.4))
    x = np.arange(len(df))
    ax.bar(x, df['cv'].values, color=colors, alpha=0.9,
           edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=20)
    ax.set_ylabel('CV = std / mean')
    ax.set_title('Coefficient of variation', pad=2)
    _clean_axes(ax)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '14_cv_bar')
    plt.close(fig)


def plot_cumulative_error(data, out_dir):
    """累积误差贡献曲线"""
    colors = get_colors(len(data))

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 2.6))
    for (name, vals), c in zip(data.items(), colors):
        v = np.sort(np.asarray(vals, dtype=np.float64))[::-1]
        cum = np.cumsum(v)
        if cum[-1] > 0:
            cum = cum / cum[-1]
        x = np.arange(1, v.size + 1) / v.size
        ax.plot(x, cum, linewidth=1.1, color=c, label=name)

    ax.plot([0, 1], [0, 1], color='#808080', linewidth=0.8,
            linestyle='--', label='Uniform')
    ax.set_xlabel('Fraction of samples (worst → best)')
    ax.set_ylabel('Cumulative error fraction')
    ax.set_title('Cumulative error contribution', pad=2)
    _clean_axes(ax)
    ax.grid(True, alpha=0.35)
    ax.legend(loc='lower right', frameon=False, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '15_cumulative_error')
    plt.close(fig)


def plot_heatmap_median_ratio(data, out_dir):
    """成对中位数比值热力图"""
    names = list(data.keys())
    n = len(names)
    medians = np.array([np.median(data[k]) for k in names])

    eps = 1e-12
    M = (medians[:, None] + eps) / (medians[None, :] + eps)

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_W_SINGLE))
    im = ax.imshow(np.log10(M), cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title('Pairwise median ratio ($\\log_{10}$)', pad=2)

    for i in range(n):
        for j in range(n):
            val = M[i, j]
            txt = f"{val:.2f}" if val < 100 else f"{val:.1e}"
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=5, color='black')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r'$\log_{10}(\mathrm{median}_i / \mathrm{median}_j)$',
                   fontsize=7)

    plt.tight_layout(pad=0.15)
    save_fig(fig, out_dir, '16_heatmap_median_ratio')
    plt.close(fig)


def plot_lognormal_fit(data, out_dir):
    """对数正态拟合叠加（每方法一个子图）"""
    names = list(data.keys())
    n = len(names)
    cols = 3
    rows = math.ceil(n / cols)
    colors = get_colors(n)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(FIG_W_DOUBLE, 1.9 * rows), squeeze=False)
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        vals = np.asarray(data[name])
        pos = vals[vals > 0]
        if pos.size < 5:
            ax.axis('off')
            continue

        logv = np.log10(pos)
        ax.hist(logv, bins=30, density=True, color=colors[i],
                alpha=0.6, edgecolor='white', linewidth=0.3,
                label='Data')
        mu, sigma = logv.mean(), logv.std(ddof=1)
        xs = np.linspace(logv.min(), logv.max(), 200)
        pdf = stats.norm.pdf(xs, mu, sigma)
        ax.plot(xs, pdf, color='#c00000', linewidth=1.0,
                label=f'N({mu:.2f}, {sigma:.2f})')

        ticks = np.arange(np.floor(logv.min()),
                          np.ceil(logv.max()) + 1, 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
        ax.set_title(name, fontsize=7, pad=1.5)
        ax.set_xlabel('abs error', labelpad=1)
        ax.set_ylabel('density', labelpad=1)
        _clean_axes(ax)
        ax.grid(True, alpha=0.35)
        ax.legend(loc='upper right', frameon=False,
                  fontsize=6, handlelength=1.2, handletextpad=0.4)

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis('off')

    plt.tight_layout(pad=0.25, h_pad=0.6, w_pad=0.5)
    save_fig(fig, out_dir, '17_lognormal_fit')
    plt.close(fig)


def _trapz(y, x):
    """兼容不同 numpy 版本的梯形积分"""
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def plot_density_all_in_one(data, out_dir, log_scale=True, bw_method=0.35,
                            filename='18_density_all_in_one',
                            x_min=1e-7,
                            legend_loc='upper left',
                            legend_order=None,
                            highlight=('MGSA', 'CNSA'),
                            highlight_lw=2.2,
                            normal_lw=1.0,
                            normal_alpha=0.55,
                            save_kde_csv=True):
    """
    所有方法的 KDE 密度曲线画在同一坐标系。

    highlight     : 需要高亮加粗的方法名（元组/列表），其余曲线淡化
    highlight_lw  : 高亮曲线线宽
    normal_lw     : 普通曲线线宽
    normal_alpha  : 普通曲线透明度
    legend_loc    : 图例位置
    legend_order  : 图例顺序（方法名列表），未列出的按原顺序接在后面
    save_kde_csv  : 是否导出 KDE 曲线数据 + 摘要 CSV

    导出文件（当 save_kde_csv=True）：
        {filename}_curves.csv   ：method, log_scale, x, density
        {filename}_summary.csv  ：每个方法 KDE 关键参数
    """
    names = list(data.keys())
    colors = get_colors(len(names))

    HIGHLIGHT_COLORS = {
        'MGSA': '#c00000',   # 深红
        'CNSA': '#1f4e79',   # 深蓝
    }
    FALLBACK_HL = ['#c00000', '#1f4e79', '#548235', '#7030a0']
    hl_set = set(highlight) if highlight else set()

    # 单栏宽 3.5 in，高度略高一点给大字体留空间
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, 3.0))

    can_log = all(np.sum(np.asarray(data[k]) > 0) >= 2 for k in names)
    use_log = log_scale and can_log

    if use_log:
        x_lo = math.log10(x_min)
    else:
        x_lo = x_min

    legend_items = {}

    # 先画普通曲线，再画高亮曲线，保证高亮在最上层
    draw_order = ([n for n in names if n not in hl_set]
                  + [n for n in names if n in hl_set])

    # ---------- 用于保存 KDE 结果 ----------
    kde_curve_records = []     # 逐点曲线
    kde_summary_records = []   # 每方法摘要

    hl_idx = 0
    for name in draw_order:
        vals = data.get(name, None)
        if vals is None:
            continue

        if name in hl_set:
            if name in HIGHLIGHT_COLORS:
                c = HIGHLIGHT_COLORS[name]
            else:
                c = FALLBACK_HL[hl_idx % len(FALLBACK_HL)]
                hl_idx += 1
            lw = highlight_lw
            alpha_fill = 0.22
            z = 5
            line_alpha = 1.0
        else:
            c = colors[names.index(name)]
            lw = normal_lw
            alpha_fill = 0.07
            z = 2
            line_alpha = normal_alpha

        v = np.asarray(vals, dtype=np.float64)
        if use_log:
            v = v[v > 0]
            v = np.log10(v)
        if v.size < 2 or v.std() == 0:
            continue

        x_hi = v.max()
        x_lo_local = x_hi - 1.0 if x_lo >= x_hi else x_lo
        xs = np.linspace(x_lo_local, x_hi, 400)

        kde = stats.gaussian_kde(v, bw_method=bw_method)
        ys = kde(xs)

        line, = ax.plot(xs, ys,
                        linewidth=lw,
                        color=c,
                        alpha=line_alpha,
                        zorder=z,
                        label=name)
        ax.fill_between(xs, 0, ys, color=c, alpha=alpha_fill,
                        linewidth=0, zorder=z - 1)

        legend_items[name] = line

        # ---------- 收集 KDE 数据 ----------
        if save_kde_csv:
            # 曲线逐点
            for xv, yv in zip(xs, ys):
                kde_curve_records.append({
                    'method': name,
                    'log_scale': int(bool(use_log)),
                    'x': float(xv),
                    'density': float(yv),
                })
            # 摘要
            peak_idx = int(np.argmax(ys))
            # gaussian_kde 的 effective bandwidth：kde.factor（乘以数据标准差）
            try:
                bw_factor = float(kde.factor)
            except Exception:
                bw_factor = float('nan')
            kde_summary_records.append({
                'method':            name,
                'log_scale':         int(bool(use_log)),
                'N_samples':         int(v.size),
                'kde_bw_method':     float(bw_method) if not isinstance(bw_method, str) else -1.0,
                'kde_bw_factor':     bw_factor,
                'bw_method_str':     str(bw_method),
                'domain_min':        float(xs[0]),
                'domain_max':        float(xs[-1]),
                'mean':              float(np.mean(v)),
                'std':               float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
                'median':            float(np.median(v)),
                'peak_x':            float(xs[peak_idx]),
                'peak_density':      float(ys[peak_idx]),
                'integral':          _trapz(ys, xs),
                'is_highlight':      int(name in hl_set),
            })

    # ---------- 坐标轴（含单位）----------
    if use_log:
        ax.set_xlim(math.log10(x_min), None)
        xmin, xmax = ax.get_xlim()
        ticks = np.arange(np.floor(xmin), np.ceil(xmax) + 1, 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
        ax.set_xlabel(r'$|z - z_{\mathrm{gt}}|$  (m)',
                      fontsize=12, labelpad=5)
        ax.set_ylabel(r'KDE density  (m$^{-1}$)',
                      fontsize=12, labelpad=5)
        ax.set_title('KDE density', fontsize=12, pad=7)
    else:
        ax.set_xlim(x_min, None)
        ax.set_xlabel(r'$|z - z_{\mathrm{gt}}|$  (m)',
                      fontsize=12, labelpad=5)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
        ax.xaxis.get_offset_text().set_fontsize(11)
        ax.set_ylabel(r'KDE density  (m$^{-1}$)',
                      fontsize=12, labelpad=5)
        ax.set_title('KDE density', fontsize=12, pad=7)

    # 刻度字号调大
    ax.tick_params(axis='both', which='major',
                   labelsize=11, length=3.2, width=0.7, pad=2.5)
    ax.tick_params(axis='both', which='minor',
                   length=1.8, width=0.6)

    _clean_axes(ax)
    ax.grid(True, which='both', alpha=0.35)

    # ---------- 图例顺序 ----------
    if legend_order is None:
        ordered_names = [n for n in data.keys() if n in legend_items]
    else:
        ordered_names = [n for n in legend_order if n in legend_items]
        for n in data.keys():
            if n in legend_items and n not in ordered_names:
                ordered_names.append(n)

    handles = [legend_items[n] for n in ordered_names]
    labels = ordered_names

    leg = ax.legend(handles, labels,
                    loc=legend_loc,
                    frameon=False,
                    fontsize=9.5,
                    handlelength=1.8,
                    handletextpad=0.5,
                    ncol=1,
                    borderaxespad=0.3,
                    labelspacing=0.32)

    # 高亮方法图例线也加粗
    for txt, handle in zip(leg.get_texts(), leg.legend_handles):
        if txt.get_text() in hl_set:
            handle.set_linewidth(highlight_lw)
            handle.set_alpha(1.0)
            handle.set_alpha(1.0)
            txt.set_fontweight('bold')

    plt.tight_layout(pad=0.2)
    save_fig(fig, out_dir, filename)
    plt.close(fig)

    # ---------- 导出 KDE 数据 ----------
    if save_kde_csv and kde_curve_records:
        curves_csv = os.path.join(out_dir, f'{filename}_curves.csv')
        curves_df = pd.DataFrame(kde_curve_records)
        curves_df.to_csv(curves_csv, index=False, float_format='%.8e')
        print(f"[保存] {curves_csv}")

        summary_csv = os.path.join(out_dir, f'{filename}_summary.csv')
        summary_df = pd.DataFrame(kde_summary_records)
        # 按峰值密度降序更直观
        summary_df = summary_df.sort_values(
            by=['is_highlight', 'peak_density'],
            ascending=[False, False]
        ).reset_index(drop=True)
        summary_df.to_csv(summary_csv, index=False, float_format='%.8f')
        print(f"[保存] {summary_csv}")


# ============================================================
# 5. 主流程
# ============================================================
def main():
    print("=" * 70)
    print("收集方法文件夹 ...")
    print("=" * 70)
    data = collect_methods(CSV_DIR, COL)
    if not data:
        print("没有找到任何有效 CSV。请检查 ROOT_DIR / CSV_NAME / 列名。")
        return

    print("\n" + "=" * 70)
    print("计算统计量 ...")
    print("=" * 70)
    stats_dict = {name: compute_stats(vals) for name, vals in data.items()}
    summary_df = pd.DataFrame(stats_dict).T
    summary_df = summary_df.sort_values('median')
    summary_df.index.name = 'method'

    csv_out = os.path.join(OUT_DIR, 'summary_statistics.csv')
    summary_df.to_csv(csv_out, float_format='%.8f')
    print(f"[保存] {csv_out}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda v: f"{v:.6g}")
    print("\n汇总统计（按 median 升序）：")
    print(summary_df)

    print("\n" + "=" * 70)
    print("长尾判定参考：")
    print("  skewness > 1  → 明显右偏（长尾）")
    print("  kurtosis > 3  → 明显重尾")
    print("  P99/P50 > 5   → 极端值相对中位数很大")
    print("  max/median 越大 → 单点极端越长")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("绘图 ...")
    print("=" * 70)

    # ---- 原有 ----
    plot_box(data, OUT_DIR)
    plot_violin(data, OUT_DIR, log_scale=True,  filename='01b_violin')
    plot_violin(data, OUT_DIR, log_scale=False, filename='01c_violin_linear')
    plot_log_box(data, OUT_DIR)
    plot_ecdf(data, OUT_DIR)
    plot_ccdf_log(data, OUT_DIR)
    plot_hist_log(data, OUT_DIR)
    plot_qq(data, OUT_DIR)
    plot_tail_metrics(summary_df, OUT_DIR)
    plot_mean_median_bar(summary_df, OUT_DIR)
    plot_quantile_curves(summary_df, OUT_DIR)

    # ---- 新增 ----
    plot_mean_std_errorbar(summary_df, OUT_DIR)
    plot_success_vs_threshold(data, OUT_DIR)
    plot_pareto(data, OUT_DIR, metric='median')
    plot_pareto(data, OUT_DIR, metric='mean')
    plot_mad_bar(summary_df, OUT_DIR)
    plot_cv_bar(summary_df, OUT_DIR)
    plot_cumulative_error(data, OUT_DIR)
    plot_heatmap_median_ratio(data, OUT_DIR)
    plot_lognormal_fit(data, OUT_DIR)

    LEGEND_ORDER = ['Var', 'ToG', 'GoG', 'Eigen',
                    'FocusNet(Sim)', 'FocusNet(Retrained)',
                    'TVGG(Sim)', 'TVGG(Retrained)',
                    'MGSA', 'CNSA']

    plot_density_all_in_one(data, OUT_DIR, log_scale=True,
                            filename='18_density_all_in_one',
                            legend_loc='upper left',
                            legend_order=LEGEND_ORDER,
                            save_kde_csv=True)
    plot_density_all_in_one(data, OUT_DIR, log_scale=False,
                            filename='18b_density_all_in_one_linear',
                            legend_loc='upper left',
                            legend_order=LEGEND_ORDER,
                            save_kde_csv=True)

    print("\n全部完成。输出目录：", OUT_DIR)


if __name__ == "__main__":
    main()