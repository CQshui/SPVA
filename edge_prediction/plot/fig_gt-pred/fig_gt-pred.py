import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

# ========== 用户可调参数 ==========
XTICK_VALUES = [250, 750]               # 硬编码的x轴刻度数值（必须为列表；面板窄, 建议只放 2 个刻度以免拥挤）
GT_LABEL_Y_OFFSET = -0.08               # GT标签相对子图底部的偏移量（负值下移，正值上移）
# =================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5


def create_combined_comparison_figure(csv_paths, method_names, output_dir, colors=None):
    os.makedirs(output_dir, exist_ok=True)

    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    n_methods = len(csv_paths)
    fig_width = 8.27
    fig_height = 2.5

    fig, axes = plt.subplots(1, n_methods, figsize=(fig_width, fig_height))
    if n_methods == 1:
        axes = [axes]

    # ---- 计算全局范围（微米）仅用于 x 轴 ----
    um_scale = 1e6
    all_gt_min = float('inf')
    all_gt_max = float('-inf')
    all_pred_min = float('inf')
    all_pred_max = float('-inf')
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        all_gt_min = min(all_gt_min, df['z_gt'].min())
        all_gt_max = max(all_gt_max, df['z_gt'].max())
        all_pred_min = min(all_pred_min, df['z_pred'].min())
        all_pred_max = max(all_pred_max, df['z_pred'].max())

    global_min = min(all_gt_min, all_pred_min)
    global_max = max(all_gt_max, all_pred_max)
    global_min_um = np.floor(global_min * um_scale)
    global_max_um = np.ceil(global_max * um_scale)
    margin_um = (global_max_um - global_min_um) * 0.02
    global_min_um -= margin_um
    global_max_um += margin_um

    metrics = []

    for i, (csv_path, method_name) in enumerate(zip(csv_paths, method_names)):
        df = pd.read_csv(csv_path)
        mae = df['abs_error'].mean()
        rmse = np.sqrt((df['abs_error'] ** 2).mean())
        r2 = 1 - np.sum((df['z_gt'] - df['z_pred']) ** 2) / np.sum((df['z_gt'] - df['z_gt'].mean()) ** 2)

        ax = axes[i]
        # 只取 80% 的点用于绘制
        df_plot = df.sample(n=int(len(df) * 0.99), random_state=42)

        gt_um = df_plot['z_gt'] * um_scale
        pred_um = df_plot['z_pred'] * um_scale

        ax.scatter(gt_um, pred_um, c=colors[i % len(colors)], s=5, alpha=0.6, linewidths=0)
        ax.plot([global_min_um, global_max_um], [global_min_um, global_max_um],
                'k--', alpha=0.5, linewidth=0.8)

        ax.set_xlim(global_min_um, global_max_um)

        # ===== 修改点：固定 y 轴范围为 -50 ~ 1050（微米） =====
        ax.set_ylim(-50, 1050)

        ax.set_aspect('equal', adjustable='box')

        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(True)

        # 设置刻度样式
        ax.tick_params(axis='both', which='major', labelsize=9, direction='in')
        # x 轴标签略小并拉开与轴的间距, 避免相邻面板刻度数字连在一起
        ax.tick_params(axis='x', labelsize=8, pad=2)
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True, min_n_ticks=2))

        # ===== 硬编码 x 轴刻度为 250, 500, 750 =====
        ax.set_xticks(XTICK_VALUES)
        ax.set_xticklabels([str(v) for v in XTICK_VALUES])

        # y轴：仅第一个子图显示标签和刻度数值
        if i == 0:
            ax.set_ylabel('Pred (μm)', fontsize=10)
            ax.tick_params(axis='y', labelleft=True)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

        ax.set_title(method_name, fontsize=10, pad=6)

        metrics.append({
            'Method': method_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })

    # 子图间无间距
    plt.subplots_adjust(wspace=0)

    # ---- 添加居中的 x 轴标签 "GT (μm)"，位置可调 ----
    ax0 = axes[0]
    bbox = ax0.get_position()
    label_y = bbox.y0 + GT_LABEL_Y_OFFSET
    fig.text(0.5, label_y, 'GT (μm)', ha='center', va='top', fontsize=10)

    plt.savefig(os.path.join(output_dir, 'combined_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ---- 性能比较条形图（不变） ----
    metrics_df = pd.DataFrame(metrics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    x_pos = np.arange(len(method_names))
    width = 0.35
    ax1.bar(x_pos - width/2, metrics_df['MAE'], width, label='MAE', color='#1f77b4', alpha=0.7)
    ax1.bar(x_pos + width/2, metrics_df['RMSE'], width, label='RMSE', color='#ff7f0e', alpha=0.7)
    ax1.set_ylabel('Error (m)', fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(method_names, fontsize=9, rotation=45)
    ax1.legend(fontsize=9, frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.bar(x_pos, metrics_df['R²'], color='#2ca02c', alpha=0.7)
    ax2.set_ylabel('R²', fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(method_names, fontsize=9, rotation=45)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    metrics_df.to_csv(os.path.join(output_dir, 'comparison_metrics.csv'), index=False)
    return metrics_df


if __name__ == "__main__":
    output_directory = r'F:\dongjiayao\Data\COCO\val\article\IEEE TIM\revise1\coco_simulation_test\000CSV'
    csv_files = [
        output_directory + r'\var.csv',
        output_directory + r'\tog.csv',
        output_directory + r'\gog.csv',
        output_directory + r'\eigen.csv',
        output_directory + r'\focusnet.csv',
        output_directory + r'\tvgg.csv',
        output_directory + r'\mgsa.csv',
        output_directory + r'\cnsa.csv',
    ]
    method_names = ['Variance', 'ToG', 'GoG', 'Eigen', 'FocusNet', 'TVGG', 'MGSA', 'CNSA']
    metrics_df = create_combined_comparison_figure(csv_files, method_names, output_directory)
    print("比较完成！生成的指标：")
    print(metrics_df)
