# -*- coding: utf-8 -*-
"""
配对显著性检验 + 性能评价表: 单一数据集, 同时输出论文 Table 5 与 Table 6。

改动要点:
1. 单路径 FOLDER(默认真实数据 exp_test\000CSVS)。
2. restrict_to_common_samples(): TVGG/FocusNet 有效样本少于其它方法,
   所有方法统一截到全体交集(少样本子集)后再统计。
3. 同时输出两张表:
   - paper_significance_table.csv   (Table 5: 配对差均值/95% CI/Wilcoxon p/Cohen's d)
   - paper_performance_table.csv    (Table 6: μe / σe / R², 共同子集上计算)
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ================= 配置 =================
FOLDER     = r'F:\dongjiayao\Data\COCO\val\article\IEEE TIM\revise1\exp_test\000CSVS'
OUT_DIR    = os.path.join(FOLDER, 'xianzhuxing')
BASELINES  = ['cnsa', 'mgsa']    # 两个基准方法，需与文件名一致（不含 .csv，小写）
ERROR_COL  = 'abs_error'         # 可改为 'rel_error'
ALPHA      = 0.05
DPI        = 300
Z_SCALE    = 1e6                 # m -> μm
N_BOOT     = 10000
os.makedirs(OUT_DIR, exist_ok=True)

# ================= 论文表格输出配置 =================
DISPLAY_NAMES = {
    'var': 'Var', 'tog': 'ToG', 'gog': 'GoG', 'eigen': 'Eigen',
    'cnsa': 'CNSA', 'mgsa': 'MGSA',
    'tvgg': 'TVGG(Sim)', 'focusnet': 'FocusNet(Sim)',
    'tvgg(sim)': 'TVGG(Sim)', 'tvgg(retrained)': 'TVGG(Retrained)',
    'focusnet(sim)': 'FocusNet(Sim)', 'focusnet(retrained)': 'FocusNet(Retrained)',
}
PAPER_ROW_ORDER = None           # None = 按 μe 自动排序(好 -> 差)
SIGNIFICANCE_TABLE_CSV = 'paper_significance_table.csv'   # Table 5
PERFORMANCE_TABLE_CSV  = 'paper_performance_table.csv'    # Table 6

# ================= 论文风格 =================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.0,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


# ================= 读取 CSV =================
_NAME_SUFFIXES = ('_off_axis_holo', '_amplitude', '_holo')


def normalize_sample_name(name):
    stem = os.path.splitext(str(name))[0]
    for s in _NAME_SUFFIXES:
        if stem.endswith(s):
            return stem[:-len(s)]
    return stem


def load_all_csv(folder):
    data = {}
    for path in sorted(glob.glob(os.path.join(folder, '*.csv'))):
        name = os.path.splitext(os.path.basename(path))[0].lower()
        if name == 'significance_analysis':
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[警告] 读取失败 {path}: {e}")
            continue

        required = {'hologram_name', 'z_gt', 'z_pred', 'abs_error'}
        missing = required - set(df.columns)
        if missing:
            print(f"[跳过] {name}: 缺少必要列 {missing}")
            continue
        if df.empty:
            print(f"[跳过] {name}: 空文件")
            continue

        df = df.dropna(subset=['hologram_name', ERROR_COL])

        if ERROR_COL == 'abs_error':
            scale = 1e5 if float(df['z_gt'].abs().median()) > 0.01 else 1.0
            if scale != 1.0:
                print(f"[提示] {name}: z_gt 为 ×1e5 缩放单位, 已换算为米")
            df['z_gt'] = df['z_gt'] / scale
            df['z_pred'] = df['z_pred'] / scale
            df['abs_error'] = (df['z_pred'] - df['z_gt']).abs()

        norm = df['hologram_name'].map(normalize_sample_name)
        if norm.duplicated().any():
            print(f"[提示] {name}: 归一化会引入重名, 保留原始文件名配对")
        else:
            df['hologram_name'] = norm
        df = df.drop_duplicates(subset=['hologram_name'])
        data[name] = df.set_index('hologram_name')
        print(f"[读取] {name}: {len(df)} 条样本")
    if not data:
        raise RuntimeError("未读取到任何有效 CSV。")
    return data


# ================= 共同样本子集 =================
def restrict_to_common_samples(data):
    """将所有方法截到全体交集(每个方法都有效的样本)上。

    TVGG/FocusNet 有效样本少于其它方法时, 必须统一用该少样本子集统计,
    否则不同配对行使用不同 n, 均值差/CI 不满足传递性。
    """
    if not data:
        return data
    common = None
    for name, df in data.items():
        idx = df.index
        common = idx if common is None else common.intersection(idx)
    common = sorted(common)
    print(f"\n[共同样本] 全体方法交集 n = {len(common)}")
    for name, df in data.items():
        print(f"  {name}: {len(df)} -> {len(common)}")
    if len(common) == 0:
        raise RuntimeError("各方法无共同样本, 无法统一统计。")
    return {name: df.loc[common].copy() for name, df in data.items()}


# ================= 计算统计量(含 R²) =================
def compute_stats(data):
    rows = []
    for name, df in data.items():
        err = df[ERROR_COL].values.astype(float)
        z_gt = df['z_gt'].values.astype(float)
        z_pred = df['z_pred'].values.astype(float)
        ss_res = float(np.sum((z_pred - z_gt) ** 2))
        ss_tot = float(np.sum((z_gt - np.mean(z_gt)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append({
            'method': name,
            'n': len(err),
            'mean': float(np.mean(err)),
            'std': float(np.std(err, ddof=1)) if len(err) > 1 else 0.0,
            'r2': r2,
        })
    return pd.DataFrame(rows).sort_values('mean')


# ================= Table 6: 性能评价表 =================
def build_performance_table(stats_df):
    """Method | μe (μm) | σe (μm) | R² , 按 μe 从好到差排列。"""
    MINUS = '−'

    def fmt(x, nd):
        x = float(x)
        return f"{MINUS}{abs(x):.{nd}f}" if x < 0 else f"{x:.{nd}f}"

    def disp(name):
        return DISPLAY_NAMES.get(name, name.upper())

    rows = []
    for _, r in stats_df.iterrows():
        rows.append({
            'Focus Metric': disp(r['method']),
            'μe (μm)': fmt(r['mean'] * Z_SCALE, 2),
            'σe (μm)': fmt(r['std'] * Z_SCALE, 2),
            'R²': fmt(r['r2'], 4),
        })
    return pd.DataFrame(rows)


# ================= 配对检验(Table 5) =================
def pairwise_vs_baseline(data, baseline, n_boot=N_BOOT, seed=0):
    base = data[baseline]
    results = []
    for name, df in data.items():
        if name == baseline:
            continue
        common = base.index.intersection(df.index)
        if len(common) == 0:
            print(f"[警告] {name} 与基准 {baseline} 无共同样本, 跳过")
            continue
        eb = base.loc[common, ERROR_COL].values.astype(float)
        eo = df.loc[common, ERROR_COL].values.astype(float)
        diff = eb - eo   # 正值表示基准误差更大（更差）

        try:
            if np.allclose(diff, 0, atol=1e-15):
                stat, p = np.nan, 1.0
            else:
                stat, p = wilcoxon(eb, eo, zero_method='wilcox',
                                   alternative='two-sided')
        except Exception:
            stat, p = np.nan, np.nan

        rng = np.random.default_rng(seed)
        n = len(diff)
        means = np.array([np.mean(diff[rng.integers(0, n, n)])
                          for _ in range(n_boot)])
        lo, hi = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])

        sd = np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        cohen_d = float(np.mean(diff) / sd) if sd and sd > 0 else np.nan

        results.append({
            'baseline': baseline,
            'compared': name,
            'n_pairs': len(common),
            'mean_err_baseline': float(np.mean(eb)),
            'mean_err_compared': float(np.mean(eo)),
            'mean_diff': float(np.mean(diff)),
            'ci_low': float(lo),
            'ci_high': float(hi),
            'wilcoxon_stat': float(stat) if not np.isnan(stat) else np.nan,
            'p_value': float(p) if not np.isnan(p) else np.nan,
            'significant': bool(p < ALPHA) if not np.isnan(p) else False,
            'cohen_d': cohen_d,
        })
    return pd.DataFrame(results)


def build_paper_table(results_df, stats_df=None):
    """Table 5: Comparison | Mean error difference (μm) | 95% CI (μm) | p | Cohen's d"""
    MINUS = '−'

    def fmt_num(x, nd):
        x = float(x)
        return f"{MINUS}{abs(x):.{nd}f}" if x < 0 else f"{x:.{nd}f}"

    def disp(name):
        return DISPLAY_NAMES.get(name, name.upper())

    if PAPER_ROW_ORDER is not None:
        row_order = PAPER_ROW_ORDER
    else:
        perf_order = stats_df.sort_values('mean')['method'].tolist()
        row_order = [(b, [m for m in perf_order if m != b]) for b in BASELINES
                     if b in perf_order]
        print("\n[表格行序] 按平均误差(好->差):",
              ' < '.join(disp(m) for m in perf_order))

    rows = []
    seen_pairs = set()
    for baseline, others in row_order:
        for other in others:
            pair = frozenset((baseline, other))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            comp_name = f"{disp(baseline)} {MINUS} {disp(other)}"
            sub = results_df[(results_df['baseline'] == baseline)
                             & (results_df['compared'] == other)]
            if sub.empty:
                rows.append({
                    'Comparison': comp_name,
                    'Mean error difference (μm)': '',
                    '95% CI (μm)': '',
                    'p value': '',
                    "Cohen's d": '',
                })
                continue
            r = sub.iloc[0]
            p = float(r['p_value'])
            rows.append({
                'Comparison': comp_name,
                'Mean error difference (μm)': fmt_num(r['mean_diff'] * Z_SCALE, 2),
                '95% CI (μm)': f"[{fmt_num(r['ci_low'] * Z_SCALE, 2)}, "
                                f"{fmt_num(r['ci_high'] * Z_SCALE, 2)}]",
                'p value': '<0.001' if p < 0.001 else f"{p:.3f}",
                "Cohen's d": fmt_num(r['cohen_d'], 3) if not np.isnan(r['cohen_d']) else '',
            })
    return pd.DataFrame(rows)


# ================= 显著性星号 =================
def _p_to_star(p):
    if np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


# ================= 主流程 =================
def main():
    data = load_all_csv(FOLDER)
    data = restrict_to_common_samples(data)   # 关键: 统一少样本子集

    stats_df = compute_stats(data)
    methods = stats_df['method'].tolist()
    n_common = len(stats_df['n'].unique()) and stats_df['n'].iloc[0]
    print(f"\n===== 各方法统计(共同子集 n={n_common}) =====")
    print(stats_df.to_string(index=False))

    # ---------- Table 6: 性能评价表 ----------
    perf_table = build_performance_table(stats_df)
    perf_path = os.path.join(OUT_DIR, PERFORMANCE_TABLE_CSV)
    perf_table.to_csv(perf_path, index=False, encoding='utf-8-sig')
    print(f"\nTable 6 性能评价表已保存：{perf_path}")
    print(perf_table.to_string(index=False))

    # ---------- Table 5: 配对显著性表 ----------
    all_results = []
    for baseline in BASELINES:
        if baseline not in data:
            print(f"[警告] 基准 {baseline} 不存在，跳过")
            continue
        all_results.append(pairwise_vs_baseline(data, baseline))
    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    csv_path = os.path.join(OUT_DIR, 'significance_analysis.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    paper_csv = build_paper_table(results_df, stats_df)
    paper_path = os.path.join(OUT_DIR, SIGNIFICANCE_TABLE_CSV)
    paper_csv.to_csv(paper_path, index=False, encoding='utf-8-sig')
    print(f"\nTable 5 配对显著性表已保存：{paper_path}")
    print(paper_csv.to_string(index=False))

    # ---------- 图1：平均绝对误差柱状图 ----------
    fig1, ax1 = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(methods))
    means_um = stats_df['mean'].values * Z_SCALE
    stds_um = stats_df['std'].values * Z_SCALE
    colors = ['#d62728' if m in BASELINES else '#1f77b4' for m in methods]

    ax1.bar(x, means_um, yerr=stds_um, capsize=2,
            color=colors, edgecolor='black', linewidth=0.5,
            error_kw={'elinewidth': 0.7, 'ecolor': 'gray'})
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha='right')
    ax1.set_ylabel('Mean absolute error (μm)')
    ax1.set_title(f'Autofocus error comparison (n={n_common})')
    ax1.grid(axis='y', linestyle=':', linewidth=0.4, alpha=0.5)
    ax1.set_axisbelow(True)

    ymax = (means_um + stds_um).max()
    for idx, name in enumerate(methods):
        if name in BASELINES:
            continue
        p_list = []
        for baseline in BASELINES:
            if baseline not in data:
                continue
            common = data[baseline].index.intersection(data[name].index)
            if len(common) == 0:
                continue
            eb = data[baseline].loc[common, ERROR_COL].values.astype(float)
            eo = data[name].loc[common, ERROR_COL].values.astype(float)
            try:
                _, p = wilcoxon(eb, eo, zero_method='wilcox', alternative='two-sided')
            except Exception:
                p = np.nan
            p_list.append(p)
        if not p_list:
            continue
        p_min = np.nanmin(p_list)
        mark = _p_to_star(p_min)
        if mark:
            ax1.text(idx, means_um[idx] + stds_um[idx] + 0.02 * ymax,
                     mark, ha='center', va='bottom', fontsize=7, color='black')

    fig1.tight_layout()
    fig1.savefig(os.path.join(OUT_DIR, 'bar_mean_error.png'))
    fig1.savefig(os.path.join(OUT_DIR, 'bar_mean_error.pdf'))
    plt.close(fig1)

    # ---------- 图2：配对差值森林图 ----------
    n_base = sum(1 for b in BASELINES if b in data)
    fig2, axes = plt.subplots(n_base, 1, figsize=(4.0, 1.8 * n_base), squeeze=False)
    axes = axes.flatten()
    k = 0
    for baseline in BASELINES:
        if baseline not in data:
            continue
        ax = axes[k]; k += 1
        res = pairwise_vs_baseline(data, baseline)
        if res.empty:
            ax.axis('off')
            continue
        res = res.sort_values('mean_diff').reset_index(drop=True)
        y = np.arange(len(res))
        mean_um = res['mean_diff'].values * Z_SCALE
        lo_um = res['ci_low'].values * Z_SCALE
        hi_um = res['ci_high'].values * Z_SCALE
        ax.errorbar(mean_um, y,
                    xerr=[mean_um - lo_um, hi_um - mean_um],
                    fmt='o', color='#1f77b4', ecolor='gray',
                    capsize=2, markersize=3, elinewidth=0.8)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(res['compared'])
        ax.set_xlabel(f'Mean error difference vs {baseline} (μm)')
        ax.set_title(f'Pairwise comparison (baseline: {baseline})')
        ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        for yi in range(len(res)):
            if res.loc[yi, 'significant']:
                mark = _p_to_star(res.loc[yi, 'p_value'])
                ax.text(hi_um[yi] + 0.03 * abs(hi_um).max(), yi, mark,
                        va='center', fontsize=7)

    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, 'forest_plot.png'))
    fig2.savefig(os.path.join(OUT_DIR, 'forest_plot.pdf'))
    plt.close(fig2)

    print(f"\n图片已保存到：{OUT_DIR}")


if __name__ == '__main__':
    main()
