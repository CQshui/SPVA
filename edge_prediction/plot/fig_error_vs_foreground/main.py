"""
File: simple_error_analysis_final_longtail.py
Purpose: Publication-quality error analysis for long-tailed distributions
         Boxplot, median-IQR bar chart, CDF, KDE with log scale
         Paired non-parametric tests with bootstrap CI and effect size
         Foreground ratio vs focus error analysis (batch for all methods)
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

# ==============================
# 1. Font Configuration
# ==============================
def setup_times_new_roman():
    available_fonts = {f.name.lower(): f.name for f in fm.fontManager.ttflist}
    times_variants = [
        'Times New Roman', 'Times', 'Liberation Serif',
        'DejaVu Serif', 'FreeSerif'
    ]
    font_found = None
    for variant in times_variants:
        if variant.lower() in available_fonts.values() or any(
            variant.lower() in f.lower() for f in available_fonts.keys()
        ):
            font_found = variant
            break
    if font_found:
        print(f"✓ Using font: {font_found}")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [font_found]
    else:
        print("⚠ Times New Roman not found. Using DejaVu Serif fallback")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    return font_found or 'DejaVu Serif'

font_name = setup_times_new_roman()

# ==============================
# 2. Enhanced Publication Settings
# ==============================
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10

plt.rcParams.update({
    'font.size': FONT_SIZE_LABEL,
    'axes.titlesize': FONT_SIZE_TITLE,
    'axes.labelsize': FONT_SIZE_LABEL,
    'xtick.labelsize': FONT_SIZE_TICK,
    'ytick.labelsize': FONT_SIZE_TICK,
    'legend.fontsize': FONT_SIZE_LEGEND,
    'figure.titlesize': FONT_SIZE_TITLE,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.0,
    'grid.linewidth': 0.6,
    'grid.alpha': 0.25,
    'grid.linestyle': ':',
    'grid.color': '#CCCCCC',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#666666',
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.borderpad': 0.5,
})

COLORS = {
    'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c',
    'red': '#d62728', 'purple': '#9467bd', 'brown': '#8c564b',
    'pink': '#e377c2', 'gray': '#7f7f7f', 'olive': '#bcbd22',
    'cyan': '#17becf'
}

# ==============================
# 3. Configuration
# ==============================
folder_path = Path(r"F:\dongjiayao\Data\COCO\val\article\IEEE TIM\revise1\exp_test\000CSVS")
if not folder_path.exists():
    folder_path = Path(".")
    print(f"⚠ Using current directory: {folder_path.absolute()}")
else:
    print(f"✓ Input folder: {folder_path}")

OUTPUT_DIR = folder_path / "statistical_analysis_results_longtail"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Output folder: {OUTPUT_DIR}")

FIRST_FILE_NAME = "ours"
OUTPUT_DPI = 1200
MAX_SIZE_CM = 8.5
MAX_SIZE_INCHES = MAX_SIZE_CM / 2.54

# ---------- 方法白名单 + 顺序（填写 CSV 文件名去掉 .csv 后的方法名） ----------
# 只有列在这里的方法会被统计和绘制；顺序 = 从左到右、从上到下。
# 留空列表 [] 表示不过滤（全部方法参与）。
CUSTOM_ORDER = [
    "var",
    "tog",
    "gog",
    "eigen",
    # "focusnet(sim)",
    "focusnet(retrained)",
    "tvgg(retrained)",
    "mgsa",
    "cnsa",
]


def norm_method_name(s):
    """忽略大小写/空格/下划线/连字符/括号的方法名规范化, 用于白名单匹配"""
    return (
        str(s).strip().lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )


_CUSTOM_ORDER_KEYS = {norm_method_name(x) for x in CUSTOM_ORDER}
# ---------------------------------------------------------------------------

print(f"📐 Square format: {MAX_SIZE_CM} cm × {MAX_SIZE_CM} cm")

# 长尾相关配置
FAILURE_THRESHOLDS_UM = [5, 10, 20, 50]
USE_LOG_SCALE_FOR_KDE = True
USE_LOG_SCALE_FOR_BOX = True
PAIRED_TEST = "wilcoxon"
BOOTSTRAP_N = 10000
BOOTSTRAP_RANDOM_SEED = 20260918

ID_COLUMN_CANDIDATES = [
    "id", "ID", "sample_id", "SampleID", "sample", "Sample",
    "image", "Image", "image_id", "ImageID",
    "filename", "Filename", "file", "File",
    "name", "Name", "hologram_name"
]

# ==============================
# 4. Data Loading + Enhanced Statistics
# ==============================
csv_files = list(folder_path.glob("*.csv"))
if not csv_files:
    print(f"❌ Error: No CSV files found in {folder_path}")
    exit()

print(f"✓ Found {len(csv_files)} CSV files")

first_file = None
other_files = []
for csv_file in csv_files:
    if csv_file.stem == FIRST_FILE_NAME:
        first_file = csv_file
    else:
        other_files.append(csv_file)

if first_file:
    csv_files_sorted = [first_file] + sorted(other_files, key=lambda x: x.name)
    print(f"✓ '{FIRST_FILE_NAME}' will be plotted first")
else:
    csv_files_sorted = sorted(csv_files, key=lambda x: x.name)
    print(f"⚠ '{FIRST_FILE_NAME}.csv' not found, using alphabetical order")

method_data = {}
method_frames = {}
method_stats = {}

for csv_file in csv_files_sorted:
    method_name = csv_file.stem
    print(f"\n📊 Processing: {method_name}")

    try:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            with open(csv_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            if ',' in first_line:
                column_names = [col.strip() for col in first_line.split(',')]
                df = pd.read_csv(csv_file, skiprows=1, header=None)
                if df.shape[1] == 1:
                    df = df[0].str.split(',', expand=True)
                if df.shape[1] == len(column_names):
                    df.columns = column_names
            else:
                raise ValueError("Unable to parse CSV file")

        abs_error_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'abs_error' in col_lower or 'absolute_error' in col_lower:
                abs_error_col = col
                break
        if abs_error_col is None:
            for col in df.columns:
                col_lower = str(col).lower()
                if 'error' in col_lower:
                    abs_error_col = col
                    break
        if abs_error_col is None:
            print(f"  ❌ No error column found")
            continue

        df = df.copy()
        df["_abs_error_numeric"] = pd.to_numeric(
            df[abs_error_col], errors='coerce'
        )

        # ---- 统一误差口径: 优先由 真值/预测 两列重算绝对误差(米制) ----
        # 部分 CSV 存储的 error 列带符号且单位混乱(米 或 ×1e5 缩放),
        # 直接使用会导致负均值、负 CV 等错误统计(log 轴绘图时还会撑爆画布)
        gt_col = next((c for c in ('z_gt', 'y_true', 'true_value_original')
                       if c in df.columns), None)
        pred_col = next((c for c in ('z_pred', 'y_pred', 'predicted_original')
                         if c in df.columns), None)
        if gt_col is not None and pred_col is not None:
            gt = pd.to_numeric(df[gt_col], errors='coerce')
            pr = pd.to_numeric(df[pred_col], errors='coerce')
            ok = gt.notna() & pr.notna()
            if ok.sum() > 0:
                scale = 1e5 if gt[ok].abs().median() > 0.01 else 1.0
                df.loc[ok, "_abs_error_numeric"] = (pr[ok] - gt[ok]).abs() / scale
                print(f"  ✓ 由 {gt_col}/{pred_col} 重算 abs_error"
                      f"{' (×1e5 缩放单位已换算为米)' if scale != 1.0 else ''}")
            else:
                df["_abs_error_numeric"] = df["_abs_error_numeric"].abs()
        else:
            df["_abs_error_numeric"] = df["_abs_error_numeric"].abs()

        valid = df["_abs_error_numeric"].notna()
        df = df.loc[valid].copy()
        if len(df) == 0:
            print(f"  ❌ No valid data")
            continue

        abs_errors = df["_abs_error_numeric"].astype(float)
        method_data[method_name] = abs_errors.reset_index(drop=True)
        method_frames[method_name] = df.reset_index(drop=True)

        q1 = float(abs_errors.quantile(0.25))
        median = float(abs_errors.quantile(0.50))
        q3 = float(abs_errors.quantile(0.75))
        p95 = float(abs_errors.quantile(0.95))
        p99 = float(abs_errors.quantile(0.99))
        skewness = float(abs_errors.skew())
        kurtosis = float(abs_errors.kurtosis())

        failure_info = {}
        for th in FAILURE_THRESHOLDS_UM:
            th_m = th * 1e-6
            cnt = int((abs_errors > th_m).sum())
            rate = cnt / len(abs_errors) * 100.0
            failure_info[th] = {'count': cnt, 'rate': rate}

        method_stats[method_name] = {
            'count': int(len(abs_errors)),
            'mean': float(abs_errors.mean()),
            'std': float(abs_errors.std()),
            'min': float(abs_errors.min()),
            'max': float(abs_errors.max()),
            'median': median,
            'q1': q1,
            'q3': q3,
            'iqr': q3 - q1,
            'p95': p95,
            'p99': p99,
            'skew': skewness,
            'kurtosis': kurtosis,
            'sem': float(abs_errors.sem()),
            'failure_info': failure_info,
        }

        fail_str = ", ".join([
            f">{th}μm: {failure_info[th]['rate']:.1f}%"
            for th in FAILURE_THRESHOLDS_UM
        ])
        print(
            f"  ✓ Loaded {len(abs_errors)} points "
            f"(median={median*1e6:.2f} μm, "
            f"IQR={ (q3-q1)*1e6:.2f} μm, "
            f"P95={p95*1e6:.2f} μm, "
            f"P99={p99*1e6:.2f} μm, "
            f"skew={skewness:.2f})"
        )
        print(f"     Failure rates: {fail_str}")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue

if not method_data:
    print("\n❌ No valid data loaded. Exiting.")
    exit()

print(f"\n✓ Successfully loaded {len(method_data)} methods")

# ==============================
# 只统计/绘制 CUSTOM_ORDER 白名单中的方法（见文件顶部）, 其余剔除
# ==============================
if CUSTOM_ORDER:
    def _in_custom_order(name):
        return str(name).strip() in CUSTOM_ORDER or norm_method_name(name) in _CUSTOM_ORDER_KEYS

    kept = [n for n in method_data if _in_custom_order(n)]
    dropped = [n for n in method_data if not _in_custom_order(n)]
    if dropped:
        print(f"⚠ 不在 CUSTOM_ORDER 中, 不统计不绘制: {dropped}")
    method_data = {k: method_data[k] for k in kept}
    method_frames = {k: method_frames[k] for k in kept}
    method_stats = {k: method_stats[k] for k in kept}
    print(f"✓ 白名单过滤后保留 {len(method_data)} 个方法: {list(method_data.keys())}")
    if not method_data:
        print("❌ CUSTOM_ORDER 过滤后没有剩余方法, 请检查名单。 Exiting.")
        exit()

# ==============================
# 4.1 Quantile / Distribution Summary
# ==============================
print("\n📋 Generating enhanced quantile statistics...")
quantile_report_data = []
for method_name, stats in method_stats.items():
    row = {
        'Method': method_name,
        'N': stats['count'],
        'Mean (μm)': stats['mean'] * 1e6,
        'SD (μm)': stats['std'] * 1e6,
        'Median (μm)': stats['median'] * 1e6,
        'Q1 (μm)': stats['q1'] * 1e6,
        'Q3 (μm)': stats['q3'] * 1e6,
        'IQR (μm)': stats['iqr'] * 1e6,
        'P95 (μm)': stats['p95'] * 1e6,
        'P99 (μm)': stats['p99'] * 1e6,
        'Max (μm)': stats['max'] * 1e6,
        'Skewness': stats['skew'],
        'Kurtosis': stats['kurtosis'],
    }
    for th in FAILURE_THRESHOLDS_UM:
        row[f'Fail>{th}μm Count'] = stats['failure_info'][th]['count']
        row[f'Fail>{th}μm Rate (%)'] = stats['failure_info'][th]['rate']
    quantile_report_data.append(row)

quantile_report_df = pd.DataFrame(quantile_report_data)
quantile_report_file = OUTPUT_DIR / "error_quantile_statistics_longtail.csv"
quantile_report_df.to_csv(quantile_report_file, index=False, float_format="%.6f")
print(f"✓ Saved: {quantile_report_file.name}")

# ==============================
# 4.2 Paired Statistical Analysis
# ==============================
def detect_id_column(df):
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for c in df.columns:
        name = str(c).lower()
        if (name == "id" or "sample_id" in name or "image_id" in name
                or "filename" in name or "file_name" in name
                or "hologram_name" in name):
            return c
    return None

def cliff_delta(x, y):
    from scipy.stats import wilcoxon
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank()
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    if w_pos + w_neg == 0:
        return 0.0
    return (w_pos - w_neg) / (w_pos + w_neg)

def bootstrap_median_difference_ci(x, y, n_boot=10000, seed=20260918):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    diff = x - y
    observed = float(np.median(diff))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_medians = np.median(diff[indices], axis=1)
    lower = float(np.percentile(boot_medians, 2.5))
    upper = float(np.percentile(boot_medians, 97.5))
    return lower, upper, observed

def paired_compare(method_a, method_b):
    df_a = method_frames[method_a].copy()
    df_b = method_frames[method_b].copy()
    id_a = detect_id_column(df_a)
    id_b = detect_id_column(df_b)

    if id_a is not None and id_b is not None:
        key_a = df_a[id_a].astype(str)
        key_b = df_b[id_b].astype(str)
        tmp_a = pd.DataFrame({"_pair_id": key_a, "a": df_a["_abs_error_numeric"].astype(float)})
        tmp_b = pd.DataFrame({"_pair_id": key_b, "b": df_b["_abs_error_numeric"].astype(float)})
        tmp_a = tmp_a.drop_duplicates("_pair_id", keep="first")
        tmp_b = tmp_b.drop_duplicates("_pair_id", keep="first")
        merged = pd.merge(tmp_a, tmp_b, on="_pair_id", how="inner")
        alignment = f"ID: {id_a} ↔ {id_b}"
    else:
        if len(df_a) != len(df_b):
            raise ValueError(f"Cannot safely perform paired test: sample counts differ ({len(df_a)} vs {len(df_b)}).")
        merged = pd.DataFrame({
            "a": df_a["_abs_error_numeric"].astype(float).to_numpy(),
            "b": df_b["_abs_error_numeric"].astype(float).to_numpy()
        })
        alignment = "row order (WARNING: no ID column detected)"

    merged = merged.dropna(subset=["a", "b"])
    if len(merged) < 2:
        raise ValueError(f"Insufficient paired samples for {method_a} vs {method_b}.")

    x = merged["a"].to_numpy(dtype=float)
    y = merged["b"].to_numpy(dtype=float)
    diff = x - y

    from scipy import stats as scipy_stats
    if PAIRED_TEST.lower() == "wilcoxon":
        if np.allclose(diff, 0):
            test_name = "Wilcoxon signed-rank"
            statistic = 0.0
            p_value = 1.0
        else:
            result = scipy_stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox", method="auto")
            test_name = "Wilcoxon signed-rank"
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
    elif PAIRED_TEST.lower() == "ttest":
        result = scipy_stats.ttest_rel(x, y, nan_policy="omit")
        test_name = "Paired t-test"
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    else:
        raise ValueError("PAIRED_TEST must be 'wilcoxon' or 'ttest'.")

    effect_size = cliff_delta(x, y)

    ci_low, ci_high, median_diff = bootstrap_median_difference_ci(
        x, y, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_RANDOM_SEED
    )

    return {
        "Method A": method_a,
        "Method B": method_b,
        "N paired": len(x),
        "Alignment": alignment,
        "Median A (μm)": np.median(x) * 1e6,
        "Median B (μm)": np.median(y) * 1e6,
        "Median Difference A-B (μm)": median_diff * 1e6,
        "95% CI Lower (μm)": ci_low * 1e6,
        "95% CI Upper (μm)": ci_high * 1e6,
        "Test": test_name,
        "Statistic": statistic,
        "P-value": p_value,
        "Effect Size (rank-biserial)": effect_size,
    }

paired_results = []
if len(method_names := list(method_data.keys())) >= 2:
    reference_method = method_names[0]
    print("\n" + "=" * 120)
    print(f"PAIRED STATISTICAL TESTS: reference = {reference_method}")
    print("=" * 120)
    for method_name in method_names[1:]:
        try:
            result = paired_compare(reference_method, method_name)
            paired_results.append(result)
            print(
                f"{reference_method} vs {method_name}: "
                f"N={result['N paired']}, "
                f"Δmedian={result['Median Difference A-B (μm)']:.3f} μm, "
                f"95% CI=[{result['95% CI Lower (μm)']:.3f}, {result['95% CI Upper (μm)']:.3f}], "
                f"{result['Test']}, p={result['P-value']:.6g}, "
                f"effect={result['Effect Size (rank-biserial)']:.3f}, "
                f"{result['Alignment']}"
            )
        except Exception as e:
            print(f"⚠ Paired analysis skipped for {reference_method} vs {method_name}: {e}")

if paired_results:
    p_vals = [r['P-value'] for r in paired_results]
    from scipy.stats import false_discovery_control
    try:
        adjusted_p = false_discovery_control(p_vals, method='bh')
    except Exception:
        n = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        sorted_p = np.array(p_vals)[sorted_idx]
        adjusted = np.empty(n)
        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = min(sorted_p[i] * n / (i + 1), 1.0)
        for i in range(n - 2, -1, -1):
            adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i+1]])
        adjusted_p = adjusted
    for r, adj_p in zip(paired_results, adjusted_p):
        r['P-value (BH adjusted)'] = adj_p

    paired_report_df = pd.DataFrame(paired_results)
    paired_report_file = OUTPUT_DIR / "paired_statistical_tests_longtail.csv"
    paired_report_df.to_csv(paired_report_file, index=False, float_format="%.8f")
    print(f"\n✓ Saved: {paired_report_file.name}")
else:
    print("\n⚠ No paired statistical results were generated.")

# ==============================
# 4.3 Console Summary
# ==============================
print("\n" + "=" * 130)
print("ENHANCED STATISTICAL SUMMARY (long-tail oriented)")
print("=" * 130)
header = (
    f"{'Method':<20}{'N':>7}{'Median':>10}{'IQR':>10}{'P95':>10}{'P99':>10}"
    f"{'Max':>10}{'Skew':>8}"
)
print(header)
print("-" * len(header))
for method_name, stats in method_stats.items():
    line = (
        f"{method_name:<20}"
        f"{stats['count']:>7d}"
        f"{stats['median']*1e6:>10.2f}"
        f"{stats['iqr']*1e6:>10.2f}"
        f"{stats['p95']*1e6:>10.2f}"
        f"{stats['p99']*1e6:>10.2f}"
        f"{stats['max']*1e6:>10.2f}"
        f"{stats['skew']:>8.2f}"
    )
    print(line)
print("=" * 130)

method_names = list(method_data.keys())
medians = [method_stats[name]['median'] * 1e6 for name in method_names]
iqrs = [method_stats[name]['iqr'] * 1e6 for name in method_names]
p95s = [method_stats[name]['p95'] * 1e6 for name in method_names]

# ==============================
# 4.4 Additional Visualizations
# ==============================
print("\n📈 Generating long-tail oriented visualizations...")

def save_figure_both(fig, stem):
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: {png_path.name}")
    print(f"✓ Saved: {pdf_path.name}")
    return png_path, pdf_path

# ------------------------------------------------
# 4.4.1 KDE with log x-axis
# ------------------------------------------------
print("  Generating KDE error distribution (log scale)...")
fig_kde, ax_kde = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
try:
    from scipy.stats import gaussian_kde
    for method_name in method_names:
        values = method_data[method_name].to_numpy(dtype=float) * 1e6
        values = values[np.isfinite(values)]
        values = values[values > 0]
        if len(values) < 2:
            continue
        if np.std(values) > 0:
            kde_func = gaussian_kde(values)
            x_min = float(np.min(values))
            x_max = float(np.max(values))
            x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 500)
            density = kde_func(x_grid)
            ax_kde.plot(x_grid, density, linewidth=2.0, label=method_name)
        else:
            x0 = float(values[0])
            ax_kde.axvline(x0, linewidth=2.0, label=method_name)
except ImportError:
    print("  ⚠ scipy not installed; using normalized histograms instead.")
    for method_name in method_names:
        values = method_data[method_name].to_numpy(dtype=float) * 1e6
        values = values[np.isfinite(values)]
        values = values[values > 0]
        if len(values) == 0:
            continue
        ax_kde.hist(values, bins=np.logspace(np.log10(values.min()), np.log10(values.max()), 30),
                    density=True, histtype='fig_step', linewidth=1.8, label=method_name)

ax_kde.set_xscale('log')
ax_kde.set_xlabel('Absolute Error (μm)', fontweight='semibold', labelpad=8)
ax_kde.set_ylabel('Probability Density', fontweight='semibold', labelpad=8)
ax_kde.set_title('Error Distribution (log scale)', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE + 1)
ax_kde.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
ax_kde.set_axisbelow(True)
ax_kde.legend(loc='best', framealpha=0.95, fancybox=False)
plt.tight_layout(pad=0.5)
save_figure_both(fig_kde, "error_kde_distribution_log")
plt.close(fig_kde)

# ------------------------------------------------
# 4.4.2 CDF plot (log x-axis)
# ------------------------------------------------
print("  Generating CDF plot (log scale)...")
fig_cdf, ax_cdf = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
for method_name in method_names:
    values = method_data[method_name].to_numpy(dtype=float) * 1e6
    values = values[np.isfinite(values)]
    values = np.sort(values)
    if len(values) == 0:
        continue
    y = np.arange(1, len(values) + 1) / len(values)
    ax_cdf.plot(values, y, linewidth=2.0, label=method_name)
ax_cdf.set_xscale('log')
ax_cdf.set_xlabel('Absolute Error (μm)', fontweight='semibold')
ax_cdf.set_ylabel('Cumulative Probability', fontweight='semibold')
ax_cdf.set_title('CDF of Absolute Error (log scale)', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE + 1)
ax_cdf.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
ax_cdf.set_axisbelow(True)
ax_cdf.legend(loc='lower right', framealpha=0.95, fancybox=False)
ax_cdf.set_ylim(0, 1.02)
plt.tight_layout(pad=0.5)
save_figure_both(fig_cdf, "error_cdf_log")
plt.close(fig_cdf)

# ------------------------------------------------
# 4.4.3 Median + IQR + P95 comparison (log y-axis)
# ------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
x_pos = np.arange(len(method_names))
q1s = [method_stats[name]['q1'] * 1e6 for name in method_names]
q3s = [method_stats[name]['q3'] * 1e6 for name in method_names]
p95s = [method_stats[name]['p95'] * 1e6 for name in method_names]

yerr_iqr = [np.array(medians) - np.array(q1s), np.array(q3s) - np.array(medians)]
ax4.errorbar(x_pos, medians, yerr=yerr_iqr, fmt='o', markersize=7,
             linewidth=1.8, capsize=5, label='Median ± IQR')
ax4.scatter(x_pos, p95s, marker='^', s=55, label='95th percentile', zorder=3)
for x, med, p95 in zip(x_pos, medians, p95s):
    ax4.text(x, p95, f'{p95:.1f}', ha='center', va='bottom', fontsize=FONT_SIZE_TICK - 1)
ax4.set_yscale('log')
ax4.set_ylabel('Absolute Error (μm, log scale)', fontweight='semibold', labelpad=8)
ax4.set_xlabel('Method', fontweight='semibold', labelpad=8)
ax4.set_title('Median, IQR and 95th Percentile (log scale)', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE + 1)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(method_names)
ax4.yaxis.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
ax4.set_axisbelow(True)
ax4.legend(loc='upper right', framealpha=0.95, fancybox=False)
plt.tight_layout(pad=0.5)
save_figure_both(fig4, "error_median_iqr_p95_log")
plt.close(fig4)

# ------------------------------------------------
# 4.4.4 Mean vs Median
# ------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
means_arr = np.array([method_stats[name]['mean'] * 1e6 for name in method_names])
medians_arr = np.array(medians)
ax5.scatter(means_arr, medians_arr, s=65, zorder=3)
xy_min = min(np.min(means_arr), np.min(medians_arr))
xy_max = max(np.max(means_arr), np.max(medians_arr))
margin = max((xy_max - xy_min) * 0.08, 1.0)
ax5.plot([xy_min - margin, xy_max + margin], [xy_min - margin, xy_max + margin],
         linestyle='--', linewidth=1.2, label='Mean = Median')
for method_name, x, y in zip(method_names, means_arr, medians_arr):
    ax5.annotate(method_name, (x, y), xytext=(5, 5), textcoords='offset points',
                 fontsize=FONT_SIZE_TICK - 1)
ax5.set_xlabel('Mean Absolute Error (μm)', fontweight='semibold')
ax5.set_ylabel('Median Absolute Error (μm)', fontweight='semibold')
ax5.set_title('Mean vs. Median Error', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE + 1)
ax5.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
ax5.set_aspect('equal', adjustable='box')
ax5.legend(loc='best', framealpha=0.95, fancybox=False)
plt.tight_layout(pad=0.5)
save_figure_both(fig5, "error_mean_vs_median")
plt.close(fig5)

# ------------------------------------------------
# 4.4.5 P95 vs Max (log y-axis)
# ------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
max_values = np.array([method_stats[name]['max'] * 1e6 for name in method_names])
width = 0.36
bars_p95 = ax6.bar(x_pos - width/2, p95s, width, label='95th percentile', alpha=0.8)
bars_max = ax6.bar(x_pos + width/2, max_values, width, label='Maximum', alpha=0.8)
for bar in list(bars_p95) + list(bars_max):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}',
             ha='center', va='bottom', fontsize=FONT_SIZE_TICK - 1)
ax6.set_yscale('log')
ax6.set_ylabel('Absolute Error (μm, log scale)', fontweight='semibold')
ax6.set_xlabel('Method', fontweight='semibold')
ax6.set_title('Tail Error Comparison (log scale)', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE + 1)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(method_names)
ax6.yaxis.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
ax6.set_axisbelow(True)
ax6.legend(loc='upper right', framealpha=0.95, fancybox=False)
plt.tight_layout(pad=0.5)
save_figure_both(fig6, "error_p95_vs_max_log")
plt.close(fig6)

# ------------------------------------------------
# 4.4.6 IQR comparison (log y-axis)
# ------------------------------------------------
fig7, ax7 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
iqr_values = np.array(iqrs)
bars = ax7.bar(x_pos, iqr_values, alpha=0.8)
for bar, value in zip(bars, iqr_values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.1f}',
             ha='center', va='bottom', fontsize=FONT_SIZE_TICK - 1)
ax7.set_yscale('log')
ax7.set_ylabel('IQR (μm, log scale)', fontweight='semibold')
ax7.set_xlabel('Method', fontweight='semibold')
ax7.set_title('Interquartile Range (log scale)', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE + 1)
ax7.set_xticks(x_pos)
ax7.set_xticklabels(method_names)
ax7.yaxis.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
ax7.set_axisbelow(True)
plt.tight_layout(pad=0.5)
save_figure_both(fig7, "error_iqr_log")
plt.close(fig7)

# ------------------------------------------------
# 4.4.7 Coefficient of variation (log y-axis)
# ------------------------------------------------
fig8, ax8 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
# CV = std/mean; mean<=0 (数据异常) 时置为 nan 并告警, 避免负值/inf 进入 log 轴
cv_values = np.array([
    method_stats[name]['std'] / method_stats[name]['mean']
    if method_stats[name]['mean'] > 0 else np.nan
    for name in method_names
])
if np.isnan(cv_values).any():
    bad = [n for n, v in zip(method_names, cv_values) if np.isnan(v)]
    print(f"⚠ 以下方法 mean<=0, CV 置为 nan: {bad}")
    cv_values = np.nan_to_num(cv_values, nan=0.0)
bars = ax8.bar(x_pos, cv_values, alpha=0.8)
for bar, value in zip(bars, cv_values):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}',
             ha='center', va='bottom', fontsize=FONT_SIZE_TICK - 1)
ax8.set_yscale('log')
ax8.set_ylabel('Coefficient of Variation (log scale)', fontweight='semibold')
ax8.set_xlabel('Method', fontweight='semibold')
ax8.set_title('Relative Error Variability', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE + 1)
ax8.set_xticks(x_pos)
ax8.set_xticklabels(method_names)
ax8.yaxis.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
ax8.set_axisbelow(True)
plt.tight_layout(pad=0.5)
save_figure_both(fig8, "error_coefficient_of_variation_log")
plt.close(fig8)

# ------------------------------------------------
# 4.4.8 Failure-rate for multiple thresholds (CDF-like)
# ------------------------------------------------
fig9, ax9 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
for method_name in method_names:
    values = method_data[method_name].to_numpy(dtype=float) * 1e6
    values = values[np.isfinite(values)]
    thresholds = np.logspace(np.log10(max(0.1, values.min())), np.log10(values.max()), 200)
    rates = [(values > th).mean() * 100 for th in thresholds]
    ax9.plot(thresholds, rates, linewidth=2.0, label=method_name)
ax9.set_xscale('log')
ax9.set_xlabel('Error threshold (μm, log scale)', fontweight='semibold')
ax9.set_ylabel('Failure rate (%)', fontweight='semibold')
ax9.set_title('Failure Rate vs Threshold', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE + 1)
ax9.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
ax9.set_axisbelow(True)
ax9.legend(loc='upper right', framealpha=0.95, fancybox=False)
ax9.set_ylim(0, 100)
plt.tight_layout(pad=0.5)
save_figure_both(fig9, "error_failure_rate_curve")
plt.close(fig9)

# ------------------------------------------------
# 4.4.9 Paired difference + 95% CI visualization
# ------------------------------------------------
if paired_results:
    fig10, ax10 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
    labels = [f"{r['Method A']} vs {r['Method B']}" for r in paired_results]
    differences = np.array([r['Median Difference A-B (μm)'] for r in paired_results])
    ci_low = np.array([r['95% CI Lower (μm)'] for r in paired_results])
    ci_high = np.array([r['95% CI Upper (μm)'] for r in paired_results])
    lower_err = differences - ci_low
    upper_err = ci_high - differences
    y = np.arange(len(labels))
    ax10.errorbar(differences, y, xerr=[lower_err, upper_err],
                  fmt='o', markersize=7, capsize=5, linewidth=1.8)
    ax10.axvline(0, linestyle='--', linewidth=1.2)
    ax10.set_yticks(y)
    ax10.set_yticklabels(labels)
    ax10.set_xlabel('Median Error Difference (μm)', fontweight='semibold')
    ax10.set_title('Paired Median Difference with 95% CI', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE + 1)
    ax10.grid(True, axis='x', linestyle=':', alpha=0.4, linewidth=0.8)
    ax10.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    save_figure_both(fig10, "paired_median_difference_95CI")
    plt.close(fig10)
else:
    print("⚠ Paired CI figure skipped because no paired results were generated.")

# ------------------------------------------------
# 4.4.10 Complete statistics table as figure
# ------------------------------------------------
fig11, ax11 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
ax11.axis('off')
table_rows = []
for method_name in method_names:
    stats = method_stats[method_name]
    table_rows.append([
        method_name,
        f"{stats['median']*1e6:.2f}",
        f"{stats['iqr']*1e6:.2f}",
        f"{stats['p95']*1e6:.2f}",
        f"{stats['p99']*1e6:.2f}",
        f"{stats['max']*1e6:.2f}"
    ])
table = ax11.table(cellText=table_rows,
                   colLabels=['Method', 'Median', 'IQR', 'P95', 'P99', 'Max'],
                   loc='center', cellLoc='center', colLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.8)
ax11.set_title('Robust Error Statistics (μm)', fontweight='bold', pad=20, fontsize=FONT_SIZE_TITLE + 1)
plt.tight_layout(pad=0.5)
save_figure_both(fig11, "robust_error_statistics_table")
plt.close(fig11)

# ==============================
# 5. Enhanced Square Boxplot (log y-axis, show outliers)
# ==============================
print("\n📈 Generating enhanced boxplot (log y-axis, outliers shown)...")
fig1, ax1 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
data_to_plot = []
for name in method_names:
    vals = method_data[name].values * 1e6
    vals = vals[vals > 0]
    data_to_plot.append(vals)

bp = ax1.boxplot(data_to_plot,
                 labels=method_names,
                 patch_artist=True,
                 showmeans=True,
                 showfliers=True,
                 widths=0.6,
                 medianprops=dict(color='#2C3E50', linewidth=2.0, solid_capstyle='round'),
                 meanprops=dict(marker='D', markerfacecolor='#E74C3C',
                                markeredgecolor='#C0392B', markersize=5,
                                markeredgewidth=1.0, zorder=3),
                 whiskerprops=dict(color='black', linewidth=1.2, linestyle='-'),
                 capprops=dict(color='black', linewidth=1.2),
                 boxprops=dict(linewidth=1.2, edgecolor='#2C3E50'),
                 flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))

color_list = list(COLORS.values())
for patch, i in zip(bp['boxes'], range(len(method_names))):
    color = color_list[i % len(color_list)]
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
    patch.set_edgecolor('#2C3E50')

ax1.set_yscale('log')
ax1.set_ylabel('Absolute Error (μm, log scale)', fontweight='semibold', labelpad=8)
ax1.set_xlabel('Method', fontweight='semibold', labelpad=8)
ax1.set_title('Error Distribution Comparison (log scale)', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE + 1)
ax1.yaxis.grid(True, linestyle=':', alpha=0.4, linewidth=0.8, color='#AAAAAA')
ax1.set_axisbelow(True)
if len(method_names) > 5:
    ax1.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICK - 0.5)
    plt.setp(ax1.xaxis.get_majorticklabels(), ha='right', rotation_mode='anchor')
else:
    ax1.tick_params(axis='x', rotation=0)
ax1.set_facecolor('#FAFAFA')
for spine in ax1.spines.values():
    spine.set_linewidth(1.2)
    spine.set_edgecolor('#666666')
legend_elements = [
    Line2D([0], [0], color='#2C3E50', linewidth=2, label='Median'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#E74C3C',
           markeredgecolor='#C0392B', markersize=5, label='Mean', linestyle='None')
]
ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.95,
           edgecolor='#666666', fancybox=False)
plt.tight_layout(pad=0.5)
save_figure_both(fig1, "error_boxplot_square_log")
plt.close(fig1)

# ==============================
# 6. Bar Chart: Median ± IQR (log y-axis)
# ==============================
print("\n📊 Generating median ± IQR bar chart (log y-axis)...")
fig2, ax2 = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))
x_pos = np.arange(len(method_names))
medians = [method_stats[name]['median'] * 1e6 for name in method_names]
iqrs = [method_stats[name]['iqr'] * 1e6 for name in method_names]
bars = ax2.bar(x_pos, medians, yerr=iqrs,
               color=[color_list[i % len(color_list)] for i in range(len(method_names))],
               alpha=0.8, edgecolor='#2C3E50', linewidth=1.2, capsize=5,
               error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capthick': 1.5})
for i, (bar, med, iqr) in enumerate(zip(bars, medians, iqrs)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + iqr,
             f'{med:.1f}', ha='center', va='bottom', fontsize=FONT_SIZE_TICK-1,
             fontweight='semibold', color='#2C3E50')
ax2.set_yscale('log')
ax2.set_ylabel('Median Absolute Error ± IQR (μm, log scale)', fontweight='semibold', labelpad=8)
ax2.set_xlabel('Method', fontweight='semibold', labelpad=8)
ax2.set_title('Median Error with IQR (log scale)', fontweight='bold', pad=15, fontsize=FONT_SIZE_TITLE+1)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(method_names)
if len(method_names) > 5:
    ax2.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICK-0.5)
    plt.setp(ax2.xaxis.get_majorticklabels(), ha='right', rotation_mode='anchor')
else:
    ax2.tick_params(axis='x', rotation=0)
ax2.yaxis.grid(True, linestyle=':', alpha=0.4, linewidth=0.8, color='#AAAAAA')
ax2.set_axisbelow(True)
ax2.set_facecolor('#FAFAFA')
for spine in ax2.spines.values():
    spine.set_linewidth(1.2)
    spine.set_edgecolor('#666666')
plt.tight_layout(pad=0.5)
save_figure_both(fig2, "error_barchart_median_iqr_log")
plt.close(fig2)

# ==============================
# 6.5 Foreground ratio vs error analysis (batch for all methods)
# ==============================
print("\n📊 Generating foreground ratio vs error analysis for all methods...")

# ---------- 配置 ----------
VOID_CSV = folder_path / "void_ratio_results.csv"
VOID_NAME_COL = "A_image"
VOID_RATIO_COL = "void_ratio_percent"
FOREGROUND_RATIO_COL = "foreground_ratio_percent"

N_BINS_VOID = 6
BIN_MODE_VOID = "quantile"
STAT_VOID = "median"
SHOW_SCATTER_VOID = False
Y_MAX_VOID = None

MAKE_INDIVIDUAL_VOID_PLOT = True
MAKE_COMBINED_VOID_PLOT = True

# ---------- 合并图自定义排序: 见文件顶部 CUSTOM_ORDER（已上移并兼作白名单） ----------
# 方法在加载后已按 CUSTOM_ORDER 过滤, 这里仅剩排序逻辑


def normalize_stem_void(name):
    if pd.isna(name):
        return ""
    stem = Path(str(name).strip()).stem
    m = re.match(r"^(.*?)_z[0-9.]+$", stem)
    return (m.group(1) if m else stem).lower()


def load_void_reference():
    if not VOID_CSV.exists():
        print(f"⚠ 未找到 void CSV：{VOID_CSV}，跳过前景比例分析。")
        return None

    df_v = pd.read_csv(VOID_CSV)
    if VOID_NAME_COL not in df_v.columns:
        print(f"⚠ void CSV 中缺少列 {VOID_NAME_COL}，现有列：{list(df_v.columns)}")
        return None
    if VOID_RATIO_COL not in df_v.columns:
        print(f"⚠ void CSV 中缺少列 {VOID_RATIO_COL}，现有列：{list(df_v.columns)}")
        return None

    df_v = df_v.copy()
    df_v["_key"] = df_v[VOID_NAME_COL].apply(normalize_stem_void)
    df_v[VOID_RATIO_COL] = pd.to_numeric(df_v[VOID_RATIO_COL], errors="coerce")
    df_v = df_v.dropna(subset=[VOID_RATIO_COL])
    df_v = df_v[df_v["_key"] != ""]
    df_v = df_v.drop_duplicates("_key", keep="first")

    # 将 void ratio 转为 foreground ratio
    df_v[FOREGROUND_RATIO_COL] = 100.0 - df_v[VOID_RATIO_COL]

    return df_v[["_key", FOREGROUND_RATIO_COL]]


def compute_void_bin_stats(df_merged):
    df = df_merged.copy()

    if BIN_MODE_VOID == "quantile":
        df["_bin"] = pd.qcut(df[FOREGROUND_RATIO_COL], q=N_BINS_VOID, duplicates="drop")
    elif BIN_MODE_VOID == "equal":
        df["_bin"] = pd.cut(df[FOREGROUND_RATIO_COL], bins=N_BINS_VOID)
    else:
        raise ValueError("BIN_MODE_VOID 必须是 'quantile' 或 'equal'")

    grp = df.groupby("_bin", observed=True)["abs_error_um"]

    centers = np.array([b.mid for b in grp.groups.keys()])
    bin_labels = [f"{b.left:.1f}–{b.right:.1f}" for b in grp.groups.keys()]

    if STAT_VOID == "mean":
        values = grp.mean().values
        errors = grp.std().values
        summary = df.groupby("_bin", observed=True)["abs_error_um"].agg(
            N="count", Mean="mean", Std="std",
            Median="median",
            P95=lambda x: np.percentile(x, 95),
            Max="max",
        ).reset_index()
    elif STAT_VOID == "median":
        values = grp.median().values
        q1 = grp.quantile(0.25).values
        q3 = grp.quantile(0.75).values
        errors = np.vstack([values - q1, q3 - values])
        summary = df.groupby("_bin", observed=True)["abs_error_um"].agg(
            N="count", Mean="mean", Std="std",
            Median="median",
            Q1=lambda x: np.percentile(x, 25),
            Q3=lambda x: np.percentile(x, 75),
            P95=lambda x: np.percentile(x, 95),
            Max="max",
        ).reset_index()
    else:
        raise ValueError("STAT_VOID 必须是 'mean' 或 'median'")

    return centers, values, errors, bin_labels, summary, df


def plot_void_error_for_method(method_name, df_void, save=True):
    df_err = method_frames[method_name].copy()

    name_col = detect_id_column(df_err)
    if name_col is None:
        print(f"  ⚠ {method_name}: 未找到名称列，跳过前景比例分析。")
        return None

    df_err["_key"] = df_err[name_col].apply(normalize_stem_void)
    df_err["abs_error_um"] = df_err["_abs_error_numeric"].astype(float) * 1e6

    merged = pd.merge(
        df_err[["_key", "abs_error_um"]],
        df_void,
        on="_key", how="inner",
    )
    merged = merged.dropna(subset=["abs_error_um", FOREGROUND_RATIO_COL])

    if len(merged) < 6:
        print(f"  ⚠ {method_name}: 合并后样本太少（{len(merged)}），跳过前景比例分析。")
        return None

    centers, values, errors, bin_labels, summary, _ = compute_void_bin_stats(merged)

    summary_out = summary.copy()
    summary_out["method"] = method_name
    summary_csv = OUTPUT_DIR / f"foreground_error_bin_summary_{method_name}.csv"
    summary_out.to_csv(summary_csv, index=False, float_format="%.4f")

    merged_out = merged[["_key", FOREGROUND_RATIO_COL, "abs_error_um"]].copy()
    merged_out["method"] = method_name
    merged_csv = OUTPUT_DIR / f"merged_foreground_error_{method_name}.csv"
    merged_out.to_csv(merged_csv, index=False, float_format="%.6f")

    if not save:
        return merged, centers, values, errors, bin_labels

    fig, ax = plt.subplots(figsize=(MAX_SIZE_INCHES, MAX_SIZE_INCHES))

    if SHOW_SCATTER_VOID:
        ax.scatter(
            merged[FOREGROUND_RATIO_COL], merged["abs_error_um"],
            s=10, alpha=0.25, edgecolor="none", color="#7f7f7f",
            label="Samples", zorder=1,
        )

    if STAT_VOID == "mean":
        ax.errorbar(
            centers, values, yerr=errors,
            fmt="o-", color="#1f77b4", ecolor="#1f77b4",
            markersize=6, linewidth=1.8, capsize=5, capthick=1.5,
            label="Mean ± SD", zorder=3,
        )
        ylabel = "Mean absolute error (μm)"
    else:
        ax.errorbar(
            centers, values, yerr=errors,
            fmt="o-", color="#1f77b4", ecolor="#1f77b4",
            markersize=6, linewidth=1.8, capsize=5, capthick=1.5,
            label="Median ± IQR", zorder=3,
        )
        ylabel = "Median absolute error (μm)"

    for c, v in zip(centers, values):
        ax.annotate(f"{v:.1f}", (c, v), xytext=(0, 6),
                    textcoords="offset points", ha="center",
                    fontsize=FONT_SIZE_TICK - 1)

    ax.set_xlabel("Foreground ratio (%)", fontweight="semibold", labelpad=8)
    ax.set_ylabel(ylabel, fontweight="semibold", labelpad=8)
    ax.set_title(f"Focus error vs foreground ratio — {method_name}",
                 fontweight="bold", pad=15, fontsize=FONT_SIZE_TITLE + 1)

    if Y_MAX_VOID is not None:
        ax.set_ylim(0, Y_MAX_VOID)

    ax.grid(True, linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", framealpha=0.95, fancybox=False)

    plt.tight_layout(pad=0.5)
    save_figure_both(fig, f"foreground_vs_error_{method_name}")
    plt.close(fig)

    return merged, centers, values, errors, bin_labels


df_void = load_void_reference()

if df_void is not None:
    print(f"✓ foreground 参考数据：{len(df_void)} 条")

    void_method_results = {}

    if MAKE_INDIVIDUAL_VOID_PLOT:
        for m_name in method_names:
            print(f"\n  → 分析 {m_name} 的前景比例-误差关系...")
            try:
                res = plot_void_error_for_method(m_name, df_void, save=True)
                if res is not None:
                    void_method_results[m_name] = res
            except Exception as e:
                print(f"  ❌ {m_name} 前景比例分析失败：{e}")

    if MAKE_COMBINED_VOID_PLOT and len(void_method_results) > 0:
        print("\n  → 生成合并对比图（论文双栏宽度，共享 y 轴 + 平方根刻度，多列布局）...")

        from matplotlib.ticker import FixedLocator, FixedFormatter

        # ---------- 打印实际方法名，便于确认 ----------
        print("合并图实际方法名：", list(void_method_results.keys()))

        # ---------- 自定义排序（见文件顶部 CUSTOM_ORDER） ----------
        _norm_order_map = {
            norm_method_name(x): i for i, x in enumerate(CUSTOM_ORDER)
        }

        def method_order_key(name):
            key = str(name).strip()

            # 精确匹配优先
            if key in CUSTOM_ORDER:
                return CUSTOM_ORDER.index(key)

            # 忽略大小写/空格/下划线/连字符等匹配
            nk = norm_method_name(key)
            if nk in _norm_order_map:
                return _norm_order_map[nk]

            # 未列入 CUSTOM_ORDER 的方法排到最后
            return len(CUSTOM_ORDER)

        sorted_items = sorted(
            void_method_results.items(),
            key=lambda kv: method_order_key(kv[0])
        )
        # ------------------------------------------------

        # ---------- 论文排版参数 ----------
        FIG_WIDTH_INCHES = 7.16
        n_methods = len(sorted_items)
        n_cols = 4
        n_rows = int(np.ceil(n_methods / n_cols))

        subplot_h = 2.05
        FIG_HEIGHT_INCHES = subplot_h * n_rows + 0.6

        PUB_FONT_TICK = 9
        PUB_FONT_LABEL = 10
        PUB_FONT_TITLE = 10.5
        PUB_FONT_SUPTITLE = 12

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()

        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'p']
        color_list_void = list(COLORS.values())

        # ---------- 统一 y 轴上限 ----------
        if Y_MAX_VOID is not None:
            y_max_shared = Y_MAX_VOID
        else:
            y_max_shared = 0
            for _, (_, _, values, errors, _) in sorted_items:
                vals = np.asarray(values, dtype=float)
                if STAT_VOID == "mean":
                    upper = vals + np.asarray(errors, dtype=float)
                else:
                    upper = vals + np.asarray(errors, dtype=float)[1]
                if upper.size > 0:
                    y_max_shared = max(y_max_shared, float(np.nanmax(upper)))
            y_max_shared *= 1.10
            if y_max_shared <= 0:
                y_max_shared = 1.0

        # ---------- 简洁 sqrt 刻度 ----------
        def make_sqrt_ticks(y_max):
            pool = [0, 10, 20, 50, 100, 200, 300, 400, 500]
            return [v for v in pool if v <= y_max * 1.02]

        ticks = make_sqrt_ticks(y_max_shared)

        for idx, (m_name, (merged, centers, values, errors, bin_labels)) in enumerate(sorted_items):
            ax = axes[idx]
            color = color_list_void[idx % len(color_list_void)]
            marker = markers[idx % len(markers)]

            if SHOW_SCATTER_VOID:
                ax.scatter(
                    merged[FOREGROUND_RATIO_COL], merged["abs_error_um"],
                    s=3, alpha=0.25, edgecolor="none", color="#7f7f7f",
                    zorder=1,
                )

            ax.errorbar(
                centers, values, yerr=errors,
                fmt=marker + '-',
                color=color, ecolor=color,
                markersize=4.0, linewidth=1.2,
                elinewidth=0.85, capsize=2.5, capthick=0.85,
                markeredgecolor='white', markeredgewidth=0.5,
                label=m_name, alpha=0.9,
                zorder=3,
            )

            ax.set_title(m_name, fontweight='bold',
                         fontsize=PUB_FONT_TITLE, pad=3)
            ax.grid(True, which='major', linestyle=":", alpha=0.35, linewidth=0.6)
            ax.set_axisbelow(True)

            # 平方根刻度
            ax.set_yscale('function', functions=(np.sqrt, np.square))
            ax.set_ylim(0, y_max_shared)

            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))

            ax.tick_params(
                axis='both', which='major',
                labelsize=PUB_FONT_TICK,
                width=0.9, length=3.2,
                labelleft=True,
                labelbottom=True,
            )

            # 每行最左侧显示 y 轴标签
            if idx % n_cols == 0:
                if STAT_VOID == "mean":
                    ax.set_ylabel("Mean abs. error (μm)",
                                  fontsize=PUB_FONT_LABEL)
                else:
                    ax.set_ylabel("Median abs. error (μm)",
                                  fontsize=PUB_FONT_LABEL)

            # 最底一行显示 x 轴标签
            if idx >= n_methods - n_cols:
                ax.set_xlabel("Foreground ratio (%)",
                              fontsize=PUB_FONT_LABEL)

        for j in range(n_methods, len(axes)):
            axes[j].axis('off')

        fig.suptitle("Focus error vs foreground ratio",
                     fontweight='bold',
                     fontsize=PUB_FONT_SUPTITLE,
                     y=0.995)

        plt.subplots_adjust(
            left=0.075,
            right=0.99,
            top=0.91,
            bottom=0.11,
            wspace=0.22,
            hspace=0.28,
        )

        save_figure_both(fig, "foreground_vs_error_all_methods")
        plt.close(fig)
    print("\n✅ 前景比例-误差分析完成。")
else:
    print("\n⚠ 未执行前景比例分析。")

# ==============================
# 7. Statistical Report
# ==============================
print("\n📋 Generating publication-ready statistical report...")
report_data = []
for method_name, stats in method_stats.items():
    row = {
        'Method': method_name,
        'Count': stats['count'],
        'Mean (μm)': f"{stats['mean']*1e6:.2f}",
        'Std (μm)': f"{stats['std']*1e6:.2f}",
        'SEM (μm)': f"{stats['sem']*1e6:.2f}",
        'Min (μm)': f"{stats['min']*1e6:.2f}",
        'Q1 (μm)': f"{stats['q1']*1e6:.2f}",
        'Median (μm)': f"{stats['median']*1e6:.2f}",
        'Q3 (μm)': f"{stats['q3']*1e6:.2f}",
        'IQR (μm)': f"{stats['iqr']*1e6:.2f}",
        'P95 (μm)': f"{stats['p95']*1e6:.2f}",
        'P99 (μm)': f"{stats['p99']*1e6:.2f}",
        'Max (μm)': f"{stats['max']*1e6:.2f}",
        'Skewness': f"{stats['skew']:.3f}",
        'Kurtosis': f"{stats['kurtosis']:.3f}",
    }
    for th in FAILURE_THRESHOLDS_UM:
        row[f'Fail>{th}μm Count'] = stats['failure_info'][th]['count']
        row[f'Fail>{th}μm Rate (%)'] = f"{stats['failure_info'][th]['rate']:.2f}"
    report_data.append(row)

report_df = pd.DataFrame(report_data)
report_file = OUTPUT_DIR / "error_statistics_report_longtail.csv"
report_df.to_csv(report_file, index=False)
print(f"✓ Saved: {report_file.name}")

manuscript_cols = [
    'Method', 'Count', 'Median (μm)', 'IQR (μm)',
    'P95 (μm)', 'P99 (μm)', 'Max (μm)', 'Skewness'
]
for th in FAILURE_THRESHOLDS_UM:
    manuscript_cols += [f'Fail>{th}μm Rate (%)']
manuscript_table = report_df[manuscript_cols].copy()
manuscript_table_file = OUTPUT_DIR / "manuscript_error_statistics_longtail.csv"
manuscript_table.to_csv(manuscript_table_file, index=False)
print(f"✓ Saved: {manuscript_table_file.name}")

print("\n" + "=" * 110)
print("SUMMARY (sorted by Median Absolute Error)")
print("=" * 110)
sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]['median'])
for rank, (method_name, stats) in enumerate(sorted_methods, 1):
    print(
        f"{rank:>2}. {method_name:<20} "
        f"Median={stats['median']*1e6:>8.2f} μm, "
        f"IQR={stats['iqr']*1e6:>8.2f} μm, "
        f"P95={stats['p95']*1e6:>8.2f} μm, "
        f"P99={stats['p99']*1e6:>8.2f} μm, "
        f"Max={stats['max']*1e6:>8.2f} μm, "
        f"Skew={stats['skew']:>6.2f}"
    )
print("=" * 110)

print("\n" + "=" * 110)
print("✅ ANALYSIS COMPLETE - LONG-TAIL ENHANCED STATISTICS")
print("=" * 110)
print(f"Generated files are ALL saved under:\n  {OUTPUT_DIR}")
print("Core visualizations:")
print("  • error_kde_distribution_log.png / .pdf")
print("  • error_cdf_log.png / .pdf")
print("  • error_median_iqr_p95_log.png / .pdf")
print("  • error_mean_vs_median.png / .pdf")
print("  • error_p95_vs_max_log.png / .pdf")
print("  • error_iqr_log.png / .pdf")
print("  • error_coefficient_of_variation_log.png / .pdf")
print("  • error_failure_rate_curve.png / .pdf")
print("  • error_boxplot_square_log.png / .pdf")
print("  • error_barchart_median_iqr_log.png / .pdf")
print("  • robust_error_statistics_table.png / .pdf")
if paired_results:
    print("  • paired_median_difference_95CI.png / .pdf")
print("Foreground ratio analysis:")
print("  • foreground_vs_error_<method>.png / .pdf  (for each method)")
print("  • foreground_vs_error_all_methods.png / .pdf")
print("  • foreground_error_bin_summary_<method>.csv")
print("  • merged_foreground_error_<method>.csv")
print("Statistics:")
print(f"  • {report_file.name}")
print(f"  • {manuscript_table_file.name}")
print(f"  • {quantile_report_file.name}")
if paired_results:
    print("  • paired_statistical_tests_longtail.csv")
print("\nFigure specifications:")
print(f"  • All figures: {MAX_SIZE_CM} cm × {MAX_SIZE_CM} cm (1:1 square)")
print(f"  • Resolution: {OUTPUT_DPI} DPI")
print(f"  • Font: {font_name}")
print(f"  • Format: PNG + PDF")
print(f"  • Units: Micrometers (μm)")
print(f"  • Robust statistics: Median, IQR, P95, P99, Max, Skewness")
print(f"  • Failure thresholds: {FAILURE_THRESHOLDS_UM} μm")
print(f"  • Paired test: {PAIRED_TEST}, BH adjusted p-values")
print(f"  • Bootstrap CI: {BOOTSTRAP_N} resamples, 95% CI, median difference")
print("=" * 110)