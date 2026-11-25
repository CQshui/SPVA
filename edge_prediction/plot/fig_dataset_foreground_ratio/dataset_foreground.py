import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
from pathlib import Path


# ============================================================
# 0. 字体与全局风格（适配单栏）
# ============================================================
def setup_font():
    available = {f.name.lower(): f.name for f in fm.fontManager.ttflist}
    for name in ['Times New Roman', 'Times', 'Liberation Serif',
                 'DejaVu Serif', 'FreeSerif']:
        if name.lower() in available.values() or any(
            name.lower() in k for k in available.keys()
        ):
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = [name]
            plt.rcParams['mathtext.fontset'] = 'dejavuserif'
            return name
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    return 'DejaVu Serif'


font_name = setup_font()

FIG_W = 3.5
FIG_H = 2.6
FONT_TICK = 8
FONT_LABEL = 9
FONT_TITLE = 9
LINE_W = 1.0

plt.rcParams.update({
    'font.size': FONT_LABEL,
    'axes.labelsize': FONT_LABEL,
    'axes.titlesize': FONT_TITLE,
    'xtick.labelsize': FONT_TICK,
    'ytick.labelsize': FONT_TICK,
    'legend.fontsize': FONT_TICK,
    'axes.linewidth': 0.8,
    'lines.linewidth': LINE_W,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.35,
    'grid.linestyle': ':',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


# ============================================================
# 1. 参数
# ============================================================
CSV_PATH = r"F:\dongjiayao\Data\AutoFocusDatabase\hologram_amplitude\void_ratio_results.csv"
GT_DIR = Path(r"F:\dongjiayao\Data\AutoFocusDatabase\hologram_amplitude\GT")
GT_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

OUTPUT_DIR = Path("foreground_ratio_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 读取 CSV
# ============================================================
df = pd.read_csv(CSV_PATH)

required_columns = ["A_image", "foreground_ratio"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"CSV 中不存在字段：{col}")

df = df[["A_image", "foreground_ratio"]].copy()
df["foreground_ratio"] = pd.to_numeric(df["foreground_ratio"], errors="coerce")
df = df.dropna(subset=["A_image", "foreground_ratio"])
df = df[(df["foreground_ratio"] >= 0) & (df["foreground_ratio"] <= 1)].copy()
df = df.reset_index(drop=True)


# ============================================================
# 2.5 额外加载 B_image，准备 GT 目录
# ============================================================
df_full = pd.read_csv(CSV_PATH)
has_b = "B_image" in df_full.columns

keep_cols = ["A_image", "foreground_ratio"]
if has_b:
    keep_cols.insert(1, "B_image")

df_full = df_full[keep_cols].copy()
df_full["foreground_ratio"] = pd.to_numeric(
    df_full["foreground_ratio"], errors="coerce"
)
df_full = df_full.dropna(subset=["A_image", "foreground_ratio"])
df_full = df_full[
    (df_full["foreground_ratio"] >= 0) & (df_full["foreground_ratio"] <= 1)
].copy()
df_full = df_full.reset_index(drop=True)


def find_gt_image(a_image_name, b_image_name=None):
    """在 GT_DIR 中查找对应图像。优先 B_image，否则用 A 主干匹配。"""
    if not GT_DIR.exists():
        return None

    if b_image_name and not pd.isna(b_image_name):
        p = GT_DIR / str(b_image_name)
        if p.exists():
            return p
        stem = Path(str(b_image_name)).stem
        for ext in GT_EXTS:
            cand = GT_DIR / f"{stem}{ext}"
            if cand.exists():
                return cand

    if a_image_name and not pd.isna(a_image_name):
        a_stem = Path(str(a_image_name)).stem
        for p in GT_DIR.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in GT_EXTS:
                continue
            if p.stem.startswith(a_stem + "_z") or p.stem == a_stem:
                return p

    return None


# ============================================================
# 3. 基本统计
# ============================================================
ratio = df["foreground_ratio"]

print("=" * 70)
print("真实实验全息图前景像素占比统计")
print("=" * 70)
print(f"样本数量       : {len(df)}")
print(f"最小值         : {ratio.min():.6f} ({ratio.min()*100:.2f}%)")
print(f"最大值         : {ratio.max():.6f} ({ratio.max()*100:.2f}%)")
print(f"均值           : {ratio.mean():.6f} ({ratio.mean()*100:.2f}%)")
print(f"标准差         : {ratio.std():.6f} ({ratio.std()*100:.2f}%)")
print(f"中位数         : {ratio.median():.6f} ({ratio.median()*100:.2f}%)")
print(f"Q1             : {ratio.quantile(0.25):.6f} ({ratio.quantile(0.25)*100:.2f}%)")
print(f"Q3             : {ratio.quantile(0.75):.6f} ({ratio.quantile(0.75)*100:.2f}%)")
print(f"IQR            : {(ratio.quantile(0.75)-ratio.quantile(0.25)):.6f}")
print(f"P5             : {ratio.quantile(0.05):.6f} ({ratio.quantile(0.05)*100:.2f}%)")
print(f"P95            : {ratio.quantile(0.95):.6f} ({ratio.quantile(0.95)*100:.2f}%)")
print("=" * 70)


# ============================================================
# 4. 保存统计结果
# ============================================================
stats = {
    "sample_count": len(ratio),
    "min": ratio.min(), "max": ratio.max(),
    "mean": ratio.mean(), "std": ratio.std(),
    "median": ratio.median(),
    "Q1": ratio.quantile(0.25), "Q3": ratio.quantile(0.75),
    "IQR": ratio.quantile(0.75) - ratio.quantile(0.25),
    "P5": ratio.quantile(0.05), "P95": ratio.quantile(0.95),
}
stats_df = pd.DataFrame([stats])
stats_df.to_csv(OUTPUT_DIR / "foreground_ratio_statistics.csv",
                index=False, encoding="utf-8-sig")


# ============================================================
# 5. 图1：前景像素占比直方图 + 代表性样本（单栏融合图）
# ============================================================
QUANTILES_FOR_THUMBS = [0.05, 0.25, 0.50, 0.75, 0.95]

thumb_rows = []
for q in QUANTILES_FOR_THUMBS:
    target = df_full["foreground_ratio"].quantile(q)
    idx = (df_full["foreground_ratio"] - target).abs().idxmin()
    row = df_full.loc[idx].copy()
    row["_target_q"] = q
    thumb_rows.append(row)

thumb_df = pd.DataFrame(thumb_rows).reset_index(drop=True)
thumb_df = thumb_df.drop_duplicates(subset=["A_image"], keep="first").reset_index(drop=True)

thumb_records = []
for _, row in thumb_df.iterrows():
    a_name = row["A_image"]
    b_name = row["B_image"] if has_b else None
    gt_path = find_gt_image(a_name, b_name)
    if gt_path is not None:
        thumb_records.append({
            "A_image": a_name,
            "B_image": b_name,
            "foreground_ratio": row["foreground_ratio"],
            "target_quantile": row["_target_q"],
            "gt_path": str(gt_path),
        })

thumbs = pd.DataFrame(thumb_records).reset_index(drop=True)
print("\n用于可视化的样本：")
if len(thumbs) > 0:
    print(thumbs.to_string(index=False))
else:
    print("⚠ 未找到可用的 GT 图像，直方图上方将不显示样本图。")

thumbs.to_csv(OUTPUT_DIR / "selected_samples_for_visualization.csv",
              index=False, encoding="utf-8-sig")


# ------------------------------------------------------------
# 构建融合图
# ------------------------------------------------------------
n_thumbs = len(thumbs)

FIG_W_FUSE = 3.5
FIG_H_FUSE = 3.3

fig = plt.figure(figsize=(FIG_W_FUSE, FIG_H_FUSE))

# 对齐边距，让缩略图行的 x 范围和直方图的 x 范围一致
LEFT = 0.14
RIGHT = 0.98
WIDTH = RIGHT - LEFT

# ---------- 直方图（下方） ----------
ax_hist = fig.add_axes([LEFT, 0.10, WIDTH, 0.55])

ax_hist.hist(
    ratio * 100,
    bins=25,
    edgecolor='black',
    linewidth=0.5,
    color='#4C72B0',
    alpha=0.85,
)
ax_hist.set_xlabel("Foreground pixel ratio (%)")
ax_hist.set_ylabel("Number of holograms")
ax_hist.grid(True, linestyle=':', alpha=0.35, linewidth=0.5)
ax_hist.set_axisbelow(True)

# ---------- 缩略图（上方） ----------
if n_thumbs > 0:
    THUMB_BOTTOM = 0.72
    THUMB_HEIGHT = 0.16
    thumb_gap = 0.010

    per_w = (WIDTH - thumb_gap * (n_thumbs - 1)) / n_thumbs

    for i, row in thumbs.iterrows():
        left_i = LEFT + i * (per_w + thumb_gap)
        ax_thumb = fig.add_axes([left_i, THUMB_BOTTOM, per_w, THUMB_HEIGHT])
        try:
            img = mpimg.imread(row["gt_path"])
        except Exception as e:
            print(f"读取失败：{row['gt_path']}  ({e})")
            ax_thumb.axis("off")
            continue

        ax_thumb.imshow(img, cmap='gray' if img.ndim == 2 else None,
                        aspect='auto')
        ax_thumb.set_xticks([])
        ax_thumb.set_yticks([])

        # 边框与直方图坐标轴一致
        for spine in ax_thumb.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('black')

        # 下方标注：与直方图 x 轴标签同样的字体
        ratio_pct = row["foreground_ratio"] * 100
        ax_thumb.set_xlabel(f"{ratio_pct:.1f}%",
                            fontsize=FONT_LABEL,
                            labelpad=1.5)

    # ---------- 缩略图行的标题，风格与坐标轴标签一致 ----------
    fig.text(
        LEFT + WIDTH / 2,
        0.915,
        "Representative holograms",
        ha='center', va='center',
        fontsize=FONT_LABEL,
        color='black',
    )

    # ---------- 直方图上用红虚线标出样本位置 ----------
    for i, row in thumbs.iterrows():
        x_val = row["foreground_ratio"] * 100
        ax_hist.axvline(
            x=x_val,
            color='#C44E52',
            linestyle='--',
            linewidth=0.6,
            alpha=0.7,
            zorder=1,
        )

plt.savefig(OUTPUT_DIR / "foreground_ratio_histogram_with_samples.png",
            dpi=600, bbox_inches='tight', pad_inches=0.02)
plt.savefig(OUTPUT_DIR / "foreground_ratio_histogram_with_samples.pdf",
            bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
print(f"\n融合图已保存：{OUTPUT_DIR / 'foreground_ratio_histogram_with_samples.png'}")


# ============================================================
# 6. 图2：排序后的前景像素占比（单栏）
# ============================================================
sorted_ratio = np.sort(ratio.values) * 100

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.plot(
    np.arange(1, len(sorted_ratio) + 1),
    sorted_ratio,
    color='#C44E52',
    linewidth=LINE_W,
)
ax.set_xlabel("Hologram index (sorted)")
ax.set_ylabel("Foreground pixel ratio (%)")
ax.grid(True, linestyle=':', alpha=0.35, linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.3)
fig.savefig(OUTPUT_DIR / "foreground_ratio_sorted.png", dpi=600,
            bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUTPUT_DIR / "foreground_ratio_sorted.pdf",
            bbox_inches='tight', pad_inches=0.02)
plt.close(fig)


# ============================================================
# 7. 图3：箱线图（单栏）
# ============================================================
fig, ax = plt.subplots(figsize=(FIG_W * 0.75, FIG_H))

ax.boxplot(
    ratio * 100,
    vert=True,
    widths=0.5,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(facecolor='#4C72B0', alpha=0.85,
                  edgecolor='black', linewidth=0.8),
    medianprops=dict(color='black', linewidth=1.2),
    whiskerprops=dict(color='black', linewidth=0.8),
    capprops=dict(color='black', linewidth=0.8),
    flierprops=dict(marker='o', markersize=2, alpha=0.5,
                    markerfacecolor='gray', markeredgecolor='none'),
)

ax.set_ylabel("Foreground pixel ratio (%)")
ax.set_xticks([])
ax.grid(True, axis='y', linestyle=':', alpha=0.35, linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.3)
fig.savefig(OUTPUT_DIR / "foreground_ratio_boxplot.png", dpi=600,
            bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUTPUT_DIR / "foreground_ratio_boxplot.pdf",
            bbox_inches='tight', pad_inches=0.02)
plt.close(fig)


# ============================================================
# 8. 输出占比最高/最低的样本
# ============================================================
print("\n前景像素占比最低的 10 组：")
print(df.sort_values("foreground_ratio").head(10).to_string(index=False))

print("\n前景像素占比最高的 10 组：")
print(df.sort_values("foreground_ratio", ascending=False).head(10).to_string(index=False))


# ============================================================
# 9. 输出不同区间的样本数量
# ============================================================
bins = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
labels = ["0–1%", "1–2%", "2–5%", "5–10%",
          "10–20%", "20–30%", "30–50%", "50–100%"]

groups = pd.cut(ratio, bins=bins, labels=labels,
                include_lowest=True, right=True)
group_counts = groups.value_counts().sort_index()

print("\n前景像素占比区间统计：")
print("=" * 50)
for label, count in group_counts.items():
    print(f"{label:>10s} : {count:4d} 组")
print("=" * 50)

group_counts.to_csv(OUTPUT_DIR / "foreground_ratio_range_counts.csv",
                    encoding="utf-8-sig")

print(f"\n全部完成，结果保存至：{OUTPUT_DIR.resolve()}")