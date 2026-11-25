import os
import numpy as np
import cv2

# ================= 配置 =================
INPUT_DIR         = r'F:\dongjiayao\Data\USAF\PROP_EDGE\CNN\prob'   # 输入路径（递归查找图像）
OUTPUT_DIR        = r'F:\dongjiayao\Data\USAF\PROP_EDGE\CNN\map'  # 输出路径（自动创建，保持子目录结构）
IMAGE_EXTENSIONS  = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
FILE_SUFFIX       = '_seg'          # 输出文件名后缀，避免与原图混淆

# 分段颜色（BGR 顺序），采用色盲友好的 Okabe-Ito 调色板
COLOR_BLACK = (0, 0, 0)        # v == 0      黑色
COLOR_LOW   = (178, 114, 0)    # 0   < v <= 1/3   蓝色 (RGB: 0,114,178)
COLOR_MID   = (115, 158, 0)    # 1/3 < v <= 2/3   绿色 (RGB: 0,158,115)
COLOR_HIGH  = (0, 94, 213)     # 2/3 < v <= 1     红色 (RGB: 213,94,0)

# 分段索引 -> 颜色查找表
PALETTE = np.array([COLOR_BLACK, COLOR_LOW, COLOR_MID, COLOR_HIGH], dtype=np.uint8)


def normalize(img):
    """逐图 min-max 归一化到 [0, 1]，全图无变化时返回全 0"""
    img = img.astype(np.float32)
    vmin, vmax = float(img.min()), float(img.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(img)
    return (img - vmin) / (vmax - vmin)


def segment_colorize(norm):
    """归一化图分段着色: 0 黑色, (0,1/3] 低, (1/3,2/3] 中, (2/3,1] 高"""
    idx = np.zeros(norm.shape, dtype=np.uint8)              # 0 = 黑色
    idx[(norm > 0.) & (norm <= 1.0 / 3.0)] = 1             # 低
    idx[(norm > 1.0 / 3.0) & (norm <= 2.0 / 3.0)] = 2       # 中
    idx[norm > 2.0 / 3.0] = 3                               # 高
    return PALETTE[idx]                                     # HxWx3, BGR


def main():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"输入路径不存在: {INPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    count = 0
    for root, _, files in os.walk(INPUT_DIR):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in IMAGE_EXTENSIONS:
                continue

            in_path = os.path.join(root, f)
            img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"警告：无法读取 {in_path}，跳过")
                continue

            out_img = segment_colorize(normalize(img))

            # 输出路径保持相对子目录结构
            rel_dir = os.path.relpath(root, INPUT_DIR)
            out_dir = OUTPUT_DIR if rel_dir == '.' else os.path.join(OUTPUT_DIR, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.splitext(f)[0] + FILE_SUFFIX + '.png')

            if cv2.imwrite(out_path, out_img):
                count += 1
                print(f"[{count:04d}] {in_path} -> {out_path}")
            else:
                print(f"警告：写入失败 {out_path}")

    print(f"完成：共处理 {count} 张图像，输出目录 {OUTPUT_DIR}")


if __name__ == '__main__':
    main()