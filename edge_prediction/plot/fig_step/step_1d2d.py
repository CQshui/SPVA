import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.ndimage import gaussian_filter

# ============================================================
# 统一论文风格
# ============================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "lines.linewidth": 1.1,
    "axes.linewidth": 0.8,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "savefig.dpi": 600,
    "figure.dpi": 300,
    "mathtext.fontset": "stix",
})

C_LINE = '#1f77b4'

# ============================================================
# 通用参数
# ============================================================
wavelength    = 633e-9
dz_list       = [0.0, 1e-3]
zoom_fraction = 0.40

# ============================================================
# Part A: 1D 阶跃函数
# ============================================================
L_1d  = 0.5e-3
N_1d  = 4096
x_1d  = np.linspace(-L_1d/2, L_1d/2, N_1d, endpoint=False)
dx_1d = x_1d[1] - x_1d[0]

I0_1d = (x_1d >= 0).astype(np.float64)
fx_1d = fftshift(fftfreq(N_1d, d=dx_1d))

intensity_1d = []
for dz in dz_list:
    if dz == 0:
        I_recon = I0_1d.copy()
    else:
        U0_fft  = fftshift(fft(ifftshift(I0_1d)))
        H       = np.exp(-1j * np.pi * wavelength * dz * fx_1d**2)
        Uz_fft  = U0_fft * H
        Uz      = fftshift(ifft(ifftshift(Uz_fft)))
        I_recon = np.abs(Uz)**2
    intensity_1d.append(I_recon)

half_1d    = (L_1d * zoom_fraction) / 2
xlim_1d_mm = half_1d * 1e3

# ============================================================
# Part B: 2D 圆形物体
# ============================================================
L_crop  = 0.5e-3
N_crop  = 1024
L_large = 2.0e-3
N_large = 4096

x_large = np.linspace(-L_large/2, L_large/2, N_large, endpoint=False)
y_large = np.linspace(-L_large/2, L_large/2, N_large, endpoint=False)
dx_large = x_large[1] - x_large[0]
X_large, Y_large = np.meshgrid(x_large, y_large)

R = 0.2e-3
I0_binary_large = ((X_large**2 + Y_large**2) >= R**2).astype(np.float64)
I0_large = gaussian_filter(I0_binary_large, sigma=2.0, mode='wrap')

fx_large = fftshift(fftfreq(N_large, d=dx_large))
fy_large = fftshift(fftfreq(N_large, d=dx_large))
FX_large, FY_large = np.meshgrid(fx_large, fy_large)

intensity_profiles = []
for dz in dz_list:
    if dz == 0:
        I_recon_large = I0_large.copy()
    else:
        U0_fft_large = fftshift(fft2(ifftshift(I0_large)))
        H_large = np.exp(-1j * np.pi * wavelength * dz *
                         (FX_large**2 + FY_large**2))
        Uz_fft_large = U0_fft_large * H_large
        Uz_large = fftshift(ifft2(ifftshift(Uz_fft_large)))
        I_recon_large = np.abs(Uz_large)**2

    center_large = N_large // 2
    start = center_large - N_crop // 2
    end   = start + N_crop
    intensity_profiles.append(I_recon_large[start:end, start:end])

half_2d    = (L_crop * zoom_fraction) / 2
xlim_2d_mm = half_2d * 1e3
extent_2d  = [-xlim_2d_mm, xlim_2d_mm, -xlim_2d_mm, xlim_2d_mm]

# ============================================================
# 合并绘图：2×2
#   上行：1D 强度曲线
#   下行：2D 强度图像
# ============================================================
fig, axes = plt.subplots(
    2, 2,
    figsize=(3.4, 3.6),
    dpi=300,
    constrained_layout=True
)

# 强制每个子图为正方形且等大
for ax in axes.ravel():
    ax.set_box_aspect(1)

titles = ['In-focus', 'Defocused']

# ---------- 上行：1D ----------
for ax, I_recon, title in zip(axes[0], intensity_1d, titles):
    ax.plot(x_1d * 1e3, I_recon, color=C_LINE, lw=1.1)
    ax.set_title(title, pad=3)
    ax.set_ylim(-0.1, 1.5)
    ax.set_xlim(-xlim_1d_mm, xlim_1d_mm)
    ax.set_xticks([])
    ax.set_yticks([])

# ---------- 下行：2D ----------
for ax, I, title in zip(axes[1], intensity_profiles, titles):
    ax.imshow(
        I,
        extent=extent_2d,
        origin='lower',
        cmap='gray',
        vmin=0, vmax=1,
        interpolation='nearest',
        aspect='auto',
    )
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])

# ---------- 强度标注 ----------
# 左列显示 Intensity 标签；右列用透明标签占位，保证两列子图宽度一致
axes[0, 0].set_ylabel('Intensity')
axes[1, 0].set_ylabel('Intensity')
for ax in [axes[0, 1], axes[1, 1]]:
    ax.set_ylabel('Intensity')
    ax.yaxis.label.set_alpha(0)   # 不可见，但占位

plt.savefig('fig_combined_defocus.pdf', bbox_inches='tight')
plt.savefig('fig_combined_defocus.png', dpi=600, bbox_inches='tight')
plt.show()