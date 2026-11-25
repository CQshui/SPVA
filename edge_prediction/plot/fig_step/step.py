import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift

# ========== 论文半栏风格设置 ==========
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "savefig.dpi": 600,
    "figure.dpi": 300,
})

# ========== Parameters ==========
wavelength = 633e-9          # wavelength (m)
L = 0.5e-3                   # computation window width (m)
N = 4096                     # number of samples
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

# Step function: x >= 0 -> 1, otherwise 0
I0 = (x >= 0).astype(np.float64)

# Frequency coordinates (angular spectrum method, centered)
fx = fftshift(fftfreq(N, d=dx))

# Defocus distances (m) — only 0 and 1 mm
dz_list = [0.0, 1e-3]

# Store reconstructed intensities
intensity_profiles = []

# Compute reconstruction for each defocus distance
for dz in dz_list:
    if dz == 0:
        I_recon = I0.copy()
    else:
        # Angular spectrum propagation
        U0_fft = fftshift(fft(ifftshift(I0)))
        H = np.exp(-1j * np.pi * wavelength * dz * fx**2)
        Uz_fft = U0_fft * H
        Uz = fftshift(ifft(ifftshift(Uz_fft)))
        I_recon = np.abs(Uz)**2
    intensity_profiles.append(I_recon)

# ========== Display zoom settings ==========
zoom_fraction = 0.40
half_width = (L * zoom_fraction) / 2
xlim_mm = half_width * 1e3

# ========== 半栏尺寸绘图 ==========
# 宽约 3.4 in ≈ 8.6 cm，适合多数期刊半栏
fig, axes = plt.subplots(
    1, 2,
    figsize=(3.4, 1.75),
    dpi=300,
    sharey=True,
    constrained_layout=True
)

for ax, dz, I_recon in zip(axes, dz_list, intensity_profiles):
    ax.plot(x * 1e3, I_recon, 'b-', linewidth=0.8)
    ax.set_title(f'$\Delta z$ = {dz*1e3:.0f} mm')
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_ylim(-0.1, 1.5)
    ax.set_xlim(-xlim_mm, xlim_mm)
    ax.tick_params(direction='in', pad=2)

axes[0].set_ylabel('Intensity')
axes[0].set_xlabel('Position $z$ (mm)')
axes[1].set_xlabel('Position $z$ (mm)')

# 保存：PDF 为矢量，适合论文；PNG 为高分辨率预览
plt.savefig('defocus_comparison.pdf', bbox_inches='tight')
plt.savefig('defocus_comparison.png', dpi=600, bbox_inches='tight')

plt.show()