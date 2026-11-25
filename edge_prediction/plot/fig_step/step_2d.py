import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift

# Parameters
wavelength = 633e-9          # wavelength (m)
L = 0.5e-3                   # computation window width (m)
N = 1024                     # number of samples (2D grid N x N)
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Create circular aperture
R = 0.2e-3                  # radius of the circle (m)
X, Y = np.meshgrid(x, y)
I0 = (X**2 + Y**2 <= R**2).astype(np.float64)

# Frequency coordinates (angular spectrum method, centered)
fx = fftshift(fftfreq(N, d=dx))
fy = fftshift(fftfreq(N, d=dy))
FX, FY = np.meshgrid(fx, fy)

# Defocus distances (m)
dz_list = [0.0, 1e-3]

# Store reconstructed intensities
intensity_profiles = []

# Compute reconstruction for each defocus distance
for dz in dz_list:
    if dz == 0:
        I_recon = I0.copy()
    else:
        # Angular spectrum propagation (2D)
        U0_fft = fftshift(fft2(ifftshift(I0)))
        H = np.exp(-1j * np.pi * wavelength * dz * (FX**2 + FY**2))
        Uz_fft = U0_fft * H
        Uz = fftshift(ifft2(ifftshift(Uz_fft)))
        I_recon = np.abs(Uz)**2
    intensity_profiles.append(I_recon)

# --- Display zoom settings ---
zoom_fraction = 0.40          # fraction of the full window to display (central region)
half_width = (L * zoom_fraction) / 2   # half width in meters
xlim_mm = half_width * 1e3    # convert to mm for plotting

# Plot: two subplots side by side, same size and layout as 1D version
fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=500)

# Use same color scale for both subplots
vmin = min(I.min() for I in intensity_profiles)
vmax = max(I.max() for I in intensity_profiles)

for ax, dz, I_recon in zip(axes, dz_list, intensity_profiles):
    # Display intensity as heatmap, limiting to central zoom region
    extent = [-xlim_mm, xlim_mm, -xlim_mm, xlim_mm]
    im = ax.imshow(I_recon, extent=extent, origin='lower', cmap='viridis',
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(f'Defocus $\Delta z$ = {dz*1e3:.0f} mm')
    ax.axis('off')               # remove axes, ticks, labels
    ax.set_aspect('equal')       # keep pixels square

plt.tight_layout()
plt.savefig('defocus_2d_circle_no_axes.png', dpi=300, bbox_inches='tight')
plt.show()