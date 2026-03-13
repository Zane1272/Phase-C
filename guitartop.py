import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation, label

import torch
import torch.nn.functional as F

if torch.backends.mps.is_available() == True:
    device = "mps"
if torch.cuda.is_available() == True:
    device = "cuda"
else:
    device = "cpu"

img = Image.open("guitar_top.png").resize((200, 200))
arr = np.array(img)

if arr.ndim == 3:
    gray = arr[:, :, 0]
else:
    gray = arr
plate = gray < 128
white = gray > 200

labels, num = label(white)
if num > 0:
    sizes = [(labels == i).sum() for i in range(1, num + 1)]
    soundhole_label = 1 + np.argmax(sizes)
    soundhole = labels == soundhole_label
else:
    soundhole = np.zeros_like(plate, bool)

plate[soundhole] = False
clamped = ~(plate | soundhole)

Nx, Ny = plate.shape

if arr.ndim == 3:
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    bridge_mask = (R > 150) & (G < 100) & (B < 100)
    bridge_mask = binary_dilation(bridge_mask, iterations=1)
else:
    bridge_mask = np.zeros_like(plate, bool)

if not np.any(bridge_mask):
    cx, cy = Nx // 2, int(Ny * 0.7)
    bridge_mask[cx - 1:cx + 2, cy - 1:cy + 2] = True

rows_with_plate = np.any(plate, axis=1)
plate_row_indices = np.where(rows_with_plate)[0]

top_row = plate_row_indices[0]
bottom_row = plate_row_indices[-1]
plate_pixel_height = bottom_row - top_row + 1

physical_height_m = 0.50  # 50 cm
dx_phys = physical_height_m / plate_pixel_height
physical_width_m = dx_phys * Ny

dx = dx_phys
dt = 0.05 * dx**2
D = 1
rho_h = 1.0
f = 200.0
omega = 2.0 * np.pi * f

plate_t = torch.tensor(plate, device=device, dtype=torch.bool)
soundhole_t = torch.tensor(soundhole, device=device, dtype=torch.bool)
clamped_t = torch.tensor(clamped, device=device, dtype=torch.bool)
bridge_t = torch.tensor(bridge_mask, device=device, dtype=torch.bool)

w = torch.zeros((1, 1, Nx, Ny), device=device, dtype=torch.float32)
w_prev = torch.zeros_like(w)

bih_kernel = torch.tensor([[0, 0, 1, 0, 0],[0, 2,-8, 2, 0],[1,-8,20,-8, 1],[0, 2,-8, 2, 0],[0, 0, 1, 0, 0]], dtype=torch.float32, device=device)

bih_kernel = bih_kernel[None, None, :, :] / (dx**4)

pixel_area = dx * dx

def step(w, w_prev, t):
    Fforce = torch.zeros_like(w)
    Fforce[0, 0][bridge_t] = pixel_area * torch.sin(torch.tensor(omega * t, device=device, dtype=torch.float32))

    bih = F.conv2d(w, bih_kernel, padding=2)

    w_next = 2 * w - w_prev - dt * dt * D / rho_h * bih + dt * dt * Fforce

    w_next[0, 0][clamped_t]   = 0.0
    w_next[0, 0][soundhole_t] = 0.0

    return w_next

steps = 10000
record_start = 0

amplitude_accum = torch.zeros_like(w)
count = 0

for n in range(steps):
    t = n * dt
    w_next = step(w, w_prev, t)
    w_prev, w = w, w_next

    if n > record_start:
        amplitude_accum += w * w
        count += 1

amplitude = torch.sqrt(amplitude_accum / max(count, 1))[0, 0]
amp_np = amplitude.cpu().numpy()
amp_np[clamped] = np.nan

extent = [0, physical_width_m * 100, physical_height_m * 100, 0]  # for cm

plt.figure(figsize=(6, 6))
plt.imshow(amp_np, cmap='inferno', origin='upper', extent=extent, aspect='equal')
plt.colorbar(label="Amplitude")
plt.show()
