import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_masks(file, nx=300, ny=300):
    img = Image.open(file).resize((nx, ny))
    arr = np.array(img)
    if arr.ndim == 3:
        gray = arr[:, :, 0]
    else:
        gray = arr
    body_mask = (gray < 128).astype(int)
    return body_mask

def compute_scale_from_mask(body_mask, real_length_cm=24.0):
    ys, xs = np.where(body_mask == 1)
    pts = np.column_stack([xs, ys])
    max_pixel_dist = 0.0
    for _ in range(2000):
        i1 = np.random.randint(len(pts))
        i2 = np.random.randint(len(pts))
        d = np.linalg.norm(pts[i1] - pts[i2])
        if d > max_pixel_dist:
            max_pixel_dist = d
    scale = real_length_cm / max_pixel_dist
    return scale, max_pixel_dist

def build_3d_volume(body_mask, scale_cm, height_cm, nz):
    z_scale_cm = height_cm / (nz - 1)
    ys, xs = np.where(body_mask == 1)
    points = []
    for (x, y) in zip(xs, ys):
        for k in range(nz):
            X_cm = x * scale_cm
            Y_cm = y * scale_cm
            Z_cm = k * z_scale_cm
            points.append([X_cm, Y_cm, Z_cm])
    return np.array(points)

def build_occupancy_grid(body_mask, nz):
    ny, nx = body_mask.shape
    occ = np.zeros((nz, ny, nx), dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            if body_mask[iy, ix] == 1:
                occ[:, iy, ix] = True
    return occ

def chord_stays_inside(p1, p2, occ_grid, scale_cm, height_cm, nz):
    z_scale_cm = height_cm / (nz - 1)
    for t in np.linspace(0, 1, 20):
        p = p1 * (1 - t) + p2 * t
        x_cm, y_cm, z_cm = p
        ix = int(x_cm / scale_cm)
        iy = int(y_cm / scale_cm)
        iz = int(z_cm / z_scale_cm)
        if ix < 0 or iy < 0 or iz < 0:
            return False
        if ix >= occ_grid.shape[2] or iy >= occ_grid.shape[1] or iz >= occ_grid.shape[0]:
            return False
        if not occ_grid[iz, iy, ix]:
            return False
    return True

def sample_3d_frequencies(points_cm, occ_grid, scale_cm, height_cm, nz, N_pairs, f_max):
    c = 343.0
    freqs = []
    amps = []
    n_pts = len(points_cm)
    for _ in range(N_pairs):
        i1 = np.random.randint(n_pts)
        i2 = np.random.randint(n_pts)
        p1 = points_cm[i1]
        p2 = points_cm[i2]
        if not chord_stays_inside(p1, p2, occ_grid, scale_cm, height_cm, nz):
            continue
        L_cm = np.linalg.norm(p1 - p2)
        L_m = L_cm / 100.0
        if L_m <= 0:
            continue
        f1 = c / (2.0 * L_m)
        n = 1
        while True:
            f = n * f1
            if f > f_max:
                break
            freqs.append(f)
            amps.append(1.0 / n)
            n += 1
    return np.array(freqs), np.array(amps)

def compute_helmholtz(body_mask, scale_cm, height_cm, soundhole_diameter_cm, top_thickness_cm=0.3):
    scale_m = scale_cm / 100.0
    height_m = height_cm / 100.0
    interior_pixels = np.sum(body_mask == 1)
    V = interior_pixels * (scale_m**2) * height_m
    r = (soundhole_diameter_cm / 2) / 100.0
    A = np.pi * r**2
    L_eff = (top_thickness_cm / 100.0) + 1.7 * r
    c = 343.0
    f_H = (c / (2 * np.pi)) * np.sqrt(A / (V * L_eff))
    return f_H

def helmholtz_eq(freqs, f_H, Q=5):
    return (freqs**2) / ((f_H**2 - freqs**2)**2 + (freqs * f_H / Q)**2)

def plot(freqs, amps, f_H):
    eq = helmholtz_eq(freqs, f_H)
    amps_eq = amps * eq
    amps_abs = amps_eq / np.sum(amps_eq)
    plt.figure(figsize=(14, 6))
    plt.hist(freqs, bins=100, weights=amps_abs, edgecolor='black')
    plt.axvline(f_H, color='red')
    plt.xlabel("Frequency Hz")
    plt.ylabel("Amplitude")
    plt.xscale("log")
    plt.xticks([400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500],["400", "500", "630", "800", "1k", "1.25k", "1.6k", "2k", "2.5k"])
    plt.grid(True)
    plt.xlim(300, 2500)
    plt.show()

if __name__ == "__main__":
    FILE = "ukulele_top.png"
    body_mask = load_masks(FILE)

    #parameters
    scale_cm, _ = compute_scale_from_mask(body_mask, real_length_cm=24.0)
    height_cm = 8.0
    soundhole_diameter_cm = 4


    nz = 100
    f_H = compute_helmholtz(body_mask, scale_cm, height_cm, soundhole_diameter_cm)
    points_cm = build_3d_volume(body_mask, scale_cm, height_cm, nz)
    occ_grid = build_occupancy_grid(body_mask, nz)
    freqs, amps = sample_3d_frequencies(points_cm, occ_grid, scale_cm, height_cm, nz,N_pairs=100000, f_max=2500.0)
    plot(freqs, amps, f_H)
