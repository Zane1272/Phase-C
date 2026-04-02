import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation, label


# -----------------------------
# User settings
# -----------------------------
IMAGE_PATH = "ukulele_top.png"
RESIZE_TO = (200, 200)

THRESHOLD_PLATE = 128
THRESHOLD_HOLE = 200

# Move hole relative to detected center
DX_HOLE = 0          # + right, - left
DY_HOLE = 0       # + down,  - up

# Resize relative to detected radius
RADIUS_SCALE = 1.00

# If True, ignore detected position and use absolute values
USE_ABSOLUTE_HOLE = False
ABS_X = 100
ABS_Y = 100
ABS_RADIUS = 18

# Plot options
SHOW_BRIDGE = True
SHOW_DEBUG_DETECTED_HOLE = True

# -----------------------------
# Load image
# -----------------------------
img = Image.open(IMAGE_PATH).resize(RESIZE_TO)
arr = np.array(img)

if arr.ndim == 3:
    gray = arr[:, :, 0]
else:
    gray = arr

rows, cols = gray.shape


# -----------------------------
# Build masks
# -----------------------------
# Guitar plate = dark region
plate_base = gray < THRESHOLD_PLATE

# Candidate hole regions = bright region
white = gray > THRESHOLD_HOLE

# Connected components of bright regions
labels, num = label(white)

if num > 0:
    sizes = [(labels == i).sum() for i in range(1, num + 1)]
    soundhole_label = 1 + np.argmax(sizes)
    detected_hole = labels == soundhole_label
else:
    detected_hole = np.zeros_like(plate_base, dtype=bool)


# -----------------------------
# Detect bridge from red pixels
# -----------------------------
if arr.ndim == 3:
    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    bridge_mask = (R > 150) & (G < 100) & (B < 100)
    bridge_mask = binary_dilation(bridge_mask, iterations=1)
else:
    bridge_mask = np.zeros_like(plate_base, dtype=bool)


# -----------------------------
# Infer detected hole center/radius
# -----------------------------
if np.any(detected_hole):
    hole_rows, hole_cols = np.where(detected_hole)

    # Bounding box of detected hole
    row_min, row_max = hole_rows.min(), hole_rows.max()
    col_min, col_max = hole_cols.min(), hole_cols.max()

    # Use bounding-box center for a more visually stable center
    detected_y = 0.5 * (row_min + row_max)
    detected_x = 0.5 * (col_min + col_max)

    # Radius estimate from bounding box
    radius_y = 0.5 * (row_max - row_min)
    radius_x = 0.5 * (col_max - col_min)
    detected_radius = 0.5 * (radius_x + radius_y)

    # Area-based radius too, just for comparison
    detected_area = float(detected_hole.sum())
    detected_radius_area = np.sqrt(detected_area / np.pi)
else:
    detected_y = rows / 2
    detected_x = cols / 2
    detected_radius = 15.0
    detected_radius_area = 15.0


# -----------------------------
# Build adjusted hole
# -----------------------------
if USE_ABSOLUTE_HOLE:
    hole_x = float(ABS_X)
    hole_y = float(ABS_Y)
    hole_radius = float(ABS_RADIUS)
else:
    hole_x = detected_x + DX_HOLE
    hole_y = detected_y + DY_HOLE
    hole_radius = detected_radius * RADIUS_SCALE

Y, X = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

adjusted_hole = (X - hole_x) ** 2 + (Y - hole_y) ** 2 <= hole_radius ** 2

# Keep new hole only where it lies inside the guitar top
adjusted_hole = adjusted_hole & plate_base

# Final plate
plate_with_hole = plate_base & (~adjusted_hole)

# Quantities
hole_area_px = adjusted_hole.sum()
plate_area_px = plate_base.sum()
removed_fraction = hole_area_px / plate_area_px if plate_area_px > 0 else np.nan


# -----------------------------
# Report
# -----------------------------
print("Detected hole center (x, y):", (round(detected_x, 2), round(detected_y, 2)))
print("Detected radius from bounding box:", round(detected_radius, 2), "px")
print("Detected radius from area:", round(detected_radius_area, 2), "px")
print("Adjusted hole center (x, y):", (round(hole_x, 2), round(hole_y, 2)))
print("Adjusted hole radius:", round(hole_radius, 2), "px")
print("Adjusted hole area:", int(hole_area_px), "pixels")
print("Fraction of plate removed:", f"{100 * removed_fraction:.2f}%")


# -----------------------------
# Debug plot: detected hole only
# -----------------------------
if SHOW_DEBUG_DETECTED_HOLE and np.any(detected_hole):
    fig_dbg, ax_dbg = plt.subplots(figsize=(5, 5))
    ax_dbg.imshow(detected_hole, cmap="gray", origin="upper")
    ax_dbg.scatter([detected_x], [detected_y], c="red", s=50, label="detected center")
    ax_dbg.set_title("Detected hole mask and center")
    ax_dbg.axis("on")
    ax_dbg.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Final plot
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 7))
canvas = np.ones((rows, cols, 3), dtype=float)

# Outside plate = light gray
canvas[~plate_base] = [0.88, 0.88, 0.88]

# Plate = dark gray
canvas[plate_with_hole] = [0.20, 0.20, 0.20]

# Adjusted hole = white
canvas[adjusted_hole] = [1.0, 1.0, 1.0]

# Bridge = red
if SHOW_BRIDGE and np.any(bridge_mask):
    canvas[bridge_mask] = [1.0, 0.0, 0.0]

ax.imshow(canvas, origin="upper")

# Mark centers
ax.scatter([detected_x], [detected_y], c="cyan", s=50, label="detected hole center")
ax.scatter([hole_x], [hole_y], c="yellow", s=50, label="adjusted hole center")

ax.set_title("Guitar top with adjustable soundhole")
ax.axis("on")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("soundhole_adjust.png", dpi=300)
plt.show()
