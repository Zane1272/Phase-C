import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label, binary_fill_holes


# ============================================================
# PARAMETERS
# ============================================================

# --- Image / detection ---
IMAGE_PATH = "ukulele_top.png"
RESIZE_TO = (200, 200)

THRESHOLD_DARK = 128
THRESHOLD_BRIGHT = 200

# --- Physical parameters ---
PHYSICAL_BODY_HEIGHT_M = 0.26   # ukulele body height [m]
CAVITY_VOLUME_M3 = 0.0025       # cavity volume [m^3]
TOP_THICKNESS_M = 0.003         # top thickness [m]
SPEED_OF_SOUND = 343.0          # [m/s]
DAMPING_ZETA = 0.05

# --- Frequency response ---
FREQ_MIN_HZ = 20
FREQ_MAX_HZ = 1000
N_FREQS = 1200

# --- Search controls ---
RADIUS_SCALE_FOR_REPORT = 1.0      # uses detected hole size for printed recommendation
RADIUS_SCALE_FOR_Y_SEARCH = 0.35   # reduced radius so valid positions can be explored
SEARCH_UPPER_FRACTION = 0.80       # search y only above this fraction of image height

# --- Plot options ---
SHOW_DEBUG_MASKS = True



def load_image_as_gray(image_path, resize_to):
    img = Image.open(image_path).resize(resize_to)
    arr = np.array(img)
    gray = arr[:, :, 0] if arr.ndim == 3 else arr
    return arr, gray


def detect_body_and_hole(gray, threshold_dark, threshold_bright):
    """
    Detect the instrument body as a filled region and detect the soundhole
    as the largest bright component inside that body.
    """
    dark = gray < threshold_dark
    body_mask = binary_fill_holes(dark)
    bright = gray > threshold_bright
    bright_inside = bright & body_mask

    labels, num = label(bright_inside)
    detected_hole = np.zeros_like(body_mask, dtype=bool)

    if num > 0:
        best_label = None
        best_size = -1

        for i in range(1, num + 1):
            region = labels == i

            
            touches_border = (
                region[0, :].any() or region[-1, :].any() or
                region[:, 0].any() or region[:, -1].any()
            )

            if not touches_border:
                size = region.sum()
                if size > best_size:
                    best_size = size
                    best_label = i

        if best_label is not None:
            detected_hole = labels == best_label

    return body_mask, detected_hole


def get_body_pixel_height(body_mask):
    rows_with_body = np.any(body_mask, axis=1)
    body_rows = np.where(rows_with_body)[0]

    if len(body_rows) == 0:
        raise ValueError("No body detected in image.")

    top_row = body_rows[0]
    bottom_row = body_rows[-1]
    return bottom_row - top_row + 1


def get_hole_geometry(hole_mask, rows, cols):
    if not np.any(hole_mask):
        # Fallback
        hole_x = cols / 2
        hole_y = rows / 2
        hole_radius_px = 12.0
        return hole_x, hole_y, hole_radius_px

    rr, cc = np.where(hole_mask)

    row_min, row_max = rr.min(), rr.max()
    col_min, col_max = cc.min(), cc.max()

    hole_y = 0.5 * (row_min + row_max)
    hole_x = 0.5 * (col_min + col_max)

    radius_y = 0.5 * (row_max - row_min)
    radius_x = 0.5 * (col_max - col_min)
    hole_radius_px = 0.5 * (radius_x + radius_y)

    return hole_x, hole_y, hole_radius_px


def build_circular_hole(rows, cols, center_x, center_y, radius_px):
    Y, X = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius_px ** 2


def hole_fits_inside_body(body_mask, hole_mask):
    return np.any(hole_mask) and np.all(body_mask[hole_mask])


def helmholtz_frequency(radius_px, body_pixel_height, physical_body_height_m,
                        cavity_volume_m3, top_thickness_m, speed_of_sound):
    """
    Convert radius from pixels to meters using the actual detected body height,
    then estimate Helmholtz frequency.
    """
    pixel_scale_m_per_px = physical_body_height_m / body_pixel_height

    radius_m = radius_px * pixel_scale_m_per_px
    area_m2 = np.pi * radius_m ** 2
    effective_neck_length_m = top_thickness_m + 1.7 * radius_m

    if area_m2 <= 0 or effective_neck_length_m <= 0 or cavity_volume_m3 <= 0:
        return np.nan, radius_m, area_m2, effective_neck_length_m, pixel_scale_m_per_px

    f_H = (speed_of_sound / (2 * np.pi)) * np.sqrt(
        area_m2 / (cavity_volume_m3 * effective_neck_length_m)
    )

    return f_H, radius_m, area_m2, effective_neck_length_m, pixel_scale_m_per_px


def coupling_factor_centered_x(center_x, center_y, cols, rows):
    """
    Simple reduced-order coupling model.
    Strongest near the geometric center, weaker near the edges.
    """
    x_norm = center_x / cols
    y_norm = center_y / rows
    coupling = np.sin(np.pi * x_norm) * np.sin(np.pi * y_norm)
    return max(coupling, 0.0)


def response_amplitude(freqs_hz, resonance_hz, coupling, damping_zeta):
    denom = np.sqrt(
        (resonance_hz ** 2 - freqs_hz ** 2) ** 2 +
        (2 * damping_zeta * resonance_hz * freqs_hz) ** 2
    )
    return coupling / denom


def body_centerline_x(body_mask):
    cols_with_body = np.any(body_mask, axis=0)
    body_cols = np.where(cols_with_body)[0]
    if len(body_cols) == 0:
        raise ValueError("No body columns detected.")
    return int(round(0.5 * (body_cols[0] + body_cols[-1])))


# ============================================================
# LOAD IMAGE
# ============================================================

arr, gray = load_image_as_gray(IMAGE_PATH, RESIZE_TO)
rows, cols = gray.shape


# ============================================================
# DETECT BODY AND ORIGINAL HOLE
# ============================================================

body_mask, detected_hole_mask = detect_body_and_hole(
    gray=gray,
    threshold_dark=THRESHOLD_DARK,
    threshold_bright=THRESHOLD_BRIGHT
)

detected_x, detected_y, detected_radius_px = get_hole_geometry(
    detected_hole_mask, rows, cols
)

body_pixel_height = get_body_pixel_height(body_mask)


# ============================================================
# DEFINE REPORTED SOUNDHOLE SIZE
# ============================================================

reported_radius_px = detected_radius_px * RADIUS_SCALE_FOR_REPORT

f_H_report, radius_m_report, area_m2_report, L_eff_report, pixel_scale = helmholtz_frequency(
    radius_px=reported_radius_px,
    body_pixel_height=body_pixel_height,
    physical_body_height_m=PHYSICAL_BODY_HEIGHT_M,
    cavity_volume_m3=CAVITY_VOLUME_M3,
    top_thickness_m=TOP_THICKNESS_M,
    speed_of_sound=SPEED_OF_SOUND
)


reported_hole_mask = build_circular_hole(
    rows, cols, detected_x, detected_y, reported_radius_px
)
reported_hole_mask = reported_hole_mask & body_mask

body_with_reported_hole = body_mask & (~reported_hole_mask)


# ============================================================
# FREQUENCY RESPONSE FOR CURRENT DETECTED HOLE
# ============================================================

freqs = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, N_FREQS)

current_coupling = coupling_factor_centered_x(
    center_x=detected_x,
    center_y=detected_y,
    cols=cols,
    rows=rows
)

current_response = response_amplitude(
    freqs_hz=freqs,
    resonance_hz=f_H_report,
    coupling=current_coupling,
    damping_zeta=DAMPING_ZETA
)


# ============================================================
# OPTIMISE ONLY ALONG Y AXIS (x fixed at centerline)
# ============================================================

x_center = body_centerline_x(body_mask)
search_radius_px = max(2, int(round(detected_radius_px * RADIUS_SCALE_FOR_Y_SEARCH)))

y_candidates = np.arange(rows)
peak_response_vs_y = np.full(rows, np.nan)
peak_frequency_vs_y = np.full(rows, np.nan)

search_y_max = int(rows * SEARCH_UPPER_FRACTION)

for y in y_candidates:
    if y > search_y_max:
        continue

    trial_hole_mask = build_circular_hole(
        rows=rows,
        cols=cols,
        center_x=x_center,
        center_y=y,
        radius_px=search_radius_px
    )

    if not hole_fits_inside_body(body_mask, trial_hole_mask):
        continue

    trial_coupling = coupling_factor_centered_x(
        center_x=x_center,
        center_y=y,
        cols=cols,
        rows=rows
    )

    trial_response = response_amplitude(
        freqs_hz=freqs,
        resonance_hz=f_H_report,
        coupling=trial_coupling,
        damping_zeta=DAMPING_ZETA
    )

    # Find peak of the response curve
    peak_idx = np.argmax(trial_response)

    peak_response_vs_y[y] = trial_response[peak_idx]
    peak_frequency_vs_y[y] = freqs[peak_idx]

if np.all(np.isnan(peak_response_vs_y)):
    raise ValueError("No valid y-positions found. Reduce RADIUS_SCALE_FOR_Y_SEARCH.")

best_y = int(np.nanargmax(peak_response_vs_y))
best_x = x_center
best_peak_response = peak_response_vs_y[best_y]

optimal_hole_mask = build_circular_hole(
    rows=rows,
    cols=cols,
    center_x=best_x,
    center_y=best_y,
    radius_px=search_radius_px
)
optimal_hole_mask = optimal_hole_mask & body_mask

body_with_optimal_hole = body_mask & (~optimal_hole_mask)



# ============================================================
# PRINT FINAL DESIGN RESULT
# ============================================================

print("\n================ DESIGN RESULT ================\n")
print("Detected body height in image:")
print(f"  {body_pixel_height} px")

print("\nPixel scale:")
print(f"  {pixel_scale * 1000:.3f} mm/pixel")

print("\nDetected original soundhole:")
print(f"  center = ({detected_x:.1f}, {detected_y:.1f}) px")
print(f"  detected radius = {detected_radius_px:.2f} px")

print("\nRecommended soundhole size:")
print(f"  radius = {radius_m_report * 1000:.2f} mm")
print(f"  diameter = {2 * radius_m_report * 1000:.2f} mm")

print("\nOptimal soundhole placement:")
print(f"  x = {best_x:.1f} px (fixed at body centerline)")
print(f"  y = {best_y:.1f} px")

print("\nAcoustic estimate:")
print(f"  Helmholtz frequency ≈ {f_H_report:.1f} Hz")
print(f"  Effective neck length ≈ {L_eff_report * 1000:.2f} mm")
print(f"  Peak response at optimal y ≈ {best_peak_response:.6e}")

print("\nInterpretation:")
print("  - Hole size mainly sets the Helmholtz resonance frequency.")
print("  - Hole y-position mainly changes coupling strength.")




# ============================================================
# PLOT 1: CURRENT GEOMETRY
# ============================================================

plt.figure(figsize=(6, 6))
plt.imshow(body_with_reported_hole, cmap="gray", origin="upper")
plt.scatter([detected_x], [detected_y], c="cyan", s=50, label="detected hole center")
plt.title("Current Ukulele Geometry")
plt.legend()
plt.axis("on")
plt.tight_layout()
if SAVE_PLOTS:
    plt.savefig("current_geometry.png", dpi=300)
plt.show()


# ============================================================
# PLOT 2: CURRENT FREQUENCY RESPONSE
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(freqs, current_response, label="response")
plt.axvline(f_H_report, color="red", linestyle="--", label=f"f_H = {f_H_report:.1f} Hz")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Relative response")
plt.title("Frequency Response for Current Soundhole")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("frequency_response.png", dpi=300)
plt.show()


# ============================================================
# PLOT 3: PEAK RESPONSE VS Y-POSITION
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(y_candidates, peak_response_vs_y, color="purple", linewidth=2)
plt.axvline(best_y, color="lime", linestyle="--", label=f"optimal y = {best_y}")
plt.xlabel("Soundhole center y-position [pixels]")
plt.ylabel("Peak response")
plt.title("Effect of Soundhole Y-Position on Response")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("response_vs_y.png", dpi=300)
plt.show()


# ============================================================
# PLOT 4: FINAL OPTIMAL PLACEMENT
# ============================================================

plt.figure(figsize=(6, 6))
plt.imshow(body_mask, cmap="gray", origin="upper")
plt.scatter([detected_x], [detected_y], c="white", s=50, label="detected hole")
plt.scatter([best_x], [best_y], c="lime", s=70, label="optimal position")
circle = plt.Circle((best_x, best_y), search_radius_px, color="lime", fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.title("Optimal Soundhole Placement (x fixed at centerline)")
plt.legend()
plt.axis("on")
plt.tight_layout()
plt.savefig("optimal_placement.png", dpi=300)
plt.show()

# ============================================================
# PLOT: PEAK FREQUENCY + AMPLITUDE vs Y POSITION
# ============================================================

valid_mask = ~np.isnan(peak_response_vs_y) & ~np.isnan(peak_frequency_vs_y)

y_plot = np.arange(rows)[valid_mask]
amp_plot = peak_response_vs_y[valid_mask]
freq_plot = peak_frequency_vs_y[valid_mask]

fig, ax1 = plt.subplots(figsize=(9, 5))

# Amplitude (left axis)
ax1.plot(y_plot, amp_plot, color="purple", linewidth=2, label="Peak amplitude")
ax1.axvline(best_y, color="lime", linestyle="--", label=f"optimal y = {best_y}")
ax1.set_xlabel("Soundhole Y position [pixels]")
ax1.set_ylabel("Peak amplitude", color="purple")
ax1.tick_params(axis="y", labelcolor="purple")
ax1.grid(True)

# Frequency (right axis)
ax2 = ax1.twinx()
ax2.plot(y_plot, freq_plot, color="red", linestyle="--", label="Peak frequency")
ax2.set_ylabel("Frequency [Hz]", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("Effect of Soundhole Position on Amplitude and Frequency")
fig.tight_layout()
plt.show()




