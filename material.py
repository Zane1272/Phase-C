# ==========================================================
# Ukulele Acoustic Simulation
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label


# ==========================================================
# 1. MATERIAL DATABASE
# ==========================================================
class Material:
    def __init__(self, name, rho, E, damping):
        self.name = name
        self.rho = rho
        self.E = E
        self.damping = damping

    def wave_speed(self):
        return np.sqrt(self.E / self.rho)


materials = [
    Material("Wood", 600, 1.0e10, 0.995),
    Material("PLA", 1250, 3.0e9, 0.993),
    Material("ABS", 1050, 2.0e9, 0.992),
    Material("Composite", 900, 5.0e9, 0.994),
]


# ==========================================================
# 2. GEOMETRY PROCESSING
# ==========================================================
class GeometryProcessorPNG:
    """
    Assumptions for the input image:
    - dark region = vibrating top plate
    - internal white region = sound hole
    - red region (if present) = bridge excitation region
    """

    def __init__(self, file, nx=140, ny=140):
        self.file = file
        self.nx = nx
        self.ny = ny

    def process(self):
        img = Image.open(self.file).convert("RGB").resize((self.nx, self.ny))
        arr = np.array(img)

        gray = np.mean(arr, axis=2)

        # Plate: dark region
        plate = gray < 128

        # White regions
        white = gray > 220
        labels, num = label(white)

        soundhole = np.zeros_like(plate, dtype=bool)

        if num > 0:
            candidates = []
            for i in range(1, num + 1):
                region = (labels == i)
                ys, xs = np.where(region)
                if len(xs) == 0:
                    continue

                touches_border = (
                    np.any(xs == 0) or np.any(xs == self.nx - 1) or
                    np.any(ys == 0) or np.any(ys == self.ny - 1)
                )

                if not touches_border:
                    candidates.append(region)

            if len(candidates) > 0:
                soundhole = max(candidates, key=lambda r: np.sum(r))

        plate[soundhole] = False
        mask = plate.astype(float)

        # Bridge mask from red pixels
        R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        bridge_mask = (R > 150) & (G < 120) & (B < 120)
        bridge_mask &= plate

        # Fallback bridge region
        if np.sum(bridge_mask) == 0:
            cx = self.nx // 2
            cy = int(self.ny * 0.35)
            bridge_mask = np.zeros_like(mask, dtype=bool)
            bridge_mask[max(cx - 2, 0):min(cx + 3, self.nx),
                        max(cy - 4, 0):min(cy + 5, self.ny)] = True
            bridge_mask &= plate

        probe = self._find_probe_point(plate)
        return mask, bridge_mask, soundhole, probe

    def _find_probe_point(self, plate):
        nx, ny = plate.shape
        candidates = [
            (nx // 2, ny // 2),
            (nx // 2, int(ny * 0.55)),
            (nx // 2, int(ny * 0.60)),
            (int(nx * 0.48), int(ny * 0.55)),
            (int(nx * 0.52), int(ny * 0.55)),
        ]

        for px, py in candidates:
            if 0 <= px < nx and 0 <= py < ny and plate[px, py]:
                return (px, py)

        xs, ys = np.where(plate)
        return (xs[len(xs)//2], ys[len(ys)//2])


# ==========================================================
# 3. WAVE SOLVER
# ==========================================================
class WaveSolver:
    def __init__(self, nx, ny, dx, dt, mask, bridge_mask):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.mask = mask
        self.bridge_mask = bridge_mask

        self.u = np.zeros((nx, ny), dtype=float)
        self.u_prev = np.zeros((nx, ny), dtype=float)

    def reset(self):
        self.u.fill(0.0)
        self.u_prev.fill(0.0)

    def apply_excitation(self, amplitude=1.0):
        self.u[self.bridge_mask] = amplitude

    def laplacian(self):
        lap = np.zeros_like(self.u)
        u = self.u

        lap[1:-1, 1:-1] = (
            u[2:, 1:-1] +
            u[:-2, 1:-1] +
            u[1:-1, 2:] +
            u[1:-1, :-2] -
            4 * u[1:-1, 1:-1]
        ) / (self.dx ** 2)

        return lap

    def step(self, c, damping):
        lap = self.laplacian()
        u_new = 2 * self.u - self.u_prev + (c ** 2) * (self.dt ** 2) * lap
        u_new *= damping
        u_new *= self.mask
        u_new = np.clip(u_new, -1e3, 1e3)

        self.u_prev = self.u.copy()
        self.u = u_new

    def energy(self):
        return np.sum(self.u ** 2)


# ==========================================================
# 4. ANALYSIS
# ==========================================================
class Analyzer:
    @staticmethod
    def fft(signal, dt):
        signal = np.asarray(signal, dtype=float)
        signal = signal - np.mean(signal)

        window = np.hanning(len(signal))
        signal_win = signal * window

        spec = np.abs(np.fft.rfft(signal_win))
        freq = np.fft.rfftfreq(len(signal_win), dt)
        return freq, spec

    @staticmethod
    def metrics(probe_signal, energy, freq, spectrum):
        probe_signal = np.asarray(probe_signal)
        energy = np.asarray(energy)

        env = np.abs(probe_signal)
        if np.max(env) > 0:
            env = env / np.max(env)

        idx = np.where(env > 0.1)[0]
        sustain_steps = idx[-1] - idx[0] if len(idx) > 1 else 0

        if len(spectrum) > 1:
            peak_idx = np.argmax(spectrum[1:]) + 1
            peak_freq = freq[peak_idx]
        else:
            peak_freq = 0.0

        total_spec = np.sum(spectrum) + 1e-12
        brightness = np.sum(spectrum[len(spectrum)//4:]) / total_spec
        energy_ratio = energy[-1] / (energy[0] + 1e-12) if len(energy) > 1 else 0.0

        return {
            "peak_freq": peak_freq,
            "brightness": brightness,
            "energy_ratio": energy_ratio,
            "sustain_steps": sustain_steps,
        }


# ==========================================================
# 5. SIMULATION
# ==========================================================
class Simulation:
    def __init__(self, mask, bridge_mask, probe, dx=0.01, total_time=0.01):
        self.mask = mask
        self.bridge_mask = bridge_mask
        self.probe = probe
        self.nx, self.ny = mask.shape
        self.dx = dx
        self.total_time = total_time

    def run(self, material):
        c = material.wave_speed()
        dt = 0.3 * self.dx / (c + 1e-12)
        steps = max(50, int(self.total_time / dt))

        solver = WaveSolver(self.nx, self.ny, self.dx, dt, self.mask, self.bridge_mask)
        solver.reset()
        solver.apply_excitation(amplitude=1.0)

        energy_hist = []
        probe_signal = []

        px, py = self.probe

        for _ in range(steps):
            solver.step(c, material.damping)

            e = solver.energy()
            if np.isnan(e) or np.isinf(e):
                print(f"⚠️ Numerical instability in {material.name}")
                break

            energy_hist.append(e)
            probe_signal.append(solver.u[px, py])

        energy_hist = np.array(energy_hist)
        probe_signal = np.array(probe_signal)

        freq, spectrum = Analyzer.fft(probe_signal, dt)
        metrics = Analyzer.metrics(probe_signal, energy_hist, freq, spectrum)

        return {
            "field": solver.u.copy(),
            "energy": energy_hist,
            "probe_signal": probe_signal,
            "freq": freq,
            "spectrum": spectrum,
            "metrics": metrics,
            "c": c,
            "dt": dt,
            "steps": steps,
        }


# ==========================================================
# 6. VISUALISATION
# ==========================================================
class Visualizer:
    COLORS = {
        "Wood": "#1f77b4",
        "PLA": "#ff7f0e",
        "ABS": "#2ca02c",
        "Composite": "#d62728",
    }

    @staticmethod
    def mask(mask, bridge_mask=None, soundhole=None, probe=None):
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
        ax.set_facecolor("white")
        ax.imshow(mask, cmap="gray")
        ax.set_title("Geometry Mask", fontsize=14, fontweight="bold")
        ax.axis("off")

        if soundhole is not None:
            ys, xs = np.where(soundhole)
            ax.scatter(xs, ys, s=2, c="cyan", label="Sound hole")

        if bridge_mask is not None:
            ys, xs = np.where(bridge_mask)
            ax.scatter(xs, ys, s=4, c="red", label="Bridge")

        if probe is not None:
            ax.scatter(probe[1], probe[0], s=45, c="blue", marker="x", label="Probe")

        if bridge_mask is not None or soundhole is not None or probe is not None:
            ax.legend(loc="lower right", fontsize=8, frameon=False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def fields(results):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor="white")
        axes = axes.flatten()

        all_fields = [data["field"] for data in results.values()]
        vmin = min(np.min(f) for f in all_fields)
        vmax = max(np.max(f) for f in all_fields)

        for ax, (name, data) in zip(axes, results.items()):
            im = ax.imshow(data["field"], cmap="hot", vmin=vmin, vmax=vmax)
            ax.set_title(name, fontsize=14, fontweight="bold")
            ax.axis("off")

        cbar = fig.colorbar(im, ax=axes, shrink=0.82, location="right")
        cbar.set_label("Vibration amplitude", fontsize=11)

        plt.suptitle("Spatial Distribution of Vibration Response", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def energy(results, max_time=0.003):
        plt.figure(figsize=(8, 5), facecolor="white")

        for name, data in results.items():
            t = np.arange(len(data["energy"])) * data["dt"]
            mask = t <= max_time
            plt.plot(
                t[mask],
                data["energy"][mask],
                label=name,
                linewidth=2.2,
                color=Visualizer.COLORS.get(name, None),
            )

        plt.yscale("log")
        plt.title("Energy Decay", fontsize=16, fontweight="bold")
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Total Energy (log scale)", fontsize=12)
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def probe_signal(results, max_time=0.003):
        plt.figure(figsize=(8, 5), facecolor="white")

        for name, data in results.items():
            t = np.arange(len(data["probe_signal"])) * data["dt"]
            mask = t <= max_time
            plt.plot(
                t[mask],
                data["probe_signal"][mask],
                label=name,
                linewidth=2.2,
                color=Visualizer.COLORS.get(name, None),
            )

        plt.title("Probe Displacement Response", fontsize=16, fontweight="bold")
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Displacement", fontsize=12)
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def spectrum(results, max_freq=1500):
        plt.figure(figsize=(8, 5), facecolor="white")

        for name, data in results.items():
            freq = data["freq"]
            spec = data["spectrum"].copy()
            spec = spec / (np.max(spec) + 1e-12)

            mask = freq <= max_freq
            plt.plot(
                freq[mask],
                spec[mask],
                label=name,
                linewidth=2.2,
                color=Visualizer.COLORS.get(name, None),
            )

        plt.title("Spectrum of Probe Response", fontsize=16, fontweight="bold")
        plt.xlabel("Frequency (Hz)", fontsize=12)
        plt.ylabel("Normalised Magnitude", fontsize=12)
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()


# ==========================================================
# 7. MAIN
# ==========================================================
if __name__ == "__main__":
    FILE = "ukulele_top.png"

    print("Processing geometry...")
    geo = GeometryProcessorPNG(FILE, nx=140, ny=140)
    mask, bridge_mask, soundhole, probe = geo.process()

    Visualizer.mask(mask, bridge_mask=bridge_mask, soundhole=soundhole, probe=probe)

    sim = Simulation(mask, bridge_mask, probe, dx=0.01, total_time=0.01)

    results = {}

    for mat in materials:
        print(f"Simulating {mat.name}...")
        results[mat.name] = sim.run(mat)

    Visualizer.fields(results)
    Visualizer.energy(results, max_time=0.003)
    Visualizer.probe_signal(results, max_time=0.003)
    Visualizer.spectrum(results, max_freq=1500)

    print("\n=== RESULTS ===")
    for name, data in results.items():
        print(
            f"{name}: "
            f"wave_speed={data['c']:.2f}, "
            f"peak_freq={data['metrics']['peak_freq']:.2f} Hz, "
            f"brightness={data['metrics']['brightness']:.4f}, "
            f"energy_ratio={data['metrics']['energy_ratio']:.4f}, "
            f"steps={data['steps']}"
        )