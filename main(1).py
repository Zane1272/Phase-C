# ==========================================================
# Ukulele Acoustic Simulation (Stable Final Version)
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.signal import find_peaks
from scipy.spatial import Delaunay

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
    Material("Composite", 900, 5.0e9, 0.994)
]

# ==========================================================
# 2. GEOMETRY PROCESSING
# ==========================================================
class GeometryProcessor:

    def __init__(self, file, nx=160, ny=160):
        self.file = file
        self.nx = nx
        self.ny = ny

    def load_mesh(self):
        mesh = trimesh.load(self.file, force='mesh')

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()

        # ⭐只取外壳
        mesh = mesh.convex_hull

        return mesh

    def project_to_2D(self, mesh):

        pts = mesh.vertices[:, :2]

        pts -= pts.min(axis=0)
        pts /= pts.max(axis=0)

        grid_x, grid_y = np.meshgrid(
            np.linspace(0,1,self.nx),
            np.linspace(0,1,self.ny)
        )

        grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T

        tri = Delaunay(pts)

        inside = tri.find_simplex(grid_points) >= 0

        mask = inside.reshape(self.nx, self.ny)

        return mask.astype(float)

    def refine_mask(self, mask):
        mask = binary_dilation(mask, iterations=2)
        mask = gaussian_filter(mask.astype(float), sigma=1.0)
        mask = (mask > 0.2).astype(float)
        return mask

    def process(self):
        mesh = self.load_mesh()
        mask = self.project_to_2D(mesh)
        mask = self.refine_mask(mask)
        return mask

# ==========================================================
# 3. WAVE SOLVER
# ==========================================================
class WaveSolver:

    def __init__(self, nx, ny, dx, dt, mask):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.mask = mask

        self.u = np.zeros((nx, ny))
        self.u_prev = np.zeros((nx, ny))

    def reset(self):
        self.u.fill(0)
        self.u_prev.fill(0)

    def apply_excitation(self):
        self.u[self.nx//2, self.ny//3] = 1.0

    def laplacian(self):
        u = self.u
        return (
            np.roll(u, 1, axis=0) +
            np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) +
            np.roll(u, -1, axis=1) -
            4*u
        ) / (self.dx**2)

    def step(self, c, damping):

        lap = self.laplacian()

        u_new = 2*self.u - self.u_prev + (c**2)*(self.dt**2)*lap

        # 阻尼
        u_new *= damping

        # 几何约束
        u_new *= self.mask

        # ⭐防爆炸
        u_new = np.clip(u_new, -1e3, 1e3)

        self.u_prev = self.u
        self.u = u_new

    def energy(self):
        return np.sum(self.u**2)

# ==========================================================
# 4. ANALYSIS
# ==========================================================
class Analyzer:

    @staticmethod
    def fft(signal, dt):
        freq = np.fft.fftfreq(len(signal), dt)
        spec = np.abs(np.fft.fft(signal))
        return freq, spec

    @staticmethod
    def metrics(energy, freq, spectrum):

        sustain = energy[-1] / (energy[0] + 1e-8)

        peak_idx = np.argmax(spectrum[1:]) + 1
        peak_freq = freq[peak_idx]

        brightness = np.sum(spectrum[len(spectrum)//4:]) / np.sum(spectrum)

        return sustain, peak_freq, brightness

# ==========================================================
# 5. SIMULATION
# ==========================================================
class Simulation:

    def __init__(self, mask):
        self.mask = mask
        self.nx, self.ny = mask.shape
        self.dx = 0.01
        self.steps = 500

    def run(self, material):

        c = material.wave_speed()

        # ⭐CFL稳定
        dt = 0.3 * self.dx / (c + 1e-8)

        solver = WaveSolver(self.nx, self.ny, self.dx, dt, self.mask)

        solver.reset()
        solver.apply_excitation()

        energy_hist = []

        for t in range(self.steps):

            solver.step(c, material.damping)

            e = solver.energy()

            if np.isnan(e) or np.isinf(e):
                print(f"⚠️ Explosion at step {t}")
                break

            energy_hist.append(e)

        energy_hist = np.array(energy_hist)

        freq, spectrum = Analyzer.fft(energy_hist, dt)
        sustain, peak_freq, brightness = Analyzer.metrics(
            energy_hist, freq, spectrum
        )

        return {
            "field": solver.u.copy(),
            "energy": energy_hist,
            "freq": freq,
            "spectrum": spectrum,
            "metrics": {
                "sustain": sustain,
                "peak_freq": peak_freq,
                "brightness": brightness
            },
            "c": c
        }

# ==========================================================
# 6. VISUALISATION
# ==========================================================
class Visualizer:

    @staticmethod
    def mask(mask):
        plt.imshow(mask, cmap='gray')
        plt.title("Geometry Mask")
        plt.axis("off")
        plt.show()

    @staticmethod
    def fields(results):
        plt.figure(figsize=(12,4))
        for i,(name,data) in enumerate(results.items()):
            plt.subplot(1,len(results),i+1)
            plt.imshow(data["field"], cmap='hot')
            plt.title(name)
            plt.axis("off")
        plt.show()

    @staticmethod
    def energy(results):
        plt.figure()
        for name,data in results.items():
            plt.plot(data["energy"], label=name)
        plt.legend()
        plt.title("Energy Decay")
        plt.show()

    @staticmethod
    def spectrum(results):
        plt.figure()
        for name,data in results.items():
            plt.plot(data["freq"][:300], data["spectrum"][:300], label=name)
        plt.legend()
        plt.title("Frequency Spectrum")
        plt.show()

# ==========================================================
# 7. MAIN
# ==========================================================
if __name__ == "__main__":

    FILE = "Ukulele.stl"

    print("Processing geometry...")
    geo = GeometryProcessor(FILE)
    mask = geo.process()

    Visualizer.mask(mask)

    sim = Simulation(mask)

    results = {}

    for mat in materials:
        print(f"Simulating {mat.name}...")
        results[mat.name] = sim.run(mat)

    Visualizer.fields(results)
    Visualizer.energy(results)
    Visualizer.spectrum(results)

    print("\n=== RESULTS ===")
    for name,data in results.items():
        print(name, data["metrics"])