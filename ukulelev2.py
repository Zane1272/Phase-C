# ==========================================================
# Ukulele Acoustic Simulation (Stable Final Version)
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import trimesh
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label
from scipy.signal import find_peaks
from scipy.spatial import Delaunay

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

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label

class GeometryProcessorPNG:

    def __init__(self, file, nx=200, ny=200):
        self.file = file
        self.nx = nx
        self.ny = ny

    def process(self):
        img = Image.open(self.file).resize((self.nx, self.ny))
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

        mask = plate.astype(float)
        mask[clamped] = 0.0
        mask[soundhole] = 0.0

        if arr.ndim == 3:
            R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            bridge_mask = (R > 150) & (G < 100) & (B < 100)
            bridge_mask = binary_dilation(bridge_mask, iterations=1)
        else:
            bridge_mask = np.zeros_like(mask, bool)

        return mask, bridge_mask

class WaveSolver:

    def __init__(self, nx, ny, dx, dt, mask, bridge_mask):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.mask = mask
        self.bridge_mask = bridge_mask

        self.u = np.zeros((nx, ny))
        self.u_prev = np.zeros((nx, ny))

    def reset(self):
        self.u.fill(0)
        self.u_prev.fill(0)

    def apply_excitation(self):
        self.u[self.bridge_mask] = 1.0

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
        u_new *= damping
        u_new *= self.mask
        u_new = np.clip(u_new, -1e3, 1e3)
        self.u_prev = self.u
        self.u = u_new

    def energy(self):
        return np.sum(self.u**2)

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

class Simulation:

    def __init__(self, mask, bridge_mask):
        self.mask = mask
        self.bridge_mask = bridge_mask
        self.nx, self.ny = mask.shape
        self.dx = 0.01
        self.steps = 500

    def run(self, material):
        c = material.wave_speed()
        dt = 0.3 * self.dx / (c + 1e-8)

        solver = WaveSolver(self.nx, self.ny, self.dx, dt, self.mask, self.bridge_mask)

        solver.reset()
        solver.apply_excitation()

        energy_hist = []
        fields_over_time = []

        for t in range(self.steps):
            solver.step(c, material.damping)
            fields_over_time.append(solver.u.copy())
            e = solver.energy()
            if np.isnan(e) or np.isinf(e):
                break
            energy_hist.append(e)

        fields_over_time = np.array(fields_over_time)
        energy_hist = np.array(energy_hist)

        freq, spectrum = Analyzer.fft(energy_hist, dt)
        sustain, peak_freq, brightness = Analyzer.metrics(energy_hist, freq, spectrum)

        return {
            "field": solver.u.copy(),
            "fields_over_time": fields_over_time,
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

if __name__ == "__main__":

    FILE = "ukulele_top.png"

    print("Processing geometry...")
    geo = GeometryProcessorPNG(FILE)
    mask, bridge_mask = geo.process()

    Visualizer.mask(mask)

    sim = Simulation(mask, bridge_mask)

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

    # np.save("pla_waves.npy", results["PLA"]["fields_over_time"])
