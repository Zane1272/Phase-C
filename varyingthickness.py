#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label

# -----------------------------
# Load geometry
# -----------------------------

def load_plate(image, nx=200, ny=200):

    img = Image.open(image).resize((nx, ny))
    arr = np.array(img)

    if arr.ndim == 3:
        gray = arr[:, :, 0]
    else:
        gray = arr

    plate = gray < 128

    return plate


# -----------------------------
# Fake vibration field
# -----------------------------

def plate_mode_shape(plate):

    Nx, Ny = plate.shape

    x = np.linspace(0, np.pi, Nx)
    y = np.linspace(0, np.pi, Ny)

    X, Y = np.meshgrid(x, y, indexing="ij")

    # simple plate mode
    field = np.sin(2*X) * np.sin(3*Y)

    field[~plate] = np.nan

    return field


# -----------------------------
# Fake acoustic pressure
# -----------------------------

def acoustic_field(field):

    pressure = np.abs(field)**2

    return pressure


# -----------------------------
# Run
# -----------------------------

FILE = "ukulele_top.png"

plate = load_plate(FILE)

mode = plate_mode_shape(plate)

pressure = acoustic_field(mode)

# -----------------------------
# Plot results
# -----------------------------

plt.figure(figsize=(6,6))
plt.title("Plate Mode Shape")
plt.imshow(mode, cmap="viridis")
plt.colorbar(label="displacement")
plt.show()


plt.figure(figsize=(6,6))
plt.title("Acoustic Pressure (fake)")
plt.imshow(pressure, cmap="inferno")
plt.colorbar(label="pressure")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation, label

import torch.nn.functional as F

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

def plate_frequency_map(image, material):

    img = Image.open(image).resize((200,200))
    arr = np.array(img)

    if arr.ndim == 3:
        gray = arr[:,:,0]
    else:
        gray = arr

    plate = gray < 128

    Nx, Ny = plate.shape

    c = material.wave_speed()

    L = 0.5        # physical plate length (m)
    k = np.pi / L

    omega_p_map = np.zeros_like(plate, dtype=float)

    omega_p_map[plate] = c * k

    return omega_p_map, plate

def plate_air_response(omega_p_map, plate, material):

    Nx, Ny = omega_p_map.shape

    m_a = 0.01
    R_p = 0.5
    R_a = 0.05
    A = 0.02
    S = 0.01
    F0 = 1.0
     # plate stiffness constant

    omega_a = 200
    a_coupling = 500

    frequencies = np.linspace(50,400,1000)
    omega = 2*np.pi*frequencies
    omega3 = omega[:,None,None]

    gamma_p = R_p
    gamma_a = R_a/m_a

    omega_c2 = a_coupling

    h = 0.003  # plate thickness (m)
    nu = 0.3

    #using the model from the paper leissa vibration of plates

    D = material.E * h**3 / (12*(1-nu**2))

    omega_p = np.sqrt(D / (m_p * h))

    u_p = 1j*omega3*F0*(omega_a**2 - omega3**2 + 1j*gamma_a*omega3)/D

    u_a = -1j*omega3*F0*(A/S)*(omega_p**2 - omega3**2 + 1j*gamma_p*omega3)/D

    rho_air = 1.2
    R_dist = 1.0

    U = A*u_p + S*u_a

    p_sound = -1j*rho_air*omega3*U/(4*np.pi*R_dist)

    u_p[:,~plate] = np.nan
    u_a[:,~plate] = np.nan
    p_sound[:,~plate] = np.nan

    return frequencies, u_p, u_a, p_sound

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

FILE = "ukulele_top.png"

print("Processing geometry...")
geo = GeometryProcessorPNG(FILE)
mask, bridge_mask = geo.process()

Visualizer.mask(mask)

def rho_plate(x,y): #to be ammended according to Joanna´s findings
    return 600



def mass_function(image, rho):

    img = Image.open(image).resize((200,200))
    arr = np.array(img)

    if arr.ndim == 3:
        gray = arr[:,:,0]
    else:
        gray = arr

    plate = gray < 128

    Nx, Ny = plate.shape
    plate_mass = np.zeros_like(plate, dtype=float)

    for x in range(Nx):
        for y in range(Ny):
            if plate[x,y]:
                plate_mass[x,y] = rho(x,y)

    rows_with_plate = np.any(plate, axis=1)
    plate_row_indices = np.where(rows_with_plate)[0]

    top_row = plate_row_indices[0]
    bottom_row = plate_row_indices[-1]
    plate_pixel_height = bottom_row - top_row + 1

    return plate_mass, plate_pixel_height


m_p,plate_pixel_height = mass_function('ukulele_top.png',rho_plate)
m_p[m_p == 0] = np.nan
# we want this to be a function 

# Derived frequencies
omega_p = np.sqrt(k_p / m_p)      # rad/s, natural freq plate
omega_a = 200.0                     # rad/s, Helmholtz frequency (example)
a_coupling = 500.0                  # coupling constant
omega_c2 = a_coupling / np.sqrt(m_p * m_a)  # coupling frequency squared

# Frequency array
frequencies = np.linspace(50, 400, 1000)  # Hz
omega = 2 * np.pi * frequencies           # rad/s
omega3 = omega[:, None, None] #make sure shapes match
# Damping
gamma_p = R_p / m_p
gamma_a = R_a / m_a

# Frequency response
D = (omega_p**2 - omega3**2 + 1j*gamma_p*omega3) * \
    (omega_a**2 - omega3**2 + 1j*gamma_a*omega3) - omega_c2**2 #match D to fit 2D plate sim


u_p = 1j * omega3 * (F0 / m_p) * (omega_a**2 - omega3**2 + 1j*gamma_a*omega3) / D
u_a = -1j * omega3 * (F0 / m_p) * (A/S) * (omega_p**2 - omega3**2 + 1j*gamma_p*omega3) / D

# Sound pressure (far field)
rho_air = 1.2      
R_dist = 1.0   # m, distance to microphone
U = A * u_p + S * u_a
p_sound = -1j * rho_air * omega3 * U / (4 * np.pi * R_dist)

# we want a map of the top plate at 440Hz, where we see the air piston movement at each position

# we want to map total top plate and air piston velocity with plastic properties to match results from paper
# making sure to uniformise formats
physical_height_m = 0.50  # 50 cm
dx_phys = physical_height_m / plate_pixel_height
physical_width_m = dx_phys * m_p.shape[1]

extent = [0, physical_width_m * 100, physical_height_m * 100, 0]  # for cm

freq = data["freq"]
idx = np.argmin(np.abs(freq-440))

results_acoustic = {}

for mat in materials:

    omega_map, plate = plate_frequency_map("ukulele_top.png", mat)

    freq, u_p, u_a, p = plate_air_response(omega_map, plate, mat)

    results_acoustic[mat.name] = {
        "freq": freq,
        "u_p": u_p,
        "u_a": u_a,
        "p": p
    }

plt.figure(figsize=(6,6))
plt.imshow(np.abs(u_p[idx]), cmap='viridis', origin='upper', extent=extent, aspect='equal')
plt.colorbar(label="Top plate velocity (magnitude)")
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(np.abs(u_a[idx]), cmap='viridis', origin='upper', extent=extent, aspect='equal')
plt.colorbar(label="Air piston velocity (magnitude)")
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(np.abs(p_sound[idx]), cmap='cividis', origin='upper', extent=extent, aspect='equal')
plt.colorbar(label="Sound pressure (magnitude)")
plt.show()

idx = np.argmin(np.abs(freq-440))

for name,data in results_acoustic.items():

    freq = data["freq"]
    idx = np.argmin(np.abs(freq-440))

    plt.figure(figsize=(6,6))
    plt.title(name + " Plate Velocity")

    plt.imshow(np.abs(data["u_p"][idx]), cmap="viridis")
    plt.colorbar()
    plt.show()
