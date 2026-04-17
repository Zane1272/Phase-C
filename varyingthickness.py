#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label

# In[1]:
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

def load_plate(image, nx=200, ny=200):

    img = Image.open(image).resize((nx, ny))
    arr = np.array(img)

    if arr.ndim == 3:
        gray = arr[:, :, 0]
    else:
        gray = arr

    plate = gray < 128

    return plate

FILE = "ukulele_top.png"

plate = load_plate(FILE)

def rho_plate(x,y):
    return 600

plate_mass = rho_plate(0,0) * plate.astype(float)

def plate_frequency_map(plate, material):

    Nx, Ny = plate.shape

    c = material.wave_speed()

    L = 0.5
    k = np.pi / L

    omega_p_map = np.zeros_like(plate, dtype=float)

    omega_p_map[plate] = c * k

    omega_p_map[~plate] = np.nan

    return omega_p_map


def plate_air_response(omega_p_map, plate, material):

    Nx, Ny = omega_p_map.shape

    # parameters
    m_a = 0.01
    R_p = 0.5
    R_a = 0.05
    A = 0.02
    S = 0.01
    F0 = 1.0

    omega_a = 200
    a_coupling = 500

    frequencies = np.linspace(50, 400, 1000)
    omega = 2*np.pi*frequencies
    omega3 = omega[:, None, None]

    gamma_p = R_p
    gamma_a = R_a/m_a

    omega_c2 = a_coupling

    # plate parameters
    h = 0.003
    nu = 0.3

    rho_p = material.rho
    m_p = rho_p * h

    # bending stiffness
    D = material.E * h**3 / (12*(1-nu**2))

    omega_p = omega_p_map

    denom = (
        (omega_p**2 - omega3**2 + 1j*gamma_p*omega3) *
        (omega_a**2 - omega3**2 + 1j*gamma_a*omega3)
        - omega_c2
    )

    u_p = 1j*omega3*F0*(omega_a**2 - omega3**2 + 1j*gamma_a*omega3)/denom
    u_a = -1j*omega3*F0*(A/S)*(omega_p**2 - omega3**2 + 1j*gamma_p*omega3)/denom

    rho_air = 1.2
    R_dist = 1.0

    U = A*u_p + S*u_a

    p_sound = -1j*rho_air*omega3*U/(4*np.pi*R_dist)

    u_p[:,~plate] = np.nan
    u_a[:,~plate] = np.nan
    p_sound[:,~plate] = np.nan

    return frequencies, u_p, u_a, p_sound

def plate_mode_shape(plate,material):

    Nx, Ny = plate.shape

    x = np.linspace(0, np.pi, Nx)
    y = np.linspace(0, np.pi, Ny)

    X, Y = np.meshgrid(x, y, indexing="ij")

    # simple plate mode
    #from Joanna´s model
    c = material.wave_speed()

    field = np.sin(2*X) * np.sin(3*Y) * (c/1000)

    field[~plate] = np.nan

    return field

def compute_plate_modes(plate, material, n_modes=6):

    Nx, Ny = plate.shape

    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    h = 0.003
    nu = 0.3

    rho = material.rho
    E = material.E

    D = E*h**3/(12*(1-nu**2))

    Lx = 0.35
    Ly = 0.25

    modes = []
    freqs = []

    for m in range(1, n_modes+1):
        for n in range(1, n_modes+1):

            phi = np.sin(m*np.pi*X) * np.sin(n*np.pi*Y)

            omega = (np.pi**2)*np.sqrt(D/(rho*h))*((m**2/Lx**2)+(n**2/Ly**2))

            phi[~plate] = np.nan

            modes.append(phi)
            freqs.append(omega)

    return np.array(modes), np.array(freqs)

def acoustic_field(field):

    pressure = np.abs(field)**2

    return pressure

def modal_response(modes, freqs, material):

    frequencies = np.linspace(50,2000,1000)
    omega = 2*np.pi*frequencies

    gamma = (1-material.damping)*50

    field = np.zeros((len(frequencies),)+modes[0].shape,dtype=complex)

    for k,phi in enumerate(modes):

        w_m = freqs[k]

        response = 1/(w_m**2 - omega[:,None,None]**2 + 1j*gamma*omega[:,None,None])

        field += response*phi

    return frequencies, field

def air_radiation(field):

    rho_air = 1.2

    pressure = np.abs(field)**2

    return pressure

#find results for each material

results = {}

for mat in materials:

    modes, freqs = compute_plate_modes(plate, mat, n_modes=5)

    f, field = modal_response(modes, freqs, mat)

    pressure = np.nanmean(air_radiation(field),axis=0)

    results[mat.name] = {"pressure":pressure}

for name, data in results.items():

    plt.figure(figsize=(6,6))
    plt.title(name + " Radiated Sound Pressure")

    plt.imshow(data["pressure"], cmap="inferno")
    plt.colorbar()

    plt.axis("off")
    plt.show()

fig, axes = plt.subplots(2,2, figsize=(10,10))

axes = axes.flatten()

for ax,(name,data) in zip(axes, results.items()):

    im = ax.imshow(data["pressure"], cmap="inferno")

    ax.set_title(name)
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)

plt.suptitle("Material Comparison of Radiated Sound Pressure")

plt.show()


FILE = "ukulele_optimal.png"

plate = load_plate(FILE)

plate_mass = rho_plate(0,0) * plate.astype(float)


results = {}

for mat in materials:

    modes, freqs = compute_plate_modes(plate, mat, n_modes=5)

    f, field = modal_response(modes, freqs, mat)

    pressure = np.nanmean(air_radiation(field),axis=0)

    results[mat.name] = {"pressure":pressure}

for name, data in results.items():

    plt.figure(figsize=(6,6))
    plt.title(name + " Radiated Sound Pressure")

    plt.imshow(data["pressure"], cmap="inferno")
    plt.colorbar()

    plt.axis("off")
    plt.show()

fig, axes = plt.subplots(2,2, figsize=(10,10))

axes = axes.flatten()

for ax,(name,data) in zip(axes, results.items()):

    im = ax.imshow(data["pressure"], cmap="inferno")

    ax.set_title(name)
    ax.axis("off")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)

plt.suptitle("Material Comparison of Radiated Sound Pressure")

plt.show()