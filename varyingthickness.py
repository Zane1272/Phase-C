#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation, label
import torch.nn.functional as F

from main import GeometryProcessor, Material

materials = [
    Material("Wood", 600, 1.0e10, 0.995),
    Material("PLA", 1250, 3.0e9, 0.993),
    Material("ABS", 1050, 2.0e9, 0.992),
    Material("Composite", 900, 5.0e9, 0.994)
] #wood definition from main(1).py

mask = GeometryProcessor.process('ukulele_top.png')
frequencies = [440]

def rho_plate(x,y,mask): #to be ammended according to Joanna´s findings
    Nx, Ny = mask.shape
    rho_map = np.zeros_like(mask, dtype=float)
    E_map = np.zeros_like(mask, dtype=float)
    damping_map = np.zeros_like(mask, dtype=float)

    for x in range(Nx):
        for y in range(Ny):
            if mask[x,y]:
                # Example: choose material per region
                rho_map[x,y] = 600         # Wood density
                E_map[x,y] = 1.0e10        # Wood Young's modulus
                damping_map[x,y] = 0.995   # Wood damping
            else:
                rho_map[x,y] = 1e-3  # negligible mass outside plate
                E_map[x,y] = 1e3
                damping_map[x,y] = 1.0

    return rho_map, E_map, damping_map
    


def mass_function(image,rho):
    'where rho must map to the side of the plate mask'
    
    #using the same mapping as top plate vibration for eaiser later integration
    img = Image.open("ukulele_top.png").resize((200, 200))
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

    #from guitartop.py
    rows_with_plate = np.any(plate, axis=1)
    plate_row_indices = np.where(rows_with_plate)[0]

    top_row = plate_row_indices[0]
    bottom_row = plate_row_indices[-1]
    plate_pixel_height = bottom_row - top_row + 1


    Nx, Ny = plate.shape

    plate_mass = np.zeros_like(plate, dtype=float)

    for x in range(Nx):
        for y in range(Ny):
            if plate[x,y]==True:
                plate_mass[x,y] = rho(x,y)
        
    return plate_mass, plate_pixel_height



m_p,plate_pixel_height = mass_function('guitar_top.png', lambda x,y: rho_plate(x,y, mask)[0])
m_p[m_p == 0] = np.nan
   # we want this to be a function 
m_a = 0.01      # kg, air piston mass
k_p = 1000.0    # N/m, top plate stiffness
R_p = 0.5       # kg/s, damping top plate
R_a = 0.05      # kg/s, damping air piston
A = 0.02        # want this as the same as above
S = 0.01        # m^2, air piston area
F0 = 1.0        # N, harmonic force amplitude


# Assuming you have a binary mask of the plate
# and the materials dictionary/list

#adapt to accept material properties format

Nx, Ny = mask.shape
rho_map = np.zeros_like(mask, dtype=float)
E_map = np.zeros_like(mask, dtype=float)
damping_map = np.zeros_like(mask, dtype=float)

# Assign material properties per pixel
for i in range(Nx):
    for j in range(Ny):
        if mask[i, j]:
            # Example: assign different materials in regions if needed
            rho_map[i, j] = 600         # Wood density
            E_map[i, j] = 1.0e10       # Wood Young's modulus
            damping_map[i, j] = 0.995  # Wood damping
        else:
            rho_map[i, j] = 1e-3       # negligible mass outside plate
            E_map[i, j] = 1e3
            damping_map[i, j] = 1.0

# Local wave speed per pixel
#from main(1).py
c_map = np.sqrt(E_map / rho_map)  


# Derived frequencies
omega_p = np.sqrt(E_map / rho_map)   #change to per pixel    # rad/s, natural freq plate
omega_a = 200.0                     # rad/s, Helmholtz frequency (example)
a_coupling = 500.0                  # coupling constant
omega_c2 = a_coupling / np.sqrt(m_p * m_a)  # coupling frequency squared

 # Hz
omega = 2 * np.pi * 440        # rad/s
omega3 = omega #make sure shapes match
# Damping
gamma_p =  R_p / rho_map *damping_map #change to per pixel
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

idx = np.argmin(np.abs(frequencies - 440))

#find total speed with material properties of wave propagation

u_total = u_p +  u_a

results_per_material = {}

for mat in materials:
    # Assign material properties
    rho_map[mask==1] = mat.rho
    E_map[mask==1] = mat.E
    damping_map[mask==1] = mat.damping

    # Per-pixel natural frequency and damping
    omega_p_map = np.sqrt(E_map / rho_map)
    gamma_p_map = R_p / rho_map * damping_map # or damping_map if you want material damping

    # Frequency-domain response (per-pixel)
    D_map = (omega_p_map**2 - omega**2 + 1j*gamma_p_map*omega) * \
            (omega_a**2 - omega**2 + 1j*gamma_a*omega) - omega_c2**2

    dx = physical_width_m / mask.shape[1]
    dy = physical_height_m / mask.shape[0]
    m_pixel = rho_map * dx * dy

    u_p_map = 1j * omega * (F0 / m_pixel) * (omega_a**2 - omega**2 + 1j*gamma_a*omega) / D_map
    u_a_map = -1j * omega * (F0 / m_pixel) * (A/S) * (omega_p_map**2 - omega**2 + 1j*gamma_p_map*omega) / D_map
    # Total top plate + air piston velocity
    u_total_map = u_p_map + mask * u_a_map

    results_per_material[mat.name] = u_total_map

plt.figure(figsize=(6,6))
plt.imshow(np.abs(u_p), cmap='viridis', origin='upper', extent=extent, aspect='equal')
plt.colorbar(label="Top plate velocity (magnitude)")
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(np.abs(u_a), cmap='viridis', origin='upper', extent=extent, aspect='equal')
plt.colorbar(label="Air piston velocity (magnitude)")
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(np.abs(p_sound), cmap='cividis', origin='upper', extent=extent, aspect='equal')
plt.colorbar(label="Sound pressure (magnitude)")
plt.show()

fig, axes = plt.subplots(1, len(materials), figsize=(18,6))

for ax, (name, u_total) in zip(axes, results_per_material.items()):
    im = ax.imshow(np.abs(u_total), cmap='viridis', origin='upper', extent=extent, aspect='equal')
    ax.set_title(name)
    ax.axis('off')

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Plate + air piston velocity")
plt.show()