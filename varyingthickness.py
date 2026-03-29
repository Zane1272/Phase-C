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

def rho_plate(x,y): #to be ammended according to Joanna´s findings
    return 1
    


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



m_p,plate_pixel_height = mass_function('guitar_top.png',rho_plate)
m_p[m_p == 0] = np.nan
   # we want this to be a function 
m_a = 0.01      # kg, air piston mass
k_p = 1000.0    # N/m, top plate stiffness
R_p = 0.5       # kg/s, damping top plate
R_a = 0.05      # kg/s, damping air piston
A = 0.02        # want this as the same as above
S = 0.01        # m^2, air piston area
F0 = 1.0        # N, harmonic force amplitude

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

idx = np.argmin(np.abs(frequencies - 440))

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