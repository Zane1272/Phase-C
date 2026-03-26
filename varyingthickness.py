#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation, label
import torch
import torch.nn.functional as F

def rho(x,y):
    return 1500*x + 1500*y
    


def mass_function(image,rho):
    'where rho must map to the side of the plate mask'
    
    #using the same mapping as top plate vibration for eaiser later integration
    img = Image.open("guitar_top.png").resize((200, 200))
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

    Nx, Ny = plate.shape

    plate_mass = np.tensor(plate.shape)

    for x, y in range(plate.shape):
        if plate.shape[x,y]=True:
            plate_mass[x,y] = rho(x,y)
    
    return plate_mass



m_p = mass_function('guitar_top.png')    # we want this to be a function 
m_a = 0.01      # kg, air piston mass
k_p = 1000.0    # N/m, top plate stiffness
R_p = 0.5       # kg/s, damping top plate
R_a = 0.05      # kg/s, damping air piston
A = 0.02        # m^2, top plate area
S = 0.01        # m^2, air piston area
F0 = 1.0        # N, harmonic force amplitude

# Derived frequencies
omega_p = np.sqrt(k_p / m_p)       # rad/s, natural freq plate
omega_a = 200.0                     # rad/s, Helmholtz frequency (example)
a_coupling = 500.0                  # coupling constant
omega_c2 = a_coupling / np.sqrt(m_p * m_a)  # coupling frequency squared

# Frequency array
frequencies = np.linspace(50, 400, 1000)  # Hz
omega = 2 * np.pi * frequencies           # rad/s

# Damping
gamma_p = R_p / m_p
gamma_a = R_a / m_a

# Frequency response
D = (omega_p**2 - omega**2 + 1j*gamma_p*omega) * (omega_a**2 - omega**2 + 1j*gamma_a*omega) - omega_c2**2

u_p = 1j * omega * (F0 / m_p) * (omega_a**2 - omega**2 + 1j*gamma_a*omega) / D
u_a = -1j * omega * (F0 / m_p) * (A/S) * (omega_p**2 - omega**2 + 1j*gamma_p*omega) / D

# Sound pressure (far field)
#rho = 1.2      #made this into a function
R_dist = 1.0   # m, distance to microphone
U = A * u_p + S * u_a
p_sound = -1j * rho * omega * U / (4 * np.pi * R_dist)

# we want a map of the top plate at 440Hz, where we see the air piston movement at each position

# we want to map total top plate and air piston velocity with plastic properties to match results from paper



