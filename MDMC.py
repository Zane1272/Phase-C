#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

m_p = 0.05      # kg, top plate mass
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
rho = 1.2      # kg/m^3, air density
R_dist = 1.0   # m, distance to microphone
U = A * u_p + S * u_a
p_sound = -1j * rho * omega * U / (4 * np.pi * R_dist)

# Plot velocity magnitude of top plate
plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(u_p), label='Top plate velocity |u_p|')
plt.plot(frequencies, np.abs(u_a), label='Air piston velocity |u_a|')
plt.plot(frequencies, np.abs(p_sound), label='Sound pressure |p|')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Coupled Guitar Top Plate & Helmholtz Resonance')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




