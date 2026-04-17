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


import numpy as np
import matplotlib.pyplot as plt


f_minus = 104.0   # Hz (first resonance)
f_h = 127.0       # Hz (Helmholtz antiresonance)
f_plus = 219.0    # Hz (second resonance)

Q_minus = 29.0
Q_plus = 25.8

m_p = 0.112       # kg (equivalent top plate mass)
V = 0.013         # m³ cavity volume (13 liters)

rho = 1.205       # air density
c = 343.0         # speed of sound
F0 = 0.2          # excitation force (N)
R_dist = 2.0      # microphone distance (m)



w_minus = 2*np.pi*f_minus
w_h = 2*np.pi*f_h
w_plus = 2*np.pi*f_plus



w_p = np.sqrt(w_minus**2 + w_plus**2 - w_h**2)



w_c2 = (w_plus**2 - w_p**2)*(w_p**2 - w_minus**2)



A2_over_mp = (w_h**2 * V) / (rho * c**2)
A = np.sqrt(A2_over_mp * m_p)

# Soundhole area estimate
S = np.pi * (0.044)**2   # ~88 mm diameter hole

# Air piston mass
m_a = rho * S * 0.02     # effective neck length ~2 cm



gamma_minus = w_minus / Q_minus
gamma_plus = w_plus / Q_plus

gamma_p = gamma_plus
gamma_a = gamma_minus



frequencies = np.linspace(60, 300, 1200)
omega = 2*np.pi*frequencies



D = ((w_p**2 - omega**2 + 1j*gamma_p*omega) *
     (w_h**2 - omega**2 + 1j*gamma_a*omega) -
     w_c2)



u_p = 1j*omega*(F0/m_p)*(w_h**2 - omega**2 + 1j*gamma_a*omega)/D



u_a = -1j*omega*(F0/m_p)*(A/S)*(w_p**2)/D



U = A*u_p + S*u_a

p_sound = -1j*rho*omega*U/(4*np.pi*R_dist)



plt.figure(figsize=(10,6))

plt.plot(frequencies,
         20*np.log10(np.abs(u_p/F0)),
         label="Plate mobility (dB re m/Ns)")

plt.plot(frequencies,
         20*np.log10(np.abs(p_sound/2e-5)),
         label="Sound pressure level (dB SPL)")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Level (dB)")
plt.title("Low-frequency guitar response (Christensen–Vistisen model)")
plt.legend()
plt.grid(True)
plt.show()

