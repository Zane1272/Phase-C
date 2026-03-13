import numpy as np
import matplotlib.pyplot as plt


# f = (c/2pi)(sqrt(A/VxL_eq))
# L_eq = t + 1.7r

# Constants
c = 343          # speed of sound (m/s)
V = 0.013        # guitar cavity volume (m^3)
t = 0.003        # top plate thickness (m)

# Radius range (meters)
r = np.linspace(0.01, 0.06, 200)

# Area from radius
A = np.pi * r**2

# Effective neck length
L_eq = t + 1.7*r

# Helmholtz frequency
f = (c/(2*np.pi)) * np.sqrt(A/(V * L_eq))

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(12,5))


ax[0].plot(r*1000, f)
ax[0].set_xlabel("Sound Hole Radius (mm)")
ax[0].set_ylabel("Helmholtz Frequency (Hz)")
ax[0].set_title("Frequency vs Sound Hole Radius")
ax[0].grid()

ax[1].plot(A*10000, f)
ax[1].set_xlabel("Sound Hole Area (cm²)")
ax[1].set_ylabel("Helmholtz Frequency (Hz)")
ax[1].set_title("Frequency vs Sound Hole Area")
ax[1].grid()

plt.tight_layout()
# Save figure
plt.savefig("soundhole_frequency_plot.png", dpi=300)
plt.show()