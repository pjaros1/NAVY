import numpy as np
import matplotlib.pyplot as plt

iterationCount = 100

x_min, x_max = -2, 2
y_min, y_max = -2, 2

resolution = 1000
zoom_factor = 0.5

# vypocet noveho rozsahu
x_range = x_max - x_min
y_range = y_max - y_min
x_min += x_range * (1 - zoom_factor) / 2
x_max -= x_range * (1 - zoom_factor) / 2
y_min += y_range * (1 - zoom_factor) / 2
y_max -= y_range * (1 - zoom_factor) / 2

# vytvoreni mrizky
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
c = x[:,np.newaxis] + y[np.newaxis,:]*1j
z = np.zeros_like(c)

for j in range(iterationCount):
    # vypocitame novou hodnotu z 
    z = z**2 + c
    # nalezeni bodu, ktere unikly z rozsahu
    mask = (np.abs(z) > 2) & (z != 0)
    # nastaveni casu uniku
    escape_time = np.zeros_like(z, dtype=float)
    escape_time[mask] = j
    # nastavim unikle body na hranicni rozsah, aby zustaly jako soucast mnoziny
    z[mask] = 2

# normalizace nevalidnich dat
escape_time = np.nan_to_num(escape_time, nan=np.nan, posinf=np.nan, neginf=np.nan, copy=False)

plt.imshow(escape_time, cmap='inferno', extent=(x_min, x_max, y_min, y_max))
plt.axis('off')
plt.show()