import matplotlib.pyplot as plt
import numpy as np
from portafolio import historia_utilidades, n_gen

# Gráfica 1: Promedio de utilidades por generación
promedios_utilidad = [np.mean(utilidades) for utilidades in historia_utilidades]
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_gen + 1), promedios_utilidad, marker='o', linestyle='-', color='b', label='Promedio de utilidades')
plt.title('Evolución del promedio de utilidades por generación', fontsize=14)
plt.xlabel('Generación', fontsize=12)
plt.ylabel('Promedio de utilidad', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# Gráfica 2: Máxima utilidad por generación
maximos_utilidad = [np.max(utilidades) for utilidades in historia_utilidades]
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_gen + 1), maximos_utilidad, marker='o', linestyle='-', color='g', label='Máxima utilidad')
plt.title('Evolución de la máxima utilidad por generación', fontsize=14)
plt.xlabel('Generación', fontsize=12)
plt.ylabel('Máxima utilidad', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()