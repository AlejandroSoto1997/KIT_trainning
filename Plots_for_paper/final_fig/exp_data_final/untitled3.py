#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:02:50 2024

@author: alejandrosoto
"""
# Ejemplo de cómo agregar anotaciones a un subplot específico en un arreglo de subplots

import matplotlib.pyplot as plt
import numpy as np

# Crear datos de ejemplo
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Crear una figura con una cuadrícula 2x2
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Graficar en axs[1, 1]
axs[1, 1].plot(x, y, label='Sine Wave')

# Agregar anotaciones en axs[1, 1]
axs[1, 1].annotate('Peak', xy=(7, np.sin(7)), xytext=(8, np.sin(7) + 0.5),
                   arrowprops=dict(facecolor='black', shrink=0.05),
                   fontsize=12, color='red')

axs[1, 1].annotate('Trough', xy=(4, np.sin(4)), xytext=(5, np.sin(4) - 0.5),
                   arrowprops=dict(facecolor='blue', shrink=0.05),
                   fontsize=12, color='blue')

axs[1, 1].text(0.2, 0.2, 'notew', 
               fontsize=12, color='color_texto', 
               ha='right', va='top')


# Configurar el subplot
axs[1, 1].set_xlabel('X-axis')
axs[1, 1].set_ylabel('Y-axis')
axs[1, 1].legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()

