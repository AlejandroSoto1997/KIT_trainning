#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:26:01 2024

@author: alejandrosoto
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Colores iniciales del degradado
colors = [
    "#a50026", "#b73b2a", "#ca5c2f", "#dd7e35", "#f49f43", "#fbbd5a", "#f9d97c",
    "#f5e9a0", "#e1f0d3", "#b4d4eb", "#8db3d9", "#5b8ab6", "#313695"
]

# Crear un mapa de colores
cmap = mcolors.LinearSegmentedColormap.from_list("custom_degradado", colors, N=13)

# Generar 13 colores a partir del mapa
generated_colors = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 13)]

# Imprimir los colores generados
print("Colores generados:")
for color in generated_colors:
    print(color)

# Mostrar los colores generados
fig, ax = plt.subplots(figsize=(10, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
for i, color in enumerate(generated_colors):
    ax.add_patch(plt.Rectangle((i / 13, 0), 1 / 13, 1, color=color))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.title("Colores Generados")
plt.show()
