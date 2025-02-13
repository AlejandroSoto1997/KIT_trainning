#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:33:26 2024

@author: alejandrosoto
"""

import matplotlib.pyplot as plt

# Definir los colores
colors = [
    "#a50026", "#b30d26", "#c01a27", "#ce2827", "#db382b", "#e34933", "#eb5a3a",
    "#f46d43", "#f67f4b", "#f99153", "#fca55d", "#fdb567", "#fdc374", "#fed283",
    "#fee090", "#fee99d", "#fff2ac", "#fffbb9", "#fbfdc7", "#f2fad6", "#e9f6e8",
    "#e0f3f8", "#d1ecf4", "#c1e4ef", "#b2ddeb", "#a3d3e6", "#92c5de", "#83b9d8",
    "#74add1", "#659bc8", "#588cc0", "#4b7db8", "#416aaf", "#3c59a6", "#36479e",
    "#313695"
]

# Crear una figura
plt.figure(figsize=(10, 10))

# Dibujar un rectángulo para cada color
for i, color in enumerate(colors):
    plt.fill_between([0, 1], i, i+1, color=color)

# Ajustar los límites y eliminar ejes
plt.xlim(0, 1)
plt.ylim(0, len(colors))
plt.axis('off')

# Mostrar el gráfico
plt.show()
