#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:14:46 2024

@author: alejandrosoto
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d

plt.style.use(['science', 'no-latex', 'bright'])

# Ruta base donde se encuentran los archivos, incluyendo Y+L0 y Y+L11
base_paths = {
    "L_-4": "./L_-4/hb",
    "L_0": "./L_0/hb",
    "L_1": "./L_1/hb",
    "L_2": "./L_2/hb",
    "L_3": "./L_3/hb",
    "L_4": "./L_4/hb",
    "L_7": "./L_7/hb",
    "L_11": "./L_11/hb",
    "Y": "./Y_0/hb",
    "Y+L0": "./Y+L0/hb",  # Nueva carpeta Y+L0
    "Y+L11": "./Y+L11/hb"  # Nueva carpeta Y+L11
}


temps = [14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92]
norm = 25 * (28 + 14)  # Normalización

def shuffled_sterr(_in):
    _chunk = 20
    N = _in.size // _chunk
    if N == 0:
        return 0
    out = 0
    for i in range(N):
        _ids = np.random.randint(0, high=_in.size, size=_chunk)
        out += np.std(_in[_ids])
    return np.sqrt(out / N) / np.sqrt(N)

def extrapolate_bulk_array(x):
    temp = x / (1 - x)
    return (1 + 1 / (2 * temp)) - np.sqrt((1 + 1. / (2 * temp)) ** 2 - 1)

def find_temp_for_fraction(temps, fractions, target_fraction=0.5):
    """Interpolar linealmente para encontrar la temperatura en la cual la fracción alcanza un valor dado."""
    f = interp1d(fractions, temps, kind='linear', fill_value="extrapolate")
    return f(target_fraction)

# Configurar estilo y fuentes
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8,
})

# Tamaño de los marcadores
marker_size = 10

# Colores especificados
colors = [
"#313695", "#416aaf", "#74add1", "#c1e4ef", "#e9f6e8", "#fee99d", "#fdb567", "#f46d43", "#db382b", "#c6171b", "#7f0d0b"
][::-1]

# Crear la figura con dos subplots en horizontal
# Crear la figura con constrained_layout para evitar problemas con tight_layout
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 4))


# Lista para almacenar los valores de temperatura a fracción 0.5
temp_at_05_values = []
temp_at_05_errors = []
labels = []
norm = None

"""

# Leer datos y graficar para cada carpeta
for i, (label, base_path) in enumerate(base_paths.items()):
    ave_data = []
    std_data = []

    for temp in temps:
        hb_file = os.path.join(base_path, f"hb_list.{temp}.dat")
        
        if os.path.exists(hb_file):
            data = np.loadtxt(hb_file)
            store_data = data[-data.size // 10:] / norm
            ave_data.append(np.average(store_data))
            std_data.append(shuffled_sterr(store_data))
        else:
            ave_data.append(np.nan)
            std_data.append(np.nan)

    ave_data = np.array(ave_data)
    std_data = np.array(std_data)
    extr = extrapolate_bulk_array(ave_data)

    ave_data = np.nan_to_num(ave_data, nan=0)
    std_data = np.nan_to_num(std_data, nan=0)
    extr = np.nan_to_num(extr, nan=0)

    color = colors[i % len(colors)]  # Usar colores en bucle

    # Reemplazar el guion bajo en la leyenda
    clean_label = label.replace("_", "")
    
    # Graficar con líneas que conectan los puntos
    ax.errorbar(temps, extr, yerr=std_data, 
            fmt='o-',  # Conectar puntos con líneas rectas
            color=color,  # Color de los datos
            markeredgecolor='k',  # Color del borde del marcador
            capsize=5,  # Tamaño de las "caps" en los errores
            label=clean_label)

    # Interpolar para encontrar la temperatura a la cual la fracción alcanza 0.5
    temp_at_05 = find_temp_for_fraction(temps, extr, target_fraction=0.5)
    temp_at_05_values.append(temp_at_05)
    temp_at_05_errors.append(np.std(std_data))  # Asumir std_data como error (puedes ajustar esto)
    labels.append(label)  # Mantener las etiquetas originales
    #print(f"Para {clean_label}, la temperatura para fracción 0.5 es aproximadamente: {temp_at_05:.2f} °C")
"""

# Leer datos y graficar para cada carpeta
for i, (label, base_path) in enumerate(base_paths.items()):
    ave_data = []
    std_data = []
    # Leer el archivo para la temperatura 14 y encontrar el valor máximo
    hb_file_14 = os.path.join(base_path, "hb_list.14.dat")
    if os.path.exists(hb_file_14):
        data_14 = np.loadtxt(hb_file_14)
        norm = np.max(data_14)  # Asignar norm como el máximo del primer archivo
    else:
        print(f"Advertencia: El archivo {hb_file_14} no existe.")

    for temp in temps:
        hb_file = os.path.join(base_path, f"hb_list.{temp}.dat")
        
        if os.path.exists(hb_file):
            # Leer el archivo .dat
            data = np.loadtxt(hb_file)
            
            # Convertir los datos a valores numéricos
            data = np.nan_to_num(data, nan=0.0)
            
            # Seleccionar la última décima parte de los datos para calcular los promedios
            store_data = data[-data.size // 10:] / norm if norm != 0 else data[-data.size // 10:]
            
            # Calcular promedio y desviación estándar
            ave_data.append(np.average(store_data))
            std_data.append(shuffled_sterr(store_data))
        else:
            ave_data.append(np.nan)
            std_data.append(np.nan)

    # Asegurar que los datos sean numéricos y manejar NaNs
    ave_data = np.array(ave_data)
    std_data = np.array(std_data)
    
    # Convertir cualquier valor no numérico o NaN a 0
    ave_data = np.nan_to_num(ave_data, nan=0.0)
    std_data = np.nan_to_num(std_data, nan=0.0)

    # Extrapolar y graficar los resultados
    extr = extrapolate_bulk_array(ave_data)

    color = colors[i % len(colors)]  # Usar colores en bucle

    # Reemplazar el guion bajo en la leyenda
    clean_label = label.replace("_", "")
    
    # Graficar con líneas que conectan los puntos
    ax.errorbar(temps, extr, yerr=std_data, 
                fmt='o-',  # Conectar puntos con líneas rectas
                color=color,  # Color de los datos
                markeredgecolor='k',  # Color del borde del marcador
                capsize=5,  # Tamaño de las "caps" en los errores
                label=clean_label)

    # Interpolar para encontrar la temperatura a la cual la fracción alcanza 0.5
    temp_at_05 = find_temp_for_fraction(temps, extr, target_fraction=0.5)
    temp_at_05_values.append(temp_at_05)
    temp_at_05_errors.append(np.std(std_data))  # Asumir std_data como error (puedes ajustar esto)
    labels.append(label)  # Mantener las etiquetas originales

ax.set_ylabel("Fraction unbounded", fontsize=10)
ax.set_xlabel("Temperature ($^{o}$C)", fontsize=10)
ax.set_ylim(-0.1, 1.1)
ax.set_xticks(np.arange(10, 110, 10))
ax.set_yticks(np.arange(0, 1.2, 0.2))
ax.text(-0.04, 1.07, 'a)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax.set_xlim(10, 100)
ax.hlines(0.5, temps[0], temps[-1], color='black')
ax.legend(loc='lower left')


# Crear un nuevo eje para el zoom
axins1 = fig.add_axes([0.38, 0.7, 0.06, 0.15])  # [left, bottom, width, height]

# Repetir el código de graficado para el nuevo eje
for i, (label, base_path) in enumerate(base_paths.items()):
    ave_data = []
    for temp in temps:
        hb_file = os.path.join(base_path, f"hb_list.{temp}.dat")
        
        if os.path.exists(hb_file):
            data = np.loadtxt(hb_file)
            if data.size > 0:
                store_data = data[-data.size // 10:] / norm if norm != 0 else data[-data.size // 10:]
                ave_data.append(np.average(store_data))
            else:
                ave_data.append(np.nan)
        else:
            ave_data.append(np.nan)

    ave_data = np.array(ave_data)
    ave_data = np.nan_to_num(ave_data, nan=0)

    color = colors[i % len(colors)]
    axins1.plot(temps, ave_data, '-', color=color)

# Ajustar límites del eje de zoom
axins1.set_xlim(55, 75)
axins1.set_ylim(0.4, 0.6)

# Opcionalmente, agregar ticks y una cuadrícula
axins1.set_xticks(np.arange(55, 85, 10))
axins1.set_yticks(np.linspace(0.4, 0.6, num=3))
ax.indicate_inset_zoom(axins1) 
axins1.grid(True)





# Valores personalizados para cada molécula
custom_values = {
    "L_-4": 16,
    "L_0": 12,
    "L_1": 11,
    "L_2": 10,
    "L_3": 9,
    "L_4": 8,
    "L_7": 5,
    "L_11": 1,
    "Y": 12,
    "Y+L11": 11,
    "Y+L0": 0# Esto será tratado de manera especial
}

# Subplot de dispersión para temperaturas a fracción 0.5
for i, (label, color) in enumerate(zip(labels, colors)):
    ax2.errorbar(custom_values[label], temp_at_05_values[i], yerr=temp_at_05_errors[i], 
                 fmt='o', color=color, 
                 markeredgecolor='k', capsize=5)

ax2.set_ylabel("$T_{m}$ ($^{o}$C)", fontsize=10)
ax2.set_xlabel("Length of the sticky end (unpaired base)", fontsize=10)
ax2.set_xticks(list(custom_values.values()))  # Establecer ticks en los valores personalizados
#ax2.set_xticklabels([label.replace("_", "") for label in labels], rotation=45, ha='right')  # Etiquetas sin guiones bajos
ax2.set_ylim(min(temp_at_05_values) - 5, max(temp_at_05_values) + 5)
ax2.text(-0.04, 1.07, 'b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
plt.savefig('all.png', dpi=300)
#ax2.grid(True)

# Ajustar el layout para que no se superpongan los subplots
# Ajustar los márgenes manualmente si es necesario
plt.subplots_adjust(wspace=0.4, left=0.1, right=0.95, bottom=0.15, top=0.9)  # Ajuste manual de los márgenes


# Guardar la imagen en un archivo PNG con alta resolución


# Mostrar la gráfica
plt.show()