#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:49:22 2024

@author: alejandrosoto
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import scienceplots
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scienceplots
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.style.use(['science', 'no-latex', 'bright'])

# Definir temperaturas de fusión y colores
melting_temperatures_1 = {
    "10": 60.69,
    "30": 63.57,
    "50": 63.89,
    "70": 64.41,
    "75": 64.77,
    "77": 67.41,
    "90": 67.90,
    "110": 68.86
}
melting_temperatures_2 = {
    "L-4":50,
    "L0": 53.71,
    "L1": 53.95,
    "L3": 54.43,
    "L4": 58.64,
    "L7": 58.76,
    "L11": 59.00
}

colors_1 = ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027']
colors_2 = [
    "#4575b4",
    "#91bfdb",
    "#e0f3f8",
    "#ffffbf",
    "#fee090",
    "#fc8d59",
    "#d73027"
]




# Crear una figura con una cuadrícula 2x2
fig, axs = plt.subplots(2, 2, figsize=(9, 7.8))

plt.subplots_adjust(left=0.07, right=0.9, top=1.0, bottom=0.05, wspace=0.15, hspace=0.18)

# Funciones auxiliares
def find_melting_temperature(df):
    index_05 = np.argmax(df['y_fit_avg'].values >= 0.5)
    melting_temperature = df.loc[index_05, 'x_fit_1']
    return melting_temperature

def find_error_tm(df, melting_temperature):
    index_05_max = np.argmax(df['y_fit_min'].values >= 0.5)
    temperature_05_max = df.loc[index_05_max, 'x_fit_1']
    error_tm = temperature_05_max - melting_temperature
    return error_tm

def generate_max_min_confidence_intervals(df):
    df['y_fit_max'] = df['y_fit_avg'] + df['confidence_interval']
    df['y_fit_min'] = df['y_fit_avg'] - df['confidence_interval']

# Función para graficar el primer set de datos
def plot_first_graph(axs):
    files = ["melting_curves_tol_e5_10_10.csv", "melting_curves_tol_e5_10_70.csv", "melting_curves_tol_e5_10_50.csv", "melting_curves_tol_e5_10_30.csv", "melting_curves_tol_e5_10_77.csv", "melting_curves_tol_e5_10_90.csv", "melting_curves_tol_e5_10_75.csv", "melting_curves_tol_e5_10_110.csv"]
    labels = ["10", "30", "50", "70", "75", "77", "90", "110"]
    ignore_columns = ["y_fit_1"]

    concentrations = ["1.30", "3.17", "5.34", "7.55", "8.15", "8.35", "9.51", "11.85"]
    
    for i, (file_name, label) in enumerate(zip(files, labels)):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df = pd.read_csv(file_path)

        if ignore_columns:
            df = df.drop(columns=ignore_columns)

        num_values = len(df.columns) - 1
        y_fit_columns = [col for col in df.columns if col.startswith("y_fit")]

        df['y_fit_avg'] = df[y_fit_columns].mean(axis=1)
        df['y_fit_std'] = df[y_fit_columns].std(axis=1)
        df['error'] = df['y_fit_std'] / np.sqrt(num_values)

        degrees_of_freedom = num_values - 1
        t_critical = t.ppf(0.975, degrees_of_freedom)

        df['confidence_interval'] = df['error']

        generate_max_min_confidence_intervals(df)

        color = colors_1[i % len(colors_1)]  # Asignar color de la lista
        melting_temperature = find_melting_temperature(df)
        axs[0, 0].plot(df["x_fit_1"], df["y_fit_avg"], label=f'{concentrations[i]} $\mu M$', alpha=0.7, color=color)
        axs[0, 0].fill_between(df["x_fit_1"], df["y_fit_min"], df["y_fit_max"], alpha=0.2, color=color)

        error_tm = find_error_tm(df, melting_temperature)
        axs[1, 0].errorbar(float(concentrations[i]), melting_temperatures_1[label], yerr=error_tm, fmt='o', color=color, capsize=5,  markersize=7, markeredgecolor='black')
        axs[1, 0].text(float(concentrations[i])-0.45, melting_temperatures_1[label]-1.5, f'{melting_temperature:.1f}', fontsize=9, verticalalignment='center', horizontalalignment='left')

    axs[0, 0].set_xlabel('Temperature ($^{o}$C)', fontsize=12)
    axs[0, 0].set_ylabel('Fraction unbounded', fontsize=12)
    axs[0, 0].set_xticks(np.arange(40, 110, 10))
    axs[0, 0].tick_params(axis='both', which='major', labelsize=10)
    axs[0, 0].legend(fontsize=10)
    axs[0, 0].text(80, 0.87, 'L12 duplex core',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotación en la segunda línea
    axs[0, 0].text(80, 0.82, '100 mM NaCl',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

    # Agregar zoom
    axins = axs[0, 0].inset_axes([0.1, 0.7, 0.25, 0.25])
    for i, (file_name, label) in enumerate(zip(files, labels)):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df_zoom = pd.read_csv(file_path)

        if ignore_columns:
            df_zoom = df_zoom.drop(columns=ignore_columns)

        num_values = len(df_zoom.columns) - 1
        y_fit_columns = [col for col in df_zoom.columns if col.startswith("y_fit")]

        df_zoom['y_fit_avg'] = df_zoom[y_fit_columns].mean(axis=1)

        color = colors_1[i % len(colors_1)]
        axins.plot(df_zoom["x_fit_1"], df_zoom["y_fit_avg"], label=f'{concentrations[i]} $\mu M$', alpha=0.7, color=color)

    axins.set_xlim(60, 70)
    axins.set_xticks(np.arange(60, 75, 5))  # Controlar los ticks en el eje X del zoom
    axins.set_ylim(0.4, 0.6)
    axins.set_yticks(np.arange(0.4, 0.7, 0.1))  # Controlar los ticks en el eje Y del zoom
    
    axins.tick_params(axis='both', which='major', labelsize=10)
    axins.grid(True)
    axs[0, 0].indicate_inset_zoom(axins)
    
    # Configurar el subplot scatter
    axs[1, 0].set_xlabel('Concentration ($\mu M$)', fontsize=12)
    axs[1, 0].set_ylabel('Melting temperature ($^{o}$C)', fontsize=12)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=10)
    axs[1, 0].set_yticks(np.arange(48, 74, 2))
    axs[1, 0].set_ylim(48, 72)
    axs[1, 0].set_xticks(np.arange(0, 13, 1))
    
    axs[1, 0].text(0.3, 71, 'L12 duplex core',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotación en la segunda línea
    axs[1, 0].text(0.3, 70, '100 mM NaCl',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
# Crear un mapa de colores suavizado
    
    concentrations = [1.3, 3.17, 5.34, 7.55, 8.15, 8.35, 9.51, 11.85]
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors_1)

# Crear un normalizador para la barra de colores, basado en los valores reales
    norm = mcolors.Normalize(vmin=min(concentrations), vmax=max(concentrations))

# Crear el ScalarMappable para la barra de colores
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

# Ajustar el tamaño y la posición de la barra de colores
    cbar_ax = fig.add_axes([0.1, 0.13, 0.33, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

# Ajustar los ticks, eliminando el cuarto y sexto valor
    ticks = [float(tick) for i, tick in enumerate(concentrations) if i not in [3, 5]]
    tick_labels = [f'{val:.1f}' for val in ticks]

# Configurar los ticks y etiquetas
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Concentration('"$\mu$"'M)')
    
    # Define your concentrations


    
    #cmap = LinearSegmentedColormap.from_list('custom_cmap', colors_1, N=len(colors_1))
    #norm = Normalize(vmin=min(concentrations), vmax=max(concentrations))
    #cbar = ColorbarBase(axs[1,0].inset_axes([0.05, 0.15, 0.7, 0.02]), cmap=cmap, norm=norm, orientation='horizontal')
    #cbar.set_label('Concentration $(^{o}C)$', fontsize=10)
    
    
    
def plot_second_graph(axs):
    files = ["melting_curves_tol_e5_10_L-4.csv","melting_curves_tol_e5_10_L1.csv", "melting_curves_tol_e5_10_L0.csv", "melting_curves_tol_e5_10_L3.csv", "melting_curves_tol_e5_10_L7.csv", "melting_curves_tol_e5_10_L4.csv", "melting_curves_tol_e5_10_L11.csv"]
    labels = ["L-4","L0", "L1", "L3", "L4", "L7", "L11"]
    ignore_columns = ["y_fit_1"]

    leng = ["16","12", "11", "9", "8", "5", "1"]

    for i, (file_name, label) in enumerate(zip(files, labels)):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df = pd.read_csv(file_path)

        if ignore_columns:
            df = df.drop(columns=ignore_columns)

        num_values = len(df.columns) - 1
        y_fit_columns = [col for col in df.columns if col.startswith("y_fit")]

        df['y_fit_avg'] = df[y_fit_columns].mean(axis=1)
        df['y_fit_std'] = df[y_fit_columns].std(axis=1)
        df['error'] = df['y_fit_std'] / np.sqrt(num_values)

        degrees_of_freedom = num_values - 1
        t_critical = t.ppf(0.975, degrees_of_freedom)

        df['confidence_interval'] = df['error']

        generate_max_min_confidence_intervals(df)

        color = colors_2[i % len(colors_2)]  # Usar colors_2 para la gráfica principal
        melting_temperature = find_melting_temperature(df)
        axs[0, 1].plot(df["x_fit_1"], df["y_fit_avg"], label=f'{label} ', alpha=0.7, linewidth=2, color=color)
        axs[0, 1].fill_between(df["x_fit_1"], df["y_fit_min"], df["y_fit_max"], alpha=0.2, color=color)
        

        error_tm = find_error_tm(df, melting_temperature)
        axs[1, 1].errorbar(float(leng[i]), melting_temperatures_2[label], yerr=error_tm, fmt='o', color=color, capsize=5, markersize=7, markeredgecolor='black')
        axs[1, 1].text(float(leng[i])-0.5, melting_temperatures_2[label]+0.8, f'{melting_temperature:.1f}', fontsize=9, verticalalignment='center', horizontalalignment='left')

    axs[0, 1].set_xlabel('Temperature ($^{o}$C)', fontsize=12)
    axs[0, 1].set_ylabel('Fraction unbounded', fontsize=12)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
    axs[0, 1].set_xticks(np.arange(40, 110, 10))
    axs[0, 1].set_xlim(38,102)
    axs[0, 1].legend(fontsize=10, loc='lower right')
# Anotación en la primera línea
    axs[0, 1].text(39.8, 1.02, '100 mM NaCl',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotación en la segunda línea
    axs[0, 1].text(39.8, 0.98, '1 $\mu$M',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))






    # Agregar zoom
    axins = axs[0, 1].inset_axes([0.45, 0.4, 0.25, 0.25])
    for i, (file_name, label) in enumerate(zip(files, labels)):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df_zoom = pd.read_csv(file_path)

        if ignore_columns:
            df_zoom = df_zoom.drop(columns=ignore_columns)

        num_values = len(df_zoom.columns) - 1
        y_fit_columns = [col for col in df_zoom.columns if col.startswith("y_fit")]

        df_zoom['y_fit_avg'] = df_zoom[y_fit_columns].mean(axis=1)

        color = colors_2[i % len(colors_2)]  # Usar colors_2 para el zoom
        axins.plot(df_zoom["x_fit_1"], df_zoom["y_fit_avg"], alpha=0.7, color=color)

    axins.set_xlim(48, 60)
    axins.set_xticks(np.arange(48, 66, 6))
    axins.set_ylim(0.4, 0.6)
    axins.set_yticks(np.arange(0.4, 0.7, 0.1))
    
    axins.tick_params(axis='both', which='major', labelsize=10)
    axins.grid(True)
    axs[0, 1].indicate_inset_zoom(axins)
    
    # Configurar el subplot scatter
    axs[1, 1].set_xlabel('Length', fontsize=12)
    axs[1, 1].set_ylabel('Melting temperature ($^{o}$C)', fontsize=12)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=10)
    axs[1, 1].set_yticks(np.arange(48, 74, 2))
    axs[1, 1].set_xticks(np.arange(0, 17, 1))
    axs[1, 1].set_ylim(48, 72)
    axs[1, 1].set_xlim(0, 16.5)
    
    axs[1, 1].text(0.3, 71, '100 mM NaCl',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotación en la segunda línea
    axs[1, 1].text(0.3, 70, '1 $\mu$M',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
    # Ocultar título y ticks del eje y en (0, 1) y (1, 1)
    axs[0, 1].set_ylabel('')  # Ocultar el título del eje y
    axs[0, 1].yaxis.set_visible(False)  # Ocultar los ticks del eje y
    
# Añadir anotación antes de ocultar ejes

    axs[1, 1].set_ylabel('')  # Ocultar el título del eje y
    axs[1, 1].yaxis.set_visible(False)  # Ocultar los ticks del eje y
 # Asegúrate de que 'labels' sea una lista o conviértelo a una lista
    labels_list = labels.index.tolist() if isinstance(labels, pd.Index) else labels

# Crear una barra de colores horizontal dentro del segundo subplot
    cmap = mcolors.ListedColormap(colors_2)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(labels_list) + 1) - 0.5, ncolors=len(labels_list))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

# Ajustar el tamaño y la posición de la barra de colores
    cbar_ax = fig.add_axes([0.523, 0.395, 0.33, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=np.arange(len(labels_list)))

# Colocar una "L" delante de cada etiqueta de los ticks
    cbar.ax.set_xticklabels([str(label) for label in labels_list])
# Anotaciones y posiciones
annotations = ['a)', 'b)', 'c)', 'd)']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
text_positions = [(-0.1, 1.05), (-0.05, 1.05), (-0.1, 1.05), (-0.05, 1.05)]  # Ajusta estas coordenadas según sea necesario

# Añadir anotaciones a los subplots
for pos, annotation, text_pos in zip(positions, annotations, text_positions):
    axs[pos].text(text_pos[0], text_pos[1], annotation, transform=axs[pos].transAxes, 
                  fontsize=12, fontweight='bold', va='top', ha='right')
    

plot_first_graph(axs)
plot_second_graph(axs)
plt.savefig('combine_exp.png', dpi=300)

plt.show()
