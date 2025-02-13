import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.colors import LinearSegmentedColormap

def log_format(x, pos):
    return f'$10^{{{int(x)}}}$'

plt.style.use(['science', 'no-latex', 'bright'])

# Directorio base
base_directory = '/Users/alejandrosoto/Documents/KIT/last_task_ale/explicit_server/SALT_0.1'

# Lista de carpetas de temperatura
temp_folders = [
    'TEMP_14', 'TEMP_16', 'TEMP_18', 'TEMP_20', 'TEMP_22', 'TEMP_24', 'TEMP_26', 'TEMP_28', 'TEMP_30', 
    'TEMP_32', 'TEMP_34', 'TEMP_36', 'TEMP_38', 'TEMP_40', 'TEMP_42', 'TEMP_44', 'TEMP_46', 'TEMP_48', 
    'TEMP_50', 'TEMP_52', 'TEMP_54', 'TEMP_56', 'TEMP_58', 'TEMP_60', 'TEMP_62', 'TEMP_64', 'TEMP_66', 
    'TEMP_68', 'TEMP_70', 'TEMP_72', 'TEMP_74', 'TEMP_76', 'TEMP_78', 'TEMP_80', 'TEMP_82', 'TEMP_84', 
    'TEMP_86'
]

# Diccionarios para almacenar las medianas y datos de energía
median_energy_dict = {}
energy_data_dict = {}

# Procesar cada carpeta de temperatura y archivo
for temp_folder in temp_folders:
    run_folder_path = os.path.join(base_directory, temp_folder, 'RUN_0')
    file_path = os.path.join(run_folder_path, 'energy_file_md.txt')

    if os.path.exists(file_path):
        # Leer el archivo de energía con pandas
        df = pd.read_csv(file_path, sep='\s+', names=['Time', 'Energy1', 'Energy2', 'Energy3'])

        # Almacenar los datos de energía para cada temperatura
        energy_data_dict[temp_folder] = df

        # Calcular y almacenar la mediana de la primera columna (Potencial)
        median_energy = df['Energy1'].median()
        median_energy_dict[temp_folder] = median_energy
        print(f'Median Potential Energy for {temp_folder}: {median_energy}')
    else:
        print(f'File not found: {file_path}')

# Crear un DataFrame a partir del diccionario de medianas
median_energy_df = pd.DataFrame(list(median_energy_dict.items()), columns=['Temperature', 'MedianPotentialEnergy'])

# Convertir la columna "Temperature" a números
median_energy_df['Temperature'] = median_energy_df['Temperature'].str.extract('(\d+)').astype(int)

# Ordenar el DataFrame por temperatura
median_energy_df.sort_values('Temperature', inplace=True)

# Lista de colores
colors = [
    "#313695", "#36479e", "#3c59a6", "#416aaf", "#4b7db8", "#588cc0", "#659bc8", "#74add1", "#83b9d8", 
    "#a3d3e6", "#b2ddeb", "#c1e4ef", "#d1ecf4", "#e0f3f8", "#e9f6e8", "#f2fad6", "#fbfdc7", "#fffbb9", 
    "#fee99d", "#fee090", "#fed283", "#fdc374", "#fdb567", "#fca55d", "#f99153", "#f67f4b", "#f46d43", 
    "#eb5a3a", "#db382b", "#ce2827", "#c01a27", "#b30d26", "#d73027", "#c6171b", "#a50f15", "#911a11", 
    "#7f0d0b"
]

# Crear un diccionario que asocia temperaturas con colores
temp_to_color = {int(folder.split('_')[1]): color for folder, color in zip(temp_folders, colors)}

# Configurar la figura con tres subplots (uno arriba y dos abajo) usando GridSpec
fig = plt.figure(figsize=(12, 8), constrained_layout=True)  # Habilitar constrained_layout
grid = plt.GridSpec(3, 2, height_ratios=[1, 0.5, 0.5], width_ratios=[1, 1], hspace=0.6, wspace=0.6)

# Primer subplot (arriba): Energy Profile para una temperatura específica (e.g., TEMP_30)
ax1 = fig.add_subplot(grid[0, :])
specific_temp = 'TEMP_60'
df_specific = energy_data_dict[specific_temp]
df_specific['LogTime'] = np.log10(df_specific['Time'])
df_inverted_specific = df_specific.iloc[::-1].reset_index(drop=True)

# Obtener el color correspondiente para la temperatura específica
specific_temp_value = int(specific_temp.split('_')[1])
color_for_specific_temp = temp_to_color[specific_temp_value]

ax1.plot(df_inverted_specific['LogTime'], df_inverted_specific['Energy1'], color=color_for_specific_temp, label="Potential Energy")
ax1.plot(df_inverted_specific['LogTime'], df_inverted_specific['Energy2'], color="black", label="Kinetic Energy")
ax1.plot(df_inverted_specific['LogTime'], df_inverted_specific['Energy3'], color="lightgrey", label="Total Energy")
ax1.set_xscale('log')
ax1.set_ylabel('Potential energy\n($k_{B}T$)', fontsize=12)
ax1.set_xlabel('Log(Time)', fontsize=12)
ax1.grid(False)
ax1.xaxis.set_major_formatter(NullFormatter())
ax1.xaxis.set_minor_formatter(NullFormatter())
#ticks = np.logspace(np.log10(1), np.log10(10), num=4)
#ax1.set_xticks(ticks)
#ax1.set_xticklabels([f"${int(tick)}$" for tick in ticks])

ax1.xaxis.set_major_formatter(FuncFormatter(log_format))
ax1.set_ylim(bottom=-1.5, top=0.5)
ax1.legend(loc='best', bbox_to_anchor=(0.6, 0.3))
ax1.text(7.03, -0.4, 'L3', fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax1.text(7.03, -0.59, "$1\\times 10^{9}$ steps", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax1.text(7.03, -0.85, f"T={specific_temp_value}" "$ ^{o}C$", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Añadir anotaciones
ax1.text(-0.01, 1.28, 'a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

# Segundo subplot (abajo izquierda): Median Potential Energy vs Temperature
ax2 = fig.add_subplot(grid[1:, 0])
ax2.scatter(median_energy_df['Temperature'], median_energy_df['MedianPotentialEnergy'], 
            marker='o', s=60, color=colors[:len(median_energy_df)], edgecolor='black')
ax2.set_xlabel('Temperature ($ ^{o}C$)', fontsize=12)
ax2.set_ylabel('Median potential energy ($k_{B}T$)', fontsize=12)
ax2.tick_params(axis='both', labelsize=10)
ax2.text(12, -0.36, 'L3', fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax2.text(12, -0.39, "$1\\times 10^{9}$ steps", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax2.text(-0.02, 1.07, 'b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

# Tercer subplot (abajo derecha): Energy Profile in logarithmic scale for all temperatures
ax3 = fig.add_subplot(grid[1:, 1])
for temp_folder, df in energy_data_dict.items():
    df['LogTime'] = np.log10(df['Time'])
    temp_value = int(temp_folder.split('_')[1])
    color = temp_to_color.get(temp_value, 'grey')
    ax3.plot(df['LogTime'], df['Energy1'], color=color, label=f'Temp {temp_value}')
ax3.set_xscale('log')
ax3.set_ylabel('Potential Energy\n($k_{B}T$)', fontsize=12)
ax3.set_xlabel('Log(Time)', fontsize=12)
#ax3.legend(loc='best', bbox_to_anchor=(1.05, 1))
ax3.grid(False)
ax3.xaxis.set_major_formatter(NullFormatter())
ax3.xaxis.set_minor_formatter(NullFormatter())
ax3.set_xticks([1, 10])
ax3.xaxis.set_major_formatter(FuncFormatter(log_format))
ax3.set_ylim(bottom=-1.5, top=0.5)
ax3.text(-0.02, 1.07, 'c)', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
"""
# Añadir un colorbar a la derecha del tercer subplot (dentro del área de subplots)
sm = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list('custom_cmap', colors), norm=plt.Normalize(vmin=min(median_energy_df['Temperature']), vmax=max(median_energy_df['Temperature'])))
sm.set_array([])
cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.6])  # Ajustar la posición del colorbar
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Temperature ($ ^{o}C$)', fontsize=12)
cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_format))
cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
"""

# Mostrar el gráfico
plt.show()
