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
"#313695", "#36479e", "#3c59a6", "#416aaf", "#4b7db8", "#588cc0", "#659bc8", "#74add1", "#83b9d8", "#a3d3e6", "#b2ddeb", "#c1e4ef", "#d1ecf4", "#e0f3f8", "#e9f6e8", "#f2fad6", "#fbfdc7", "#fffbb9", "#fee99d", "#fee090", "#fed283", "#fdc374", "#fdb567", "#fca55d", "#f99153", "#f67f4b", "#f46d43", "#eb5a3a", "#db382b", "#ce2827", "#c01a27", "#b30d26", "#d73027", "#c6171b", "#a50f15", "#911a11", "#7f0d0b"
]

# Crear un diccionario que asocia temperaturas con colores
temp_to_color = {int(folder.split('_')[1]): color for folder, color in zip(temp_folders, colors)}

# Configurar la figura con tres subplots (uno arriba y dos abajo) usando GridSpec
fig = plt.figure(figsize=(9, 8))
grid = plt.GridSpec(3, 2, height_ratios=[1, 0.4, 0.4], hspace=0.6, wspace=0.1)  # Ajusta la proporción de altura


# Primer subplot (arriba): Energy Profile para una temperatura específica (e.g., TEMP_30)
ax1 = fig.add_subplot(grid[0, :])
specific_temp = 'TEMP_60'
df_specific = energy_data_dict[specific_temp]


norm_value = 1  # Reemplaza con el valor de normalización apropiado
last_10_percent = df_specific[-df.shape[0]//1:].reset_index(drop=True)
 # Normalizar los datos de 'Energy1'
normalized_energy = last_10_percent['Energy1'] / norm_value
normalized_energy_1 = last_10_percent['Energy2'] / norm_value
normalized_energy_2 = last_10_percent['Energy3'] / norm_value

 # Crear una nueva columna 'LogTime' si es necesario
if 'LogTime' not in last_10_percent.columns:
    last_10_percent['LogTime'] = last_10_percent['Time']
specific_temp_value = int(specific_temp.split('_')[1])
color_for_specific_temp = temp_to_color[specific_temp_value]
# Graficar usando semilogx
#color = colors[i % len(colors)]  # Asegurarse de que el índice esté dentro del rango de colores
ax1.semilogx(np.arange(0, last_10_percent.shape[0]), normalized_energy, label="Potential energy", color=color_for_specific_temp)
ax1.semilogx(np.arange(0, last_10_percent.shape[0]), normalized_energy_1, label="Kinetic energy", color="black")
ax1.semilogx(np.arange(0, last_10_percent.shape[0]), normalized_energy_2, label="Total energy", color="lightgrey")



"""
df_specific['LogTime'] = df_specific['Time']
#df_specific['LogTime'] = np.log10(df_specific['Time'])
df_inverted_specific = df_specific.iloc[::-1].reset_index(drop=True)

# Obtener el color correspondiente para la temperatura específica
specific_temp_value = int(specific_temp.split('_')[1])
color_for_specific_temp = temp_to_color[specific_temp_value]

ax1.plot(df_inverted_specific['LogTime'], df_inverted_specific['Energy1'], color=color_for_specific_temp, label="Potential Energy")
ax1.plot(df_inverted_specific['LogTime'], df_inverted_specific['Energy2'], color="black", label="Kinetic Energy")
ax1.plot(df_inverted_specific['LogTime'], df_inverted_specific['Energy3'], color="lightgrey", label="Total Energy")
"""
ax1.set_xscale('log')
ax1.set_ylabel('Potential energy\n($k_{B}T$)', fontsize=12)
ax1.set_xlabel('Time (sim units)', fontsize=12)
ax1.set_xlim(10**0,10**5.5)
ax1.grid(False)
#ax1.xaxis.set_major_formatter(NullFormatter())
#ax1.xaxis.set_minor_formatter(NullFormatter())
#ax1.set_xticks([1, 10])
#ax1.xaxis.set_major_formatter(FuncFormatter(log_format))

ax1.set_ylim(bottom=-1.5, top=0.5)
ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
ax1.text(10**4.7, -0.5,"$T=$" f'{specific_temp_value}' "$ ^{o}C$", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax1.text(10**4.7, -0.63, "L3", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax1.text(10**4.7, -0.70, "$1\\times 10^{9} [steps]$", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax1.text(10**4.7, -0.83, "$1\mu$M", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax1.text(10**4.7, -0.96, "100 mM NaCl", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))



# Añadir anotaciones
ax1.text(-0.01, 1.28, 'a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

# Segundo subplot (abajo izquierda): Median Potential Energy vs Temperature
ax2 = fig.add_subplot(grid[1:, 0])
ax2.scatter(median_energy_df['Temperature'], median_energy_df['MedianPotentialEnergy'], 
            marker='o', s=60, color=colors[:len(median_energy_df)], edgecolor='black')
ax2.set_xlabel('Temperature ($ ^{o}C$)', fontsize=12)
ax2.set_ylabel('Average potential energy ($k_{B}T$)\n Potential energy ($k_{B}T$)', fontsize=12)
ax2.tick_params(axis='both', labelsize=10)
ax2.text(12, -0.3, 'L3', fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax2.text(12, -0.36, "$1\\times 10^{9} [steps] $", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax2.text(12, -0.44, "$1\mu$M", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax2.text(12, -0.52, "100 mM NaCl", fontsize=10, color='black', ha='left', va='top', 
         bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
ax2.text(-0.02, 1.07, 'b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax2.set_ylim(bottom=-1.5, top=-0.25)
# Tercer subplot (abajo derecha): Energy Profiles with Logarithmic Time Scale para todas las temperaturas
ax3 = fig.add_subplot(grid[1:, 1])
#ale appr
"""
for i, (temp_folder, df) in enumerate(energy_data_dict.items()):
    df['LogTime'] = df['Time']
    #df['LogTime'] = np.log10(df['Time'])
    df_inverted = df.iloc[::-1].reset_index(drop=True)
    color = colors[i % len(colors)]  # Asegurarse de que el índice esté dentro del rango de colores
    ax3.plot(df_inverted['LogTime'], df_inverted['Energy1'], label=temp_folder, color=color)
"""
#ema

norm_value = 1  # Reemplaza con el valor de normalización apropiado

for i, (temp_folder, df) in enumerate(energy_data_dict.items()):
    # Solo toma el último 10% de los datos
    last_10_percent = df.iloc[-df.shape[0]//1:].reset_index(drop=True)
    
    # Normalizar los datos de 'Energy1'
    normalized_energy = last_10_percent['Energy1'] / norm_value
    
    # Crear una nueva columna 'LogTime' si es necesario
    if 'LogTime' not in last_10_percent.columns:
        last_10_percent['LogTime'] = last_10_percent['Time']
    
    # Graficar usando semilogx
    color = colors[i % len(colors)]  # Asegurarse de que el índice esté dentro del rango de colores
    ax3.semilogx(np.arange(0, last_10_percent.shape[0]), normalized_energy, label=temp_folder, color=color)

ax3.set_xscale('log')
ax3.set_xlabel('Time (sim units)', fontsize=12)
ax3.set_ylabel('Potential Energy ($k_{B}T$)', fontsize=12)
ax3.set_ylim(bottom=-1.4, top=-0.3)
ax3.set_yticks([])
ax3.set_ylabel('')

# Configurar los ticks manualmente para el eje x
#ax3.xaxis.set_major_formatter(NullFormatter())
ax3.xaxis.set_minor_formatter(NullFormatter())
#ax3.set_xticks([1, 10])
ax3.set_ylim(bottom=-1.5, top=-0.25)
ax3.set_xlim(10**0,10**5.5)
#ax3.xaxis.set_major_formatter(FuncFormatter(log_format))
# Crear la barra de colores
cmap = LinearSegmentedColormap.from_list('temp_colormap', colors)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(temp_to_color.keys()), vmax=max(temp_to_color.keys())))
sm.set_array([])

# Añadir la barra de colores dentro del subplot
cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.28])  # Ajustar las coordenadas y el tamaño de la barra de colores
cb = plt.colorbar(sm, cax=cb_ax)
cb.set_label('Temperature ($ ^{o}C$)', fontsize=10)
cb.ax.tick_params(labelsize=10)
cb.ax.yaxis.set_ticks_position('left')

# Añadir sombra ajustable al final del gráfico
shade_start = 10000 * 0.1  # Punto de inicio de la sombra (en log(Time))
shade_end = 10000  # Punto final de la sombra (en log(Time))
ax3.fill_between(last_10_percent['LogTime'], -1.5, 0.5, 
                 where=(last_10_percent['LogTime'] >= shade_start) & (last_10_percent['LogTime'] <= shade_end),
                 color='lightgrey', alpha=0.8)

# Ajustar el diseño y mostrar el gráfico
plt.savefig('combined_plots_with_specific_temp.png', dpi=400)
plt.show()