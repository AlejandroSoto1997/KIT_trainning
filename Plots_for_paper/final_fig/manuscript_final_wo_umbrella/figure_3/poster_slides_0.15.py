import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.ticker import FuncFormatter, NullFormatter

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
        energy_data_dict[temp_folder] = df['Energy1'].values

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

# Leer el archivo de energía para el segundo gráfico (ejemplo de la primera carpeta TEMP_30)
example_folder = 'TEMP_80'
run_folder_path = os.path.join(base_directory, example_folder, 'RUN_0')
file_path = os.path.join(run_folder_path, 'energy_file_md.txt')
df = pd.read_csv(file_path, sep='\s+', names=['Time', 'Energy1', 'Energy2', 'Energy3'])

# Calcular el valor medio de la energía potencial
mean_potential_energy = df['Energy1'].mean()

# Invertir los datos del DataFrame y calcular LogTime
df['LogTime'] = np.log10(df['Time'])
df_inverted = df.iloc[::-1].reset_index(drop=True)

# Configurar la figura con dos subplots
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))

# Ajustar tamaño de fuente general
font_size_axis_labels = 12
font_size_ticks = 12
font_size_legend = 10

# Primer subplot: Median Potential Energy vs Temperature
axs[1].scatter(median_energy_df['Temperature'], median_energy_df['MedianPotentialEnergy'], 
               marker='o', s=60, color='#018571', edgecolor='black')
axs[1].set_xlabel('Temperature ($ ^{o}C$)', fontsize=font_size_axis_labels)
axs[1].set_ylabel('Mean potential energy ($k_{B}T$)', fontsize=font_size_axis_labels)
axs[1].tick_params(axis='both', labelsize=font_size_ticks)

# Segundo subplot: Energy Profiles with Logarithmic Time Scale
axs[0].plot(df_inverted['LogTime'], df_inverted['Energy2'], label='Kinetic energy', color='#bababa')
axs[0].plot(df_inverted['LogTime'], df_inverted['Energy3'], label='Total energy', color='#404040')
axs[0].plot(df_inverted['LogTime'], df_inverted['Energy1'], label='Potential energy', color='#018571')
axs[0].axhline(mean_potential_energy, color='r', linestyle='--', label=f'Mean value: {mean_potential_energy:.2f}' "$k_{B}T$", linewidth=2)



# Añadir sombra ajustable al final del gráfico
shade_start = 0.5
shade_end = 0
axs[0].fill_between(df_inverted['LogTime'], -1.5, 0.5, where=(df_inverted['LogTime'] <= shade_start), color='#018571', alpha=0.2)

# Configurar escala logarítmica y ejes
axs[0].set_xscale('log')  # Escala logarítmica para el eje x
axs[0].set_xlabel('Log(Time)', fontsize=font_size_axis_labels)
axs[0].set_ylabel('Energy ($k_{B}T$)', fontsize=font_size_axis_labels)
axs[0].legend(loc='best', bbox_to_anchor=(0.6, 0.3), fontsize=font_size_legend)
axs[0].set_ylim(bottom=-1.5, top=0.5)

# Establecer los límites del eje x
axs[0].set_xlim(left=10**0, right=10**1)

# Configurar los ticks manualmente
def log_format(x, pos):
    if x == 1:
        return '$10^0$'
    elif x == 10:
        return '$10^1$'
    else:
        return ''

# Ocultar todos los ticks
axs[0].xaxis.set_major_formatter(NullFormatter())  # Oculta los ticks
axs[0].xaxis.set_minor_formatter(NullFormatter())  # Oculta los ticks menores

# Añadir solo los ticks que deseas mostrar
axs[0].set_xticks([1, 10])  # Solo los ticks deseados
axs[0].xaxis.set_major_formatter(FuncFormatter(log_format))  # Etiquetas personalizadas en formato 10^x

# Ajustar el tamaño de los ticks del eje x
axs[0].tick_params(axis='x', labelsize=font_size_ticks)
axs[0].tick_params(axis='y', labelsize=font_size_ticks)

# Quitar el grid del primer subplot
axs[0].grid(False)

# Ajustar el diseño y mostrar el gráfico
plt.tight_layout()
plt.savefig('combined_plots_1.png', dpi=400)
plt.show()
