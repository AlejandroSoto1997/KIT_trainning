import pandas as pd
import os
import matplotlib.pyplot as plt

# Directorio base
base_directory = '/Users/alejandrosoto/Downloads/KIT/Tools/oxDNA/Task 1 Francesco/Y shape /relaxing diff cluster '  # Reemplaza con la ruta de tu directorio base

# Lista de subcarpetas
subfolders = [
    'salt_0.15_58C', 'salt_0.1_58C', 'salt_0.2_58C',
    'salt_0.15_55C', 'salt_0.1_55C', 'salt_0.2_55C',
    'salt_0.15_60C', 'salt_0.1_60C', 'salt_0.2_60C'
]

# Diccionario para almacenar los promedios
average_energy_dict = {}

# Recorre las subcarpetas y procesa los archivos
for subfolder in subfolders:
    folder_path = os.path.join(base_directory, subfolder, 'linker', 'duplex+se')
    file_path = os.path.join(folder_path, 'energy_file_md.txt')

    if os.path.exists(file_path):
        # Lee el archivo de energía con pandas
        df = pd.read_csv(file_path, sep='\s+', names=['Time', 'Energy1', 'Energy2', 'Energy3'])

        # Calcula y almacena el valor promedio de la tercera columna
        average_energy = df['Energy3'].mean()
        average_energy_dict[subfolder] = average_energy
        print(f'Average Total Energy for {subfolder}: {average_energy}')
    else:
        print(f'File not found: {file_path}')

# Crea un DataFrame a partir del diccionario
average_energy_df = pd.DataFrame(list(average_energy_dict.items()), columns=['Subfolder', 'AverageEnergy'])

# Separa la columna "Subfolder" en tres columnas usando el guion bajo como separador
average_energy_df[['Salt', 'salt_conc', 'temp']] = average_energy_df['Subfolder'].str.split('_', expand=True)

# Elimina la letra "C" y convierte la columna "temp" a número
average_energy_df['temp'] = average_energy_df['temp'].str.rstrip('C').astype(float)
# Convierte la columna "salt_conc" a número
average_energy_df['salt_conc'] = average_energy_df['salt_conc'].astype(float)

# Ordena el DataFrame por temperatura y concentración de sal
average_energy_df.sort_values(['temp', 'salt_conc'], inplace=True)

# Crea dos gráficos subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Salt Concentration vs Temperature
for salt_conc, group_df in average_energy_df.groupby('salt_conc'):
    axes[0].plot(group_df['temp'], group_df['AverageEnergy'], marker='o', label=f'Salt conc. = {salt_conc} mM')

axes[0].set_title('Representative total energy (eq) vs temperature')
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Representative total energy (eq)')
axes[0].legend()
axes[0].grid(True)

# Gráfico 2: Temperature vs Salt Concentration
for temp, group_df in average_energy_df.groupby('temp'):
    axes[1].plot(group_df['salt_conc'], group_df['AverageEnergy'], marker='o', label=f'Temp. = {temp}°C')

axes[1].set_title('Representative total energy (eq) vs salt concentration')
axes[1].set_xlabel('Salt concentration (mM)')
axes[1].set_ylabel('Representative total energy (eq)')
axes[1].legend()
axes[1].grid(True)

# Ajusta el paso en el eje x de la segunda gráfica
axes[1].xaxis.set_major_locator(plt.MultipleLocator(0.05))

plt.show()
