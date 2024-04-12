import os
import pandas as pd
import matplotlib.pyplot as plt
import statistics

# Directorio base
base_directory = '/Users/alejandrosoto/Documents/KIT/Task 4 Francesco/100 mM/L1'

# Lista de subcarpetas a procesar
subfolders = ['10','25', '30', '35', '40', '45', '50', '52', '55', '57', '60', '62' ,'65' ,'67', '70','72','75','77', '80', '82','85' ]

# Diccionarios para almacenar las estadísticas de enlaces rotos por temperatura
median_dict = {}
mean_dict = {}
stdev_dict = {}

# Recorre las subcarpetas especificadas y procesa los archivos
for subfolder in subfolders:
    folder_path = os.path.join(base_directory, subfolder)
    hb_count_file_path = os.path.join(folder_path, 'hb_count.dat')

    if os.path.exists(hb_count_file_path):
        # Leer el archivo hb_count.dat y obtener los valores
        with open(hb_count_file_path, 'r') as file:
            values = [int(line.strip()) for line in file.readlines() if line.strip()]

        # Calcular las estadísticas deseadas de los enlaces rotos
        if values:
            median = statistics.median(values)
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
        else:
            median = 0
            mean = 0
            stdev = 0

        # Almacenar las estadísticas en los diccionarios
        median_dict[subfolder] = median
        mean_dict[subfolder] = mean
        stdev_dict[subfolder] = stdev

        print(f'Statistics for {subfolder}: Median={median}, Mean={mean}, Stdev={stdev}')
    else:
        print(f'File not found: {hb_count_file_path}')

# Crear DataFrames para almacenar las estadísticas
median_df = pd.DataFrame(list(median_dict.items()), columns=['Temperature (°C)', 'Median'])
mean_df = pd.DataFrame(list(mean_dict.items()), columns=['Temperature (°C)', 'Mean'])
stdev_df = pd.DataFrame(list(stdev_dict.items()), columns=['Temperature (°C)', 'Stdev'])

# Calcular la fracción de enlaces rotos normalizado por temperatura usando la 
#mean
"""
min_median = min(mean_dict.values())
max_median = max(mean_dict.values())
fraction_broken_bonds_dict = {}
for temp, mean in mean_dict.items():
    fraction = (mean - min_median) / (max_median - min_median) if max_median != min_median else 0
    fraction_broken_bonds_dict[temp] = fraction
"""
#median
min_median = min(median_dict.values())
max_median = max(median_dict.values())
fraction_broken_bonds_dict = {}
for temp, median in mean_dict.items():
    fraction = (median - min_median) / (max_median - min_median) if max_median != min_median else 0
    fraction_broken_bonds_dict[temp] = fraction

# Obtener las temperaturas y fracciones formadas como listas ordenadas
temperatures = list(fraction_broken_bonds_dict.keys())
fractions = list(fraction_broken_bonds_dict.values())

# Ordenar las temperaturas y fracciones
sorted_temperatures = sorted(temperatures, key=lambda x: int(x))
sorted_fractions = [fraction_broken_bonds_dict[temp] for temp in sorted_temperatures]

# Calcular las diferencias entre las temperaturas adyacentes
temperature_diffs = [int(sorted_temperatures[i + 1]) - int(sorted_temperatures[i]) for i in range(len(sorted_temperatures) - 1)]

# Calcular las posiciones en el eje x teniendo en cuenta las diferencias de temperatura
positions = [0]
current_position = 0
for diff in temperature_diffs:
    current_position += diff
    positions.append(current_position)

# Graficar la fracción de enlaces rotos normalizado vs posición espacial real
plt.plot(positions, sorted_fractions, marker='o', linestyle='--', color='green')
plt.xticks(positions, sorted_temperatures)  # Mostrar las temperaturas en el eje x
plt.xlabel('Temperature (°C)')
plt.ylabel('Fraction of Formed Bonds')
plt.title('Fraction Formed Bonds vs Temperature')
plt.grid(True)
plt.show()

# Resultados de estadísticas en DataFrames
print("Median DataFrame:")
print(median_df)
print("\nMean DataFrame:")
print(mean_df)
print("\nStdev DataFrame:")
print(stdev_df)
