import os
import pandas as pd
import matplotlib.pyplot as plt
import statistics

# Directorio base
base_directory = '/Users/alejandrosoto/Documents/KIT/Task 4 Francesco/100 mM/YSHAPE'

# Lista de subcarpetas a procesar
subfolders = ['10','25', '30', '35', '40', '45','50', '52', '55', '50', '52', '55', '57', '60', '62', '65' ,'67','70','72','75','77', '80', '82','85' ]

#rem point: '50', '52', '55', '57', '60', '62' ,'65' ,'67', '70','72','75','77', '80', '82','85' 

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
"""
# Calcular la fracción de enlaces rotos normalizado por temperatura usando la mediana
min_median = min(median_dict.values())
max_median = max(median_dict.values())

fraction_broken_bonds_dict = {}
for temp, median in median_dict.items():
    fraction = (median - min_median) / (max_median - min_median) if max_median != min_median else 0
    fraction_broken_bonds_dict[temp] = fraction
# Calcular la división de la mediana de número de enlaces entre la desviación estándar por temperatura



median_stdev_division_dict = {}
for temp, median in median_dict.items():
    stdev = stdev_dict[temp]  # Obtener la desviación estándar correspondiente a la temperatura
    stdev_min = stdev_df.iloc[0]['Stdev']  # Desviación estándar correspondiente a la temperatura mínima
    stdev_max = stdev_df.iloc[-1]['Stdev']  # Desviación estándar correspondiente a la temperatura máxima
    
    if stdev == 0 or median - min_median == 0 or max_median - min_median == 0:
        division = 0
    else:
        term1 = ((stdev**2 + stdev_min**2)**0.5 / (median - min_median))**2 if median - min_median != 0 else 0
        term2 = ((stdev_max**2 + stdev_min**2)**0.5 / (max_median - min_median))**2 if max_median - min_median != 0 else 0
        division = ((median - min_median) / (max_median - min_median)) * (term1 + term2)**0.5

    median_stdev_division_dict[temp] = division

    


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
plt.plot(positions, sorted_fractions, marker='o', linestyle='--', color='green', label='Fraction of Formed Bonds')

# Obtener las barras de error (division de la mediana entre la desviación estándar)
errors = list(median_stdev_division_dict.values())

# Graficar las barras de error
plt.errorbar(positions, sorted_fractions, yerr=errors, fmt='none', ecolor='red', capsize=5, label='Std. deviation of the calculated fraction')

plt.xticks(positions, sorted_temperatures)  # Mostrar las temperaturas en el eje x
plt.xlabel('Temperature (°C)')
plt.ylabel('Fraction of Formed Bonds')
plt.title('Formed Bonds vs Temperature')
plt.grid(True)
plt.legend()
plt.show()

# Resultados de estadísticas en DataFrames
print("Median DataFrame:")
print(median_df)
print("\nMean DataFrame:")
print(mean_df)
print("\nStdev DataFrame:")
print(stdev_df)
print("\nMinimum Bonds Standard Deviation:", stdev_df.iloc[0]['Stdev'])
print("Maximum Bonds Standard Deviation:", stdev_df.iloc[-1]['Stdev'])
"""

# Calcular la fracción de enlaces rotos normalizado por temperatura usando el promedio
min_mean = min(mean_dict.values())
max_mean = max(mean_dict.values())

fraction_broken_bonds_dict = {}
for temp, mean in mean_dict.items():
    fraction = (mean - min_mean) / (max_mean - min_mean) if max_mean != min_mean else 0
    fraction_broken_bonds_dict[temp] = fraction

# Calcular la división del promedio de número de enlaces entre la desviación estándar por temperatura
mean_stdev_division_dict = {}
for temp, mean in mean_dict.items():
    stdev = stdev_dict[temp]  # Obtener la desviación estándar correspondiente a la temperatura
    stdev_min = stdev_df.iloc[0]['Stdev']  # Desviación estándar correspondiente a la temperatura mínima
    stdev_max = stdev_df.iloc[-1]['Stdev']  # Desviación estándar correspondiente a la temperatura máxima
    
    if stdev == 0 or mean - min_mean == 0 or max_mean - min_mean == 0:
        division = 0
    else:
        term1 = ((stdev**2 + stdev_min**2)**0.5 / (mean - min_mean))**2 if mean - min_mean != 0 else 0
        term2 = ((stdev_max**2 + stdev_min**2)**0.5 / (max_mean - min_mean))**2 if max_mean - min_mean != 0 else 0
        division = ((mean - min_mean) / (max_mean - min_mean)) * (term1 + term2)**0.5

    mean_stdev_division_dict[temp] = division

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
plt.plot(positions, sorted_fractions, marker='o', linestyle='--', color='blue', label='Fraction of Formed Bonds')

# Obtener las barras de error (división del promedio de la mediana entre la desviación estándar)
errors = list(mean_stdev_division_dict.values())

# Graficar las barras de error
plt.errorbar(positions, sorted_fractions, yerr=errors, fmt='none', ecolor='red', capsize=5, label='Std. deviation of the calculated fraction')

plt.xticks(positions, sorted_temperatures)  # Mostrar las temperaturas en el eje x
plt.xlabel('Temperature (°C)')
plt.ylabel('Fraction of Formed Bonds (θ)')
plt.title('Formed Bonds vs Temperature')
plt.grid(True)
plt.legend()
plt.show()
