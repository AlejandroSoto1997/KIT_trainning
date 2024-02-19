import os
import pandas as pd
import matplotlib.pyplot as plt

# Directorio base
base_directory = '/Users/alejandrosoto/Documents/KIT/Task 3 Iliya/francesco suggestion/0.10/partial_results_trial-08.02.24/task2'

# Lista de subcarpetas
subfolders = ['10', '15', '20', '25', '30', '35', '40', '45', '50','51', '55', '60', '65', '70','75','80','85']

# Diccionario para almacenar el número total de enlaces por temperatura
total_bond_count_dict = {}

# Recorre las subcarpetas y procesa los archivos
for subfolder in subfolders:
    folder_path = os.path.join(base_directory, subfolder, 'linker', 'duple+se')
    hb_list_file_path = os.path.join(folder_path, 'hb_list.dat')

    if os.path.exists(hb_list_file_path):
        # Leer el archivo hb_list.dat y contar el número total de enlaces
        with open(hb_list_file_path, 'r') as file:
            lines = file.readlines()
            bond_count = sum(1 for line in lines[1:] if line.strip() and ' ' in line)  # Ignorar encabezado y líneas en blanco

        # Almacenar el número total de enlaces en el diccionario
        total_bond_count_dict[subfolder] = bond_count
        print(f'Total number of bonds for {subfolder}: {bond_count}')
    else:
        print(f'File not found: {hb_list_file_path}')

# Calcular la fracción de enlaces promedio por temperatura
bond_count_10C = total_bond_count_dict['10']
bond_count_85C = total_bond_count_dict['85']
fraction_bond_count_dict = {}
for temp, bond_count in total_bond_count_dict.items():
    fraction = 1 - (bond_count_10C - bond_count) / (bond_count_10C - bond_count_85C)
    
    #fraction = bond_count/bond_count_10C
    fraction_bond_count_dict[temp] = fraction

# Guardar los datos como un DataFrame de pandas
df = pd.DataFrame(list(fraction_bond_count_dict.items()), columns=['Temperature (°C)', 'Fraction Formed'])
# Guardar el DataFrame como un archivo CSV
csv_file_path = 'fraction_data_steps_10_6_hb_print_1_jMD_trial2.csv'
df.to_csv(csv_file_path, index=False)

# Graficar la fracción de enlaces promedio vs temperatura
fig, ax = plt.subplots()
ax.plot(fraction_bond_count_dict.keys(), fraction_bond_count_dict.values(), marker='o', linestyle='--', color='red')
ax.set_title('Fraction Formed Bonds vs Temperature')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Fraction of Formed Bonds')

plt.grid(True)
plt.show()
