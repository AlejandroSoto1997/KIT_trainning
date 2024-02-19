"""
import os

def modify_temperature(root_folder):
    # Recorre todas las carpetas dentro de la nueva ruta
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            # Extrae la temperatura del nombre de la carpeta
            try:
                temperature = int(folder_name)
            except ValueError:
                print(f"El nombre de la carpeta '{folder_name}' no es un número entero. Saltando...")
                continue

            # Busca los archivos 'input_MD.dat' en cada carpeta
            for subdir, _, files in os.walk(folder_path):
                for file_name in files:
                    if file_name == 'input_MD.dat':
                        file_path = os.path.join(subdir, file_name)
                        # Abre el archivo 'input_MD.dat' y lee su contenido
                        with open(file_path, 'r') as file:
                            lines = file.readlines()

                        # Actualiza las líneas con el nuevo valor de temperatura
                        new_lines = []
                        for line in lines:
                            if 'T =' in line:
                                new_line = f'T = {temperature}C\n'
                            else:
                                new_line = line
                            new_lines.append(new_line)

                        # Guarda los cambios en el archivo 'input_MD.dat'
                        with open(file_path, 'w') as file:
                            file.writelines(new_lines)

# Especifica la nueva ruta donde se encuentran los archivos 'input_MD.dat'
root_folder = "/Users/alejandrosoto/Documents/KIT/Task 4 Francesco/100 mM/L1"
# Modifica los archivos 'input_MD.dat' para que la temperatura coincida con el nombre de la carpeta
modify_temperature(root_folder)
"""
#change values of the codes
import os

def modify_input_files(root_folder):
    # Define the values for the lines you want to update
    print_conf_interval_value = 100000.0
    print_energy_every_value = 100000.0
    print_every_value = 1000.0
    steps_value = 1e7

    # Recorre todas las carpetas dentro de la nueva ruta
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            # Busca los archivos 'input_MD.dat' en cada carpeta
            for subdir, _, files in os.walk(folder_path):
                for file_name in files:
                    if file_name == 'input_MD.dat':
                        file_path = os.path.join(subdir, file_name)
                        # Abre el archivo 'input_MD.dat' y lee su contenido
                        with open(file_path, 'r') as file:
                            lines = file.readlines()

                        # Actualiza las líneas con los nuevos valores
                        new_lines = []
                        for line in lines:
                            if 'print_conf_interval' in line:
                                new_line = f'print_conf_interval = {print_conf_interval_value}\n'
                            elif 'print_energy_every' in line:
                                new_line = f'print_energy_every = {print_energy_every_value}\n'
                            elif 'print_every' in line:
                                new_line = f'print_every = {print_every_value}\n'
                            elif 'steps' in line:
                                new_line = f'steps = {steps_value}\n'
                            else:
                                new_line = line  # Mantén la línea sin cambios
                            new_lines.append(new_line)

                        # Guarda los cambios en el archivo 'input_MD.dat'
                        with open(file_path, 'w') as file:
                            file.writelines(new_lines)

# Especifica la nueva ruta donde se encuentran los archivos 'input_MD.dat'
root_folder = "/Users/alejandrosoto/Documents/KIT/Task 4 Francesco/100 mM/L1"
# Modifica los archivos 'input_MD.dat' con los nuevos valores de las líneas específicas
modify_input_files(root_folder)

