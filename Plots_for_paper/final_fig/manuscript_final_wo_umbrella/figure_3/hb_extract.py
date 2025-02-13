#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:25:47 2024

@author: alejandrosoto
"""

import os
import shutil

# Ruta base donde están los folders de temperatura
base_path = "/Users/alejandrosoto/Documents/KIT/last_task_ale/explicit_server/SALT_0.1"

# Lista de temperaturas disponibles
temperatures = range(14, 88, 2)

# Crear la carpeta "L_3" si no existe
destination_folder = os.path.join(base_path, "L_3")
os.makedirs(destination_folder, exist_ok=True)

# Iterar sobre las carpetas de temperatura
for temp in temperatures:
    temp_folder = f"TEMP_{temp}"
    source_file = os.path.join(base_path, temp_folder, "RUN_0", "hb_list.dat")
    
    if os.path.exists(source_file):
        # Nuevo nombre de archivo con la temperatura
        new_file_name = f"hb_list.{temp}.dat"
        destination_file = os.path.join(destination_folder, new_file_name)
        
        # Copiar el archivo al folder "L_3"
        shutil.copy(source_file, destination_file)
        print(f"Archivo copiado: {new_file_name}")
    else:
        print(f"Archivo no encontrado en {temp_folder}")

print("Proceso completado.")
