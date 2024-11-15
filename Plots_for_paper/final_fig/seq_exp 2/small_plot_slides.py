#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:03:49 2024

@author: alejandrosoto
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'no-latex', 'bright'])

def read_melting_curves_nupack(directory):
    dfs_nupack = []
    for conc_dir in sorted(os.listdir(directory)):
        if any(char.isdigit() for char in conc_dir):  # Verificar si el directorio contiene dígitos
            conc_path = os.path.join(directory, conc_dir)
            if os.path.isdir(conc_path):
                # Extraer los caracteres numéricos del nombre del directorio y convertirlo a float
                molecule = float(''.join(filter(str.isdigit, conc_dir)))
                file_path = os.path.join(conc_path, f"{molecule}.csv")
                if os.path.isfile(file_path):  # Verificar si el archivo existe
                    df = pd.read_csv(file_path)
                    dfs_nupack.append((molecule, df))
    return sorted(dfs_nupack, key=lambda x: x[0])  # Ordenar por concentración ascendente

def find_melting_temperature(df):
    # Encontrar el índice de la temperatura más cercana a la fracción 0.5
    idx1 = np.abs(df.iloc[:, 1] - 0.5).idxmin()
    
    # Encontrar los índices de los puntos más cercanos a ambos lados del punto 0.5
    idx2 = idx1 + 1 if df.iloc[idx1, 1] < 0.5 else idx1 - 1
    
    # Realizar interpolación lineal entre los puntos más cercanos
    x1, y1 = df.iloc[idx1, 0], df.iloc[idx1, 1]
    x2, y2 = df.iloc[idx2, 0], df.iloc[idx2, 1]
    tm = x1 + ((0.5 - y1) * (x2 - x1) / (y2 - y1))
    
    return tm

# Directorios donde se encuentran los archivos CSV
directory_bases_profiles = "/Users/alejandrosoto/Downloads/seq_exp/0.1M"
directory_nupack = "/Users/alejandrosoto/Downloads/seq_exp/0.1M"

# Plot de los perfiles de desaparición de bases
archivos_bases_profiles = [archivo for archivo in os.listdir(directory_bases_profiles) if archivo.endswith(".csv")]

# Lista para almacenar los nombres de los archivos y los valores de Tm
tm_info = []

# Iterar sobre los archivos CSV y encontrar Tm para cada uno
for archivo in archivos_bases_profiles:
    df = pd.read_csv(os.path.join(directory_bases_profiles, archivo))
    tm = find_melting_temperature(df)
    tm_info.append((archivo, tm))

# Imprimir los nombres de los archivos y los valores de Tm
print("Información de Tm:")
for nombre_archivo, tm in tm_info:
    print(f"Archivo: {nombre_archivo}, Tm: {tm:.2f} °C")
    df_tm_info = pd.DataFrame(tm_info, columns=['molecule', 'Tm'])
    df_tm_info[['molecule_clean', 'type_of_file']] = df_tm_info['molecule'].str.split('.c', expand=True)
    df_tm_info.drop(columns=['molecule'], inplace=True)
    df_tm_info.drop(columns=['type_of_file'], inplace=True)
    df_tm_info[['type_of_mol','molecule_clean_id']] = df_tm_info['molecule_clean'].str.split('L', expand=True)
    df_tm_info['molecule_clean_id'] = df_tm_info['molecule_clean_id'].astype(float)
    df_tm_info = df_tm_info.sort_values(by='molecule_clean_id')
    df_tm_info = df_tm_info.reset_index(drop=True)
    df_tm_info['nucleotides'] = range(12, -1, -1)
    print(df_tm_info)

# Graficar la primera curva de desaparición de bases
fig, axs = plt.subplots(1, 1, figsize=(6, 6))

df1 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[11]))
axs.plot(df1["Temperature(C)"], df1["Fraction of bases unpaired at equilibrium"], label=f'L0 ' "$(T_{m}=$"f'{find_melting_temperature(df1):.2f}$^\circ$C)', color='#9e0142')

df2 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[12]))
axs.plot(df2["Temperature(C)"], df2["Fraction of bases unpaired at equilibrium"], label=f'L1 ' "$(T_{m}=$"f'{find_melting_temperature(df2):.2f}$^\circ$C)', color='#d53e4f')

df3 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[10]))
axs.plot(df3["Temperature(C)"], df3["Fraction of bases unpaired at equilibrium"], label=f'L2 ' "$(T_{m}=$"f'{find_melting_temperature(df3):.2f}$^\circ$C)', color='#f46d43')

df4 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[9]))
axs.plot(df4["Temperature(C)"], df4["Fraction of bases unpaired at equilibrium"], label=f'L3 ' "$(T_{m}=$"f'{find_melting_temperature(df4):.2f}$^\circ$C)', color='#fdae61')

df5 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[6]))
axs.plot(df5["Temperature(C)"], df5["Fraction of bases unpaired at equilibrium"], label=f'L4 ' "$(T_{m}=$"f'{find_melting_temperature(df5):.2f}$^\circ$C)', color='#fee08b')

df6 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[5]))
axs.plot(df6["Temperature(C)"], df6["Fraction of bases unpaired at equilibrium"], label=f'L5 ' "$(T_{m}=$"f'{find_melting_temperature(df6):.2f}$^\circ$C)', color='#ffffbf')

df7 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[7]))
axs.plot(df7["Temperature(C)"], df7["Fraction of bases unpaired at equilibrium"], label=f'L6 ' "$(T_{m}=$"f'{find_melting_temperature(df7):.2f}$^\circ$C)', color='#e6f598')

df8 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[8]))
axs.plot(df8["Temperature(C)"], df8["Fraction of bases unpaired at equilibrium"], label=f'L7 ' "$(T_{m}=$"f'{find_melting_temperature(df8):.2f}$^\circ$C)', color='#abdda4')

df9 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[1]))
axs.plot(df9["Temperature(C)"], df9["Fraction of bases unpaired at equilibrium"], label=f'L8 ' "$(T_{m}=$"f'{find_melting_temperature(df9):.2f}$^\circ$C)', color='#66c2a5')

df10 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[0]))
axs.plot(df10["Temperature(C)"], df10["Fraction of bases unpaired at equilibrium"], label=f'L9 ' "$(T_{m}=$"f'{find_melting_temperature(df10):.2f}$^\circ$C)', color='#3288bd')

df11 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[3]))
axs.plot(df11["Temperature(C)"], df11["Fraction of bases unpaired at equilibrium"], label=f'L10 ' "$(T_{m}=$"f'{find_melting_temperature(df11):.2f}$^\circ$C)', color='#9673b9')

df12 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[4]))
axs.plot(df12["Temperature(C)"], df12["Fraction of bases unpaired at equilibrium"], label=f'L11 ' "$(T_{m}=$"f'{find_melting_temperature(df12):.2f}$^\circ$C)', color='#5e4fa2')

df13 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[2]))
axs.plot(df13["Temperature(C)"], df13["Fraction of bases unpaired at equilibrium"], label=f'duplex ' "$(T_{m}=$"f'{find_melting_temperature(df13):.2f}$^\circ$C)', color='#2c1e50')

#ax.plot(df_tm_info['con_wo_units'], df_tm_info['Tm'], marker='o', linestyle='-')
axs.set_ylim(0, 1)
#axs.set_yticks(np.arange(0, 1.1, 0.1))
axs.set_xlabel('Temperature ($^{o}$C)')
axs.set_ylabel('Fraction unbounded')

plt.savefig('nupack_con_scr.png', dpi=300)
plt.show()