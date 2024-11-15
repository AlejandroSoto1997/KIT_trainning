#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:55:40 2024

@author: alejandrosoto
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science', 'no-latex', 'bright'])

def read_melting_curves_nupack(directory):
    dfs_nupack = []
    for conc_dir in sorted(os.listdir(directory)):
        if any(char.isdigit() for char in conc_dir):  # Verificar si el directorio contiene dígitos
            conc_path = os.path.join(directory, conc_dir)
            if os.path.isdir(conc_path):
                # Extraer los caracteres numéricos del nombre del directorio y convertirlo a float
                concentration = float(''.join(filter(str.isdigit, conc_dir)))
                file_path = os.path.join(conc_path, f"{concentration}.csv")
                if os.path.isfile(file_path):  # Verificar si el archivo existe
                    df = pd.read_csv(file_path)
                    dfs_nupack.append((concentration, df))
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
directory_bases_profiles = "./duplex"
directory_nupack = "./"

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

df_tm_info = pd.DataFrame(tm_info, columns=['Concentracion', 'Tm'])

# Dividir el nombre del archivo en dos columnas separadas solo si contiene un guion bajo
if df_tm_info['Concentracion'].str.contains('_').any():
    df_tm_info[['linker', 'Concentracion_ext']] = df_tm_info['Concentracion'].str.split('_', expand=True)
    df_tm_info[['con_w_units', 'type_of_file']] = df_tm_info['Concentracion_ext'].str.split('.c', expand=True)
    df_tm_info.drop(columns=['Concentracion_ext'], inplace=True)
    df_tm_info[['con_wo_units', 'units']] = df_tm_info['con_w_units'].str.split('uM', expand=True)
    df_tm_info.drop(columns=['type_of_file'], inplace=True)
    df_tm_info.drop(columns=['con_w_units'], inplace=True)
    df_tm_info.drop(columns=['units'], inplace=True)
    df_tm_info['con_wo_units'] = df_tm_info['con_wo_units'].astype(float)
    df_tm_info = df_tm_info.sort_values(by='con_wo_units')
    df_tm_info = df_tm_info.reset_index(drop=True)
else:
    # Si no contiene un guion bajo, extraer la parte numérica del nombre del archivo
    df_tm_info['con_wo_units'] = df_tm_info['Concentracion'].str.extract(r'(\d+\.\d+|\d+)', expand=False).astype(float)
    df_tm_info = df_tm_info.sort_values(by='con_wo_units')
    df_tm_info = df_tm_info.reset_index(drop=True)

print(df_tm_info)

# Graficar cada perfil de desaparición de bases por separado
# Graficar la primera curva de desaparición de bases
fig, axs = plt.subplots(2, 1, figsize=(7.5, 12))

df1 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[7]))
axs[0].plot(df1["Temperature(C)"], df1["Fraction of bases unpaired at equilibrium"], color = '#d53e4f', label=f"{df_tm_info['con_wo_units'].iloc[0]:.2f} $\mu M$")
axs[0].legend(title="Concentration", loc='best', fontsize=16, title_fontsize=14)

df2 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[1]))
axs[0].plot(df2["Temperature(C)"], df2["Fraction of bases unpaired at equilibrium"],color = '#f46d43', label=f"{df_tm_info['con_wo_units'].iloc[1]} $\mu M$")
axs[0].legend(loc='best', fontsize=16, title_fontsize=14)

df3 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[6]))
axs[0].plot(df3["Temperature(C)"], df3["Fraction of bases unpaired at equilibrium"],color = '#fdae61', label=f"{df_tm_info['con_wo_units'].iloc[2]} $\mu M$")
axs[0].legend(loc='best', fontsize=16, title_fontsize=14)

df4 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[3]))
axs[0].plot(df4["Temperature(C)"], df4["Fraction of bases unpaired at equilibrium"],color = '#fee08b', label=f"{df_tm_info['con_wo_units'].iloc[3]} $\mu M$")
axs[0].legend(loc='best', fontsize=16, title_fontsize=14)

df5 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[5]))
axs[0].plot(df5["Temperature(C)"], df5["Fraction of bases unpaired at equilibrium"],color = '#e6f598', label=f"{df_tm_info['con_wo_units'].iloc[4]} $\mu M$")
axs[0].legend(loc='best', fontsize=16, title_fontsize=14)

df6 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[2]))
axs[0].plot(df6["Temperature(C)"], df6["Fraction of bases unpaired at equilibrium"],color = '#abdda4', label=f"{df_tm_info['con_wo_units'].iloc[5]} $\mu M$")
axs[0].legend(loc='best', fontsize=16, title_fontsize=14)

df7 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[4]))
axs[0].plot(df7["Temperature(C)"], df7["Fraction of bases unpaired at equilibrium"],color = '#66c2a5', label=f"{df_tm_info['con_wo_units'].iloc[6]} $\mu M$")
axs[0].legend(loc='best', fontsize=16, title_fontsize=14)

df8 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[0]))
axs[0].plot(df8["Temperature(C)"], df8["Fraction of bases unpaired at equilibrium"],color = '#3288bd', label=f"{df_tm_info['con_wo_units'].iloc[7]} $\mu M$")
axs[0].legend(loc='best', fontsize=16, title_fontsize=14)

#d53e4f
#f46d43
#fdae61
#fee08b
#e6f598
#abdda4
#66c2a5
#3288bd

# Configuración del zoom
axins = axs[0].inset_axes([0.12, 0.7, 0.25, 0.25])  # (x0, y0, width, height) del panel secundario
axins.plot(df1["Temperature(C)"], df1["Fraction of bases unpaired at equilibrium"], color = '#d53e4f')
axins.plot(df2["Temperature(C)"], df2["Fraction of bases unpaired at equilibrium"], color = '#f46d43')
axins.plot(df3["Temperature(C)"], df3["Fraction of bases unpaired at equilibrium"], color = '#fdae61')
axins.plot(df4["Temperature(C)"], df4["Fraction of bases unpaired at equilibrium"], color = '#fee08b')
axins.plot(df5["Temperature(C)"], df5["Fraction of bases unpaired at equilibrium"], color = '#e6f598')
axins.plot(df6["Temperature(C)"], df6["Fraction of bases unpaired at equilibrium"], color = '#abdda4')
axins.plot(df7["Temperature(C)"], df7["Fraction of bases unpaired at equilibrium"], color = '#66c2a5')
axins.plot(df8["Temperature(C)"], df8["Fraction of bases unpaired at equilibrium"], color = '#3288bd')

axins.set_xlim(65, 71)
axins.set_ylim(0.46, 0.54)
axins.set_xticks(np.arange(65, 72, 2))
axins.set_yticks(np.arange(0.46, 0.56, 0.04))
axins.grid(True)

# Cambiar el tamaño de la fuente de los ticks del gráfico de zoom
axins.tick_params(axis='both', which='major', labelsize=16)
# Configuración del tamaño de fuente de los ticks del primer subplot
axs[0].tick_params(axis='both', which='major', labelsize=25)  # Aquí ajusta el tamaño de fuente según tus preferencias

# Añadir guías al gráfico principal para mostrar el área de zoom
axs[0].indicate_inset_zoom(axins, edgecolor="black")
axs[0].set_ylim(0, 1)
axs[0].set_yticks(np.arange(0, 1.2, 0.2))
axs[0].set_xlabel('Temperature ($^{o}$C)', fontsize=25)
axs[0].set_ylabel('Fraction unbounded', fontsize=25)

# Graficar el segundo subplot (scatter plot de Tm vs Concentración)
colors = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
axs[1].scatter(df_tm_info['con_wo_units'], df_tm_info['Tm'], marker='o', color=colors, s=150,edgecolor='black')

# Añadir etiquetas de datos (Tm) a cada punto con posición controlada
# Añadir etiquetas de datos (Tm) a cada punto con posición controlada
for i, txt in enumerate(df_tm_info['Tm']):
    x = df_tm_info['con_wo_units'][i]
    y = df_tm_info['Tm'][i]
    if i == 4:  # Ajuste específico para la segunda etiqueta
        axs[1].text(x - 0.2, y + 0.14, f'{txt:.2f}', fontsize=14, ha='center', va='bottom')
    elif i == 5:  # Ajuste específico para la cuarta etiqueta
        axs[1].text(x + 0.22, y - 0.3, f'{txt:.2f}', fontsize=14, ha='center', va='bottom')
    else:
        axs[1].text(x - 0.01, y + 0.08, f'{txt:.1f}', fontsize=14, ha='center', va='bottom')


# Configurar el resto de detalles del segundo subplot
axs[1].set_xlabel('Concentration ($\mu M$)', fontsize=25)
axs[1].set_ylabel('Melting temperature ($^{o}$C)', fontsize=25)
axs[1].set_yticks(np.arange(67, 72, 1))
axs[1].set_xticks(np.arange(0, 14, 2)) 
#axs[1].set_ylim(66.5, 68)
axs[1].tick_params(axis='both', which='major', labelsize=25)





plt.tight_layout()

# Guardar la figura y mostrarla
plt.savefig('nupack_con_scr_paper.png', dpi=300)



plt.show()
