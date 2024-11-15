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
from scipy.interpolate import interp1d
import scienceplots
import matplotlib.colors as mcolors
import matplotlib.cm as cm


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
fig, axs = plt.subplots(2, 2, figsize=(9, 7.8))
#plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.06, hspace=0.16)
plt.subplots_adjust(left=0.07, right=0.9, top=1.0, bottom=0.05, wspace=0.15, hspace=0.18)
#fig.subplots_adjust(wspace=1, hspace=0)

# Colores definidos
colors = ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027']


# Graficar datos con los colores especificados
df1 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[7]))
axs[0,0].plot(df1["Temperature(C)"], df1["Fraction of bases unpaired at equilibrium"], color=colors[0], label=f"{df_tm_info['con_wo_units'].iloc[0]:.2f} $\mu M$")

df2 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[1]))
axs[0,0].plot(df2["Temperature(C)"], df2["Fraction of bases unpaired at equilibrium"], color=colors[1], label=f"{df_tm_info['con_wo_units'].iloc[1]} $\mu M$")

df3 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[6]))
axs[0,0].plot(df3["Temperature(C)"], df3["Fraction of bases unpaired at equilibrium"], color=colors[2], label=f"{df_tm_info['con_wo_units'].iloc[2]} $\mu M$")

df4 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[3]))
axs[0,0].plot(df4["Temperature(C)"], df4["Fraction of bases unpaired at equilibrium"], color=colors[3], label=f"{df_tm_info['con_wo_units'].iloc[3]} $\mu M$")

df5 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[5]))
axs[0,0].plot(df5["Temperature(C)"], df5["Fraction of bases unpaired at equilibrium"], color=colors[4], label=f"{df_tm_info['con_wo_units'].iloc[4]} $\mu M$")

df6 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[2]))
axs[0,0].plot(df6["Temperature(C)"], df6["Fraction of bases unpaired at equilibrium"], color=colors[5], label=f"{df_tm_info['con_wo_units'].iloc[5]} $\mu M$")

df7 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[4]))
axs[0,0].plot(df7["Temperature(C)"], df7["Fraction of bases unpaired at equilibrium"], color=colors[6], label=f"{df_tm_info['con_wo_units'].iloc[6]} $\mu M$")

df8 = pd.read_csv(os.path.join(directory_bases_profiles, archivos_bases_profiles[0]))
axs[0,0].plot(df8["Temperature(C)"], df8["Fraction of bases unpaired at equilibrium"], color=colors[7], label=f"{df_tm_info['con_wo_units'].iloc[7]} $\mu M$")

# Configurar la leyenda
axs[0,0].legend(loc='best', fontsize=10, title_fontsize=12)

axs[0, 0].text(80, 0.87, 'L12 duplex core',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotación en la segunda línea
axs[0, 0].text(80, 0.82, '100 mM NaCl',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

#d53e4f
#f46d43
#fdae61
#fee08b
#e6f598
#abdda4
#66c2a5
#3288bd

# Configuración del zoom
axins = axs[0,0].inset_axes([0.12, 0.7, 0.25, 0.25])  # (x0, y0, width, height) del panel secundario
axins.plot(df1["Temperature(C)"], df1["Fraction of bases unpaired at equilibrium"], color=colors[0])
axins.plot(df2["Temperature(C)"], df2["Fraction of bases unpaired at equilibrium"], color=colors[1])
axins.plot(df3["Temperature(C)"], df3["Fraction of bases unpaired at equilibrium"], color=colors[2])
axins.plot(df4["Temperature(C)"], df4["Fraction of bases unpaired at equilibrium"], color=colors[3])
axins.plot(df5["Temperature(C)"], df5["Fraction of bases unpaired at equilibrium"], color=colors[4])
axins.plot(df6["Temperature(C)"], df6["Fraction of bases unpaired at equilibrium"], color=colors[5])
axins.plot(df7["Temperature(C)"], df7["Fraction of bases unpaired at equilibrium"], color=colors[6])
axins.plot(df8["Temperature(C)"], df8["Fraction of bases unpaired at equilibrium"], color=colors[7])


axins.set_xlim(65, 71)
axins.set_ylim(0.46, 0.54)
axins.set_xticks(np.arange(65, 72, 3))
axins.set_yticks(np.arange(0.46, 0.56, 0.04))
axins.grid(True)

# Cambiar el tamaño de la fuente de los ticks del gráfico de zoom
axins.tick_params(axis='both', which='major', labelsize=10)
# Configuración del tamaño de fuente de los ticks del primer subplot
axs[0,0].tick_params(axis='both', which='major', labelsize=10)  # Aquí ajusta el tamaño de fuente según tus preferencias

# Añadir guías al gráfico principal para mostrar el área de zoom
axs[0,0].indicate_inset_zoom(axins, edgecolor="black")
axs[0,0].set_ylim(0, 1)
axs[0,0].set_yticks(np.arange(0, 1.1, 0.1))
axs[0,0].set_xlabel('Temperature ($^{o}$C)', fontsize=12)
axs[0,0].set_ylabel('Fraction unbounded', fontsize=12)

# Graficar el segundo subplot (scatter plot de Tm vs Concentración)
#colors = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
colors = ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027']

axs[1,0].scatter(df_tm_info['con_wo_units'], df_tm_info['Tm'], marker='o', color=colors, s=150,edgecolor='black')

# Añadir etiquetas de datos (Tm) a cada punto con posición controlad



# Añadir etiquetas de datos (Tm) a cada punto con posición controlada
for i, txt in enumerate(df_tm_info['Tm']):
    x = df_tm_info['con_wo_units'][i]
    y = df_tm_info['Tm'][i]
    if i == 4:  # Ajuste específico para la segunda etiqueta
        axs[1,0].text(x - 0.2, y + 0.2, f'{txt:.2f}', fontsize=9, ha='center', va='bottom')
    elif i == 5:  # Ajuste específico para la cuarta etiqueta
        axs[1,0].text(x + 0.22, y - 0.7, f'{txt:.2f}', fontsize=9, ha='center', va='bottom')
    else:
        axs[1,0].text(x - 0.01, y - 0.7, f'{txt:.1f}', fontsize=9, ha='center', va='bottom')


# Configurar el resto de detalles del segundo subplot
axs[1,0].set_xlabel('Concentration ($\mu M$)', fontsize=12)
axs[1,0].set_ylabel('Melting temperature ($^{o}$C)', fontsize=12)
axs[1,0].set_yticks(np.arange(62, 72, 1))
axs[1,0].set_xticks(np.arange(0, 13, 1)) 
#axs[1,0].set_ylim(66.5, 68)
axs[1,0].tick_params(axis='both', which='major', labelsize=10)

# Crear un mapa de colores suavizado
cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

# Crear un normalizador para la barra de colores, basado en los valores reales
norm = mcolors.Normalize(vmin=min(df_tm_info['con_wo_units']), vmax=max(df_tm_info['con_wo_units']))

# Crear el ScalarMappable para la barra de colores
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Ajustar el tamaño y la posición de la barra de colores
cbar_ax = fig.add_axes([0.1, 0.13, 0.33, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

# Ajustar los ticks, eliminando el cuarto y sexto valor
ticks = [tick for i, tick in enumerate(df_tm_info['con_wo_units']) if i not in [3, 5]]
tick_labels = [f'{val:.1f}' for val in ticks]

cbar.set_ticks(ticks)
cbar.set_ticklabels(tick_labels)
cbar.set_label('Concentration('"$\mu$"'M)')


axs[1, 0].text(0.3, 70.8, 'L12 duplex core',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotación en la segunda línea
axs[1, 0].text(0.3, 70.3, '100 mM NaCl',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))


#plt.tight_layout()

# Guardar la figura y mostrarla
#plt.savefig('nupack_con_scr_paper_combine.png', dpi=300)

# Directorios donde se encuentran los archivos CSV
directory_bases_profiles_1 = "./0.1M"
directory_nupack_1 = "./0.1M"

# Plot de los perfiles de desaparición de bases
archivos_bases_profiles_1 = [archivo for archivo in os.listdir(directory_bases_profiles_1) if archivo.endswith(".csv")]

# Lista para almacenar los nombres de los archivos y los valores de Tm
tm_info_1 = []

# Iterar sobre los archivos CSV y encontrar Tm para cada uno
for archivo in archivos_bases_profiles_1:
    df = pd.read_csv(os.path.join(directory_bases_profiles_1, archivo))
    tm = find_melting_temperature(df)
    tm_info_1.append((archivo, tm))

# Imprimir los nombres de los archivos y los valores de Tm
print("Información de Tm:")
for nombre_archivo, tm in tm_info_1:
    print(f"Archivo: {nombre_archivo}, Tm: {tm:.2f} °C")
    df_tm_info_1 = pd.DataFrame(tm_info_1, columns=['molecule', 'Tm'])
    # Dividir el nombre del archivo en dos columnas separadas
    #df_tm_info_1[['linker', 'Concentracion_ext']] = df_tm_info_1['Concentracion'].str.split('_', expand=True)
    df_tm_info_1[['molecule_clean', 'type_of_file']] = df_tm_info_1['molecule'].str.split('.c', expand=True)
    df_tm_info_1.drop(columns=['molecule'], inplace=True)
    df_tm_info_1.drop(columns=['type_of_file'], inplace=True)
    df_tm_info_1[['type_of_mol','molecule_clean_id']] = df_tm_info_1['molecule_clean'].str.split('L', expand=True)
    
    #df_tm_info_1[['con_wo_units', 'units']] = df_tm_info_1['con_w_units'].str.split('uM', expand=True)
    #df_tm_info_1.drop(columns=['type_of_file'], inplace=True)
    #df_tm_info_1.drop(columns=['con_w_units'], inplace=True)
    #df_tm_info_1.drop(columns=['units'], inplace=True)
    df_tm_info_1['molecule_clean_id'] = df_tm_info_1['molecule_clean_id'].astype(float)
    df_tm_info_1 = df_tm_info_1.sort_values(by='molecule_clean_id')
    df_tm_info_1 = df_tm_info_1.reset_index(drop=True)
    df_tm_info_1['nucleotides'] = range(12, -1, -1)
    
    # Eliminar las columnas intermedias que ya no necesitamos
    #df_tm_info_1.drop(columns=['Concentracion_ext', 'ext'], inplace=True)

    print(df_tm_info_1)
# Graficar cada perfil de desaparición de bases por separado
# Graficar la primera curva de desaparición de bases
# Graficar la primera curva de desaparición de bases

colors_1 = [
"#313695", "#416aaf", "#659bc8", "#a3d3e6", "#d1ecf4", "#f2fad6", "#fee99d", "#fdc374", "#f99153", "#e34933", "#c01a27", "#c6171b", "#7f0d0b"
]


# Leer y graficar los datos, añadiendo la leyenda a cada gráfico
df1 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[11]))
axs[0,1].plot(df1["Temperature(C)"], df1["Fraction of bases unpaired at equilibrium"], color = colors_1[0], label = df_tm_info_1['molecule_clean'].iloc[0])

df2 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[12]))
axs[0,1].plot(df2["Temperature(C)"], df2["Fraction of bases unpaired at equilibrium"], color = colors_1[1], label = df_tm_info_1['molecule_clean'].iloc[1])

df3 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[10]))
axs[0,1].plot(df3["Temperature(C)"], df3["Fraction of bases unpaired at equilibrium"], color = colors_1[2], label = df_tm_info_1['molecule_clean'].iloc[2])

df4 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[9]))
axs[0,1].plot(df4["Temperature(C)"], df4["Fraction of bases unpaired at equilibrium"], color = colors_1[3], label = df_tm_info_1['molecule_clean'].iloc[3])

df5 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[6]))
axs[0,1].plot(df5["Temperature(C)"], df5["Fraction of bases unpaired at equilibrium"], color = colors_1[4], label = df_tm_info_1['molecule_clean'].iloc[4])

df6 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[5]))
axs[0,1].plot(df6["Temperature(C)"], df6["Fraction of bases unpaired at equilibrium"], color = colors_1[5], label = df_tm_info_1['molecule_clean'].iloc[5])

df7 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[7]))
axs[0,1].plot(df7["Temperature(C)"], df7["Fraction of bases unpaired at equilibrium"], color = colors_1[6], label = df_tm_info_1['molecule_clean'].iloc[6])

df8 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[8]))
axs[0,1].plot(df8["Temperature(C)"], df8["Fraction of bases unpaired at equilibrium"], color = colors_1[7], label = df_tm_info_1['molecule_clean'].iloc[7])

df9 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[1]))
axs[0,1].plot(df9["Temperature(C)"], df9["Fraction of bases unpaired at equilibrium"], color = colors_1[8], label = df_tm_info_1['molecule_clean'].iloc[8])

df10 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[0]))
axs[0,1].plot(df10["Temperature(C)"], df10["Fraction of bases unpaired at equilibrium"], color = colors_1[9], label = df_tm_info_1['molecule_clean'].iloc[9])

df11 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[3]))
axs[0,1].plot(df11["Temperature(C)"], df11["Fraction of bases unpaired at equilibrium"], color = colors_1[10], label = df_tm_info_1['molecule_clean'].iloc[10])

df12 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[4]))
axs[0,1].plot(df12["Temperature(C)"], df12["Fraction of bases unpaired at equilibrium"], color = colors_1[11], label = df_tm_info_1['molecule_clean'].iloc[11])

df13 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[2]))
axs[0,1].plot(df13["Temperature(C)"], df13["Fraction of bases unpaired at equilibrium"], color = colors_1[12], label = df_tm_info_1['molecule_clean'].iloc[12])

# Añadir leyenda con dos columnas
axs[0,1].legend(loc='best', fontsize=10, ncol=2)

# Configuración del zoom
axins = axs[0,1].inset_axes([0.15, 0.7, 0.25, 0.25])  # (x0, y0, width, height) del panel secundario
# Aplicar los colores a los plots
axins.plot(df1["Temperature(C)"], df1["Fraction of bases unpaired at equilibrium"], color=colors_1[0])
axins.plot(df2["Temperature(C)"], df2["Fraction of bases unpaired at equilibrium"], color=colors_1[1])
axins.plot(df3["Temperature(C)"], df3["Fraction of bases unpaired at equilibrium"], color=colors_1[2])
axins.plot(df4["Temperature(C)"], df4["Fraction of bases unpaired at equilibrium"], color=colors_1[3])
axins.plot(df5["Temperature(C)"], df5["Fraction of bases unpaired at equilibrium"], color=colors_1[4])
axins.plot(df6["Temperature(C)"], df6["Fraction of bases unpaired at equilibrium"], color=colors_1[5])
axins.plot(df7["Temperature(C)"], df7["Fraction of bases unpaired at equilibrium"], color=colors_1[6])
axins.plot(df8["Temperature(C)"], df8["Fraction of bases unpaired at equilibrium"], color=colors_1[7])
axins.plot(df9["Temperature(C)"], df9["Fraction of bases unpaired at equilibrium"], color=colors_1[8])
axins.plot(df10["Temperature(C)"], df10["Fraction of bases unpaired at equilibrium"], color=colors_1[9])
axins.plot(df11["Temperature(C)"], df11["Fraction of bases unpaired at equilibrium"], color=colors_1[10])
axins.plot(df12["Temperature(C)"], df12["Fraction of bases unpaired at equilibrium"], color=colors_1[11])
axins.plot(df13["Temperature(C)"], df13["Fraction of bases unpaired at equilibrium"], color=colors_1[12])
axins.set_xlim(59, 67)
axins.set_xlim(59, 67)
axins.set_ylim(0.46, 0.54)
axins.set_xticks(np.arange(59, 68, 4))
axins.set_yticks(np.arange(0.46, 0.56, 0.04))
axins.grid(True)

# Cambiar el tamaño de la fuente de los ticks del gráfico de zoom
axins.tick_params(axis='both', which='major', labelsize=10)
# Configuración del tamaño de fuente de los ticks del primer subplot
axs[0,1].tick_params(axis='both', which='major', labelsize=10)  # Aquí ajusta el tamaño de fuente según tus preferencias

# Añadir guías al gráfico principal para mostrar el área de zoom
axs[0,1].indicate_inset_zoom(axins, edgecolor="black")
#ax.plot(df_tm_info_1['con_wo_units'], df_tm_info_1['Tm'], marker='o', linestyle='-')
axs[0,1].set_ylim(0, 1)
axs[0,1].set_yticks(np.arange(0, 1.1, 0.1))
axs[0,1].set_xlabel('Temperature ($^{o}$C)', fontsize=12)

axs[0,1].tick_params(axis='y', which='both', left=True, right=True, labelleft=False, labelright=False)

#axs[0,1].set_ylabel('Fraction unbounded', fontsize=12)
axs[0, 1].text(80, 0.67, '100 mM NaCl',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotación en la segunda línea
axs[0, 1].text(80, 0.63, '1 $\mu$M',  
               fontsize=10, color='black',  
               ha='left', va='top',
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))



#colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2','#5e4fa2']
#colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2', '#2c1e50']
#colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd','#9673b9', '#5e4fa2', '#2c1e50']

colors = [
"#313695", "#416aaf", "#659bc8", "#a3d3e6", "#d1ecf4", "#f2fad6", "#fee99d", "#fdc374", "#f99153", "#e34933", "#c01a27", "#c6171b", "#7f0d0b"
]

axs[1,1].scatter(df_tm_info_1['nucleotides'], df_tm_info_1['Tm'], marker='o', color=colors, s=150, edgecolor='black')

# Añadir etiquetas de datos (Tm) a cada punto con posición controlada
for i, txt in enumerate(df_tm_info_1['Tm']):
    x = df_tm_info_1['nucleotides'][i]
    y = df_tm_info_1['Tm'][i]
    # Ajustar las coordenadas x e y según sea necesario
    axs[1,1].text(x + 0.03, y + 0.25, f'{txt:.1f}', fontsize=9, ha='center', va='bottom')
    
axs[1,1].scatter(df_tm_info_1['nucleotides'], df_tm_info_1['Tm'], marker='o', color=colors, s=150, edgecolor='black')
#axs[1,1].set_title('Tm vs Concentración')
axs[1,1].set_xlabel('Length sticky end (no. nucleotides)',fontsize=12)
#axs[1,1].set_ylabel('Melting temperature ($^{o}$C)',fontsize=12)
axs[1,1].set_yticks(np.arange(62, 72, 1))
axs[1,1].set_xticks(np.arange(0, 13, 1))  # Especificar los ticks del eje y
axs[1,1].tick_params(axis='both', which='major', labelsize=10)
#axs[1,1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
axs[1,1].tick_params(axis='y', which='both', left=True, right=True, labelleft=False, labelright=False)
#axs[1,1].set_ylim(66.5, 68)

axs[1, 1].text(0.0, 70.8, '100 mM NaCl',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotación en la segunda línea
axs[1, 1].text(0.0, 70.4, '1 $\mu$M',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

"""
# Anotaciones para cada subplot
annotations = ['a)', 'b)', 'c)', 'd)']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
for pos, annotation in zip(positions, annotations):
    axs[pos].text(-0.056, 1.05, annotation, transform=axs[pos].transAxes, 
                  fontsize=12, fontweight='bold', va='top', ha='right')
"""
# Anotaciones y posiciones
annotations = ['a)', 'b)', 'c)', 'd)']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
text_positions = [(-0.1, 1.05), (-0.05, 1.05), (-0.1, 1.05), (-0.05, 1.05)]  # Ajusta estas coordenadas según sea necesario

# Añadir anotaciones a los subplots
for pos, annotation, text_pos in zip(positions, annotations, text_positions):
    axs[pos].text(text_pos[0], text_pos[1], annotation, transform=axs[pos].transAxes, 
                  fontsize=12, fontweight='bold', va='top', ha='right')

# Crear una barra de colores horizontal dentro del segundo subplot
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=np.arange(len(df_tm_info_1) + 1) - 0.5, ncolors=len(df_tm_info_1))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Ajustar el tamaño y la posición de la barra de colores
cbar_ax = fig.add_axes([0.533, 0.4, 0.33, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=np.arange(len(df_tm_info_1)))

# Colocar una "L" delante de cada etiqueta de los ticks
cbar.ax.set_xticklabels(['L' + str(label) for label in df_tm_info_1.index])

# Etiqueta de la barra de colores
cbar.set_label('Linkers')

plt.savefig('nupack_con_scr_paper.png', dpi=300)

#plt.tight_layout()
plt.show()
