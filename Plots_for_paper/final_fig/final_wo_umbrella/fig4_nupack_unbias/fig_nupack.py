#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:46:18 2025

@author: alejandrosoto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:14:46 2024

@author: alejandrosoto
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scienceplots
import matplotlib.colors as mcolors
import matplotlib.cm as cm


plt.style.use(['science', 'no-latex', 'bright'])

# Ruta base donde se encuentran los archivos, incluyendo Y+L0 y Y+L11
base_paths = {
    "L_-4": "./L_-4/hb",
    "L_0": "./L_0/hb",
    "L_1": "./L_1/hb",
    "L_2": "./L_2/hb",
    "L_3": "./L_3/hb",
    "L_4": "./L_4/hb",
    "L_7": "./L_7/hb",
    "L_11": "./L_11/hb",
    "Y+L11": "./Y+L11/hb",  # Nueva carpeta Y+L11
    "Y+L0": "./Y+L0/hb",  # Nueva carpeta Y+L0
    "Y": "./Y_0/hb"
    
    
}

temps = [14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92]
norm = 25 * (28 + 14)  # Normalización

def shuffled_sterr(_in):
    _chunk = 20
    N = _in.size // _chunk
    if N == 0:
        return 0
    out = 0
    for i in range(N):
        _ids = np.random.randint(0, high=_in.size, size=_chunk)
        out += np.std(_in[_ids])
    return np.sqrt(out / N) / np.sqrt(N)

def extrapolate_bulk_array(x):
    temp = x / (1 - x)
    return (1 + 1 / (2 * temp)) - np.sqrt((1 + 1. / (2 * temp)) ** 2 - 1)

def find_temp_for_fraction(temps, fractions, target_fraction=0.5):
    """Interpolar linealmente para encontrar la temperatura en la cual la fracción alcanza un valor dado."""
    f = interp1d(fractions, temps, kind='linear', fill_value="extrapolate")
    return f(target_fraction)

# Configurar estilo y fuentes
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8,
})

# Tamaño de los marcadores
marker_size = 10

# Colores especificados
colors = [
"#313695", "#416aaf", "#74add1", "#c1e4ef", "#e9f6e8", "#fee99d", "#fdb567", "#f46d43", "#db382b", "#c6171b", "#7f0d0b"
][::-1]

# Crear la figura con dos subplots en horizontal
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Aquí cambiamos el nombre de 'ax' y 'ax2' a 'axs[0]' y 'axs[0,1]'

# Lista para almacenar los valores de temperatura a fracción 0.5
temp_at_05_values = []
temp_at_05_errors = []
labels = []
norm = None

# Leer datos y graficar para cada carpeta
for i, (label, base_path) in enumerate(base_paths.items()):
    ave_data = []
    std_data = []
    # Leer el archivo para la temperatura 14 y encontrar el valor máximo
    hb_file_14 = os.path.join(base_path, "hb_list.14.dat")
    if os.path.exists(hb_file_14):
        data_14 = np.loadtxt(hb_file_14)
        norm = np.max(data_14)  # Asignar norm como el máximo del primer archivo
    else:
        print(f"Advertencia: El archivo {hb_file_14} no existe.")

    for temp in temps:
        hb_file = os.path.join(base_path, f"hb_list.{temp}.dat")
        
        if os.path.exists(hb_file):
            # Leer el archivo .dat
            data = np.loadtxt(hb_file)
            
            # Convertir los datos a valores numéricos
            data = np.nan_to_num(data, nan=0.0)
            
            # Seleccionar la última décima parte de los datos para calcular los promedios
            store_data = data[-data.size // 10:] / norm if norm != 0 else data[-data.size // 10:]
            
            # Calcular promedio y desviación estándar
            ave_data.append(np.average(store_data))
            std_data.append(shuffled_sterr(store_data))
        else:
            ave_data.append(np.nan)
            std_data.append(np.nan)

    # Asegurar que los datos sean numéricos y manejar NaNs
    ave_data = np.array(ave_data)
    std_data = np.array(std_data)
    
    # Convertir cualquier valor no numérico o NaN a 0
    ave_data = np.nan_to_num(ave_data, nan=0.0)
    std_data = np.nan_to_num(std_data, nan=0.0)

    # Extrapolar y graficar los resultados
    extr = extrapolate_bulk_array(ave_data)

    color = colors[i % len(colors)]  # Usar colores en bucle

    # Reemplazar el guion bajo en la leyenda
    clean_label = label.replace("_", "")
    """
    # Graficar con líneas que conectan los puntos
    axs[0].errorbar(temps, extr, yerr=std_data, 
                fmt='o-',  # Conectar puntos con líneas rectas
                color=color,  # Color de los datos
                markeredgecolor='k',  # Color del borde del marcador
                capsize=5,  # Tamaño de las "caps" en los errores
                label=clean_label)

    # Interpolar para encontrar la temperatura a la cual la fracción alcanza 0.5
    temp_at_05 = find_temp_for_fraction(temps, extr, target_fraction=0.5)
    temp_at_05_values.append(temp_at_05)
    temp_at_05_errors.append(np.std(std_data))  # Asumir std_data como error (puedes ajustar esto)
    labels.append(label)  # Mantener las etiquetas originales
    """





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

# Crear un DataFrame con la información de Tm
    df_tm_info_1 = pd.DataFrame(tm_info_1, columns=['molecule', 'Tm'])

# Separar y procesar el nombre de las moléculas
    df_tm_info_1[['molecule_clean', 'type_of_file']] = df_tm_info_1['molecule'].str.split('.c', expand=True)
    df_tm_info_1.drop(columns=['molecule', 'type_of_file'], inplace=True)

# Procesar los archivos que comienzan con "L"
    df_tm_info_1[['type_of_mol', 'molecule_clean_id']] = df_tm_info_1['molecule_clean'].str.extract(r'([A-Za-z]+)([0-9\-]+)')

# Asignar valores personalizados para "Y0", "Y1" y "Y12"
    custom_nucleotide_values = {'Y0': 12, 'Y1': 11, 'Y12': 0}
    df_tm_info_1['nucleotides'] = df_tm_info_1.apply(
    lambda row: custom_nucleotide_values[row['molecule_clean']] if row['molecule_clean'] in custom_nucleotide_values else None,
    axis=1
    )

# Para los demás ("L"), asignar valores basados en molecule_clean_id
    mask_l = df_tm_info_1['type_of_mol'] == 'L'
    df_tm_info_1.loc[mask_l, 'molecule_clean_id'] = df_tm_info_1.loc[mask_l, 'molecule_clean_id'].astype(float)
    df_tm_info_1.loc[mask_l, 'nucleotides'] = range(16, 16 - mask_l.sum(), -1)

# Resetear índice y mostrar el DataFrame final
    df_tm_info_1['molecule_clean_id'] = df_tm_info_1['molecule_clean_id'].astype(float)
    df_tm_info_1 = df_tm_info_1.sort_values(by='nucleotides', ascending=False).reset_index(drop=True)
    

    
    # Eliminar las columnas intermedias que ya no necesitamos
    #df_tm_info_1.drop(columns=['Concentracion_ext', 'ext'], inplace=True)

    print(df_tm_info_1)
# Graficar cada perfil de desaparición de bases por separado
# Graficar la primera curva de desaparición de bases
# Graficar la primera curva de desaparición de bases

colors_1 = [
"#313695", "#3c59a6", "#588cc0", "#74add1", "#a3d3e6", "#c1e4ef", "#e9f6e8", "#fbfdc7", "#fee99d", "#fed283", "#fca55d", "#f67f4b", "#e34933", "#ce2827", "#d73027", "#a50f15", "#7f0d0b"
]


# Leer y graficar los datos, añadiendo la leyenda a cada gráfico
df1 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[15]))
axs[0].plot(df1["Temperature(C)"], 1-df1["Fraction of bases unpaired at equilibrium"], color = colors_1[0], label = df_tm_info_1['molecule_clean'].iloc[4])

#df2 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[14]))
#axs[0].plot(df2["Temperature(C)"], 1-df2["Fraction of bases unpaired at equilibrium"], color = colors_1[1], label = df_tm_info_1['molecule_clean'].iloc[6])

#df3 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[3]))
#axs[0].plot(df3["Temperature(C)"], 1-df3["Fraction of bases unpaired at equilibrium"], color = colors_1[2], label = df_tm_info_1['molecule_clean'].iloc[16])

df4 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[0]))
axs[0].plot(df4["Temperature(C)"], 1-df4["Fraction of bases unpaired at equilibrium"], color = colors_1[3], label = df_tm_info_1['molecule_clean'].iloc[0])

df5 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[13]))
axs[0].plot(df5["Temperature(C)"], 1-df5["Fraction of bases unpaired at equilibrium"], color = colors_1[4], label = df_tm_info_1['molecule_clean'].iloc[14])

df6 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[16]))
axs[0].plot(df6["Temperature(C)"], 1-df5["Fraction of bases unpaired at equilibrium"], color = colors_1[5], label = df_tm_info_1['molecule_clean'].iloc[15])

df7 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[12]))
axs[0].plot(df7["Temperature(C)"], 1-df7["Fraction of bases unpaired at equilibrium"], color = colors_1[6], label = df_tm_info_1['molecule_clean'].iloc[13])

df8 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[11]))
axs[0].plot(df8["Temperature(C)"], 1-df8["Fraction of bases unpaired at equilibrium"], color = colors_1[7], label = df_tm_info_1['molecule_clean'].iloc[12])

df9 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[8]))
axs[0].plot(df9["Temperature(C)"], 1-df9["Fraction of bases unpaired at equilibrium"], color = colors_1[8], label = df_tm_info_1['molecule_clean'].iloc[9])

df10 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[7]))
axs[0].plot(df10["Temperature(C)"], 1-df10["Fraction of bases unpaired at equilibrium"], color = colors_1[9], label = df_tm_info_1['molecule_clean'].iloc[8])

df11 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[9]))
axs[0].plot(df11["Temperature(C)"], 1-df11["Fraction of bases unpaired at equilibrium"], color = colors_1[10], label = df_tm_info_1['molecule_clean'].iloc[10])

df12 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[10]))
axs[0].plot(df12["Temperature(C)"], 1-df12["Fraction of bases unpaired at equilibrium"], color = colors_1[11], label = df_tm_info_1['molecule_clean'].iloc[11])

df13 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[2]))
axs[0].plot(df13["Temperature(C)"], 1-df13["Fraction of bases unpaired at equilibrium"], color = colors_1[12], label = df_tm_info_1['molecule_clean'].iloc[2])

df14 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[1]))
axs[0].plot(df14["Temperature(C)"], 1-df14["Fraction of bases unpaired at equilibrium"], color = colors_1[13], label = df_tm_info_1['molecule_clean'].iloc[1])

df15 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[5]))
axs[0].plot(df15["Temperature(C)"], 1-df15["Fraction of bases unpaired at equilibrium"], color = colors_1[14], label = df_tm_info_1['molecule_clean'].iloc[5])

df16 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[6]))
axs[0].plot(df16["Temperature(C)"], 1-df16["Fraction of bases unpaired at equilibrium"], color = colors_1[15], label = df_tm_info_1['molecule_clean'].iloc[7])

df17 = pd.read_csv(os.path.join(directory_bases_profiles_1, archivos_bases_profiles_1[4]))
axs[0].plot(df17["Temperature(C)"], 1-df17["Fraction of bases unpaired at equilibrium"], color = colors_1[16], label = df_tm_info_1['molecule_clean'].iloc[3])
# Añadir leyenda con dos columnas
axs[0].legend(loc='lower left', fontsize=10, ncol=2)

axs[0].text(-0.04, 1.07, 'a)', transform=axs[0].transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

# Configuración del zoom
axins = axs[0].inset_axes([0.7, 0.58, 0.25, 0.25])  # (x0, y0, width, height) del panel secundario
# Aplicar los colores a los plots
axins.plot(df1["Temperature(C)"], df1["Fraction of bases unpaired at equilibrium"], color=colors_1[0])
#axins.plot(df2["Temperature(C)"], df2["Fraction of bases unpaired at equilibrium"], color=colors_1[1])
#axins.plot(df3["Temperature(C)"], df3["Fraction of bases unpaired at equilibrium"], color=colors_1[2])
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
axins.plot(df14["Temperature(C)"], df14["Fraction of bases unpaired at equilibrium"], color=colors_1[13])
axins.plot(df15["Temperature(C)"], df15["Fraction of bases unpaired at equilibrium"], color=colors_1[14])
axins.plot(df16["Temperature(C)"], df16["Fraction of bases unpaired at equilibrium"], color=colors_1[15])
axins.plot(df17["Temperature(C)"], df17["Fraction of bases unpaired at equilibrium"], color=colors_1[16])


axins.set_xlim(52, 65)

axins.set_ylim(0.46, 0.54)
axins.set_xticks(np.arange(52, 73, 8))
axins.set_yticks(np.arange(0.46, 0.56, 0.04))
axs[0].set_ylabel("Fraction unbounded", fontsize=12)
axins.grid(True)

# Cambiar el tamaño de la fuente de los ticks del gráfico de zoom
axins.tick_params(axis='both', which='major', labelsize=10)
# Configuración del tamaño de fuente de los ticks del primer subplot
axs[0].tick_params(axis='both', which='major', labelsize=10)  # Aquí ajusta el tamaño de fuente según tus preferencias

# Añadir guías al gráfico principal para mostrar el área de zoom
axs[0].indicate_inset_zoom(axins, edgecolor="black")
#ax.plot(df_tm_info_1['con_wo_units'], df_tm_info_1['Tm'], marker='o', linestyle='-')
axs[0].set_ylim(0, 1)
axs[0].set_yticks(np.arange(0, 1.2, 0.2))
axs[0].set_xlabel('Temperature ($^{o}$C)', fontsize=12)
axs[0].axhline(y=0.5, color='black', alpha=0.4, linestyle='--', linewidth=1, xmin=0.47, xmax=1)
#axs[0].tick_params(axis='y', which='both', left=True, right=True, labelleft=False, labelright=False)

#axs[0].set_ylabel('Fraction unbounded', fontsize=12)
axs[0].text(80, 0.97, '100 mM NaCl',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotación en la segunda línea
axs[0].text(80, 0.93, '1 $\mu$M',  
               fontsize=10, color='black',  
               ha='left', va='top',
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))




#colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2','#5e4fa2']
#colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2', '#2c1e50']
#colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd','#9673b9', '#5e4fa2', '#2c1e50']

# Ordenar el DataFrame por la columna Tm
df_tm_info_1 = df_tm_info_1.sort_values(by='Tm', ascending=True).reset_index(drop=True)

# Función para excluir filas específicas por índice o valores
def excluir_filas(df, excluir_indices=None, excluir_valores=None, columna=None):
    if excluir_indices:
        df = df.drop(excluir_indices, axis=0).reset_index(drop=True)
    if excluir_valores and columna:
        df = df[~df[columna].isin(excluir_valores)].reset_index(drop=True)
    return df

# Definir los índices o valores a excluir
indices_a_excluir = [0, 1]  # Índices a excluir
#valores_a_excluir = [68.5]  # Valores a excluir en la columna Tm

# Aplicar exclusión de filas
df_tm_info_1 = excluir_filas(df_tm_info_1, excluir_indices=indices_a_excluir)
#df_tm_info_1 = excluir_filas(df_tm_info_1, excluir_valores=valores_a_excluir, columna='Tm')

# Definir colores (gradiente inverso)
colors = [
    "#313695", "#3c59a6", "#588cc0", "#74add1", "#a3d3e6", "#c1e4ef",
    "#e9f6e8", "#fbfdc7", "#fee99d", "#fed283", "#fca55d", "#f67f4b",
    "#e34933", "#ce2827", "#d73027", "#a50f15", "#7f0d0b"
][::1]

# Asignar colores a los puntos según el índice de orden
assigned_colors = [colors[i % len(colors)] for i in range(len(df_tm_info_1))]

# Crear el scatter plot con colores asignados
axs[1].scatter(
    df_tm_info_1['nucleotides'], df_tm_info_1['Tm'],
    marker='o', color=assigned_colors, s=150, edgecolor='black'
)

# Añadir etiquetas de datos (Tm) a cada punto con posición controlada
for i, txt in enumerate(df_tm_info_1['Tm']):
    x = df_tm_info_1['nucleotides'][i]
    y = df_tm_info_1['Tm'][i]
    axs[1].text(x + 0.03, y + 0.39, f'{txt:.1f}', fontsize=9, ha='center', va='bottom')

# Configurar los ejes y apariencia
axs[1].set_xlabel('Length sticky end (no. nucleotides)', fontsize=12)
axs[1].set_xticks(np.arange(0, 17, 1))  # Especificar los ticks del eje x
axs[1].tick_params(axis='both', which='major', labelsize=10)
axs[1].set_ylim(50, 70)
axs[1].set_ylabel('Melting temperature $(^{o}C)$', fontsize=12)
axs[1].text(-0.04, 1.07, 'b)', transform=axs[1].transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
#axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
#axs[1].tick_params(axis='y', which='both', left=True, right=True, labelleft=False, labelright=False)
#axs[1].set_ylim(66.5, 68)
# Crear una barra de colores horizontal dentro del segundo subplot
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=np.arange(len(df_tm_info_1) + 1) - 0.5, ncolors=len(df_tm_info_1))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Ajustar el tamaño y la posición de la barra de colores
cbar_ax = fig.add_axes([0.59, 0.22, 0.38, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=np.arange(len(df_tm_info_1['molecule_clean'])))
cbar.ax.set_xticklabels(df_tm_info_1['molecule_clean'])  # Usar las etiquetas limpias
# Controlar el tamaño del texto de las etiquetas de la barra de colores
cbar.ax.tick_params(labelsize=7)  # Ajusta el tamaño de las etiquetas (puedes cambiar el número)

cbar.set_label('Molecules')


plt.tight_layout()
plt.savefig('exp_comp.png', dpi=400)
plt.show()
