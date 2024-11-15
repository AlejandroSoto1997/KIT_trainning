import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scienceplots
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.style.use(['science', 'no-latex', 'bright'])

# Definir los colores para cada linker
colors = [
    "#4575b4", "#91bfdb", "#e0f3f8", "#ffffbf", "#fee090", "#fc8d59", "#d73027"
]

linkers = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']

# Longitud del sticky end de los linkers
sticky_end_lengths = {
    'L0': 12,
    'L1': 11,
    'L2': 10,
    'L3': 9,
    'L4': 8,
    'L5': 7,
    'L6': 6
}

# Cargar los datos desde los archivos CSV
df_L0_L1 = pd.read_csv("L0_L1.csv")
df_L2_L3 = pd.read_csv("L2_L3.csv")
df_YL1_L4_L5_L6 = pd.read_csv("YL1_L4_L5_L6.csv")

# Convertir a valores numéricos y eliminar cualquier fila de encabezado restante
df_L0_L1 = df_L0_L1.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
df_L2_L3 = df_L2_L3.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
df_YL1_L4_L5_L6 = df_YL1_L4_L5_L6.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

# Definir las columnas de interés
columns_of_interest = {
    'L0': {'h': (0, 1), 'c': (2, 3)},
    'L1': {'h': (12, 13), 'c': (14, 15)},
    'L2': {'h': (0, 1), 'c': (2, 3)},
    'L3': {'h': (12, 13), 'c': (14, 15)},
    'L4': {'h': (8, 9), 'c': (10, 11)},
    'L5': {'h': (20, 21), 'c': (22, 23)},
    'L6': {'h': (24, 25), 'c': (26, 27)}
}

fusion_temperatures = []
error_bars = []

# Crear la figura y los subplots
fig, axs = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [1, 1]})

for i, (linker, col_indices) in enumerate(columns_of_interest.items()):
    # Datos de calentamiento
    if linker in ['L0', 'L1']:
        df_selected = df_L0_L1[[df_L0_L1.columns[col_indices['h'][0]], df_L0_L1.columns[col_indices['h'][1]]]]
    elif linker in ['L2', 'L3']:
        df_selected = df_L2_L3[[df_L2_L3.columns[col_indices['h'][0]], df_L2_L3.columns[col_indices['h'][1]]]]
    else:
        df_selected = df_YL1_L4_L5_L6[[df_YL1_L4_L5_L6.columns[col_indices['h'][0]], df_YL1_L4_L5_L6.columns[col_indices['h'][1]]]]

    temp_h_col, abs_h_col = df_selected.columns[0], df_selected.columns[1]
    df_h = df_selected.apply(pd.to_numeric, errors='coerce')
    
    # Datos de enfriamiento y volteo
    if linker in ['L0', 'L1']:
        df_selected_c = df_L0_L1[[df_L0_L1.columns[col_indices['c'][0]], df_L0_L1.columns[col_indices['c'][1]]]]
    elif linker in ['L2', 'L3']:
        df_selected_c = df_L2_L3[[df_L2_L3.columns[col_indices['c'][0]], df_L2_L3.columns[col_indices['c'][1]]]]
    else:
        df_selected_c = df_YL1_L4_L5_L6[[df_YL1_L4_L5_L6.columns[col_indices['c'][0]], df_YL1_L4_L5_L6.columns[col_indices['c'][1]]]]

    temp_c_col, abs_c_col = df_selected_c.columns[0], df_selected_c.columns[1]
    df_c = df_selected_c.apply(pd.to_numeric, errors='coerce')
    df_c = df_c.iloc[::-1].reset_index(drop=True)  # Voltear los datos de enfriamiento

    # Normalización
    min_h = df_h[abs_h_col].min()
    max_h = df_h[abs_h_col].max()
    min_c = df_c[abs_c_col].min()
    max_c = df_c[abs_c_col].max()

    df_h[f'Absorbance_Norm_{linker}_h'] = (df_h[abs_h_col] - min_h) / (max_h - min_h)
    df_c[f'Absorbance_Norm_{linker}_c'] = (df_c[abs_c_col] - min_c) / (max_c - min_c)

    # Calcular el promedio entre las curvas
    min_len = min(len(df_h[temp_h_col]), len(df_c[temp_c_col]))
    temp_range = df_h[temp_h_col].iloc[:min_len]
    avg_absorbance = (df_h[f'Absorbance_Norm_{linker}_h'].iloc[:min_len] + df_c[f'Absorbance_Norm_{linker}_c'].iloc[:min_len]) / 2

    # Interpolación lineal para encontrar la temperatura correspondiente a la fracción 0.5
    interp_func = interp1d(avg_absorbance, temp_range, bounds_error=False, fill_value='extrapolate')
    temperature_at_05 = interp_func(0.5)
    fusion_temperatures.append((sticky_end_lengths[linker], temperature_at_05))

    # Calcular el error como la diferencia en temperatura entre las curvas de calentamiento y enfriamiento cuando la fracción es 0.5
    temp_h_05 = interp_func(0.5)
    temp_c_05 = interp1d(df_c[f'Absorbance_Norm_{linker}_c'], df_c[temp_c_col], bounds_error=False, fill_value='extrapolate')(0.5)
    error_bars.append(abs(temp_h_05 - temp_c_05))

    # Graficar el promedio
    axs[0].plot(temp_range, avg_absorbance, linestyle='-', color=colors[i], label=f'{linker}: $T_{{m}}={temperature_at_05:.2f} ^{{\circ}}C$')
    axs[0].legend(loc='best')
    # Llenar el área entre las curvas de calentamiento y enfriamiento con una sombra suave
    axs[0].fill_between(df_h[temp_h_col], df_h[f'Absorbance_Norm_{linker}_h'], df_c[f'Absorbance_Norm_{linker}_c'], color=colors[i], alpha=0.2)

# Configurar el subplot principal
axs[0].set_xlabel('Temperature $(^{o}C)$')
axs[0].set_ylabel('Normalized absorbance @ 260 nm')
axs[0].text(25, 0.4, '100 mM NaCl',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotación en la segunda línea
axs[0].text(25, 0.36, r'$\approx 1 \ \mu\mathrm{M}$',  
               fontsize=10, color='black',  
               ha='left', va='top', 
               bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
axs[0].grid(False)

# Añadir el zoom en el primer subplot
axins = axs[0].inset_axes([0.72, 0.1, 0.24, 0.24])  # Ajusta estos valores según necesites

for i, (linker, col_indices) in enumerate(columns_of_interest.items()):
    # Datos de calentamiento
    if linker in ['L0', 'L1']:
        df_selected = df_L0_L1[[df_L0_L1.columns[col_indices['h'][0]], df_L0_L1.columns[col_indices['h'][1]]]]
    elif linker in ['L2', 'L3']:
        df_selected = df_L2_L3[[df_L2_L3.columns[col_indices['h'][0]], df_L2_L3.columns[col_indices['h'][1]]]]
    else:
        df_selected = df_YL1_L4_L5_L6[[df_YL1_L4_L5_L6.columns[col_indices['h'][0]], df_YL1_L4_L5_L6.columns[col_indices['h'][1]]]]

    temp_h_col, abs_h_col = df_selected.columns[0], df_selected.columns[1]
    df_h = df_selected.apply(pd.to_numeric, errors='coerce')
    
    # Datos de enfriamiento y volteo
    if linker in ['L0', 'L1']:
        df_selected_c = df_L0_L1[[df_L0_L1.columns[col_indices['c'][0]], df_L0_L1.columns[col_indices['c'][1]]]]
    elif linker in ['L2', 'L3']:
        df_selected_c = df_L2_L3[[df_L2_L3.columns[col_indices['c'][0]], df_L2_L3.columns[col_indices['c'][1]]]]
    else:
        df_selected_c = df_YL1_L4_L5_L6[[df_YL1_L4_L5_L6.columns[col_indices['c'][0]], df_YL1_L4_L5_L6.columns[col_indices['c'][1]]]]

    temp_c_col, abs_c_col = df_selected_c.columns[0], df_selected_c.columns[1]
    df_c = df_selected_c.apply(pd.to_numeric, errors='coerce')
    df_c = df_c.iloc[::-1].reset_index(drop=True)  # Voltear los datos de enfriamiento

    # Normalización
    min_h = df_h[abs_h_col].min()
    max_h = df_h[abs_h_col].max()
    min_c = df_c[abs_c_col].min()
    max_c = df_c[abs_c_col].max()

    df_h[f'Absorbance_Norm_{linker}_h'] = (df_h[abs_h_col] - min_h) / (max_h - min_h)
    df_c[f'Absorbance_Norm_{linker}_c'] = (df_c[abs_c_col] - min_c) / (max_c - min_c)

    # Calcular el promedio entre las curvas
    min_len = min(len(df_h[temp_h_col]), len(df_c[temp_c_col]))
    temp_range = df_h[temp_h_col].iloc[:min_len]
    avg_absorbance = (df_h[f'Absorbance_Norm_{linker}_h'].iloc[:min_len] + df_c[f'Absorbance_Norm_{linker}_c'].iloc[:min_len]) / 2

    # Interpolación lineal para encontrar la temperatura correspondiente a la fracción 0.5
    interp_func = interp1d(avg_absorbance, temp_range, bounds_error=False, fill_value='extrapolate')
    temperature_at_05 = interp_func(0.5)

    # Graficar el promedio en el zoom
    axins.plot(temp_range, avg_absorbance, linestyle='-', color=colors[i], alpha=0.7)
#axs[0].indicate_inset_zoom(axins)
axins.set_xlim(60, 68)
axins.set_xticks(np.arange(60, 72, 4))
axins.set_ylim(0.4, 0.6)
axins.set_yticks(np.arange(0.4, 0.7, 0.1))
axins.grid(True)
axs[0].indicate_inset_zoom(axins)

#

# Graficar el subplot de temperaturas de fusión vs longitud del sticky end con barras de error
for i, (length, temp) in enumerate(fusion_temperatures):
    axs[1].errorbar(length, temp, yerr=error_bars[i], fmt='o', color=colors[i], markeredgecolor='black', capsize=5)

axs[1].set_xlabel('Sticky end length (nucleotide)')
axs[1].set_ylabel('Melting temperature $(^{o}C)$')
axs[1].set_ylim(60,67)

axs[1].grid(False)

# Crear una barra de colores horizontal dentro del segundo subplot
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=np.arange(len(linkers) + 1) - 0.5, ncolors=len(linkers))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Ajustar el tamaño y la posición de la barra de colores
cbar_ax = fig.add_axes([0.72, 0.25, 0.25, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=np.arange(len(linkers)))
cbar.ax.set_xticklabels(linkers)
cbar.set_label('Linkers')

plt.tight_layout()
plt.savefig('exp_iliya.png', dpi=400)
plt.show()
