import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter
from matplotlib.colors import Normalize
import scienceplots

# Función para formatear el eje x en escala logarítmica
def log_format(x, pos):
    return f'$10^{{{int(x)}}}$'

# Estilo de plot
plt.style.use(['science', 'no-latex', 'bright'])

# Directorio base
base_directory = '/Users/alejandrosoto/Documents/KIT/last_task_ale/explicit_server/SALT_0.1'

# Lista de carpetas de temperatura
temp_folders = [
    'TEMP_14', 'TEMP_16', 'TEMP_18', 'TEMP_20', 'TEMP_22', 'TEMP_24', 'TEMP_26', 'TEMP_28', 'TEMP_30', 
    'TEMP_32', 'TEMP_34', 'TEMP_36', 'TEMP_38', 'TEMP_40', 'TEMP_42', 'TEMP_44', 'TEMP_46', 'TEMP_48', 
    'TEMP_50', 'TEMP_52', 'TEMP_54', 'TEMP_56', 'TEMP_58', 'TEMP_60', 'TEMP_62', 'TEMP_64', 'TEMP_66', 
    'TEMP_68', 'TEMP_70', 'TEMP_72', 'TEMP_74', 'TEMP_76', 'TEMP_78', 'TEMP_80', 'TEMP_82', 'TEMP_84', 
    'TEMP_86'
]

# Diccionarios para almacenar las medianas y datos de energía
median_energy_dict = {}
energy_data_dict = {}

# Procesar cada carpeta de temperatura y archivo
for temp_folder in temp_folders:
    run_folder_path = os.path.join(base_directory, temp_folder, 'RUN_0')
    file_path = os.path.join(run_folder_path, 'energy_file_md.txt')

    if os.path.exists(file_path):
        # Leer el archivo de energía con pandas
        df = pd.read_csv(file_path, sep='\s+', names=['Time', 'Energy1', 'Energy2', 'Energy3'])

        # Almacenar los datos de energía para cada temperatura
        energy_data_dict[temp_folder] = df

        # Calcular y almacenar la mediana de la primera columna (Potencial)
        median_energy = df['Energy1'].median()
        median_energy_dict[temp_folder] = median_energy
        print(f'Median Potential Energy for {temp_folder}: {median_energy}')
    else:
        print(f'File not found: {file_path}')

# Crear un DataFrame a partir del diccionario de medianas
median_energy_df = pd.DataFrame(list(median_energy_dict.items()), columns=['Temperature', 'MedianPotentialEnergy'])

# Convertir la columna "Temperature" a números
median_energy_df['Temperature'] = median_energy_df['Temperature'].str.extract('(\d+)').astype(int)

# Ordenar el DataFrame por temperatura
median_energy_df.sort_values('Temperature', inplace=True)

# Lista de colores
colors = [
"#313695", "#36479e", "#3c59a6", "#416aaf", "#4b7db8", "#588cc0", "#659bc8", "#74add1", "#83b9d8", 
"#a3d3e6", "#b2ddeb", "#c1e4ef", "#d1ecf4", "#e0f3f8", "#e9f6e8", "#f2fad6", "#fbfdc7", "#fffbb9", 
"#fee99d", "#fee090", "#fed283", "#fdc374", "#fdb567", "#fca55d", "#f99153", "#f67f4b", "#f46d43", 
"#eb5a3a", "#db382b", "#ce2827", "#c01a27", "#b30d26", "#d73027", "#c6171b", "#a50f15", "#911a11", "#7f0d0b"
]

# Crear un diccionario que asocia temperaturas con colores
temp_to_color = {int(folder.split('_')[1]): color for folder, color in zip(temp_folders, colors)}

# Configurar la figura con subplots usando GridSpec
fig, axs = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [1, 1]})

norm_value = 1  # Reemplaza con el valor de normalización apropiado

for i, (temp_folder, df) in enumerate(energy_data_dict.items()):
    # Solo toma el último 10% de los datos
    last_10_percent = df.iloc[-df.shape[0]//1:].reset_index(drop=True)
    
    # Normalizar los datos de 'Energy1'
    normalized_energy = last_10_percent['Energy1'] / norm_value
    
    # Graficar usando semilogx
    color = colors[i % len(colors)]  # Asegurarse de que el índice esté dentro del rango de colores
    axs[0].semilogx(np.arange(0, last_10_percent.shape[0]), normalized_energy, label=temp_folder, color=color)

axs[0].set_xscale('log')
axs[0].set_xlabel('Time (sim units)', fontsize=12)
axs[0].set_ylabel('Potential Energy ($k_{B}T$)', fontsize=12)
axs[0].set_ylim(bottom=-1.4, top=-0.3)
axs[0].xaxis.set_minor_formatter(NullFormatter())
axs[0].set_xlim(10**0, 10**5.5)
axs[0].text(1.2*10**4, -0.38, 'L3',  
           fontsize=10, color='black',  
           ha='left', va='top', 
           bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

# Anotaciones adicionales
axs[0].text(1.18*10**4, -0.41, '$1\\times 10^{9} [steps] $',  
           fontsize=10, color='black',  
           ha='left', va='top', 
           bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))

axs[0].text(1.2*10**4, -0.455, '1 $\mu$M',  
           fontsize=10, color='black',  
           ha='left', va='top', 
           bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
axs[0].text(1.2*10**4, -0.5, '100 mM NaCl',  
           fontsize=10, color='black',  
           ha='left', va='top', 
           bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.1))
# Crear la barra de colores
cmap = LinearSegmentedColormap.from_list('temp_colormap', colors)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(temp_to_color.keys()), vmax=max(temp_to_color.keys())))
sm.set_array([])

# Añadir la barra de colores dentro del subplot
cb_ax = fig.add_axes([0.9, 0.22, 0.02, 0.6])  # Ajustar las coordenadas y el tamaño de la barra de colores
cb = plt.colorbar(sm, cax=cb_ax)
cb.set_label('Temperature ($ ^{o}$C)', fontsize=12)
cb.ax.tick_params(labelsize=12)

# Añadir sombra ajustable al final del gráfico
shade_start = 10000 * 0.1  # Punto de inicio de la sombra (en log(Time))
shade_end = 10000  # Punto final de la sombra (en log(Time))
axs[0].fill_between(last_10_percent['Time'], -1.5, 0.5, 
                 where=(last_10_percent['Time'] >= shade_start) & (last_10_percent['Time'] <= shade_end),
                 color='lightgrey', alpha=0.8)

# Ruta base donde están las carpetas TEMP_X
base_path = "/Users/alejandrosoto/Documents/KIT/last_task_ale/explicit_server/SALT_0.1"

# Temperaturas (extraídas de las carpetas TEMP_X)
Temps = np.arange(14, 86, 2)

# Define el valor de normalización numérico
norm_value = 25 * (28 + 14)

# Lista de colores
colors = [
    "#313695", "#36479e", "#3c59a6", "#416aaf", "#4b7db8", "#588cc0", "#659bc8",
    "#74add1", "#83b9d8", "#92c5de", "#a3d3e6", "#b2ddeb", "#c1e4ef", "#d1ecf4",
    "#e0f3f8", "#e9f6e8", "#f2fad6", "#fbfdc7", "#fffbb9", "#fff2ac", "#fee99d",
    "#fee090", "#fed283", "#fdc374", "#fdb567", "#fca55d", "#f99153", "#f67f4b",
    "#f46d43", "#eb5a3a", "#e34933", "#db382b", "#ce2827", "#c01a27", "#b30d26",
    "#d73027", "#c6171b", "#a50f15", "#911a11", "#7f0d0b"
]


def shuffled_sterr(_in):
    _chunk = 20
    N = _in.size // _chunk
    if N == 0:
        return 0
    out = 0
    for i in range(N):
        _ids = np.random.randint(0, high=_in.size, size=_chunk)
        out += np.std(_in[_ids])
    return np.sqrt(out/N) / np.sqrt(N)

def extrapolate_bulk(fraction):
    phi = fraction / (1 - fraction)
    if phi == 0:
        return 0
    else:
        x = 1 + 1 / (2 * phi)
        return x - np.sqrt(np.fabs(x*x - 1))

def extrapolate_bulk_array(x):
    temp = x / (1 - x)
    return (1 + 1 / (2 * temp)) - np.sqrt((1 + 1. / (2 * temp))**2 - 1)

# Usar un contexto local para rcParams
with plt.rc_context({
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.titlesize': 12,
}):
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))
    norm = Normalize(vmin=min(Temps), vmax=max(Temps))


    for i, T in enumerate(Temps):
        temp_path = os.path.join(base_path, f"TEMP_{T}")
        store_data = []

        for r in range(30):
            run_path = os.path.join(temp_path, f"RUN_{r}")
            hb_file = os.path.join(run_path, "hb_list.dat")

            if os.path.exists(hb_file):
                data = np.loadtxt(hb_file)
                store_data.extend(data[-data.size//20:] / norm_value)
                axs[1].semilogx(np.arange(0, data.size//10+1), data[-data.size//10:] / norm_value, color=colors[i % len(colors)])

        store_data = np.asarray(store_data)


axs[1].set_xlabel("Time (sim units)", fontsize=12)
#axs[1].tick_params(left=False, labelleft=False)
axs[1].set_ylabel("Fraction unbounded", fontsize=12)
axs[1].set_yticks(np.arange(0, 1.2, 0.2))
    # Ajustar los ticks logarítmicos y sumar 1 a cada uno
    
    #
    #
  # Configurar los límites del eje x
axs[1].set_xlim(10**-1, 10**4.5)

# Definir los ticks como antes
log_ticks = np.array([10**-1, 10**0, 10**1, 10**2, 10**3, 10**4])

# Aplicar los ticks al eje x
axs[1].set_xticks(log_ticks)

# Cambiar las etiquetas de los ticks para que se muestren como si empezaran en 10^0
axs[1].set_xticklabels([f"$10^{{{int(t)}}}$" for t in range(0, 6)])
    # Sombrear un rango específico
shade_start = 10**2  # Punto de inicio de la sombra (en log(Time))
shade_end = 10**3  # Punto final de la sombra (en log(Time))
axs[1].axvspan(shade_start+1, shade_end+1, color='lightgrey', alpha=0.5)
    # Anotaciones y posiciones
# Añadir anotaciones a los subplots
annotations = ['a)', 'b)']
positions = [0, 1]
text_positions = [(-0.15, 1.05), (-0.15, 1.05)] 

for ann, pos, text_pos in zip(annotations, positions, text_positions):
    axs[pos].text(text_pos[0], text_pos[1], ann, 
                  transform=axs[pos].transAxes, 
                  fontsize=14, fontweight='bold', 
                  ha='center', va='center', 
                  bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.1))

# Ajustar el diseño y mostrar el gráfico
plt.tight_layout()
plt.savefig('combined_plots_with_annotations.png', dpi=400)
plt.show()
 
# Ajustar el diseño y mostrar el gráfico
plt.tight_layout()
plt.savefig('combined_plots_with_specific_temp.png', dpi=400)
plt.show()
