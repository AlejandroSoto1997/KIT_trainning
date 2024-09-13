import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex', 'bright'])

# Cargar los datos desde el archivo CSV
df_simulated = pd.read_csv("sim_output_umbrella_matthew_sample.csv")
df_experimental = pd.read_excel("melting_curves_Y_L0_L1.xlsx")
df_experimental_1 = pd.read_csv("Export_Data_AsyaL1_try2.csv")
fran_1 = pd.read_csv("fran_20_linker_l2_0.001.csv")

fran_1['Fraction_unf'] = 1 - fran_1['Fraction']
fran_x = fran_1['Temperature']
fran_y = fran_1['Fraction_unf']

# Limpiar los datos de df_experimental_1
df_experimental_1.columns = df_experimental_1.iloc[0]  # Establecer la primera fila como nombres de columnas
df_experimental_1 = df_experimental_1.drop(index=0).reset_index(drop=True)  # Eliminar la primera fila y restablecer los índices

# Convertir a valores numéricos
df_experimental_1 = df_experimental_1.apply(pd.to_numeric, errors='coerce')

# Cambiar el nombre de la columna 'Abs' a 'Abs_h'
df_experimental_1.columns.values[1] = 'Abs_h'
df_experimental_1.columns.values[0] = 'Temp_h'

# Calcular el mínimo y máximo de la columna 'Abs_h'
min_val = df_experimental_1['Abs_h'].min()
max_val = df_experimental_1['Abs_h'].max()

# Normalizar la columna 'Abs_h'
df_experimental_1['Abs_h_norm'] = (df_experimental_1['Abs_h'] - min_val) / (max_val - min_val)
temp_exp1 = df_experimental_1['Temp_h']
abs_norm1 = df_experimental_1['Abs_h_norm']

# Extraer las columnas de interés de los datos simulados y experimentales
temp_simulated = df_simulated['Temperature']
inverted_finfs2_simulated = df_simulated['Inverted Finfs 2']
temp_experimental = df_experimental['T_L0100_h']
transmitance_fraction_experimental = df_experimental['A_L0100_h']

# Configurar estilo y fuentes
plt.rcParams.update({
    'font.size': 25,        # Tamaño de fuente general
    'font.family': 'serif',
    'axes.labelsize': 25,   # Tamaño de fuente de las etiquetas de los ejes
    'xtick.labelsize': 25,  # Tamaño de fuente de los números del eje x
    'ytick.labelsize': 25,  # Tamaño de fuente de los números del eje y
    'legend.fontsize': 25,  # Tamaño de fuente de la leyenda
})

# Tamaño de los marcadores
marker_size = 10

# Crear la figura con un tamaño personalizado
plt.figure(figsize=(14, 6))

# Graficar los datos con bordes negros
plt.plot(temp_simulated, inverted_finfs2_simulated, marker='o', linestyle='-', color='#bae4bc', label='Advanced sampling', markersize=marker_size, linewidth=2, markeredgewidth=1, markeredgecolor='k')
plt.plot(temp_experimental, transmitance_fraction_experimental, marker='o', linestyle='-', color='#7bccc4', label='Exp. Data', markersize=marker_size, linewidth=2, markeredgewidth=1, markeredgecolor='k')
#plt.plot(temp_exp1, abs_norm1, marker='o', linestyle='-', color='#1c9099', label='Exp. Data 2', markersize=marker_size, linewidth=2, markeredgewidth=1, markeredgecolor='k')
plt.plot(fran_x, fran_y, marker='o', linestyle='-', color='#43a2ca', label='Direct sampling', markersize=marker_size, linewidth=2, markeredgewidth=1, markeredgecolor='k')

plt.xlabel('Temperature $(^{o}C)$')
plt.ylabel('Absorbance @ 260 nm\nUnbounded Frac.')  # Salto de línea añadido aquí
plt.legend()
plt.grid(False)  # Eliminar el grid
plt.tight_layout()

# Guardar la imagen en un archivo PNG con alta resolución
plt.savefig('comparison_plot.png', dpi=300)

# Mostrar la gráfica
plt.show()
