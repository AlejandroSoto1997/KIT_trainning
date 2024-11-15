import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex', 'bright'])

# Definir los colores para cada linker
colors = {
    'L4': {'main': 'blue', 'shade': 'lightblue'},
    'L5': {'main': 'green', 'shade': 'lightgreen'},
    'L6': {'main': 'red', 'shade': 'lightcoral'}
}

# Cargar los datos desde el archivo CSV
df_YL1_L4_L5_L6 = pd.read_csv("YL1_L4_L5_L6.csv")

# Convertir a valores numéricos y eliminar cualquier fila de encabezado restante
df_YL1_L4_L5_L6 = df_YL1_L4_L5_L6.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

# Definir las columnas de interés para L4, L5 y L6
columns = {
    'L4': {'h': (8, 9), 'c': (10, 11)},
    'L5': {'h': (20, 21), 'c': (22, 23)},
    'L6': {'h': (24, 25), 'c': (26, 27)}
}

# Crear la figura
plt.figure(figsize=(12, 8))

for linker, col_indices in columns.items():
    # Datos de calentamiento
    temp_h_col, abs_h_col = df_YL1_L4_L5_L6.columns[col_indices['h'][0]], df_YL1_L4_L5_L6.columns[col_indices['h'][1]]
    df_h = df_YL1_L4_L5_L6[[temp_h_col, abs_h_col]].apply(pd.to_numeric, errors='coerce')

    # Datos de enfriamiento y volteo
    temp_c_col, abs_c_col = df_YL1_L4_L5_L6.columns[col_indices['c'][0]], df_YL1_L4_L5_L6.columns[col_indices['c'][1]]
    df_c = df_YL1_L4_L5_L6[[temp_c_col, abs_c_col]].apply(pd.to_numeric, errors='coerce')
    df_c = df_c.iloc[::-1].reset_index(drop=True)  # Voltear los datos de enfriamiento

    # Normalización
    min_h = df_h[abs_h_col].min()
    max_h = df_h[abs_h_col].max()
    min_c = df_c[abs_c_col].min()
    max_c = df_c[abs_c_col].max()

    df_h[f'Absorbance_Norm_{linker}_h'] = (df_h[abs_h_col] - min_h) / (max_h - min_h)
    df_c[f'Absorbance_Norm_{linker}_c'] = (df_c[abs_c_col] - min_c) / (max_c - min_c)

    # Graficar las curvas de calentamiento y enfriamiento
    plt.plot(df_h[temp_h_col], df_h[f'Absorbance_Norm_{linker}_h'], marker='o', linestyle='-', color=colors[linker]['main'], label=f'{linker} h')
    plt.plot(df_c[temp_c_col], df_c[f'Absorbance_Norm_{linker}_c'], marker='o', linestyle='-', color=colors[linker]['main'], label=f'{linker} c')

    # Llenar el área entre las curvas con una sombra suave
    plt.fill_between(df_h[temp_h_col], df_h[f'Absorbance_Norm_{linker}_h'], df_c[f'Absorbance_Norm_{linker}_c'], color=colors[linker]['shade'], alpha=0.3)

# Configurar la gráfica
plt.xlabel('Temperature (°C)')
plt.ylabel('Normalized Absorbance')
plt.legend()
plt.grid(False)
plt.tight_layout()

# Guardar la gráfica
plt.savefig('L4_L5_L6_heating_cooling_plot_colored_shaded.png', dpi=300)

# Mostrar la gráfica
plt.show()




    
#L4 h: 8 y 9
#L4 c: 10 y 11
#L5 h: 20 y 21
#L5 c: 22 y 23
#L6 c: 24 y 25
#L6 c: 26 y 27


"""



# Mostrar los nombres de las columnas para que puedas seleccionar manualmente
print("Columnas disponibles en el archivo:")
for idx, col in enumerate(df_YL1_L4_L5_L6.columns):
    print(f"{idx}: {col}")

# Solicitar al usuario que seleccione el índice de la columna de temperatura y de absorbancia
temp_col_idx = int(input("Selecciona el índice de la columna de temperatura: "))
abs_col_idx = int(input("Selecciona el índice de la columna de absorbancia: "))

# Obtener los nombres de las columnas seleccionadas
temp_col = df_YL1_L4_L5_L6.columns[temp_col_idx]
abs_col = df_YL1_L4_L5_L6.columns[abs_col_idx]

# Extraer las columnas seleccionadas
df_selected = df_YL1_L4_L5_L6[[temp_col, abs_col]]

# Convertir a valores numéricos
df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

# Graficar los datos seleccionados
plt.figure(figsize=(10, 6))
plt.plot(df_selected[temp_col], df_selected[abs_col], marker='o', linestyle='-', label=abs_col)

plt.xlabel('Temperature (°C)')
plt.ylabel('Absorbance (non-normalized)')
plt.legend()
plt.grid(False)
plt.tight_layout()

# Guardar la gráfica
plt.savefig('selected_plot.png', dpi=300)

# Mostrar la gráfica
plt.show()

"""


"""
# Cambiar el nombre de las columnas si es necesario
df_YL1_L4_L5_L6.columns.values[0] = 'Temperature'
df_YL1_L4_L5_L6.columns.values[1] = 'Absorbance'

# Calcular el mínimo y máximo de la columna 'Absorbance'
min_val = df_YL1_L4_L5_L6['Absorbance'].min()
max_val = df_YL1_L4_L5_L6['Absorbance'].max()

# Normalizar la columna 'Absorbance'
df_YL1_L4_L5_L6['Absorbance_Norm'] = (df_YL1_L4_L5_L6['Absorbance'] - min_val) / (max_val - min_val)
temp_exp = df_YL1_L4_L5_L6['Temperature']
abs_norm = df_YL1_L4_L5_L6['Absorbance_Norm']

# Hacer lo mismo con otros conjuntos de datos si es necesario
# ...

# Tamaño de los marcadores
marker_size = 10

# Crear la figura con un tamaño personalizado
plt.figure(figsize=(14, 6))

# Graficar los datos procesados
plt.plot(temp_exp, abs_norm, marker='o', linestyle='-', color='#bae4bc', label='YL1_L4_L5_L6', markersize=marker_size, linewidth=2, markeredgewidth=1, markeredgecolor='k')
# Puedes añadir más líneas con otros conjuntos de datos

plt.xlabel('Temperature $(^{o}C)$')
plt.ylabel('Normalized Absorbance')
plt.legend()
plt.grid(False)  # Eliminar el grid
plt.tight_layout()

# Guardar la imagen en un archivo PNG con alta resolución
plt.savefig('comparison_plot.png', dpi=300)

# Mostrar la gráfica
plt.show()
"""