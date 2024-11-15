#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:10:19 2024

@author: alejandrosoto
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scienceplots

plt.style.use(['science', 'no-latex', 'bright'])

# Cargar los datos desde el archivo CSV
df_YL1_L4_L5_L6 = pd.read_csv("YL1_L4_L5_L6.csv")

# Convertir a valores numéricos y eliminar cualquier fila de encabezado restante
df_YL1_L4_L5_L6 = df_YL1_L4_L5_L6.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

# Definir las columnas de interés para L4
temp_h_col = df_YL1_L4_L5_L6.columns[8]
abs_h_col = df_YL1_L4_L5_L6.columns[9]
temp_c_col = df_YL1_L4_L5_L6.columns[10]
abs_c_col = df_YL1_L4_L5_L6.columns[11]

# Datos de calentamiento
df_h = df_YL1_L4_L5_L6[[temp_h_col, abs_h_col]].apply(pd.to_numeric, errors='coerce')

# Datos de enfriamiento y volteo
df_c = df_YL1_L4_L5_L6[[temp_c_col, abs_c_col]].apply(pd.to_numeric, errors='coerce')
df_c = df_c.iloc[::-1].reset_index(drop=True)  # Voltear los datos de enfriamiento

# Normalización
min_h = df_h[abs_h_col].min()
max_h = df_h[abs_h_col].max()
min_c = df_c[abs_c_col].min()
max_c = df_c[abs_c_col].max()

df_h['Absorbance_Norm_h'] = (df_h[abs_h_col] - min_h) / (max_h - min_h)
df_c['Absorbance_Norm_c'] = (df_c[abs_c_col] - min_c) / (max_c - min_c)

# Crear la figura
plt.figure(figsize=(10, 6))

# Calcular el promedio entre las curvas
min_len = min(len(df_h[temp_h_col]), len(df_c[temp_c_col]))
temp_range = df_h[temp_h_col].iloc[:min_len]
avg_absorbance = (df_h['Absorbance_Norm_h'].iloc[:min_len] + df_c['Absorbance_Norm_c'].iloc[:min_len]) / 2

# Interpolación lineal para encontrar la temperatura correspondiente a la fracción 0.5
interp_func = interp1d(avg_absorbance, temp_range, bounds_error=False, fill_value='extrapolate')
temperature_at_05 = interp_func(0.5)

# Graficar el promedio
plt.plot(temp_range, avg_absorbance, linestyle='--', color='black', label="L4: $T_{m}=$"f'{temperature_at_05:.2f}'"$^{o}C$")

# Llenar el área entre las curvas de calentamiento y enfriamiento con una sombra suave
plt.fill_between(df_h[temp_h_col], df_h['Absorbance_Norm_h'], df_c['Absorbance_Norm_c'], color='gray', alpha=0.3)

# Añadir una línea horizontal en la fracción 0.5
#plt.axhline(y=0.5, color='red', linestyle='--', label='Fraction 0.5')



# Graficar el punto donde la fracción 0.5 corta la curva promedio
#plt.plot(temperature_at_05, 0.5, 'ro', label=f'Temperature at 0.5: {temperature_at_05:.2f} °C')

# Configurar la gráfica
plt.xlabel('Temperature $(^{o}C)$')
plt.ylabel('Normalized Absorbance')
plt.legend()
plt.grid(False)
plt.tight_layout()

# Guardar la gráfica
plt.savefig('L4_heating_cooling_average_with_fraction_0.5_only_average.png', dpi=300)

# Mostrar la gráfica
plt.show()

print(f"Temperature corresponding to 0.5 absorbance fraction: {temperature_at_05:.2f} °C")
