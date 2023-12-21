#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:01:22 2023

@author: alejandrosoto
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Lee el archivo de energía con pandas
file_path = 'energy_file_md.txt'  # Reemplaza con el nombre real de tu archivo
df = pd.read_csv(file_path, sep='\s+', names=['Time', 'Energy1', 'Energy2', 'Energy3'])

# Grafica las tres energías en función del tiempo en escala logarítmica
plt.figure(figsize=(10, 6)) 
plt.plot(df['Time'], np.log(-df['Energy1']), label='Potential energy')
plt.plot(df['Time'], np.log(df['Energy2']), label='Kinetic energy')
plt.plot(df['Time'], np.log(-df['Energy3']), label='Total energy')

# Configura el gráfico
plt.title('Energías en función del tiempo (escala logarítmica)')
plt.xlabel('Tiempo')
plt.ylabel('Log(Energía)')
plt.legend()
plt.grid(True)
plt.show()
