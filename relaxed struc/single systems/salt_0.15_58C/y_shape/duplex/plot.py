
#energy_plts_md
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# Load energy data from your file (replace 'energy_file.txt' with your file name)
data = np.loadtxt('energy_file_md.txt')

# Selecciona la segunda columna (índice 1) de tus datos
time_data = data[:, 0]
energy_data = data[:, 1]
energy_k_data = data[:, 2]
energy_t_data = data[:, 3]

# Añade una pequeña constante para evitar log(0) y posibles problemas con ceros
small_constant = 1e-10
energy_data_log = np.log(np.abs(energy_data) + small_constant)

# Parámetro para el promedio móvil (ajústalo según tus necesidades)
window_size = 50

# Calcula el promedio móvil
smoothed_data = np.convolve(energy_data_log, np.ones(window_size)/window_size, mode='valid')

# Ajusta los índices para que coincidan con la longitud de los datos suavizados
steps_smoothed = np.arange(1 + window_size//2, len(smoothed_data) + 1 + window_size//2)

# Obtiene el número de pasos de la simulación
num_steps = len(energy_data_log)

# Crea una lista de números de pasos para el eje x
steps = np.arange(1, num_steps + 1)

# Crea el primer gráfico con escala logarítmica
fig, ax1 = plt.subplots()
ax1.plot(steps, energy_data_log, label='Log(Energy)', color='blue')
ax1.plot(steps_smoothed, smoothed_data, label=f'Smoothed Log(Energy) (Window Size={window_size})', color='orange')
ax1.set_yscale('log')  # Aplica escala logarítmica en el eje y

# Configura el eje y secundario para la energía sin modificaciones
ax2 = ax1.twinx()
ax2.plot(steps, energy_data, label='Energy (Original Scale)', color='green')

# Calcula el valor promedio de la curva de energía
average_energy = np.mean(np.abs(energy_data))
ax2.axhline(y=-average_energy, color='r', linestyle='--', label='Average Energy')

# Etiquetas y leyendas
ax1.set_xlabel('Monte Carlo Steps')
ax1.set_ylabel('Log(Energy)', color='blue')
ax2.set_ylabel('Energy (Original Scale)', color='green')
ax1.set_title('Energy Evolution with Logarithmic Scale and Smoothed Data')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()