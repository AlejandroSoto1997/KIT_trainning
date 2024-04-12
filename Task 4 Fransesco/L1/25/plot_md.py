import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Lee el archivo de energía con pandas
file_path = 'energy_file_md.txt'  # Reemplaza con el nombre real de tu archivo
df = pd.read_csv(file_path, sep='\s+', names=['Time', 'Energy1', 'Energy2', 'Energy3'])

# Calcula el valor medio de la energía potencial
mean_potential_energy = df['Energy1'].mean()

# Configura la figura con tres subgráficos
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Subgráfico 1: Escala logarítmica en el eje x
axes[0].plot(np.log(df['Time']), df['Energy1'], label='Potential energy')
axes[0].axhline(mean_potential_energy, color='r', linestyle='--', label=f'Mean Potential Energy: {mean_potential_energy:.2f}')
axes[0].plot(np.log(df['Time']), df['Energy2'], label='Kinetic energy')
axes[0].plot(np.log(df['Time']), df['Energy3'], label='Total energy')
axes[0].set_xscale('log')
axes[0].set_title('Logarithmic Time Scale')
axes[0].set_xlabel('Log(Time)')
axes[0].set_ylabel('Energy')
axes[0].legend()
axes[0].grid(True)
axes[0].set_ylim(bottom=-1.5, top=0.5)  # Ajusta los límites en el eje y

# Subgráfico 2: Escala logarítmica en el eje x
axes[1].plot(np.log(df['Time']), df['Energy1'], label='Potential energy')
axes[1].axhline(mean_potential_energy, color='r', linestyle='--', label=f'Mean Potential Energy: {mean_potential_energy:.2f}')
axes[1].plot(np.log(df['Time']), df['Energy2'], label='Kinetic energy')
axes[1].plot(np.log(df['Time']), df['Energy3'], label='Total energy')
#axes[1].set_xscale('log')
axes[1].set_title('Logarithmic Time Scale')
axes[1].set_xlabel('Log(Time)')
axes[1].set_ylabel('Energy')
axes[1].legend()
axes[1].grid(True)
axes[1].set_ylim(bottom=-1.5, top=0.5)  # Ajusta los límites en el eje y

# Subgráfico 3: Escala lineal en el eje x con escala científica
axes[2].plot(df['Time'] * 3.03 / 1e+12, df['Energy1'], label='Potential energy')
axes[2].axhline(mean_potential_energy, color='r', linestyle='--', label=f'Mean Potential Energy: {mean_potential_energy:.2f}')
axes[2].plot(df['Time'] * 3.03 / 1e+12, df['Energy2'], label='Kinetic energy')
axes[2].plot(df['Time'] * 3.03 / 1e+12, df['Energy3'], label='Total energy')
axes[2].set_title('Linear Time Scale')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Energy')
axes[2].legend()
axes[2].grid(True)
axes[2].set_ylim(bottom=-1.5, top=0.5)

# Ajusta el diseño y aplica la escala científica en el eje x
plt.tight_layout()
for ax in axes:
    ax.xaxis.set_major_formatter(ScalarFormatter())

# Muestra el gráfico
plt.show()
plt.savefig('energy_profile.png')
