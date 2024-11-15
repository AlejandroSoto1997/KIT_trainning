import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

plt.style.use(['science', 'no-latex', 'bright'])

# Temperaturas de fusión obtenidas previamente
melting_temperatures = {
    "10": 60.69,
    "30": 63.57,
    "50": 63.89,
    "70": 64.41,
    "75": 64.77,
    "77": 67.41,
    "90": 67.90,
    "110": 68.86
}

colors = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']

# Funciones auxiliares
def find_melting_temperature(df):
    index_05 = np.argmax(df['y_fit_avg'] >= 0.5)
    melting_temperature = df.loc[index_05, 'x_fit_1']
    return melting_temperature, index_05

def find_error_tm(df):
    melting_temperature, index_05 = find_melting_temperature(df)
    index_05_max = np.argmax(df['y_fit_min'] >= 0.5)
    temperature_05_max = df.loc[index_05_max, 'x_fit_1']
    error_tm = temperature_05_max - melting_temperature
    return error_tm

def generate_max_min_confidence_intervals(df):
    df['y_fit_max'] = df['y_fit_avg'] + df['confidence_interval']
    df['y_fit_min'] = df['y_fit_avg'] - df['confidence_interval']

def plot_error_bars_with_confidence_interval(files, labels=None, ignore_columns=None):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    concentrations = ["1.30", "3.17", "5.34", "7.55", "8.15", "8.35", "9.51", "11.85"]
    
    for i, (file_name, label) in enumerate(zip(files, labels)):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df = pd.read_csv(file_path)

        print("DataFrame after reading file:", file_path)
        print(df.head())  # Imprimir las primeras filas del DataFrame

        if ignore_columns:
            df = df.drop(columns=ignore_columns)

        num_values = len(df.columns) - 1
        y_fit_columns = [col for col in df.columns if col.startswith("y_fit")]

        df['y_fit_avg'] = df[y_fit_columns].mean(axis=1)
        df['y_fit_std'] = df[y_fit_columns].std(axis=1)
        df['error'] = df['y_fit_std'] / np.sqrt(num_values)

        degrees_of_freedom = num_values - 1
        t_critical = t.ppf(0.975, degrees_of_freedom)

        df['confidence_interval'] = df['error']

        generate_max_min_confidence_intervals(df)

        color = colors[i % len(colors)]  # Asignar color de la lista
        melting_temperature, index_05 = find_melting_temperature(df)
        axs[0].plot(df["x_fit_1"], df["y_fit_avg"], label=f'{concentrations[i]} $\mu M$', alpha=0.7, color=color)
        axs[0].fill_between(df["x_fit_1"], df["y_fit_min"], df["y_fit_max"], alpha=0.2, color=color)

        # Agregar las etiquetas de Tm con sus errores al scatter plot
        error_tm = find_error_tm(df)
        axs[1].errorbar(float(concentrations[i]), melting_temperatures[label], yerr=error_tm, fmt='o', color=color, capsize=5,  markersize=10, markeredgecolor='black')
        if i == 7:  # Último valor
            axs[1].text(float(concentrations[i]) - 0.2, melting_temperatures[label], f'{melting_temperature:.1f}'"$^{o}$C", fontsize=14, verticalalignment='center', horizontalalignment='right')
        else:
            axs[1].text(float(concentrations[i]) + 0.2, melting_temperatures[label], f'{melting_temperature:.1f}'"$^{o}$C", fontsize=14, verticalalignment='center', horizontalalignment='left')

    axs[0].set_xlabel('Temperature ($^{o}$C)', fontsize=25)
    axs[0].set_ylabel('Fraction unbounded', fontsize=25)
    #axs[0].set_yticks(np.arange(0, 1.1, 0.1))
    axs[0].tick_params(axis='both', which='major', labelsize=25)
    axs[0].legend(fontsize=16)

    # Agregar zoom
    axins = axs[0].inset_axes([0.6, 0.1, 0.3, 0.3])
    for i, (file_name, label) in enumerate(zip(files, labels)):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df_zoom = pd.read_csv(file_path)

        if ignore_columns:
            df_zoom = df_zoom.drop(columns=ignore_columns)

        num_values = len(df_zoom.columns) - 1
        y_fit_columns = [col for col in df_zoom.columns if col.startswith("y_fit")]

        df_zoom['y_fit_avg'] = df_zoom[y_fit_columns].mean(axis=1)

        color = colors[i % len(colors)]
        axins.plot(df_zoom["x_fit_1"], df_zoom["y_fit_avg"], label=f'{label}', alpha=0.7, color=color)

    axins.set_xlim(60, 70)
    axins.set_xticks(np.arange(60, 74, 4))  # Controlar los ticks en el eje X del zoom
    axins.set_ylim(0.4, 0.6)
    axins.set_yticks(np.arange(0.4, 0.7, 0.1))  # Controlar los ticks en el eje Y del zoom
    
    axins.tick_params(axis='both', which='major', labelsize=16)
    axins.grid(True)
    axs[0].indicate_inset_zoom(axins)

    # Configurar el subplot scatter
    axs[1].set_xlabel('Concentration ($\mu M$)', fontsize=25)
    axs[1].set_ylabel('Melting temperature ($^{o}$C)', fontsize=25)
    axs[1].tick_params(axis='both', which='major', labelsize=25)
    axs[1].set_yticks(np.arange(60, 72, 2))
    axs[1].set_xticks(np.arange(0, 14, 2))
    plt.tight_layout()

    plt.savefig('error_bars_with_confidence_interval_subplot_paper.png', dpi=300)
    plt.show()

files = ["melting_curves_tol_e5_10_10.csv", "melting_curves_tol_e5_10_70.csv", "melting_curves_tol_e5_10_50.csv", "melting_curves_tol_e5_10_30.csv", "melting_curves_tol_e5_10_77.csv", "melting_curves_tol_e5_10_90.csv", "melting_curves_tol_e5_10_75.csv", "melting_curves_tol_e5_10_110.csv"]
labels = ["10", "30", "50", "70", "75", "77", "90", "110"]
ignore_columns = ["y_fit_1"]

plot_error_bars_with_confidence_interval(files, labels, ignore_columns)
