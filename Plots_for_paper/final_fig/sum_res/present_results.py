import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import scienceplots

plt.style.use(['science', 'no-latex', 'bright'])

# Temperaturas de fusión obtenidas previamente
melting_temperatures = {
    "L0": 53.71,
    "L1": 53.95,
    "L3": 54.43,
    "L4": 58.64,
    "L7": 58.76,
    "L11": 59.00
}

def find_melting_temperature(df):
    # Encontrar el índice donde y_fit_avg es igual o mayor a 0.5
    index_05 = np.argmax(df['y_fit_avg'].values >= 0.5)
    melting_temperature = df.loc[index_05, 'x_fit_1']
    return melting_temperature

def find_error_tm(df, melting_temperature):
    index_05_max = np.argmax(df['y_fit_min'].values >= 0.5)
    temperature_05_max = df.loc[index_05_max, 'x_fit_1']
    error_tm = temperature_05_max - melting_temperature
    return error_tm

def generate_max_min_confidence_intervals(df):
    df['y_fit_max'] = df['y_fit_avg'] + df['confidence_interval']
    df['y_fit_min'] = df['y_fit_avg'] - df['confidence_interval']
    
    
"""   
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
"""
def plot_error_bars_with_confidence_interval(files, labels=None, ignore_columns=None):
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 12))
    #leng = ["1", "5", "8", "9", "11", "12"]  # Lista de valores enteros correspondientes
    leng = ["12", "11", "9", "8", "5", "1"]

    # Definir colores personalizados
    colors = ['#d53e4f', '#fc8d59', '#fee08b', '#e6f598', '#99d594', '#3288bd']

    for i, (file_name, label) in enumerate(zip(files, labels)):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df = pd.read_csv(file_path)

        if ignore_columns:
            df = df.drop(columns=ignore_columns)

        num_values = len(df.columns) - 1
        y_fit_columns = [col for col in df.columns if col.startswith("y_fit")]

        df['y_fit_avg'] = df[y_fit_columns].mean(axis=1)
        df['y_fit_std'] = df[y_fit_columns].std(axis=1)
        df['error'] = df['y_fit_std'] / np.sqrt(num_values)

        degrees_of_freedom = num_values - 1
        t_critical = t.ppf(0.975, degrees_of_freedom)

        df['confidence_interval'] =  df['error']

        generate_max_min_confidence_intervals(df)

        color = colors[i % len(colors)]  # Ciclo de colores si hay más perfiles que colores definidos
        melting_temperature = find_melting_temperature(df)
        axs[0].plot(df["x_fit_1"], df["y_fit_avg"], label=f'{label} ', alpha=0.7, linewidth=1, color=color)
        axs[0].fill_between(df["x_fit_1"], df["y_fit_min"], df["y_fit_max"], alpha=0.2, color=color)

        # Agregar puntos al segundo subplot
        melting_temp = melting_temperatures[label]
        integer_value = int(leng[i])  # Usar el valor entero correspondiente de leng
        axs[1].scatter(integer_value, melting_temp, color=color, alpha=0.7, s=100, edgecolor='black')

        # Ajustar la anotación del punto
        # Ajustar la anotación del punto
        if i in [0, 1]:  # Si el índice del label es 0 o 1
            axs[1].text(integer_value - 1.5, melting_temperatures[label] -0.5, f'{melting_temperatures[label]:.1f}'"$^{o}$C", fontsize=14, horizontalalignment='left')
        else:
            axs[1].text(integer_value + 0.2, melting_temperatures[label], f'{melting_temperatures[label]:.1f}'"$^{o}$C", fontsize=14, verticalalignment='center', horizontalalignment='left')
        """
        if label == labels[0,1]:  # Si es el último punto
            axs[1].text(integer_value -0.8, melting_temp + 0.4, f'{melting_temp:.1f}'"$^{o}$C", fontsize=14, horizontalalignment='left')
        else:
            axs[1].text(integer_value + 0.2, melting_temp -0.2, f'{melting_temp:.1f}'"$^{o}$C", fontsize=14, verticalalignment='center', horizontalalignment='left')
        """
        # Agregar barras de error vertical
        error_tm = find_error_tm(df, melting_temperature)
        axs[1].errorbar(integer_value, melting_temp, yerr=error_tm, fmt='none', color=color, capsize=5)

    axs[0].set_xlabel('Temperature ($^{o}$C)', fontsize=25)
    axs[0].set_ylabel('Fraction unbounded', fontsize=25)
    axs[0].tick_params(axis='both', which='major', labelsize=25)
    axs[0].legend(fontsize=16)
    
    # Agregar zoom
    axins = axs[0].inset_axes([0.4, 0.1, 0.3, 0.3])
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

    axins.set_xlim(52, 60)
    axins.set_xticks(np.arange(52, 64, 4))  # Controlar los ticks en el eje X del zoom
    axins.set_ylim(0.4, 0.6)
    axins.set_yticks(np.arange(0.4, 0.7, 0.1))  # Controlar los ticks en el eje Y del zoom
    
    axins.tick_params(axis='both', which='major', labelsize=16)
    axins.grid(True)
    axs[0].indicate_inset_zoom(axins)

    axs[1].set_xlabel('Length sticky ends (no. of nucleotides)', fontsize=25)
    axs[1].set_ylabel('Melting temperature ($^{o}$C)', fontsize=25)
    axs[1].tick_params(axis='both', which='major', labelsize=25)
    axs[1].set_yticks(np.arange(53, 61, 1))
    axs[1].set_xticks(np.arange(0, 16, 4))  
    axs[1].set_ylim(53, 60)
    plt.tight_layout()
    plt.savefig('error_bars_with_confidence_interval_subplot_paper.png', dpi=300)
    plt.show()



files = ["melting_curves_tol_e5_10_L1.csv", "melting_curves_tol_e5_10_L0.csv", "melting_curves_tol_e5_10_L3.csv", "melting_curves_tol_e5_10_L7.csv", "melting_curves_tol_e5_10_L4.csv", "melting_curves_tol_e5_10_L11.csv"]
labels = ["L0", "L1", "L3", "L4", "L7", "L11"]
ignore_columns = ["y_fit_1"]

plot_error_bars_with_confidence_interval(files, labels, ignore_columns)
