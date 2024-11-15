import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

plt.style.use(['science', 'no-latex', 'bright'])

def find_melting_temperature_1(df):
    index_05 = np.argmax(df['y_fit_avg'] >= 0.5)
    melting_temperature = df.loc[index_05, 'x_fit_1']
    return melting_temperature, index_05

def find_melting_temperature_2(df):
    idx1 = np.abs(df.iloc[:, 1] - 0.5).idxmin()
    idx2 = idx1 + 1 if df.iloc[idx1, 1] < 0.5 else idx1 - 1
    x1, y1 = df.iloc[idx1, 0], df.iloc[idx1, 1]
    x2, y2 = df.iloc[idx2, 0], df.iloc[idx2, 1]
    tm = x1 + ((0.5 - y1) * (x2 - x1) / (y2 - y1))
    return tm

def find_error_tm(df):
    melting_temperature, index_05 = find_melting_temperature_1(df)
    index_05_max = np.argmax(df['y_fit_min'] >= 0.5)
    temperature_05_max = df.loc[index_05_max, 'x_fit_1']
    error_tm = temperature_05_max - melting_temperature
    return error_tm

def generate_max_min_confidence_intervals(df):
    df['y_fit_max'] = df['y_fit_avg'] + df['confidence_interval']
    df['y_fit_min'] = df['y_fit_avg'] - df['confidence_interval']

def plot_error_bars_with_confidence_interval(files, labels=None, ignore_columns=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    for file_name, label in zip(files, labels):
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

        df['confidence_interval'] = df['error']

        generate_max_min_confidence_intervals(df)

        ax.plot(df["x_fit_1"], df["y_fit_avg"], label=f'{label} (Tm={find_melting_temperature_1(df)[0]:.2f}$^\circ$C $\pm$ {find_error_tm(df):.2f}$^\circ$C)', alpha=0.7)
        ax.fill_between(df["x_fit_1"], df["y_fit_min"], df["y_fit_max"], alpha=0.2)

    ax.set_xlabel('Temperature ($^{o}$C)', fontsize=16)
    ax.set_ylabel('Fraction unbounded', fontsize=16)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    axins = ax.inset_axes([0.6, 0.1, 0.3, 0.3])
    for file_name, label in zip(files, labels):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df_zoom = pd.read_csv(file_path)

        if ignore_columns:
            df_zoom = df_zoom.drop(columns=ignore_columns)

        num_values = len(df_zoom.columns) - 1
        y_fit_columns = [col for col in df_zoom.columns if col.startswith("y_fit")]

        df_zoom['y_fit_avg'] = df_zoom[y_fit_columns].mean(axis=1)

        axins.plot(df_zoom["x_fit_1"], df_zoom["y_fit_avg"], label=f'{label}', alpha=0.7)

    axins.set_xlim(55, 60)
    axins.set_ylim(0.4, 0.6)
    axins.grid(True)
    ax.indicate_inset_zoom(axins)

    plt.grid(False)
    plt.title('umbrella sampling ipy_oxdna', fontsize=18)
    return fig, ax

def read_melting_curves_nupack(file_paths):
    dfs_nupack = []
    for file_path in file_paths:
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))
            dfs_nupack.append((folder_name, df))
        else:
            print(f"File not found: {file_path}")
    return dfs_nupack

# Lista de rutas de archivos CSV de Nupack
file_paths = [
    "./nupack/nupack/L0/melting_curves_tol_e5_10_L0_nupack.csv",
    "./nupack/nupack/L1/melting_curves_tol_e5_10_L1_nupack.csv",
    "./nupack/nupack/L2/melting_curves_tol_e5_10_L2_nupack.csv",
    "./nupack/nupack/L3/melting_curves_tol_e5_10_L3_nupack.csv",
    "./nupack/nupack/L4/melting_curves_tol_e5_10_L4_nupack.csv",
    "./nupack/nupack/duplex/melting_curves_tol_e5_10_duplex_nupack.csv",
    # Añade más rutas de archivos CSV aquí si es necesario
]

# Leer los archivos CSV de Nupack
dfs_nupack = read_melting_curves_nupack(file_paths)


def find_melting_temperature(df):
    idx1 = np.abs(df.iloc[:, 1] - 0.5).idxmin()
    idx2 = idx1 + 1 if df.iloc[idx1, 1] < 0.5 else idx1 - 1
    x1, y1 = df.iloc[idx1, 0], df.iloc[idx1, 1]
    x2, y2 = df.iloc[idx2, 0], df.iloc[idx2, 1]
    tm = x1 + ((0.5 - y1) * (x2 - x1) / (y2 - y1))
    return tm

def plot_melting_curves(dfs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    for folder_name, df in dfs:
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'{folder_name} (Tm={find_melting_temperature(df):.2f}$^\circ$C)')
    
    axins = ax.inset_axes([0.6, 0.1, 0.3, 0.3])
    for folder_name, df in dfs:
        axins.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'{folder_name} (Tm={find_melting_temperature(df):.2f}$^\circ$C)')
    axins.set_xlim(60, 70)
    axins.set_ylim(0.4, 0.6)
    axins.set_xticks(np.arange(60, 71, 2))
    axins.set_yticks(np.arange(0.4, 0.61, 0.05))
    axins.grid(True)
    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.set_xlabel('Temperature ($^{o}$C)', fontsize=16)
    ax.set_ylabel('', fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(False)
    plt.title('nupack', fontsize=18)
    
    return fig, ax

# Directorio donde se encuentran los archivos CSV de nupack
directory_nupack = "nupack"

# Leer los archivos CSV de nupack
#dfs_nupack = read_melting_curves_nupack(directory_nupack)

# Definir las variables files, labels e ignore_columns
files = ["melting_curves_tol_e5_10_L0.csv", "melting_curves_tol_e5_10_L1.csv", "melting_curves_tol_e5_10_L2.csv","melting_curves_tol_e5_10_L3.csv","melting_curves_tol_e5_10_L4.csv", "melting_curves_tol_e5_10_L.csv"]
labels = ["L0", "L1", "L2","L3","L4", "duplex"]
ignore_columns = ["y_fit_1"]

# Crear la figura y los subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Graficar la primera función en el primer subplot
plot_error_bars_with_confidence_interval(files, labels, ignore_columns, ax=axs[0])

# Graficar la segunda función en el segundo subplot
plot_melting_curves(dfs_nupack, ax=axs[1])

plt.tight_layout()
plt.show()
