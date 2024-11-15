import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import scienceplots

plt.style.use(['science', 'no-latex', 'bright'])

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
    fig, ax = plt.subplots(figsize=(10, 10))

    # Custom colors
    colors = ['#8c510a', '#d8b365', '#f6e8c3', '#f5f5f5', '#c7eae5', '#5ab4ac', '#01665e']

    for i, (file_name, label) in enumerate(zip(files, labels)):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df = pd.read_csv(file_path)

        print("DataFrame after reading file:", file_path)
        print(df.head())  # Print the first few rows of the DataFrame

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

        # Plotting main data with custom color
        ax.plot(df["x_fit_1"], df["y_fit_avg"], label=f'{label} (Tm={find_melting_temperature(df)[0]:.2f}$^\circ$C $\pm$ {find_error_tm(df):.2f}$^\circ$C)', alpha=0.7, color=colors[i])
        
        # Adding confidence intervals with the same color
        ax.fill_between(df["x_fit_1"], df["y_fit_min"], df["y_fit_max"], alpha=0.2, color=colors[i])

    # Customizing plot elements for main plot
    ax.set_xlabel('Temperature ($^{o}$C)', fontsize=24)
    ax.set_ylabel('Fraction unbounded', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18)  # Adjust label size as desired
    ax.legend(fontsize=16)

    # Inset zoom
    axins = ax.inset_axes([0.6, 0.5, 0.3, 0.3])
    for i, (file_name, label) in enumerate(zip(files, labels)):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df_zoom = pd.read_csv(file_path)

        if ignore_columns:
            df_zoom = df_zoom.drop(columns=ignore_columns)

        num_values = len(df_zoom.columns) - 1
        y_fit_columns = [col for col in df_zoom.columns if col.startswith("y_fit")]

        df_zoom['y_fit_avg'] = df_zoom[y_fit_columns].mean(axis=1)

        axins.plot(df_zoom["x_fit_1"], df_zoom["y_fit_avg"], label=f'{label}', alpha=0.7, color=colors[i])

    axins.set_xlim(60, 70)
    axins.set_xticks(np.arange(60, 74, 4))
    axins.set_ylim(0.4, 0.6)
    axins.tick_params(axis='both', which='major', labelsize=16)  # Adjust label size as desired for zoom
    axins.set_yticks(np.arange(0.4, 0.61, 0.1))  # Adjust y ticks as desired for zoom
    axins.grid(True)
    ax.indicate_inset_zoom(axins)

    plt.grid(False)
    #plt.title('Melting Curves with Confidence Intervals')  # Optional title
    plt.savefig('error_bars_with_confidence_interval.png', dpi=300)
    plt.show()

files = ["melting_curves_tol_e5_10_10.csv","melting_curves_tol_e5_10_70.csv", "melting_curves_tol_e5_10_30.csv", "melting_curves_tol_e5_10_77.csv", "melting_curves_tol_e5_10_90.csv", "melting_curves_tol_e5_10_75.csv", "melting_curves_tol_e5_10_110.csv"]
labels = ["10","30","70", "75","77", "90", "110"]
ignore_columns = ["y_fit_1"]

plot_error_bars_with_confidence_interval(files, labels, ignore_columns)
