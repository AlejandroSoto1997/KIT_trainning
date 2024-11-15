import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex', 'bright'])

def plot_curves_with_profiles(files, labels=None, ignore_columns=None):
    fig, ax = plt.subplots(figsize=(20, 20))

    for file_name, label in zip(files, labels):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df = pd.read_csv(file_path)

        print("DataFrame after reading file:", file_path)
        print(df.head())  # Imprimir las primeras filas del DataFrame

        if ignore_columns:
            df = df.drop(columns=ignore_columns)

        x_fit_columns = [col for col in df.columns if col.startswith("x_fit")]
        y_fit_columns = [col for col in df.columns if col.startswith("y_fit")]

        df['y_fit_avg'] = df[y_fit_columns].mean(axis=1)
        df['y_fit_min'] = df[y_fit_columns].min(axis=1)
        df['y_fit_max'] = df[y_fit_columns].max(axis=1)

        # Plot individual profiles with dotted lines
        for y_fit_col in y_fit_columns:
            ax.plot(df[x_fit_columns[0]], df[y_fit_col], linestyle='dotted', alpha=0.5, label=f'{label} (Individual Profile)')

        # Plot average curve
        ax.plot(df[x_fit_columns[0]], df['y_fit_avg'], label=f'{label} (Average Curve)', alpha=0.7)
        
        # Plot shaded area between min and max
        ax.fill_between(df[x_fit_columns[0]], df['y_fit_min'], df['y_fit_max'], alpha=0.2)

    ax.set_xlabel('Temperature ($^{o}$C)')
    ax.set_ylabel('Fraction unbounded')
    plt.legend()

    # Agregar zoom
    axins = ax.inset_axes([0.6, 0.5, 0.3, 0.3])
    for file_name, label in zip(files, labels):
        folder_name = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(folder_name, file_name)
        df_zoom = pd.read_csv(file_path)

        if ignore_columns:
            df_zoom = df_zoom.drop(columns=ignore_columns)

        x_fit_columns = [col for col in df_zoom.columns if col.startswith("x_fit")]
        y_fit_columns = [col for col in df_zoom.columns if col.startswith("y_fit")]
 
        df_zoom['y_fit_avg'] = df_zoom[y_fit_columns].mean(axis=1)

        axins.plot(df_zoom[x_fit_columns[0]], df_zoom['y_fit_avg'], label=f'{label}', alpha=0.7)

    axins.set_xlim(60, 70)
    axins.set_ylim(0.4, 0.6)
    axins.grid(True)
    ax.indicate_inset_zoom(axins)

    plt.grid(False)
    #plt.title('Melting Curves with Max-Min Shading')
    plt.savefig('curves_with_profiles.png', dpi=300)
    plt.show()

files = ["melting_curves_tol_e5_10_10.csv","melting_curves_tol_e5_10_30.csv","melting_curves_tol_e5_10_50.csv"]
labels = ["10","30","50"]
ignore_columns = []

plot_curves_with_profiles(files, labels, ignore_columns)
