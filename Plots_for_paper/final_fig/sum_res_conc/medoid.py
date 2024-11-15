import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids

plt.style.use(['science', 'no-latex', 'bright'])

def plot_representative_curves(files, labels=None, ignore_columns=None):
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

        x_values = np.array([df[col].values for col in x_fit_columns])
        y_values = np.array([df[col].values for col in y_fit_columns])

        y_combined = np.vstack(y_values)
        dissimilarity_matrix = cdist(y_combined, y_combined, metric='euclidean')

        kmedoids = KMedoids(n_clusters=1, metric='precomputed', method='pam', init='heuristic')
        kmedoids.fit(dissimilarity_matrix)

        medoid_index = kmedoids.medoid_indices_[0]

        ax.plot(x_values[medoid_index], y_values[medoid_index], label=f'{label} (Medoid Curve)', alpha=0.7)

    ax.set_xlabel('Temperature ($^{o}$C)')
    ax.set_ylabel('Fraction unbounded')
    plt.legend()
    plt.savefig('representative_curves.png', dpi=300)
    plt.show()

files = ["melting_curves_tol_e5_10_10.csv","melting_curves_tol_e5_10_30.csv","melting_curves_tol_e5_10_50.csv", "melting_curves_tol_e5_10_70.csv", "melting_curves_tol_e5_10_75.csv", "melting_curves_tol_e5_10_77.csv", "melting_curves_tol_e5_10_90.csv", "melting_curves_tol_e5_10_110.csv"]
labels = ["10","30","50","70", "75","77", "90", "110"]
ignore_columns = ["y_fit_1"]

plot_representative_curves(files, labels, ignore_columns)
