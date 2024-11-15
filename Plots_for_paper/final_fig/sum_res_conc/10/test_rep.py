import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids

# Read the CSV data
data = pd.read_csv('melting_curves_tol_e5_10_10.csv')

# Extract the x and y values for each slice
x_values = np.array([data[f'x_fit_{i}'].values for i in range(1, 11)])
y_values = np.array([data[f'y_fit_{i}'].values for i in range(1, 11)])

# Combine all y-values for clustering
y_combined = np.vstack(y_values)

# Compute pairwise distances between curves
dissimilarity_matrix = cdist(y_combined, y_combined, metric='euclidean')

# Apply K-Medoids
kmedoids = KMedoids(n_clusters=1, metric='precomputed', method='pam', init='heuristic')
kmedoids.fit(dissimilarity_matrix)

# Find the index of the medoid
medoid_index = kmedoids.medoid_indices_[0]

# Define the colors
colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#9673b9', '#5e4fa2', '#2c1e50']

# Plot the data
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Subplot 1: All curves without fill_between
for i in range(len(x_values)):
    axs[0].plot(x_values[i], y_values[i], color=colors[i], label=f'Slice {i + 1}')
axs[0].set_xlabel('X-axis')
axs[0].set_ylabel('Y-axis')
axs[0].set_title('All Curves')
axs[0].legend()

# Subplot 2: Representative curve (Medoid) without fill_between
axs[1].plot(x_values[medoid_index], y_values[medoid_index], color=colors[medoid_index], label=f'Representative Curve (Slice {medoid_index + 1})')
axs[1].set_xlabel('X-axis')
axs[1].set_ylabel('Y-axis')
axs[1].set_title('Most Representative Curve (Medoid)')
axs[1].legend()

# Show plot
plt.tight_layout()
plt.show()
