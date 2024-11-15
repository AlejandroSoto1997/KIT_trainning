#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:46:48 2024

@author: alejandrosoto
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data
data = pd.read_csv('melting_curves_tol_e5_10_110.csv')

# Extract the x and y values for each slice
x_values = [data[f'x_fit_{i}'] for i in range(1, 11)]
y_values = [data[f'y_fit_{i}'] for i in range(1, 11)]

# Define the colors
colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#9673b9', '#5e4fa2', '#2c1e50']

# Plot the data
plt.figure(figsize=(10, 6))

for i in range(len(x_values)):
    plt.plot(x_values[i], y_values[i], color=colors[i], label=f'Slice {i + 1}')
    if i > 0:
        plt.fill_between(x_values[i], y_values[i - 1], y_values[i], color=colors[i], alpha=0.3)

# Adding the last fill_between to transition to the first color
plt.fill_between(x_values[-1], y_values[-1], y_values[0], color=colors[0], alpha=0.3)

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Spectral Plot with Transition Colors')
plt.legend()

# Show plot
plt.show()
