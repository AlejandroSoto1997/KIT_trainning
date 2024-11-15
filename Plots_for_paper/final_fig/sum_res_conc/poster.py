#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:42:20 2024

@author: alejandrosoto
"""

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
    fig, ax = plt.subplots(figsize=(20, 20))

    for file_name, label in zip(files, labels):
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

        df['confidence_interval'] =  df['error']

        generate_max_min_confidence_intervals(df)

        ax.plot(df["x_fit_1"], df["y_fit_avg"], label=f'{label} (Tm={find_melting_temperature(df)[0]:.2f}$^\circ$C $\pm$ {find_error_tm(df):.2f}$^\circ$C)', alpha=0.7)
        ax.fill_between(df["x_fit_1"], df["y_fit_min"], df["y_fit_max"], alpha=0.2)

  
    ax.set_xlabel('Temperature ($^{o}$C)')
    ax.set_ylabel('Fraction unbounded')
    plt.legend()

files = ["melting_curves_tol_e5_10_50.csv"]
labels = ["50"]
ignore_columns = []
plt.grid(False)
    #plt.title('Melting Curves with Confidence Intervals')
plt.savefig('error_bars_with_confidence_interval_poster.png', dpi=300)
#plt.show()

plot_error_bars_with_confidence_interval(files, labels, ignore_columns)
