import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap

# Ruta base donde están las carpetas TEMP_X
base_path = "/Users/alejandrosoto/Downloads/L_0/SALT_0.1"

# Temperaturas (extraídas de las carpetas TEMP_X)
Temps = np.arange(14, 94, 2)

# Define el valor de normalización numérico
norm_value = 25 * (28 + 14)

# Lista de colores
colors = [
    "#313695", "#36479e", "#3c59a6", "#416aaf", "#4b7db8", "#588cc0", "#659bc8",
    "#74add1", "#83b9d8", "#92c5de", "#a3d3e6", "#b2ddeb", "#c1e4ef", "#d1ecf4",
    "#e0f3f8", "#e9f6e8", "#f2fad6", "#fbfdc7", "#fffbb9", "#fff2ac", "#fee99d",
    "#fee090", "#fed283", "#fdc374", "#fdb567", "#fca55d", "#f99153", "#f67f4b",
    "#f46d43", "#eb5a3a", "#e34933", "#db382b", "#ce2827", "#c01a27", "#b30d26",
    "#d73027", "#c6171b", "#a50f15", "#911a11", "#7f0d0b"
]

def shuffled_sterr(_in):
    _chunk = 20
    N = _in.size // _chunk
    if N == 0:
        return 0
    out = 0
    for i in range(N):
        _ids = np.random.randint(0, high=_in.size, size=_chunk)
        out += np.std(_in[_ids])
    return np.sqrt(out/N) / np.sqrt(N)

def extrapolate_bulk(fraction):
    phi = fraction / (1 - fraction)
    if phi == 0:
        return 0
    else:
        x = 1 + 1 / (2 * phi)
        return x - np.sqrt(np.fabs(x*x - 1))

def extrapolate_bulk_array(x):
    temp = x / (1 - x)
    return (1 + 1 / (2 * temp)) - np.sqrt((1 + 1. / (2 * temp))**2 - 1)

# Usar un contexto local para rcParams
with plt.rc_context({
    'font.size': 25,
    'axes.labelsize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 18,
    'axes.titlesize': 24,
}):
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))
    norm = Normalize(vmin=min(Temps), vmax=max(Temps))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    ave_data = []
    std_data = []

    for i, T in enumerate(Temps):
        temp_path = os.path.join(base_path, f"TEMP_{T}")
        store_data = []

        for r in range(30):
            run_path = os.path.join(temp_path, f"RUN_{r}")
            hb_file = os.path.join(run_path, "hb_list.dat")

            if os.path.exists(hb_file):
                data = np.loadtxt(hb_file)
                store_data.extend(data[-data.size//10:] / norm_value)
                axs[1].semilogx(np.arange(0, data.size//10+1), data[-data.size//10:] / norm_value, color=colors[i % len(colors)])

        store_data = np.asarray(store_data)
        ave_data.append(np.average(store_data[store_data.size//2:]))
        std_data.append(shuffled_sterr(store_data[store_data.size//2:]))

    ave_data = np.asarray(ave_data)
    std_data = np.asarray(std_data)
    extr = extrapolate_bulk_array(ave_data)

    ave_data = np.nan_to_num(ave_data, nan=0)
    std_data = np.nan_to_num(std_data, nan=0)
    extr = np.nan_to_num(extr, nan=0)

    for i, T in enumerate(Temps):
        axs[0].scatter(T, ave_data[i], color=colors[i % len(colors)], s=40, edgecolor='black', zorder=5)

    axs[0].errorbar(Temps, ave_data, yerr=std_data, marker='o', color='black', capsize=5, linestyle='none', label='Single molecule')
    axs[0].errorbar(Temps, extr, yerr=std_data, marker='s', color='black', capsize=5, linestyle='none', label='Bulk extrapolation')

    axs[0].set_ylabel("Fraction unbounded", fontsize=20)
    axs[0].set_xlabel("Temperature $(^{o}C)$", fontsize=20)
    axs[0].hlines(0.5, 10, 100, color='black', linestyles='dashed')
    axs[0].legend(loc='lower left')
    axs[0].grid(False)

    cbar = ColorbarBase(axs[0].inset_axes([1.05, 0.15, 0.02, 0.7]), cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Temperature $(^{o}C)$', fontsize=20)

    axs[1].set_xlabel("Time", fontsize=20)
    axs[1].tick_params(left=False, labelleft=False)

    plt.tight_layout()
    plt.savefig('L0_custom_with_colorbar_inside_subplot.png', dpi=150, format='png', bbox_inches='tight')
    plt.show()

result = np.vstack((Temps, ave_data, extr, std_data)).T
print(result)
