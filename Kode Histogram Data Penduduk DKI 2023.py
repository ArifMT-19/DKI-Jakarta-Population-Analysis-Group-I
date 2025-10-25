import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

file_name = 'Jumlah Penduduk Menurut Kabupaten_Kota di Provinsi DKI Jakarta , 2023.csv'

try:
    df = pd.read_csv(file_name, skiprows=3, header=None)
    df.columns = ['Wilayah', 'Jumlah_Penduduk']
    df_cleaned = df.iloc[:-1].copy()
    df_cleaned['Jumlah_Penduduk'] = df_cleaned['Jumlah_Penduduk'].astype(int)

    bins_manual = [0,500000,1000000,1500000,2000000,2500000,3000000,3500000,4000000]
    fig, ax = plt.subplots(figsize=(10, 6))

    n, bins, patches = ax.hist(
        df_cleaned['Jumlah_Penduduk'],
        bins=bins_manual,
        edgecolor='black',
        color='steelblue',
        alpha=0.6
    )

    x_vals = np.linspace(df_cleaned['Jumlah_Penduduk'].min(),
                         df_cleaned['Jumlah_Penduduk'].max(), 200)
    kde = gaussian_kde(df_cleaned['Jumlah_Penduduk'])
    kde_scaled = kde(x_vals) * len(df_cleaned) * (bins[1]-bins[0])
    ax.plot(x_vals, kde_scaled, color='red', linewidth=2, label='Kurva Density')

    ax.set_xlim(0, 4000000)

    for count, patch in zip(n, patches):
        if count > 0:
            ax.annotate(
                f'{int(count)} Wilayah',
                xy=(patch.get_x() + patch.get_width() / 2, count),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom'
            )

    ax.set_title('Histogram Distribusi Penduduk DKI Jakarta 2023 dengan Kurva Density', fontsize=16)
    ax.set_xlabel('Kelompok Jumlah Penduduk', fontsize=12)
    ax.set_ylabel('Frekuensi (Jumlah Wilayah)', fontsize=12)
    ax.legend()

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1_000_000:.1f} Juta'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    populasi = df_cleaned['Jumlah_Penduduk']
    mean_val = populasi.mean()
    median_val = populasi.median()
    mode_val = populasi.mode()
    variance_val = populasi.var()
    std_dev_val = populasi.std()

    print(f"Mean (Rata-rata)   : {mean_val:,.2f}")
    print(f"Median (Nilai Tengah): {median_val:,.2f}")
    if mode_val.empty:
        print("Mode (Modus)         : Tidak ada (semua nilai unik)")
    else:
        print(f"Mode (Modus)         : {mode_val.to_string(index=False)}")
    print(f"Variansi             : {variance_val:,.2f}")
    print(f"Standar Deviasi      : {std_dev_val:,.2f}")

except FileNotFoundError:
    print(f"Error: File '{file_name}' tidak ditemukan.")
except Exception as e:
    print(f"Terjadi error: {e}")