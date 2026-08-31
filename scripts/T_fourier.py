"""
En el siguiente archivo se realizará un algoritmo para extraer la transformada de fourier 
de la trayectoria del centro de presiones
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from Procesamiento_baropodometrias import read_data

TRAYECTORIA_COP_X, TRAYECTORIA_COP_Y = read_data()
def FFT_COP():

    PATH = "Baropodometrias procesadas"

    for archivo in TRAYECTORIA_COP_X:
        if archivo not in TRAYECTORIA_COP_Y:
            continue

        
        cop_x = TRAYECTORIA_COP_X[archivo]
        cop_y = TRAYECTORIA_COP_Y[archivo]

        datos_x = np.array(cop_x, dtype=float)
        datos_x = datos_x[~np.isnan(datos_x)]
        datos_x = datos_x - np.mean(datos_x)

        datos_y = np.array(cop_y, dtype=float)
        datos_y = datos_y[~np.isnan(datos_y)]
        datos_y = datos_y - np.mean(datos_y)

        if len(datos_x) == 0 or len(datos_y) == 0:
            continue

        n = min(len(datos_x), len(datos_y))
        signal_x = datos_x[:n]
        signal_y = datos_y[:n]

        freq_x = np.fft.fftfreq(n)
        freq_y = np.fft.fftfreq(n)

        tf_x = np.fft.fft(signal_x)
        tf_y = np.fft.fft(signal_y)

        magnitud_x = np.abs(tf_x)
        magnitud_y = np.abs(tf_y)

        dominant_x = freq_x[np.argmax(magnitud_x)]
        dominant_y = freq_y[np.argmax(magnitud_y)]

        band_x = max(0.05, abs(dominant_x) * 0.25)
        band_y = max(0.05, abs(dominant_y) * 0.25)

        mask_anom_x = np.abs(np.abs(freq_x) - abs(dominant_x)) > band_x
        mask_anom_y = np.abs(np.abs(freq_y) - abs(dominant_y)) > band_y

        fft_fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        ax_x.plot(freq_x, magnitud_x, color="tab:blue", label="COP_x")
        ax_x.set_ylabel("|FFT|")
        ax_x.set_xlim(-2.0, 2.0)
        ax_x.grid(True, which="both", ls="--", alpha=0.5)
        ax_x.set_title(f"Transformada de Fourier CoP - {os.path.splitext(archivo)[0]}")
        ax_x.legend()

        ax_y.plot(freq_y, magnitud_y, color="tab:orange", label="COP_y")
        ax_y.set_ylabel("|FFT|")
        ax_y.set_xlabel("f")
        ax_y.set_xlim(-2.0, 2.0)
        ax_y.grid(True, which="both", ls="--", alpha=0.5)
        ax_y.legend()

        anom_fig, (ax_anom_x, ax_anom_y) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        ax_anom_x.plot(freq_x[mask_anom_x], magnitud_x[mask_anom_x], color="tab:blue", label="COP_x anómalo")
        ax_anom_x.set_ylabel("|FFT|")
        ax_anom_x.set_title(f"Frecuencias anómalas - {os.path.splitext(archivo)[0]}")
        ax_anom_x.grid(True, ls="--", alpha=0.5)
        ax_anom_x.legend()

        ax_anom_y.plot(freq_y[mask_anom_y], magnitud_y[mask_anom_y], color="tab:orange", label="COP_y anómalo")
        ax_anom_y.set_ylabel("|FFT|")
        ax_anom_y.set_xlabel("f")
        ax_anom_y.grid(True, ls="--", alpha=0.5)
        ax_anom_y.legend()

        ax_anom_x.set_xlim(-2.0, 2.0)
        ax_anom_y.set_xlim(-2.0, 2.0)

        nombre_sin_ext = os.path.splitext(archivo)[0]
        carpeta_persona = os.path.join(PATH, nombre_sin_ext)
        os.makedirs(carpeta_persona, exist_ok=True)

        ruta_fft = os.path.join(carpeta_persona, f"{nombre_sin_ext}_FFT.png")
        ruta_anom = os.path.join(carpeta_persona, f"{nombre_sin_ext}_Frecuencias_anomalas.png")

        fft_fig.savefig(ruta_fft)
        anom_fig.savefig(ruta_anom)

        plt.close(fft_fig)
        plt.close(anom_fig)


if __name__ == "__main__":

    FFT_COP()