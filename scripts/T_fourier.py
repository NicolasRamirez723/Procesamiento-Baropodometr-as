"""
En el siguiente archivo se realizará un algoritmo para extraer la transformada de fourier 
de la trayectoria del centro de presiones
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from Procesamiento_baropodometrias import read_data
from scipy.signal import welch

TRAYECTORIA_COP_X, TRAYECTORIA_COP_Y,TRAYECTORIA_COP_X_R,TRAYECTORIA_COP_X_L= read_data(verificar=False)

def calcular_f95(frecuencias,psd):

    potencia_acumulada = np.cumsum(psd)
    potencia_total = potencia_acumulada[-1]
    umbral_95 = 0.95 * potencia_total
    
    # Encontrar el índice donde se supera el 95%
    idx_95 = np.where(potencia_acumulada >= umbral_95)[0][0]
    return frecuencias[idx_95]


def calcular_ISF(cop_x_r,cop_x_l):

    """
    Calcula el índice de simetría frecuencial, recibe como parámetros la trayectoria del centro de 
    presiones de cada pie
    """
    if len(cop_x_l) != len(cop_x_r):

        raise "Los datos deben de coincidir en dimensión"
        

    datos_limpios_r = np.array(cop_x_r,dtype=float)

    datos_limpios_r = datos_limpios_r[~np.isnan(datos_limpios_r)
    ]
    datos_limpios_r = datos_limpios_r - np.mean(datos_limpios_r)

    datos_limpios_l = np.array(cop_x_l,dtype=float)
    
    datos_limpios_l = datos_limpios_l[~np.isnan(datos_limpios_l)
    ]
    datos_limpios_l = datos_limpios_l - np.mean(datos_limpios_l)


    n = min(len(datos_limpios_r),len(datos_limpios_l))

    datos_grafica_r = datos_limpios_r[:n]

    datos_grafica_l = datos_limpios_l[:n]

    frecuencias_r,PSD_R = welch(datos_grafica_r,nperseg=256) 

    frecuencias_l, PSD_L = welch(datos_grafica_l,nperseg=256)

    
    f95_r = calcular_f95(frecuencias=frecuencias_r,psd=PSD_R)

    f95_l = calcular_f95(frecuencias=frecuencias_l,psd=PSD_L)

    print(f"{PSD_L[0]}")

    isf = (abs(f95_r - f95_l) / (0.5 * (f95_r + f95_l))) * 100

    
    """
    Para determinar el indice de simetria debemos tomar un cociente entre la frecuencia que acapara el 95% 
    de la potencia en cada PSD
    """

    return isf


def FFT_COP():

    """
    Procesa las trayectorias del centro de presión para x, y, x de miembro izquierdo y derecho y calcula el ISF
    """

    PATH = "Baropodometrias procesadas"

    for archivo in TRAYECTORIA_COP_X:
        if archivo not in TRAYECTORIA_COP_Y:
            continue


        num_paciente = archivo.split("_")[0][-1]
        
        num_muestra = archivo.split(".")[0][-1] 

        nombre_sin_ext = os.path.splitext(archivo)[0]

        carpeta_persona = os.path.join(PATH,f"Subject{num_paciente}",f"Muestra{num_muestra}")
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



        #Guardamos las imagenes
        ruta_fft = os.path.join(carpeta_persona, f"{nombre_sin_ext}_FFT.png")
        ruta_anom = os.path.join(carpeta_persona, f"{nombre_sin_ext}_Frecuencias_anomalas.png")



        fft_fig.savefig(ruta_fft)
        anom_fig.savefig(ruta_anom)

        plt.close(fft_fig)
        plt.close(anom_fig)

        #Agregamos el ISF al txt de la persona

        ISF = calcular_ISF(cop_x_r=TRAYECTORIA_COP_X_R[archivo],cop_x_l=TRAYECTORIA_COP_X_L[archivo])
        path_txt = os.path.join(carpeta_persona,f"resumen_{num_muestra}.txt")

        with open(path_txt,"a",newline="", encoding="utf-8") as file:

            file.write(f"\nISF = {ISF}\n")





if __name__ == "__main__":

    FFT_COP()

    