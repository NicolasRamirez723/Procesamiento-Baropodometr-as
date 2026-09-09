#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import sys

# Importar filtros útiles
try:
    from scipy.signal import savgol_filter, medfilt
    _HAS_SCI_PKG = True
except Exception:
    _HAS_SCI_PKG = False

# ---------------------------
# Lectura robusta .txt
# ---------------------------
def leer_baropodometria(ruta_txt):
    tiempo_ms, fr, cx = [], [], []
    with open(ruta_txt, "r", encoding="utf-8", errors="ignore") as f:
        for linea in f:
            campos = linea.strip().split()
            if len(campos) < 7:
                continue
            try:
                t_val = float(campos[0])
                fr_val = float(campos[3])
                cx_raw = campos[5]
                cx_val = float(cx_raw) if cx_raw.lower() != "nan" else np.nan
            except Exception:
                continue
            tiempo_ms.append(t_val)
            fr.append(fr_val)
            cx.append(cx_val)

    if len(tiempo_ms) < 10:
        raise ValueError(f"No se pudieron extraer datos válidos de {ruta_txt}.")

    tiempo_ms = np.array(tiempo_ms, dtype=float)
    t = (tiempo_ms - tiempo_ms[0]) / 1000.0  # ms -> s
    cop_x = np.array(cx, dtype=float) / 1000.0  # mm -> m

    # interpolar NaNs en cop_x
    if np.any(np.isnan(cop_x)):
        idx = np.arange(len(cop_x))
        good = ~np.isnan(cop_x)
        if good.sum() > 0:
            cop_x[np.isnan(cop_x)] = np.interp(idx[np.isnan(cop_x)], idx[good], cop_x[good])
        else:
            cop_x = np.zeros_like(cop_x)

    fuerza_r = np.array(fr, dtype=float)
    dt_medio = np.median(np.diff(t))
    fs = 1.0 / dt_medio if dt_medio > 0 else 100.0

    return t, fuerza_r, cop_x, fs

# ---------------------------
# Suavizado de fuerza y detección robusta de apoyos
# ---------------------------
def _suavizar_fuerza(fuerza_r, fs, win_ms=40):
    """
    Suaviza la señal de fuerza para evitar cruces espurios.
    win_ms: ventana en ms para el filtro (por defecto 40 ms).
    """
    n = len(fuerza_r)
    if n == 0:
        return fuerza_r
    win_samples = max(1, int(round((win_ms / 1000.0) * fs)))
    if win_samples % 2 == 0:
        win_samples += 1
    if _HAS_SCI_PKG and win_samples >= 3:
        try:
            return medfilt(fuerza_r, kernel_size=win_samples)
        except Exception:
            pass
    # fallback: media móvil simple
    k = win_samples
    kernel = np.ones(k) / k
    padded = np.pad(fuerza_r, (k//2, k-1-k//2), mode='edge')
    smooth = np.convolve(padded, kernel, mode='valid')
    return smooth[:n]

def estimar_frecuencia_zancada(t, fuerza_r, fs=None, return_step_and_stride=False, umbral_frac=0.10):
    """
    Estima frecuencia de paso (steps/s) y de zancada (strides/s).
    - fs: frecuencia de muestreo (Hz). Si se pasa, se usa para suavizar la fuerza.
    - umbral_frac: fracción del máximo de fuerza para definir apoyo.
    Devuelve f_stride por defecto; si return_step_and_stride=True devuelve (f_step, f_stride).
    """
    # Suavizar fuerza para evitar detecciones espurias
    if fs is not None:
        fuerza_s = _suavizar_fuerza(fuerza_r, fs, win_ms=40)
    else:
        fuerza_s = fuerza_r.copy()

    # Umbral relativo más robusto
    maxf = np.nanmax(fuerza_s) if np.any(~np.isnan(fuerza_s)) else 1.0
    umbral = max(1e-6, umbral_frac * maxf)

    apoyo = fuerza_s > umbral

    # Eliminar pulsos muy cortos (ruido)
    if fs is not None:
        min_ms = 50  # eliminar pulsos < 50 ms
        min_samples = max(1, int(round((min_ms / 1000.0) * fs)))
    else:
        min_samples = 3

    segs = []
    in_seg = False
    start = 0
    for i, val in enumerate(apoyo):
        if val and not in_seg:
            in_seg = True
            start = i
        elif not val and in_seg:
            in_seg = False
            segs.append((start, i-1))
    if in_seg:
        segs.append((start, len(apoyo)-1))
    apoyo_clean = np.zeros_like(apoyo, dtype=bool)
    for s, e in segs:
        if (e - s + 1) >= min_samples:
            apoyo_clean[s:e+1] = True

    cruces = np.where(np.diff(apoyo_clean.astype(int)) == 1)[0]
    if len(cruces) < 2:
        if return_step_and_stride:
            return 0.9, 0.45
        return 0.9

    tiempos_cruce = t[cruces]
    periodos_paso = np.diff(tiempos_cruce)  # periodo de paso (s)

    # eliminar outliers en periodos (p. ej. > 2.5x mediana o < 0.4x mediana)
    med = np.median(periodos_paso)
    if med <= 0 or np.isnan(med):
        if return_step_and_stride:
            return 0.9, 0.45
        return 0.9
    valid = (periodos_paso > 0.4 * med) & (periodos_paso < 2.5 * med)
    if valid.sum() < 1:
        periodos_paso_use = periodos_paso
    else:
        periodos_paso_use = periodos_paso[valid]

    med_paso = np.median(periodos_paso_use)
    f_step = 1.0 / med_paso if med_paso > 0 else 0.0
    f_stride = f_step / 2.0

    # limitar a rango humano plausible para evitar valores espurios
    f_stride = float(np.clip(f_stride, 0.2, 2.5))  # zancadas/s
    f_step = float(np.clip(f_step, 0.4, 5.0))      # pasos/s

    if return_step_and_stride:
        return f_step, f_stride
    return f_stride

# ---------------------------
# Filtro de Kalman Cinemático (sin cambios funcionales)
# ---------------------------
class KF_Cinematico_CoG:
    def __init__(self, dt):
        self.A = np.array([[1.0, dt],[0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[1e-4, 0.0],[0.0, 1e-3]])
        self.R = np.array([[2.5]])
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1.0

    def predecir(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def actualizar(self, z):
        y_pred = self.H @ self.x
        residuo = z - y_pred[0]
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K.flatten() * residuo)
        self.P = (np.eye(2) - K @ self.H) @ self.P

    def correr(self, cop_x_array):
        n = len(cop_x_array)
        cog_x_est = np.zeros(n)
        if n > 0:
            self.x[0] = cop_x_array[0]
            self.x[1] = 0.0
        for k in range(n):
            self.predecir()
            self.actualizar(cop_x_array[k])
            cog_x_est[k] = self.x[0]
        return cog_x_est

def calcular_CoG_X(cop_x, fs):
    dt = 1.0 / fs
    kf = KF_Cinematico_CoG(dt=dt)
    mean_cop_x = np.nanmean(cop_x)
    cop_centered = cop_x - mean_cop_x
    cog_centered = kf.correr(cop_centered)
    return cog_centered + mean_cop_x

# ---------------------------
# CoG Ideal (Onda Senoidal)
# ---------------------------
def generar_CoG_X_ideal(t, f_stride, mean_x, cycles_per_stride=1.0, amplitude_m=0.015):
    """
    f_stride: frecuencia de zancada (zancadas/s).
    cycles_per_stride: cuántos ciclos por zancada (1 -> 1 ciclo por zancada;
                       2 -> 1 ciclo por paso).
    """
    f_eff = f_stride * cycles_per_stride
    omega = 2.0 * np.pi * f_eff
    x_ideal = mean_x + amplitude_m * np.sin(omega * (t - t[0]))
    return x_ideal

# ---------------------------
# Suavizado visual
# ---------------------------
def suavizar_serie(x):
    if _HAS_SCI_PKG and len(x) > 31:
        try:
            return savgol_filter(x, window_length=31, polyorder=3)
        except Exception:
            pass
    # fallback: media móvil 11
    k = 11
    kernel = np.ones(k) / k
    x_padded = np.pad(x, (k//2, k-1-k//2), mode='edge')
    x_smooth = np.convolve(x_padded, kernel, mode='valid')
    return x_smooth[:len(x)]

# ---------------------------
# Gráfica completa 
# ---------------------------
def graficar_comparacion_ML(t, cog_x_calc, cog_x_ideal, archivo, titulo):
    t_seg = t
    cx_mm = suavizar_serie(cog_x_calc) * 1000.0
    ix_mm = cog_x_ideal * 1000.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_seg, cx_mm, color='steelblue', label='CoG Calculado (Paciente)', linewidth=1.5)
    ax.plot(t_seg, ix_mm, color='darkorange', linestyle='--', label='CoG Ideal (Simétrico)', linewidth=1.2, alpha=0.9)

    ax.set_title(titulo, fontsize=14, pad=15)
    ax.set_xlabel("Tiempo (Segundos)", fontsize=11)
    ax.set_ylabel("Oscilación Lateral (mm)", fontsize=11)

    min_y = min(np.min(cx_mm), np.min(ix_mm))
    max_y = max(np.max(cx_mm), np.max(ix_mm))
    margen = max((max_y - min_y) * 0.15, 5.0)
    ax.set_ylim(min_y - margen, max_y + margen)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    fig.savefig(archivo, dpi=300)
    plt.close(fig)

# ---------------------------
# Pipeline Principal
# ---------------------------
def main():
    repo_root = "."
    base_data_dir = "data"
    if len(sys.argv) > 1:
        base_data_dir = sys.argv[1]

    patron = Path(repo_root) / base_data_dir / "muestras" / "The nature of functional variability in plantar pressure during a range of controlled walking speeds" / "*.txt"
    archivos = sorted(glob.glob(str(patron)))

    if not archivos:
        print("No se encontraron archivos en la ruta de muestras.")
        return

    grupos = {}
    for a in archivos:
        sj = Path(a).name.split("_")[0]
        grupos.setdefault(sj, []).append(a)

    salida_root = Path(repo_root) / "Baropodometrias procesadas" / "Modelos de CoG"
    salida_root.mkdir(parents=True, exist_ok=True)

    for sujeto, rutas in grupos.items():
        print(f"\nProcesando {sujeto} ({len(rutas)} caminatas)...")

        out_dir = salida_root / sujeto
        out_dir.mkdir(parents=True, exist_ok=True)

        for r in rutas:
            nombre_archivo = Path(r).stem
            print(f"  --> Analizando caminata: {nombre_archivo}")

            t, fr, cx, fs = leer_baropodometria(r)

            cog_x = calcular_CoG_X(cx, fs)

            # estimación robusta: suavizar fuerza y usar umbral más conservador
            try:
                f_step, f_stride = estimar_frecuencia_zancada(t, fr, fs=fs, return_step_and_stride=True, umbral_frac=0.10)
            except TypeError:
                f_stride = estimar_frecuencia_zancada(t, fr, fs=fs, umbral_frac=0.10)
                f_step = f_stride * 2.0

            print(f"    f_step (paso) = {f_step:.3f} Hz, f_stride (zancada) = {f_stride:.3f} Hz")

            # centrar para visualización
            cog_x = cog_x - np.nanmean(cog_x)
            mean_x = 0.0

            # generar ideal: 1 ciclo por zancada (cycles_per_stride=1.0)
            cog_x_ideal = generar_CoG_X_ideal(t, f_stride, mean_x, cycles_per_stride=1.0, amplitude_m=0.015)

            ruta_grafica = out_dir / f"CoG_ML_{nombre_archivo}.png"
            titulo_grafica = f"{sujeto} - Estabilidad Medio-Lateral ({nombre_archivo})"

            graficar_comparacion_ML(t, cog_x, cog_x_ideal, ruta_grafica, titulo_grafica)
            print(f"    guardado: {ruta_grafica}")

    print("\nProcesamiento completo.")

if __name__ == "__main__":
    main()
