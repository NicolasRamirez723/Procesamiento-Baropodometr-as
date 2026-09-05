#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisis_ml_cog.py

Pipeline de Baropodometría enfocado EXCLUSIVAMENTE en la Oscilación Medio-Lateral (Eje X).
Compara el CoG X real vs un CoG X Ideal (Onda Senoidal).

Entradas: txt en `data/muestras/...`
Salidas: Imágenes .png en `Baropodometrías procesadas/`
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import sys
from scipy.linalg import expm

try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

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
                fr_val = float(campos[3]) # Fuerza pie derecho (para zancada)
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
    t = (tiempo_ms - tiempo_ms[0]) / 1000.0
    cop_x = np.array(cx, dtype=float) / 1000.0 # Convertir a metros

    # Interpolar NaNs si existen
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

def estimar_frecuencia_zancada(t, fuerza_r):
    apoyo = fuerza_r > (0.02 * np.max(fuerza_r) + 1e-6)
    cruces = np.where(np.diff(apoyo.astype(int)) == 1)[0]
    if len(cruces) < 2:
        return 0.9 # Valor por defecto si no detecta pasos
    periodos_paso = np.diff(t[cruces])
    periodo_zancada_medio = 2.0 * np.median(periodos_paso)
    if periodo_zancada_medio <= 0:
        return 0.9
    return 1.0 / periodo_zancada_medio

# ---------------------------
# Filtro de Kalman 1D (Solo Eje X)
# ---------------------------
class KF_1D_X:
    def __init__(self, dt, g=9.81, l_st=0.90):
        omega0_sq = g / l_st
        # Modelo de péndulo simple frontal
        A = np.array([[0.0, 1.0], [omega0_sq, 0.0]])
        B = np.array([[0.0], [-omega0_sq]])
        
        M_aug = np.zeros((3, 3))
        M_aug[:2, :2] = A
        M_aug[:2, 2:] = B
        M_exp = expm(M_aug * dt)
        
        self.Ad = M_exp[:2, :2]
        self.Bd = M_exp[:2, 2:].flatten()
        self.Q = np.diag((1e-6, 1e-4))
        self.R = np.array([[1e-4]])
        self.H = np.array([[1.0, 0.0]])
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1e-2

    def predecir(self, u):
        self.x = self.Ad @ self.x + self.Bd * u
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q

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
        for k in range(n):
            if k > 0:
                self.predecir(cop_x_array[k - 1])
            self.actualizar(cop_x_array[k])
            cog_x_est[k] = self.x[0]
        return cog_x_est

def calcular_CoG_X(cop_x, fs):
    dt = 1.0 / fs
    kf = KF_1D_X(dt=dt)
    mean_cop_x = np.nanmean(cop_x)
    cop_centered = cop_x - mean_cop_x
    cog_centered = kf.correr(cop_centered)
    return cog_centered + mean_cop_x

# ---------------------------
# CoG Ideal (Onda Senoidal)
# ---------------------------
def generar_CoG_X_ideal(t, f_z, mean_x):
    # Frecuencia de la onda: 1 zancada = 1 ciclo completo (derecha-izquierda)
    omega = 2 * np.pi * (f_z / 2.0) 
    amplitud_lateral = 0.015 # 15 mm hacia cada lado (30 mm total)
    
    # Onda senoidal centrada exactamente en la media del paciente
    x_ideal = mean_x + amplitud_lateral * np.sin(omega * (t - t[0]))
    return x_ideal

def suavizar_serie(x):
    if _HAS_SAVGOL and len(x) > 15:
        return savgol_filter(x, window_length=15, polyorder=3)
    return x

# ---------------------------
# Gráfica Única y Limpia
# ---------------------------
def graficar_comparacion_ML(t, cog_x_calc, cog_x_ideal, archivo, sujeto):
    t_seg = t # Mantenemos en segundos para mejor lectura clínica
    
    # Suavizado para visualización y conversión a mm
    cx_mm = suavizar_serie(cog_x_calc) * 1000.0
    ix_mm = cog_x_ideal * 1000.0

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t_seg, cx_mm, color='steelblue', label='CoG Calculado (Paciente)', linewidth=1.5)
    ax.plot(t_seg, ix_mm, color='darkorange', linestyle='--', label='CoG Ideal (Simétrico)', linewidth=2, alpha=0.9)

    ax.set_title(f"{sujeto} - Estabilidad Medio-Lateral (Eje X)", fontsize=14, pad=15)
    ax.set_xlabel("Tiempo (Segundos)", fontsize=11)
    ax.set_ylabel("Oscilación Lateral (mm)", fontsize=11)
    
    # Ajuste dinámico del eje Y para que no se vea aplastado
    min_y = min(np.min(cx_mm), np.min(ix_mm))
    max_y = max(np.max(cx_mm), np.max(ix_mm))
    margen = (max_y - min_y) * 0.15
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

    salida_root = Path(repo_root) / "Baropodometrías procesadas"
    salida_root.mkdir(parents=True, exist_ok=True)

    for sujeto, rutas in grupos.items():
        print(f"Procesando {sujeto} ({len(rutas)} archivos)...")
        listas_cog_x, listas_t, listas_fz = [], [], []
        
        for r in rutas:
            t, fr, cx, fs = leer_baropodometria(r)
            cog_x = calcular_CoG_X(cx, fs)
            fz = estimar_frecuencia_zancada(t, fr)
            
            listas_cog_x.append(cog_x)
            listas_t.append(t)
            listas_fz.append(fz)

        mlen = min(len(x) for x in listas_cog_x)
        if mlen == 0: continue
        
        # Promediar los trials del paciente
        cog_x_prom = np.nanmean([x[:mlen] for x in listas_cog_x], axis=0)
        t_comun = listas_t[0][:mlen]
        fz_media = np.nanmean(listas_fz)
        mean_x = np.nanmean(cog_x_prom)

        # Generar Ideal
        cog_x_ideal = generar_CoG_X_ideal(t_comun, fz_media, mean_x)

        # Guardar resultados
        out_dir = salida_root / sujeto
        out_dir.mkdir(parents=True, exist_ok=True)
        ruta_grafica = out_dir / f"CoG_Oscilacion_MedioLateral_{sujeto}.png"

        graficar_comparacion_ML(t_comun, cog_x_prom, cog_x_ideal, ruta_grafica, sujeto)
        print(f"  --> Generado {ruta_grafica.name}")

if __name__ == "__main__":
    main()  