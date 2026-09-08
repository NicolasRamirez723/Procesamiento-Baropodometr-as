import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import sys

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
    t = (tiempo_ms - tiempo_ms[0]) / 1000.0
    cop_x = np.array(cx, dtype=float) / 1000.0 

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
# CORRECCIÓN: estimar frecuencia de zancada (stride)
# ---------------------------
def estimar_frecuencia_zancada(t, fuerza_r, return_step_and_stride=False):
    """
    Estima frecuencia de zancada (1 zancada = 2 pasos).
    - Detecta inicios de apoyo del pie derecho (cruces).
    - periodos_paso = tiempos entre inicios de apoyo (periodo de paso).
    - periodo_zancada = 2 * median(periodos_paso).
    Devuelve f_zancada = 1 / periodo_zancada.
    Si return_step_and_stride=True devuelve (f_step, f_stride).
    """
    apoyo = fuerza_r > (0.02 * np.max(fuerza_r) + 1e-6)
    cruces = np.where(np.diff(apoyo.astype(int)) == 1)[0]
    if len(cruces) < 2:
        # fallback razonable
        if return_step_and_stride:
            return 0.9, 0.45
        return 0.9

    # tiempos de los cruces
    tiempos_cruce = t[cruces]
    periodos_paso = np.diff(tiempos_cruce)  # esto es el periodo de paso (s)
    med_paso = np.median(periodos_paso)

    # frecuencia de paso (steps/s)
    f_step = 1.0 / med_paso if med_paso > 0 else 0.0
    # frecuencia de zancada (stride) = 1 / (2 * periodo_paso_medio)
    periodo_zancada = 2.0 * med_paso
    f_stride = 1.0 / periodo_zancada if periodo_zancada > 0 else 0.0

    if return_step_and_stride:
        return f_step, f_stride

    # por compatibilidad con tu pipeline, devolvemos f_stride (frecuencia de zancada)
    return f_stride

# ---------------------------
# Filtro de Kalman Cinemático (Estable y Fuertemente Amortiguado)
# (sin cambios funcionales)
# ---------------------------
class KF_Cinematico_CoG:
    def __init__(self, dt):
        self.A = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([
            [1e-4, 0.0],
            [0.0,  1e-3]
        ])
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
# CoG Ideal (Onda Senoidal) - ahora usa f_stride por defecto
# ---------------------------
def generar_CoG_X_ideal(t, f_stride, mean_x, cycles_per_stride=1.0):
    """
    Genera ideal en X.
    - f_stride: frecuencia de zancada (zancadas/s).
    - cycles_per_stride: cuántos ciclos queremos por zancada (1 -> 1 ciclo por zancada).
      Si quisieras 1 ciclo por paso, usar cycles_per_stride=2.0 (porque 1 zancada = 2 pasos).
    """
    # frecuencia efectiva para la señal ideal
    f_eff = f_stride * cycles_per_stride
    omega = 2.0 * np.pi * f_eff
    amplitud_lateral = 0.015  # 15 mm en metros
    x_ideal = mean_x + amplitud_lateral * np.sin(omega * (t - t[0]))
    return x_ideal

def suavizar_serie(x):
    if _HAS_SAVGOL and len(x) > 31:
        return savgol_filter(x, window_length=31, polyorder=3)
    return x

# ---------------------------
# Gráfica Automática (sin cambios importantes)
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
    
    # Zoom dinámico: recorta a percentiles centrales para mejorar legibilidad
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
# Pipeline Principal (usa f_stride)
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

            # obtenemos tanto frecuencia de paso como de zancada para diagnóstico
            f_step, f_stride = estimar_frecuencia_zancada(t, fr, return_step_and_stride=True)
            print(f"    f_step (paso) = {f_step:.3f} Hz, f_stride (zancada) = {f_stride:.3f} Hz")

            # Elegimos la frecuencia de zancada para la ideal (1 ciclo por zancada)
            # Si prefieres 1 ciclo por paso, usa cycles_per_stride=2.0
            cog_x = cog_x - np.nanmean(cog_x)
            mean_x = 0.0
            cog_x_ideal = generar_CoG_X_ideal(t, f_stride, mean_x, cycles_per_stride=1.0)

            ruta_grafica = out_dir / f"CoG_ML_{nombre_archivo}.png"
            titulo_grafica = f"{sujeto} - Estabilidad Medio-Lateral ({nombre_archivo})"
            
            graficar_comparacion_ML(t, cog_x, cog_x_ideal, ruta_grafica, titulo_grafica)

if __name__ == "__main__":
    main()
