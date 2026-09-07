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

def estimar_frecuencia_zancada(t, fuerza_r):
    apoyo = fuerza_r > (0.02 * np.max(fuerza_r) + 1e-6)
    cruces = np.where(np.diff(apoyo.astype(int)) == 1)[0]
    if len(cruces) < 2:
        return 0.9 
    periodos_paso = np.diff(t[cruces])
    periodo_zancada_medio = 2.0 * np.median(periodos_paso)
    if periodo_zancada_medio <= 0:
        return 0.9
    return 1.0 / periodo_zancada_medio
# ---------------------------
# Filtro de Kalman Cinemático (Estable y Fuertemente Amortiguado)
# ---------------------------
class KF_Cinematico_CoG:
    def __init__(self, dt):
        # Modelo de estado puro y 100% estable: X = [Posición, Velocidad]
        self.A = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        
        # Observamos directamente la posición (El CoP es una observación ruidosa del CoG)
        self.H = np.array([[1.0, 0.0]])
        
       # Q: Ruido de proceso (permite la oscilación natural del paso)
        self.Q = np.array([
            [1e-4, 0.0],
            [0.0,  1e-3]
        ])
        
        # R: Ruido de medición equilibrado (ni tan sensible ni tan plano)
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
            self.x[0] = cop_x_array[0] # Arranca exactamente en la primera posición real
            self.x[1] = 0.0            # Velocidad inicial en cero
            
        for k in range(n):
            self.predecir()
            self.actualizar(cop_x_array[k])
            cog_x_est[k] = self.x[0]
            
        return cog_x_est
def calcular_CoG_X(cop_x, fs):
    dt = 1.0 / fs
    # Ya no dependemos de g ni de l_st para la estabilidad del filtro
    kf = KF_Cinematico_CoG(dt=dt)
    mean_cop_x = np.nanmean(cop_x)
    cop_centered = cop_x - mean_cop_x
    cog_centered = kf.correr(cop_centered)
    return cog_centered + mean_cop_x


# ---------------------------
# CoG Ideal (Onda Senoidal)
# ---------------------------
def generar_CoG_X_ideal(t, f_z, mean_x):
    omega = 2 * np.pi * (f_z / 2.0) 
    amplitud_lateral = 0.015 # 15 mm hacia cada lado
    x_ideal = mean_x + amplitud_lateral * np.sin(omega * (t - t[0]))
    return x_ideal

def suavizar_serie(x):
    if _HAS_SAVGOL and len(x) > 31:
        return savgol_filter(x, window_length=31, polyorder=3)
    return x

# ---------------------------
# Gráfica Automática
# ---------------------------
def graficar_comparacion_ML(t, cog_x_calc, cog_x_ideal, archivo, titulo):
    t_seg = t 
    cx_mm = suavizar_serie(cog_x_calc) * 1000.0
    ix_mm = cog_x_ideal * 1000.0

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t_seg, cx_mm, color='steelblue', label='CoG Calculado (Paciente)', linewidth=1.5)
    ax.plot(t_seg, ix_mm, color='darkorange', linestyle='--', label='CoG Ideal (Simétrico)', linewidth=2, alpha=0.9)

    ax.set_title(titulo, fontsize=14, pad=15)
    ax.set_xlabel("Tiempo (Segundos)", fontsize=11)
    ax.set_ylabel("Oscilación Lateral (mm)", fontsize=11)
    
    # Zoom dinámico: Se ajusta perfectamente al máximo y mínimo de los datos reales
    min_y = min(np.min(cx_mm), np.min(ix_mm))
    max_y = max(np.max(cx_mm), np.max(ix_mm))
    margen = max((max_y - min_y) * 0.15, 5.0) # Asegura un margen mínimo visual
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
            fz = estimar_frecuencia_zancada(t, fr)
            
            # Centramos ambas ondas en 0 para visualización perfecta
            cog_x = cog_x - np.nanmean(cog_x) 
            mean_x = 0.0 

            cog_x_ideal = generar_CoG_X_ideal(t, fz, mean_x)

            ruta_grafica = out_dir / f"CoG_ML_{nombre_archivo}.png"
            titulo_grafica = f"{sujeto} - Estabilidad Medio-Lateral ({nombre_archivo})"
            
            graficar_comparacion_ML(t, cog_x, cog_x_ideal, ruta_grafica, titulo_grafica)

if __name__ == "__main__":
    main()