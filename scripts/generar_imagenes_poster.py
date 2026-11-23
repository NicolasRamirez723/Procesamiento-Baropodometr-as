"""Genera figuras de alta resolución y estética unificada para el póster científico.

Las figuras se construyen a partir de los resultados procesados del repositorio
(CSV estadísticos, resultados de RMSE/Kalman y registros baropodométricos)
y se guardan en la carpeta `Poster_Imagenes/`.
"""

import os
import re
import sys
import tempfile
from itertools import islice
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_baro_cache"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from Simulador_CoG_KKF_y_RMSE import (
    calcular_CoG_X,
    calcular_rmse,
    estimar_frecuencia_zancada,
    generar_CoG_X_ideal,
    leer_baropodometria,
    suavizar_serie,
)
from calculo_ID import mapear_pies


# Paleta coherente con el póster (azul/turquesa UNER + acentos contrastantes)
TEAL_PRIMARY = "#2B9EB3"
TEAL_DARK = "#155E75"
TEAL_LIGHT = "#A5DCE6"
NAVY = "#1B365D"
CORAL = "#E76F51"
AMBER = "#E9A825"
SLATE = "#475569"
GRID_COLOR = "#E2E8F0"

SPEED_LABELS = ["1.1", "1.3", "1.5", "1.7", "1.9"]
MUESTRA_TO_VEL = {1: 1.1, 3: 1.3, 5: 1.5, 7: 1.7, 9: 1.9}
SPEED_PALETTE = ["#1D4ED8", "#0EA5E9", "#14B8A6", "#F59E0B", "#E76F51"]

OUT_DIR = Path("Poster_Imagenes")
DATA_DIR = Path(
    "data/muestras/The nature of functional variability in plantar pressure during a range of controlled walking speeds"
)


def configurar_estilo():
    """Configura parámetros globales de Matplotlib para lectura clara en póster."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#94A3B8",
            "axes.linewidth": 1.0,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelcolor": NAVY,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.color": "#1E293B",
            "ytick.color": "#1E293B",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def cargar_datos_combinados() -> pd.DataFrame:
    """Combina estadisticas.csv y Resultados_Estadisticos_RMSE.csv por sujeto y muestra."""
    df_est = pd.read_csv(os.path.join("Estadisticas", "estadisticas.csv"))
    df_rmse = pd.read_csv(
        os.path.join("Baropodometrias procesadas", "Resultados_Estadisticos_RMSE.csv")
    )

    df_rmse["num paciente"] = df_rmse["Sujeto"].str.replace("Subject", "", regex=False).astype(int)
    df_rmse["muestra"] = df_rmse["Archivo"].str.extract(r"_1p(\d+)").astype(int)

    df = pd.merge(
        df_est,
        df_rmse[["num paciente", "muestra", "Frec_Zancada_Hz", "RMSE_mm"]],
        on=["num paciente", "muestra"],
        how="outer",
    )
    df["velocidad"] = df["muestra"].map(MUESTRA_TO_VEL)
    df["asimetria_cm"] = (
        df["longitud del miembro mayor"] - df["longitud estimada del miembro corto"]
    ) * 100.0
    df["diff_ID"] = (
        df["promedio impulso derecho"] - df["promedio impulso izquierdo"]
    ).abs()
    return df.sort_values(["velocidad", "num paciente"]).reset_index(drop=True)


def _formatear_p(p_val: float) -> str:
    if p_val < 0.001:
        return "p < 0.001"
    return f"p = {p_val:.3f}"


def generar_fig1_metricas_velocidad(df: pd.DataFrame):
    """Figura 1: Evolución de RMSE, frecuencia de zancada, ΔID e ISF según la velocidad."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    fig.suptitle(
        "Efecto de la velocidad de marcha sobre la estabilidad y simetría plantar (n = 16 sujetos)",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        y=0.97,
    )

    paneles = [
        (axes[0, 0], "RMSE_mm", "A) Error del CoG medio-lateral (RMSE)", "RMSE (mm)", TEAL_PRIMARY),
        (axes[0, 1], "Frec_Zancada_Hz", "B) Frecuencia de zancada", "Frecuencia (Hz)", "#0284C7"),
        (axes[1, 0], "diff_ID", "C) Diferencia de Impulso Dinámico (ΔID)", r"$|\mathrm{ID}_{\mathrm{der}} - \mathrm{ID}_{\mathrm{izq}}|\ (\mathrm{N}\cdot\mathrm{s})$", "#0D9488"),
        (axes[1, 1], "ISF", "D) Índice de Simetría Frecuencial (ISF)", "ISF (%)", CORAL),
    ]

    rng = np.random.default_rng(42)
    velocidades = sorted(df["velocidad"].dropna().unique())
    posiciones = np.arange(len(velocidades))

    for ax, col, titulo, ylabel, color in paneles:
        grupos = [df.loc[df["velocidad"] == v, col].dropna().values for v in velocidades]

        bp = ax.boxplot(
            grupos,
            positions=posiciones,
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": NAVY, "linewidth": 2.2},
            whiskerprops={"color": SLATE, "linewidth": 1.3},
            capprops={"color": SLATE, "linewidth": 1.3},
            boxprops={"edgecolor": color, "linewidth": 1.5},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.24)

        for idx, valores in enumerate(grupos):
            jitter = rng.uniform(-0.13, 0.13, size=len(valores))
            ax.scatter(
                np.full_like(valores, posiciones[idx], dtype=float) + jitter,
                valores,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                s=34,
                alpha=0.82,
                zorder=3,
            )

        medianas = [np.median(g) for g in grupos]
        ax.plot(
            posiciones,
            medianas,
            color=NAVY,
            linestyle="--",
            linewidth=1.5,
            marker="o",
            markersize=5,
            zorder=4,
            label="Mediana",
        )

        sub = df[["velocidad", col]].dropna()
        rho, p_val = spearmanr(sub["velocidad"], sub[col])

        y_min, y_max = float(sub[col].min()), float(sub[col].max())
        rango = max(y_max - y_min, 1e-6)
        ax.set_ylim(y_min - 0.08 * rango, y_max + 0.28 * rango)

        ax.text(
            0.97,
            0.95,
            f"Spearman ρ = {rho:+.2f}\n({_formatear_p(p_val)})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=NAVY,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#F8FAFC",
                edgecolor=TEAL_PRIMARY,
                alpha=0.95,
            ),
        )

        ax.set_title(titulo, pad=8, color=NAVY, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(posiciones)
        ax.set_xticklabels([f"{v:.1f} m/s" for v in velocidades])

    axes[1, 0].set_xlabel("Velocidad de marcha en cinta")
    axes[1, 1].set_xlabel("Velocidad de marcha en cinta")

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    out_path = OUT_DIR / "01_metricas_por_velocidad.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def parsear_resultados_spearman(path_txt: str):
    """Extrae la matriz de ρ y p-Holm desde Estadisticas/resultados_estadisticos.txt."""
    with open(path_txt, "r", encoding="utf-8") as f:
        texto = f.read()

    bloques = re.split(r"\n(?=Muestra \d+)", texto.strip())
    muestras = []
    datos = {
        "ΔL vs. ISF": [],
        "ΔL vs. ΔID": [],
        "ΔL vs. RMSE (CoG)": [],
        "ISF vs. ΔID": [],
    }
    holm_texto = {k: [] for k in datos}

    for bloque in bloques:
        m = re.search(r"Muestra (\d+)", bloque)
        if not m:
            continue
        muestra = int(m.group(1))
        muestras.append(MUESTRA_TO_VEL[muestra])

        r_isf = re.search(r"Spearman asimetría de longitud .*? e ISF:\s*([-+]?\d+\.\d+)", bloque)
        r_id = re.search(r"Spearman asimetría de longitud y diferencia absoluta de impulsos:\s*([-+]?\d+\.\d+)", bloque)
        r_rmse = re.search(
            r"Spearman RMSE y asimetría de longitud:.*?rho=([-+]?\d+\.\d+).*?p-Holm=([-+]?\d+\.\d+)",
            bloque,
        )
        r_isf_id = re.search(
            r"Spearman ISF y diferencia absoluta de impulsos:.*?rho=([-+]?\d+\.\d+).*?p-Holm=([-+]?\d+\.\d+)",
            bloque,
        )

        datos["ΔL vs. ISF"].append(float(r_isf.group(1)))
        holm_texto["ΔL vs. ISF"].append("n.s.")

        datos["ΔL vs. ΔID"].append(float(r_id.group(1)))
        holm_texto["ΔL vs. ΔID"].append("n.s.")

        datos["ΔL vs. RMSE (CoG)"].append(float(r_rmse.group(1)))
        holm_texto["ΔL vs. RMSE (CoG)"].append(rf"$p_{{\mathrm{{Holm}}}}={float(r_rmse.group(2)):.2f}$")

        datos["ISF vs. ΔID"].append(float(r_isf_id.group(1)))
        holm_texto["ISF vs. ΔID"].append(rf"$p_{{\mathrm{{Holm}}}}={float(r_isf_id.group(2)):.2f}$")

    return muestras, datos, holm_texto


def generar_fig2_heatmap_spearman():
    """Figura 2: Mapa de calor de correlaciones de Spearman por velocidad."""
    velocidades, datos, holm_texto = parsear_resultados_spearman(
        os.path.join("Estadisticas", "resultados_estadisticos.txt")
    )
    etiquetas_filas = list(datos.keys())
    matriz = np.array([datos[k] for k in etiquetas_filas])

    cmap = LinearSegmentedColormap.from_list(
        "teal_coral", [CORAL, "#F8FAFC", TEAL_PRIMARY]
    )

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    im = ax.imshow(matriz, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")

    ax.set_xticks(np.arange(len(velocidades)))
    ax.set_xticklabels([f"{v:.1f} m/s\n(n={'15' if v==1.1 else '16'})" for v in velocidades], fontweight="bold")
    ax.set_yticks(np.arange(len(etiquetas_filas)))
    ax.set_yticklabels(etiquetas_filas, fontweight="bold")

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            val = matriz[i, j]
            sub = holm_texto[etiquetas_filas[i]][j]
            color_txt = "white" if abs(val) > 0.38 else NAVY
            ax.text(
                j,
                i - 0.08,
                f"ρ = {val:+.2f}",
                ha="center",
                va="center",
                color=color_txt,
                fontsize=11.5,
                fontweight="bold",
            )
            ax.text(
                j,
                i + 0.22,
                f"({sub})",
                ha="center",
                va="center",
                color=color_txt,
                fontsize=8.8,
                alpha=0.9,
            )

    # Líneas divisorias limpias entre celdas
    ax.set_xticks(np.arange(-0.5, len(velocidades), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(etiquetas_filas), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Coeficiente de Spearman (ρ)", color=NAVY, fontweight="bold")

    ax.set_title(
        "Correlación de Spearman (ρ) entre dismetría (ΔL) y métricas biomecánicas por velocidad",
        pad=14,
        color=NAVY,
        fontsize=12.6,
        fontweight="bold",
    )
    fig.text(
        0.47,
        0.02,
        r"Ninguna correlación resultó estadísticamente significativa tras corrección de Holm-Bonferroni ($p_{\mathrm{Holm}} \geq 0.44;\ p_{\mathrm{perm}} \geq 0.089$).",
        ha="center",
        fontsize=9.5,
        style="italic",
        color=SLATE,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = OUT_DIR / "02_heatmap_spearman.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generar_fig3_cog_kalman_zoom():
    """Figura 3: Comparación temporal del CoG (Kalman vs Ideal) en baja y alta velocidad."""
    casos = [
        ("Subject10_1p1.txt", "1.1 m/s (Velocidad baja)", TEAL_PRIMARY),
        ("Subject10_1p9.txt", "1.9 m/s (Velocidad alta)", "#0284C7"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True, sharey=True)
    fig.suptitle(
        "Trayectoria medio-lateral del Centro de Gravedad (CoG): Filtro de Kalman vs. Modelo Simétrico (Sujeto 10)",
        fontsize=13.5,
        fontweight="bold",
        color=NAVY,
        y=0.97,
    )

    t_ini, t_fin = 20.0, 32.0
    for ax, (archivo, etiqueta, color_real) in zip(axes, casos):
        t, fr, cx, fs = leer_baropodometria(str(DATA_DIR / archivo))
        cog_x = calcular_CoG_X(cx, fs)
        cog_x = cog_x - np.nanmean(cog_x)
        _, f_stride = estimar_frecuencia_zancada(
            t, fr, fs=fs, return_step_and_stride=True, umbral_frac=0.10
        )
        cog_ideal = generar_CoG_X_ideal(t, f_stride, 0.0, cycles_per_stride=1.0, amplitude_m=0.015)
        rmse_mm = calcular_rmse(cog_x * 1000.0, cog_ideal * 1000.0)

        mask = (t >= t_ini) & (t <= t_fin)
        t_win = t[mask] - t_ini
        real_mm = suavizar_serie(cog_x[mask]) * 1000.0
        ideal_mm = cog_ideal[mask] * 1000.0

        ax.fill_between(
            t_win,
            real_mm,
            ideal_mm,
            color=TEAL_LIGHT,
            alpha=0.38,
            label="Desviación respecto al patrón ideal",
        )
        ax.plot(
            t_win,
            real_mm,
            color=color_real,
            linewidth=2.3,
            label="CoG estimado (Filtro de Kalman)",
        )
        ax.plot(
            t_win,
            ideal_mm,
            color=CORAL,
            linestyle="--",
            linewidth=1.9,
            label="Modelo sinusoidal ideal simétrico (±15 mm)",
        )

        ax.axhline(0, color="#CBD5E1", linewidth=0.9, zorder=1)
        ax.set_title(
            f"{etiqueta}   |   Frec. zancada = {f_stride:.2f} Hz   |   RMSE global = {rmse_mm:.2f} mm",
            loc="left",
            color=NAVY,
            fontsize=11.5,
            pad=6,
        )
        ax.set_ylabel("Desplaz. ML (mm)")
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-68, 68)

    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor=TEAL_PRIMARY,
        fontsize=9.5,
    )
    axes[1].set_xlabel("Tiempo en régimen estacionario (s) [ventana t = 20–32 s]")

    fig.tight_layout(rect=[0, 0.01, 1, 0.89])
    out_path = OUT_DIR / "03_cog_kalman_velocidad.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generar_fig4_dispersion_isf_id(df: pd.DataFrame):
    """Figura 4: Dispersión entre ISF y diferencia de Impulso Dinámico por velocidad."""
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    velocidades = sorted(df["velocidad"].dropna().unique())
    resumen_spearman = []

    for vel, color in zip(velocidades, SPEED_PALETTE):
        sub = df[df["velocidad"] == vel].dropna(subset=["ISF", "diff_ID"])
        rho, p_val = spearmanr(sub["ISF"], sub["diff_ID"])
        resumen_spearman.append(f"{vel:.1f} m/s: ρ = {rho:+.2f}")
        ax.scatter(
            sub["ISF"],
            sub["diff_ID"],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=72,
            alpha=0.88,
            label=f"{vel:.1f} m/s (ρ = {rho:+.2f})",
            zorder=3,
        )

    ax.set_title(
        "Relación entre Índice de Simetría Frecuencial (ISF) y Diferencia de Impulso Dinámico (ΔID)",
        fontsize=12.0,
        fontweight="bold",
        color=NAVY,
        pad=12,
    )
    ax.set_xlabel("Índice de Simetría Frecuencial — ISF (%)")
    ax.set_ylabel("Diferencia absoluta de Impulso Dinámico — ΔID (N·s)")
    ax.set_ylim(-3, 86)
    ax.set_xlim(-6, 165)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Velocidad de marcha",
        title_fontsize=10.5,
        fontsize=9.8,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor=TEAL_PRIMARY,
    )

    rho_global, p_global = spearmanr(
        df.dropna(subset=["ISF", "diff_ID"])["ISF"],
        df.dropna(subset=["ISF", "diff_ID"])["diff_ID"],
    )
    ax.text(
        0.025,
        0.95,
        f"Tendencia global (n = 79): ρ = {rho_global:+.2f} ({_formatear_p(p_global)})\n"
        "Sin acoplamiento robusto entre asimetría frecuencial y de carga",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=NAVY,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F8FAFC",
            edgecolor=TEAL_PRIMARY,
            alpha=0.94,
        ),
    )

    fig.tight_layout()
    out_path = OUT_DIR / "04_dispersion_isf_id.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generar_fig5_segmentacion_impulso():
    """Figura 5: Mapa de presión plantar segmentado y curvas fuerza-tiempo con ID sombreado."""
    path = DATA_DIR / "Subject1_1p1.txt"

    # 1) Leer fuerzas puntuales del encabezado para la ventana fuerza-tiempo
    tiempos_ms, f_izq, f_der = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lstrip().startswith("Platform"):
                break
            campos = line.strip().split()
            if len(campos) >= 7:
                try:
                    tiempos_ms.append(float(campos[0]))
                    f_izq.append(float(campos[2]))
                    f_der.append(float(campos[3]))
                except ValueError:
                    continue

    t_s = (np.array(tiempos_ms) - tiempos_ms[0]) / 1000.0
    f_izq = np.array(f_izq)
    f_der = np.array(f_der)

    # 2) Seleccionar un frame nítido de doble apoyo donde ambos pies estén bien separados
    mejor_frame = None
    mejor_score = -1.0
    umbral_y = 27.928  # Umbral exacto calculado para Subject1_1p1 en resumen_1.txt
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in islice(f, 30000, 90000):
            vals = line.strip().split()
            if len(vals) == 129 and vals[0].startswith("y"):
                m = [[float(v) for v in vals[1:]]]
                for _ in range(55):
                    l2 = f.readline().strip().split()
                    if len(l2) == 129 and l2[0].startswith("y"):
                        m.append([float(v) for v in l2[1:]])
                if len(m) == 56:
                    arr = np.array(m)
                    grupos = mapear_pies(arr)
                    if len(grupos) == 2:
                        ys = [gr[0][3] for gr in grupos]
                        sizes = [len(gr) for gr in grupos]
                        if min(ys) < 24.5 and max(ys) > 33.0 and min(sizes) >= 60:
                            score = min(sizes) * float(arr.max())
                            if score > mejor_score:
                                mejor_score = score
                                mejor_frame = arr

    # Recortar columnas activas para mejor encuadre visual
    cols_activas = np.where(mejor_frame.max(axis=0) > 0.1)[0]
    c_min = max(0, int(cols_activas.min()) - 6)
    c_max = min(mejor_frame.shape[1] - 1, int(cols_activas.max()) + 6)
    mapa_recortado = mejor_frame[:, c_min : c_max + 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9), gridspec_kw={"width_ratios": [1.05, 1.25]})
    fig.suptitle(
        "Segmentación plantar e integración del Impulso Dinámico (ID) — Ejemplo representativo (Sujeto 1, 1.1 m/s)",
        fontsize=12.6,
        fontweight="bold",
        color=NAVY,
        y=0.98,
    )

    # Panel A: Mapa de presión de doble apoyo
    mapa_masked = np.ma.masked_where(mapa_recortado <= 0.05, mapa_recortado)
    cmap_presion = plt.cm.YlGnBu.copy()
    cmap_presion.set_bad(color="#F8FAFC")

    im = ax1.imshow(mapa_masked, cmap=cmap_presion, aspect="auto", origin="lower")
    ax1.axhline(umbral_y, color=CORAL, linestyle="--", linewidth=2.2, label=f"Umbral separación (y = {umbral_y:.1f})")
    ax1.text(0.04, 0.78, "Pie Derecho", transform=ax1.transAxes, color=NAVY, fontweight="bold", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=TEAL_PRIMARY, alpha=0.92))
    ax1.text(0.04, 0.16, "Pie Izquierdo", transform=ax1.transAxes, color=NAVY, fontweight="bold", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=TEAL_PRIMARY, alpha=0.92))
    ax1.set_title("A) Mapa de presión en doble apoyo (N/cm²)", loc="left", color=NAVY, fontsize=11.5)
    ax1.set_xlabel("Eje longitudinal (celdas × 8.46 mm)")
    ax1.set_ylabel("Eje transversal (celdas × 8.46 mm)")
    ax1.legend(loc="upper right", fontsize=9.0, frameon=True, facecolor="white")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label("Presión (N/cm²)", fontsize=9.5, color=NAVY)

    # Panel B: Curvas fuerza-tiempo y área = Impulso Dinámico
    mask_t = (t_s >= 10.0) & (t_s <= 13.8)
    tw = t_s[mask_t] - 10.0
    fi = f_izq[mask_t]
    fd = f_der[mask_t]

    ax2.fill_between(tw, 0, fi, color=TEAL_PRIMARY, alpha=0.24)
    ax2.fill_between(tw, 0, fd, color=CORAL, alpha=0.22)
    ax2.plot(tw, fi, color=TEAL_PRIMARY, linewidth=2.1, label=r"Pie Izquierdo ($\overline{\mathrm{ID}}_{\mathrm{izq}} = 203.3\ \mathrm{N}\cdot\mathrm{s}$)")
    ax2.plot(tw, fd, color=CORAL, linewidth=2.1, label=r"Pie Derecho ($\overline{\mathrm{ID}}_{\mathrm{der}} = 227.1\ \mathrm{N}\cdot\mathrm{s}$)")

    ax2.set_title("B) Integración espacio-temporal: ID = ∫ F(t) dt", loc="left", color=NAVY, fontsize=11.5)
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Fuerza vertical de reacción (N)")
    ax2.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(0, max(fi.max(), fd.max()) * 1.22)
    ax2.legend(loc="upper right", fontsize=9.3, frameon=True, facecolor="white", edgecolor="#CBD5E1")

    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    out_path = OUT_DIR / "05_segmentacion_plantar_impulso.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    configurar_estilo()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = cargar_datos_combinados()

    generadas = [
        generar_fig1_metricas_velocidad(df),
        generar_fig2_heatmap_spearman(),
        generar_fig3_cog_kalman_zoom(),
        generar_fig4_dispersion_isf_id(df),
        generar_fig5_segmentacion_impulso(),
    ]
    for ruta in generadas:
        print(f"Generada: {ruta}")


if __name__ == "__main__":
    main()
