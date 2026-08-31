from itertools import islice
import matplotlib.pyplot as plt
import os
import numpy as np
"""
Datos plataforma:
128 columnas
56 lineas
Ancho de celda = 8.46mm
Largo de celda = 8.46mm

"""
mapa_presiones = []
linea_inicio_datos = 30000
filas_por_mapa = 56
frames_por_montaje = 5


def recortar_huella(frame):
    frame = np.asarray(frame)
    filas, columnas = np.where(frame > 0)

    if len(filas) == 0:
        return frame

    margen = 2
    fila_inicio = max(filas.min() - margen, 0)
    fila_fin = min(filas.max() + margen + 1, frame.shape[0])
    columna_inicio = max(columnas.min() - margen, 0)
    columna_fin = min(columnas.max() + margen + 1, frame.shape[1])

    return np.rot90(frame[fila_inicio:fila_fin, columna_inicio:columna_fin])


def generar_funcion_P(graficar=True):
    # A partir de los registros baropodométricos se reconstruye una función de la presión
    # en función de x, y y t.
    todos_los_frames = []

    PATH = "data/muestras/The nature of functional variability in plantar pressure during a range of controlled walking speeds"

    for archivo in sorted(os.listdir(path=PATH)):

        if not archivo.endswith(".txt"):
            continue

        NEW_PATH = os.path.join(PATH, archivo)
        nombre_base = os.path.splitext(archivo)[0]
        paciente = nombre_base.split("_")[0]

        with open(NEW_PATH, "r", encoding="utf-8") as archive:
            frames_mapa_calor = []

            for line in islice(archive, linea_inicio_datos, None):
                valores = line.strip().split()

                if len(valores) != 129 or not valores[0].startswith("y"):
                    continue

                try:
                    primera_fila = [float(valor) for valor in valores[1:]]
                except ValueError:
                    continue

                mapa_presiones = [primera_fila]

                for fila in islice(archive, filas_por_mapa - 1):
                    valores = fila.strip().split()

                    if len(valores) != 129 or not valores[0].startswith("y"):
                        break

                    try:
                        mapa_presiones.append([float(valor) for valor in valores[1:]])
                    except ValueError:
                        break

                if len(mapa_presiones) == filas_por_mapa:
                    frames_mapa_calor.append(mapa_presiones)

            todos_los_frames.append(frames_mapa_calor)

            if graficar:
                path_dir = os.path.join(
                    "Baropodometrias procesadas",
                    paciente,
                    "Mapa de calor",
                    nombre_base,
                )
                os.makedirs(path_dir, exist_ok=True)

                for inicio in range(0, len(frames_mapa_calor), frames_por_montaje):
                    grupo = frames_mapa_calor[inicio:inicio + frames_por_montaje]
                    huellas = [recortar_huella(frame) for frame in grupo]
                    escala_maxima = max(np.max(huella) for huella in huellas)

                    fig, axes = plt.subplots(1, len(huellas), figsize=(10, 4))
                    axes = np.atleast_1d(axes)

                    for posicion, (ax, huella) in enumerate(zip(axes, huellas)):
                        imagen = ax.imshow(
                            huella,
                            cmap="jet",
                            origin="upper",
                            aspect="equal",
                            vmin=0,
                            vmax=escala_maxima,
                            interpolation="nearest",
                        )
                        ax.set_title(f"Frame {inicio + posicion}")
                        ax.axis("off")

                    fig.colorbar(imagen, ax=axes.tolist(), label="Presión", shrink=0.8)
                    fig.suptitle(f"Paciente {paciente}, registro {nombre_base}")
                    fig.tight_layout()

                    numero_montaje = inicio // frames_por_montaje
                    dir_destino = os.path.join(
                        path_dir,
                        f"montaje_{numero_montaje:04d}.png",
                    )
                    fig.savefig(dir_destino, dpi=150, bbox_inches="tight")
                    plt.close(fig)

    return todos_los_frames


if __name__ == "__main__":
    #Como dato extra se genera el mapa de calor de la baropodometría dinámica para
    #visualizarla frame a frame
    frames_placa = generar_funcion_P()

    print(f"Para el frame 10 el mapa de presiones será: f{frames_placa[9]}")



