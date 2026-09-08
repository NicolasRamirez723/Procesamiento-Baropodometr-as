from itertools import islice
from collections import deque
import os
import numpy as np
"""
Datos plataforma:
128 columnas
56 lineas
Ancho de celda = 8.46mm
Largo de celda = 8.46mm
presion: N/cm2
"""
mapa_presiones = []
linea_inicio_datos = 30000
filas_por_mapa = 56

import numpy as np
from scipy.ndimage import label, center_of_mass

def mapear_pies(frame):
    """
    Función optimizada con SciPy.
    Recibe un frame de la placa de presión e identifica todas las manchas de
    apoyo. El umbral de separación se calcula después de leer los frames, por
    lo que esta función no clasifica todavía derecha e izquierda.
    """
    frame = np.asarray(frame)
    mascara = frame > 0

    if not np.any(mascara):
        return []

    matriz_etiquetada, num_grupos = label(mascara)
    etiquetas = np.arange(1, num_grupos + 1)
    centros = center_of_mass(frame, matriz_etiquetada, etiquetas)

    grupos = []
    for i, centro in enumerate(centros):
        id_grupo = i + 1
        centro = np.asarray(centro).ravel()
        y_CM_grupo = float(centro[0])
        filas_grupo, cols_grupo = np.where(matriz_etiquetada == id_grupo)
        valores_grupo = frame[filas_grupo, cols_grupo]
        grupos.append(list(zip(
            valores_grupo,
            filas_grupo,
            cols_grupo,
            [y_CM_grupo] * len(valores_grupo),
        )))

    return grupos


def integrar_impulsos(fuerzas, tiempos, dt_default):
    """Integra cada apoyo con la regla trapezoidal entre sus extremos."""
    impulsos = []
    impulso_actual = 0.0
    apoyo = False
    fuerza_anterior = 0.0

    for indice, fuerza in enumerate(fuerzas):
        if indice == 0 or tiempos[indice] is None or tiempos[indice - 1] is None:
            dt_frame = dt_default
        else:
            dt_frame = (tiempos[indice] - tiempos[indice - 1]) / 1000.0
            if dt_frame <= 0:
                dt_frame = dt_default

        if fuerza > 0:
            if not apoyo:
                apoyo = True
                fuerza_anterior = 0.0
            impulso_actual += (fuerza_anterior + fuerza) * dt_frame / 2
            fuerza_anterior = fuerza
        elif apoyo:
            impulso_actual += fuerza_anterior * dt_frame / 2
            impulsos.append(impulso_actual)
            impulso_actual = 0.0
            fuerza_anterior = 0.0
            apoyo = False

    if apoyo:
        impulsos.append(impulso_actual)

    return impulsos


def CALCULO_ID():

    """
    Se calcula el impuslo dinámico promedio asociado a cada pie durante cada muestra
    """

    # A partir de los registros baropodométricos se reconstruye una función de la presión para cada pie
    # en función de x, y y t.


    # Presión en N/cm²; las celdas miden 0.846 cm por lado.
    area_celda = 0.846 ** 2
    dt_default = 0.010
    PATH = "data/muestras/The nature of functional variability in plantar pressure during a range of controlled walking speeds"

    for archivo in sorted(os.listdir(path=PATH)):

        if not archivo.endswith(".txt"):
            continue

        NEW_PATH = os.path.join(PATH, archivo)
        num_paciente = archivo.split("_")[0][-1]

        num_muestra = archivo.split(".")[0][-1]

        frames_procesados = []
        tiempos_frames = []
        centros_masa = []
        tiempo_mapa_actual = None
        with open(NEW_PATH, "r", encoding="utf-8") as archive:

            for line in islice(archive, linea_inicio_datos, None):
                valores = line.strip().split()

                if len(valores) >= 3 and valores[0].lower().startswith("time,"):
                    try:
                        tiempo_mapa_actual = float(valores[-1])
                    except ValueError:
                        pass
                    continue

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

                grupos = mapear_pies(mapa_presiones)
                frames_procesados.append(grupos)
                tiempos_frames.append(tiempo_mapa_actual)
                centros_masa.extend(grupo[0][3] for grupo in grupos if grupo)

        if not centros_masa:
            print(f"{archivo}: no se encontraron apoyos.")
            continue

        # El umbral se obtiene de la lectura completa de esta muestra.
        umbral = float(np.mean(centros_masa))
        fuerzas_derechas = []
        fuerzas_izquierdas = []

        for grupos in frames_procesados:
            pie_derecho = [celda for grupo in grupos if grupo[0][3] > umbral for celda in grupo]
            pie_izquierdo = [celda for grupo in grupos if grupo[0][3] <= umbral for celda in grupo]
            fuerzas_derechas.append(sum(celda[0] for celda in pie_derecho) * area_celda)
            fuerzas_izquierdas.append(sum(celda[0] for celda in pie_izquierdo) * area_celda)

        impulsos_derechos = integrar_impulsos(fuerzas_derechas, tiempos_frames, dt_default)
        impulsos_izquierdos = integrar_impulsos(fuerzas_izquierdas, tiempos_frames, dt_default)
        promedio_derecho = float(np.mean(impulsos_derechos)) if impulsos_derechos else 0.0
        promedio_izquierdo = float(np.mean(impulsos_izquierdos)) if impulsos_izquierdos else 0.0

        resultado = (
            f"{archivo}: umbral y={umbral:.3f}, "
            f"pisadas derecha={len(impulsos_derechos)}, "
            f"promedio impulso derecho={promedio_derecho:.4f} N·s | "
            f"pisadas izquierda={len(impulsos_izquierdos)}, "
            f"promedio impulso izquierdo={promedio_izquierdo:.4f} N·s\n"
        )
        print(resultado, end="")

        ruta_salida = os.path.join(
            "Baropodometrias Procesadas",
            f"Subject{num_paciente}",
            f"Muestra{num_muestra}",
            f"resumen_{num_muestra}.txt",
        )
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        with open(ruta_salida, "a", newline="", encoding="utf-8") as archive:
            archive.write(resultado)

if __name__ == "__main__":
    # Validación rápida de una ventana sin recorrer todo el dataset.
    #prueba_1000_frames(1000)

    # Si querés correr la versión completa, descomentá estas líneas:
    frames_placa = CALCULO_ID()
    # print(f"Para el frame 10 el mapa de presiones será: f{frames_placa[9]}")


