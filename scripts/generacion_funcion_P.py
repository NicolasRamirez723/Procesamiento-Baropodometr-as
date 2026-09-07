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

"""
mapa_presiones = []
linea_inicio_datos = 30000
filas_por_mapa = 56

import numpy as np
from scipy.ndimage import label, center_of_mass

def mapear_pies(frame):
    """
    Función optimizada con SciPy.
    Recibe un frame de la placa de presión, identifica las manchas de apoyo,
    calcula su centro de masa y clasifica las coordenadas en pie derecho (y > 28)
    o izquierdo (y <= 28).
    """
    frame = np.asarray(frame)
    pie_izquierdo, pie_derecho = [], []
    
    mascara = frame > 0
    
    
    if not np.any(mascara):
        return pie_derecho, pie_izquierdo

    
    matriz_etiquetada, num_grupos = label(mascara)
    
    
    etiquetas = np.arange(1, num_grupos + 1)
    centros = center_of_mass(frame, matriz_etiquetada, etiquetas)
    
    
    if num_grupos == 1:
        centros = [centros]

   
    for i, centro in enumerate(centros):
        id_grupo = i + 1
        centro = np.asarray(centro).ravel()
        y_CM = float(centro[0])
        
       
        filas_grupo, cols_grupo = np.where(matriz_etiquetada == id_grupo)
        
        valores_grupo = frame[filas_grupo, cols_grupo]
        

        datos_grupo = list(zip(valores_grupo, filas_grupo, cols_grupo))
        
        if y_CM > 28:
            pie_derecho.extend(datos_grupo)
        else:
            pie_izquierdo.extend(datos_grupo)

    return pie_derecho, pie_izquierdo
"""
def prueba_1000_frames(max_frames=1000):
    Prueba la segmentación de pies en una cantidad acotada de frames.
    PATH = "data/muestras/The nature of functional variability in plantar pressure during a range of controlled walking speeds"

    frames_analizados = 0
    total_derecho = 0
    total_izquierdo = 0
    total_ambos = 0
    total_vacios = 0

    for archivo in sorted(os.listdir(path=PATH)):
        if not archivo.endswith(".txt"):
            continue

        NEW_PATH = os.path.join(PATH, archivo)

        with open(NEW_PATH, "r", encoding="utf-8") as archive:
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

                pie_derecho, pie_izquierdo = mapear_pies(mapa_presiones)

                y_cm_d = np.mean([fila for _, fila, _ in pie_derecho]) if pie_derecho else None
                y_cm_i = np.mean([fila for _, fila, _ in pie_izquierdo]) if pie_izquierdo else None

                print(
                    f"Frame {frames_analizados}: "
                    f"Ycm_d={y_cm_d} | celdas_d={len(pie_derecho)} | "
                    f"Ycm_i={y_cm_i} | celdas_i={len(pie_izquierdo)}"
                )

                if pie_derecho and not pie_izquierdo:
                    total_derecho += 1
                elif pie_izquierdo and not pie_derecho:
                    total_izquierdo += 1
                elif pie_derecho and pie_izquierdo:
                    total_ambos += 1
                else:
                    total_vacios += 1

                frames_analizados += 1

                if frames_analizados >= max_frames:
                    break

            if frames_analizados >= max_frames:
                break

    print(f"Frames analizados: {frames_analizados}")
    print(f"Solo derecho: {total_derecho}")
    print(f"Solo izquierdo: {total_izquierdo}")
    print(f"Ambos: {total_ambos}")
    print(f"Vacios: {total_vacios}")

    return {
        "frames_analizados": frames_analizados,
        "solo_derecho": total_derecho,
        "solo_izquierdo": total_izquierdo,
        "ambos": total_ambos,
        "vacios": total_vacios,
    }

"""
def generar_funcion_P():
    time = 0
    # A partir de los registros baropodométricos se reconstruye una función de la presión
    # en función de x, y y t.


    frames_pie_derecho = []
    frames_pie_izquierdo = []



    PATH = "data/muestras/The nature of functional variability in plantar pressure during a range of controlled walking speeds"

    for archivo in sorted(os.listdir(path=PATH)):

        if not archivo.endswith(".txt"):
            continue

        NEW_PATH = os.path.join(PATH, archivo)

        with open(NEW_PATH, "r", encoding="utf-8") as archive:

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

                pie_derecho,pie_izquierdo = mapear_pies(mapa_presiones)

                print(f"Leyendo frame t={time} ms | archivo={archivo}")
                frames_pie_derecho.append((pie_derecho,time))

                frames_pie_izquierdo.append((pie_izquierdo,time))

                time += 10

    return frames_pie_derecho,frames_pie_izquierdo


if __name__ == "__main__":
    # Validación rápida de una ventana sin recorrer todo el dataset.
    #prueba_1000_frames(1000)

    # Si querés correr la versión completa, descomentá estas líneas:
    frames_placa = generar_funcion_P()
    # print(f"Para el frame 10 el mapa de presiones será: f{frames_placa[9]}")


