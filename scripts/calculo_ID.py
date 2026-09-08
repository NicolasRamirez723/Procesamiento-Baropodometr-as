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

        impulsos_derecho = []
        impulsos_izquierdo = []
        impulso_derecho_actual = 0.0
        impulso_izquierdo_actual = 0.0
        apoyo_derecho = False
        apoyo_izquierdo = False

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


def CALCULO_ID():

    """
    Se calcula el impuslo dinámico promedio asociado a cada pie durante cada muestra
    """

    # A partir de los registros baropodométricos se reconstruye una función de la presión para cada pie
    # en función de x, y y t.


    # Presión en N/cm²; las celdas miden 0.846 cm por lado.
    area_celda = 0.846 ** 2
    dt = 0.010



    PATH = "data/muestras/The nature of functional variability in plantar pressure during a range of controlled walking speeds"

    for archivo in sorted(os.listdir(path=PATH)):

        if not archivo.endswith(".txt"):
            continue

        NEW_PATH = os.path.join(PATH, archivo)
        num_paciente = archivo.split("_")[0][-1]
        
        num_muestra = archivo.split(".")[0][-1]
                        
        impulsos_derecho = []
        impulsos_izquierdo = []
        impulso_derecho_actual = 0.0
        impulso_izquierdo_actual = 0.0
        apoyo_derecho = False
        apoyo_izquierdo = False

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

                presion_derecha = sum(
                    presion for presion, _, _ in pie_derecho
                )
                presion_izquierda = sum(
                    presion for presion, _, _ in pie_izquierdo
                )

                fuerza_derecha = presion_derecha * area_celda
                fuerza_izquierda = presion_izquierda * area_celda

                if presion_derecha > 0:
                    apoyo_derecho = True
                    impulso_derecho_actual += fuerza_derecha * dt
                elif apoyo_derecho:
                    impulsos_derecho.append(impulso_derecho_actual)
                    impulso_derecho_actual = 0.0
                    apoyo_derecho = False

                if presion_izquierda > 0:
                    apoyo_izquierdo = True
                    impulso_izquierdo_actual += fuerza_izquierda * dt
                elif apoyo_izquierdo:
                    impulsos_izquierdo.append(impulso_izquierdo_actual)
                    impulso_izquierdo_actual = 0.0
                    apoyo_izquierdo = False

                

        if apoyo_derecho:
            impulsos_derecho.append(impulso_derecho_actual)
        if apoyo_izquierdo:
            impulsos_izquierdo.append(impulso_izquierdo_actual)

        promedio_derecho = (
            np.mean(impulsos_derecho) if impulsos_derecho else 0.0
        )
        promedio_izquierdo = (
            np.mean(impulsos_izquierdo) if impulsos_izquierdo else 0.0
        )

        print(
            f"{archivo}: "
            f"pisadas derecha={len(impulsos_derecho)}, "
            f"promedio impulso derecho={promedio_derecho:.4f} N·s | "
            f"pisadas izquierda={len(impulsos_izquierdo)}, "
            f"promedio impulso izquierdo={promedio_izquierdo:.4f} N·s"
        )

        PATH_ESCRITURA = os.path.join("Baropodometrias Procesadas",f"Subject{num_paciente}",f"Muestra{num_muestra}",f"resumen_{num_muestra}.txt")

        with open(PATH_ESCRITURA,"a",newline="") as archive:

            archive.write(f"{archivo}: "
                        f"pisadas derecha={len(impulsos_derecho)}, "
                        f"promedio impulso derecho={promedio_derecho:.4f} N·s | "
                        f"pisadas izquierda={len(impulsos_izquierdo)}, "
                        f"promedio impulso izquierdo={promedio_izquierdo:.4f} N·s\n")

if __name__ == "__main__":
    # Validación rápida de una ventana sin recorrer todo el dataset.
    #prueba_1000_frames(1000)

    # Si querés correr la versión completa, descomentá estas líneas:
    frames_placa = CALCULO_ID()
    # print(f"Para el frame 10 el mapa de presiones será: f{frames_placa[9]}")


