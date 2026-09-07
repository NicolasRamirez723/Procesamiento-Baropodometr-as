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



def mapear_pies(frame):

    """
    La función recibe un frame completo de la placa de presión y mapea la presión no nula 
    registrada, se asume que si la posición está por encima de y=28, entonces corresponde al pie 
    derecho y si está por debajo al pie izquierdo, esto ya que la persona está caminando de izquierda a derecha
    """

    frame = np.asarray(frame)

    """
    La lógica es esta, se recorre el frame y cuando se detecta una posición no nula se empieza a realizar un
    mapeo alrededor de esta, es decir, si es la posición i,j, analizaremos un cuadrado alrededor de este valor
    y guardaremos la posición, si volvemos a encontrar un valor no nulo repetimos el proceso hasta que se mapea por completo
    la presión plantar del pie correspondiente
    """
    filas_tot, col_tot = frame.shape
    visitados = set()
    pie_izquierdo, pie_derecho = [], []

    for num_fila in range(filas_tot):
        for num_columna in range(col_tot):
            
            
            if frame[num_fila, num_columna] > 0 and (num_fila, num_columna) not in visitados:
                
                
                datos_pie_actual = []
                
                pendientes = deque([(num_fila,num_columna)])
                visitados.add((num_fila, num_columna))

                while pendientes:
                    f, c = pendientes.popleft()
                    valor = frame[f, c]
                    datos_pie_actual.append((valor, f, c))
                    
                    vecinos = [(f, c+1), (f, c-1), (f+1, c), (f-1, c)]
                    for vec_f, vec_c in vecinos:
                        if 0 <= vec_f < filas_tot and 0 <= vec_c < col_tot:
                            if (vec_f, vec_c) not in visitados:
                                if frame[vec_f, vec_c] > 0:
                                    visitados.add((vec_f, vec_c))
                                    pendientes.append((vec_f, vec_c))

                
                y_CM = np.mean([fila for valor, fila, col in datos_pie_actual])

                # Clasificación
                # Se acumula cada componente conectado en su pie correspondiente;
                # no se sobrescribe el pie completo en cada región encontrada.
                if y_CM > 28:
                    pie_derecho.extend(datos_pie_actual)
                else:
                    pie_izquierdo.extend(datos_pie_actual)

    return pie_derecho, pie_izquierdo




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


"""                      
def validar_primeros_frames(n_frames=1000):
    PATH = "data/muestras/The nature of functional variability in plantar pressure during a range of controlled walking speeds"
    time = 0

    for archivo in sorted(os.listdir(path=PATH)):
        if not archivo.endswith(".txt"):
            continue

        NEW_PATH = os.path.join(PATH, archivo)

        with open(NEW_PATH, "r", encoding="utf-8") as archive:
            frames_validados = 0

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

                y_cm_d = np.mean([f for _, f, _ in pie_derecho]) if pie_derecho else None
                y_cm_i = np.mean([f for _, f, _ in pie_izquierdo]) if pie_izquierdo else None

                print(
                    f"Frame {frames_validados}: t={time} ms | "
                    f"derecho={len(pie_derecho)} celdas | "
                    f"izquierdo={len(pie_izquierdo)} celdas | "
                    f"yCM_d={y_cm_d} | yCM_i={y_cm_i}"
                )

                frames_validados += 1
                time += 10

                if frames_validados >= n_frames:
                    return

"""

if __name__ == "__main__":
    # Validación rápida de los primeros frames antes de correr todo el dataset.
    #validar_primeros_frames(1000)

    
    frames_placa = generar_funcion_P()
    



