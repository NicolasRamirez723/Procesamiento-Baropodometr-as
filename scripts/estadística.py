import os
import matplotlib.pyplot as plt
import numpy as np

def estimar_longitud(t_mayor,t_menor,longitud_mayor) -> float:

    """
    Se determinar la longitd de la pierna restante al asumir un modelo de pendulo invertido,
    se toma como referencia de marcha normal a la de menor velocidad, es decir, el 1er registro
    """

    return (t_menor/t_mayor) * longitud_mayor

def estadisticas(path="Baropodometrias Procesadas") -> list:

    """
    Se analizan todas los resultados de cada persona para obtener una lista que contenga
    las variables calculadas junto con su asimetría longitudinal, de esta manera vamos a poder 
    calcular estadísticas que permitan construir conclusiones
    """
    long = {"1":0.905, "2":1, "3":0.83, "4":0.925, "5":0.965, "6":1.3,"7":0.885,"8":0.905,"9":0.85,"10":0.92,"11":1,"12":0.94,"13":0.83,"14":0.89,"15":0.85,"16":0.96}
    #Lectura de los txt
    PATH = path
    num_paciente = 1
    for direc in os.listdir(path=PATH):

        print(f"Leyendo registros analizados paciente {direc}")

        try:
            num_paciente = float(direc[-2:])

            num_paciente = direc[-2:]
        except ValueError:
            num_paciente = direc[-1:]
        for muestra in os.listdir(os.path.join(PATH,direc)):

            for data in os.listdir(os.path.join(PATH,direc,muestra)):

                if data.endswith("txt") and muestra.endswith("1"):

                    archivo = os.path.join(PATH,direc,muestra,data)
                    with open(archivo,"r", encoding="utf-8", errors="ignore") as archive:
                        tiempo_l = None
                        tiempo_r = None
                        for line in archive:

                            try:

                                tiempo = line.split(",")[0]

                                if tiempo.split(":")[0] == "Tiempo de apoyo medio pie izquierdo":

                                    tiempo_l = float(tiempo.split(":")[1])

                                elif tiempo.split(":")[0] == "Tiempo de apoyo medio pie derecho":

                                    tiempo_r = float(tiempo.split(":")[1])
                                
                            except Exception:
                                continue

                    if tiempo_l is not None and tiempo_r is not None:
                        print(f"Longitud de miembro corto estimada: {estimar_longitud(max(tiempo_l,tiempo_r),min(tiempo_r,tiempo_l),longitud_mayor=long[str(num_paciente)])}")
    

if __name__ == "__main__":

    estadisticas()


