import csv
import os
import re


PATH_CSV = os.path.join("Estadisticas", "estadisticas.csv")


def estimar_longitud(t_mayor: float, t_menor: float, longitud_mayor: float) -> float:
    """Estima la longitud del miembro corto a partir de los tiempos de apoyo."""
    return (t_menor / t_mayor) * longitud_mayor


def leer_resumen(archivo: str) -> dict:
    """Extrae las métricas de un archivo resumen."""
    metricas = {
        "tiempo_izquierdo": None,
        "tiempo_derecho": None,
        "fuerza_izquierda": None,
        "fuerza_derecha": None,
        "isf": None,
        "impulso_derecho": None,
        "impulso_izquierdo": None,
    }

    with open(archivo, "r", encoding="utf-8", errors="ignore") as resumen:
        contenido = resumen.read()

    numero = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    patrones = {
        "tiempo_izquierdo": rf"Tiempo de apoyo medio pie izquierdo:\s*{numero}",
        "tiempo_derecho": rf"Tiempo de apoyo medio pie derecho:\s*{numero}",
        "fuerza_izquierda": rf"fuerza puntual media pie izquierdo\s*{numero}",
        "fuerza_derecha": rf"fuerza puntual media pie derecho\s*{numero}",
        "isf": rf"ISF\s*=\s*{numero}",
        "impulso_derecho": rf"promedio impulso derecho\s*=\s*{numero}",
        "impulso_izquierdo": rf"promedio impulso izquierdo\s*=\s*{numero}",
    }

    for nombre, patron in patrones.items():
        coincidencia = re.search(patron, contenido)
        if coincidencia:
            metricas[nombre] = float(coincidencia.group(1))

    return metricas


def estadisticas(path="Baropodometrias procesadas") -> list:
    """Genera un CSV con las métricas de todos los resúmenes disponibles."""
    longitudes = {
        "1": 0.905, "2": 1, "3": 0.83, "4": 0.925,
        "5": 0.965, "6": 1.3, "7": 0.885, "8": 0.905,
        "9": 0.85, "10": 0.92, "11": 1, "12": 0.94,
        "13": 0.83, "14": 0.89, "15": 0.85, "16": 0.96,
    }
    encabezados = [
        "num paciente",
        "muestra",
        "tiempo de apoyo menor",
        "tiempo de apoyo mayor",
        "fuerza puntual promedio izquierda",
        "fuerza puntual promedio derecha",
        "ISF",
        "promedio impulso derecho",
        "promedio impulso izquierdo",
        "longitud estimada del miembro corto",
        "longitud del miembro mayor",
    ]
    filas = []

    for paciente in sorted(os.listdir(path)):
        ruta_paciente = os.path.join(path, paciente)
        coincidencia_paciente = re.fullmatch(r"Subject(\d+)", paciente)
        if not os.path.isdir(ruta_paciente) or not coincidencia_paciente:
            continue

        num_paciente = coincidencia_paciente.group(1)
        print(f"Leyendo registros analizados paciente {num_paciente}")

        for muestra in sorted(os.listdir(ruta_paciente)):
            ruta_muestra = os.path.join(ruta_paciente, muestra)
            coincidencia_muestra = re.fullmatch(r"Muestra(\d+)", muestra)
            if not os.path.isdir(ruta_muestra) or not coincidencia_muestra:
                continue

            resumenes = [
                archivo for archivo in os.listdir(ruta_muestra)
                if archivo.startswith("resumen_") and archivo.endswith(".txt")
            ]
            for nombre_archivo in resumenes:
                metricas = leer_resumen(os.path.join(ruta_muestra, nombre_archivo))
                tiempos = [metricas["tiempo_izquierdo"], metricas["tiempo_derecho"]]
                if any(valor is None for valor in tiempos):
                    print(f"Se omite {nombre_archivo}: faltan tiempos de apoyo")
                    continue

                tiempo_menor = min(tiempos)
                tiempo_mayor = max(tiempos)
                longitud_mayor = longitudes.get(num_paciente)
                longitud_corta = (
                    estimar_longitud(tiempo_mayor, tiempo_menor, longitud_mayor)
                    if longitud_mayor is not None and tiempo_mayor != 0
                    else None
                )
                filas.append([
                    int(num_paciente),
                    int(coincidencia_muestra.group(1)),
                    tiempo_menor,
                    tiempo_mayor,
                    metricas["fuerza_izquierda"],
                    metricas["fuerza_derecha"],
                    metricas["isf"],
                    metricas["impulso_derecho"],
                    metricas["impulso_izquierdo"],
                    longitud_corta,
                    longitud_mayor,
                ])

    os.makedirs(os.path.dirname(PATH_CSV), exist_ok=True)
    with open(PATH_CSV, "w", newline="", encoding="utf-8") as archivo_csv:
        escritor = csv.writer(archivo_csv)
        escritor.writerow(encabezados)
        escritor.writerows(filas)

    print(f"CSV generado: {PATH_CSV} ({len(filas)} filas)")
    return filas


if __name__ == "__main__":
    estadisticas()


