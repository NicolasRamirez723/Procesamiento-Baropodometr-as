import csv
import math
import os
import re

import statsmodels.api as sm


PATH_CSV = os.path.join("Estadisticas", "estadisticas.csv")
PATH_RESULTADOS = os.path.join("Estadisticas", "resultados_estadisticos.txt")
LONGITUD_CORTA = "longitud estimada del miembro corto"
LONGITUD_MAYOR = "longitud del miembro mayor"
ISF = "ISF"
IMPULSO_DERECHO = "promedio impulso derecho"
IMPULSO_IZQUIERDO = "promedio impulso izquierdo"
MUESTRA = "muestra"


def estimar_longitud(t_mayor: float, t_menor: float, longitud_mayor: float) -> float:
    """Estima la longitud del miembro corto a partir de los tiempos de apoyo."""
    return (t_menor / t_mayor) * longitud_mayor


def leer_resumen(archivo: str) -> dict:
    """Extrae las métricas de un archivo resumen."""
    metricas: dict[str, float | None] = {
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


def leer_estadisticas(path=PATH_CSV) -> list:
    """Lee las filas del CSV de estadísticas."""
    with open(path, "r", newline="", encoding="utf-8") as archivo_csv:
        return list(csv.DictReader(archivo_csv))


def diferencia_impulso(fila: dict) -> float:
    """Calcula el impulso dinámico como la diferencia absoluta entre pies."""
    return abs(float(fila[IMPULSO_DERECHO]) - float(fila[IMPULSO_IZQUIERDO]))


def agrupar_por_muestra(filas: list) -> dict:
    """Agrupa los registros por número de muestra."""
    grupos = {}
    for fila in filas:
        muestra = int(fila[MUESTRA])
        grupos.setdefault(muestra, []).append(fila)
    return dict(sorted(grupos.items()))


def _rango_promedio(valores: list) -> list:
    """Asigna rangos promedio cuando hay valores empatados."""
    ordenados = sorted(enumerate(valores), key=lambda elemento: elemento[1])
    rangos = [0.0] * len(valores)
    inicio = 0

    while inicio < len(ordenados):
        fin = inicio + 1
        while fin < len(ordenados) and ordenados[fin][1] == ordenados[inicio][1]:
            fin += 1
        rango = (inicio + 1 + fin) / 2
        for posicion in range(inicio, fin):
            rangos[ordenados[posicion][0]] = rango
        inicio = fin

    return rangos


def _correlacion_spearman(x: list, y: list):
    """Calcula Spearman con la fórmula de rangos y corrección para empates."""
    if len(x) != len(y) or len(x) < 2:
        return None

    rango_x = _rango_promedio(x)
    rango_y = _rango_promedio(y)
    cantidad = len(x)

    def correccion_empates(valores: list) -> int:
        frecuencias = {}
        for valor in valores:
            frecuencias[valor] = frecuencias.get(valor, 0) + 1
        return sum(frecuencia**3 - frecuencia for frecuencia in frecuencias.values())

    base = cantidad**3 - cantidad
    denominador_x = base - correccion_empates(x)
    denominador_y = base - correccion_empates(y)
    if denominador_x == 0 or denominador_y == 0:
        return None

    diferencia_cuadrados = sum(
        (rango_x[i] - rango_y[i]) ** 2 for i in range(cantidad)
    )
    numerador = (denominador_x + denominador_y) / 2 - 6 * diferencia_cuadrados
    return numerador / math.sqrt(denominador_x * denominador_y)


def _spearman_asimetria_isf_muestra(filas: list):
    pares = [
        (
            float(fila[LONGITUD_MAYOR]) - float(fila[LONGITUD_CORTA]),
            float(fila[ISF]),
        )
        for fila in filas
        if fila.get(LONGITUD_MAYOR) and fila.get(LONGITUD_CORTA) and fila.get(ISF)
    ]
    return _correlacion_spearman(
        [par[0] for par in pares], [par[1] for par in pares]
    )


def spearman_asimetria_isf(filas: list) -> dict:
    """Calcula Spearman entre asimetría e ISF independientemente por muestra."""
    return {
        muestra: _spearman_asimetria_isf_muestra(filas_muestra)
        for muestra, filas_muestra in agrupar_por_muestra(filas).items()
    }


def _spearman_asimetria_impulso_muestra(filas: list):
    pares = [
        (
            float(fila[LONGITUD_MAYOR]) - float(fila[LONGITUD_CORTA]),
            diferencia_impulso(fila),
        )
        for fila in filas
        if fila.get(LONGITUD_MAYOR)
        and fila.get(LONGITUD_CORTA)
        and fila.get(IMPULSO_DERECHO)
        and fila.get(IMPULSO_IZQUIERDO)
    ]
    return _correlacion_spearman(
        [par[0] for par in pares], [par[1] for par in pares]
    )


def spearman_asimetria_impulso(filas: list) -> dict:
    """Calcula Spearman entre asimetría y diferencia de impulsos por muestra."""
    return {
        muestra: _spearman_asimetria_impulso_muestra(filas_muestra)
        for muestra, filas_muestra in agrupar_por_muestra(filas).items()
    }


def _regresion_isf_impulso_muestra(filas: list) -> dict | None:
    pares = [
        (diferencia_impulso(fila), float(fila[ISF]))
        for fila in filas
        if fila.get(ISF) and fila.get(IMPULSO_DERECHO) and fila.get(IMPULSO_IZQUIERDO)
    ]
    if len(pares) < 3:
        return None

    impulso = [par[0] for par in pares]
    isf = [par[1] for par in pares]
    modelo = sm.OLS(isf, sm.add_constant(impulso)).fit()
    return {
        "intercepto": float(modelo.params[0]),
        "pendiente": float(modelo.params[1]),
        "p_value": float(modelo.pvalues[1]),
        "r_squared": float(modelo.rsquared),
    }


def regresion_isf_impulso(filas: list) -> dict:
    """Ajusta la regresión de ISF e impulso dinámico por muestra."""
    return {
        muestra: _regresion_isf_impulso_muestra(filas_muestra)
        for muestra, filas_muestra in agrupar_por_muestra(filas).items()
    }


def guardar_resultados(filas: list, path=PATH_RESULTADOS) -> str:
    """Guarda en un TXT los cálculos estadísticos separados por muestra."""
    grupos = agrupar_por_muestra(filas)
    spearman_asimetria = spearman_asimetria_isf(filas)
    spearman_asimetria_impulso_resultados = spearman_asimetria_impulso(filas)
    regresiones = regresion_isf_impulso(filas)

    def formato_numero(valor) -> str:
        return "No calculable" if valor is None else f"{valor:.6f}"

    lineas = ["Resultados estadísticos por muestra", ""]
    for muestra, filas_muestra in grupos.items():
        lineas.append(f"Muestra {muestra} (n={len(filas_muestra)})")
        lineas.append(
            "Spearman asimetría de longitud (longitud mayor - menor) e ISF: "
            f"{formato_numero(spearman_asimetria[muestra])}"
        )
        lineas.append(
            "Spearman asimetría de longitud y diferencia absoluta de impulsos: "
            f"{formato_numero(spearman_asimetria_impulso_resultados[muestra])}"
        )

        regresion = regresiones[muestra]
        if regresion is None:
            lineas.append("Regresión lineal: No calculable")
        else:
            lineas.append(
                "Regresión lineal (ISF = intercepto + pendiente * diferencia de impulsos): "
                f"intercepto={formato_numero(regresion['intercepto'])}, "
                f"pendiente={formato_numero(regresion['pendiente'])}, "
                f"p-valor={formato_numero(regresion['p_value'])}, "
                f"R²={formato_numero(regresion['r_squared'])}"
            )
        lineas.append("")

    directorio = os.path.dirname(path)
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    with open(path, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas))
    return path


if __name__ == "__main__":
    datos = leer_estadisticas()
    ruta_resultados = guardar_resultados(datos)
    print(f"Resultados guardados en: {ruta_resultados}")


