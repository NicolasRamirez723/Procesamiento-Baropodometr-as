import csv
import math
import os
import random
import re

import matplotlib.pyplot as plt
import statsmodels.api as sm


PATH_CSV = os.path.join("Estadisticas", "estadisticas.csv")
PATH_RESULTADOS = os.path.join("Estadisticas", "resultados_estadisticos.txt")
PATH_GRAFICA_ISF_ID = os.path.join("Estadisticas", "mapa_dispersion_isf_id.png")
LONGITUD_CORTA = "longitud estimada del miembro corto"
LONGITUD_MAYOR = "longitud del miembro mayor"
ISF = "ISF"
IMPULSO_DERECHO = "promedio impulso derecho"
IMPULSO_IZQUIERDO = "promedio impulso izquierdo"
MUESTRA = "muestra"


def estimar_longitud(t_mayor: float, t_menor: float, longitud_mayor: float) -> float:
    """Estima la longitud del miembro corto a partir de los tiempos de apoyo."""
    return ((t_menor / t_mayor)**2) * longitud_mayor


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


def graficar_isf_id(filas: list, path=PATH_GRAFICA_ISF_ID) -> str:
    """Guarda la dispersión ISF-ID, diferenciando las muestras por color."""
    grupos = {}
    for fila in filas:
        if not all(
            _es_numero_valido(fila.get(columna))
            for columna in (ISF, IMPULSO_DERECHO, IMPULSO_IZQUIERDO, MUESTRA)
        ):
            continue

        muestra = int(fila[MUESTRA])
        grupos.setdefault(muestra, ([], []))
        valores_isf, valores_id = grupos[muestra]
        valores_isf.append(float(fila[ISF]))
        valores_id.append(diferencia_impulso(fila))

    if not grupos:
        raise ValueError("No hay datos numéricos válidos para graficar ISF e ID.")

    figura, eje = plt.subplots(figsize=(9, 6))
    for muestra, (valores_isf, valores_id) in sorted(grupos.items()):
        eje.scatter(valores_isf, valores_id, label=f"Muestra {muestra}", alpha=0.75)

    eje.set_title("Relación entre ISF e ID por muestra")
    eje.set_xlabel("ISF")
    eje.set_ylabel("ID: diferencia absoluta de impulsos (N·s)")
    eje.grid(True, alpha=0.3)
    eje.legend(title="Muestra")
    figura.tight_layout()

    directorio = os.path.dirname(path)
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    figura.savefig(path, dpi=150)
    plt.close(figura)
    return path


def _es_numero_valido(valor) -> bool:
    if valor is None or str(valor).strip() == "":
        return False
    try:
        return math.isfinite(float(valor))
    except (TypeError, ValueError):
        return False


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


def _spearman_isf_impulso_muestra(filas: list, semilla: int) -> dict | None:
    pares = [
        (float(fila[ISF]), diferencia_impulso(fila))
        for fila in filas
        if all(
            _es_numero_valido(fila.get(columna))
            for columna in (ISF, IMPULSO_DERECHO, IMPULSO_IZQUIERDO)
        )
    ]
    if len(pares) < 2:
        return None

    isf = [par[0] for par in pares]
    impulso = [par[1] for par in pares]
    rho = _correlacion_spearman(isf, impulso)
    if rho is None:
        return None

    permutaciones = 9999
    generador = random.Random(semilla)
    extremos = 0
    for _ in range(permutaciones):
        rho_permutado = _correlacion_spearman(isf, generador.sample(impulso, len(impulso)))
        if rho_permutado is not None and abs(rho_permutado) >= abs(rho) - 1e-12:
            extremos += 1

    return {
        "n": len(pares),
        "rho": rho,
        "p_value": (extremos + 1) / (permutaciones + 1),
    }


def spearman_isf_impulso(filas: list) -> dict:
    """Calcula Spearman ISF-impulso y p por permutación dentro de cada muestra."""
    resultados = {
        muestra: _spearman_isf_impulso_muestra(filas_muestra, 20261007 + muestra)
        for muestra, filas_muestra in agrupar_por_muestra(filas).items()
    }

    ordenados = sorted(
        (
            (muestra, resultado["p_value"])
            for muestra, resultado in resultados.items()
            if resultado is not None
        ),
        key=lambda elemento: elemento[1],
    )
    cantidad_pruebas = len(ordenados)
    p_ajustado_previo = 0.0
    for posicion, (muestra, p_value) in enumerate(ordenados):
        p_ajustado = min(1.0, (cantidad_pruebas - posicion) * p_value)
        p_ajustado_previo = max(p_ajustado_previo, p_ajustado)
        resultado = resultados[muestra]
        if resultado is not None:
            resultado["p_value_holm"] = p_ajustado_previo

    return resultados



def guardar_resultados(filas: list, path=PATH_RESULTADOS) -> str:
    """Guarda en un TXT los cálculos estadísticos separados por muestra."""
    grupos = agrupar_por_muestra(filas)
    spearman_asimetria = spearman_asimetria_isf(filas)
    spearman_asimetria_impulso_resultados = spearman_asimetria_impulso(filas)
    spearman_isf_impulso_resultados = spearman_isf_impulso(filas)

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
        spearman_relacion = spearman_isf_impulso_resultados[muestra]
        if spearman_relacion is None:
            lineas.append("Spearman ISF y diferencia absoluta de impulsos: No calculable")
        else:
            lineas.append(
                "Spearman ISF y diferencia absoluta de impulsos: "
                f"n={spearman_relacion['n']}, rho={formato_numero(spearman_relacion['rho'])}, "
                f"p-permutación={formato_numero(spearman_relacion['p_value'])}, "
                f"p-Holm={formato_numero(spearman_relacion['p_value_holm'])}"
            )
        lineas.append("")

    directorio = os.path.dirname(path)
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    with open(path, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas))
    return path


if __name__ == "__main__":
    estadisticas()
    datos = leer_estadisticas()
    ruta_resultados = guardar_resultados(datos)
    print(f"Resultados guardados en: {ruta_resultados}")
    ruta_grafica = graficar_isf_id(datos)
    print(f"Mapa de dispersión ISF-ID guardado en: {ruta_grafica}")

