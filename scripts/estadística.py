import csv
import math
import os
import random
import re
import statistics
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_baro_cache"))

import matplotlib.pyplot as plt
import statsmodels.api as sm


PATH_CSV = os.path.join("Estadisticas", "estadisticas.csv")
PATH_CSV_RMSE = os.path.join(
    "Baropodometrias procesadas", "Resultados_Estadisticos_RMSE.csv"
)
PATH_RESULTADOS = os.path.join("Estadisticas", "resultados_estadisticos.txt")
PATH_RESUMEN_DESCRIPTIVO = os.path.join(
    "Estadisticas", "mediana_varianza_por_muestra.csv"
)
PATH_GRAFICA_ISF_ID = os.path.join("Estadisticas", "mapa_dispersion_isf_id.png")
LONGITUD_CORTA = "longitud estimada del miembro corto"
LONGITUD_MAYOR = "longitud del miembro mayor"
ISF = "ISF"
IMPULSO_DERECHO = "promedio impulso derecho"
IMPULSO_IZQUIERDO = "promedio impulso izquierdo"
MUESTRA = "muestra"
MUESTRA_A_VELOCIDAD = {1: 1.1, 3: 1.3, 5: 1.5, 7: 1.7, 9: 1.9}


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


def leer_rmse(path=PATH_CSV_RMSE) -> list:
    """Lee el RMSE y asigna cada registro a su sujeto y número de muestra."""
    with open(path, "r", newline="", encoding="utf-8") as archivo_csv:
        filas = list(csv.DictReader(archivo_csv))

    for fila in filas:
        coincidencia = re.fullmatch(r"Subject(\d+)_\d+p(\d+)", fila["Archivo"])
        if coincidencia is None:
            raise ValueError(f"Nombre de archivo RMSE no reconocido: {fila['Archivo']}")
        if int(fila["Sujeto"].removeprefix("Subject")) != int(
            coincidencia.group(1)
        ):
            raise ValueError(
                f"El sujeto no coincide con el archivo: {fila['Archivo']}"
            )
        fila["num paciente"] = coincidencia.group(1)
        fila[MUESTRA] = coincidencia.group(2)

    return filas


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


def _prueba_spearman_permutacion(x: list, y: list, semilla: int) -> dict | None:
    if len(x) != len(y) or len(x) < 2:
        return None

    rho = _correlacion_spearman(x, y)
    if rho is None:
        return None

    permutaciones = 9999
    generador = random.Random(semilla)
    extremos = 0
    for _ in range(permutaciones):
        rho_permutado = _correlacion_spearman(x, generador.sample(y, len(y)))
        if rho_permutado is not None and abs(rho_permutado) >= abs(rho) - 1e-12:
            extremos += 1

    return {
        "n": len(x),
        "rho": rho,
        "p_value": (extremos + 1) / (permutaciones + 1),
    }


def _ajustar_p_holm(resultados: dict) -> dict:
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


def _indexar_rmse_por_sujeto_muestra(filas_rmse: list) -> dict:
    """Indexa frecuencia de zancada y RMSE por (sujeto, muestra) validando unicidad."""
    indice = {}
    for fila in filas_rmse:
        clave = (int(fila["num paciente"]), int(fila[MUESTRA]))
        if clave in indice:
            raise ValueError(
                f"Hay más de un RMSE para sujeto y muestra {clave}."
            )
        registro = {}
        for columna in ("Frec_Zancada_Hz", "RMSE_mm"):
            if _es_numero_valido(fila.get(columna)):
                registro[columna] = float(fila[columna])
        indice[clave] = registro
    return indice


def _spearman_rmse_asimetria_muestra(
    filas: list, rmse_por_sujeto_muestra: dict, muestra: int, semilla: int
) -> dict | None:
    pares = []
    for fila in filas:
        columnas_longitud = (LONGITUD_MAYOR, LONGITUD_CORTA)
        if not all(
            _es_numero_valido(fila.get(columna)) for columna in columnas_longitud
        ):
            continue

        clave = (int(fila["num paciente"]), muestra)
        rmse = rmse_por_sujeto_muestra.get(clave)
        if rmse is not None:
            asimetria = float(fila[LONGITUD_MAYOR]) - float(fila[LONGITUD_CORTA])
            pares.append((asimetria, rmse))

    asimetria = [par[0] for par in pares]
    rmse = [par[1] for par in pares]
    return _prueba_spearman_permutacion(asimetria, rmse, semilla)


def spearman_rmse_asimetria(filas: list, filas_rmse: list) -> dict:
    """Calcula Spearman RMSE-asimetría por muestra y ajusta p con Holm."""
    indice_rmse = _indexar_rmse_por_sujeto_muestra(filas_rmse)
    rmse_por_sujeto_muestra = {
        clave: reg["RMSE_mm"]
        for clave, reg in indice_rmse.items()
        if "RMSE_mm" in reg
    }

    resultados = {
        muestra: _spearman_rmse_asimetria_muestra(
            filas_muestra, rmse_por_sujeto_muestra, muestra, 20261021 + muestra
        )
        for muestra, filas_muestra in agrupar_por_muestra(filas).items()
    }

    return _ajustar_p_holm(resultados)


def _interpretar_correlacion(rho: float | None) -> str:
    if rho is None:
        return "No calculable."

    magnitud = abs(rho)
    if magnitud < 0.2:
        intensidad = "muy débil"
    elif magnitud < 0.4:
        intensidad = "débil"
    elif magnitud < 0.6:
        intensidad = "moderada"
    elif magnitud < 0.8:
        intensidad = "fuerte"
    else:
        intensidad = "muy fuerte"

    if rho == 0:
        return "Sin relación monotónica entre RMSE y diferencia de longitud."

    direccion = "positiva" if rho > 0 else "negativa"
    return f"Relación {intensidad} {direccion} entre RMSE y diferencia de longitud."


def _spearman_isf_impulso_muestra(filas: list, semilla: int) -> dict | None:
    pares = [
        (float(fila[ISF]), diferencia_impulso(fila))
        for fila in filas
        if all(
            _es_numero_valido(fila.get(columna))
            for columna in (ISF, IMPULSO_DERECHO, IMPULSO_IZQUIERDO)
        )
    ]
    isf = [par[0] for par in pares]
    impulso = [par[1] for par in pares]
    return _prueba_spearman_permutacion(isf, impulso, semilla)


def spearman_isf_impulso(filas: list) -> dict:
    """Calcula Spearman ISF-impulso y p por permutación dentro de cada muestra."""
    resultados = {
        muestra: _spearman_isf_impulso_muestra(filas_muestra, 20261007 + muestra)
        for muestra, filas_muestra in agrupar_por_muestra(filas).items()
    }

    return _ajustar_p_holm(resultados)


def _valor_columna(fila: dict, columna: str) -> float | None:
    return float(fila[columna]) if _es_numero_valido(fila.get(columna)) else None


def _valor_diferencia_impulso(fila: dict) -> float | None:
    if _es_numero_valido(fila.get(IMPULSO_DERECHO)) and _es_numero_valido(
        fila.get(IMPULSO_IZQUIERDO)
    ):
        return diferencia_impulso(fila)
    return None


def _valor_asimetria_longitud(fila: dict, factor: float = 1.0) -> float | None:
    if _es_numero_valido(fila.get(LONGITUD_MAYOR)) and _es_numero_valido(
        fila.get(LONGITUD_CORTA)
    ):
        return (float(fila[LONGITUD_MAYOR]) - float(fila[LONGITUD_CORTA])) * factor
    return None


_ESPECIFICACIONES_ESTADISTICAS = [
    ("tiempo de apoyo menor", "ms", "ms²", lambda f: _valor_columna(f, "tiempo de apoyo menor")),
    ("tiempo de apoyo mayor", "ms", "ms²", lambda f: _valor_columna(f, "tiempo de apoyo mayor")),
    (
        "fuerza puntual promedio izquierda",
        "N",
        "N²",
        lambda f: _valor_columna(f, "fuerza puntual promedio izquierda"),
    ),
    (
        "fuerza puntual promedio derecha",
        "N",
        "N²",
        lambda f: _valor_columna(f, "fuerza puntual promedio derecha"),
    ),
    ("ISF", "%", "%²", lambda f: _valor_columna(f, ISF)),
    ("promedio impulso derecho", "N·s", "(N·s)²", lambda f: _valor_columna(f, IMPULSO_DERECHO)),
    ("promedio impulso izquierdo", "N·s", "(N·s)²", lambda f: _valor_columna(f, IMPULSO_IZQUIERDO)),
    (
        "longitud estimada del miembro corto",
        "m",
        "m²",
        lambda f: _valor_columna(f, LONGITUD_CORTA),
    ),
    ("longitud del miembro mayor", "m", "m²", lambda f: _valor_columna(f, LONGITUD_MAYOR)),
    ("diferencia absoluta de impulsos", "N·s", "(N·s)²", _valor_diferencia_impulso),
    ("asimetría de longitud (m)", "m", "m²", lambda f: _valor_asimetria_longitud(f, 1.0)),
    ("asimetría de longitud (cm)", "cm", "cm²", lambda f: _valor_asimetria_longitud(f, 100.0)),
]

_ESPECIFICACIONES_RMSE = [
    ("frecuencia de zancada", "Hz", "Hz²", "Frec_Zancada_Hz"),
    ("RMSE CoG", "mm", "mm²", "RMSE_mm"),
]


def _resumir_valores(
    nombre: str, unidad: str, unidad_varianza: str, valores: list[float]
) -> dict:
    """Calcula n, mediana y varianza muestral (n-1) para una serie de valores."""
    n = len(valores)
    return {
        "variable": nombre,
        "unidad": unidad,
        "unidad_varianza": unidad_varianza,
        "n": n,
        "mediana": statistics.median(valores) if n >= 1 else None,
        "varianza_muestral": statistics.variance(valores) if n >= 2 else None,
    }


def estadisticas_descriptivas_por_muestra(
    filas: list, filas_rmse: list | None = None
) -> dict:
    """Calcula mediana y varianza muestral (n-1) por muestra para todas las variables."""
    indice_rmse = (
        _indexar_rmse_por_sujeto_muestra(filas_rmse)
        if filas_rmse is not None
        else None
    )
    resumen = {}

    for muestra, filas_muestra in agrupar_por_muestra(filas).items():
        items = []
        for nombre, unidad, unidad_var, extractor in _ESPECIFICACIONES_ESTADISTICAS:
            valores = [
                val for val in (extractor(fila) for fila in filas_muestra)
                if val is not None
            ]
            items.append(_resumir_valores(nombre, unidad, unidad_var, valores))

        if indice_rmse is not None:
            for nombre, unidad, unidad_var, col_rmse in _ESPECIFICACIONES_RMSE:
                valores = []
                for fila in filas_muestra:
                    if not _es_numero_valido(fila.get("num paciente")):
                        continue
                    clave = (int(fila["num paciente"]), muestra)
                    val = indice_rmse.get(clave, {}).get(col_rmse)
                    if val is not None:
                        valores.append(val)
                items.append(_resumir_valores(nombre, unidad, unidad_var, valores))

        resumen[muestra] = items

    return resumen


def guardar_mediana_varianza_csv(
    resumen_descriptivo: dict, path=PATH_RESUMEN_DESCRIPTIVO
) -> str:
    """Exporta un CSV con mediana y varianza muestral (n-1) por muestra y variable."""
    encabezados = [
        "muestra",
        "velocidad_m_s",
        "variable",
        "unidad",
        "unidad_varianza",
        "n",
        "mediana",
        "varianza_muestral",
    ]
    directorio = os.path.dirname(path)
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as archivo_csv:
        escritor = csv.writer(archivo_csv)
        escritor.writerow(encabezados)
        for muestra, items in sorted(resumen_descriptivo.items()):
            velocidad = MUESTRA_A_VELOCIDAD.get(muestra, "")
            for item in items:
                escritor.writerow([
                    muestra,
                    velocidad,
                    item["variable"],
                    item["unidad"],
                    item["unidad_varianza"],
                    item["n"],
                    item["mediana"],
                    item["varianza_muestral"],
                ])

    return path


def guardar_resultados(
    filas: list, path=PATH_RESULTADOS, filas_rmse: list | None = None
) -> str:
    """Guarda en un TXT los cálculos estadísticos separados por muestra."""
    grupos = agrupar_por_muestra(filas)
    spearman_asimetria = spearman_asimetria_isf(filas)
    spearman_asimetria_impulso_resultados = spearman_asimetria_impulso(filas)
    spearman_isf_impulso_resultados = spearman_isf_impulso(filas)
    spearman_rmse_resultados = (
        spearman_rmse_asimetria(filas, filas_rmse)
        if filas_rmse is not None
        else {}
    )
    resumen_descriptivo = estadisticas_descriptivas_por_muestra(
        filas, filas_rmse=filas_rmse
    )

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
        if filas_rmse is not None:
            spearman_rmse = spearman_rmse_resultados[muestra]
            if spearman_rmse is None:
                lineas.append("Spearman RMSE y asimetría de longitud: No calculable")
            else:
                lineas.append(
                    "Spearman RMSE y asimetría de longitud: "
                    f"n={spearman_rmse['n']}, rho={formato_numero(spearman_rmse['rho'])}, "
                    f"p-permutación={formato_numero(spearman_rmse['p_value'])}, "
                    f"p-Holm={formato_numero(spearman_rmse['p_value_holm'])}"
                )
                lineas.append(
                    f"Interpretación: {_interpretar_correlacion(spearman_rmse['rho'])}"
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
        lineas.append("Estadísticos descriptivos (mediana y varianza muestral, n-1):")
        for item in resumen_descriptivo[muestra]:
            lineas.append(
                f"  - {item['variable']} ({item['unidad']}): "
                f"n={item['n']}, mediana={formato_numero(item['mediana'])}, "
                f"varianza={formato_numero(item['varianza_muestral'])} {item['unidad_varianza']}"
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
    datos_rmse = leer_rmse()
    resumen_descriptivo = estadisticas_descriptivas_por_muestra(
        datos, filas_rmse=datos_rmse
    )
    ruta_descriptivo = guardar_mediana_varianza_csv(resumen_descriptivo)
    print(f"Resumen mediana y varianza guardado en: {ruta_descriptivo}")
    ruta_resultados = guardar_resultados(datos, filas_rmse=datos_rmse)
    print(f"Resultados guardados en: {ruta_resultados}")
    ruta_grafica = graficar_isf_id(datos)
    print(f"Mapa de dispersión ISF-ID guardado en: {ruta_grafica}")
