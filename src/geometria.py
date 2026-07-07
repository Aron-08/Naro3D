"""
geometria.py — Auditoría y normalización topológica de figuras ya generadas
por el pipeline de ia_interprete.py (paso 1/2/2b), antes de exportarlas a
cad2d.py (BlockEntity) o a simulador_aerodinamico.py (contorno CFD).

Esta skill NO genera geometría nueva: audita la que ya salió validada de
paso 2b y, cuando puede, la corrige de forma determinística (cerrar un
contorno abierto, normalizar orientación para exportar). Si el problema es
grave (auto-intersección, ramificación), no la corrige — devuelve
`apto_para_destino=False` para que quien la llame decida si reintenta el
paso 1 con un prompt más estricto.

Filosofía (ver skill 02_skill_geometria.md, y skill 00 — filtro de ruido):
    - Todo lo topológico (contorno cerrado, orientación, auto-intersección,
      área, perímetro, centroide) es matemática exacta: se resuelve 100% en
      Python, nunca se le pregunta al modelo.
    - El LLM se usa SOLO para clasificar semánticamente un mecanismo
      (engranaje/polea/tornillo/genérico) cuando uso_destino="mecanismo",
      porque eso no tiene una fórmula cerrada — es la única parte de esta
      skill que no es puramente determinística.

Coordenadas: se trabaja en el plano (x,y) relativo [0,1] que ya usa el
pipeline (z se ignora para topología — las figuras de este proyecto son
wireframes casi planos; ver nota en ubicacion.py sobre el mismo tema).
"""

import math

import modelos   # modelo/temperatura de esta skill vienen de modelos_config.json ("geometria")
from geo_utils import segmentos_cruzan as _segmentos_cruzan  # test de cruce: única
                                                               # implementación, ver geo_utils.py


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

UMBRAL_CIERRE       = 0.05     # distancia máxima para cerrar automáticamente un contorno abierto
AREA_MINIMA         = 0.0005   # por debajo de esto, la geometría se considera degenerada (colapsada)
# La temperatura y el modelo de esta skill ahora se controlan desde
# modelos_config.json (bloque "geometria"), no aca.


# ---------------------------------------------------------------------------
# Grafo de conexiones — grados, componentes
# ---------------------------------------------------------------------------

def _construir_grafo(conexiones: list, n_puntos: int) -> dict:
    adyacencia = {i: [] for i in range(n_puntos)}
    for i, j in conexiones:
        if 0 <= i < n_puntos and 0 <= j < n_puntos and i != j:
            adyacencia[i].append(j)
            adyacencia[j].append(i)
    return adyacencia


def _grados(conexiones: list, n_puntos: int) -> list:
    grados = [0] * n_puntos
    for i, j in conexiones:
        if 0 <= i < n_puntos and 0 <= j < n_puntos and i != j:
            grados[i] += 1
            grados[j] += 1
    return grados


def _detectar_componentes(adyacencia: dict, n_puntos: int) -> list:
    """Componentes conexas del grafo (listas de índices de punto)."""
    visitados = set()
    componentes = []
    for inicio in range(n_puntos):
        if inicio in visitados:
            continue
        pila = [inicio]
        comp = []
        while pila:
            u = pila.pop()
            if u in visitados:
                continue
            visitados.add(u)
            comp.append(u)
            for v in adyacencia[u]:
                if v not in visitados:
                    pila.append(v)
        componentes.append(comp)
    return componentes


def _ordenar_ciclo(adyacencia: dict, componente: list) -> list | None:
    """Recorre un ciclo simple (todos los nodos de `componente` con grado 2
    dentro del ciclo) y devuelve el orden de índices de punto al caminarlo.
    Devuelve None si no es un ciclo simple recorrible (no debería pasar si
    ya se filtró por grado==2, pero se protege igual)."""
    if len(componente) < 3:
        return None
    inicio = componente[0]
    orden = [inicio]
    anterior = None
    actual = inicio
    for _ in range(len(componente) + 1):
        vecinos = adyacencia[actual]
        if len(vecinos) != 2:
            return None
        siguiente = vecinos[0] if vecinos[0] != anterior else vecinos[1]
        if siguiente == inicio:
            return orden if len(orden) == len(componente) else None
        orden.append(siguiente)
        anterior, actual = actual, siguiente
    return None   # no cerró en la cantidad esperada de pasos: no es un ciclo simple


# ---------------------------------------------------------------------------
# Área (shoelace), perímetro, centroide de un contorno ya ordenado
# ---------------------------------------------------------------------------

def _area_shoelace(pts_orden: list) -> float:
    """Área con signo (fórmula del zapatero). El signo depende del sentido
    de recorrido — no representa una "orientación correcta/incorrecta" en sí
    misma, solo indica si el recorrido usado va en un sentido u otro. Lo
    importante para exportar es la CONSISTENCIA entre contornos, no el signo
    en abstracto (ver contornos_para_exportar)."""
    n = len(pts_orden)
    s = 0.0
    for i in range(n):
        x1, y1 = pts_orden[i][0], pts_orden[i][1]
        x2, y2 = pts_orden[(i + 1) % n][0], pts_orden[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _perimetro(pts_orden: list) -> float:
    n = len(pts_orden)
    total = 0.0
    for i in range(n):
        x1, y1 = pts_orden[i][0], pts_orden[i][1]
        x2, y2 = pts_orden[(i + 1) % n][0], pts_orden[(i + 1) % n][1]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _centroide_poligono(pts_orden: list, area_con_signo: float) -> tuple:
    if abs(area_con_signo) < 1e-9:
        xs = [p[0] for p in pts_orden]
        ys = [p[1] for p in pts_orden]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    n = len(pts_orden)
    cx = cy = 0.0
    for i in range(n):
        x1, y1 = pts_orden[i][0], pts_orden[i][1]
        x2, y2 = pts_orden[(i + 1) % n][0], pts_orden[(i + 1) % n][1]
        cruz = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cruz
        cy += (y1 + y2) * cruz
    factor = 1.0 / (6.0 * area_con_signo)
    return (cx * factor, cy * factor)


# ---------------------------------------------------------------------------
# Auto-intersección (segmentos que se cruzan sin ser adyacentes)
# ---------------------------------------------------------------------------

# (test de orientación / cruce de segmentos: ver geo_utils.py, importado arriba)


def detectar_autointerseccion(puntos: list, conexiones: list) -> list:
    """Devuelve la lista de pares de índices de arista (en `conexiones`) que
    se cruzan entre sí, ignorando pares que comparten un vértice (eso es
    normal en un polígono, no es auto-intersección)."""
    cruces = []
    n = len(conexiones)
    for a in range(n):
        i1, j1 = conexiones[a]
        for b in range(a + 1, n):
            i2, j2 = conexiones[b]
            if len({i1, j1, i2, j2}) < 4:
                continue   # comparten un vértice: adyacentes, no cuenta
            if _segmentos_cruzan(puntos[i1], puntos[j1], puntos[i2], puntos[j2]):
                cruces.append((a, b))
    return cruces


# ---------------------------------------------------------------------------
# Área / centroide aproximados cuando la figura es solo primitivas (sin puntos)
# ---------------------------------------------------------------------------

def _area_primitivas(primitivas: list) -> float:
    total = 0.0
    for p in primitivas:
        tipo = p.get("tipo")
        if tipo == "circulo":
            total += math.pi * p["r"] ** 2
        elif tipo == "rectangulo":
            total += p["ancho"] * p["alto"]
        elif tipo == "elipse":
            total += math.pi * p["rx"] * p["ry"]
        # esfera/cubo/cilindro son volúmenes 3D, no aportan "área de planta"
        # directa comparable — se ignoran para este total 2D.
    return total


def _centroide_primitivas(primitivas: list) -> tuple:
    if not primitivas:
        return (0.5, 0.5)
    xs, ys = [], []
    for p in primitivas:
        tipo = p.get("tipo")
        if tipo == "rectangulo":
            xs.append(p["x"] + p["ancho"] / 2)
            ys.append(p["y"] + p["alto"] / 2)
        else:
            xs.append(p.get("cx", 0.5))
            ys.append(p.get("cy", 0.5))
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ---------------------------------------------------------------------------
# Auditoría principal
# ---------------------------------------------------------------------------

def auditar_geometria(figura: dict, uso_destino: str = "render_only") -> dict:
    """Audita `figura` ({puntos, conexiones, primitivas}) y devuelve un
    reporte completo. No modifica `figura`. `uso_destino` en
    {"cfd", "mecanismo", "render_only"} — cfd/mecanismo exigen contorno
    cerrado, sin ramificaciones y sin auto-intersección; render_only es
    tolerante (una figura decorativa con un punto colgante igual se dibuja).
    """
    puntos = figura.get("puntos", [])
    conexiones = figura.get("conexiones", [])
    primitivas = figura.get("primitivas", [])
    n = len(puntos)
    correcciones: list = []

    # --- Caso: solo primitivas, sin puntos/líneas ------------------------
    # Círculos, rectángulos, etc. son inherentemente cerrados: no necesitan
    # auditoría de contorno.
    if n == 0:
        return {
            "cerrado": True,
            "orientacion": None,
            "auto_interseccion": False,
            "cruces": [],
            "area": _area_primitivas(primitivas),
            "perimetro": None,
            "centroide": _centroide_primitivas(primitivas),
            "apto_para_destino": True,
            "correcciones_aplicadas": [],
            "conexiones_corregidas": conexiones,
            "colgantes": [],
            "ramificados": [],
            "aislados": [],
            "contornos": [],
        }

    # --- Grados de cada punto ---------------------------------------------
    grados = _grados(conexiones, n)
    colgantes = [i for i, g in enumerate(grados) if g == 1]
    ramificados = [i for i, g in enumerate(grados) if g > 2]
    aislados = [i for i, g in enumerate(grados) if g == 0]

    conexiones_corregidas = list(conexiones)

    # --- Intento de auto-cierre: solo el caso simple (exactamente 2 colgantes,
    # sin ramificaciones) — si hay más de 2 o hay ramificación, no se adivina
    # cuál cerrar, se deja para que decida el paso 2b/reintento de paso 1.
    if len(colgantes) == 2 and not ramificados:
        p1, p2 = colgantes
        d = math.hypot(puntos[p1][0] - puntos[p2][0], puntos[p1][1] - puntos[p2][1])
        if d < UMBRAL_CIERRE:
            conexiones_corregidas.append([p1, p2])
            correcciones.append(
                f"  Contorno abierto: se agregó conexión P{p1}-P{p2} "
                f"(distancia {d:.3f}) para cerrarlo."
            )
            grados = _grados(conexiones_corregidas, n)
            colgantes = [i for i, g in enumerate(grados) if g == 1]

    cerrado = (not colgantes) and (not ramificados) and (not aislados) and n >= 3

    # --- Componentes y contornos ordenados ---------------------------------
    adyacencia = _construir_grafo(conexiones_corregidas, n)
    componentes = [c for c in _detectar_componentes(adyacencia, n) if c and set(c) - set(aislados)]

    contornos = []   # cada uno: {"orden": [...], "puntos": [...], "area": float, "perimetro": float}
    area_total = 0.0
    perimetro_total = 0.0
    cx_acum = cy_acum = peso_acum = 0.0

    if cerrado:
        for comp in componentes:
            orden = _ordenar_ciclo(adyacencia, comp)
            if orden is None:
                cerrado = False   # componente no es un ciclo simple recorrible: audit falla
                continue
            pts_orden = [puntos[i] for i in orden]
            area_c = _area_shoelace(pts_orden)
            perim_c = _perimetro(pts_orden)
            cxc, cyc = _centroide_poligono(pts_orden, area_c)
            contornos.append({
                "orden": orden, "puntos": pts_orden,
                "area": area_c, "perimetro": perim_c, "centroide": (cxc, cyc),
            })
            area_total += abs(area_c)
            perimetro_total += perim_c
            cx_acum += cxc * abs(area_c)
            cy_acum += cyc * abs(area_c)
            peso_acum += abs(area_c)

    if contornos and peso_acum > 0:
        centroide = (cx_acum / peso_acum, cy_acum / peso_acum)
        area = area_total
        perimetro = perimetro_total
        orientacion = "positiva" if contornos[0]["area"] > 0 else "negativa"
    else:
        xs = [p[0] for p in puntos]
        ys = [p[1] for p in puntos]
        centroide = (sum(xs) / len(xs), sum(ys) / len(ys))
        area = None
        perimetro = None
        orientacion = None

    # --- Auto-intersección (se chequea siempre, cerrado o no) --------------
    cruces = detectar_autointerseccion(puntos, conexiones_corregidas)
    auto_interseccion = len(cruces) > 0

    area_degenerada = area is not None and abs(area) <= AREA_MINIMA

    # --- Aptitud según destino ----------------------------------------------
    if uso_destino in ("cfd", "mecanismo"):
        apto = cerrado and not auto_interseccion and not area_degenerada
    else:
        apto = True   # render_only: se dibuja igual aunque no sea un contorno perfecto

    return {
        "cerrado": cerrado,
        "orientacion": orientacion,
        "auto_interseccion": auto_interseccion,
        "cruces": cruces,
        "area": area,
        "perimetro": perimetro,
        "centroide": centroide,
        "apto_para_destino": apto,
        "correcciones_aplicadas": correcciones,
        "conexiones_corregidas": conexiones_corregidas,
        "colgantes": colgantes,
        "ramificados": ramificados,
        "aislados": aislados,
        "contornos": contornos,
    }


# ---------------------------------------------------------------------------
# Filtro de ruido / validación final (capa 3, ver skill 00)
# ---------------------------------------------------------------------------

def validar_y_corregir_geometria(figura: dict, uso_destino: str = "render_only",
                                  solo_diagnostico: bool = False) -> tuple[dict, bool, list[str]]:
    """Devuelve (figura_corregida, es_valida, advertencias).
    `figura_corregida` solo difiere de `figura` en `conexiones` (por el
    auto-cierre); nunca se tocan `puntos` ni `primitivas` — esta skill no
    inventa coordenadas nuevas, solo cierra huecos triviales.

    `solo_diagnostico=True` (ver plan_kernel_parametrico.md, sección 8.1):
    para geometría que ya nació correcta por construcción (kernel
    paramétrico, `ensamblador.py`), esta función deja de actuar como
    corrector activo y pasa a ser una red de seguridad de solo lectura —
    loguea cualquier violación encontrada (señal de un bug real en
    `_anclar_contacto` o similar, no de un modelo chico alucinando
    coordenadas) pero NUNCA modifica `conexiones` ni cierra nada
    automáticamente. `figura_corregida` es entonces una copia idéntica de
    `figura`, y `es_valida` refleja el resultado crudo de la auditoría sin
    intento de reparación previo."""
    if solo_diagnostico:
        reporte = auditar_geometria(figura, uso_destino)
        advertencias = []
        if reporte["colgantes"]:
            advertencias.append(
                f"  [diagnóstico] Puntos colgantes: {reporte['colgantes']} — inesperado en "
                f"geometría paramétrica, revisar ensamblador.py"
            )
        if reporte["ramificados"]:
            advertencias.append(
                f"  [diagnóstico] Puntos ramificados: {reporte['ramificados']} — inesperado en "
                f"geometría paramétrica, revisar ensamblador.py"
            )
        if reporte["aislados"]:
            advertencias.append(
                f"  [diagnóstico] Puntos aislados: {reporte['aislados']} — inesperado en "
                f"geometría paramétrica, revisar ensamblador.py"
            )
        if reporte["auto_interseccion"]:
            advertencias.append(
                f"  [diagnóstico] Auto-intersección en aristas {reporte['cruces']} — inesperado "
                f"en geometría paramétrica, revisar ensamblador.py"
            )
        if reporte["area"] is not None and abs(reporte["area"]) <= AREA_MINIMA:
            advertencias.append(
                f"  [diagnóstico] Área degenerada ({reporte['area']:.5f}) — inesperado en "
                f"geometría paramétrica, revisar ensamblador.py"
            )
        for aviso in advertencias:
            print(f"[geometria][solo_diagnostico]{aviso}")
        figura_sin_tocar = dict(figura)
        return figura_sin_tocar, reporte["apto_para_destino"], advertencias

    reporte = auditar_geometria(figura, uso_destino)
    advertencias = list(reporte["correcciones_aplicadas"])

    if reporte["colgantes"]:
        advertencias.append(f"  Puntos colgantes sin cerrar: {reporte['colgantes']} — violación grave")
    if reporte["ramificados"]:
        advertencias.append(f"  Puntos con más de 2 conexiones (ramificación): {reporte['ramificados']} — violación grave")
    if reporte["aislados"]:
        advertencias.append(f"  Puntos sin ninguna conexión: {reporte['aislados']} — violación grave")
    if reporte["auto_interseccion"]:
        advertencias.append(f"  Auto-intersección en aristas {reporte['cruces']} — violación grave")
    if reporte["area"] is not None and abs(reporte["area"]) <= AREA_MINIMA:
        advertencias.append(f"  Área degenerada ({reporte['area']:.5f}) — geometría casi colapsada")

    figura_corregida = {
        **figura,
        "conexiones": reporte["conexiones_corregidas"],
    }
    return figura_corregida, reporte["apto_para_destino"], advertencias


# ---------------------------------------------------------------------------
# Exportación: contornos normalizados, listos para CAD/CFD
# ---------------------------------------------------------------------------

def contornos_para_exportar(figura: dict, uso_destino: str = "cfd",
                             orientacion_deseada: str = "positiva") -> list:
    """Devuelve una lista de contornos cerrados, cada uno como lista de
    (x,y) YA en el orden de recorrido correcto para exportar — normalizando
    todos al mismo sentido (`orientacion_deseada`), para no repetir a mano
    el fix de "eje Y invertido" cada vez que cambia el origen de la figura.

    No modifica la figura original. Si la geometría no es apta (auto-
    intersección, contorno abierto), devuelve lista vacía — quien llama debe
    chequear `validar_y_corregir_geometria` antes si quiere saber por qué.
    """
    reporte = auditar_geometria(figura, uso_destino)
    if not reporte["apto_para_destino"]:
        return []

    resultado = []
    for c in reporte["contornos"]:
        pts = c["puntos"]
        area_signo = c["area"]
        va_al_reves = (
            (orientacion_deseada == "positiva" and area_signo < 0) or
            (orientacion_deseada == "negativa" and area_signo > 0)
        )
        resultado.append(list(reversed(pts)) if va_al_reves else list(pts))
    return resultado


# ---------------------------------------------------------------------------
# Clasificación semántica de mecanismos (única parte con LLM)
# ---------------------------------------------------------------------------

SYSTEM_CLASIFICAR_MECANISMO = """Clasificás la silueta de una pieza mecánica 2D ya generada, para saber
qué tipo de mecanismo es. Recibís el contorno como lista de puntos (x,y) y el área/perímetro
ya calculados. NO inventés coordenadas, solo clasificá.

Respondé ÚNICAMENTE con este formato, sin texto adicional:

TIPO: <engranaje|polea|tornillo|husillo|generico>
EJE: cx,cy

Reglas:
  - "engranaje": contorno con muchos vértices regulares alrededor de un centro (dientes).
  - "polea": contorno redondeado (muchos puntos formando un círculo/anillo), sin dientes.
  - "tornillo"/"husillo": contorno alargado con un patrón repetitivo a lo largo de un eje.
  - "generico": cualquier otra silueta mecánica que no encaje claramente en las anteriores.
  - EJE es el centro de rotación/simetría de la pieza — normalmente coincide con el centroide
    que ya te paso, salvo que la silueta sea claramente asimétrica respecto de él.

=== EJEMPLO ===
entrada: contorno con 24 vértices distribuidos en anillo irregular alrededor de (0.50,0.50),
área 0.045, perímetro 0.98
TIPO: engranaje
EJE: 0.50,0.50
"""


def _parsear_clasificacion(texto: str, centroide_defecto: tuple) -> dict:
    resultado = {"tipo": "generico", "eje": centroide_defecto}
    if not texto:
        return resultado
    import re
    m_tipo = re.search(r"^TIPO\s*:\s*(\w+)", texto, re.MULTILINE)
    if m_tipo and m_tipo.group(1).lower() in ("engranaje", "polea", "tornillo", "husillo", "generico"):
        resultado["tipo"] = m_tipo.group(1).lower()
    m_eje = re.search(r"^EJE\s*:\s*([\d.]+)\s*,\s*([\d.]+)", texto, re.MULTILINE)
    if m_eje:
        try:
            resultado["eje"] = (float(m_eje.group(1)), float(m_eje.group(2)))
        except ValueError:
            pass
    return resultado


def clasificar_mecanismo(figura: dict, reporte: dict | None = None) -> dict:
    """Solo tiene sentido llamarla cuando uso_destino == 'mecanismo'. Usa el
    LLM únicamente para la etiqueta semántica; el centroide de respaldo
    siempre sale del cálculo geométrico exacto (nunca del modelo)."""
    reporte = reporte or auditar_geometria(figura, "mecanismo")
    centroide_defecto = reporte["centroide"] or (0.5, 0.5)

    if not reporte["contornos"]:
        return {"tipo": "generico", "eje": centroide_defecto}

    contorno = reporte["contornos"][0]
    n_vertices = len(contorno["puntos"])
    resumen = (
        f"contorno con {n_vertices} vértices, "
        f"área {abs(contorno['area']):.4f}, perímetro {contorno['perimetro']:.4f}, "
        f"centroide ({centroide_defecto[0]:.2f},{centroide_defecto[1]:.2f})"
    )

    texto = modelos.llamar(
        "geometria",
        messages=[
            {"role": "system", "content": SYSTEM_CLASIFICAR_MECANISMO},
            {"role": "user", "content": f"entrada: {resumen}"},
        ],
    )
    return _parsear_clasificacion(texto, centroide_defecto)


# ---------------------------------------------------------------------------
# API de conveniencia — pensada para llamar justo antes de exportar
# ---------------------------------------------------------------------------

def preparar_para_cad(figura: dict) -> dict:
    """Todo lo que necesita cad2d.py para crear un BlockEntity a partir de
    una figura ya generada: contornos normalizados + área/perímetro/centroide.
    Devuelve None si la geometría no es apta (avisos ya impresos)."""
    figura_ok, es_valida, advertencias = validar_y_corregir_geometria(figura, "mecanismo")
    for a in advertencias:
        print(f"[geometria]{a}")
    if not es_valida:
        return None
    reporte = auditar_geometria(figura_ok, "mecanismo")
    return {
        "contornos": contornos_para_exportar(figura_ok, "mecanismo"),
        "area": reporte["area"],
        "perimetro": reporte["perimetro"],
        "centroide": reporte["centroide"],
    }


def preparar_para_cfd(figura: dict) -> dict | None:
    """Contorno(s) listos para exportar_a_cfd() en simulador_aerodinamico.py.
    Devuelve None si la geometría no es apta (avisos ya impresos)."""
    figura_ok, es_valida, advertencias = validar_y_corregir_geometria(figura, "cfd")
    for a in advertencias:
        print(f"[geometria]{a}")
    if not es_valida:
        return None
    return {"contornos": contornos_para_exportar(figura_ok, "cfd")}


if __name__ == "__main__":
    print("=== Cuadrado simple, cerrado y bien formado ===")
    cuadrado = {
        "puntos": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
        "conexiones": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "primitivas": [],
    }
    r = auditar_geometria(cuadrado, "cfd")
    print("cerrado:", r["cerrado"], "| área:", round(r["area"], 5),
          "| perímetro:", round(r["perimetro"], 4), "| centroide:", r["centroide"],
          "| apto:", r["apto_para_destino"])

    print("\n=== Contorno abierto (falta la última conexión) ===")
    abierto = {
        "puntos": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.601]],
        "conexiones": [[0, 1], [1, 2], [2, 3]],
        "primitivas": [],
    }
    fig_corr, valido, avisos = validar_y_corregir_geometria(abierto, "cfd")
    print("válido:", valido, "| conexiones corregidas:", fig_corr["conexiones"])
    for a in avisos:
        print(a)

    print("\n=== Figura en forma de 'ocho' (auto-intersección real) ===")
    ocho = {
        "puntos": [[0.3, 0.3], [0.7, 0.7], [0.7, 0.3], [0.3, 0.7]],
        "conexiones": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "primitivas": [],
    }
    r2 = auditar_geometria(ocho, "cfd")
    print("auto_interseccion:", r2["auto_interseccion"], "| cruces:", r2["cruces"],
          "| apto:", r2["apto_para_destino"])

    print("\n=== Exportar contorno normalizado ===")
    contornos = contornos_para_exportar(cuadrado, "cfd", orientacion_deseada="positiva")
    print("contornos listos para exportar:", contornos)