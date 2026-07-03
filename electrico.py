"""
electrico.py — Skill 06: ley de Ohm, potencia disipada, y análisis de una RED
de componentes conectados (no solo un objeto aislado) mediante análisis nodal
modificado (MNA). Pensado para conectar con objetos.py (resistividad ya
presente en la ficha desde el día 1: "resistencia_electrica_ohm_m") y con
termodinamica.py (el calentamiento I²R es una fuente de calor más, entra al
mismo modelo de cuerpo concentrado que ya existe — un cable con sobrecorriente
es el mismo problema físico que una pieza cerca de una llama, solo cambia de
dónde sale la potencia).

Filosofía (ver skill 00 — filtro de ruido, mismo criterio que termodinamica.py
y calculo_estructural.py): el LLM NUNCA resuelve el circuito. Resolver una red
de resistencias es álgebra lineal exacta (Python + numpy, análisis nodal) —
no hay "criterio" que decidir ahí, a diferencia de estructural/térmico donde
había que elegir qué fórmula de libro aplicaba. Acá el LLM se usa solo para:
  1. Clasificar la topología en palabras simples (serie/paralelo/red general/
     abierto) para el mensaje al usuario — la resolución real siempre es la
     red completa vía resolver_red(), sea cual sea la clasificación.
  2. Redactar una advertencia corta si hay sobrecorriente/sobretensión/
     cortocircuito — el número que dispara la advertencia ya lo calculó
     Python, el LLM solo la redacta en lenguaje llano.

Unidades: SI (Ω, V, A, W, m, m²). Igual que termodinamica.py/calculo_estructural.py,
la escena vive en coordenadas normalizadas [0,1]³, no en metros — la
resistencia de un objeto-conductor se deriva de resistividad + masa/densidad
(ficha de objetos.py), nunca de coordenadas de pantalla. Ver
`geometria_electrica_desde_ficha()`.

Requiere lo mismo que ia_interprete.py (Ollama corriendo) SOLO para
evaluar_circuito(); resolver_red() es Python/numpy puro y no toca el modelo —
es justamente la función pensada para llamarse en cada tick de un modo
interactivo (ver modos.py), sin costo de LLM.
"""

import re

import numpy as np

import modelos   # modelo/temperatura de esta skill vienen de modelos_config.json ("electrico")


# ---------------------------------------------------------------------------
# Constantes de referencia (fijas, nunca se le preguntan al modelo)
# ---------------------------------------------------------------------------

# La temperatura y el modelo de esta skill ahora se controlan desde
# modelos_config.json (bloque "electrico"), no acá.

# Umbral resistividad conductor/aislante para detección de topología por
# contacto (Ω·m). Por debajo: conduce. Por encima: corta el circuito ahí
# aunque los objetos se estén tocando (ej. un cable de cobre apoyado sobre
# una base de madera no cierra el circuito a través de la base).
UMBRAL_RESISTIVIDAD_CONDUCTOR_OHM_M = 1e-4

H_CONVECCION_AIRE_LIBRE_W_M2K = 10.0   # mismo valor de referencia que termodinamica.py

_VALORES_SEGURIDAD_ELECTRICO = {
    "metal_generico":    {"resistencia_electrica_ohm_m": 2.0e-8},   # ~cobre
    "plastico_generico": {"resistencia_electrica_ohm_m": 1.0e13},   # aislante
    "mineral_generico":  {"resistencia_electrica_ohm_m": 1.0e10},
    "organico_generico":  {"resistencia_electrica_ohm_m": 1.0e12},
}


# ---------------------------------------------------------------------------
# Geometría eléctrica: resistencia de un objeto-conductor, siempre en Python
# ---------------------------------------------------------------------------

def geometria_electrica_desde_ficha(peso_kg: float, densidad_kg_m3: float,
                                     resistencia_electrica_ohm_m: float,
                                     longitud_m: float | None = None,
                                     area_seccion_m2: float | None = None) -> dict:
    """Devuelve {"longitud_m", "area_seccion_m2", "resistencia_ohm"}.

    Mismo criterio que geometria_termica_desde_ficha(): el volumen sale de
    masa/densidad (ya resueltos por objetos.py), nunca de coordenadas de
    pantalla. Si no se conoce la longitud real del conductor (ej. un cable
    con forma definida), se aproxima como cuerpo compacto (cubo equivalente):
    longitud = volumen^(1/3), área = volumen^(2/3) — da una resistencia
    conservadora (ni muy cable largo ni muy pastilla corta).
    """
    densidad = densidad_kg_m3 if densidad_kg_m3 and densidad_kg_m3 > 0 else 1000.0
    masa = peso_kg if peso_kg and peso_kg > 0 else 0.001
    volumen = masa / densidad

    if longitud_m and longitud_m > 0:
        longitud = longitud_m
        area = area_seccion_m2 if (area_seccion_m2 and area_seccion_m2 > 0) else (volumen / longitud)
    else:
        longitud = volumen ** (1.0 / 3.0)
        area = volumen ** (2.0 / 3.0)

    area = max(area, 1e-12)
    resistividad = resistencia_electrica_ohm_m if resistencia_electrica_ohm_m and \
        resistencia_electrica_ohm_m > 0 else _VALORES_SEGURIDAD_ELECTRICO["metal_generico"]["resistencia_electrica_ohm_m"]
    resistencia_ohm = resistividad * longitud / area

    return {
        "longitud_m": longitud,
        "area_seccion_m2": area,
        "resistencia_ohm": resistencia_ohm,
    }


# ---------------------------------------------------------------------------
# Ley de Ohm y potencia — Python puro, sin estado, seguras de llamar cada tick
# ---------------------------------------------------------------------------

def corriente_a(tension_v: float, resistencia_ohm: float) -> float:
    return tension_v / resistencia_ohm if resistencia_ohm > 0 else float("inf")


def tension_v(corriente_a_: float, resistencia_ohm: float) -> float:
    return corriente_a_ * resistencia_ohm


def potencia_disipada_w(corriente_a_: float, resistencia_ohm: float) -> float:
    """P = I²R — es la magnitud que alimenta el modo de falla térmico de un
    conductor con sobrecorriente (ver corriente_maxima_por_calentamiento_a)."""
    return (corriente_a_ ** 2) * resistencia_ohm


def resistencia_equivalente_serie(*resistencias_ohm: float) -> float:
    return sum(r for r in resistencias_ohm if r not in (None, float("inf")))


def resistencia_equivalente_paralelo(*resistencias_ohm: float) -> float:
    inversas = [1.0 / r for r in resistencias_ohm if r and r > 0]
    return 1.0 / sum(inversas) if inversas else float("inf")


def corriente_maxima_por_calentamiento_a(resistencia_ohm: float, area_expuesta_m2: float,
                                          temp_ambiente_c: float, temp_max_servicio_c: float,
                                          h_w_m2k: float = H_CONVECCION_AIRE_LIBRE_W_M2K) -> float:
    """Corriente de régimen permanente que hace que el conductor se estabilice
    justo en su temperatura máxima de servicio (mismo modelo de convección
    simple que termodinamica.py): en régimen, potencia disipada = potencia
    convectada, P = h·A·ΔT, y P = I²R → I_max = sqrt(h·A·ΔT / R)."""
    delta_t = (temp_max_servicio_c or 0.0) - (temp_ambiente_c or 20.0)
    if resistencia_ohm <= 0 or delta_t <= 0 or area_expuesta_m2 <= 0:
        return 0.0
    return (h_w_m2k * area_expuesta_m2 * delta_t / resistencia_ohm) ** 0.5


# ---------------------------------------------------------------------------
# Topología por contacto (Python puro, determinístico) — "tocarse = conducir"
# ---------------------------------------------------------------------------
# Cada objeto-conductor es un resistor con dos terminales (sus dos extremos,
# a lo largo de su dimensión mayor). Cuando dos objetos se tocan en la escena
# (mismo test AABB con margen que ya usa ubicacion.py para colisión, acá
# reimplementado local y trivial a propósito — no vale la pena acoplar este
# módulo a los internals privados de ubicacion.py por 6 líneas de geometría,
# a diferencia del test de cruce de segmentos de geo_utils.py, que sí era
# grande y con casos borde reales que ameritaban una sola fuente de verdad),
# sus terminales más cercanos quedan eléctricamente unidos (mismo nodo) — así
# se arma la red sin pedirle nada al LLM.

def _bboxes_tocan(a: dict, b: dict, margen: float = 0.01) -> bool:
    return (a["x_min"] - margen <= b["x_max"] and b["x_min"] - margen <= a["x_max"] and
            a["y_min"] - margen <= b["y_max"] and b["y_min"] - margen <= a["y_max"] and
            a.get("z_min", 0) - margen <= b.get("z_max", 0) and
            b.get("z_min", 0) - margen <= a.get("z_max", 0))


class _UnionFind:
    """Union-Find mínimo para fusionar terminales en contacto en un mismo
    nodo eléctrico. No hace falta nada más sofisticado: la cantidad de
    terminales es 2 por objeto, siempre chica (mismo orden que las figuras
    del proyecto, <30 objetos en escena)."""

    def __init__(self):
        self._padre: dict[str, str] = {}

    def raiz(self, x: str) -> str:
        self._padre.setdefault(x, x)
        while self._padre[x] != x:
            self._padre[x] = self._padre[self._padre[x]]
            x = self._padre[x]
        return x

    def unir(self, a: str, b: str) -> None:
        ra, rb = self.raiz(a), self.raiz(b)
        if ra != rb:
            self._padre[ra] = rb


def construir_red_desde_contacto(objetos_en_escena: list[dict],
                                  fichas_por_nombre: dict[str, dict],
                                  margen: float = 0.02) -> tuple[list[dict], dict[str, str]]:
    """objetos_en_escena: lista de {"nombre", "bbox"} (mismo shape que
    ubicacion.objetos_en_escena_actual()). fichas_por_nombre: {nombre: ficha
    de propiedades ya resuelta (con resistencia_electrica_ohm_m, peso_kg,
    densidad_kg_m3)}.

    Devuelve (elementos_resistivos, mapa_terminales) donde elementos_resistivos
    ya está listo para resolver_red(), y mapa_terminales indica a qué nodo
    equivalente quedó cada terminal "<nombre>#A"/"<nombre>#B" — útil para que
    quien llame sepa dónde conectar una fuente externa (ej. "bateria#A").

    Objetos cuya ficha no tenga resistencia_electrica_ohm_m por debajo de
    UMBRAL_RESISTIVIDAD_CONDUCTOR_OHM_M actúan como aislante: se los incluye
    como resistor de valor altísimo (no cortan la topología, pero en la
    práctica no dejan pasar corriente apreciable — más robusto para el
    solver que directamente omitirlos, que puede dejar nodos flotantes).
    """
    uf = _UnionFind()
    for i, obj_a in enumerate(objetos_en_escena):
        for obj_b in objetos_en_escena[i + 1:]:
            if _bboxes_tocan(obj_a["bbox"], obj_b["bbox"], margen):
                # Se tocan: unir el terminal más cercano de cada uno. Sin
                # conocer la orientación real, aproximar con "B de A" y
                # "A de B" — es una simplificación (no siempre el terminal
                # geométricamente más cercano), documentada a propósito:
                # para un breadboard virtual manejado con la mano, la
                # topología exacta importa menos que "si se tocan, conducen".
                uf.unir(f"{obj_a['nombre']}#B", f"{obj_b['nombre']}#A")

    elementos = []
    mapa_terminales = {}
    for obj in objetos_en_escena:
        nombre = obj["nombre"]
        ficha = fichas_por_nombre.get(nombre, {})
        resistividad = ficha.get("resistencia_electrica_ohm_m")
        geo = geometria_electrica_desde_ficha(
            peso_kg=ficha.get("peso_kg", 0.0),
            densidad_kg_m3=ficha.get("densidad_kg_m3", 0.0),
            resistencia_electrica_ohm_m=resistividad or 1e13,
        )
        term_a, term_b = f"{nombre}#A", f"{nombre}#B"
        nodo_a, nodo_b = uf.raiz(term_a), uf.raiz(term_b)
        mapa_terminales[term_a] = nodo_a
        mapa_terminales[term_b] = nodo_b
        elementos.append({
            "nombre": nombre,
            "a": nodo_a,
            "b": nodo_b,
            "resistencia_ohm": geo["resistencia_ohm"],
            "conductor": bool(resistividad and resistividad < UMBRAL_RESISTIVIDAD_CONDUCTOR_OHM_M),
        })
    return elementos, mapa_terminales


# ---------------------------------------------------------------------------
# Resolución de la red — análisis nodal modificado (MNA), Python/numpy puro
# ---------------------------------------------------------------------------
# Esta es la función que responde "cómo fluye la corriente en los demás
# componentes conectados": no es serie ni paralelo a mano, es el caso general
# (cualquier grafo de resistores + fuentes de tensión). Es la parte de esta
# skill que NUNCA llama al LLM — pensada para el tick de un modo interactivo.

def resolver_red(elementos_resistivos: list[dict], fuentes: list[dict],
                  nodo_referencia: str = "GND") -> dict:
    """
    elementos_resistivos: [{"nombre", "a", "b", "resistencia_ohm"}, ...]
    fuentes:              [{"nombre", "nodo_pos", "nodo_neg", "tension_v"}, ...]

    Devuelve {"voltajes_nodo": {nodo: V}, "corrientes_elemento": {nombre: A
    (positiva de "a" hacia "b")}, "corrientes_fuente": {nombre: A},
    "potencia_disipada_w": {nombre: W}, "circuito_abierto": bool}.

    "circuito_abierto"=True cuando la matriz sale singular (nodos aislados,
    ninguna fuente cierra el lazo) — se devuelve todo en 0 en vez de romper,
    mismo criterio de "nunca None, nunca bloquear el pipeline" del resto del
    proyecto (skill 00).
    """
    nodos = set()
    for e in elementos_resistivos:
        nodos.add(e["a"]); nodos.add(e["b"])
    for f in fuentes:
        nodos.add(f["nodo_pos"]); nodos.add(f["nodo_neg"])
    nodos.discard(nodo_referencia)
    nodos = sorted(nodos)
    idx = {n: i for i, n in enumerate(nodos)}
    n = len(nodos)
    m = len(fuentes)

    if n == 0:
        return {"voltajes_nodo": {}, "corrientes_elemento": {e["nombre"]: 0.0 for e in elementos_resistivos},
                "corrientes_fuente": {f["nombre"]: 0.0 for f in fuentes},
                "potencia_disipada_w": {e["nombre"]: 0.0 for e in elementos_resistivos},
                "circuito_abierto": True}

    G = np.zeros((n, n))
    for e in elementos_resistivos:
        r = e["resistencia_ohm"] if e["resistencia_ohm"] and e["resistencia_ohm"] > 0 else 1e15
        g = 1.0 / r
        a, b = e["a"], e["b"]
        if a != nodo_referencia:
            G[idx[a], idx[a]] += g
        if b != nodo_referencia:
            G[idx[b], idx[b]] += g
        if a != nodo_referencia and b != nodo_referencia:
            G[idx[a], idx[b]] -= g
            G[idx[b], idx[a]] -= g

    B = np.zeros((n, m))
    for k, f in enumerate(fuentes):
        p, ng = f["nodo_pos"], f["nodo_neg"]
        if p != nodo_referencia:
            B[idx[p], k] += 1.0
        if ng != nodo_referencia:
            B[idx[ng], k] -= 1.0

    A_sup = np.hstack([G, B])
    A_inf = np.hstack([B.T, np.zeros((m, m))])
    A = np.vstack([A_sup, A_inf]) if m > 0 else G
    z = np.zeros(n + m)
    for k, f in enumerate(fuentes):
        z[n + k] = f["tension_v"]

    try:
        x = np.linalg.solve(A, z)
    except np.linalg.LinAlgError:
        return {"voltajes_nodo": {nd: 0.0 for nd in nodos},
                "corrientes_elemento": {e["nombre"]: 0.0 for e in elementos_resistivos},
                "corrientes_fuente": {f["nombre"]: 0.0 for f in fuentes},
                "potencia_disipada_w": {e["nombre"]: 0.0 for e in elementos_resistivos},
                "circuito_abierto": True}

    voltajes_nodo = {nodo_referencia: 0.0}
    for nd, i in idx.items():
        voltajes_nodo[nd] = float(x[i])

    corrientes_elemento, potencias = {}, {}
    for e in elementos_resistivos:
        r = e["resistencia_ohm"] if e["resistencia_ohm"] and e["resistencia_ohm"] > 0 else 1e15
        i_elem = (voltajes_nodo[e["a"]] - voltajes_nodo[e["b"]]) / r
        corrientes_elemento[e["nombre"]] = i_elem
        potencias[e["nombre"]] = (i_elem ** 2) * r

    corrientes_fuente = {f["nombre"]: float(x[n + k]) for k, f in enumerate(fuentes)}

    return {
        "voltajes_nodo": voltajes_nodo,
        "corrientes_elemento": corrientes_elemento,
        "corrientes_fuente": corrientes_fuente,
        "potencia_disipada_w": potencias,
        "circuito_abierto": False,
    }


# ---------------------------------------------------------------------------
# Razonamiento guiado (única parte con LLM) — clasificar, no resolver
# ---------------------------------------------------------------------------

SYSTEM_ELECTRICO = """Analizás una red eléctrica ya resuelta (voltajes y corrientes ya calculados
por Python con análisis nodal exacto). NO resolvés el circuito. Tu trabajo es clasificar la
topología en palabras simples y redactar una advertencia corta si corresponde.

Formato de salida EXACTO, una línea por dato, sin texto adicional:
TOPOLOGIA: <serie|paralelo|red_general|abierto>
MODO_FALLA: <ninguno|sobrecorriente|sobretension|sobrecalentamiento|cortocircuito>
NOTA: una frase corta, opcional

Rangos de referencia:
- Un circuito "abierto" no tiene camino cerrado entre la fuente y sus terminales.
- "Cortocircuito": una rama con resistencia total menor a 1 ohm entre los dos polos de una fuente.
- Nunca inventes valores de tensión/corriente/potencia — esos ya están calculados; tu NOTA es
  solo texto para el usuario, nunca una fuente de datos numéricos.

Ejemplo:
TOPOLOGIA: red_general
MODO_FALLA: sobrecalentamiento
NOTA: la resistencia R2 está disipando más potencia de la que puede evacuar por convección.
"""


def _parsear_respuesta_electrica(texto: str) -> dict:
    resultado = {"topologia": "red_general", "modo_falla": "ninguno", "nota": ""}
    if not texto:
        return resultado
    m = re.search(r"TOPOLOGIA:\s*(\w+)", texto, re.IGNORECASE)
    if m:
        resultado["topologia"] = m.group(1).lower()
    m = re.search(r"MODO_FALLA:\s*(\w+)", texto, re.IGNORECASE)
    if m:
        resultado["modo_falla"] = m.group(1).lower()
    m = re.search(r"NOTA:\s*(.+)", texto, re.IGNORECASE)
    if m:
        resultado["nota"] = m.group(1).strip()
    return resultado


def _decidir_criterio_electrico(resumen: dict) -> dict:
    import json
    texto = modelos.llamar(
        "electrico",
        messages=[
            {"role": "system", "content": SYSTEM_ELECTRICO},
            {"role": "user", "content": json.dumps(resumen, ensure_ascii=False)},
        ],
    )
    return _parsear_respuesta_electrica(texto)


def validar_y_corregir_electrico(decision_llm: dict, riesgo_python: str) -> tuple[dict, bool, list[str]]:
    """Gana siempre Python (riesgo_python, ya calculado contra corriente
    máxima real de cada elemento) — igual criterio que termodinamica.py y
    calculo_estructural.py. El LLM solo aporta topología+nota."""
    resultado = dict(decision_llm)
    advertencias = []
    if resultado.get("modo_falla") != riesgo_python:
        advertencias.append(
            f"  MODO_FALLA del modelo ('{resultado.get('modo_falla')}') no coincide con "
            f"el riesgo calculado ('{riesgo_python}') — se usa el calculado."
        )
        resultado["modo_falla"] = riesgo_python
    return resultado, True, advertencias


def evaluar_circuito(elementos_resistivos: list[dict], fuentes: list[dict],
                      corrientes_maximas_a: dict[str, float] | None = None,
                      nodo_referencia: str = "GND") -> dict:
    """API de consulta puntual (con LLM, para mostrarle al usuario un resumen
    en palabras). Para el tick de un modo interactivo, llamar resolver_red()
    directo — esta función solo agrega clasificación + nota una vez.

    corrientes_maximas_a: {nombre_elemento: A} — opcional, si se conoce el
    límite térmico de cada componente (ver corriente_maxima_por_calentamiento_a).
    """
    calculo = resolver_red(elementos_resistivos, fuentes, nodo_referencia)

    riesgo = "ninguno"
    if calculo["circuito_abierto"]:
        riesgo = "abierto"
    elif corrientes_maximas_a:
        for nombre, i in calculo["corrientes_elemento"].items():
            limite = corrientes_maximas_a.get(nombre)
            if limite and abs(i) > limite:
                riesgo = "sobrecalentamiento"
                break

    resumen = {
        "voltajes_nodo": calculo["voltajes_nodo"],
        "corrientes_elemento": calculo["corrientes_elemento"],
        "potencia_disipada_w": calculo["potencia_disipada_w"],
        "riesgo_calculado": riesgo,
    }
    decision = _decidir_criterio_electrico(resumen)
    decision, _, advertencias = validar_y_corregir_electrico(decision, riesgo)

    return {
        **calculo,
        "topologia": decision["topologia"],
        "modo_falla": decision["modo_falla"],
        "nota": decision["nota"],
        "riesgo": riesgo,
        "advertencias": advertencias,
    }


if __name__ == "__main__":
    print("=== Caso simple: dos resistores en serie con una pila de 9V ===")
    elementos = [
        {"nombre": "R1", "a": "V+", "b": "N1", "resistencia_ohm": 100.0},
        {"nombre": "R2", "a": "N1", "b": "GND", "resistencia_ohm": 220.0},
    ]
    fuentes = [{"nombre": "pila", "nodo_pos": "V+", "nodo_neg": "GND", "tension_v": 9.0}]
    r = resolver_red(elementos, fuentes)
    for k, v in r.items():
        print(f"  {k}: {v}")