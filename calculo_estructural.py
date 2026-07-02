"""
calculo_estructural.py — Tensión, factor de seguridad, deflexión y pandeo de
un objeto ya definido (geometría + material) sometido a un caso de carga.
Pensado para conectarse con MechanismRegistry de cad2d.py (engranajes, ejes,
tornillos con un torque/fuerza de diseño) y con la ficha de objetos.py.

Filosofía (ver skill 05_skill_calculo_estructural.md, y skill 00 — filtro de
ruido): el LLM NUNCA hace la cuenta de resistencia de materiales de punta a
punta. Se usa solo para elegir CRITERIO — qué fórmula aplica (viga/columna/
eje), qué resistencia usar como referencia, y el modo de falla dominante.
Toda la aritmética (viga simple, esfuerzo axial, torsión, pandeo de Euler) es
Python puro con fórmulas de libro. Si la elección del LLM no matchea con la
regla determinística tipo_carga+tipo_apoyo -> fórmula, Python gana siempre
(igual criterio que termodinamica.py con RIESGO/BIOT_ESTIMADO).

Conexión con los otros módulos nuevos:
    - geometria.py   : da el área/perímetro exactos de la silicueta 2D ya
                        auditada (auditar_geometria), en unidades de escena.
    - ubicacion.py    : da el bounding box real de la figura (calcular_bbox),
                        del que se saca la dimensión mayor (longitud) y menor
                        (espesor característico) de la pieza.
    Como el proyecto trabaja en coordenadas de escena normalizadas [0,1]³
    (no metros — ver la misma nota en termodinamica.py), la conversión a
    unidades reales necesita un factor de escala explícito
    (`escala_m_por_unidad`) que quien llama debe indicar; nunca se asume 1:1
    en silencio. Ver `geometria_estructural_desde_figura()`.

Requiere lo mismo que ia_interprete.py (Ollama corriendo + el modelo
configurado).
"""

import math
import re

from ia_interprete import _llamar_modelo   # reutiliza el wrapper de Ollama ya probado
import geometria as geo                    # área/perímetro exactos de la sección (skill 02)
from ubicacion import calcular_bbox        # bbox real de la figura (skill 01)


TEMPERATURA_ESTRUCTURAL = 0.15   # baja: esta skill decide criterio, no crea contenido

_PALABRAS_CARGA_CICLICA = ("repetid", "vibrac", "ciclic", "cíclic", "fatiga", "alternante")

# Fallback si falta limite_fatiga_mpa en la ficha (la skill 03 lo agrega,
# pero puede no estar generado todavía) — 0.45x tracción es la regla que ya
# usa 03_skill_ciencia_materiales.md como referencia típica.
_FACTOR_FATIGA_DEFECTO = 0.45
# Fallback si falta resistencia_corte_mpa — estimación de Tresca (τ_adm ≈ 0.6·σ_tracción).
_FACTOR_CORTE_DEFECTO = 0.6


# ---------------------------------------------------------------------------
# Conexión con geometria.py + ubicacion.py — ficha geométrica real (Python puro)
# ---------------------------------------------------------------------------

def geometria_estructural_desde_figura(figura: dict, escala_m_por_unidad: float,
                                        longitud_m: float | None = None,
                                        tipo_apoyo: str = "simple") -> dict:
    """Arma la ficha {"area_seccion_m2", "espesor_caracteristico_m",
    "longitud_m", "tipo_apoyo", "forma_seccion"} que necesita evaluar_carga(),
    a partir de una figura ya generada por el pipeline — sin volver a
    preguntarle nada de geometría al LLM (regla 4 del filtro de ruido).

    - área de la sección: geometria.auditar_geometria(figura, "mecanismo")
      ya calcula el área exacta (shoelace) de la silueta 2D auditada.
    - dimensiones: ubicacion.calcular_bbox(...) da el bounding box real de
      la figura; la dimensión mayor se usa como longitud (si no se pasa
      `longitud_m` explícito) y la menor como espesor característico.
    - forma_seccion: "circular" si la figura tiene una primitiva de círculo
      o esfera, "rectangular" en cualquier otro caso (aproximación).

    `escala_m_por_unidad` es OBLIGATORIO y explícito a propósito: las
    coordenadas de escena están en [0,1]³, no en metros, así que no hay un
    factor "por defecto" seguro — pasar 1.0 a mano si de verdad se quiere
    tratar la escena como metros.
    """
    reporte = geo.auditar_geometria(figura, "mecanismo")
    area_escena = abs(reporte["area"]) if reporte["area"] is not None else 0.0
    area_seccion_m2 = area_escena * (escala_m_por_unidad ** 2)

    bbox = calcular_bbox(figura.get("puntos", []), figura.get("primitivas", []))
    ancho_escena = bbox["x_max"] - bbox["x_min"]
    alto_escena = bbox["y_max"] - bbox["y_min"]
    dimension_mayor = max(ancho_escena, alto_escena) or 0.01
    dimension_menor = (min(ancho_escena, alto_escena) or dimension_mayor)

    longitud_final = longitud_m if longitud_m is not None else dimension_mayor * escala_m_por_unidad
    espesor_caracteristico_m = dimension_menor * escala_m_por_unidad
    if espesor_caracteristico_m <= 0 and area_seccion_m2 > 0:
        # Figura degenerada en un eje (línea): aproximar con sección cuadrada
        # equivalente en área, para no dividir por cero más adelante.
        espesor_caracteristico_m = math.sqrt(area_seccion_m2)

    forma_seccion = "rectangular"
    for prim in figura.get("primitivas", []):
        if prim.get("tipo") in ("circulo", "esfera", "cilindro"):
            forma_seccion = "circular"
            break

    return {
        "area_seccion_m2": area_seccion_m2,
        "espesor_caracteristico_m": espesor_caracteristico_m,
        "longitud_m": longitud_final,
        "tipo_apoyo": tipo_apoyo,
        "forma_seccion": forma_seccion,
    }


# ---------------------------------------------------------------------------
# Propiedades de sección (momento de inercia, distancia a fibra extrema, polar)
# ---------------------------------------------------------------------------

def _propiedades_seccion(area_seccion_m2: float, espesor_caracteristico_m: float,
                          forma_seccion: str) -> dict:
    """Devuelve {"I", "c", "J", "diametro"} — fórmulas de libro, Python puro.
    Para sección rectangular se asume espesor_caracteristico_m = altura h de
    la sección (la dimensión que flexiona) y se despeja el ancho b = A/h.
    Para circular, espesor_caracteristico_m se toma como diámetro."""
    h = espesor_caracteristico_m if espesor_caracteristico_m > 0 else math.sqrt(max(area_seccion_m2, 1e-9))

    if forma_seccion == "circular":
        d = h
        I = math.pi * d ** 4 / 64.0
        J = math.pi * d ** 4 / 32.0
        c = d / 2.0
    else:
        b = area_seccion_m2 / h if h > 0 else 0.0
        I = b * h ** 3 / 12.0
        c = h / 2.0
        # Aproximación (NO exacta para rectangular): usar la sección circular
        # equivalente en área solo para estimar la constante de torsión. Un
        # eje de torsión de verdad casi siempre es circular; si esto se usa
        # con forma_seccion="rectangular" el resultado de torsión es una
        # cota aproximada, no un valor de ingeniería definitivo.
        d_equiv = math.sqrt(4.0 * area_seccion_m2 / math.pi) if area_seccion_m2 > 0 else h
        J = math.pi * d_equiv ** 4 / 32.0

    return {"I": I, "c": c, "J": J}


# ---------------------------------------------------------------------------
# Mapeo determinístico tipo_carga + tipo_apoyo -> fórmula (Python, tabla fija)
# ---------------------------------------------------------------------------

def _formula_determinista(tipo_carga: str, tipo_apoyo: str) -> str:
    tipo_carga = (tipo_carga or "").lower()
    tipo_apoyo = (tipo_apoyo or "").lower()

    if tipo_carga == "torsion":
        return "eje_torsion"
    if tipo_carga == "axial":
        return "columna_compresion"
    if tipo_apoyo in ("empotrado", "empotrada", "voladizo", "cantilever"):
        return "viga_empotrada"
    return "viga_simple"   # puntual/distribuida con apoyo simple (default)


def _es_carga_ciclica(carga: dict) -> bool:
    texto = (carga.get("descripcion") or carga.get("nota") or "").lower()
    return any(p in texto for p in _PALABRAS_CARGA_CICLICA)


def _elegir_resistencia_referencia(tipo_carga: str, material: dict, ciclica: bool) -> tuple[str, float]:
    """Regla determinística (Python, tabla fija — ver skill 05): devuelve
    (etiqueta, valor_mpa). Gana siempre sobre lo que decida el LLM."""
    tipo_carga = (tipo_carga or "").lower()

    if ciclica:
        limite = material.get("limite_fatiga_mpa")
        if not limite or limite <= 0:
            traccion = material.get("resistencia_traccion_mpa") or 0.0
            limite = _FACTOR_FATIGA_DEFECTO * traccion
        return "fatiga", limite

    if tipo_carga == "torsion":
        corte = material.get("resistencia_corte_mpa")
        if not corte or corte <= 0:
            traccion = material.get("resistencia_traccion_mpa") or 0.0
            corte = _FACTOR_CORTE_DEFECTO * traccion
        return "corte", corte

    if tipo_carga == "axial":
        compresion = material.get("resistencia_compresion_mpa") or 0.0
        return "compresion", compresion

    # Flexión (viga simple/empotrada, puntual/distribuida): la fibra
    # traccionada es el punto débil, especialmente en materiales frágiles
    # (hormigón, cerámicos) — ver tabla de la skill 05.
    traccion = material.get("resistencia_traccion_mpa") or 0.0
    return "traccion", traccion


# ---------------------------------------------------------------------------
# Fórmulas de resistencia de materiales (Python puro, ver skill 05)
# ---------------------------------------------------------------------------

def factor_seguridad(resistencia_referencia_mpa: float, sigma_pa: float) -> float:
    """FS = resistencia_referencia / sigma_calculada. Nunca se redondea hacia
    arriba en el mensaje al usuario — eso lo maneja quien presenta el resultado."""
    if sigma_pa <= 0:
        return float("inf")
    return (resistencia_referencia_mpa * 1e6) / sigma_pa


def pandeo_euler(modulo_elasticidad_pa: float, inercia_m4: float, longitud_m: float) -> float:
    """F_crit = pi^2 * E * I / L^2 — carga crítica de pandeo (columna
    biarticulada, caso más común y más conservador de los 4 casos de Euler)."""
    if longitud_m <= 0:
        return float("inf")
    return (math.pi ** 2) * modulo_elasticidad_pa * inercia_m4 / (longitud_m ** 2)


def _viga_simple(F: float, L: float, E: float, I: float, c: float, distribuida: bool) -> dict:
    if distribuida:
        w = F / L if L > 0 else 0.0
        M = w * L ** 2 / 8.0
        delta = 5.0 * w * L ** 4 / (384.0 * E * I) if E * I > 0 else 0.0
    else:
        M = F * L / 4.0
        delta = F * L ** 3 / (48.0 * E * I) if E * I > 0 else 0.0
    sigma = M * c / I if I > 0 else float("inf")
    return {"sigma_pa": sigma, "deflexion_m": delta, "momento_nm": M}


def _viga_empotrada(F: float, L: float, E: float, I: float, c: float, distribuida: bool) -> dict:
    if distribuida:
        w = F / L if L > 0 else 0.0
        M = w * L ** 2 / 2.0
        delta = w * L ** 4 / (8.0 * E * I) if E * I > 0 else 0.0
    else:
        M = F * L
        delta = F * L ** 3 / (3.0 * E * I) if E * I > 0 else 0.0
    sigma = M * c / I if I > 0 else float("inf")
    return {"sigma_pa": sigma, "deflexion_m": delta, "momento_nm": M}


def _columna_compresion(F: float, L: float, E: float, I: float, area_m2: float,
                         espesor_m: float) -> dict:
    sigma = F / area_m2 if area_m2 > 0 else float("inf")
    elongacion = F * L / (area_m2 * E) if area_m2 * E > 0 else 0.0
    esbeltez = L / espesor_m if espesor_m > 0 else 0.0
    resultado = {"sigma_pa": sigma, "deflexion_m": elongacion, "esbeltez": esbeltez,
                 "pandea": False, "carga_critica_n": None}
    if esbeltez > 10:
        f_crit = pandeo_euler(E, I, L)
        resultado["carga_critica_n"] = f_crit
        resultado["pandea"] = f_crit < F   # la carga aplicada ya supera la crítica de Euler
    return resultado


def _eje_torsion(T: float, J: float, r: float) -> dict:
    """T se interpreta en N·m (torque), no en N — a diferencia de las demás
    fórmulas, donde carga.magnitud_n es una fuerza. Documentar esto al pasar
    `carga` con tipo="torsion": magnitud_n = torque en N·m."""
    tau = T * r / J if J > 0 else float("inf")
    return {"sigma_pa": tau, "deflexion_m": None, "momento_nm": T}


def _calcular(formula: str, geometria: dict, material: dict, carga: dict) -> dict:
    F = carga.get("magnitud_n", 0.0)
    L = geometria.get("longitud_m", 0.0)
    E = (material.get("modulo_elasticidad_gpa") or 0.0) * 1e9
    area = geometria.get("area_seccion_m2", 0.0)
    espesor = geometria.get("espesor_caracteristico_m", 0.0)
    forma = geometria.get("forma_seccion", "rectangular")
    seccion = _propiedades_seccion(area, espesor, forma)
    distribuida = (carga.get("tipo", "").lower() == "distribuida")

    if formula == "viga_simple":
        r = _viga_simple(F, L, E, seccion["I"], seccion["c"], distribuida)
    elif formula == "viga_empotrada":
        r = _viga_empotrada(F, L, E, seccion["I"], seccion["c"], distribuida)
    elif formula == "eje_torsion":
        r = _eje_torsion(F, seccion["J"], seccion["c"])
    else:   # columna_compresion / axial_simple
        r = _columna_compresion(F, L, E, seccion["I"], area, espesor)

    r["formula"] = formula
    r["I_m4"] = seccion["I"]
    return r


# ---------------------------------------------------------------------------
# Razonamiento guiado (única parte con LLM) — elegir criterio, no calcular
# ---------------------------------------------------------------------------

SYSTEM_ESTRUCTURAL = """Analizás si una pieza ya definida (geometría + material) aguanta un caso de
carga. NO hacés ninguna cuenta numérica: tensión, factor de seguridad y deflexión ya los
calcula Python con fórmulas exactas de resistencia de materiales. Tu trabajo es elegir el
CRITERIO correcto y redactar una advertencia corta si corresponde.

Recibís un JSON con "geometria" (área de sección, espesor característico, longitud, tipo
de apoyo, forma de sección), "material" (resistencias, módulo de elasticidad) y "carga"
(tipo, magnitud, posición).

Respondé ÚNICAMENTE con este formato, un dato por línea, sin texto adicional:

FORMULA: <viga_simple|viga_empotrada|columna_compresion|eje_torsion>
RESISTENCIA_REFERENCIA: <traccion|compresion|corte|fatiga>
MODO_FALLA: <fractura_fragil|fluencia_ductil|fatiga|pandeo>
NOTA: una frase corta, opcional

Reglas:
  - FORMULA es tu lectura del caso, pero Python la recalcula con una tabla fija
    (tipo_carga + tipo_apoyo) y tiene prioridad si difieren.
  - RESISTENCIA_REFERENCIA: para materiales frágiles (hormigón, cerámicos, vidrio) en
    flexión, la tracción es el punto débil incluso si la carga "aplasta"; para carga axial
    de columna, compresión; para torsión, corte; si la carga es repetida/vibración/cíclica,
    siempre fatiga, nunca tracción simple.
  - MODO_FALLA: "fractura_fragil" para materiales frágiles cerca del límite, "fluencia_ductil"
    para metales dúctiles, "fatiga" si la carga es cíclica, "pandeo" si la pieza es esbelta
    (columna larga y delgada) y la carga es de compresión.
  - NOTA: nunca metas números ahí (tensiones, FS, deflexiones) — se ignoran igual.

=== EJEMPLO 1 ===
entrada: viga de hormigón, apoyo simple, carga puntual centrada de 2000N, material frágil
FORMULA: viga_simple
RESISTENCIA_REFERENCIA: traccion
MODO_FALLA: fractura_fragil
NOTA: el hormigón es mucho más débil a tracción que a compresión, revisar esa fibra

=== EJEMPLO 2 ===
entrada: columna de acero esbelta (L/espesor > 15), carga axial de compresión de 5000N
FORMULA: columna_compresion
RESISTENCIA_REFERENCIA: compresion
MODO_FALLA: pandeo
NOTA: esbeltez alta, el pandeo de Euler probablemente gobierna antes que la compresión simple
"""


def _parsear_respuesta_estructural(texto: str) -> dict:
    resultado = {
        "formula_llm": "viga_simple",
        "resistencia_referencia_llm": "traccion",
        "modo_falla_llm": "fluencia_ductil",
        "nota": "",
    }
    if not texto:
        return resultado

    m = re.search(r"^FORMULA\s*:\s*(\w+)", texto, re.MULTILINE)
    if m and m.group(1).lower() in ("viga_simple", "viga_empotrada", "columna_compresion", "eje_torsion"):
        resultado["formula_llm"] = m.group(1).lower()

    m = re.search(r"^RESISTENCIA_REFERENCIA\s*:\s*(\w+)", texto, re.MULTILINE)
    if m and m.group(1).lower() in ("traccion", "compresion", "corte", "fatiga"):
        resultado["resistencia_referencia_llm"] = m.group(1).lower()

    m = re.search(r"^MODO_FALLA\s*:\s*(\w+)", texto, re.MULTILINE)
    if m and m.group(1).lower() in ("fractura_fragil", "fluencia_ductil", "fatiga", "pandeo"):
        resultado["modo_falla_llm"] = m.group(1).lower()

    m = re.search(r"^NOTA\s*:\s*(.+)$", texto, re.MULTILINE)
    if m:
        resultado["nota"] = m.group(1).strip()

    return resultado


def _decidir_criterio(geometria: dict, material: dict, carga: dict) -> dict:
    resumen = (
        f'{{"geometria": {{"area_seccion_m2": {geometria.get("area_seccion_m2", 0):.6f}, '
        f'"espesor_caracteristico_m": {geometria.get("espesor_caracteristico_m", 0):.5f}, '
        f'"longitud_m": {geometria.get("longitud_m", 0):.4f}, '
        f'"tipo_apoyo": "{geometria.get("tipo_apoyo", "simple")}", '
        f'"forma_seccion": "{geometria.get("forma_seccion", "rectangular")}"}}, '
        f'"material": {{"resistencia_traccion_mpa": {material.get("resistencia_traccion_mpa", 0)}, '
        f'"resistencia_compresion_mpa": {material.get("resistencia_compresion_mpa", 0)}, '
        f'"modulo_elasticidad_gpa": {material.get("modulo_elasticidad_gpa", 0)}}}, '
        f'"carga": {{"tipo": "{carga.get("tipo", "puntual")}", '
        f'"magnitud_n": {carga.get("magnitud_n", 0)}, '
        f'"posicion_relativa": {carga.get("posicion_relativa", 0.5)}}}}}'
    )
    texto = _llamar_modelo(
        messages=[
            {"role": "system", "content": SYSTEM_ESTRUCTURAL},
            {"role": "user", "content": f"entrada: {resumen}"},
        ],
        num_predict=-1,
        temperatura=TEMPERATURA_ESTRUCTURAL,
    )
    return _parsear_respuesta_estructural(texto)


# ---------------------------------------------------------------------------
# Filtro de ruido (capa 3 — skill 00): Python es SIEMPRE la fuente numérica
# ---------------------------------------------------------------------------

def validar_y_corregir_estructural(decision_llm: dict, contexto: dict) -> tuple[dict, bool, list[str]]:
    """decision_llm: salida de _parsear_respuesta_estructural().
    contexto: {"formula_python", "resistencia_referencia_python", "modo_falla_python"}
    ya resueltos determinísticamente. Devuelve (decision_final, es_valida, advertencias).
    Nunca bloqueante: un desacuerdo se loguea, pero Python siempre gana."""
    advertencias: list[str] = []
    resultado = dict(decision_llm)

    formula_python = contexto["formula_python"]
    if resultado["formula_llm"] != formula_python:
        advertencias.append(
            f"  FORMULA del modelo ('{resultado['formula_llm']}') no matchea con la tabla "
            f"determinística tipo_carga+tipo_apoyo ('{formula_python}') — se usa esta última."
        )
    resultado["formula"] = formula_python

    ref_python = contexto["resistencia_referencia_python"]
    if resultado["resistencia_referencia_llm"] != ref_python:
        advertencias.append(
            f"  RESISTENCIA_REFERENCIA del modelo ('{resultado['resistencia_referencia_llm']}') "
            f"no coincide con la regla determinística ('{ref_python}') — se usa esta última."
        )
    resultado["resistencia_referencia"] = ref_python

    resultado["modo_falla"] = contexto.get("modo_falla_python") or resultado["modo_falla_llm"]

    return resultado, True, advertencias


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

def evaluar_carga(geometria: dict, material: dict, carga: dict) -> dict:
    """Punto de entrada de la skill. Todo dato numérico de salida es Python
    puro (fórmulas de resistencia de materiales); el LLM solo aportó el
    criterio (fórmula/resistencia de referencia/modo de falla, con Python
    como fuente de verdad si difieren) y la nota.

    geometria : ver geometria_estructural_desde_figura() — o armarla a mano
                con {"area_seccion_m2", "espesor_caracteristico_m",
                "longitud_m", "tipo_apoyo", "forma_seccion"}.
    material  : ficha de objetos.py (resistencia_traccion_mpa,
                resistencia_compresion_mpa, resistencia_corte_mpa,
                modulo_elasticidad_gpa, limite_fatiga_mpa si existe).
    carga     : {"tipo": "puntual"|"distribuida"|"torsion"|"axial",
                 "magnitud_n": float,   # N, salvo torsión: N·m (torque)
                 "posicion_relativa": float,   # informativo, no usado en las fórmulas actuales
                 "descripcion": str}    # opcional, para detectar carga cíclica

    Devuelve:
        {
          "formula": str, "resistencia_referencia": str, "modo_falla": str,
          "sigma_mpa": float, "resistencia_referencia_mpa": float,
          "factor_seguridad": float, "aguanta": bool, "margen_bajo": bool,
          "deflexion_m": float | None, "pandea": bool | None,
          "carga_critica_pandeo_n": float | None,
          "nota": str, "advertencias": [str, ...],
        }
    """
    advertencias: list[str] = []

    tipo_carga = carga.get("tipo", "puntual")
    tipo_apoyo = geometria.get("tipo_apoyo", "simple")
    ciclica = _es_carga_ciclica(carga)

    formula_python = _formula_determinista(tipo_carga, tipo_apoyo)
    ref_etiqueta, ref_valor_mpa = _elegir_resistencia_referencia(tipo_carga, material, ciclica)
    if ref_valor_mpa <= 0:
        advertencias.append(
            f"  resistencia_referencia ('{ref_etiqueta}') es 0 o no está en la ficha del "
            f"material — el factor de seguridad calculado no es confiable, revisar la ficha."
        )

    calculo = _calcular(formula_python, geometria, material, carga)

    # Modo de falla, calculado también en Python (no solo cualitativo del LLM):
    modo_falla_python = None
    if ciclica:
        modo_falla_python = "fatiga"
    elif calculo.get("pandea"):
        modo_falla_python = "pandeo"
    else:
        ratio_fragilidad = 0.0
        traccion = material.get("resistencia_traccion_mpa") or 0.0
        compresion = material.get("resistencia_compresion_mpa") or 0.0
        if compresion > 0:
            ratio_fragilidad = traccion / compresion
        modo_falla_python = "fractura_fragil" if ratio_fragilidad and ratio_fragilidad < 0.15 else "fluencia_ductil"

    contexto = {
        "formula_python": formula_python,
        "resistencia_referencia_python": ref_etiqueta,
        "modo_falla_python": modo_falla_python,
    }

    decision = _decidir_criterio(geometria, material, carga)
    decision, _, adv_filtro = validar_y_corregir_estructural(decision, contexto)
    advertencias.extend(adv_filtro)

    fs = factor_seguridad(ref_valor_mpa, calculo["sigma_pa"])
    aguanta = fs >= 1.0
    margen_bajo = fs < 1.5

    if not aguanta:
        advertencias.append(
            f"  Factor de seguridad {fs:.2f} < 1.0 — la pieza NO aguanta esta carga con la "
            f"fórmula '{formula_python}' y referencia '{ref_etiqueta}'."
        )
    elif margen_bajo:
        advertencias.append(
            f"  Factor de seguridad {fs:.2f} < 1.5 — aguanta, pero con margen bajo."
        )

    return {
        "formula": decision["formula"],
        "resistencia_referencia": decision["resistencia_referencia"],
        "modo_falla": decision["modo_falla"],
        "sigma_mpa": calculo["sigma_pa"] / 1e6,
        "resistencia_referencia_mpa": ref_valor_mpa,
        "factor_seguridad": fs,
        "aguanta": aguanta,
        "margen_bajo": margen_bajo,
        "deflexion_m": calculo.get("deflexion_m"),
        "pandea": calculo.get("pandea"),
        "carga_critica_pandeo_n": calculo.get("carga_critica_n"),
        "nota": decision["nota"],
        "advertencias": advertencias,
    }


if __name__ == "__main__":
    print("=== Conexión geometria.py + ubicacion.py: viga rectangular a partir de una figura ===")
    viga = {
        "puntos": [[0.3, 0.48], [0.7, 0.48], [0.7, 0.52], [0.3, 0.52]],
        "conexiones": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "primitivas": [],
    }
    # escala arbitraria de ejemplo: 1 unidad de escena = 1.5 metros reales
    geom = geometria_estructural_desde_figura(viga, escala_m_por_unidad=1.5, tipo_apoyo="simple")
    print("geometría derivada:", {k: round(v, 4) if isinstance(v, float) else v for k, v in geom.items()})

    material_acero = {
        "material": "acero",
        "resistencia_traccion_mpa": 400.0,
        "resistencia_compresion_mpa": 400.0,
        "modulo_elasticidad_gpa": 200.0,
    }
    carga_puntual = {"tipo": "puntual", "magnitud_n": 5000.0, "posicion_relativa": 0.5}

    try:
        r = evaluar_carga(geom, material_acero, carga_puntual)
        print("formula:", r["formula"], "| sigma:", round(r["sigma_mpa"], 2), "MPa",
              "| FS:", round(r["factor_seguridad"], 2), "| aguanta:", r["aguanta"],
              "| deflexion:", round(r["deflexion_m"] * 1000, 3), "mm")
        for a in r["advertencias"]:
            print(a)
    except Exception as e:
        print(f"(omitido, requiere Ollama corriendo: {e})")

    print("\n=== Columna esbelta de hormigón (frágil), sin LLM: solo verificación de mapeo/pandeo ===")
    columna = geometria_estructural_desde_figura(
        {"puntos": [[0.48, 0.1], [0.52, 0.1], [0.52, 0.9], [0.48, 0.9]],
         "conexiones": [[0, 1], [1, 2], [2, 3], [3, 0]], "primitivas": []},
        escala_m_por_unidad=3.0, tipo_apoyo="simple",
    )
    print("geometría columna:", {k: round(v, 4) if isinstance(v, float) else v for k, v in columna.items()})
    formula = _formula_determinista("axial", columna["tipo_apoyo"])
    print("formula determinística para carga axial:", formula)
