"""
ubicacion.py — Cálculo de posición 3D de cada objeto dentro de la escena
compartida del entorno virtual (EntornoVirtual). Resuelve "dónde va" un
objeto recién generado por el pipeline de ia_interprete.py, sin que choque
con lo que ya está puesto, y sin pedirle al modelo aritmética que Python
puede resolver solo.

Filosofía (ver skill 01_skill_ubicacion_espacial.md):
    - El LLM SOLO interpreta la intención de un pedido en lenguaje natural
      ("al lado del bloque de cemento", "arriba de la mesa, tocándola"):
      a qué objeto es relativo, qué tipo de relación espacial, de qué lado.
    - Python calcula SIEMPRE el número exacto: centro final, resolución de
      colisiones (AABB), clamps a los límites de la escena. Nunca se confía
      en que el LLM devuelva una coordenada float correcta.
    - Si no hay pedido explícito del usuario, no se llama al modelo en
      absoluto: se usa colocación por defecto (primer hueco libre cerca del
      centro de la escena). Esto ahorra una llamada al modelo en el caso
      más común (agregar un objeto sin indicar dónde).

Coordenadas: todo este módulo trabaja en el mismo espacio relativo [0,1]^3
que usa ia_interprete.py / entorno_virtual.py ANTES de agregar_figura()
(x,y normalizados de panel, z: 0=cámara, 1=fondo, 0.5=plano neutro).

Requisitos: mismo Ollama/modelo que ia_interprete.py (se reutiliza su
wrapper _llamar_modelo, no se abre una conexión nueva).
"""

import math
import re

import modelos   # modelo/temperatura de esta skill vienen de modelos_config.json ("ubicacion_espacial")


# ---------------------------------------------------------------------------
# Constantes de escena
# ---------------------------------------------------------------------------

MARGEN_MINIMO       = 0.04   # separación mínima deseada entre bboxes (unidades relativas)
TOLERANCIA_MARGEN   = 0.02   # por debajo de esto se considera "violación leve" (se clampa sin avisar fuerte)
MARGEN_CAMARA       = 0.03   # margen antes del borde del panel [0,1] para no salir de cámara
PISO_Y              = 0.82   # nivel por defecto de "apoyo" (estante virtual) cuando no hay referencia
MAX_ITER_COLISION   = 12     # tope de iteraciones al resolver colisiones múltiples
# La temperatura y el modelo de esta skill ahora se controlan desde
# modelos_config.json (bloque "ubicacion_espacial"), no acá.


# ---------------------------------------------------------------------------
# Registro de escena — qué hay puesto y dónde
# ---------------------------------------------------------------------------
# No se recalcula a partir de EntornoVirtual (eso implicaría convertir de
# mundo/píxeles a relativo con las dimensiones del panel). En cambio, este
# módulo es la fuente de verdad de "dónde está cada objeto en espacio
# relativo" — main.py debe llamar registrar_objeto()/eliminar_objeto() en
# los mismos puntos donde ya llama entorno.agregar_figura() / figuras.pop().

_escena: dict[str, dict] = {}   # nombre -> {"bbox": {...}, "centro": (x,y,z), "apoya": str}


def registrar_objeto(nombre: str, bbox: dict, centro: tuple, apoya: str) -> None:
    """Guarda/actualiza la posición de `nombre` en el registro de escena."""
    _escena[nombre] = {"bbox": dict(bbox), "centro": tuple(centro), "apoya": apoya}


def eliminar_objeto(nombre: str) -> None:
    """Saca a `nombre` del registro de escena (llamar junto con figuras.pop())."""
    _escena.pop(nombre, None)


def objetos_en_escena_actual() -> list[dict]:
    """Lista de dicts {nombre, bbox, centro, apoya} — snapshot para pasarle a
    calcular_ubicacion() o para depuración."""
    return [{"nombre": n, **datos} for n, datos in _escena.items()]


# ---------------------------------------------------------------------------
# Bounding box de una figura en espacio relativo [0,1]^3
# ---------------------------------------------------------------------------

def _bbox_vacio(cx=0.5, cy=0.5, cz=0.5) -> dict:
    return {"x_min": cx, "x_max": cx, "y_min": cy, "y_max": cy, "z_min": cz, "z_max": cz}


def _expandir(bbox: dict, x: float, y: float, z: float | None = None) -> None:
    bbox["x_min"] = min(bbox["x_min"], x)
    bbox["x_max"] = max(bbox["x_max"], x)
    bbox["y_min"] = min(bbox["y_min"], y)
    bbox["y_max"] = max(bbox["y_max"], y)
    if z is not None:
        bbox["z_min"] = min(bbox["z_min"], z)
        bbox["z_max"] = max(bbox["z_max"], z)


def calcular_bbox(puntos: list, primitivas: list | None = None) -> dict:
    """Bounding box relativo [0,1]^3 de una figura ya generada (puntos +
    primitivas), tal como salen del paso 2/2b de ia_interprete.py. Puramente
    determinístico — nunca se le pide esto al modelo.
    """
    primitivas = primitivas or []
    if not puntos and not primitivas:
        return _bbox_vacio()

    bbox = None

    for p in puntos:
        x, y = p[0], p[1]
        z = p[2] if len(p) > 2 else 0.5
        if bbox is None:
            bbox = _bbox_vacio(x, y, z)
        else:
            _expandir(bbox, x, y, z)

    for prim in primitivas:
        tipo = prim.get("tipo")
        cz = prim.get("cz", 0.5)

        if tipo == "circulo":
            cx, cy, r = prim["cx"], prim["cy"], prim["r"]
            xs, ys, zs = (cx - r, cx + r), (cy - r, cy + r), (cz, cz)
        elif tipo == "rectangulo":
            x, y, w, h = prim["x"], prim["y"], prim["ancho"], prim["alto"]
            xs, ys, zs = (x, x + w), (y, y + h), (cz, cz)
        elif tipo == "elipse":
            cx, cy, rx, ry = prim["cx"], prim["cy"], prim["rx"], prim["ry"]
            xs, ys, zs = (cx - rx, cx + rx), (cy - ry, cy + ry), (cz, cz)
        elif tipo == "esfera":
            cx, cy, cz3, r = prim["cx"], prim["cy"], prim["cz"], prim["r"]
            xs, ys, zs = (cx - r, cx + r), (cy - r, cy + r), (cz3 - r, cz3 + r)
        elif tipo == "cubo":
            cx, cy, cz3 = prim["cx"], prim["cy"], prim["cz"]
            w, h, d = prim["ancho"], prim["alto"], prim.get("profundo", prim["ancho"])
            xs, ys, zs = (cx - w / 2, cx + w / 2), (cy - h / 2, cy + h / 2), (cz3 - d / 2, cz3 + d / 2)
        elif tipo == "cilindro":
            cx, cy, cz3, r, alto = prim["cx"], prim["cy"], prim["cz"], prim["r"], prim["alto"]
            xs, ys, zs = (cx - r, cx + r), (cy - alto / 2, cy + alto / 2), (cz3 - r, cz3 + r)
        else:
            continue

        if bbox is None:
            bbox = _bbox_vacio(xs[0], ys[0], zs[0])
        _expandir(bbox, xs[0], ys[0], zs[0])
        _expandir(bbox, xs[1], ys[1], zs[1])

    return bbox or _bbox_vacio()


def _centro_de_bbox(bbox: dict) -> tuple:
    return (
        (bbox["x_min"] + bbox["x_max"]) / 2,
        (bbox["y_min"] + bbox["y_max"]) / 2,
        (bbox["z_min"] + bbox["z_max"]) / 2,
    )


def _tamano_bbox(bbox: dict) -> tuple:
    return (
        bbox["x_max"] - bbox["x_min"],
        bbox["y_max"] - bbox["y_min"],
        bbox["z_max"] - bbox["z_min"],
    )


def trasladar_bbox(bbox: dict, dx: float, dy: float, dz: float) -> dict:
    return {
        "x_min": bbox["x_min"] + dx, "x_max": bbox["x_max"] + dx,
        "y_min": bbox["y_min"] + dy, "y_max": bbox["y_max"] + dy,
        "z_min": bbox["z_min"] + dz, "z_max": bbox["z_max"] + dz,
    }


def _bbox_centrado_en(bbox_local: dict, centro: tuple) -> dict:
    """Devuelve el bbox local reubicado para que quede centrado en `centro`."""
    cx0, cy0, cz0 = _centro_de_bbox(bbox_local)
    return trasladar_bbox(bbox_local, centro[0] - cx0, centro[1] - cy0, centro[2] - cz0)


# ---------------------------------------------------------------------------
# Colisión AABB (axis-aligned bounding box) en espacio relativo
# ---------------------------------------------------------------------------

def _bboxes_solapan(a: dict, b: dict, margen: float = 0.0) -> bool:
    return not (
        a["x_max"] + margen <= b["x_min"] or b["x_max"] + margen <= a["x_min"] or
        a["y_max"] + margen <= b["y_min"] or b["y_max"] + margen <= a["y_min"] or
        a["z_max"] + margen <= b["z_min"] or b["z_max"] + margen <= a["z_min"]
    )


UMBRAL_PROFUNDIDAD = 0.05   # por debajo de esto, una figura se considera "plana" en Z


def _vector_separacion(a: dict, b: dict, margen: float) -> tuple:
    """Vector mínimo de traslación (MTV) para separar `a` de `b` con el margen
    pedido, empujando `a` en la dirección de menor penetración. Devuelve
    (dx, dy, dz) a sumarle al centro de `a`.

    Nota: la mayoría de las figuras de este proyecto son wireframes casi
    planos (z≈0.5 constante, sin K:/Y:/S: 3D). Si AMBAS figuras son planas,
    separarlas "en Z" matemáticamente no solapa sus bboxes pero en pantalla
    se ven superpuestas igual (la cámara mira casi de frente). Por eso, en
    ese caso se descarta Z como eje de separación y se fuerza X o Y."""
    pen_x = min(a["x_max"] + margen, b["x_max"] + margen) - max(a["x_min"], b["x_min"])
    pen_y = min(a["y_max"] + margen, b["y_max"] + margen) - max(a["y_min"], b["y_min"])
    pen_z = min(a["z_max"] + margen, b["z_max"] + margen) - max(a["z_min"], b["z_min"])

    ambas_planas = (
        (a["z_max"] - a["z_min"]) < UMBRAL_PROFUNDIDAD and
        (b["z_max"] - b["z_min"]) < UMBRAL_PROFUNDIDAD
    )

    penetraciones = {"x": pen_x, "y": pen_y, "z": pen_z}
    if ambas_planas:
        penetraciones["z"] = float("inf")  # nunca elegir Z como eje de escape entre dos figuras planas

    eje = min(penetraciones, key=lambda k: abs(penetraciones[k]) if penetraciones[k] > 0 else float("inf"))
    monto = penetraciones[eje]
    if monto <= 0 or monto == float("inf"):
        return (0.0, 0.0, 0.0)

    ca = _centro_de_bbox(a)
    cb = _centro_de_bbox(b)

    # Dirección: alejar el centro de `a` del centro de `b` sobre el eje elegido
    idx = {"x": 0, "y": 1, "z": 2}[eje]
    direccion = 1.0 if ca[idx] >= cb[idx] else -1.0
    if ca[idx] == cb[idx]:
        direccion = 1.0  # centros coincidentes: convención fija para no quedar trabado

    offset = [0.0, 0.0, 0.0]
    offset[idx] = direccion * monto
    return tuple(offset)


def _resolver_colisiones(bbox_local: dict, centro_inicial: tuple,
                          objetos: list, margen: float = MARGEN_MINIMO) -> tuple:
    """Ajusta `centro_inicial` iterativamente hasta que el bbox (centrado ahí)
    no se solape con ningún objeto de la escena, o hasta agotar iteraciones.
    Devuelve (centro_final, bbox_final, colisiones_sin_resolver: bool).
    """
    centro = list(centro_inicial)
    for _ in range(MAX_ITER_COLISION):
        bbox_actual = _bbox_centrado_en(bbox_local, tuple(centro))
        colisión = None
        for obj in objetos:
            if _bboxes_solapan(bbox_actual, obj["bbox"], margen):
                colisión = obj
                break
        if colisión is None:
            return tuple(centro), bbox_actual, False

        dx, dy, dz = _vector_separacion(bbox_actual, colisión["bbox"], margen)
        if (dx, dy, dz) == (0.0, 0.0, 0.0):
            # No se pudo calcular una dirección útil (bboxes degenerados) — cortar
            break
        centro[0] += dx
        centro[1] += dy
        centro[2] += dz

    bbox_final = _bbox_centrado_en(bbox_local, tuple(centro))
    sigue_colisionando = any(_bboxes_solapan(bbox_final, o["bbox"], margen) for o in objetos)
    return tuple(centro), bbox_final, sigue_colisionando


def _clamp_centro_a_camara(centro: tuple, bbox_local: dict) -> tuple:
    """Evita que el objeto quede parcialmente fuera del panel [0,1]^3."""
    sx, sy, sz = _tamano_bbox(bbox_local)
    lo_x, hi_x = MARGEN_CAMARA + sx / 2, 1 - MARGEN_CAMARA - sx / 2
    lo_y, hi_y = MARGEN_CAMARA + sy / 2, 1 - MARGEN_CAMARA - sy / 2
    lo_z, hi_z = MARGEN_CAMARA + sz / 2, 1 - MARGEN_CAMARA - sz / 2
    cx = min(max(centro[0], lo_x), hi_x) if hi_x > lo_x else 0.5
    cy = min(max(centro[1], lo_y), hi_y) if hi_y > lo_y else 0.5
    cz = min(max(centro[2], lo_z), hi_z) if hi_z > lo_z else 0.5
    return (cx, cy, cz)


# ---------------------------------------------------------------------------
# Colocación por defecto (sin pedido del usuario) — sin LLM
# ---------------------------------------------------------------------------

def _lugar_libre_por_defecto(bbox_local: dict, objetos: list) -> tuple:
    """Punto de partida: centro de escena a la altura del "piso" virtual.
    Si ya hay algo ahí, se resuelve como cualquier otra colisión."""
    centro_inicial = (0.5, PISO_Y, 0.5)
    centro, _, _ = _resolver_colisiones(bbox_local, centro_inicial, objetos)
    return centro


# ---------------------------------------------------------------------------
# Interpretación de un pedido en lenguaje natural (única parte con LLM)
# ---------------------------------------------------------------------------

SYSTEM_UBICACION = """Interpretás dónde debe ir un objeto nuevo dentro de una escena 3D, según lo
que pidió el usuario. NO calculás coordenadas numéricas: solo identificás la intención.

Recibís un JSON con:
  - "objeto_nuevo": nombre del objeto a ubicar.
  - "objetos_en_escena": lista de nombres de objetos ya presentes.
  - "pedido": frase del usuario en lenguaje natural.

Respondé ÚNICAMENTE con este formato, un dato por línea, sin texto adicional:

REFERENCIA: <nombre_exacto_de_objetos_en_escena|piso|ninguno>
REFERENCIA2: <nombre_exacto_de_objetos_en_escena|ninguno>
RELACION: <al_lado|arriba|abajo|encima_tocando|debajo_tocando|entre|centro_escena>
DIRECCION: <izquierda|derecha|frente|atras|ninguna>
ROT_Z: <grados, 0 si no se menciona orientación>

Reglas:
  - REFERENCIA debe ser un nombre que aparece TEXTUAL en "objetos_en_escena", nunca inventado.
    Si el pedido no menciona ningún objeto existente, usar REFERENCIA: ninguno.
  - REFERENCIA2 solo se usa si RELACION es "entre" (ej. "entre el bloque y la silla").
    En cualquier otro caso, REFERENCIA2: ninguno.
  - "encima_tocando" es para cuando un objeto se apoya sobre otro (ej. "arriba de la mesa,
    apoyado"). "arriba" sin más es una posición relativa sin contacto (más alto, separado).
  - DIRECCION solo aplica a "al_lado" (izquierda/derecha) — para las demás relaciones usar "ninguna".
  - Si el pedido no da información de rotación, ROT_Z: 0.

=== EJEMPLOS ===

pedido: "al lado del bloque de cemento, sin tocarlo"
REFERENCIA: Bloque de cemento
REFERENCIA2: ninguno
RELACION: al_lado
DIRECCION: derecha
ROT_Z: 0

pedido: "arriba de la mesa, apoyado sobre ella"
REFERENCIA: mesa
REFERENCIA2: ninguno
RELACION: encima_tocando
DIRECCION: ninguna
ROT_Z: 0

pedido: "entre el bloque de cemento y el cubo de plastico"
REFERENCIA: Bloque de cemento
REFERENCIA2: cubo de plastico
RELACION: entre
DIRECCION: ninguna
ROT_Z: 0

pedido: "en el centro, de canto"
REFERENCIA: ninguno
REFERENCIA2: ninguno
RELACION: centro_escena
DIRECCION: ninguna
ROT_Z: 90
"""


def _parsear_respuesta_ubicacion(texto: str) -> dict:
    """Parser tolerante del formato REFERENCIA:/RELACION:/etc. Nunca devuelve
    None: ante cualquier dato faltante, usa el default más conservador."""
    resultado = {
        "referencia": "ninguno",
        "referencia2": "ninguno",
        "relacion": "centro_escena",
        "direccion": "ninguna",
        "rot_z": 0.0,
    }
    if not texto:
        return resultado

    m_ref = re.search(r"^REFERENCIA\s*:\s*(.+)$", texto, re.MULTILINE)
    if m_ref:
        resultado["referencia"] = m_ref.group(1).strip()

    m_ref2 = re.search(r"^REFERENCIA2\s*:\s*(.+)$", texto, re.MULTILINE)
    if m_ref2:
        resultado["referencia2"] = m_ref2.group(1).strip()

    m_rel = re.search(r"^RELACION\s*:\s*(\w+)", texto, re.MULTILINE)
    if m_rel:
        resultado["relacion"] = m_rel.group(1).strip().lower()

    m_dir = re.search(r"^DIRECCION\s*:\s*(\w+)", texto, re.MULTILINE)
    if m_dir:
        resultado["direccion"] = m_dir.group(1).strip().lower()

    m_rot = re.search(r"^ROT_Z\s*:\s*(-?[\d.]+)", texto, re.MULTILINE)
    if m_rot:
        try:
            resultado["rot_z"] = float(m_rot.group(1))
        except ValueError:
            pass

    return resultado


def _interpretar_pedido(pedido_usuario: str, nombre_nuevo: str, objetos: list) -> dict:
    """Llama al modelo SOLO si hay un pedido explícito. Devuelve el dict de
    _parsear_respuesta_ubicacion(), nunca None."""
    if not pedido_usuario or not pedido_usuario.strip():
        return _parsear_respuesta_ubicacion("")

    nombres_disponibles = [o["nombre"] for o in objetos]
    contenido = (
        f'{{"objeto_nuevo": "{nombre_nuevo}", '
        f'"objetos_en_escena": {nombres_disponibles}, '
        f'"pedido": "{pedido_usuario}"}}'
    )
    texto = modelos.llamar(
        "ubicacion_espacial",
        messages=[
            {"role": "system", "content": SYSTEM_UBICACION},
            {"role": "user", "content": contenido},
        ],
    )
    interpretacion = _parsear_respuesta_ubicacion(texto)

    # Filtro de ruido: la REFERENCIA tiene que existir textual en la escena.
    # Si el modelo inventó un nombre que no está, se descarta (violación leve,
    # no grave — no vale la pena reintentar por esto, se cae a colocación libre).
    if interpretacion["referencia"] not in nombres_disponibles and interpretacion["referencia"] != "piso":
        interpretacion["referencia"] = "ninguno"
    if interpretacion["referencia2"] not in nombres_disponibles:
        interpretacion["referencia2"] = "ninguno"
    if interpretacion["referencia"] == "ninguno":
        interpretacion["relacion"] = "centro_escena"

    return interpretacion


# ---------------------------------------------------------------------------
# Traducir la intención (LLM) a un centro exacto (Python)
# ---------------------------------------------------------------------------

def _bbox_de_referencia(nombre_referencia: str, objetos: list) -> dict | None:
    for o in objetos:
        if o["nombre"] == nombre_referencia:
            return o["bbox"]
    return None


def _centro_por_relacion(interpretacion: dict, bbox_local: dict, objetos: list) -> tuple:
    """Calcula el centro objetivo ANTES de resolver colisiones, puramente a
    partir de la geometría (nunca del LLM)."""
    sx, sy, sz = _tamano_bbox(bbox_local)
    relacion = interpretacion["relacion"]
    ref_bbox = _bbox_de_referencia(interpretacion["referencia"], objetos)

    if relacion == "entre":
        ref2_bbox = _bbox_de_referencia(interpretacion["referencia2"], objetos)
        if ref_bbox and ref2_bbox:
            c1 = _centro_de_bbox(ref_bbox)
            c2 = _centro_de_bbox(ref2_bbox)
            return ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2, (c1[2] + c2[2]) / 2)
        # Falta alguna referencia: cae a colocación libre
        return _lugar_libre_por_defecto(bbox_local, objetos)

    if ref_bbox is None or relacion == "centro_escena":
        return _lugar_libre_por_defecto(bbox_local, objetos)

    cr = _centro_de_bbox(ref_bbox)
    rx, ry, rz = _tamano_bbox(ref_bbox)

    if relacion == "al_lado":
        signo = -1.0 if interpretacion["direccion"] == "izquierda" else 1.0
        dx = signo * (rx / 2 + sx / 2 + MARGEN_MINIMO)
        return (cr[0] + dx, cr[1], cr[2])

    if relacion == "arriba":
        # "arriba" sin contacto: más alto (y menor) y separado en Y
        dy = -(ry / 2 + sy / 2 + MARGEN_MINIMO)
        return (cr[0], cr[1] + dy, cr[2])

    if relacion == "abajo":
        dy = (ry / 2 + sy / 2 + MARGEN_MINIMO)
        return (cr[0], cr[1] + dy, cr[2])

    if relacion == "encima_tocando":
        # Apoyado justo sobre el borde superior de la referencia (y_min de la ref)
        return (cr[0], ref_bbox["y_min"] - sy / 2, cr[2])

    if relacion == "debajo_tocando":
        return (cr[0], ref_bbox["y_max"] + sy / 2, cr[2])

    # Relación no reconocida: default conservador
    return _lugar_libre_por_defecto(bbox_local, objetos)


# ---------------------------------------------------------------------------
# Filtro de ruido / validación final (capa 3, ver skill 00)
# ---------------------------------------------------------------------------

def validar_y_corregir_ubicacion(propuesta: dict, objetos: list) -> tuple[dict, bool, list[str]]:
    """propuesta: {"centro", "bbox_local", "apoya", "rot_z"}
    Devuelve (propuesta_corregida, es_valida, advertencias)."""
    advertencias: list[str] = []
    centro = propuesta["centro"]
    bbox_local = propuesta["bbox_local"]

    centro_clampado = _clamp_centro_a_camara(centro, bbox_local)
    if centro_clampado != tuple(centro):
        advertencias.append(
            f"  Centro {tuple(round(v, 3) for v in centro)} clampado a "
            f"{tuple(round(v, 3) for v in centro_clampado)} (fuera de cámara)"
        )
    centro = centro_clampado

    bbox_final = _bbox_centrado_en(bbox_local, centro)
    colision_grave = False
    for o in objetos:
        if _bboxes_solapan(bbox_final, o["bbox"], margen=0.0):
            colision_grave = True
            advertencias.append(f"  Sigue solapado con '{o['nombre']}' tras resolución — violación grave")

    if propuesta.get("apoya", "ninguno") not in (["piso", "ninguno"] + [o["nombre"] for o in objetos]):
        advertencias.append(f"  'apoya' referencia un objeto inexistente ('{propuesta['apoya']}'), se usa 'piso'")
        propuesta = {**propuesta, "apoya": "piso"}

    resultado = {**propuesta, "centro": centro, "bbox": bbox_final}
    return resultado, (not colision_grave), advertencias


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

def calcular_ubicacion(nombre: str, datos_figura: dict, pedido_usuario: str = "") -> dict:
    """Calcula dónde debe ir `datos_figura` (con la forma {puntos, conexiones,
    primitivas} que ya sale de ia_interprete.py) dentro de la escena actual.

    No modifica `datos_figura` ni el registro de escena — usar
    `ubicar_y_registrar()` para el flujo completo (calcular + aplicar +
    registrar), que es lo que normalmente se llama desde main.py.

    Devuelve:
        {
          "centro": (cx, cy, cz),
          "offset": (dx, dy, dz),        # para trasladar la figura original
          "bbox": {...},                 # bbox final ya ubicado
          "bbox_local": {...},           # bbox original, sin trasladar
          "apoya": str,
          "rot_z": float,
          "advertencias": [str, ...],
        }
    """
    objetos = objetos_en_escena_actual()
    bbox_local = calcular_bbox(datos_figura.get("puntos", []), datos_figura.get("primitivas", []))

    interpretacion = _interpretar_pedido(pedido_usuario, nombre, objetos)
    centro_objetivo = _centro_por_relacion(interpretacion, bbox_local, objetos)

    # La resolución de colisiones es SIEMPRE Python puro, incluso si el
    # centro vino de una relación con referencia (el LLM decidió "al lado
    # de X", pero no garantiza que no choque con un tercer objeto).
    centro_resuelto, _, sigue_colisionando = _resolver_colisiones(bbox_local, centro_objetivo, objetos)

    apoya = interpretacion["referencia"] if interpretacion["relacion"] in (
        "encima_tocando", "debajo_tocando"
    ) else "piso"

    propuesta = {
        "centro": centro_resuelto,
        "bbox_local": bbox_local,
        "apoya": apoya,
        "rot_z": interpretacion["rot_z"],
    }

    final, es_valida, advertencias = validar_y_corregir_ubicacion(propuesta, objetos)
    if not es_valida:
        for a in advertencias:
            print(f"[ubicacion]{a}")

    cx0, cy0, cz0 = _centro_de_bbox(bbox_local)
    offset = (final["centro"][0] - cx0, final["centro"][1] - cy0, final["centro"][2] - cz0)

    return {
        "centro": final["centro"],
        "offset": offset,
        "bbox": final["bbox"],
        "bbox_local": bbox_local,
        "apoya": final["apoya"],
        "rot_z": final["rot_z"],
        "advertencias": advertencias,
    }


def aplicar_offset_a_figura(datos_figura: dict, offset: tuple) -> dict:
    """Traslada puntos y primitivas de `datos_figura` según `offset`, y
    clampa cada coordenada a [0,1] por las dudas (última red de seguridad
    antes de pasarle esto a entorno.agregar_figura)."""
    dx, dy, dz = offset

    def _clamp01(v):
        return min(max(v, 0.0), 1.0)

    puntos_nuevos = []
    for p in datos_figura.get("puntos", []):
        x, y = p[0] + dx, p[1] + dy
        if len(p) > 2:
            z = _clamp01(p[2] + dz)
            puntos_nuevos.append((_clamp01(x), _clamp01(y), z))
        else:
            puntos_nuevos.append((_clamp01(x), _clamp01(y)))

    primitivas_nuevas = []
    for prim in datos_figura.get("primitivas", []):
        p = dict(prim)
        if "cx" in p:
            p["cx"] = _clamp01(p["cx"] + dx)
        if "cy" in p:
            p["cy"] = _clamp01(p["cy"] + dy)
        if "cz" in p:
            p["cz"] = _clamp01(p["cz"] + dz)
        if "x" in p:   # rectángulo
            p["x"] = _clamp01(p["x"] + dx)
        if "y" in p:
            p["y"] = _clamp01(p["y"] + dy)
        primitivas_nuevas.append(p)

    return {
        "puntos": puntos_nuevos,
        "conexiones": datos_figura.get("conexiones", []),
        "primitivas": primitivas_nuevas,
    }


def ubicar_y_registrar(nombre: str, datos_figura: dict, pedido_usuario: str = "") -> dict:
    """Flujo completo pensado para llamarse directo desde main.py, en el
    mismo lugar donde hoy se llama entorno.agregar_figura():

        ubicado = ubicacion.ubicar_y_registrar(nombre, datos, pedido_usuario)
        entorno.agregar_figura(ubicado["puntos"], ubicado["conexiones"],
                               primitivas_relativas=ubicado["primitivas"],
                               nombre=nombre)

    Ya deja la figura trasladada Y registrada en la escena — no hace falta
    llamar registrar_objeto() aparte.
    """
    resultado = calcular_ubicacion(nombre, datos_figura, pedido_usuario)
    datos_ubicados = aplicar_offset_a_figura(datos_figura, resultado["offset"])
    registrar_objeto(nombre, resultado["bbox"], resultado["centro"], resultado["apoya"])
    return {**datos_ubicados, "_ubicacion": resultado}


if __name__ == "__main__":
    print("=== Prueba de ubicacion.py (sin modelo, colocación libre) ===")
    figura_a = {"puntos": [[0.4, 0.6], [0.6, 0.6], [0.6, 0.75], [0.4, 0.75]],
                "conexiones": [[0, 1], [1, 2], [2, 3], [3, 0]], "primitivas": []}
    r1 = ubicar_y_registrar("bloque A", figura_a)
    print("bloque A ->", r1["_ubicacion"]["centro"], r1["_ubicacion"]["advertencias"])

    figura_b = {"puntos": [[0.4, 0.6], [0.6, 0.6], [0.6, 0.75], [0.4, 0.75]],
                "conexiones": [[0, 1], [1, 2], [2, 3], [3, 0]], "primitivas": []}
    r2 = ubicar_y_registrar("bloque B", figura_b)
    print("bloque B ->", r2["_ubicacion"]["centro"], r2["_ubicacion"]["advertencias"])
    print("Se solapan?", _bboxes_solapan(r1["_ubicacion"]["bbox"], r2["_ubicacion"]["bbox"]))

    print("\n=== Prueba con pedido en lenguaje natural (requiere Ollama corriendo) ===")
    figura_c = {"puntos": [[0.45, 0.55], [0.55, 0.55], [0.55, 0.68], [0.45, 0.68]],
                "conexiones": [[0, 1], [1, 2], [2, 3], [3, 0]], "primitivas": []}
    try:
        r3 = ubicar_y_registrar("bloque C", figura_c, "al lado del bloque A, sin tocarlo")
        print("bloque C ->", r3["_ubicacion"]["centro"], r3["_ubicacion"]["advertencias"])
    except Exception as e:
        print(f"(omitido: {e})")