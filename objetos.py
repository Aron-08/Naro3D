"""
objetos.py — Catálogo de objetos del entorno: geometría + propiedades físicas.

Un objeto del catálogo NO es dos cosas separadas: es UN registro con dos partes que se
generan en momentos distintos pero pertenecen al mismo objeto, identificado por su
descripción (el mismo texto que se usa para dibujarlo en `entorno_virtual`):

    {
        "nombre":      "silla de madera",
        "figura":      { puntos, conexiones, primitivas }   <- geometría (ia_interprete)
        "propiedades": { material, peso_kg, ... }            <- ficha física (este módulo)
    }

Orden de generación — IMPORTANTE:
    Primero se pide la geometría y se dibuja. Recién cuando eso terminó se pide la ficha
    de propiedades físicas. NUNCA se piden las dos cosas en paralelo: en hardware sin GPU
    dedicada (como las PCs que usa este proyecto), tener dos respuestas del modelo
    "vivas" al mismo tiempo duplica la memoria que Ollama necesita y puede colgar el
    proceso o tirar el modelo. `crear_objeto()` implementa exactamente esa secuencia y
    avisa con callbacks en cada etapa para que quien dibuja no tenga que esperar a que
    las propiedades estén listas.

Funciones principales:
    crear_objeto(descripcion, callback_figura, callback_propiedades)
        -> orquesta TODO: geometría primero, propiedades después, en secuencia.
    generar_propiedades(descripcion)        -> solo la ficha de propiedades (paso 2 solo)
    actualizar_propiedades(propiedades, pedido) -> ficha recalculada según un pedido
    guardar_objeto / cargar_objeto / listar_objetos / eliminar_objeto -> persistencia

Corriendo este archivo directamente (`python objetos.py`) se abre el editor visual completo
de editor_visual.py (EditorVisual: catálogo, propiedades, apariencia) — no hay un panel
separado acá, para no mantener dos UI tkinter duplicadas para lo mismo.

Requiere lo mismo que ia_interprete.py (Ollama corriendo + el modelo descargado).
"""

import json
import os
import time

import ia_interprete as ia  # reutiliza _extraer_json / catálogo (las llamadas al modelo pasan por modelos.py)
import modelos              # modelo/temperatura de cada skill vienen de modelos_config.json
import geometria as geo     # auditoría topológica (skill 02) antes de guardar/dibujar cualquier figura
import termodinamica as term  # análisis térmico bajo demanda (skill 04) sobre un objeto ya creado
import calculo_estructural as est  # tensión/FS/deflexión bajo demanda (skill 05) sobre un objeto ya creado
import electrico as elec    # ley de Ohm / red de componentes bajo demanda (skill 06)


# ---------------------------------------------------------------------------
# Definición de la ficha de propiedades físicas
# ---------------------------------------------------------------------------

# (clave, etiqueta visible, unidad, es_numerico)
CAMPOS = [
    ("material",                    "Material",                        "",        False),
    ("peso_kg",                     "Peso",                             "kg",      True),
    ("densidad_kg_m3",              "Densidad",                         "kg/m³",   True),
    ("resistencia_traccion_mpa",    "Resistencia a la tracción",        "MPa",     True),
    ("resistencia_corte_mpa",       "Resistencia al corte",             "MPa",     True),
    ("resistencia_compresion_mpa",  "Resistencia a la compresión",      "MPa",     True),
    ("limite_elastico_mpa",         "Límite elástico (fluencia)",       "MPa",     True),
    ("modulo_elasticidad_gpa",      "Módulo de elasticidad (Young)",    "GPa",     True),
    ("dureza",                      "Dureza",                           "",        False),
    ("resistencia_electrica_ohm_m", "Resistividad eléctrica",           "Ω·m",     True),
    ("conductividad_termica_w_mk",  "Conductividad térmica",            "W/(m·K)", True),
    ("punto_fusion_c",              "Punto de fusión",                  "°C",      True),
    ("coef_friccion",               "Coeficiente de fricción",          "",        True),
    ("notas",                       "Notas",                            "",        False),
]

_CLAVES = [c[0] for c in CAMPOS]
_CAMPOS_NUM = {c[0] for c in CAMPOS if c[3]}

_PROPIEDADES_VACIAS = {clave: (0.0 if clave in _CAMPOS_NUM else "") for clave in _CLAVES}
_FIGURA_VACIA = {"puntos": [], "conexiones": [], "primitivas": []}


# ---------------------------------------------------------------------------
# Ficha de materiales EXTENDIDA (skill 03_ciencia_materiales) — paso 1.5
# opcional, solo cuando termodinamica.py o calculo_estructural.py van a
# necesitar estos 5 campos que la ficha básica de arriba no tiene. Se genera
# bajo demanda (no en crear_objeto) y se cachea por material — ver
# generar_propiedades_extendidas() / _propiedades_extendidas_actualizadas().
# ---------------------------------------------------------------------------

CAMPOS_EXTENDIDOS = [
    ("coef_dilatacion_termica_1_k", "Coef. de dilatación térmica", "1/K",      True),
    ("calor_especifico_j_kgk",      "Calor específico",            "J/(kg·K)", True),
    ("modulo_poisson",              "Módulo de Poisson",           "",         True),
    ("limite_fatiga_mpa",           "Límite de fatiga",            "MPa",      True),
    ("temperatura_max_servicio_c",  "Temperatura máx. de servicio","°C",       True),
]

_CLAVES_EXT = [c[0] for c in CAMPOS_EXTENDIDOS]
_PROPIEDADES_EXTENDIDAS_VACIAS = {clave: 0.0 for clave in _CLAVES_EXT}

# Plantillas de seguridad por categoría (misma heurística de palabras clave
# que termodinamica._categoria_material, mismos 4 baldes) — nunca se deja un
# campo en 0 silencioso, un 0 en calor_especifico o Poisson rompe cualquier
# cálculo térmico/estructural después (división por cero incluida).
_VALORES_SEGURIDAD_MATERIAL = {
    "metal_generico":    {"modulo_poisson": 0.30, "calor_especifico_j_kgk": 460,  "coef_dilatacion_termica_1_k": 1.2e-5},
    "plastico_generico": {"modulo_poisson": 0.38, "calor_especifico_j_kgk": 1500, "coef_dilatacion_termica_1_k": 7e-5},
    "mineral_generico":  {"modulo_poisson": 0.20, "calor_especifico_j_kgk": 900,  "coef_dilatacion_termica_1_k": 1.0e-5},
    "organico_generico": {"modulo_poisson": 0.35, "calor_especifico_j_kgk": 1800, "coef_dilatacion_termica_1_k": 0.5e-5},
}


def _categoria_material_ext(nombre_material: str) -> str:
    """Misma heurística por palabras clave que termodinamica._categoria_material,
    para que la plantilla de seguridad elegida sea consistente entre skills."""
    texto = (nombre_material or "").lower()
    if any(p in texto for p in ("acero", "aluminio", "cobre", "hierro", "metal", "bronce", "titanio")):
        return "metal_generico"
    if any(p in texto for p in ("plastico", "pvc", "abs", "pla", "nylon", "polimero")):
        return "plastico_generico"
    if any(p in texto for p in ("madera", "tela", "cuero", "papel", "organico")):
        return "organico_generico"
    return "mineral_generico"


# ---------------------------------------------------------------------------
# Persistencia en disco (objetos_db.json: { nombre: {figura, propiedades, ...} })
# ---------------------------------------------------------------------------

DB_PATH = "objetos_db.json"


def _cargar_db() -> dict:
    if not os.path.exists(DB_PATH):
        return {}
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[objetos] Error al leer {DB_PATH}: {e}")
        return {}


def _guardar_db(db: dict) -> None:
    try:
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[objetos] Error al guardar {DB_PATH}: {e}")


def listar_objetos() -> list[str]:
    """Devuelve los nombres de todos los objetos guardados, ordenados alfabéticamente."""
    return sorted(_cargar_db().keys())


def cargar_objeto(nombre: str) -> dict | None:
    """Devuelve el registro combinado de `nombre`: {nombre, figura, propiedades, ...}."""
    return _cargar_db().get(nombre)


def guardar_objeto(nombre: str, figura: dict | None = None,
                    propiedades: dict | None = None,
                    propiedades_extendidas: dict | None = None) -> dict:
    """Crea o actualiza el registro de `nombre`.

    Se le puede pasar solo `figura` (recién dibujada, todavía sin características), solo
    `propiedades` (llegaron después, o se editaron a mano), `propiedades_extendidas` (skill
    03_ciencia_materiales, generada bajo demanda la primera vez que hace falta para un
    cálculo térmico o estructural), o cualquier combinación de las tres. Lo que no se pase
    se completa con lo que ya hubiera guardado antes, o con una plantilla vacía si es la
    primera vez. Esto es lo que permite guardar el objeto en pasos sucesivos sin pisar lo
    anterior: primero geometría, después propiedades, y más adelante (bajo demanda) la
    ficha extendida de materiales.
    """
    db = _cargar_db()
    existente = db.get(nombre, {})

    figura_final = figura if figura is not None else existente.get("figura", dict(_FIGURA_VACIA))
    propiedades_final = _normalizar_propiedades(
        propiedades if propiedades is not None else existente.get("propiedades", {})
    )
    propiedades_extendidas_final = (
        propiedades_extendidas if propiedades_extendidas is not None
        else existente.get("propiedades_extendidas", {})
    )

    registro = {
        "nombre": nombre,
        "figura": figura_final,
        "propiedades": propiedades_final,
        "propiedades_extendidas": propiedades_extendidas_final,
        "creado": existente.get("creado", time.strftime("%Y-%m-%d %H:%M:%S")),
        "actualizado": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    db[nombre] = registro
    _guardar_db(db)
    print(f"[objetos] '{nombre}' guardado en {DB_PATH}")
    return registro


def eliminar_objeto(nombre: str) -> None:
    db = _cargar_db()
    if nombre in db:
        del db[nombre]
        _guardar_db(db)
        print(f"[objetos] '{nombre}' eliminado de {DB_PATH}")


# ---------------------------------------------------------------------------
# Prompts para el modelo (propiedades físicas)
# ---------------------------------------------------------------------------

SYSTEM_PROPIEDADES_FISICAS = """Sos un asistente técnico que asigna propiedades físicas y de
material realistas a objetos del mundo real, para que puedan simularse dentro de un entorno
virtual (cálculo de peso al moverlos, si se rompen al aplicarles fuerza, si conducen
electricidad o calor, etc.)

Te dan el nombre o la descripción de un objeto. Respondé SOLO con un objeto JSON, sin texto
antes ni después, sin markdown, con EXACTAMENTE estas claves (los valores numéricos son
números puros, sin unidades ni texto dentro):

{
  "material": string,                          // material principal o estructural
  "peso_kg": number,                            // peso real aproximado del objeto completo
  "densidad_kg_m3": number,                     // densidad del material principal
  "resistencia_traccion_mpa": number,           // resistencia última a la tracción
  "resistencia_corte_mpa": number,              // resistencia al corte
  "resistencia_compresion_mpa": number,         // resistencia a la compresión
  "limite_elastico_mpa": number,                // límite elástico / fluencia
  "modulo_elasticidad_gpa": number,             // módulo de Young
  "dureza": string,                             // ej: "120 HB", "6 Mohs", "60 Shore A"
  "resistencia_electrica_ohm_m": number,        // resistividad eléctrica del material
  "conductividad_termica_w_mk": number,
  "punto_fusion_c": number,
  "coef_friccion": number,                      // coef. de fricción aprox. contra una superficie típica
  "notas": string                               // observaciones breves útiles para la simulación
}

Reglas:
  - Usá valores realistas y conocidos para el material correspondiente (ej: el acero ronda
    200 GPa de módulo de elasticidad y ~7850 kg/m³ de densidad; no inventes números al azar).
  - Si el objeto está hecho de varios materiales, elegí el predominante o estructural y
    aclaralo en "notas" (ej: "estructura de madera, herrajes de acero").
  - Toda propiedad tiene un valor realista aunque parezca poco relevante (ej: la madera seca
    SÍ tiene una resistividad eléctrica muy alta; un objeto de tela tiene resistencias
    mecánicas bajas pero no nulas).
  - No agregues claves extra ni elimines ninguna de las pedidas.
  - No expliques nada fuera del JSON.
"""

SYSTEM_ACTUALIZAR_PROPIEDADES = """Tenés la ficha de propiedades físicas de un objeto (JSON) y
un pedido del usuario para corregirla, completarla o actualizarla.

Devolvé el JSON COMPLETO actualizado, con las mismas claves que recibiste. Cambiá lo que el
pedido indica explícitamente y RECALCULÁ cualquier otro valor relacionado para que la ficha
siga siendo físicamente coherente entre sí (ejemplo: si el material pasa de "acero" a
"aluminio", recalculá densidad, resistencias, módulo de elasticidad, punto de fusión y
resistividad eléctrica acordes al aluminio real, no solo cambies el nombre del material).

Respondé SOLO con el JSON actualizado, sin texto antes ni después, sin markdown."""

SYSTEM_PROPIEDADES_EXTENDIDAS = """Sos un asistente técnico de ciencia de materiales. Te dan la
ficha básica de un material, YA CONFIRMADA, como contexto de solo lectura — no la vuelvas a
inventar ni la cambies. Tu única tarea es completar 5 campos adicionales que esa ficha básica
no tiene, necesarios para cálculos térmicos y estructurales posteriores.

Respondé SOLO con un objeto JSON, sin texto antes ni después, sin markdown, con EXACTAMENTE
estas claves (valores numéricos puros, sin unidades ni texto dentro):

{
  "coef_dilatacion_termica_1_k": number,   // 1/K, ej. acero ≈ 1.2e-5, aluminio ≈ 2.3e-5
  "calor_especifico_j_kgk": number,        // J/(kg·K), ej. agua = 4186, acero ≈ 490
  "modulo_poisson": number,                // adimensional, 0.0-0.5, ej. acero ≈ 0.30, hormigón ≈ 0.20
  "limite_fatiga_mpa": number,             // esfuerzo alternante que soporta indefinidamente, típicamente 0.4-0.5x resistencia_traccion_mpa
  "temperatura_max_servicio_c": number     // temperatura antes de perder propiedades mecánicas significativamente
}

Tabla de referencia (no te alejes de estos rangos sin una razón física clara para el material dado):
| Material típico              | Poisson    | Dilatación (1/K)     | Calor esp. (J/kg·K) | Fatiga (fracción de tracción)          |
|-------------------------------|-----------|-----------------------|----------------------|------------------------------------------|
| Acero                         | 0.27-0.30 | 1.1e-5 - 1.3e-5       | 450-500              | 0.45-0.50                                 |
| Aluminio                      | 0.32-0.35 | 2.2e-5 - 2.4e-5       | 890-900              | 0.30-0.40                                 |
| Hormigón                      | 0.15-0.22 | 0.9e-5 - 1.2e-5       | 880-1000             | no aplica (frágil, usar ≈0.55x compresión)|
| Plástico (PLA/ABS genérico)   | 0.35-0.40 | 6e-5 - 9e-5           | 1200-1900            | 0.20-0.30                                 |
| Madera                        | 0.30-0.45 | 0.3e-5 - 0.6e-5 (fibra)| 1200-2700            | no aplica de forma simple                 |

Reglas:
  - Si el material de la ficha cambia respecto a un pedido anterior, recalculá los 5 campos
    juntos y coherentes entre sí para el material nuevo; nunca dejes un campo con el valor
    de un material distinto.
  - No agregues claves extra ni elimines ninguna de las pedidas.
  - No expliques nada fuera del JSON.
"""


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------

def _normalizar_propiedades(datos: dict) -> dict:
    """Completa claves faltantes con la plantilla vacía y fuerza los tipos correctos,
    para que una respuesta parcial o mal tipada del modelo no rompa el resto del programa."""
    resultado = dict(_PROPIEDADES_VACIAS)
    if not isinstance(datos, dict):
        return resultado
    for clave in _CLAVES:
        if clave not in datos:
            continue
        valor = datos[clave]
        if clave in _CAMPOS_NUM:
            try:
                resultado[clave] = float(valor)
            except (ValueError, TypeError):
                pass
        else:
            resultado[clave] = str(valor)
    return resultado


# ---------------------------------------------------------------------------
# Llamadas al modelo — solo la parte de propiedades
# ---------------------------------------------------------------------------

def generar_propiedades(descripcion: str) -> dict | None:
    """Le pide al modelo una ficha de propiedades físicas completa para `descripcion`.
    No toca la geometría: se usa como paso 2, después de que la figura ya existe."""
    print(f"[objetos] Generando propiedades físicas para '{descripcion}'...")
    contenido = modelos.llamar(
        "propiedades_fisicas_basicas",
        messages=[
            {"role": "system", "content": SYSTEM_PROPIEDADES_FISICAS},
            {"role": "user", "content": descripcion},
        ],
    )
    datos = ia._extraer_json(contenido)
    if not datos:
        print("[objetos] El modelo no devolvió un JSON parseable.")
        return None

    print("[objetos] ✓ Propiedades generadas.")
    return _normalizar_propiedades(datos)


def actualizar_propiedades(propiedades: dict, pedido: str) -> dict | None:
    """Le manda al modelo la ficha actual de propiedades más un `pedido` en lenguaje
    natural, y le pide que devuelva la ficha corregida/recalculada."""
    print(f"[objetos] Actualizando propiedades. Pedido: '{pedido}'")
    actuales = {k: propiedades.get(k, _PROPIEDADES_VACIAS[k]) for k in _CLAVES}
    contenido_usuario = (
        f"Ficha actual:\n{json.dumps(actuales, ensure_ascii=False)}\n\n"
        f"Pedido del usuario: {pedido}"
    )
    contenido = modelos.llamar(
        "propiedades_fisicas_actualizar",
        messages=[
            {"role": "system", "content": SYSTEM_ACTUALIZAR_PROPIEDADES},
            {"role": "user", "content": contenido_usuario},
        ],
    )
    datos = ia._extraer_json(contenido)
    if not datos:
        print("[objetos] El modelo no devolvió un JSON parseable en la actualización.")
        return None

    print("[objetos] ✓ Propiedades actualizadas.")
    return _normalizar_propiedades(datos)


# ---------------------------------------------------------------------------
# Ficha extendida de materiales (skill 03_ciencia_materiales)
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, ConfigDict, field_validator
    _PYDANTIC_DISPONIBLE = True
except ImportError:
    _PYDANTIC_DISPONIBLE = False
    print("[objetos] pydantic no está instalado; la skill 'materiales' (JSON nativo de "
          "Ollama) usa solo el clamp físico de validar_y_corregir_material, sin la capa "
          "extra de tipado estricto. Para activarla: pip install pydantic --break-system-packages")


if _PYDANTIC_DISPONIBLE:
    class _FichaMaterialExtendidaPydantic(BaseModel):
        """Esquema estricto para la salida JSON (format=\"json\") de la skill 03
        (ciencia de materiales). NO reemplaza validar_y_corregir_material —
        esa sigue siendo la única fuente de verdad de rangos físicos. Esto
        solo garantiza tipo/forma antes de que los números lleguen ahí, para
        que un campo con basura (string donde va un número, NaN, un objeto
        anidado) se descarte acá en vez de colarse como 0.0 silencioso."""
        model_config = ConfigDict(extra="ignore")

        coef_dilatacion_termica_1_k: float | None = None
        calor_especifico_j_kgk: float | None = None
        modulo_poisson: float | None = None
        limite_fatiga_mpa: float | None = None
        temperatura_max_servicio_c: float | None = None

        @field_validator("*", mode="before")
        @classmethod
        def _vacio_a_none(cls, v):
            # El modelo local a veces manda "" o "null" como texto en vez de
            # omitir la clave directamente; tratarlo como ausente en vez de
            # dejar que el cast a float tire una excepción más abajo.
            if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "nan"):
                return None
            return v


def _normalizar_propiedades_extendidas(datos: dict) -> dict:
    """Mismo criterio que _normalizar_propiedades: completa claves faltantes
    y fuerza tipos numéricos, para que una respuesta parcial o mal tipada del
    modelo no rompa nada río abajo.

    Si pydantic está disponible, primero pasa `datos` por
    `_FichaMaterialExtendidaPydantic` (capa de tipado estricto de la skill
    03) — cualquier campo que no castee limpio a float queda en None y cae
    en la plantilla de seguridad de `validar_y_corregir_material`, en vez de
    quedar como un 0.0 indistinguible de un valor real."""
    resultado = dict(_PROPIEDADES_EXTENDIDAS_VACIAS)
    if not isinstance(datos, dict):
        return resultado

    if _PYDANTIC_DISPONIBLE:
        try:
            validado = _FichaMaterialExtendidaPydantic.model_validate(datos)
            datos = validado.model_dump(exclude_none=True)
        except Exception as e:
            print(f"[objetos][ciencia_materiales] Ficha extendida no pasó el esquema Pydantic "
                  f"({e}); se completa con la plantilla de seguridad.")
            datos = {}

    for clave in _CLAVES_EXT:
        if clave not in datos:
            continue
        try:
            resultado[clave] = float(datos[clave])
        except (ValueError, TypeError):
            pass
    return resultado


def validar_y_corregir_material(datos: dict, propiedades_base: dict) -> tuple[dict, bool, list[str]]:
    """Filtro de ruido (capa 3, skill 00) para la ficha extendida de
    materiales. Nunca deja pasar valores físicamente imposibles y nunca
    devuelve un campo en 0 silencioso — mismo criterio que
    termodinamica._sanear_material y calculo_estructural."""
    d = dict(datos)
    avisos: list[str] = []
    categoria = _categoria_material_ext(propiedades_base.get("material", ""))
    plantilla = _VALORES_SEGURIDAD_MATERIAL[categoria]

    poisson = d.get("modulo_poisson", 0.0)
    if not (0.0 <= poisson <= 0.5):
        avisos.append(f"modulo_poisson fuera de rango físico ({poisson}); clamp a {plantilla['modulo_poisson']}")
        d["modulo_poisson"] = plantilla["modulo_poisson"]

    traccion = propiedades_base.get("resistencia_traccion_mpa") or 0.0
    fatiga = d.get("limite_fatiga_mpa", 0.0)
    if traccion > 0 and fatiga > traccion:
        nuevo = 0.45 * traccion
        avisos.append(f"limite_fatiga_mpa ({fatiga}) mayor que resistencia_traccion_mpa ({traccion}); clamp a {nuevo:.2f}")
        d["limite_fatiga_mpa"] = nuevo
    elif fatiga <= 0 and traccion > 0:
        d["limite_fatiga_mpa"] = 0.45 * traccion
        avisos.append("limite_fatiga_mpa faltante; estimado como 0.45x resistencia_traccion_mpa")

    calor_especifico = d.get("calor_especifico_j_kgk", 0.0)
    if not calor_especifico or calor_especifico <= 0:
        avisos.append(f"calor_especifico_j_kgk faltante o inválido; usando plantilla '{categoria}'")
        d["calor_especifico_j_kgk"] = plantilla["calor_especifico_j_kgk"]

    dilatacion = d.get("coef_dilatacion_termica_1_k", 0.0)
    if not dilatacion or dilatacion <= 0:
        avisos.append(f"coef_dilatacion_termica_1_k faltante o inválido; usando plantilla '{categoria}'")
        d["coef_dilatacion_termica_1_k"] = plantilla["coef_dilatacion_termica_1_k"]

    temp_max = d.get("temperatura_max_servicio_c", 0.0)
    if temp_max <= -273.15 or temp_max == 0.0:
        avisos.append("temperatura_max_servicio_c faltante o inválida; usando valor conservador de 200°C")
        d["temperatura_max_servicio_c"] = 200.0

    # Todo lo de arriba se resuelve por clamp/plantilla en el momento — nunca
    # hace falta reintento contra el modelo para esta skill (a diferencia de
    # geometría o ubicación, acá no hay "colisión" que perseguir).
    return d, True, avisos


def generar_propiedades_extendidas(nombre: str, propiedades_base: dict) -> dict:
    """Skill 03 (ciencia de materiales) — paso 1.5 opcional que EXTIENDE la
    ficha básica ya generada por generar_propiedades() con los 5 campos que
    termodinamica.py y calculo_estructural.py necesitan y la ficha básica no
    tiene: coef_dilatacion_termica_1_k, calor_especifico_j_kgk,
    modulo_poisson, limite_fatiga_mpa, temperatura_max_servicio_c.

    Nunca se le vuelve a preguntar al modelo lo que ya está en
    `propiedades_base` (regla 4 del filtro de ruido, skill 00) — se le manda
    como contexto de solo lectura. Nunca devuelve None: si el modelo no
    responde o responde mal, se completa entero con la plantilla de
    seguridad de la categoría del material (skill 00, política "nunca None
    en cadena")."""
    material = propiedades_base.get("material", "")
    print(f"[objetos] Generando propiedades extendidas de material para '{nombre}' ({material})...")

    contexto = {
        "material": material,
        "densidad_kg_m3": propiedades_base.get("densidad_kg_m3", 0.0),
        "modulo_elasticidad_gpa": propiedades_base.get("modulo_elasticidad_gpa", 0.0),
        "resistencia_traccion_mpa": propiedades_base.get("resistencia_traccion_mpa", 0.0),
        "resistencia_compresion_mpa": propiedades_base.get("resistencia_compresion_mpa", 0.0),
        "dureza": propiedades_base.get("dureza", ""),
    }
    # Skill "materiales" en modelos_config.json (phi4-mini:3.8b + modo JSON
    # nativo de Ollama): validar_y_corregir_material() de abajo es la capa 3
    # (Pydantic/clamp) que exige el contrato de la skill 00.
    contenido = modelos.llamar(
        "materiales",
        messages=[
            {"role": "system", "content": SYSTEM_PROPIEDADES_EXTENDIDAS},
            {"role": "user", "content": json.dumps(contexto, ensure_ascii=False)},
        ],
    )
    datos = ia._extraer_json(contenido)
    if not datos:
        print("[objetos] El modelo no devolvió JSON parseable para propiedades extendidas; "
              "se completa entero con la plantilla de seguridad.")
        datos = {}

    datos = _normalizar_propiedades_extendidas(datos)
    datos, _, avisos = validar_y_corregir_material(datos, propiedades_base)
    for aviso in avisos:
        print(f"[objetos][ciencia_materiales] {aviso}")

    datos["_material_cacheado"] = material  # para invalidar el cache si el material cambia
    print("[objetos] ✓ Propiedades extendidas de material generadas.")
    return datos


def _propiedades_extendidas_actualizadas(nombre: str, registro: dict) -> dict:
    """Devuelve la ficha extendida de `registro`, generándola (skill 03) si
    todavía no existe o si el material base cambió desde la última vez que
    se generó. Mismo criterio de cache que el resto de las skills nuevas
    (00_skill_filtro_ruido_datos.md): no se recalcula si el material no
    cambió, así una actualización de geometría no tira el cálculo térmico ya
    hecho, y viceversa."""
    propiedades = registro.get("propiedades", {})
    material_actual = propiedades.get("material", "")
    ext = registro.get("propiedades_extendidas") or {}

    if ext.get("_material_cacheado") == material_actual and material_actual:
        return ext

    ext = generar_propiedades_extendidas(nombre, propiedades)
    guardar_objeto(nombre, propiedades_extendidas=ext)
    return ext


# ---------------------------------------------------------------------------
# Orquestación: geometría primero, propiedades después — NUNCA en paralelo
# ---------------------------------------------------------------------------

def crear_objeto(descripcion: str, callback_figura=None, callback_propiedades=None) -> dict | None:
    """Crea un objeto completo del entorno: la geometría y sus propiedades físicas,
    pedidas EN SECUENCIA, nunca al mismo tiempo.

    1) Pide la geometría (ia_interprete.generar_figura). Si sale bien, guarda ese avance
       y llama a `callback_figura(nombre, registro)` — pensado para que quien llama pueda
       dibujarla en el entorno de inmediato, sin esperar el paso 2.
    2) Recién ahí, con el primer pedido al modelo ya resuelto y liberado, pide la ficha de
       propiedades físicas (generar_propiedades). Si sale bien, actualiza el registro
       guardado y llama a `callback_propiedades(nombre, registro)`.

    Por qué en secuencia y no en paralelo: pedirle al modelo dos cosas distintas al mismo
    tiempo obliga a tener (al menos) dos contextos vivos en Ollama simultáneamente. En
    hardware sin GPU dedicada y con RAM limitada, eso puede colgar el proceso o forzar a
    Ollama a descargar y recargar el modelo entre pedidos, mucho más lento que hacerlo uno
    detrás del otro.

    Devuelve el registro final (con propiedades, si se pudieron generar) o None si ni
    siquiera la geometría se pudo generar. Si la geometría sale bien pero las propiedades
    fallan, devuelve igual el registro con la figura y propiedades vacías —el objeto queda
    en el entorno, solo que sin ficha física todavía (se puede regenerar después con
    "Actualizar con IA" en el panel).
    """
    print(f"[objetos] === Creando objeto '{descripcion}' (geometría → propiedades) ===")

    # 1) Geometría primero. Esto es lo único que hace falta para poder dibujar.
    figura = ia.generar_figura(descripcion)
    if not figura:
        print(f"[objetos] No se pudo generar la geometría de '{descripcion}'. Se cancela.")
        return None

    # 1b) Auditoría topológica (skill 02_geometria, módulo geometria.py) — 100%
    # determinística, sin volver a llamar al modelo. Cierra contornos casi
    # cerrados y normaliza orientación en silencio; solo avisa fuerte si hay
    # auto-intersección real o geometría degenerada. Acá el destino es
    # "render_only" (escena 3D del entorno) — CAD/CFD auditan de nuevo, más
    # estricto, recién al exportar (ver geo.preparar_para_cad/preparar_para_cfd).
    figura_auditada, geo_ok, geo_avisos = geo.validar_y_corregir_geometria(figura, "render_only")
    for aviso in geo_avisos:
        print(f"[objetos][geometria]{aviso}")
    if geo_ok:
        figura = figura_auditada
    else:
        print(f"[objetos] '{descripcion}': geometría con violaciones graves; se dibuja igual "
              f"(uso_destino=render_only) pero no sería apta si luego se exporta a CAD/CFD.")

    registro = guardar_objeto(descripcion, figura=figura)
    if callback_figura:
        callback_figura(descripcion, registro)

    # 2) Recién ahora se pide la ficha de propiedades físicas del MISMO objeto.
    propiedades = generar_propiedades(descripcion)
    if propiedades is not None:
        registro = guardar_objeto(descripcion, propiedades=propiedades)
    else:
        print(f"[objetos] Geometría OK, pero fallaron las propiedades de '{descripcion}'.")

    if callback_propiedades:
        callback_propiedades(descripcion, registro if propiedades is not None else None)

    print(f"[objetos] === '{descripcion}' terminado ===")
    return registro


def analizar_termico_objeto(nombre: str, condiciones: dict) -> dict | None:
    """Skill 04 (termodinamica.py) aplicada bajo demanda a un objeto YA creado.

    Se activa cuando el usuario pregunta algo térmico de un objeto existente
    ("¿cuánto tarda en enfriarse el bloque de cemento?", "¿qué temperatura
    alcanza la cámara de combustión?"). Nunca vuelve a pedirle geometría o
    material al LLM: usa la ficha ya guardada en objetos_db.json (regla 4 del
    filtro de ruido — skill 00) y solo llama al modelo para el criterio
    térmico (modelo concentrado/gradiente, riesgo).

    condiciones: ver termodinamica.analizar_termico — como mínimo conviene
    pasar "temp_ambiente_c", "temp_fuente_c" y "tipo_proceso"; para combustión
    H2/O2 agregar "masa_h2_kg".

    Devuelve el dict de termodinamica.analizar_termico(), o None si el
    objeto no existe o todavía no tiene ficha de propiedades generada.
    """
    registro = cargar_objeto(nombre)
    if not registro or not registro.get("propiedades"):
        print(f"[objetos] '{nombre}' no existe o todavía no tiene ficha de propiedades; "
              f"no se puede analizar térmicamente.")
        return None

    # Skill 03 (ciencia_materiales), bajo demanda: la ficha básica no trae
    # calor_especifico_j_kgk/coef_dilatacion_termica_1_k/temperatura_max_servicio_c,
    # así que sin esto termodinamica.py siempre caía a sus valores genéricos
    # por categoría en vez de usar el material real del objeto.
    propiedades_extendidas = _propiedades_extendidas_actualizadas(nombre, registro)
    material_completo = {**registro["propiedades"], **propiedades_extendidas}
    material_completo.pop("_material_cacheado", None)

    geometria_termica = term.geometria_termica_desde_ficha(
        peso_kg=material_completo.get("peso_kg", 0.0),
        densidad_kg_m3=material_completo.get("densidad_kg_m3", 0.0),
    )

    try:
        resultado = term.analizar_termico(geometria_termica, material_completo, condiciones)
    except Exception as e:
        print(f"[objetos] No se pudo completar el análisis térmico de '{nombre}': {e}")
        return None

    for aviso in resultado["advertencias"]:
        print(f"[objetos][termodinamica]{aviso}")

    return resultado


def evaluar_carga_objeto(nombre: str, carga: dict, escala_m_por_unidad: float,
                          tipo_apoyo: str = "simple", longitud_m: float | None = None) -> dict | None:
    """Skill 05 (calculo_estructural.py) aplicada bajo demanda a un objeto YA
    creado. Se activa cuando el usuario define/edita un mecanismo con carga
    esperada ("este engranaje va a transmitir 5 Nm") o pregunta directamente
    ("¿aguanta el bloque de cemento si le pongo 200 kg encima?").

    Reutiliza la figura y la ficha de propiedades ya guardadas — nunca le
    vuelve a pedir geometría o material al LLM (regla 4 del filtro de
    ruido). La ficha geométrica real sale de
    calculo_estructural.geometria_estructural_desde_figura(), que a su vez
    conecta geometria.py (área exacta de la sección) y ubicacion.py (bbox
    real de la figura) — ver ese módulo para el detalle.

    escala_m_por_unidad: obligatorio, sin default silencioso — la escena
    vive en coordenadas [0,1]³, no en metros (mismo criterio que
    analizar_termico_objeto/termodinamica.py).
    carga: {"tipo": "puntual"|"distribuida"|"torsion"|"axial", "magnitud_n": float,
            "posicion_relativa": float, "descripcion": str (opcional, para detectar
            carga cíclica)}.

    Devuelve el dict de calculo_estructural.evaluar_carga(), o None si el
    objeto no existe o todavía no tiene figura+ficha de propiedades.
    """
    registro = cargar_objeto(nombre)
    if not registro or not registro.get("figura") or not registro.get("propiedades"):
        print(f"[objetos] '{nombre}' no existe o le falta figura/propiedades; "
              f"no se puede evaluar la carga estructural.")
        return None

    # Skill 03 (ciencia_materiales), bajo demanda: la ficha básica no trae
    # limite_fatiga_mpa/modulo_poisson, así que sin esto
    # _elegir_resistencia_referencia() de calculo_estructural.py siempre
    # caía a resistencia_traccion/compresion en vez de poder usar fatiga
    # real cuando la carga es cíclica.
    propiedades_extendidas = _propiedades_extendidas_actualizadas(nombre, registro)
    material_completo = {**registro["propiedades"], **propiedades_extendidas}
    material_completo.pop("_material_cacheado", None)

    geometria_estructural = est.geometria_estructural_desde_figura(
        registro["figura"], escala_m_por_unidad, longitud_m=longitud_m, tipo_apoyo=tipo_apoyo,
    )

    try:
        resultado = est.evaluar_carga(geometria_estructural, material_completo, carga)
    except Exception as e:
        print(f"[objetos] No se pudo completar el cálculo estructural de '{nombre}': {e}")
        return None

    for aviso in resultado["advertencias"]:
        print(f"[objetos][calculo_estructural]{aviso}")

    return resultado


def evaluar_circuito_objeto(nombres: list[str], fuentes: list[dict],
                             margen_contacto: float = 0.02) -> dict | None:
    """Skill 06 (electrico.py) aplicada a un grupo de objetos YA creados y
    colocados en la escena ("¿cuánta corriente pasa por cada cable si
    conecto la pila acá?"). La topología sale de ubicacion.py (quién toca a
    quién) — nunca se le pregunta al LLM ni a la mano qué está conectado con
    qué, es geometría (regla 4 del filtro de ruido, igual criterio que
    evaluar_carga_objeto/analizar_termico_objeto).

    nombres: objetos a incluir en la red (deben existir y tener figura+propiedades).
    fuentes: [{"nombre", "nodo_pos", "nodo_neg", "tension_v"}] — la polaridad de
    una fuente no se puede inferir de la geometría, así que se pasa explícita;
    nodo_pos/nodo_neg típicamente son terminales "<objeto>#A"/"<objeto>#B" de
    electrico.construir_red_desde_contacto().

    Devuelve el dict de electrico.evaluar_circuito(), o None si falta algún objeto.
    """
    import ubicacion as ubi

    fichas = {}
    for nombre in nombres:
        registro = cargar_objeto(nombre)
        if not registro or not registro.get("propiedades"):
            print(f"[objetos] '{nombre}' no existe o le falta la ficha de propiedades; "
                  f"no se puede evaluar el circuito.")
            return None
        fichas[nombre] = registro["propiedades"]

    objetos_escena = [o for o in ubi.objetos_en_escena_actual() if o["nombre"] in fichas]
    elementos, _mapa = elec.construir_red_desde_contacto(objetos_escena, fichas, margen_contacto)

    try:
        resultado = elec.evaluar_circuito(elementos, fuentes)
    except Exception as e:
        print(f"[objetos] No se pudo completar el análisis eléctrico: {e}")
        return None

    for aviso in resultado["advertencias"]:
        print(f"[objetos][electrico]{aviso}")

    return resultado


# ---------------------------------------------------------------------------
# Panel gráfico
# ---------------------------------------------------------------------------
# El catálogo de objetos ya tiene un editor completo en editor_visual.py
# (EditorVisual: crear, guardar, actualizar con IA, eliminar, más color y
# escala) — corriendo `python objetos.py` directamente se abre ESE panel en
# vez de duplicar uno más chico acá. Mantener dos paneles tkinter separados
# para lo mismo es la clase de duplicación que termina desincronizada (un
# botón "Eliminar" que hace algo distinto en cada uno, por ejemplo); si en
# algún momento hace falta un modo "solo propiedades, sin apariencia", es
# mejor agregarlo como opción dentro de EditorVisual que resucitar un panel
# aparte acá.

def _lanzar_panel():
    import editor_visual
    editor_visual._lanzar()


if __name__ == "__main__":
    _lanzar_panel()