"""
ensamblador.py — Kernel paramétrico determinístico (ver plan_kernel_parametrico.md).

Reemplaza, para los objetos que puede resolver, a los pasos -1/0a/1/2a/2b de
`ia_interprete.py`: en vez de pedirle a un modelo chico que escriba
coordenadas de escena a mano (y que "case" a ojo los decimales de dos partes
que se tocan), el LLM solo decide QUÉ partes tiene el objeto, de qué FORMA
son (catálogo cerrado) y cómo se TOCAN entre sí — exactamente lo mismo que
ya hace bien en el resto del proyecto (ver 00_skill_filtro_ruido_datos.md).
Python resuelve el álgebra de contacto/simetría con un único cálculo exacto
por relación, nunca dos escrituras de texto que "deberían" coincidir.

Este módulo NO llama al LLM. La integración con el modelo (parseo de la
respuesta JSON, validación Pydantic, fallback al pipeline viejo) vive en
`objetos.py :: generar_geometria_parametrica` — acá solo está el kernel
puro: dado un conjunto de `Parte` ya validadas, producir la geometría.

Convención de espacio: todo este módulo trabaja en centímetros reales,
centrado en el origen (0,0,0) del objeto — igual que las fábricas de
`malla.py`. La posición GLOBAL dentro de la escena [0,1]^3 la sigue
decidiendo `ubicacion.py`, sin cambios; este módulo solo resuelve la forma
interna del objeto en su propio espacio local (ver sección 12 del plan:
"ubicacion.py: sin cambios").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import malla as malla_mod
from malla import Malla, PX_POR_CM

try:
    from pydantic import BaseModel, field_validator
    _PYDANTIC_DISPONIBLE = True
except ImportError:
    _PYDANTIC_DISPONIBLE = False


# ---------------------------------------------------------------------------
# Catálogo cerrado de formas (ver plan, sección 4.2)
# ---------------------------------------------------------------------------

FORMAS_VALIDAS = {"caja", "cilindro", "esfera", "prisma_triangular", "tubo"}

# Claves obligatorias de dims_cm por forma — usado tanto para validar como
# para armar mensajes de error útiles ("te faltó 'alto'").
_CLAVES_POR_FORMA = {
    "caja": ("ancho", "alto", "profundo"),
    "cilindro": ("radio", "alto"),
    "esfera": ("radio",),
    "prisma_triangular": ("ancho", "alto", "profundo"),
    "tubo": ("radio_externo", "radio_interno", "alto"),
}

_LADOS_VALIDOS = {"izquierda", "derecha", "abajo", "arriba", "atras", "adelante"}


class ErrorEnsamble(Exception):
    """Error de datos (no de física): contacto a parte inexistente, ciclo de
    dependencias, forma no reconocida. Nunca se corrige en silencio — quien
    llama (objetos.py) debe caer al fallback del pipeline viejo."""


# ---------------------------------------------------------------------------
# 4.1 — Parte: unidad de composición
# ---------------------------------------------------------------------------

@dataclass
class Parte:
    nombre: str                     # "Asiento", "Pata_1" — único dentro del objeto
    forma: str                      # una de FORMAS_VALIDAS
    dims_cm: dict                   # claves dependen de la forma (ver _CLAVES_POR_FORMA)
    contacto: str | None = None     # "toca:<lado_propio>=<lado_otro>:<Parte>" |
                                     # "simetrica_a:<Parte>" | None
    color: str | None = None
    notas: str = ""

    def __post_init__(self):
        if self.forma not in FORMAS_VALIDAS:
            raise ErrorEnsamble(
                f"Forma '{self.forma}' no está en el catálogo cerrado "
                f"({sorted(FORMAS_VALIDAS)})."
            )
        faltantes = [k for k in _CLAVES_POR_FORMA[self.forma] if k not in self.dims_cm]
        if faltantes:
            raise ErrorEnsamble(
                f"Parte '{self.nombre}' (forma={self.forma}) no trae las claves "
                f"obligatorias {faltantes} en dims_cm."
            )


# ---------------------------------------------------------------------------
# Validación Pydantic de la salida del LLM (capa 3, ver skill 00 y sección
# 7.2 del plan) — mismo patrón que _FichaMaterialExtendidaPydantic en
# objetos.py: garantiza tipo/forma antes de construir cualquier `Parte`.
# ---------------------------------------------------------------------------

if _PYDANTIC_DISPONIBLE:
    class ParteLLM(BaseModel):
        nombre: str
        forma: str
        dims_cm: dict
        contacto: str | None = None

        @field_validator("forma")
        @classmethod
        def _forma_valida(cls, v):
            if v not in FORMAS_VALIDAS:
                raise ValueError(f"forma '{v}' no está en el catálogo cerrado")
            return v

        @field_validator("dims_cm")
        @classmethod
        def _dims_numericas(cls, v):
            # El modelo local a veces manda strings numéricos ("45") en vez
            # de números — se aceptan y castean acá, nunca se propaga un
            # str a las fábricas de malla.py (fallarían con TypeError feo).
            limpio = {}
            for k, val in v.items():
                try:
                    limpio[k] = float(val)
                except (TypeError, ValueError):
                    raise ValueError(f"dims_cm['{k}']={val!r} no es numérico")
            return limpio

    class ComposicionLLM(BaseModel):
        partes: list[ParteLLM]
        factible: bool


    def partes_desde_json_llm(datos: dict) -> "list[Parte] | None":
        """Convierte la respuesta cruda (ya parseada a dict) del paso A en una
        lista de `Parte`, o None si no es factible / no valida. Nunca lanza:
        cualquier problema de parseo o de esquema devuelve None, para que
        `objetos.py` caiga al fallback sin excepciones no controladas (ver
        sección 7.2 del plan)."""
        try:
            comp = ComposicionLLM.model_validate(datos)
        except Exception:
            return None
        if not comp.factible or not comp.partes:
            return None
        try:
            return [
                Parte(nombre=p.nombre, forma=p.forma, dims_cm=p.dims_cm, contacto=p.contacto)
                for p in comp.partes
            ]
        except ErrorEnsamble:
            return None
else:
    def partes_desde_json_llm(datos: dict) -> None:
        return None


# ---------------------------------------------------------------------------
# 6.1 — fabricar_malla: dispatcher forma -> fábrica de malla.py
# ---------------------------------------------------------------------------

def fabricar_malla(parte: Parte) -> Malla:
    d = parte.dims_cm
    if parte.forma == "caja":
        return malla_mod.malla_cubo(d["ancho"], d["alto"], d["profundo"])
    if parte.forma == "cilindro":
        return malla_mod.malla_cilindro(d["radio"], d["alto"])
    if parte.forma == "esfera":
        return malla_mod.malla_esfera(d["radio"])
    if parte.forma == "prisma_triangular":
        return malla_mod.malla_prisma_triangular(d["ancho"], d["alto"], d["profundo"])
    if parte.forma == "tubo":
        return malla_mod.malla_tubo(d["radio_externo"], d["radio_interno"], d["alto"])
    raise ErrorEnsamble(f"Forma '{parte.forma}' sin fábrica asociada.")   # no debería pasar, Parte ya valida


# ---------------------------------------------------------------------------
# 6.4 — _anclar_contacto: el corazón matemático del reemplazo
# ---------------------------------------------------------------------------

_EJE_POR_LADO = {
    "izquierda": (0, -1), "derecha": (0, +1),
    "abajo":     (1, -1), "arriba":  (1, +1),
    "atras":     (2, -1), "adelante": (2, +1),
}


def _extremo_en_eje(malla: Malla, eje: int, signo: int) -> float:
    valores = [v[eje] for v in malla.vertices]
    return max(valores) if signo > 0 else min(valores)


def _anclar_contacto(malla: Malla, malla_otra: Malla, centro_otra: tuple,
                      lado_propio: str, lado_otro: str) -> tuple:
    """Calcula el centro EXACTO donde `malla` debe ir para que su cara
    `lado_propio` coincida con la cara `lado_otro` de `malla_otra` (ya
    ubicada en `centro_otra`). Álgebra pura — nunca hay dos números
    'parecidos' que deberían ser iguales: hay un solo cálculo, así que la
    unión es matemáticamente exacta (bit a bit), no una aproximación que un
    modelo de 4B intentó copiar a mano.

    Los dos ejes que no participan del contacto se alinean centrados con
    `centro_otra` por defecto (mismo criterio que "encima, centrado" en
    ubicacion.py para relaciones sin offset explícito)."""
    if lado_propio not in _EJE_POR_LADO:
        raise ErrorEnsamble(f"Lado propio '{lado_propio}' inválido ({sorted(_EJE_POR_LADO)}).")
    if lado_otro not in _EJE_POR_LADO:
        raise ErrorEnsamble(f"Lado otro '{lado_otro}' inválido ({sorted(_EJE_POR_LADO)}).")

    eje_p, signo_p = _EJE_POR_LADO[lado_propio]
    eje_o, signo_o = _EJE_POR_LADO[lado_otro]

    extremo_propio = _extremo_en_eje(malla, eje_p, signo_p)
    extremo_otro = _extremo_en_eje(malla_otra, eje_o, signo_o) + centro_otra[eje_o]

    centro = [0.0, 0.0, 0.0]
    for eje in range(3):
        if eje == eje_p:
            continue
        centro[eje] = centro_otra[eje]
    centro[eje_p] = extremo_otro - extremo_propio
    return tuple(centro)


# ---------------------------------------------------------------------------
# Parseo del campo `contacto` (idéntico en semántica a SYSTEM_FIGURA_
# RAZONAMIENTO / 01_skill_ubicacion_espacial.md — ver sección 4.3 del plan)
# ---------------------------------------------------------------------------

def _parsear_contacto_toca(contacto: str) -> tuple[str, str, str]:
    """'toca:abajo=arriba:Cuerpo' -> ('abajo', 'arriba', 'Cuerpo')."""
    resto = contacto[len("toca:"):]
    if "=" not in resto or ":" not in resto:
        raise ErrorEnsamble(f"Formato de contacto inválido: '{contacto}'.")
    lados, nombre_otra = resto.split(":", 1)
    if "=" not in lados:
        raise ErrorEnsamble(f"Formato de contacto inválido: '{contacto}'.")
    lado_propio, lado_otro = lados.split("=", 1)
    lado_propio, lado_otro, nombre_otra = lado_propio.strip(), lado_otro.strip(), nombre_otra.strip()
    if lado_propio not in _LADOS_VALIDOS or lado_otro not in _LADOS_VALIDOS:
        raise ErrorEnsamble(f"Lados inválidos en contacto '{contacto}'.")
    if not nombre_otra:
        raise ErrorEnsamble(f"Contacto sin nombre de parte de referencia: '{contacto}'.")
    return lado_propio, lado_otro, nombre_otra


def _nombre_referenciado(contacto: str) -> str | None:
    """Nombre de la Parte de la que depende `contacto`, o None si no depende
    de ninguna (contacto=None). Usado por `_orden_topologico`."""
    if contacto is None:
        return None
    if contacto.startswith("toca:"):
        return _parsear_contacto_toca(contacto)[2]
    if contacto.startswith("simetrica_a:"):
        nombre = contacto.split(":", 1)[1].strip()
        if not nombre:
            raise ErrorEnsamble(f"Contacto 'simetrica_a' sin nombre: '{contacto}'.")
        return nombre
    raise ErrorEnsamble(f"Contacto no reconocido: '{contacto}'.")


# ---------------------------------------------------------------------------
# 6.3 — Orden topológico + resolución completa del ensamble
# ---------------------------------------------------------------------------

def _orden_topologico(partes: list[Parte]) -> list[Parte]:
    """Ordena `partes` de forma que toda Parte con `contacto` quede después
    de la Parte que referencia. Detecta ciclos (A toca B, B toca A) y
    referencias a partes inexistentes ANTES de intentar fabricar nada —
    fallar rápido con un error claro, nunca en silencio (ver sección 6.5
    del plan)."""
    por_nombre = {p.nombre: p for p in partes}
    if len(por_nombre) != len(partes):
        raise ErrorEnsamble("Hay nombres de Parte repetidos; cada nombre debe ser único.")

    dependencias: dict[str, str | None] = {}
    for p in partes:
        dep = _nombre_referenciado(p.contacto)
        if dep is not None and dep not in por_nombre:
            raise ErrorEnsamble(
                f"'{p.nombre}' referencia contacto con '{dep}', que no existe entre las partes."
            )
        dependencias[p.nombre] = dep

    resueltas: list[str] = []
    visitando: set[str] = set()
    visitadas: set[str] = set()

    def _visitar(nombre: str, pila: tuple[str, ...] = ()):
        if nombre in visitadas:
            return
        if nombre in visitando:
            ciclo = " -> ".join(pila + (nombre,))
            raise ErrorEnsamble(f"Ciclo de dependencias de contacto detectado: {ciclo}")
        visitando.add(nombre)
        dep = dependencias[nombre]
        if dep is not None:
            _visitar(dep, pila + (nombre,))
        visitando.discard(nombre)
        visitadas.add(nombre)
        resueltas.append(nombre)

    for p in partes:
        _visitar(p.nombre)

    return [por_nombre[n] for n in resueltas]


def resolver_ensamble(partes: list[Parte]) -> dict[str, tuple[Malla, tuple]]:
    """Devuelve {nombre: (malla, centro_cm)} con todas las restricciones de
    contacto/simetría resueltas exactamente. Nunca le pide nada al LLM —
    esta función es 100% determinística y, dadas las mismas `partes`,
    siempre produce exactamente el mismo resultado (ver tests en
    tests/test_ensamblador.py)."""
    if not partes:
        raise ErrorEnsamble("resolver_ensamble() recibió una lista vacía de partes.")

    orden = _orden_topologico(partes)
    resultado: dict[str, tuple[Malla, tuple]] = {}

    for parte in orden:
        malla = fabricar_malla(parte)

        if parte.contacto is None:
            centro = (0.0, 0.0, 0.0)   # ancla en el origen local del objeto;
                                        # la posición global la da ubicacion.py después

        elif parte.contacto.startswith("toca:"):
            lado_propio, lado_otro, nombre_otra = _parsear_contacto_toca(parte.contacto)
            malla_otra, centro_otra = resultado[nombre_otra]   # ya resuelta por _orden_topologico
            centro = _anclar_contacto(malla, malla_otra, centro_otra, lado_propio, lado_otro)

        elif parte.contacto.startswith("simetrica_a:"):
            nombre_otra = parte.contacto.split(":", 1)[1].strip()
            _malla_otra, centro_otra = resultado[nombre_otra]
            centro = (-centro_otra[0], centro_otra[1], centro_otra[2])   # reflejo exacto en X

        else:
            raise ErrorEnsamble(f"Contacto no reconocido: '{parte.contacto}'")

        resultado[parte.nombre] = (malla, centro)

    return resultado


# ---------------------------------------------------------------------------
# Empaquetado: ensamble -> una sola Malla fusionada, centrada en su propio
# centroide de bounding box, lista para malla_ia_async/biblioteca_mallas
# (mismo formato: vértices+caras) y para optimizacion_malla.decimar() si el
# objeto termina con más triángulos que LOD_BAJO.
# ---------------------------------------------------------------------------

def fusionar_ensamble(resultado: dict[str, tuple[Malla, tuple]]) -> Malla:
    """Combina todas las (malla, centro) de `resolver_ensamble()` en una
    sola Malla, con los vértices ya trasladados a su centro y toda la malla
    recentrada para que el origen quede en el centro de su propio bounding
    box (mismo criterio que `malla_desde_stl`, que resta el centroide del
    bbox del STL cargado) — así el objeto ensamblado se comporta como
    cualquier otra Malla local del proyecto: centrada, sin transformar."""
    vertices: list[tuple] = []
    caras: list[tuple] = []

    for _nombre, (malla, centro) in resultado.items():
        offset = len(vertices)
        cx, cy, cz = centro
        vertices.extend((vx + cx, vy + cy, vz + cz) for vx, vy, vz in malla.vertices)
        caras.extend((a + offset, b + offset, c + offset) for a, b, c in malla.caras)

    if not vertices:
        return Malla()

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    ccx = (min(xs) + max(xs)) / 2.0
    ccy = (min(ys) + max(ys)) / 2.0
    ccz = (min(zs) + max(zs)) / 2.0

    vertices_centrados = [(vx - ccx, vy - ccy, vz - ccz) for vx, vy, vz in vertices]
    return Malla(vertices=vertices_centrados, caras=caras)


def ensamblar_partes(partes: list[Parte]) -> Malla:
    """API de conveniencia: resuelve el ensamble completo y devuelve una
    única Malla fusionada en cm, centrada en el origen — lista para que
    `objetos.py::generar_geometria_parametrica` la escale a `radio_bounding`
    relativo (dividiendo por PX_POR_CM y por la escala de escena, ver
    plan sección 5.2) antes de pasarla a
    EntornoVirtual.agregar_figura_desde_malla()."""
    resultado = resolver_ensamble(partes)
    return fusionar_ensamble(resultado)


if __name__ == "__main__":
    print("=== Ensamble sin LLM: silla simple (asiento + respaldo + 4 patas) ===")
    partes_silla = [
        Parte("Asiento", "caja", {"ancho": 45, "alto": 5, "profundo": 45}),
        Parte("Respaldo", "caja", {"ancho": 45, "alto": 45, "profundo": 5},
              contacto="toca:abajo=arriba:Asiento"),
        Parte("Pata_1", "cilindro", {"radio": 2, "alto": 45},
              contacto="toca:arriba=abajo:Asiento"),
    ]
    r = resolver_ensamble(partes_silla)
    for nombre, (malla, centro) in r.items():
        print(f"  {nombre}: centro={tuple(round(c, 3) for c in centro)}, "
              f"vértices={malla.num_vertices()}, caras={malla.num_caras()}")

    malla_final = fusionar_ensamble(r)
    print(f"Malla fusionada: {malla_final.num_vertices()} vértices, "
          f"{malla_final.num_caras()} caras, radio_bounding={malla_final.radio_bounding():.3f} cm")

    print("\n=== Caso de error: contacto a parte inexistente ===")
    try:
        resolver_ensamble([Parte("X", "caja", {"ancho": 1, "alto": 1, "profundo": 1},
                                  contacto="toca:abajo=arriba:NoExiste")])
    except ErrorEnsamble as e:
        print(f"  ErrorEnsamble (esperado): {e}")

    print("\n=== Caso de error: ciclo de dependencias ===")
    try:
        resolver_ensamble([
            Parte("A", "caja", {"ancho": 1, "alto": 1, "profundo": 1}, contacto="toca:abajo=arriba:B"),
            Parte("B", "caja", {"ancho": 1, "alto": 1, "profundo": 1}, contacto="toca:abajo=arriba:A"),
        ])
    except ErrorEnsamble as e:
        print(f"  ErrorEnsamble (esperado): {e}")
