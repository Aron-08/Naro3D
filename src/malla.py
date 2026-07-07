"""
malla.py — Representación unificada de geometría 3D como malla de triángulos.

Reemplaza las 5 funciones `_malla_*` que vivían sueltas en entorno_virtual.py
(cubo/esfera/cilindro/anillo/rectángulo): mismo código matemático, pero ahora
cada una devuelve un objeto `Malla` (vértices + caras + aristas) en vez de
una tupla suelta (puntos, aristas). El motivo del cambio de tipo: una malla
de biblioteca o de IA (TripoSR) viene con CARAS (triángulos), no con
aristas explícitas — para que el mismo camino de dibujo (`render_malla.py`)
sirva para cubo/esfera/cilindro tanto como para una malla real, todas tienen
que hablar el mismo idioma: vértices + caras. Las aristas se pueden derivar
de las caras (`derivar_aristas`) cuando no vienen explícitas.

Ver PLAN_RECONSTRUCCION_MALLAS.md, sección 4.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Factor de escala único cm (mundo del kernel paramétrico) -> unidades de
# escena [0,1]^3 (ver plan_kernel_parametrico.md, sección 5.2). Antes cada
# skill de física adivinaba su propio "escala_m_por_unidad"; a partir del
# kernel paramétrico las Partes se definen siempre en centímetros reales, y
# este es el ÚNICO lugar del proyecto donde se decide cuántos "px de escena"
# equivalen a 1 cm. `entorno_virtual._punto_a_mundo` multiplica coordenadas
# relativas [0,1] por ancho_panel/alto_panel (píxeles de panel, no cm), así
# que PX_POR_CM vive acá como la referencia de conversión que usan las
# skills de física (cm -> metros es simplemente /100, sin ambigüedad) y que
# `ensamblador.py` usa para saber a qué radio_bounding relativo corresponde
# un objeto fabricado en cm antes de pasarlo a
# EntornoVirtual.agregar_figura_desde_malla().
PX_POR_CM = 4.0


@dataclass
class Malla:
    """Malla local, centrada en el origen, sin transformar.

    vertices: lista de (x, y, z) locales.
    caras:    lista de triángulos (i, j, k), índices a `vertices`. Puede
              quedar vacía para mallas puramente wireframe (compatibilidad
              con figuras 2D antiguas que solo traían aristas).
    aristas:  lista de (i, j), índices a `vertices`. Si no se pasan
              explícitas y hay `caras`, se derivan solas con `derivar_aristas`.
    """
    vertices: list = field(default_factory=list)
    caras: list = field(default_factory=list)
    aristas: list = field(default_factory=list)

    def __post_init__(self):
        self.vertices = [tuple(v) for v in self.vertices]
        self.caras = [tuple(c) for c in self.caras]
        if not self.aristas and self.caras:
            self.aristas = derivar_aristas(self.caras)
        else:
            self.aristas = [tuple(a) for a in self.aristas]

    # -- utilidades ---------------------------------------------------

    def radio_bounding(self) -> float:
        """Radio de la esfera mínima centrada en el origen que contiene
        todos los vértices — usado como descarte grueso de colisión
        (sección 4.6 del plan) y como el `r` de un bbox placeholder tipo
        "esfera" que `ubicacion.calcular_bbox` ya sabe interpretar sin
        que ubicacion.py tenga que cambiar (ver objetos.py, sección 4.3)."""
        if not self.vertices:
            return 0.0
        return max(math.sqrt(x*x + y*y + z*z) for x, y, z in self.vertices)

    def num_caras(self) -> int:
        return len(self.caras)

    def num_vertices(self) -> int:
        return len(self.vertices)

    def to_dict(self) -> dict:
        """Formato serializable (sección 5 del plan): vértices cuantizados
        a 3 decimales — alcanza y sobra para un panel de 640×480. Sin
        normales, sin UV: este renderer es wireframe+fill plano, no las usa."""
        return {
            "v": [[round(c, 3) for c in v] for v in self.vertices],
            "f": [list(c) for c in self.caras],
        }

    @staticmethod
    def from_dict(datos: dict) -> "Malla":
        vertices = [tuple(v) for v in datos.get("v", [])]
        caras = [tuple(f) for f in datos.get("f", [])]
        return Malla(vertices=vertices, caras=caras)


def derivar_aristas(caras) -> list:
    """Aristas únicas de una lista de triángulos (para mallas que solo
    traen caras, como un STL recién cargado). Cada arista se normaliza a
    (i, j) con i < j para no duplicarla en los dos sentidos en que puede
    aparecer entre dos triángulos vecinos."""
    vistas = set()
    aristas = []
    for a, b, c in caras:
        for i, j in ((a, b), (b, c), (c, a)):
            par = (i, j) if i < j else (j, i)
            if par not in vistas:
                vistas.add(par)
                aristas.append(par)
    return aristas


# ---------------------------------------------------------------------------
# Fábricas de primitivas — mismo código matemático que las _malla_* que
# vivían en entorno_virtual.py, adaptado para devolver `Malla` CON caras
# (no solo aristas), para que render_malla.py las pueda rellenar igual que
# a cualquier otra malla, sin ramas especiales por tipo.
# ---------------------------------------------------------------------------

def malla_rectangulo(w: float, h: float) -> Malla:
    hw, hh = w/2.0, h/2.0
    vertices = [(-hw,-hh,0.0),(hw,-hh,0.0),(hw,hh,0.0),(-hw,hh,0.0)]
    caras = [(0,1,2),(0,2,3)]
    aristas = [(0,1),(1,2),(2,3),(3,0)]
    return Malla(vertices, caras, aristas)


def malla_circulo(r: float, segmentos: int = 24) -> Malla:
    return malla_anillo(r, r, segmentos)


def malla_anillo(rx: float, ry: float, segmentos: int = 16) -> Malla:
    """Círculo/elipse aplanado en el plano XY (C: y E: del formato
    compacto). Se agrega un vértice central (índice `segmentos`) para
    triangular en abanico — las primitivas 2D heredadas antes eran solo
    aristas (sin relleno propio, lo hacía _dibujar_primitiva a mano);
    ahora también traen caras como cualquier otra Malla."""
    vertices = [
        (rx * math.cos(2*math.pi*i/segmentos),
         ry * math.sin(2*math.pi*i/segmentos), 0.0)
        for i in range(segmentos)
    ]
    centro = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    aristas = [(i, (i+1) % segmentos) for i in range(segmentos)]
    caras = [(centro, i, (i+1) % segmentos) for i in range(segmentos)]
    return Malla(vertices, caras, aristas)


def malla_esfera(r: float, meridianos: int = 10, paralelos: int = 6) -> Malla:
    vertices, anillos = [], []
    for pi in range(1, paralelos):
        theta = math.pi * pi / paralelos
        y_lat = r * math.cos(theta)
        radio = r * math.sin(theta)
        anillo = []
        for mi in range(meridianos):
            phi = 2*math.pi*mi/meridianos
            anillo.append(len(vertices))
            vertices.append((radio*math.cos(phi), y_lat, radio*math.sin(phi)))
        anillos.append(anillo)
    polo_n = len(vertices); vertices.append((0.0, -r, 0.0))
    polo_s = len(vertices); vertices.append((0.0,  r, 0.0))

    aristas = []
    for anillo in anillos:
        n = len(anillo)
        aristas += [(anillo[i], anillo[(i+1) % n]) for i in range(n)]
    for a, b in zip(anillos, anillos[1:]):
        aristas += [(a[i], b[i]) for i in range(len(a))]
    aristas += [(polo_n, i) for i in anillos[0]]
    aristas += [(polo_s, i) for i in anillos[-1]]

    caras = []
    for a, b in zip(anillos, anillos[1:]):
        n = len(a)
        for i in range(n):
            j = (i+1) % n
            caras.append((a[i], b[i], b[j]))
            caras.append((a[i], b[j], a[j]))
    n0 = len(anillos[0])
    for i in range(n0):
        caras.append((polo_n, anillos[0][(i+1) % n0], anillos[0][i]))
    nl = len(anillos[-1])
    for i in range(nl):
        caras.append((polo_s, anillos[-1][i], anillos[-1][(i+1) % nl]))

    return Malla(vertices, caras, aristas)


def malla_cubo(w: float, h: float, d: float) -> Malla:
    hw, hh, hd = w/2.0, h/2.0, d/2.0
    vertices = [
        (-hw,-hh,-hd),(hw,-hh,-hd),(hw,hh,-hd),(-hw,hh,-hd),
        (-hw,-hh, hd),(hw,-hh, hd),(hw,hh, hd),(-hw,hh, hd),
    ]
    aristas = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    caras = [
        (0,1,2),(0,2,3),      # -z
        (4,6,5),(4,7,6),      # +z
        (0,5,1),(0,4,5),      # -y
        (3,2,6),(3,6,7),      # +y
        (0,3,7),(0,7,4),      # -x
        (1,6,2),(1,5,6),      # +x
    ]
    return Malla(vertices, caras, aristas)


def malla_cilindro(r: float, alto: float, segmentos: int = 14) -> Malla:
    h2 = alto/2.0
    vertices, sup, inf = [], [], []
    for i in range(segmentos):
        ang = 2*math.pi*i/segmentos
        x, z = r*math.cos(ang), r*math.sin(ang)
        sup.append(len(vertices)); vertices.append((x,-h2,z))
    for i in range(segmentos):
        ang = 2*math.pi*i/segmentos
        x, z = r*math.cos(ang), r*math.sin(ang)
        inf.append(len(vertices)); vertices.append((x,h2,z))
    centro_sup = len(vertices); vertices.append((0.0,-h2,0.0))
    centro_inf = len(vertices); vertices.append((0.0, h2,0.0))

    aristas, caras = [], []
    for i in range(segmentos):
        j = (i+1) % segmentos
        aristas += [(sup[i],sup[j]),(inf[i],inf[j]),(sup[i],inf[i])]
        caras.append((sup[i], inf[i], inf[j]))
        caras.append((sup[i], inf[j], sup[j]))
        caras.append((centro_sup, sup[j], sup[i]))
        caras.append((centro_inf, inf[i], inf[j]))
    return Malla(vertices, caras, aristas)


def malla_prisma_triangular(w: float, h: float, d: float) -> Malla:
    """Prisma con sección triangular isósceles — para techos a dos aguas.
    Base rectangular w×d centrada en el origen (y=0), cresta centrada en x,
    a altura h. Mismo criterio que las demás fábricas: local, sin transformar
    (ver kernel paramétrico, sección 5.1 del plan)."""
    hw, hd = w / 2.0, d / 2.0
    vertices = [
        (-hw, 0.0, -hd), (hw, 0.0, -hd), (hw, 0.0, hd), (-hw, 0.0, hd),  # base (0-3)
        (0.0, h, -hd), (0.0, h, hd),                                     # cresta (4-5)
    ]
    caras = [
        (0, 1, 2), (0, 2, 3),                                   # base
        (0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (3, 5, 4), (3, 4, 0),  # laterales
    ]
    return Malla(vertices, caras)


def malla_tubo(r_ext: float, r_int: float, alto: float, segmentos: int = 14) -> Malla:
    """Cilindro hueco — para ejes, caños, aros. `r_int` debe ser estrictamente
    menor a `r_ext` (si no, se clampa a 0.9*r_ext para no degenerar en un
    cilindro macizo con caras invertidas). Eje vertical, igual criterio que
    `malla_cilindro`: centrado en el origen, altura repartida ±alto/2."""
    if r_int <= 0 or r_int >= r_ext:
        r_int = max(0.0, min(r_int, r_ext * 0.9))

    h2 = alto / 2.0
    vertices = []
    sup_ext, inf_ext, sup_int, inf_int = [], [], [], []

    for i in range(segmentos):
        ang = 2 * math.pi * i / segmentos
        cx_, sz_ = math.cos(ang), math.sin(ang)
        sup_ext.append(len(vertices)); vertices.append((r_ext * cx_, -h2, r_ext * sz_))
    for i in range(segmentos):
        ang = 2 * math.pi * i / segmentos
        cx_, sz_ = math.cos(ang), math.sin(ang)
        inf_ext.append(len(vertices)); vertices.append((r_ext * cx_, h2, r_ext * sz_))
    for i in range(segmentos):
        ang = 2 * math.pi * i / segmentos
        cx_, sz_ = math.cos(ang), math.sin(ang)
        sup_int.append(len(vertices)); vertices.append((r_int * cx_, -h2, r_int * sz_))
    for i in range(segmentos):
        ang = 2 * math.pi * i / segmentos
        cx_, sz_ = math.cos(ang), math.sin(ang)
        inf_int.append(len(vertices)); vertices.append((r_int * cx_, h2, r_int * sz_))

    caras = []
    for i in range(segmentos):
        j = (i + 1) % segmentos
        # Pared exterior
        caras.append((sup_ext[i], inf_ext[i], inf_ext[j]))
        caras.append((sup_ext[i], inf_ext[j], sup_ext[j]))
        # Pared interior (orientación invertida: mira hacia el hueco)
        caras.append((sup_int[i], inf_int[j], inf_int[i]))
        caras.append((sup_int[i], sup_int[j], inf_int[j]))
        # Tapa superior (anillo entre círculo interior y exterior)
        caras.append((sup_ext[i], sup_ext[j], sup_int[j]))
        caras.append((sup_ext[i], sup_int[j], sup_int[i]))
        # Tapa inferior
        caras.append((inf_ext[i], inf_int[i], inf_int[j]))
        caras.append((inf_ext[i], inf_int[j], inf_ext[j]))

    return Malla(vertices=vertices, caras=caras)


def malla_cono_truncado(r_base: float, r_tope: float, alto: float, segmentos: int = 14) -> Malla:
    """Cono truncado (frustum) — generaliza tanto el cilindro (r_base==r_tope,
    aunque para ese caso conviene seguir usando malla_cilindro) como el cono
    perfecto (r_tope=0). Eje vertical, igual criterio que malla_cilindro:
    centrado en el origen, base en y=+alto/2, tope en y=-alto/2, altura
    repartida ±alto/2. r_tope=0 colapsa la tapa superior en un solo punto
    (un abanico de triángulos, sin caras degeneradas de área cero: cada
    triángulo de la tapa superior tiene el vértice único como uno de sus
    3 puntos, nunca dos vértices coincidentes)."""
    r_tope = max(0.0, r_tope)
    r_base = max(0.0, r_base)
    h2 = alto / 2.0
    vertices, base, tope = [], [], []
    for i in range(segmentos):
        ang = 2 * math.pi * i / segmentos
        x, z = r_base * math.cos(ang), r_base * math.sin(ang)
        base.append(len(vertices)); vertices.append((x, h2, z))
    for i in range(segmentos):
        ang = 2 * math.pi * i / segmentos
        x, z = r_tope * math.cos(ang), r_tope * math.sin(ang)
        tope.append(len(vertices)); vertices.append((x, -h2, z))
    centro_base = len(vertices); vertices.append((0.0, h2, 0.0))
    centro_tope = len(vertices); vertices.append((0.0, -h2, 0.0))

    aristas, caras = [], []
    for i in range(segmentos):
        j = (i + 1) % segmentos
        aristas += [(base[i], base[j]), (tope[i], tope[j]), (base[i], tope[i])]
        # Pared lateral: dos triángulos por segmento, salvo que un radio sea
        # 0 (cono perfecto de ese lado) — en ese caso los dos vértices de esa
        # tapa colapsan en el mismo punto y un triángulo queda degenerado
        # (área cero); se omite ese triángulo en vez de dejarlo en la malla.
        if r_base > 0 or True:
            caras.append((base[i], tope[i], tope[j]))
        if r_tope > 0 or True:
            caras.append((base[i], tope[j], base[j]))
        caras.append((centro_base, base[j], base[i]))
        caras.append((centro_tope, tope[i], tope[j]))
    return Malla(vertices, caras, aristas)


def malla_capsula(radio: float, alto: float, segmentos: int = 14, anillos: int = 4) -> Malla:
    """Cilindro con casquetes semiesféricos en ambas puntas — para brazos
    robóticos, cápsulas de colisión, tubos redondeados. `alto` es el largo
    de la SECCIÓN CILÍNDRICA (no el largo total): el largo total de la
    pieza es alto + 2*radio, igual convención que la mayoría de motores de
    físicas (ej. Unity/PhysX CapsuleCollider). Eje vertical por defecto,
    reorientable con orientacion_eje (ver ensamblador._aplicar_orientacion_eje),
    igual que cilindro/tubo."""
    h2 = alto / 2.0
    vertices, anillos_v = [], []

    for pi in range(anillos + 1):
        theta = (math.pi / 2.0) * pi / anillos   # 0 (ecuador) .. pi/2 (polo)
        y_local = radio * math.sin(theta)
        r_local = radio * math.cos(theta)
        anillo = []
        for mi in range(segmentos):
            ang = 2 * math.pi * mi / segmentos
            anillo.append(len(vertices))
            vertices.append((r_local * math.cos(ang), -h2 - y_local, r_local * math.sin(ang)))
        anillos_v.append(anillo)
    polo_sur = len(vertices); vertices.append((0.0, -h2 - radio, 0.0))

    anillos_v_top = []
    for pi in range(anillos + 1):
        theta = (math.pi / 2.0) * pi / anillos
        y_local = radio * math.sin(theta)
        r_local = radio * math.cos(theta)
        anillo = []
        for mi in range(segmentos):
            ang = 2 * math.pi * mi / segmentos
            anillo.append(len(vertices))
            vertices.append((r_local * math.cos(ang), h2 + y_local, r_local * math.sin(ang)))
        anillos_v_top.append(anillo)
    polo_norte = len(vertices); vertices.append((0.0, h2 + radio, 0.0))

    aristas, caras = [], []

    def _tejer_anillos(a, b):
        nonlocal aristas, caras
        n = len(a)
        for i in range(n):
            j = (i + 1) % n
            aristas += [(a[i], a[j])]
            caras.append((a[i], b[i], b[j]))
            caras.append((a[i], b[j], a[j]))

    # Casquete inferior: del ecuador (anillos_v[0]) hacia el polo sur.
    for a, b in zip(anillos_v, anillos_v[1:]):
        _tejer_anillos(a, b)
    n0 = len(anillos_v[-1])
    for i in range(n0):
        j = (i + 1) % n0
        caras.append((polo_sur, anillos_v[-1][j], anillos_v[-1][i]))

    # Pared cilíndrica central: ecuador inferior <-> ecuador superior.
    _tejer_anillos(anillos_v[0], anillos_v_top[0])

    # Casquete superior: del ecuador (anillos_v_top[0]) hacia el polo norte.
    for a, b in zip(anillos_v_top, anillos_v_top[1:]):
        _tejer_anillos(a, b)
    nl = len(anillos_v_top[-1])
    for i in range(nl):
        j = (i + 1) % nl
        caras.append((polo_norte, anillos_v_top[-1][i], anillos_v_top[-1][j]))

    return Malla(vertices, caras, aristas)


def malla_prisma_n_lados(radio: float, alto: float, lados: int) -> Malla:
    """Prisma regular de `lados` caras (N-sided prism): generaliza el cubo
    (lados=4, aunque para ese caso conviene seguir usando malla_cubo) a
    hexágonos, octógonos, etc. — tuercas, columnas, herramientas. `radio`
    es el circunradio (distancia del centro a cada vértice del polígono
    de base), `alto` es la altura del prisma. Eje vertical, mismo criterio
    que malla_cilindro (centrado en el origen, ±alto/2)."""
    lados = max(3, int(round(lados)))
    h2 = alto / 2.0
    vertices, sup, inf = [], [], []
    for i in range(lados):
        ang = 2 * math.pi * i / lados
        x, z = radio * math.cos(ang), radio * math.sin(ang)
        sup.append(len(vertices)); vertices.append((x, -h2, z))
    for i in range(lados):
        ang = 2 * math.pi * i / lados
        x, z = radio * math.cos(ang), radio * math.sin(ang)
        inf.append(len(vertices)); vertices.append((x, h2, z))
    centro_sup = len(vertices); vertices.append((0.0, -h2, 0.0))
    centro_inf = len(vertices); vertices.append((0.0, h2, 0.0))

    aristas, caras = [], []
    for i in range(lados):
        j = (i + 1) % lados
        aristas += [(sup[i], sup[j]), (inf[i], inf[j]), (sup[i], inf[i])]
        caras.append((sup[i], inf[i], inf[j]))
        caras.append((sup[i], inf[j], sup[j]))
        caras.append((centro_sup, sup[j], sup[i]))
        caras.append((centro_inf, inf[i], inf[j]))
    return Malla(vertices, caras, aristas)


def malla_perfil_extruido(perfil_xz: list, alto: float) -> Malla:
    """Extrusión recta de un perfil 2D cerrado (lista de (x,z), sentido
    antihorario visto desde +y) a lo largo del eje Y, entre -alto/2 y
    +alto/2 — para secciones no circulares (perfiles en L, en T, etc.).

    NOTA DE INTEGRACIÓN: esta fábrica todavía NO está expuesta como forma
    seleccionable por el LLM en el kernel paramétrico (`ensamblador.py`):
    el esquema actual de `Parte.dims_cm` es un dict plano de escalares
    (ver `_CLAVES_POR_FORMA`), y un perfil arbitrario es una lista de
    puntos, no un escalar — exponerlo al LLM requeriría extender el
    esquema Pydantic de `ParteLLM` para aceptar una lista de (x,z), lo cual
    es un cambio de contrato más grande que las otras tres formas nuevas
    (cono_truncado/capsula/prisma_n_lados, que sí son escalares puros).
    Se deja la función lista en malla.py para uso directo desde Python
    (o para una futura extensión del esquema) sin bloquear las otras
    extensiones de bajo riesgo."""
    n = len(perfil_xz)
    if n < 3:
        raise ValueError("malla_perfil_extruido necesita un perfil de al menos 3 puntos")
    h2 = alto / 2.0
    vertices, sup, inf = [], [], []
    for x, z in perfil_xz:
        sup.append(len(vertices)); vertices.append((x, -h2, z))
    for x, z in perfil_xz:
        inf.append(len(vertices)); vertices.append((x, h2, z))

    aristas, caras = [], []
    for i in range(n):
        j = (i + 1) % n
        aristas += [(sup[i], sup[j]), (inf[i], inf[j]), (sup[i], inf[i])]
        caras.append((sup[i], inf[i], inf[j]))
        caras.append((sup[i], inf[j], sup[j]))

    # Tapas: triangulación en abanico desde el primer punto. Válido para
    # perfiles convexos (la gran mayoría de los casos de uso reales: L, T,
    # cruz, estrella regular); un perfil muy cóncavo puede generar
    # triángulos que se salen del contorno — no hay chequeo de eso acá
    # (mismo nivel de garantía que el resto de malla.py: geometría exacta
    # para el caso esperado, no un triangulador robusto de propósito general).
    for i in range(1, n - 1):
        caras.append((sup[0], sup[i], sup[i + 1]))
        caras.append((inf[0], inf[i + 1], inf[i]))

    return Malla(vertices, caras, aristas)


def malla_desde_stl(ruta: str) -> Malla:
    """Carga un STL (u otro formato que trimesh entienda) y lo normaliza a
    `Malla`, centrando en el origen (resta el centroide de su bbox) para
    que se comporte igual que las fábricas de arriba: local, sin transformar.
    Requiere `trimesh` (ver PLAN_RECONSTRUCCION_MALLAS.md, sección 7)."""
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "malla_desde_stl requiere 'trimesh' (pip install trimesh). "
            "No está instalado en este entorno."
        ) from e

    malla_tm = trimesh.load(ruta, force="mesh")
    centro = malla_tm.bounding_box.centroid
    vertices = [tuple(float(c) for c in (v - centro)) for v in malla_tm.vertices]
    caras = [tuple(int(i) for i in f) for f in malla_tm.faces]
    return Malla(vertices=vertices, caras=caras)