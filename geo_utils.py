"""
geo_utils.py — Primitivas geométricas 2D puras, sin dependencias del resto del
proyecto (a propósito: es el punto común entre ia_interprete.py y geometria.py,
que SÍ tienen una dependencia direccional entre sí -- geometria.py hace
`from ia_interprete import _llamar_modelo` -- así que este test de cruce de
segmentos no puede vivir en ninguno de los dos sin generar un import circular.
Vive acá, y ambos lo importan).

Antes había dos copias de este mismo test (una en cada módulo) escritas en
momentos distintos; esta es la versión que queda como única fuente de verdad,
con tolerancia de punto flotante y manejo de casos colineales/de contacto.
"""


def signo_orientacion(p, q, r) -> int:
    """Signo del producto cruz (q-p) x (r-q): de qué lado de la recta p-q cae r.
    0 si los tres puntos son colineales (dentro de una tolerancia numérica)."""
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else -1


def punto_en_segmento(p, q, r) -> bool:
    """Asumiendo que p, q, r son colineales, True si q cae dentro del rango
    (bounding box) de p-r, es decir si q está sobre el segmento p-r."""
    return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
            min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)


def segmentos_cruzan(a1, a2, b1, b2) -> bool:
    """True si el segmento a1-a2 cruza al segmento b1-b2, incluyendo el caso
    de segmentos colineales que se superponen (tocarse solo en un extremo
    compartido no cuenta -- eso lo filtra quien llama, antes de invocar esto,
    descartando pares de aristas que comparten un vértice)."""
    o1 = signo_orientacion(a1, a2, b1)
    o2 = signo_orientacion(a1, a2, b2)
    o3 = signo_orientacion(b1, b2, a1)
    o4 = signo_orientacion(b1, b2, a2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and punto_en_segmento(a1, b1, a2):
        return True
    if o2 == 0 and punto_en_segmento(a1, b2, a2):
        return True
    if o3 == 0 and punto_en_segmento(b1, a1, b2):
        return True
    if o4 == 0 and punto_en_segmento(b1, a2, b2):
        return True
    return False
