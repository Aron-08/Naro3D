"""
figura.py — Figura3D: forma del entorno, ya sea wireframe/primitivas
heredadas del fallback LLM o una Malla real de biblioteca/IA, con estado
de colisión "congelada"/"activa".

Extraído de entorno_virtual.py (antes vivía junto con cámara y gestos), sin
cambios de fondo respecto a la versión anterior salvo los agregados de la
sección 4.6 del plan: una Figura3D ahora puede llevar una Malla real
(`malla_lod_baja`/`malla_lod_alta`) en vez del wireframe de puntos/
primitivas heredado, con un `radio_bounding` cacheado (en unidades de
mundo) para el descarte grueso de colisión.
"""

from __future__ import annotations

import cv2
import numpy as np

import malla as malla_mod
import render_malla

GRACIA_FRAMES_ACTIVA = 30  # ~1s a 30fps: colchón tras soltar la figura antes de volver a "congelada"


_COLOR_NOMBRE_A_BGR = {
    "rojo":   (0, 0, 255),
    "verde":  (0, 180, 0),
    "azul":   (255, 0, 0),
    "amarillo": (0, 220, 220),
    "naranja": (0, 140, 255),
    "morado": (200, 0, 200),
    "rosa":   (203, 192, 255),
    "cian":   (255, 255, 0),
    "gris":   (140, 140, 140),
    "blanco": (255, 255, 255),
    "negro":  (0, 0, 0),
    "marron": (42, 42, 165),
    "marrón": (42, 42, 165),
}

_DEFAULT_PRIMITIVE_COLORS = {
    "cubo":      (120, 170, 220),
    "cilindro":  (120, 200, 160),
    "esfera":    (160, 140, 240),
    "rectangulo":(215, 160, 120),
    "circulo":   (120, 220, 190),
    "elipse":    (160, 200, 220),
}

_PALETTE = [
    (190, 120, 240),
    (80, 200, 170),
    (180, 160, 70),
    (90, 180, 240),
    (220, 100, 110),
    (140, 220, 120),
]


def _parse_color_value(value):
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(int(max(0, min(255, v))) for v in value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        valor = value.strip().lower()
        if valor in _COLOR_NOMBRE_A_BGR:
            return _COLOR_NOMBRE_A_BGR[valor]
        if valor.startswith("#") and len(valor) == 7:
            try:
                r = int(valor[1:3], 16); g = int(valor[3:5], 16); b = int(valor[5:7], 16)
                return (b, g, r)
            except ValueError:
                return None
        if valor.startswith("0x") and len(valor) == 8:
            try:
                r = int(valor[2:4], 16); g = int(valor[4:6], 16); b = int(valor[6:8], 16)
                return (b, g, r)
            except ValueError:
                return None
    return None


def _color_para_primitiva(prim, idx):
    if "color" in prim:
        parsed = _parse_color_value(prim["color"])
        if parsed is not None:
            return parsed
    tipo = prim.get("tipo")
    if tipo in _DEFAULT_PRIMITIVE_COLORS:
        return _DEFAULT_PRIMITIVE_COLORS[tipo]
    return _PALETTE[idx % len(_PALETTE)]


def _fill_poly_alpha(panel, polygon, color, alpha=0.28):
    if polygon is None or len(polygon) < 3:
        return
    overlay = panel.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0, dst=panel)


class Figura3D:
    """Forma del entorno en 3D.

    Dos orígenes de geometría posibles, no excluyentes pero normalmente uno
    u otro:
      - Wireframe heredado: `puntos`/`conexiones`/`primitivas`, tal como
        salía del fallback LLM (ia_interprete.py, pasos -1..2b). Igual que
        siempre.
      - Malla real: `malla_lod_baja`/`malla_lod_alta` (biblioteca o IA),
        ubicada en `centro` (mundo). Se dibuja con
        render_malla.dibujar_malla(), un solo camino sin importar el
        origen de la malla.

    `estado` ("congelada"/"activa") y `radio_bounding` sostienen el
    descarte grueso de colisión (sección 4.6 del plan): con una malla de
    cientos de caras, comparar CADA arista contra la mano en cada frame no
    escala; salvo que la figura esté "activa" (agarrada, + colchón de
    frames tras soltarla), solo se compara el centro proyectado contra
    `radio_bounding` — ver EntornoVirtual, que es quien decide cuándo
    cambia el estado (necesita la lista de gestos, no vive acá).
    """

    def __init__(self, puntos, conexiones,
                 color_normal=(0,200,200), color_tocado=(0,0,255),
                 primitivas=None, nombre="", propiedades=None,
                 malla_lod_baja=None, malla_lod_alta=None,
                 centro=(0.0, 0.0, 0.0), radio_bounding: float = 0.0):
        self.puntos      = puntos
        self.conexiones  = conexiones
        self.color_normal  = color_normal
        self.color_tocado  = color_tocado
        self.primitivas  = primitivas or []
        self.tocado      = False
        self.nombre      = nombre       # descripción usada para generarla (ej: "silla de madera")
        self.propiedades = propiedades  # dict con ficha física, o None mientras no llegó

        self.malla_lod_baja = malla_lod_baja   # Malla | None
        self.malla_lod_alta = malla_lod_alta   # Malla | None
        self.centro          = list(centro)     # centro de la malla, en mundo
        self.radio_bounding  = radio_bounding   # en unidades de mundo (mismo espacio que self.centro)
        self.estado                = "congelada"
        self._frames_gracia_activa = 0

    def asignar_propiedades(self, propiedades: dict):
        """Recibe la ficha física (paso 2 de objetos.crear_objeto) y la une a la figura
        ya dibujada. Se llama desde el bucle principal al consumir _cola_propiedades."""
        self.propiedades = propiedades

    def actualizar_malla(self, malla_lod_baja, malla_lod_alta, radio_bounding=None):
        """Reemplaza la geometría de una figura ya dibujada por su malla
        real, cuando malla_ia_async termina en background (sección 3, paso
        3 del plan). La figura sigue siendo "la misma" (mismo nombre,
        mismas propiedades, misma posición): solo cambia de qué está
        hecha. Si venía del fallback LLM (con puntos/primitivas propios),
        esos se vacían: la malla real los reemplaza, no coexisten."""
        self.malla_lod_baja = malla_lod_baja
        self.malla_lod_alta = malla_lod_alta
        if radio_bounding is not None:
            self.radio_bounding = radio_bounding
        self.puntos = []
        self.primitivas = []

    def es_malla(self) -> bool:
        return self.malla_lod_baja is not None

    def malla_activa_actual(self):
        """LOD alto si la figura está agarrada (o en el colchón de
        gracia tras soltarla) y ese LOD existe; LOD bajo en cualquier
        otro caso (incluye "todavía no llegó el LOD alto de biblioteca/IA")."""
        if self.estado == "activa" and self.malla_lod_alta is not None:
            return self.malla_lod_alta
        return self.malla_lod_baja

    # ------------------------------------------------------------------
    # Geometría en espacio mundo
    # ------------------------------------------------------------------

    def segmentos_mundo(self):
        for i, j in self.conexiones:
            yield self.puntos[i], self.puntos[j]

    def primitivas_segmentos_mundo(self):
        for prim in self.primitivas:
            cx, cy, cz = prim["centro"]
            for i, j in prim["aristas"]:
                lx,ly,lz = prim["puntos_locales"][i]
                pa = (lx+cx, ly+cy, lz+cz)
                lx,ly,lz = prim["puntos_locales"][j]
                pb = (lx+cx, ly+cy, lz+cz)
                yield pa, pb

    def malla_segmentos_mundo(self):
        """Aristas de la malla activa (LOD alto si está agarrada, si no
        LOD bajo) en espacio mundo. Se usa SOLO en el fine-test de
        colisión de EntornoVirtual, y solo cuando la figura ya está
        "activa" — con la figura "congelada" ni se llama a esto."""
        m = self.malla_activa_actual()
        if m is None:
            return
        cx, cy, cz = self.centro
        for i, j in m.aristas:
            lx, ly, lz = m.vertices[i]
            pa = (lx+cx, ly+cy, lz+cz)
            lx, ly, lz = m.vertices[j]
            pb = (lx+cx, ly+cy, lz+cz)
            yield pa, pb

    def trasladar(self, dx, dy, dz):
        self.puntos = [(x+dx, y+dy, z+dz) for x,y,z in self.puntos]
        for prim in self.primitivas:
            prim["centro"][0] += dx
            prim["centro"][1] += dy
            prim["centro"][2] += dz
        if self.es_malla():
            self.centro[0] += dx
            self.centro[1] += dy
            self.centro[2] += dz

    def profundidad_media(self, proyector):
        if self.es_malla():
            return proyector(tuple(self.centro))[1]
        puntos = self.puntos or [tuple(p["centro"]) for p in self.primitivas]
        if not puntos:
            return 0.0
        return sum(proyector(p)[1] for p in puntos) / len(puntos)

    # ------------------------------------------------------------------
    # Dibujo de primitivas heredadas (sin cambios de fondo respecto a la
    # versión anterior — se conserva SOLO para objetos que ya vinieron del
    # fallback LLM; todo lo nuevo de biblioteca/IA es una Malla y pasa por
    # render_malla.dibujar_malla, no por acá).
    # ------------------------------------------------------------------

    def _puntos_mundo_primitiva(self, prim):
        cx, cy, cz = prim["centro"]
        return [(lx + cx, ly + cy, lz + cz)
                for lx, ly, lz in prim["puntos_locales"]]

    def _color_primitiva(self, prim, idx):
        return _color_para_primitiva(prim, idx)

    def _dibujar_primitiva_heredada(self, panel, prim, proyector, color, alpha=0.28):
        if "puntos_locales" not in prim or "centro" not in prim:
            return
        puntos_mundo = self._puntos_mundo_primitiva(prim)
        proyectados = [proyector(p) for p in puntos_mundo]
        puntos2d = [pt for pt, _ in proyectados]
        z_vals = [z for _, z in proyectados]
        tipo = prim.get("tipo")

        if tipo in ("rectangulo", "circulo", "elipse"):
            polygon = np.array(puntos2d, dtype=np.int32)
            _fill_poly_alpha(panel, polygon, color, alpha)
        elif tipo == "esfera":
            if len(puntos2d) >= 3:
                pts = np.array(puntos2d, dtype=np.int32)
                hull = cv2.convexHull(pts)
                _fill_poly_alpha(panel, hull, color, alpha)
        elif tipo == "cubo":
            faces = [
                (0,1,2,3), (4,5,6,7),
                (0,1,5,4), (2,3,7,6),
                (1,2,6,5), (0,3,7,4),
            ]
            face_polygons = []
            for face in faces:
                poly = np.array([puntos2d[i] for i in face], dtype=np.int32)
                depth = sum(z_vals[i] for i in face) / len(face)
                face_polygons.append((depth, poly))
            for _, poly in sorted(face_polygons, key=lambda item: item[0], reverse=True):
                _fill_poly_alpha(panel, poly, color, alpha)
        elif tipo == "cilindro":
            if len(puntos2d) >= 6:
                n = len(puntos2d) // 2
                top = np.array(puntos2d[:n], dtype=np.int32)
                bottom = np.array(puntos2d[n:], dtype=np.int32)
                side = np.array(puntos2d[:n] + puntos2d[n:][::-1], dtype=np.int32)
                face_polygons = [
                    (sum(z_vals[:n]) / n, top),
                    (sum(z_vals[n:]) / n, bottom),
                    (sum(z_vals) / len(z_vals), side),
                ]
                for _, poly in sorted(face_polygons, key=lambda item: item[0], reverse=True):
                    _fill_poly_alpha(panel, poly, color, alpha)
        else:
            return

        if len(puntos2d) >= 2:
            edge_color = tuple(max(0, int(c * 0.75)) for c in color)
            for i, j in prim["aristas"]:
                pa, _ = proyectados[i]
                pb, _ = proyectados[j]
                cv2.line(panel, pa, pb, edge_color, 2)

    def dibujar(self, panel, proyector):
        color = self.color_tocado if self.tocado else self.color_normal

        if self.es_malla():
            m = self.malla_activa_actual()
            render_malla.dibujar_malla(panel, m, tuple(self.centro), proyector, color)
            return

        for idx, prim in enumerate(self.primitivas):
            prim_color = self._color_primitiva(prim, idx)
            self._dibujar_primitiva_heredada(panel, prim, proyector, prim_color)

        for a, b in self.segmentos_mundo():
            pa, _ = proyector(a)
            pb, _ = proyector(b)
            cv2.line(panel, pa, pb, color, 4)
        for p in self.puntos:
            pp, _ = proyector(p)
            cv2.circle(panel, pp, 5, color, -1)


# ---------------------------------------------------------------------------
# Retrocompatibilidad: Figura (alias 2D para el módulo antiguo)
# ---------------------------------------------------------------------------
class Figura(Figura3D):
    """Alias de Figura3D expuesto para compatibilidad con el código anterior."""
    def __init__(self, puntos, conexiones,
                 color_normal=(0,200,200), color_tocado=(0,0,255),
                 primitivas=None, nombre="", propiedades=None):
        puntos3d = [(p[0], p[1], p[2] if len(p)>2 else 0.0) for p in puntos]
        primitivas3d = []
        for prim in (primitivas or []):
            tipo = prim.get("tipo")
            if tipo == "circulo":
                r = prim["r"] * 400
                m = malla_mod.malla_anillo(r, r)
                primitivas3d.append({"centro":[prim["cx"]*640, prim["cy"]*480, 0.0],
                                     "puntos_locales": m.vertices, "aristas": m.aristas})
            elif tipo == "rectangulo":
                w = prim["ancho"]*640; h = prim["alto"]*480
                m = malla_mod.malla_rectangulo(w, h)
                primitivas3d.append({"centro":[(prim["x"]+prim["ancho"]/2)*640,
                                               (prim["y"]+prim["alto"]/2)*480, 0.0],
                                     "puntos_locales": m.vertices, "aristas": m.aristas})
        super().__init__(puntos3d, conexiones, color_normal, color_tocado,
                         primitivas3d, nombre=nombre, propiedades=propiedades)
