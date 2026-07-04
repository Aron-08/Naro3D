"""
render_malla.py — Camino único de dibujo para cualquier Malla: cubo,
esfera, cilindro, malla de biblioteca o malla de IA son todas "una Malla"
a esta altura, no hay casos especiales por tipo. Reemplaza el if/elif de
6 ramas que tenía `Figura3D._dibujar_primitiva` en la versión anterior.

Ver PLAN_RECONSTRUCCION_MALLAS.md, sección 4.5.
"""

from __future__ import annotations

import cv2
import numpy as np


def _fill_poly_alpha(panel, polygon, color, alpha=0.28):
    if polygon is None or len(polygon) < 3:
        return
    overlay = panel.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0, dst=panel)


def dibujar_malla(panel, malla, centro, proyector, color, alpha=0.28, grosor_arista=1):
    """Dibuja `malla` (vértices locales, sin transformar) ubicada en
    `centro` (x,y,z mundo), con `proyector` (mismo (x,y,z) -> ((sx,sy),
    z_cam) que usa Camara/EntornoVirtual para todo lo demás).

    Painter's algorithm: se proyectan todos los vértices una sola vez, se
    ordenan las CARAS por profundidad media (más lejos primero) y se van
    rellenando en ese orden — mismo criterio que antes se aplicaba por
    FIGURA completa en EntornoVirtual.dibujar, ahora a nivel de cara
    individual (necesario para que una malla cóncava se vea bien, no solo
    primitivas convexas como antes).
    """
    if not malla.vertices:
        return

    cx, cy, cz = centro
    proyectados = [proyector((lx + cx, ly + cy, lz + cz)) for lx, ly, lz in malla.vertices]
    puntos2d = [pt for pt, _ in proyectados]
    z_vals = [z for _, z in proyectados]

    if malla.caras:
        caras_ordenadas = sorted(
            malla.caras,
            key=lambda cara: sum(z_vals[i] for i in cara) / len(cara),
            reverse=True,
        )
        for cara in caras_ordenadas:
            poly = np.array([puntos2d[i] for i in cara], dtype=np.int32)
            _fill_poly_alpha(panel, poly, color, alpha)

    if malla.aristas:
        edge_color = tuple(max(0, int(c * 0.75)) for c in color)
        for i, j in malla.aristas:
            cv2.line(panel, puntos2d[i], puntos2d[j], edge_color, grosor_arista)
