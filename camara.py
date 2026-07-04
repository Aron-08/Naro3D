"""
camara.py — Cámara virtual: proyección de perspectiva + gestos de cámara
(zoom/dolly de dos manos, rotación continua de una mano) + brújula.

Extraído de EntornoVirtual, que antes mezclaba cámara, gestos, colisión y
dibujo en una sola clase de 812 líneas. El arrastre de figura (puño
cerrado) se queda en EntornoVirtual porque necesita la lista de figuras,
no es puramente cámara. Ver PLAN_RECONSTRUCCION_MALLAS.md, sección 4.7.

Controles (sin cambios de comportamiento respecto a la versión anterior):
  DOS MANOS con pulgar+índice+medio abiertos (anular y meñique cerrados)
      separarlas  → avanzar cámara (dolly in).
      juntarlas   → retroceder cámara (dolly out).
  UNA MANO con índice+medio abiertos (pulgar, anular y meñique cerrados)
      mover la muñeca de arriba a abajo      → pitch (eje X).
      mover la muñeca de izquierda a derecha → yaw (eje Y).
      El giro es continuo, grado a grado, proporcional a cuánto se movió
      la muñeca en ese frame (ver ESCALA_ROTAR).
"""

from __future__ import annotations

import math
import cv2
import numpy as np


def _matriz_rotacion_eje_angulo(eje, angulo):
    """Rodrigues: matriz 3x3 que gira `angulo` rad alrededor del vector unitario `eje`."""
    x, y, z = eje
    c, s = math.cos(angulo), math.sin(angulo)
    C = 1 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Sensibilidad de gestos de cámara (mismos valores que la versión anterior
# de entorno_virtual.py — sin cambios de comportamiento).
# ---------------------------------------------------------------------------
UMBRAL_DOLLY = 1.5    # px/frame separación mínima entre manos para dolly
ESCALA_DOLLY = 3.0    # cuánto cambia distancia_camara por px de separación

UMBRAL_ROTAR_90_PX        = 70.0
ESCALA_ROTAR              = (math.pi / 2.0) / UMBRAL_ROTAR_90_PX   # rad por píxel
ANGULO_MAX_ROTACION_FRAME = 0.25   # rad (~14°) — tope de giro por frame

# Colchón de gracia (ver comentario original en entorno_virtual.py): el
# pulgar es la señal más ruidosa de dedos_extendidos() y girar la muñeca
# -para rotar- es justo lo que produce posturas oblicuas que lo confunden.
# Sin este colchón, un solo frame mal clasificado corta el gesto y se
# pierde la posición anterior.
GRACIA_FRAMES_ROTAR = 6


def _es_gesto_zoom(estados):
    """Pulgar, índice y medio extendidos; anular y meñique cerrados."""
    if not estados or len(estados) < 5:
        return False
    pulgar, indice, medio, anular, menique = estados[:5]
    return pulgar and indice and medio and not anular and not menique


def _es_gesto_rotar(estados):
    """Índice y medio extendidos; pulgar, anular y meñique cerrados ("tijera")."""
    if not estados or len(estados) < 5:
        return False
    pulgar, indice, medio, anular, menique = estados[:5]
    return (not pulgar) and indice and medio and not anular and not menique


class Camara:
    """Proyección de perspectiva + gestos de zoom/rotar sobre un panel de
    `ancho_panel` x `alto_panel` píxeles. Misma convención de coordenadas
    que el resto del proyecto: world space en píxeles de panel, z centrado
    en 0 (ver docstring de nivel de módulo en entorno_virtual.py)."""

    def __init__(self, ancho_panel, alto_panel):
        self.ancho_panel = ancho_panel
        self.alto_panel  = alto_panel

        self.rotacion         = np.eye(3, dtype=np.float64)
        self.distancia_camara = 950.0
        self.distancia_min    = 350.0
        self.distancia_max    = 2600.0
        self.focal             = 620.0
        self.centro_x          = ancho_panel / 2.0
        self.centro_y          = alto_panel  / 2.0

        self._rotar_anterior = {}   # etiqueta -> (cx,cy) del frame anterior con gesto activo
        self._rotar_gracia   = {}   # etiqueta -> frames de tolerancia restantes
        self._dolly_anterior = {}   # etiqueta -> (cx,cy) anterior

    # ------------------------------------------------------------------
    # Proyección
    # ------------------------------------------------------------------

    def proyectar(self, punto3d):
        """(x,y,z) mundo -> ((sx,sy) pantalla, z_cam). Aplica rotación
        global (pivoteando en el centro del panel) y proyección de
        perspectiva."""
        x, y, z = punto3d
        v = self.rotacion @ np.array([x - self.centro_x, y - self.centro_y, z])
        z_cam = max(v[2] + self.distancia_camara, 1.0)
        factor = self.focal / z_cam
        sx = int(v[0]*factor + self.centro_x)
        sy = int(v[1]*factor + self.centro_y)
        return (sx, sy), float(z_cam)

    # ------------------------------------------------------------------
    # Gestos de cámara (zoom/dolly + rotar)
    # ------------------------------------------------------------------

    def actualizar_gestos_camara(self, manos) -> bool:
        """Aplica dolly y rotación sobre la cámara según `manos` (mismo
        formato que EntornoVirtual.actualizar_gestos). Devuelve True si se
        aplicó dolly este frame, para que EntornoVirtual no interprete
        además un gesto de rotar simultáneo con esas mismas manos."""
        manos_zoom = {m["etiqueta"]: m for m in manos
                      if _es_gesto_zoom(m.get("estados")) and m["etiqueta"] in ("Izquierda", "Derecha")}

        dolly_aplicado = False
        if len(manos_zoom) == 2:
            izq = manos_zoom["Izquierda"]
            der = manos_zoom["Derecha"]
            prev = self._dolly_anterior
            if "Izquierda" in prev and "Derecha" in prev:
                dx_izq = izq["centro"][0] - prev["Izquierda"][0]
                dx_der = der["centro"][0] - prev["Derecha"][0]
                separacion = (dx_der - dx_izq) / 2.0
                if abs(separacion) > UMBRAL_DOLLY:
                    nueva = self.distancia_camara - separacion * ESCALA_DOLLY
                    self.distancia_camara = max(self.distancia_min,
                                                min(self.distancia_max, nueva))
                    dolly_aplicado = True
            self._dolly_anterior["Izquierda"] = izq["centro"]
            self._dolly_anterior["Derecha"]   = der["centro"]
        else:
            self._dolly_anterior.clear()

        manos_rotar = {m["etiqueta"]: m for m in manos if _es_gesto_rotar(m.get("estados"))}
        etiqueta_activa = None
        if len(manos_rotar) == 1 and not dolly_aplicado:
            etiqueta_activa = next(iter(manos_rotar))

        for etiqueta in list(self._rotar_anterior.keys()):
            if etiqueta == etiqueta_activa:
                continue
            self._rotar_gracia[etiqueta] = self._rotar_gracia.get(etiqueta, GRACIA_FRAMES_ROTAR) - 1
            if self._rotar_gracia[etiqueta] <= 0:
                self._rotar_anterior.pop(etiqueta, None)
                self._rotar_gracia.pop(etiqueta, None)

        if etiqueta_activa is not None:
            mano = manos_rotar[etiqueta_activa]
            cx, cy = mano["centro"]
            self._rotar_gracia[etiqueta_activa] = GRACIA_FRAMES_ROTAR
            anterior = self._rotar_anterior.get(etiqueta_activa)

            if anterior is None:
                self._rotar_anterior[etiqueta_activa] = (cx, cy)
            else:
                px, py = anterior
                dx, dy = cx - px, cy - py
                if dx or dy:
                    if abs(dy) >= abs(dx):
                        eje, delta = (1.0, 0.0, 0.0), dy
                    else:
                        eje, delta = (0.0, 1.0, 0.0), dx
                    angulo = -delta * ESCALA_ROTAR
                    angulo = max(-ANGULO_MAX_ROTACION_FRAME, min(ANGULO_MAX_ROTACION_FRAME, angulo))
                    R = _matriz_rotacion_eje_angulo(eje, angulo)
                    self.rotacion = R @ self.rotacion
                self._rotar_anterior[etiqueta_activa] = (cx, cy)

        return dolly_aplicado

    # ------------------------------------------------------------------
    # Brújula
    # ------------------------------------------------------------------

    def dibujar_brujula(self, panel):
        """Brújula 3D en la esquina inferior izquierda: ejes X(rojo), Y(verde), Z(naranja)."""
        ox, oy = 45, self.alto_panel - 45
        largo = 28
        for eje, color, nombre in (
            (np.array([largo, 0.0, 0.0]), (60, 60, 255),   "X"),
            (np.array([0.0, largo, 0.0]), (60, 255, 60),   "Y"),
            (np.array([0.0, 0.0, largo]), (60, 180, 255),  "Z"),
        ):
            v = self.rotacion @ eje
            punta = (int(ox + v[0]), int(oy + v[1]))
            cv2.line(panel, (ox, oy), punta, color, 2)
            cv2.putText(panel, nombre, (punta[0]+3, punta[1]+3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.circle(panel, (ox, oy), 3, (220, 220, 220), -1)
