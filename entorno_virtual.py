"""
entorno_virtual.py — Orquestador delgado del entorno 3D: compone `Camara`
(camara.py) + lista de `Figura3D` (figura.py), delega dibujo de mallas a
`render_malla.dibujar_malla()` y delega la creación de geometría local a
`malla.py`. Antes esta clase mezclaba cámara, gestos, colisión y dibujo en
812 líneas; ahora es la capa que las compone (ver PLAN_RECONSTRUCCION_
MALLAS.md, sección 4.7).

API pública sin cambios respecto a la versión anterior (main.py no necesita
tocarse para esto): `proyectar`, `agregar_figura`, `asignar_propiedades_a_
figura`, `actualizar`, `actualizar_gestos`, `dibujar`, atributo `.figuras`.
Se agregan dos métodos nuevos para el flujo de biblioteca/IA:
`agregar_figura_desde_malla` y `actualizar_malla_de_figura`.

---------------------------------------------------------------------------
Convención de coordenadas (sin cambios)
---------------------------------------------------------------------------
Las figuras viven en un espacio "mundo" 3D, en unidades de píxel del panel:
  x ∈ [0, ancho_panel], y ∈ [0, alto_panel]   (igual que en la versión 2D)
  z centrado en 0, con el mismo orden de magnitud que x/y (ver
  escala_profundidad). z=0 es el "plano neutro" de pantalla; z negativo se
  acerca a la cámara, z positivo se aleja.

La IA (fallback) sigue mandando coordenadas relativas en [0,1]; el tercer
valor (z) es OPCIONAL en el formato compacto: si no viene, se asume 0.5.

Las mallas de biblioteca/IA (sección 5 del plan) también viven en espacio
relativo, con la MISMA convención que ya usan las primitivas ("esfera":
r relativo, escalado por min(ancho_panel, alto_panel)): sus vértices y su
`radio_bounding` son fracciones de min(ancho_panel, alto_panel) — una malla
no distingue x/y/z al escalar (a diferencia de un rectángulo con w/h
independientes), así que una escala uniforme es la que preserva la forma.
---------------------------------------------------------------------------
"""

import math

import numpy as np

import malla as malla_mod
from camara import Camara
from figura import Figura3D, Figura, GRACIA_FRAMES_ACTIVA  # noqa: F401 (Figura se re-exporta)

__all__ = ["EntornoVirtual", "Figura3D", "Figura"]


# ---------------------------------------------------------------------------
# EntornoVirtual 3D
# ---------------------------------------------------------------------------

class EntornoVirtual:
    """Entorno físico 3D sobre el panel, renderizado en wireframe/relleno
    plano con proyección de perspectiva propia (OpenCV puro).

    Controles por gesto (ver actualizar_gestos):
      PUÑO CERRADO + MOVIMIENTO
          → arrastrar figura tocada (x, y, z). Además activa el estado
            "activa" de la figura (LOD alto + fine-test de colisión,
            sección 4.6 del plan) mientras dura el arrastre y un colchón
            de frames después de soltarla.
      Gestos de cámara (zoom/dolly, rotar): ver camara.Camara.
    """

    def __init__(self, ancho_panel, alto_panel):
        self.ancho_panel = ancho_panel
        self.alto_panel  = alto_panel
        self.radio_colision = 14
        self.escala_profundidad = min(ancho_panel, alto_panel)

        self.camara = Camara(ancho_panel, alto_panel)

        # Estado de gestos (arrastre de figura — necesita self.figuras, no
        # es puramente cámara, por eso se queda acá y no en Camara).
        self.arrastres = {}   # etiqueta -> {figura, centro_anterior (x,y,z)}

        # Arranca vacío: no hay ninguna figura por defecto. Las figuras las
        # agrega el usuario vía objetos.crear_objeto() -> agregar_figura()
        # (o agregar_figura_desde_malla() para biblioteca/IA).
        self.figuras = []

    # ------------------------------------------------------------------
    # Proyección (delegada a Camara)
    # ------------------------------------------------------------------

    def proyectar(self, punto3d):
        return self.camara.proyectar(punto3d)

    # Acceso directo a los atributos de cámara más consultados desde afuera
    # (motor_estereo y similares), para no romper nada que ya los lea como
    # `entorno.rotacion` / `entorno.distancia_camara` en vez de
    # `entorno.camara.rotacion`.
    @property
    def rotacion(self):
        return self.camara.rotacion

    @property
    def distancia_camara(self):
        return self.camara.distancia_camara

    # ------------------------------------------------------------------
    # Agregar figuras — camino heredado (fallback LLM: puntos/primitivas)
    # ------------------------------------------------------------------

    def agregar_figura(self, puntos_relativos, conexiones,
                       color_normal=(0,200,200), color_tocado=(0,0,255),
                       primitivas_relativas=None, nombre="", propiedades=None):
        """Agrega una figura nueva al entorno a partir de geometría en
        formato heredado (la que sale de ia_interprete.py / ubicacion.py).

        puntos_relativos:
            lista de (x,y) o (x,y,z), valores en [0,1].
            z opcional; si falta se asume 0.5 (plano neutro).

        primitivas_relativas: lista de dicts con coordenadas en [0,1]:
            2D heredadas:
                {"tipo":"circulo",    "cx","cy","r",  ["cz"]}
                {"tipo":"rectangulo", "x","y","ancho","alto", ["cz"]}
                {"tipo":"elipse",     "cx","cy","rx","ry",    ["cz"]}
            3D:
                {"tipo":"esfera",    "cx","cy","cz","r"}
                {"tipo":"cubo",      "cx","cy","cz","ancho","alto","profundo"}
                {"tipo":"cilindro",  "cx","cy","cz","r","alto"}

        nombre: descripción del objeto (se usa para encontrarlo después,
            tanto para asignar propiedades como para reemplazar su
            geometría por una malla real cuando llega de IA en background).

        propiedades: ficha física inicial (opcional).
        """
        puntos_mundo = [self._punto_a_mundo(p) for p in puntos_relativos]
        primitivas_mundo = [
            pm for pm in (self._primitiva_a_mundo(p)
                          for p in (primitivas_relativas or []))
            if pm is not None
        ]
        figura = Figura3D(puntos_mundo, conexiones, color_normal, color_tocado,
                          primitivas_mundo, nombre=nombre, propiedades=propiedades)
        self.figuras.append(figura)
        return figura

    # ------------------------------------------------------------------
    # Agregar figuras — camino nuevo (biblioteca/IA: Malla real)
    # ------------------------------------------------------------------

    def agregar_figura_desde_malla(self, centro_relativo, malla_lod_baja,
                                    malla_lod_alta=None, nombre="", propiedades=None,
                                    color_normal=(0,200,200), color_tocado=(0,0,255),
                                    radio_bounding_relativo=None):
        """Agrega una figura nueva al entorno a partir de una Malla real
        (biblioteca_mallas.buscar() o malla_ia_async, ya decimadas por
        optimizacion_malla). `centro_relativo` es (x,y[,z]) en [0,1], igual
        que `puntos_relativos` en agregar_figura -- normalmente ya viene
        resuelto por ubicacion.ubicar_y_registrar() usando un bbox
        placeholder tipo "esfera" (ver objetos.py, sección 4.3 del plan).
        """
        centro_mundo = self._punto_a_mundo(centro_relativo)
        escala = min(self.ancho_panel, self.alto_panel)
        malla_baja_mundo = _malla_escalada(malla_lod_baja, escala)
        malla_alta_mundo = _malla_escalada(malla_lod_alta, escala) if malla_lod_alta else None
        radio_rel = (radio_bounding_relativo if radio_bounding_relativo is not None
                     else malla_lod_baja.radio_bounding())

        figura = Figura3D([], [], color_normal, color_tocado, primitivas=[],
                          nombre=nombre, propiedades=propiedades,
                          malla_lod_baja=malla_baja_mundo, malla_lod_alta=malla_alta_mundo,
                          centro=centro_mundo, radio_bounding=radio_rel * escala)
        self.figuras.append(figura)
        return figura

    def actualizar_malla_de_figura(self, nombre: str, malla_lod_baja,
                                    malla_lod_alta=None,
                                    radio_bounding_relativo=None) -> bool:
        """Busca la figura `nombre` (por ejemplo la que dibujó el fallback
        LLM mientras la generación IA todavía corría en background) y
        reemplaza su geometría por la malla real ya terminada -- sección
        3, paso 3 del plan: 'reemplaza la figura primitiva por la malla
        real'. Devuelve True si encontró la figura."""
        for figura in self.figuras:
            if figura.nombre == nombre:
                escala = min(self.ancho_panel, self.alto_panel)
                malla_baja_mundo = _malla_escalada(malla_lod_baja, escala)
                malla_alta_mundo = _malla_escalada(malla_lod_alta, escala) if malla_lod_alta else None
                radio = (radio_bounding_relativo * escala if radio_bounding_relativo is not None
                         else figura.radio_bounding)
                figura.actualizar_malla(malla_baja_mundo, malla_alta_mundo, radio)
                print(f"[entorno] Malla real de IA asignada a '{nombre}' (reemplaza la geometría del fallback).")
                return True
        print(f"[entorno] No se encontró figura con nombre '{nombre}' para asignar la malla.")
        return False

    def asignar_propiedades_a_figura(self, nombre: str, propiedades: dict) -> bool:
        """Busca la primera figura del entorno cuyo `nombre` coincida y le asigna
        las propiedades físicas. Devuelve True si encontró la figura, False si no.
        Se llama desde el bucle principal al consumir _cola_propiedades."""
        for figura in self.figuras:
            if figura.nombre == nombre:
                figura.asignar_propiedades(propiedades)
                print(f"[entorno] Propiedades asignadas a '{nombre}'.")
                return True
        print(f"[entorno] No se encontró figura con nombre '{nombre}'.")
        return False

    def _punto_a_mundo(self, p):
        x, y = p[0], p[1]
        z = p[2] if len(p) > 2 else 0.5
        return (
            x * self.ancho_panel,
            y * self.alto_panel,
            (z - 0.5) * self.escala_profundidad,
        )

    def _primitiva_a_mundo(self, prim):
        tipo = prim.get("tipo")
        cz_rel = prim.get("cz", 0.5)
        cz = (cz_rel - 0.5) * self.escala_profundidad

        if tipo == "circulo":
            r = prim["r"] * min(self.ancho_panel, self.alto_panel)
            m = malla_mod.malla_anillo(r, r)
            cx = prim["cx"] * self.ancho_panel
            cy = prim["cy"] * self.alto_panel

        elif tipo == "elipse":
            rx = prim["rx"] * self.ancho_panel
            ry = prim["ry"] * self.alto_panel
            m = malla_mod.malla_anillo(rx, ry)
            cx = prim["cx"] * self.ancho_panel
            cy = prim["cy"] * self.alto_panel

        elif tipo == "rectangulo":
            w = prim["ancho"] * self.ancho_panel
            h = prim["alto"]  * self.alto_panel
            m = malla_mod.malla_rectangulo(w, h)
            cx = (prim["x"] + prim["ancho"]/2.0) * self.ancho_panel
            cy = (prim["y"] + prim["alto"] /2.0) * self.alto_panel

        elif tipo == "esfera":
            r = prim["r"] * min(self.ancho_panel, self.alto_panel)
            m = malla_mod.malla_esfera(r)
            cx = prim["cx"] * self.ancho_panel
            cy = prim["cy"] * self.alto_panel

        elif tipo == "cubo":
            w = prim["ancho"] * self.ancho_panel
            h = prim["alto"]  * self.alto_panel
            d = prim.get("profundo", prim["ancho"]) * self.escala_profundidad
            m = malla_mod.malla_cubo(w, h, d)
            cx = prim["cx"] * self.ancho_panel
            cy = prim["cy"] * self.alto_panel

        elif tipo == "cilindro":
            r = prim["r"]    * min(self.ancho_panel, self.alto_panel)
            h = prim["alto"] * self.alto_panel
            m = malla_mod.malla_cilindro(r, h)
            cx = prim["cx"] * self.ancho_panel
            cy = prim["cy"] * self.alto_panel

        else:
            return None

        prim_mundo = dict(prim)
        prim_mundo["centro"] = [cx, cy, cz]
        prim_mundo["puntos_locales"] = m.vertices
        prim_mundo["aristas"] = m.aristas
        return prim_mundo

    # ------------------------------------------------------------------
    # Colisión (en espacio de pantalla)
    # ------------------------------------------------------------------
    # Descarte grueso vs. fine-test (sección 4.6 del plan): una figura
    # heredada (puntos/primitivas, típicamente 6-12 aristas) sigue
    # comparándose arista por arista como siempre, no hay necesidad de
    # optimizar eso. Una figura con Malla (cientos de caras potenciales)
    # solo hace el fine-test arista-por-arista cuando está "activa"; si
    # está "congelada" se compara únicamente el centro proyectado contra
    # `radio_bounding` -- un solo chequeo, sin iterar ni una arista.
    # ------------------------------------------------------------------

    @staticmethod
    def _dist_punto_seg_2d(p, a, b):
        px, py = p;  ax, ay = a;  bx, by = b
        abx, aby = bx-ax, by-ay
        apx, apy = px-ax, py-ay
        L2 = abx*abx + aby*aby
        if L2 == 0:
            return math.hypot(apx, apy)
        t = max(0.0, min(1.0, (apx*abx + apy*aby) / L2))
        return math.hypot(px-(ax+t*abx), py-(ay+t*aby))

    def _radio_pantalla(self, figura):
        """Radio aproximado en píxeles de pantalla del `radio_bounding` de
        `figura` (mundo), para el descarte grueso: proyecta el centro y un
        punto desplazado `radio_bounding` sobre el eje X, y mide la
        distancia entre las dos proyecciones."""
        cx, cy, cz = figura.centro
        p0, _ = self.proyectar((cx, cy, cz))
        p1, _ = self.proyectar((cx + figura.radio_bounding, cy, cz))
        return math.hypot(p1[0]-p0[0], p1[1]-p0[1])

    def _tocada_por_puntos(self, figura, puntos_pantalla) -> bool:
        """True si algún punto de mano (2D pantalla) toca `figura`, con el
        camino grueso o fino según corresponda."""
        if figura.es_malla() and figura.estado != "activa":
            centro2d, _ = self.proyectar(tuple(figura.centro))
            radio_px = max(self._radio_pantalla(figura), self.radio_colision)
            for p in puntos_pantalla:
                if math.hypot(p[0]-centro2d[0], p[1]-centro2d[1]) <= radio_px:
                    return True
            return False

        segs = self._segmentos_proy(figura)
        for p in puntos_pantalla:
            for a, b in segs:
                if self._dist_punto_seg_2d(p, a, b) <= self.radio_colision:
                    return True
        return False

    def _segmentos_proy(self, figura):
        """Todas las aristas "finas" de la figura proyectadas a 2D con la
        cámara actual. Para una figura con Malla, esto SOLO se llama
        cuando ya se decidió hacer el fine-test (estado "activa") --
        _tocada_por_puntos filtra antes con el descarte grueso."""
        seg = []
        for a, b in figura.segmentos_mundo():
            pa, _ = self.proyectar(a)
            pb, _ = self.proyectar(b)
            seg.append((pa, pb))
        for a, b in figura.primitivas_segmentos_mundo():
            pa, _ = self.proyectar(a)
            pb, _ = self.proyectar(b)
            seg.append((pa, pb))
        if figura.es_malla():
            for a, b in figura.malla_segmentos_mundo():
                pa, _ = self.proyectar(a)
                pb, _ = self.proyectar(b)
                seg.append((pa, pb))
        return seg

    def actualizar(self, puntos_mano):
        """Detecta qué figuras están tocadas por los puntos 2D de pantalla de la mano."""
        for figura in self.figuras:
            figura.tocado = self._tocada_por_puntos(figura, puntos_mano)

    def _figura_tocada(self, puntos_pantalla):
        for figura in self.figuras:
            if self._tocada_por_puntos(figura, puntos_pantalla):
                return figura
        return None

    # ------------------------------------------------------------------
    # Estado congelada/activa (sección 4.6 del plan)
    # ------------------------------------------------------------------

    def _actualizar_estados_malla(self):
        """Se llama una vez por frame: la figura agarrada ahora mismo (está
        en self.arrastres) queda "activa" con el colchón completo de
        gracia; el resto de las figuras con Malla decrementan su colchón y
        vuelven a "congelada" cuando se agota. No itera ninguna arista acá
        -- es aritmética de enteros, no colisión."""
        figuras_agarradas = {arr["figura"] for arr in self.arrastres.values()}
        for figura in self.figuras:
            if not figura.es_malla():
                continue
            if figura in figuras_agarradas:
                figura.estado = "activa"
                figura._frames_gracia_activa = GRACIA_FRAMES_ACTIVA
            elif figura._frames_gracia_activa > 0:
                figura._frames_gracia_activa -= 1
                figura.estado = "activa"
            else:
                figura.estado = "congelada"

    # ------------------------------------------------------------------
    # Gestos
    # ------------------------------------------------------------------

    def actualizar_gestos(self, manos):
        """Aplica los gestos de mano sobre la cámara y las figuras del entorno.

        `manos`: lista de dicts, uno por mano detectada en el frame actual:
            "etiqueta"    : "Izquierda"/"Derecha" (estable entre frames).
            "puntos"      : puntos 2D de pantalla [(x,y), ...].
            "centro"      : (x, y) del centro de palma en pantalla.
            "profundidad" : float — z de MediaPipe del centro (diferencias relativas).
            "estados"     : [Pulgar, Índice, Medio, Anular, Meñique] booleanos.
            "cerrada"     : True si los 5 dedos están cerrados (puño).
        """
        etiquetas_presentes = {m["etiqueta"] for m in manos}

        # ---- Cámara (zoom/dolly + rotar): delegado a Camara ----
        self.camara.actualizar_gestos_camara(manos)

        # ---- PUÑO CERRADO → arrastrar figura (x, y, z) ----
        for mano in manos:
            etiqueta = mano["etiqueta"]
            if not mano["cerrada"]:
                self.arrastres.pop(etiqueta, None)
                continue

            cx, cy = mano["centro"]
            cz = mano.get("profundidad", 0.0)
            arrastre = self.arrastres.get(etiqueta)

            if arrastre is None:
                figura = self._figura_tocada(mano["puntos"])
                if figura is not None:
                    self.arrastres[etiqueta] = {
                        "figura": figura,
                        "centro_anterior": (cx, cy, cz),
                    }
            else:
                cx0, cy0, cz0 = arrastre["centro_anterior"]
                dx, dy = cx-cx0, cy-cy0
                dz = (cz-cz0) * ESCALA_ARRASTRE_Z * self.ancho_panel
                if dx or dy or dz:
                    arrastre["figura"].trasladar(dx, dy, dz)
                arrastre["centro_anterior"] = (cx, cy, cz)

        for etiqueta in list(self.arrastres.keys()):
            if etiqueta not in etiquetas_presentes:
                del self.arrastres[etiqueta]

        self._actualizar_estados_malla()

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------

    def dibujar(self, panel):
        # Pintor's algorithm: más lejanas primero
        for figura in sorted(self.figuras,
                             key=lambda f: -f.profundidad_media(self.proyectar)):
            figura.dibujar(panel, self.proyectar)
        self.camara.dibujar_brujula(panel)


ESCALA_ARRASTRE_Z = 1.0    # multiplica delta de profundidad de mano al arrastrar


def _malla_escalada(malla_obj, escala: float):
    """Escala uniformemente los vértices de `malla_obj` (relativos, ver
    docstring de nivel de módulo) por `escala` = min(ancho_panel,
    alto_panel), para pasarlos a espacio mundo. `radio_bounding` se escala
    aparte, por quien llama, con el mismo factor."""
    vertices_mundo = [(x*escala, y*escala, z*escala) for x, y, z in malla_obj.vertices]
    return malla_mod.Malla(vertices=vertices_mundo, caras=malla_obj.caras, aristas=malla_obj.aristas)