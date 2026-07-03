import math
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Convención de coordenadas
# ---------------------------------------------------------------------------
# Las figuras viven en un espacio "mundo" 3D, en unidades de píxel del panel:
#   x ∈ [0, ancho_panel], y ∈ [0, alto_panel]   (igual que en la versión 2D)
#   z centrado en 0, con el mismo orden de magnitud que x/y (ver escala_profundidad
#   en EntornoVirtual). z=0 es el "plano neutro" de pantalla; z negativo se acerca
#   a la cámara, z positivo se aleja.
#
# La IA sigue mandando coordenadas relativas en [0,1] (igual que antes); el tercer
# valor (z) es OPCIONAL en el formato compacto: si no viene, se asume 0.5, que cae
# justo en el plano neutro y reproduce el comportamiento 2D anterior sin cambios.
# ---------------------------------------------------------------------------


def _normalizar_vec(v):
    n = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if n < 1e-9:
        return (0.0, 0.0, 1.0)
    return (v[0]/n, v[1]/n, v[2]/n)


def _cruz(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def _punto(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


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
# Mallas locales de primitivas 3D (centradas en el origen, sin transformar)
# Cada función devuelve (puntos_locales, aristas) — wireframe puro.
# ---------------------------------------------------------------------------

def _malla_anillo(rx, ry, segmentos=16):
    """Círculo/elipse aplanado en el plano XY (C: y E:)."""
    puntos = [
        (rx * math.cos(2*math.pi*i/segmentos),
         ry * math.sin(2*math.pi*i/segmentos), 0.0)
        for i in range(segmentos)
    ]
    aristas = [(i, (i+1) % segmentos) for i in range(segmentos)]
    return puntos, aristas


def _malla_rectangulo(w, h):
    hw, hh = w/2.0, h/2.0
    puntos = [(-hw,-hh,0.0),(hw,-hh,0.0),(hw,hh,0.0),(-hw,hh,0.0)]
    aristas = [(0,1),(1,2),(2,3),(3,0)]
    return puntos, aristas


def _malla_esfera(r, meridianos=10, paralelos=6):
    puntos, anillos = [], []
    for pi in range(1, paralelos):
        theta = math.pi * pi / paralelos
        y_lat = r * math.cos(theta)
        radio = r * math.sin(theta)
        anillo = []
        for mi in range(meridianos):
            phi = 2*math.pi*mi/meridianos
            anillo.append(len(puntos))
            puntos.append((radio*math.cos(phi), y_lat, radio*math.sin(phi)))
        anillos.append(anillo)
    polo_n = len(puntos); puntos.append((0.0, -r, 0.0))
    polo_s = len(puntos); puntos.append((0.0,  r, 0.0))
    aristas = []
    for anillo in anillos:
        n = len(anillo)
        aristas += [(anillo[i], anillo[(i+1)%n]) for i in range(n)]
    for a, b in zip(anillos, anillos[1:]):
        aristas += [(a[i], b[i]) for i in range(len(a))]
    aristas += [(polo_n, i) for i in anillos[0]]
    aristas += [(polo_s, i) for i in anillos[-1]]
    return puntos, aristas


def _malla_cubo(w, h, d):
    hw, hh, hd = w/2.0, h/2.0, d/2.0
    puntos = [
        (-hw,-hh,-hd),(hw,-hh,-hd),(hw,hh,-hd),(-hw,hh,-hd),
        (-hw,-hh, hd),(hw,-hh, hd),(hw,hh, hd),(-hw,hh, hd),
    ]
    aristas = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]
    return puntos, aristas


def _malla_cilindro(r, alto, segmentos=14):
    h2 = alto/2.0
    puntos, sup, inf = [], [], []
    for i in range(segmentos):
        ang = 2*math.pi*i/segmentos
        x, z = r*math.cos(ang), r*math.sin(ang)
        sup.append(len(puntos)); puntos.append((x,-h2,z))
    for i in range(segmentos):
        ang = 2*math.pi*i/segmentos
        x, z = r*math.cos(ang), r*math.sin(ang)
        inf.append(len(puntos)); puntos.append((x,h2,z))
    aristas = []
    for i in range(segmentos):
        j = (i+1)%segmentos
        aristas += [(sup[i],sup[j]),(inf[i],inf[j]),(sup[i],inf[i])]
    return puntos, aristas


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
                r = int(valor[1:3], 16)
                g = int(valor[3:5], 16)
                b = int(valor[5:7], 16)
                return (b, g, r)
            except ValueError:
                return None
        if valor.startswith("0x") and len(valor) == 8:
            try:
                r = int(valor[2:4], 16)
                g = int(valor[4:6], 16)
                b = int(valor[6:8], 16)
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


# ---------------------------------------------------------------------------
# Figura3D
# ---------------------------------------------------------------------------

class Figura3D:
    """Forma del entorno en 3D: esqueleto de puntos+líneas más primitivas wireframe.
    Cambia de color cuando la mano la toca (evaluado en proyección de pantalla).

    Cada figura lleva su `nombre` (la descripción que se usó para generarla) y sus
    `propiedades` físicas (material, peso, resistencias, etc.) como parte del mismo
    objeto.  Las propiedades se asignan en dos tiempos: primero se dibuja (con
    `propiedades=None`), y cuando llegan desde `objetos.crear_objeto` se completan
    con `asignar_propiedades()` — sin necesidad de redibujar nada.
    """

    def __init__(self, puntos, conexiones,
                 color_normal=(0,200,200), color_tocado=(0,0,255),
                 primitivas=None, nombre="", propiedades=None):
        self.puntos      = puntos
        self.conexiones  = conexiones
        self.color_normal  = color_normal
        self.color_tocado  = color_tocado
        self.primitivas  = primitivas or []
        self.tocado      = False
        self.nombre      = nombre       # descripción usada para generarla (ej: "silla de madera")
        self.propiedades = propiedades  # dict con ficha física, o None mientras no llegó

    def asignar_propiedades(self, propiedades: dict):
        """Recibe la ficha física (paso 2 de objetos.crear_objeto) y la une a la figura
        ya dibujada. Se llama desde el bucle principal al consumir _cola_propiedades."""
        self.propiedades = propiedades

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

    def trasladar(self, dx, dy, dz):
        self.puntos = [(x+dx, y+dy, z+dz) for x,y,z in self.puntos]
        for prim in self.primitivas:
            prim["centro"][0] += dx
            prim["centro"][1] += dy
            prim["centro"][2] += dz

    def profundidad_media(self, proyector):
        puntos = self.puntos or [tuple(p["centro"]) for p in self.primitivas]
        if not puntos:
            return 0.0
        return sum(proyector(p)[1] for p in puntos) / len(puntos)

    def _puntos_mundo_primitiva(self, prim):
        cx, cy, cz = prim["centro"]
        return [(lx + cx, ly + cy, lz + cz)
                for lx, ly, lz in prim["puntos_locales"]]

    def _color_primitiva(self, prim, idx):
        return _color_para_primitiva(prim, idx)

    def _dibujar_primitiva(self, panel, prim, proyector, color, alpha=0.28):
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
        for idx, prim in enumerate(self.primitivas):
            prim_color = self._color_primitiva(prim, idx)
            self._dibujar_primitiva(panel, prim, proyector, prim_color)

        color = self.color_tocado if self.tocado else self.color_normal
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
# Si alguna parte del código importa "Figura" (en vez de "Figura3D"), esto
# evita un ImportError sin cambiar comportamiento visible: Figura3D acepta
# puntos 2D (los eleva a z=0 internamente si ya son tuplas de 2).
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
                locales, aristas = _malla_anillo(r, r)
                primitivas3d.append({"centro":[prim["cx"]*640, prim["cy"]*480, 0.0],
                                     "puntos_locales": locales, "aristas": aristas})
            elif tipo == "rectangulo":
                w = prim["ancho"]*640; h = prim["alto"]*480
                locales, aristas = _malla_rectangulo(w, h)
                primitivas3d.append({"centro":[(prim["x"]+prim["ancho"]/2)*640,
                                               (prim["y"]+prim["alto"]/2)*480, 0.0],
                                     "puntos_locales": locales, "aristas": aristas})
        super().__init__(puntos3d, conexiones, color_normal, color_tocado,
                         primitivas3d, nombre=nombre, propiedades=propiedades)


# ---------------------------------------------------------------------------
# Sensibilidad de gestos de cámara
# ---------------------------------------------------------------------------
UMBRAL_DOLLY        = 1.5    # px/frame separación mínima entre manos para dolly
ESCALA_DOLLY        = 3.0    # cuánto cambia distancia_camara por px de separación
ESCALA_ARRASTRE_Z   = 1.0    # multiplica delta de profundidad de mano al arrastrar

# Rotación continua por gesto (ver _es_gesto_rotar): cada frame, el gesto
# rota el entorno en proporción a cuánto se movió la muñeca respecto al
# frame anterior (no es un salto de 90°/180°, es grado a grado a medida que
# la mano se mueve). ESCALA_ROTAR está calibrada para que desplazar la
# muñeca UMBRAL_ROTAR_90_PX píxeles de panel equivalga a 90° acumulados
# ("mano vertical a horizontal"); el doble de eso equivale a 180°
# ("vertical a vertical opuesta"). ANGULO_MAX_ROTACION_FRAME limita el
# giro de un solo frame para que un salto de posición por ruido de
# MediaPipe (mano perdida y reencontrada, etc.) no gire de golpe.
UMBRAL_ROTAR_90_PX         = 70.0
ESCALA_ROTAR               = (math.pi / 2.0) / UMBRAL_ROTAR_90_PX   # rad por píxel
ANGULO_MAX_ROTACION_FRAME  = 0.25   # rad (~14°) — tope de giro por frame

# El pulgar es la señal más ruidosa de dedos_extendidos() (ver _pulgar_extendido
# en main.py: usa 3 criterios justamente porque falla en "posturas oblicuas") y
# girar la muñeca -para hacer el gesto de rotar- es EXACTAMENTE lo que produce
# esas posturas oblicuas. Sin colchón, un solo frame mal clasificado (pulgar
# leído como abierto) corta el gesto y se pierde la posición anterior, así
# que el próximo frame bueno arranca de cero en vez de seguir acumulando.
# Este número de frames de tolerancia evita eso: mientras el gesto se haya
# visto hace poco, un frame aislado sin el patrón exacto no lo cancela.
GRACIA_FRAMES_ROTAR = 6


def _es_gesto_zoom(estados):
    """Gesto de zoom: pulgar, índice y medio extendidos; anular y meñique
    cerrados. `estados` es la lista de 5 booleanos [Pulgar, Índice, Medio,
    Anular, Meñique] que devuelve dedos_extendidos() en main.py."""
    if not estados or len(estados) < 5:
        return False
    pulgar, indice, medio, anular, menique = estados[:5]
    return pulgar and indice and medio and not anular and not menique


def _es_gesto_rotar(estados):
    """Gesto de rotación: índice y medio extendidos; pulgar, anular y
    meñique cerrados (como una "tijera")."""
    if not estados or len(estados) < 5:
        return False
    pulgar, indice, medio, anular, menique = estados[:5]
    return (not pulgar) and indice and medio and not anular and not menique


# ---------------------------------------------------------------------------
# EntornoVirtual 3D
# ---------------------------------------------------------------------------

class EntornoVirtual:
    """Entorno físico 3D sobre el panel, renderizado en wireframe con
    proyección de perspectiva propia (OpenCV puro, sin dependencias nuevas).

    Controles por gesto (ver actualizar_gestos):
      PUÑO CERRADO + MOVIMIENTO
          → arrastrar figura tocada (x, y, z).
      DOS MANOS con pulgar+índice+medio abiertos (anular y meñique cerrados)
          separarlas  → avanzar cámara (dolly in).
          juntarlas   → retroceder cámara (dolly out).
      UNA MANO con índice+medio abiertos (pulgar, anular y meñique cerrados)
          mover la muñeca de arriba a abajo      → gira el entorno tumbando
                                                     el eje Z (pitch, eje X).
          mover la muñeca de izquierda a derecha → gira el plano XY, "mano
                                                     acostada" (yaw, eje Y).
          El giro es continuo, grado a grado, proporcional a cuánto se
          movió la muñeca en ese frame (ver ESCALA_ROTAR): no es un salto
          fijo, se mueve la mano y el entorno gira junto con ella.
    """

    def __init__(self, ancho_panel, alto_panel):
        self.ancho_panel = ancho_panel
        self.alto_panel  = alto_panel
        self.radio_colision = 14
        self.escala_profundidad = min(ancho_panel, alto_panel)

        # Cámara virtual
        self.rotacion         = np.eye(3, dtype=np.float64)
        self.distancia_camara = 950.0
        self.distancia_min    = 350.0
        self.distancia_max    = 2600.0
        self.focal            = 620.0
        self.centro_x         = ancho_panel / 2.0
        self.centro_y         = alto_panel  / 2.0

        # Estado de gestos
        self.arrastres         = {}   # etiqueta -> {figura, centro_anterior (x,y,z)}
        self._rotar_anterior   = {}   # etiqueta -> (cx,cy) muñeca en el frame anterior con el gesto activo
        self._rotar_gracia     = {}   # etiqueta -> frames de tolerancia restantes (ver GRACIA_FRAMES_ROTAR)
        self._dolly_anterior   = {}   # etiqueta -> (cx,cy) anterior

        # Arranca vacío: no hay ninguna figura por defecto. Las figuras las
        # agrega el usuario vía objetos.crear_objeto() -> agregar_figura().
        self.figuras = []

    # ------------------------------------------------------------------
    # Proyección
    # ------------------------------------------------------------------

    def proyectar(self, punto3d):
        """(x,y,z) mundo -> ((sx,sy) pantalla, z_cam).
        Aplica rotación global del entorno (pivoteando en el centro del panel)
        y proyección de perspectiva."""
        x, y, z = punto3d
        v = self.rotacion @ np.array([x - self.centro_x, y - self.centro_y, z])
        z_cam = max(v[2] + self.distancia_camara, 1.0)
        factor = self.focal / z_cam
        sx = int(v[0]*factor + self.centro_x)
        sy = int(v[1]*factor + self.centro_y)
        return (sx, sy), float(z_cam)

    # ------------------------------------------------------------------
    # Agregar figuras
    # ------------------------------------------------------------------

    def agregar_figura(self, puntos_relativos, conexiones,
                       color_normal=(0,200,200), color_tocado=(0,0,255),
                       primitivas_relativas=None, nombre="", propiedades=None):
        """Agrega una figura nueva al entorno.

        puntos_relativos:
            lista de (x,y) o (x,y,z), valores en [0,1].
            z opcional; si falta se asume 0.5 (plano neutro, comportamiento 2D).

        primitivas_relativas: lista de dicts con coordenadas en [0,1]:
            2D heredadas:
                {"tipo":"circulo",    "cx","cy","r",  ["cz"]}
                {"tipo":"rectangulo", "x","y","ancho","alto", ["cz"]}
                {"tipo":"elipse",     "cx","cy","rx","ry",    ["cz"]}
            Nuevas 3D:
                {"tipo":"esfera",    "cx","cy","cz","r"}
                {"tipo":"cubo",      "cx","cy","cz","ancho","alto","profundo"}
                {"tipo":"cilindro",  "cx","cy","cz","r","alto"}

        nombre:
            Descripción del objeto (la misma que se usó para generarlo). Se almacena
            en la figura para poder encontrarla después y asignarle las propiedades
            cuando llegan del segundo paso de objetos.crear_objeto.

        propiedades:
            Ficha física inicial (opcional). Normalmente llega en None y se completa
            después con asignar_propiedades_a_figura().
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
            locales, aristas = _malla_anillo(r, r)
            cx = prim["cx"] * self.ancho_panel
            cy = prim["cy"] * self.alto_panel

        elif tipo == "elipse":
            rx = prim["rx"] * self.ancho_panel
            ry = prim["ry"] * self.alto_panel
            locales, aristas = _malla_anillo(rx, ry)
            cx = prim["cx"] * self.ancho_panel
            cy = prim["cy"] * self.alto_panel

        elif tipo == "rectangulo":
            w = prim["ancho"] * self.ancho_panel
            h = prim["alto"]  * self.alto_panel
            locales, aristas = _malla_rectangulo(w, h)
            cx = (prim["x"] + prim["ancho"]/2.0) * self.ancho_panel
            cy = (prim["y"] + prim["alto"] /2.0) * self.alto_panel

        elif tipo == "esfera":
            r = prim["r"] * min(self.ancho_panel, self.alto_panel)
            locales, aristas = _malla_esfera(r)
            cx = prim["cx"] * self.ancho_panel
            cy = prim["cy"] * self.alto_panel

        elif tipo == "cubo":
            w = prim["ancho"] * self.ancho_panel
            h = prim["alto"]  * self.alto_panel
            d = prim.get("profundo", prim["ancho"]) * self.escala_profundidad
            locales, aristas = _malla_cubo(w, h, d)
            cx = prim["cx"] * self.ancho_panel
            cy = prim["cy"] * self.alto_panel

        elif tipo == "cilindro":
            r = prim["r"]    * min(self.ancho_panel, self.alto_panel)
            h = prim["alto"] * self.alto_panel
            locales, aristas = _malla_cilindro(r, h)
            cx = prim["cx"] * self.ancho_panel
            cy = prim["cy"] * self.alto_panel

        else:
            return None

        prim_mundo = dict(prim)
        prim_mundo["centro"] = [cx, cy, cz]
        prim_mundo["puntos_locales"] = locales
        prim_mundo["aristas"] = aristas
        return prim_mundo

    # ------------------------------------------------------------------
    # Colisión (en espacio de pantalla)
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

    def _segmentos_proy(self, figura):
        """Todas las aristas de la figura proyectadas a 2D con la cámara actual."""
        seg = []
        for a, b in figura.segmentos_mundo():
            pa, _ = self.proyectar(a)
            pb, _ = self.proyectar(b)
            seg.append((pa, pb))
        for a, b in figura.primitivas_segmentos_mundo():
            pa, _ = self.proyectar(a)
            pb, _ = self.proyectar(b)
            seg.append((pa, pb))
        return seg

    def actualizar(self, puntos_mano):
        """Detecta qué figuras están tocadas por los puntos 2D de pantalla de la mano."""
        for figura in self.figuras:
            segs = self._segmentos_proy(figura)
            tocado = False
            for p in puntos_mano:
                for a, b in segs:
                    if self._dist_punto_seg_2d(p, a, b) <= self.radio_colision:
                        tocado = True
                        break
                if tocado:
                    break
            figura.tocado = tocado

    def _figura_tocada(self, puntos_pantalla):
        for figura in self.figuras:
            segs = self._segmentos_proy(figura)
            for p in puntos_pantalla:
                for a, b in segs:
                    if self._dist_punto_seg_2d(p, a, b) <= self.radio_colision:
                        return figura
        return None

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
            "estados"     : [Pulgar, Índice, Medio, Anular, Meñique] booleanos
                             (extendido/cerrado), para los gestos de zoom y rotar.
            "cerrada"     : True si los 5 dedos están cerrados (puño).
        """
        etiquetas_presentes = {m["etiqueta"] for m in manos}
        manos_zoom = {m["etiqueta"]: m for m in manos
                      if _es_gesto_zoom(m.get("estados")) and m["etiqueta"] in ("Izquierda","Derecha")}

        # ---- ZOOM: pulgar+índice+medio abiertos en ambas manos
        #      (anular y meñique cerrados) → dolly (separar=acercar, juntar=alejar) ----
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

        # ---- ROTAR: índice+medio abiertos en una mano
        #      (pulgar, anular y meñique cerrados) → gira continuo, grado a
        #      grado, proporcional a cuánto se mueve la muñeca este frame.
        #      Vertical (arriba↔abajo)   → pitch alrededor del eje X (tumba el eje Z).
        #      Horizontal (izq↔der)      → yaw alrededor del eje Y (plano XY, "mano acostada").
        manos_rotar = {m["etiqueta"]: m for m in manos if _es_gesto_rotar(m.get("estados"))}

        etiqueta_activa = None
        if len(manos_rotar) == 1 and not dolly_aplicado:
            etiqueta_activa = next(iter(manos_rotar))

        # Colchón de gracia: una mano que ya tenía el gesto iniciado no pierde
        # su posición anterior por un frame aislado sin el patrón exacto (ver
        # GRACIA_FRAMES_ROTAR) — solo se descarta si pasa varios frames seguidos
        # sin volver a matchear.
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
                # Primer frame del gesto: solo fija el punto de partida,
                # todavía no hay movimiento que convertir en giro.
                self._rotar_anterior[etiqueta_activa] = (cx, cy)
            else:
                px, py = anterior
                dx, dy = cx - px, cy - py

                if dx or dy:
                    if abs(dy) >= abs(dx):
                        eje, delta = (1.0, 0.0, 0.0), dy       # pitch: tumba el eje Z
                    else:
                        eje, delta = (0.0, 1.0, 0.0), dx       # yaw: plano XY (mano acostada)

                    angulo = -delta * ESCALA_ROTAR
                    angulo = max(-ANGULO_MAX_ROTACION_FRAME,
                                min(ANGULO_MAX_ROTACION_FRAME, angulo))
                    R = _matriz_rotacion_eje_angulo(eje, angulo)
                    self.rotacion = R @ self.rotacion

                self._rotar_anterior[etiqueta_activa] = (cx, cy)

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

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------

    def dibujar(self, panel):
        # Pintor's algorithm: más lejanas primero
        for figura in sorted(self.figuras,
                             key=lambda f: -f.profundidad_media(self.proyectar)):
            figura.dibujar(panel, self.proyectar)
        self._dibujar_brujula(panel)

    def _dibujar_brujula(self, panel):
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