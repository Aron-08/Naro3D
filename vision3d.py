"""
vision3d.py — Visión 3D real sobre el panel: paralaje de movimiento (head
tracking, el truco de Johnny Lee con el sensor de la Wii), proyección fuera
de eje (off-axis / asymmetric frustum), estéreo binocular por anaglifo
rojo/cian, y profundidad reforzada con sombra de contacto (la oclusión y el
tamaño relativo ya son gratis: vienen del pintor's algorithm + la perspectiva
que ya tiene entorno_virtual.py).

MÓDULO DESCONECTADO: ningún otro archivo del proyecto lo importa todavía.
No modifica ni necesita modificar entorno_virtual.py — ver el porqué y las
instrucciones de conexión más abajo.

-------------------------------------------------------------------------
Por qué proyección fuera de eje y no la perspectiva simétrica que ya hay
-------------------------------------------------------------------------
`EntornoVirtual.proyectar()` (entorno_virtual.py, línea ~339) hace una
perspectiva de foco fijo: la cámara virtual está siempre centrada y mirando
de frente. Eso dibuja un mundo *dentro de un dibujo* — gires la cabeza como
gires, ves siempre la misma imagen. Para que la ventana se sienta como una
caja con un agujero (parallax real), el plano de proyección tiene que quedar
FIJO en el espacio — es el borde físico del panel en el monitor, que no se
mueve — y el punto de vista tiene que ser la posición real del ojo del
usuario, que sí se mueve. La matriz de proyección se recalcula cada frame en
función de esa posición: eso es exactamente una asymmetric/off-axis frustum,
y es matemáticamente equivalente a la transformación por punto que usa
`proyector_fuera_de_eje()` más abajo (ver Kooima, "Generalized Perspective
Projection", 2009 — es la misma idea que usó Johnny Lee con el sensor IR de
un Wiimote apuntando al usuario en vez de a la pantalla).

-------------------------------------------------------------------------
Por qué no hace falta tocar entorno_virtual.py
-------------------------------------------------------------------------
`Figura3D.dibujar(panel, proyector)` ya recibe la función de proyección como
parámetro — no está atada a `EntornoVirtual.proyectar`. Este módulo arma su
propia función de proyección (fuera de eje, una por ojo) y reimplementa el
mismo bucle de pintor's algorithm que usa `EntornoVirtual.dibujar()`, pero
iterando directamente sobre `entorno.figuras` y usando `entorno.rotacion`,
`entorno.centro_x/centro_y` (todos atributos públicos que ya existen). Cero
cambios de comportamiento en entorno_virtual.py; este módulo solo lo *lee*.

-------------------------------------------------------------------------
Sobre gafas polarizadas
-------------------------------------------------------------------------
No se implementan acá. Gafas rojo/cian filtran color — cualquier monitor
sirve. Gafas polarizadas filtran por *ángulo de polarización de la luz*, algo
que un monitor común no emite: hace falta hardware específico (un monitor
"passive 3D" con líneas entrelazadas pares/impares polarizadas distinto, o
dos proyectores con filtros cruzados). Si en algún momento hay ese hardware,
`_render_ojo()` ya deja los dos frames (izquierdo/derecho) separados antes de
componer el anaglifo — el punto de entrelazado por líneas se agrega ahí sin
tocar el resto del módulo.

-------------------------------------------------------------------------
Instrucciones de conexión (NO aplicadas — ver más abajo qué tocar)
-------------------------------------------------------------------------
Ver el mensaje de chat que acompaña este archivo: tiene el detalle línea por
línea de qué agregar en main.py (import, instanciar RastreadorCabeza y
MotorEstereo, alternar el FaceMesh con el procesamiento de manos, y
reemplazar los dos call-sites de `entorno.dibujar(panel)` por un wrapper que
elige entre el render normal y el 3D real). entorno_virtual.py no se toca.
"""

import math

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


# ---------------------------------------------------------------------------
# Constantes físicas y de calibración
# ---------------------------------------------------------------------------

# Distancia interpupilar promedio de un adulto. Varía persona a persona
# (rango real ~5.4-7.4cm); usarla como constante fija introduce un error de
# escala en la distancia estimada, pero ese error es el mismo en todos los
# frames (es un factor multiplicativo constante para una persona dada), así
# que no rompe el paralaje: solo corre la "profundidad cero" un poco más
# cerca o lejos de lo real. Para el efecto visual (que la escena reaccione a
# dónde está la cabeza) no hace falta más precisión que esta.
IPD_PROMEDIO_CM = 6.3

# Separación entre ojos para el render estéreo. Es el equivalente de
# "6.5cm a cada lado" que se pide en la consigna, pero medido de punta a
# punta (no de centro a cada lado), por eso el motor usa la mitad.
SEPARACION_OJOS_CM = 6.5


class CalibracionPantalla:
    """Constantes físicas del setup real (monitor + cámara). Con valores por
    defecto razonables, pero para que el paralaje "cierre" bien (que el
    punto donde debería estar cada figura coincida con dónde se ve) conviene
    calibrar esto una vez por PC. Los dos equipos del proyecto son gubernamentales
    y distintos entre sí, así que esto casi seguro necesita un valor por máquina.

    Cómo calibrar a mano (2 minutos, sin herramientas):
      1. ancho_cm / alto_cm: medí con una regla el rectángulo del monitor
         que ocupa la ventana del panel (no el monitor entero si la ventana
         no está maximizada).
      2. offset_camara_y_cm: distancia vertical entre el centro del monitor
         y la webcam (típicamente montada arriba: valor positivo si la
         cámara está por ENCIMA del centro de la pantalla).
      3. fov_horizontal_deg: campo visual horizontal de la webcam. Si no lo
         sabés, 60° es un valor típico de webcam integrada/USB genérica;
         66-70° es más típico de webcams "gran angular". Para afinarlo:
         parate a una distancia conocida (ej. 60cm, medida con la regla),
         fijate el ancho en píxeles que mide tu propia cara en el frame de
         MP_ANCHO x MP_ALTO, y ajustá fov_horizontal_deg hasta que
         `distancia_cm` (ver RastreadorCabeza) devuelva ~60cm parado ahí.
      4. distancia_default_cm: a qué distancia asumir que está el usuario
         cuando todavía no se detectó ninguna cara (arranque, o cara fuera
         de cuadro) — usar la distancia habitual de uso.
    """

    def __init__(self, ancho_cm=34.0, alto_cm=19.0, offset_camara_y_cm=10.0,
                 fov_horizontal_deg=60.0, distancia_default_cm=55.0):
        self.ancho_cm = ancho_cm
        self.alto_cm = alto_cm
        self.offset_camara_y_cm = offset_camara_y_cm
        self.fov_horizontal_deg = fov_horizontal_deg
        self.distancia_default_cm = distancia_default_cm


def _focal_equivalente_px(ancho_frame_px, fov_horizontal_deg):
    """Focal de la webcam en píxeles, derivada de su FOV horizontal asumido.
    Es el mismo modelo pinhole que usa cualquier cámara: focal_px =
    (ancho/2) / tan(fov/2). Con esto se puede pasar de tamaños en píxeles a
    distancias reales en cm sin necesidad de calibración de cámara formal
    (chessboard, etc.) — suficiente para este uso, no para medición precisa."""
    return (ancho_frame_px / 2.0) / math.tan(math.radians(fov_horizontal_deg) / 2.0)


# ---------------------------------------------------------------------------
# Head tracking — Face Landmarker liviano (mismo criterio que mp_hands: API
# de Solutions, no la API de Tasks nueva, para no sumar la descarga de un
# .task aparte).
# ---------------------------------------------------------------------------

class RastreadorCabeza:
    """Estima la posición 3D de la cabeza del usuario respecto a la cámara,
    usando la distancia interpupilar en píxeles (landmarks de iris) como
    proxy de distancia — el mismo principio que el hack de Johnny Lee, solo
    que ahí usaba dos LEDs IR de separación conocida en vez de los ojos.

    Se corre cada `cada_n_frames` frames (alternando con el tracking de
    manos, igual que la consigna: "alternando cada 100ms, depende de qué
    rinda mejor"): el FaceMesh es liviano, pero sumarlo TODOS los frames
    junto con Hands en un equipo sin GPU dedicada es plata que este proyecto
    no tiene para gastar (ver el comentario de MP_COMPLEJIDAD en main.py,
    mismo tipo de trade-off). Cuando se saltea un frame, devuelve la última
    posición conocida — igual que el "ghost mode" de manos en main.py, para
    que la cabeza no "salte" entre detecciones.
    """

    def __init__(self, ancho_frame_px, alto_frame_px, calibracion,
                 cada_n_frames=2, alpha_suavizado=0.35,
                 min_confianza_deteccion=0.5, min_confianza_seguimiento=0.5):
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # obligatorio: sin esto no hay landmarks
                                     # de iris (468-477), y sin iris no hay
                                     # forma liviana de medir distancia
                                     # interpupilar en píxeles.
            min_detection_confidence=min_confianza_deteccion,
            min_tracking_confidence=min_confianza_seguimiento,
        )
        self._ancho_frame_px = ancho_frame_px
        self._alto_frame_px = alto_frame_px
        self._calibracion = calibracion
        self._focal_px = _focal_equivalente_px(ancho_frame_px, calibracion.fov_horizontal_deg)
        self._cada_n_frames = max(1, cada_n_frames)
        self._alpha = alpha_suavizado
        self._contador = 0
        self._ultima_pos_cm = None   # persistencia entre detecciones (ver docstring)

    def procesar(self, frame_rgb):
        """`frame_rgb`: mismo frame (sin el margen negro que se usa para
        manos — la cara no lo necesita, ver instrucciones de conexión) ya
        convertido a RGB, tamaño ancho_frame_px x alto_frame_px.

        Devuelve (ex_cm, ey_cm, ez_cm): posición del punto medio entre ambos
        ojos, en cm, en un sistema centrado en la PANTALLA (no en la
        cámara — ya se le resta el offset físico cámara→centro de pantalla).
        ez_cm es la distancia del ojo al plano de la pantalla.
        Devuelve None solo si nunca se detectó ninguna cara todavía (ni en
        este frame ni en ninguno anterior) — quien llama debe manejar ese
        caso (ver MotorEstereo.renderizar, usa distancia_default_cm).
        """
        self._contador += 1
        if self._contador % self._cada_n_frames != 0:
            return self._ultima_pos_cm

        resultados = self._face_mesh.process(frame_rgb)
        if not resultados.multi_face_landmarks:
            return self._ultima_pos_cm

        lm = resultados.multi_face_landmarks[0].landmark
        # Con refine_landmarks=True: 468 = centro iris derecho (del sujeto),
        # 473 = centro iris izquierdo. El frame ya llega espejado (mismo
        # cv2.flip que usa main.py para manos), así que "derecho/izquierdo"
        # acá es como los ve la cámara, no como los siente el usuario — no
        # importa para esta cuenta, es simétrica.
        iris_a, iris_b = lm[468], lm[473]

        ipd_px = math.hypot(
            (iris_a.x - iris_b.x) * self._ancho_frame_px,
            (iris_a.y - iris_b.y) * self._alto_frame_px,
        )
        if ipd_px < 1.0:
            return self._ultima_pos_cm

        distancia_cm = (IPD_PROMEDIO_CM * self._focal_px) / ipd_px

        cx_px = (iris_a.x + iris_b.x) / 2.0 * self._ancho_frame_px
        cy_px = (iris_a.y + iris_b.y) / 2.0 * self._alto_frame_px

        # Proyección inversa del modelo pinhole: de (px, distancia) a cm
        # reales en el plano perpendicular a la cámara a esa distancia.
        ex_cm = (cx_px - self._ancho_frame_px / 2.0) * distancia_cm / self._focal_px
        ey_cm = (cy_px - self._alto_frame_px / 2.0) * distancia_cm / self._focal_px
        ey_cm -= self._calibracion.offset_camara_y_cm  # cámara -> centro de pantalla

        nueva_pos = (ex_cm, ey_cm, distancia_cm)

        if self._ultima_pos_cm is None:
            self._ultima_pos_cm = nueva_pos
        else:
            a = self._alpha
            self._ultima_pos_cm = tuple(
                a * nuevo + (1.0 - a) * anterior
                for nuevo, anterior in zip(nueva_pos, self._ultima_pos_cm)
            )
        return self._ultima_pos_cm

    def cerrar(self):
        self._face_mesh.close()


# ---------------------------------------------------------------------------
# Proyección fuera de eje (off-axis / asymmetric frustum)
# ---------------------------------------------------------------------------

def proyector_fuera_de_eje(entorno, ojo_cm, calibracion, ancho_panel_px, alto_panel_px):
    """Devuelve una función `punto3d -> ((sx,sy), z_cam)`, con la misma forma
    que `entorno.proyectar`, pero que en vez de una cámara de foco fijo usa
    el ojo real del usuario (`ojo_cm`, en cm, centrado en la pantalla) como
    punto de vista, contra un plano de proyección fijo (el panel).

    Reusa `entorno.rotacion` (así el objeto sigue girando con el gesto de
    rotación existente) y `entorno.centro_x/centro_y` (mismo pivote), pero
    ignora `entorno.focal`/`entorno.distancia_camara`: esos eran los
    parámetros de la cámara virtual de foco fijo que esto reemplaza.
    """
    px_por_cm = ancho_panel_px / calibracion.ancho_cm
    ex_px = ojo_cm[0] * px_por_cm
    ey_px = ojo_cm[1] * px_por_cm
    dist_ojo_px = max(ojo_cm[2] * px_por_cm, 1.0)

    rotacion = entorno.rotacion
    centro_x, centro_y = entorno.centro_x, entorno.centro_y

    def proyector(punto3d):
        x, y, z = punto3d
        v = rotacion @ np.array([x - centro_x, y - centro_y, z])

        # Transformación proyectiva por punto para un ojo arbitrario contra
        # un plano fijo (equivalente al frustum asimétrico completo, ver
        # docstring del módulo). denom es la distancia del ojo al punto a lo
        # largo de z; se acota para no dividir por ~0 en casos degenerados.
        denom = dist_ojo_px - v[2]
        if -1.0 < denom < 1.0:
            denom = -1.0 if denom < 0 else 1.0
        factor = dist_ojo_px / denom

        sx = int(centro_x + ex_px + (v[0] - ex_px) * factor)
        sy = int(centro_y + ey_px + (v[1] - ey_px) * factor)
        return (sx, sy), float(denom)

    return proyector


# ---------------------------------------------------------------------------
# Render por ojo: mismo pintor's algorithm que EntornoVirtual.dibujar(), pero
# parametrizado por proyector — y con una sombra de contacto simple para
# reforzar la profundidad además de la oclusión (que ya viene gratis del
# orden de dibujo) y el tamaño relativo (que ya viene gratis de la
# perspectiva).
# ---------------------------------------------------------------------------

def _centro_figura(figura):
    puntos = figura.puntos or [tuple(p["centro"]) for p in figura.primitivas]
    if not puntos:
        return None
    n = len(puntos)
    cx = sum(p[0] for p in puntos) / n
    cy = sum(p[1] for p in puntos) / n
    cz = sum(p[2] for p in puntos) / n
    return (cx, cy, cz)


def _dibujar_sombra_contacto(panel, figura, proyector, alto_panel_px):
    """Sombra de contacto simple (no física, no raytraceada): un óvalo
    oscuro debajo de cada figura, más grande y más tenue cuanto más lejos
    está — mismo truco que usa cualquier motor 2.5D liviano para reforzar
    "dónde toca el piso" sin tener que simular luces."""
    centro = _centro_figura(figura)
    if centro is None:
        return
    (sx, _sy), z_cam = proyector(centro)
    if z_cam <= 1.0:
        return
    y_piso = int(alto_panel_px * 0.94)
    escala = max(1.0, 4000.0 / z_cam)  # figuras más cerca -> sombra más grande
    ancho_sombra = int(18 * escala)
    alto_sombra = int(6 * escala)
    if ancho_sombra < 2 or alto_sombra < 1:
        return
    overlay = panel.copy()
    cv2.ellipse(overlay, (sx, y_piso), (ancho_sombra, alto_sombra), 0, 0, 360,
                (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, dst=panel)


def _render_ojo(entorno, proyector, ancho_panel_px, alto_panel_px, con_sombra=True):
    panel = np.zeros((alto_panel_px, ancho_panel_px, 3), dtype=np.uint8)
    figuras_ordenadas = sorted(entorno.figuras, key=lambda f: -f.profundidad_media(proyector))
    if con_sombra:
        for figura in figuras_ordenadas:
            _dibujar_sombra_contacto(panel, figura, proyector, alto_panel_px)
    for figura in figuras_ordenadas:
        # Pintor's algorithm: al estar ordenadas de más lejos a más cerca,
        # las cercanas se dibujan encima -> oclusión correcta gratis.
        figura.dibujar(panel, proyector)
    # EntornoVirtual.dibujar() también dibuja la brújula de ejes (mismo
    # método, no se reimplementa acá — es solo lectura, cero cambios en
    # entorno_virtual.py). La brújula no pasa por `proyector`: usa
    # entorno.rotacion directo y una posición de pantalla fija, así que
    # dibujarla igual en cada ojo del anaglifo da el mismo píxel en ambos
    # renders (sin fantasma rojo/cian) — es el resultado correcto para un
    # HUD que no tiene paralaje propio.
    entorno._dibujar_brujula(panel)
    return panel


def _componer_anaglifo(panel_izq, panel_der):
    """Anaglifo rojo/cian: canal rojo del ojo izquierdo, canales verde+azul
    del ojo derecho. cv2 usa orden BGR -> el canal rojo es el índice 2."""
    salida = panel_der.copy()
    salida[:, :, 2] = panel_izq[:, :, 2]
    return salida


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

MODO_OFFAXIS = "offaxis"   # paralaje de movimiento, mono (sin gafas)
MODO_ANAGLIFO = "anaglifo"  # paralaje + estéreo binocular (gafas rojo/cian)


class MotorEstereo:
    """Punto de entrada único para renderizar el entorno con visión 3D real.
    Ver `renderizar()`."""

    def __init__(self, calibracion, separacion_ojos_cm=SEPARACION_OJOS_CM):
        self._calibracion = calibracion
        self._separacion_ojos_cm = separacion_ojos_cm

    def renderizar(self, entorno, ojo_cm, ancho_panel_px, alto_panel_px, modo=MODO_ANAGLIFO):
        if ojo_cm is None:
            # Todavía no se detectó ninguna cara (arranque en frío): cámara
            # fija centrada a la distancia default, sin paralaje real hasta
            # que RastreadorCabeza tenga su primera lectura.
            ojo_cm = (0.0, 0.0, self._calibracion.distancia_default_cm)

        if modo == MODO_OFFAXIS:
            proyector = proyector_fuera_de_eje(
                entorno, ojo_cm, self._calibracion, ancho_panel_px, alto_panel_px)
            return _render_ojo(entorno, proyector, ancho_panel_px, alto_panel_px)

        mitad = self._separacion_ojos_cm / 2.0
        ojo_izq = (ojo_cm[0] - mitad, ojo_cm[1], ojo_cm[2])
        ojo_der = (ojo_cm[0] + mitad, ojo_cm[1], ojo_cm[2])

        proyector_izq = proyector_fuera_de_eje(
            entorno, ojo_izq, self._calibracion, ancho_panel_px, alto_panel_px)
        proyector_der = proyector_fuera_de_eje(
            entorno, ojo_der, self._calibracion, ancho_panel_px, alto_panel_px)

        panel_izq = _render_ojo(entorno, proyector_izq, ancho_panel_px, alto_panel_px)
        panel_der = _render_ojo(entorno, proyector_der, ancho_panel_px, alto_panel_px)
        return _componer_anaglifo(panel_izq, panel_der)