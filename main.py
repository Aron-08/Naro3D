import math
import threading
import time
import tkinter as tk

import cv2
import numpy as np
import mediapipe as mp

from entorno_virtual import EntornoVirtual
import objetos as obj
import ubicacion       # crear_objeto orquesta geometría → propiedades en secuencia
import editor_visual    # panel para editar propiedades físicas, color y escala
import optimizacion_objetos as opt_obj  # calidad dinámica en tiempo real (ver módulo)
import malla as malla_module  # Malla.from_dict() para reconstruir mallas de biblioteca/IA (PLAN_RECONSTRUCCION_MALLAS.md)
from ui_thread import en_hilo_ui  # tkinter no es thread-safe: ver ui_thread.py

mp_hands = mp.solutions.hands

# ---------------------------------------------------------------------------
# Configuración de cámara
# ---------------------------------------------------------------------------
# Capturar a mayor resolución le da a MediaPipe más píxeles por mano → mejora
# la detección a distancia y la precisión de landmarks. Solo para detección;
# el frame se reduce a PANEL_W×PANEL_H antes del overlay y del imshow.
CAM_INDICE  = 0
CAM_ANCHO   = 1920
CAM_ALTO    = 1080
CAM_FPS     = 30

# ---------------------------------------------------------------------------
# Resolución de trabajo para MediaPipe (separada de la de captura)
# ---------------------------------------------------------------------------
# Antes, MediaPipe procesaba el frame completo a resolución de captura
# (1280x720). Esa es la causa principal de los FPS bajos: la inferencia de
# MediaPipe escala aproximadamente con la cantidad de píxeles de entrada.
# Acá se separa "qué tan nítido es el video" (CAM_ANCHO/ALTO, ahora más alto
# que antes) de "qué tan grande es la imagen que ve MediaPipe" (MP_ANCHO/ALTO,
# bien más chica). Los landmarks que devuelve MediaPipe son coordenadas
# normalizadas en [0,1], así que mapean igual de bien a cualquier resolución
# de destino (PANEL_W/PANEL_H, frame de video, etc.) sin importar a qué
# tamaño se haya procesado la inferencia. Resultado: más píxeles en el video
# Y más FPS al mismo tiempo, porque son dos cosas independientes.
MP_ANCHO = 640
MP_ALTO  = 360

# ---------------------------------------------------------------------------
# Margen de detección ("padding") — arregla la mano "desapareciendo" de cerca
# ---------------------------------------------------------------------------
# El palm detector de MediaPipe está entrenado con encuadres tipo selfie,
# donde la mano ocupa una porción moderada del frame con margen alrededor.
# Cuando acercás la mano a la cámara y empieza a llenar casi todo el cuadro,
# el score de detección cae por debajo del umbral y la mano se "pierde",
# aunque siga completamente visible. Agregar un borde negro alrededor de la
# imagen ANTES de pasarla a MediaPipe reduce la proporción que ocupa la mano
# dentro del frame procesado (sin achicar los píxeles reales de la mano), lo
# que mantiene el detector en su rango de confianza entrenado.
# Esto NO arregla el caso en que la mano queda físicamente fuera del campo
# de visión de la cámara (dedos cortados en el borde de la imagen real):
# eso requiere alejar un poco la mano o usar una cámara de mayor FOV.
MARGEN_DETECCION = 0.30  # fracción de MP_ANCHO/MP_ALTO agregada como borde

# ---------------------------------------------------------------------------
# Configuración de MediaPipe
# ---------------------------------------------------------------------------
# model_complexity=1 usa el modelo completo en lugar del lite:
#   · Precisión de landmarks ~5 pp mejor (especialmente en poses oblicuas).
#   · Detecta manos más alejadas (el modelo ve detalles más finos).
#   · ~30-50 ms más lento por frame en CPU moderada; irrelevante si la GPU
#     o el backend ONNX de MediaPipe corre en hardware.
MP_COMPLEJIDAD      = 1

# min_detection_confidence: umbral sobre el score del detector de palma.
# Bajarlo de 0.6 a 0.45 permite detectar manos pequeñas/lejanas cuyo
# score no llega al umbral anterior. Costo: leve aumento de falsos positivos
# en fondos muy texturizados. Subir a 0.7+ si hay fantasmas en el fondo.
MP_CONF_DETECCION   = 0.40

# min_tracking_confidence: si el score del tracker cae por debajo de este
# umbral, MediaPipe abandona el tracking y corre el detector completo desde
# cero. Subirlo de 0.60 a 0.70 hace que la re-detección se dispare antes de
# que el tracking haya colapsado del todo → la mano se recupera más rápido
# en movimientos bruscos o poses extremas.
MP_CONF_SEGUIMIENTO = 0.70

# ---------------------------------------------------------------------------
# CLAHE (realce de contraste adaptativo local)
# ---------------------------------------------------------------------------
# Se aplica al canal L en espacio LAB antes de enviar el frame a MediaPipe.
# Mejora la detección cuando la iluminación es baja, despareja o hay reflejos.
# Solo afecta al frame que procesa MediaPipe; el video visible usa el frame
# original para que el color se vea natural.
CLAHE_CLIP   = 2.0    # factor de recorte de contraste (>3 pixela bordes)
CLAHE_GRID   = (8, 8) # tamaño de cada región local en píxeles

# ---------------------------------------------------------------------------
# Persistencia ("ghost frames")
# ---------------------------------------------------------------------------
# Cuando MediaPipe no devuelve ninguna mano, se reutilizan los últimos
# landmarks válidos durante MAX_GHOST_FRAMES frames antes de declarar
# "sin mano". Cubre el hueco que dejan los movimientos rápidos y los
# cambios de iluminación momentáneos sin introducir lag perceptible.
MAX_GHOST_FRAMES = 8   # ampliado para dar tiempo al fallback de recuperar

# ---------------------------------------------------------------------------
# Detector de fallback (static_image_mode=True)
# ---------------------------------------------------------------------------
# Cuando el tracker principal pierde la mano, este detector corre en paralelo:
# redetecta desde cero sin depender del estado anterior (no hay tracking).
# Más lento que el tracker, pero nunca queda "atascado" por haberla perdido
# en un frame previo. Se activa tras FRAMES_ANTES_FALLBACK frames sin mano.
#
# GAMMA_FALLBACK: la imagen se aclara antes de dársela al fallback.
# Ayuda cuando la mano queda en sombra y el contraste baja.
# 1.0 = sin cambio. 1.8 = bastante más brillante. 2.5 = muy agresivo.
FRAMES_ANTES_FALLBACK = 1      # frames sin detección antes de activar fallback
GAMMA_FALLBACK        = 1.8    # brightening gamma para el fallback
MP_CONF_FALLBACK      = 0.35   # umbral de detección más bajo para el fallback

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=MP_COMPLEJIDAD,
    min_detection_confidence=MP_CONF_DETECCION,
    min_tracking_confidence=MP_CONF_SEGUIMIENTO,
)

cap = cv2.VideoCapture(CAM_INDICE)
# MJPG reduce la carga del bus USB/UVC: la cámara comprime antes de enviar,
# lo que permite alcanzar 30 fps en resoluciones altas sin saturar el ancho
# de banda (solo descomprimimos en CPU una vez, no en cada frame raw).
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_ANCHO)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_ALTO)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
# Buffer de 1 frame: por default el backend V4L2/UVC encola varios frames.
# Si el procesamiento (MediaPipe) tarda más que el intervalo entre frames,
# esa cola se llena y vamos mostrando/procesando frames cada vez más viejos
# — el video "atrasa". Eso es lo que se percibe como que la mano se mueve
# rápido y "desaparece": en realidad se está procesando dónde estaba la mano
# hace varios frames. Con buffer=1 siempre se descarta lo viejo y se toma
# el frame más reciente disponible. No todos los backends lo respetan al
# 100%, por eso además se usa el hilo de captura de abajo, que logra lo
# mismo de forma más confiable.
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class CapturaHilo:
    """Lee frames de la cámara en un hilo separado y deja siempre disponible
    el último frame capturado. El hilo principal (el que corre MediaPipe)
    nunca espera a la cámara ni se queda mostrando frames atrasados: si
    llega más rápido que la cámara reusa el último frame, y si la cámara
    entregó varios frames mientras el hilo principal estaba ocupado, los
    intermedios se descartan automáticamente (solo importa el más nuevo).
    Esto es lo que de verdad sube los FPS percibidos en cámaras USB/UVC."""

    def __init__(self, captura):
        self._captura = captura
        self._frame = None
        self._lock = threading.Lock()
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def _bucle(self):
        while self._activo:
            ret, frame = self._captura.read()
            if not ret:
                continue
            with self._lock:
                self._frame = frame

    def leer(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def detener(self):
        self._activo = False
        self._hilo.join(timeout=1.0)
        self._captura.release()


captura_hilo = CapturaHilo(cap)

# Índices de landmarks de mediapipe que nos interesan
PUNTAS = [4, 8, 12, 16, 20]          # pulgar, índice, medio, anular, meñique
NUDILLOS = [2, 5, 9, 13, 17]         # base de cada dedo (MCP; en el pulgar es su MCP)
INTERMEDIAS = [3, 6, 10, 14, 18]     # articulación donde se dobla la falange proximal con la media (IP/PIP)
BASE_PALMA = [0, 5, 9, 13, 17]       # muñeca + nudillos (MCP) -> promedio = centro de palma

NOMBRES_DEDOS = ["Pulgar", "Índice", "Medio", "Anular", "Meñique"]

# Tamaño del panel donde vive el entorno virtual (y donde se dibuja la mano "donde está", sin recentrarla)
# Subido de 640x480 a 960x720: el costo de dibujar con cv2 es insignificante
# comparado con la inferencia de MediaPipe, así que esto no cuesta FPS — es
# pura ganancia de nitidez en la ventana final.
PANEL_W, PANEL_H = 960, 720

entorno = EntornoVirtual(PANEL_W, PANEL_H)

# Calidad dinámica: baja/sube el LOD de las figuras con Malla real según los
# fps reales del bucle (ver optimizacion_objetos.py). No decima nada en
# caliente, solo elige entre los LOD ya precalculados al crear/optimizar
# cada objeto — costo despreciable de llamar cada frame.
gestor_calidad = opt_obj.GestorCalidadDinamica()

# --- vision3d: head tracking + proyección fuera de eje + estéreo anaglifo ---
# Ver vision3d.py para el detalle. calibracion_pantalla tiene valores por
# defecto razonables pero conviene medir el setup real una vez (regla sobre
# el monitor, offset de la webcam) — ver el docstring de CalibracionPantalla.
# Cero costo si _modo_vision3d queda en None: renderizar_entorno() cae
# directo a entorno.dibujar(panel), el camino de siempre.
import vision3d

calibracion_pantalla = vision3d.CalibracionPantalla(
    ancho_cm=34.0, alto_cm=19.0,
    offset_camara_y_cm=10.0,
    fov_horizontal_deg=60.0,
    distancia_default_cm=55.0,
)
rastreador_cabeza = vision3d.RastreadorCabeza(MP_ANCHO, MP_ALTO, calibracion_pantalla, cada_n_frames=2)
motor_estereo = vision3d.MotorEstereo(calibracion_pantalla)
_modo_vision3d = None   # None | vision3d.MODO_OFFAXIS | vision3d.MODO_ANAGLIFO
cabeza_cm = None        # última posición conocida de la cabeza (cm, centrado en pantalla)


def renderizar_entorno(panel):
    """Reemplaza los call-sites directos de entorno.dibujar(panel): si no hay
    ningún modo de vision3d activo, es exactamente lo de siempre (mismo
    panel, mismo costo, dibuja in-place). Con un modo activo, motor_estereo
    re-renderiza todo el frame en función de dónde está la cabeza y el
    resultado se copia in-place sobre `panel` (panel[:] = resultado) — así
    el call-site no necesita reasignar la variable, se comporta exactamente
    como el entorno.dibujar(panel) que reemplaza."""
    if _modo_vision3d is None:
        entorno.dibujar(panel)
        return panel
    resultado = motor_estereo.renderizar(entorno, cabeza_cm, PANEL_W, PANEL_H, modo=_modo_vision3d)
    panel[:] = resultado
    return panel


# Cola de figuras generadas por la IA, listas para agregar al entorno.
# Solo se hace append() desde el hilo de tkinter y pop(0) desde el hilo de OpenCV;
# la GIL de Python hace que ambas operaciones sean atómicas, así que no necesitamos un Lock.
_cola_figuras     = []   # (nombre, datos_figura, pedido_usuario, malla_info) — la figura ya se puede dibujar
_cola_propiedades = []   # (nombre, propiedades)  — llegan después, en el paso 2

# ---------------------------------------------------------------------------
# Skill 01 (ubicación espacial) — detectar si la descripción trae un pedido
# de posición explícito, para no llamar al modelo de ubicación en el caso
# más común (agregar un objeto sin indicar dónde va). Ver 01_skill_ubicacion_
# espacial.md: "si no hay pedido explícito, no se llama al modelo en absoluto".
# ---------------------------------------------------------------------------
_PALABRAS_UBICACION = (
    "al lado", "al costado", "arriba", "abajo", "encima", "debajo",
    "entre", "tocando", "apoyado", "apoyada", "sobre", "junto",
    "cerca", "de canto", "centro", "izquierda", "derecha",
)


def _extraer_pedido_ubicacion(descripcion: str) -> str:
    """Si la descripción menciona una relación espacial, se reusa el mismo
    texto como pedido_usuario para ubicacion.calcular_ubicacion() (esa
    función interpreta con el LLM SOLO la parte que le sirve). Si no hay
    ninguna palabra clave de posición, devuelve "" para que ubicacion.py
    use la colocación por defecto sin gastar una llamada al modelo."""
    texto = descripcion.lower()
    if any(palabra in texto for palabra in _PALABRAS_UBICACION):
        return descripcion
    return ""


# ---------------------------------------------------------------------------
# Panel de control (ventana tkinter en hilo separado)
# ---------------------------------------------------------------------------

def _generar_y_encolar(descripcion, root, label_estado, lista_box):
    """Corre en un hilo daemon: llama a objetos.crear_objeto y encola la figura en cuanto
    la geometría está lista (sin esperar las propiedades físicas), y encola las propiedades
    cuando llegan en el paso 2 — todo en secuencia, sin dos modelos vivos al mismo tiempo.

    Ojo con tkinter acá: este método (y los callbacks de abajo) corren en un hilo que NO
    es el que llamó root.mainloop(). Nunca se toca un widget directo (label_estado.config,
    lista_box.insert, etc.) — todo pasa por en_hilo_ui(root, ...), que lo reprograma en el
    hilo correcto con root.after(0, ...). Sin esto, tocar la UI desde acá es undefined
    behavior: anda casi siempre pero puede colgar o corromper la ventana de forma no
    reproducible, típicamente cuando dos objetos se están generando casi al mismo tiempo.

    _cola_figuras / _cola_propiedades SÍ se pueden tocar directo desde este hilo: son
    listas de Python, no widgets de tkinter, y su atomicidad la da la GIL (ver el
    comentario donde se declaran), no el hecho de estar en el hilo de la UI.
    """
    en_hilo_ui(root, label_estado.config, text=f"Dibujando '{descripcion}'…")
    pedido_ubicacion = _extraer_pedido_ubicacion(descripcion)

    def al_dibujar(nombre, _registro):
        # El registro en disco ya tiene la figura guardada; lo que necesitamos para el
        # entorno son los datos de geometría que vienen en _registro["figura"] (bbox
        # placeholder si vino de biblioteca, wireframe real si vino del fallback LLM) y,
        # si el HIT fue de biblioteca, la Malla real en _registro["malla"] (ver
        # objetos.py::crear_objeto y PLAN_RECONSTRUCCION_MALLAS.md, sección 3).
        datos = _registro.get("figura") if _registro else None
        malla_info = _registro.get("malla") if _registro else None
        if datos:
            _cola_figuras.append((nombre, datos, pedido_ubicacion, malla_info))
            en_hilo_ui(root, lista_box.insert, tk.END, nombre)
        en_hilo_ui(root, label_estado.config, text=f"'{nombre}' dibujado. Pidiendo características…")

    def al_tener_propiedades(nombre, registro):
        if registro is not None:
            props = registro.get("propiedades")
            if props:
                _cola_propiedades.append((nombre, props))
        en_hilo_ui(
            root, label_estado.config,
            text=(f"'{nombre}' completo." if registro else
                  f"'{nombre}' sin características (error en paso 2).")
        )

    registro = obj.crear_objeto(
        descripcion,
        callback_figura=al_dibujar,
        callback_propiedades=al_tener_propiedades,
    )
    if registro is None:
        en_hilo_ui(root, label_estado.config, text=f"No se pudo generar '{descripcion}'.")


def _on_agregar(entry, root, label_estado, lista_box):
    descripcion = entry.get().strip()
    if not descripcion:
        return
    entry.delete(0, tk.END)
    threading.Thread(
        target=_generar_y_encolar,
        args=(descripcion, root, label_estado, lista_box),
        daemon=True,
    ).start()


def _on_eliminar(lista_box):
    """Elimina la figura seleccionada de la lista y del entorno.
    Ya no hay ninguna figura por defecto protegida: cualquier índice
    seleccionado se puede eliminar."""
    sel = lista_box.curselection()
    if not sel:
        return
    idx = sel[0]
    nombre = lista_box.get(idx)
    lista_box.delete(idx)
    if idx < len(entorno.figuras):
        entorno.figuras.pop(idx)
    # Sacarlo también del registro de ubicacion.py — si no, el hueco que
    # ocupaba sigue "reservado" (bbox fantasma) para la resolución de
    # colisiones de la próxima figura que se agregue.
    ubicacion.eliminar_objeto(nombre)


def _lanzar_panel():
    root = tk.Tk()
    root.title("Entorno virtual — objetos")
    root.resizable(False, False)

    # --- Título ---
    tk.Label(root, text="Objetos en el entorno", font=("Helvetica", 11, "bold")).pack(
        pady=(12, 4), padx=12
    )

    # --- Lista de figuras ---
    frame_lista = tk.Frame(root)
    frame_lista.pack(fill=tk.BOTH, padx=12, pady=4)

    scrollbar = tk.Scrollbar(frame_lista, orient=tk.VERTICAL)
    lista_box = tk.Listbox(
        frame_lista,
        height=8,
        width=32,
        yscrollcommand=scrollbar.set,
        selectmode=tk.SINGLE,
        font=("Helvetica", 10),
    )
    scrollbar.config(command=lista_box.yview)
    lista_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Arranca vacía: no hay ninguna figura por defecto en la lista.

    # --- Input + botón Agregar ---
    frame_input = tk.Frame(root)
    frame_input.pack(fill=tk.X, padx=12, pady=(8, 4))

    entry = tk.Entry(frame_input, font=("Helvetica", 10))
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    entry.bind("<Return>", lambda e: _on_agregar(entry, root, label_estado, lista_box))

    btn_agregar = tk.Button(
        frame_input,
        text="Agregar",
        command=lambda: _on_agregar(entry, root, label_estado, lista_box),
    )
    btn_agregar.pack(side=tk.LEFT, padx=(6, 0))

    # --- Botón Eliminar ---
    tk.Button(
        root,
        text="Eliminar seleccionado",
        command=lambda: _on_eliminar(lista_box),
    ).pack(pady=(4, 4), padx=12, fill=tk.X)

    # --- Botón Editor visual (propiedades, color, escala) ---
    tk.Button(
        root,
        text="Editor visual…",
        command=lambda: editor_visual.abrir_editor_visual(root),
    ).pack(pady=(0, 4), padx=12, fill=tk.X)

    # --- Label de estado ---
    label_estado = tk.Label(
        root,
        text="Escribí una figura y presioná Agregar.",
        font=("Helvetica", 9),
        fg="gray",
        wraplength=230,
    )
    label_estado.pack(pady=(0, 12), padx=12)

    root.mainloop()


# Lanzar el panel en un hilo separado para que no bloquee el bucle de OpenCV
threading.Thread(target=_lanzar_panel, daemon=True).start()


# ---------------------------------------------------------------------------
# Funciones de procesamiento de mano (sin cambios respecto al original)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Detección de dedos — sistema multi-criterio
# ---------------------------------------------------------------------------
#
# Problema del sistema anterior:
#   · _sigue_para_adelante solo usa 2D (x,y) y falla cuando la mano está
#     inclinada o rotada en profundidad: el dot-product puede dar positivo con
#     un dedo claramente cerrado si la muñeca apunta hacia la cámara.
#   · El criterio del pulgar (distancia al meñique) falla en posturas oblicuas.
#   · Ninguno de los dos detecta "punta cerca del centro de la palma" ni
#     "punta alineada/encima del nudillo o la falange" — los dos síntomas más
#     obvios de un dedo cerrado.
#
# Nuevo sistema: TRES señales independientes por dedo, votación por mayoría.
#
#   1. DIRECCIÓN (dot-product original, mejorado con z de MediaPipe)
#      v1 = MCP→PIP, v2 = PIP→TIP; si v1·v2 > 0 en los tres ejes → extendido.
#      Usar z reduce falsos positivos con la mano ladeada.
#
#   2. DISTANCIA PUNTA–PALMA
#      Si la punta está a menos de UMBRAL_CERCANIA_PALMA veces el tamaño de la
#      palma (distancia muñeca–nudillo medio), el dedo está cerrado sin importar
#      su dirección.  Es la señal más robusta para el puño.
#
#   3. ALINEACIÓN PUNTA–FALANGE
#      Si la punta queda "atrás" del nudillo intermedio (PIP) proyectada sobre
#      el eje del dedo (es decir, la punta no avanza más allá del PIP en la
#      dirección MCP→PIP), el dedo está definitivamente cerrado.
#
# El pulgar tiene su propio camino: usa las tres señales adaptadas a su
# anatomía (se abre lateralmente, no hacia arriba).
#
# Con 2 o más señales indicando "cerrado" → cerrado; si solo 1 → extendido.
# Esto da histéresis implícita y reduce el parpadeo frame a frame.
# ---------------------------------------------------------------------------

# Fracción del tamaño de la palma por debajo del cual la punta se considera
# "pegada a la palma" → dedo cerrado.  0.65 es conservador: solo fuerza
# cierre cuando la punta está realmente muy cerca del centro.
UMBRAL_CERCANIA_PALMA = 0.65

# Fracción mínima de "avance" de la punta más allá del PIP en la dirección
# del dedo para que cuente como extendido en el criterio de alineación.
# Un valor negativo significa que toleramos que la punta quede un poco detrás
# del PIP (semiflexión) antes de marcarla como cerrada.
UMBRAL_AVANCE_PUNTA = -0.02


def _tamanio_palma(lm):
    """Distancia 3D entre la muñeca (0) y el nudillo del dedo medio (9).
    Sirve como referencia de escala para comparar distancias dentro de la mano.
    Trabaja en coordenadas normalizadas de MediaPipe (x,y ∈ [0,1]; z relativo)."""
    w, m = lm[0], lm[9]
    return math.sqrt((w.x - m.x)**2 + (w.y - m.y)**2 + (w.z - m.z)**2) or 1e-6


def _centro_palma_3d(lm):
    """Centro de la palma en 3D: promedio de muñeca + 4 nudillos MCP."""
    indices = [0, 5, 9, 13, 17]
    cx = sum(lm[i].x for i in indices) / len(indices)
    cy = sum(lm[i].y for i in indices) / len(indices)
    cz = sum(lm[i].z for i in indices) / len(indices)
    return cx, cy, cz


def _dot3(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


def _criterio_direccion(mcp, pip, tip):
    """Señal 1: dot-product en 3D entre v(MCP→PIP) y v(PIP→TIP).
    True = extendido (ambos tramos apuntan al mismo lado)."""
    v1x, v1y, v1z = pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z
    v2x, v2y, v2z = tip.x - pip.x,  tip.y - pip.y,  tip.z - pip.z
    return _dot3(v1x, v1y, v1z, v2x, v2y, v2z) > 0


def _criterio_distancia_palma(tip, centro_palma, escala):
    """Señal 2: distancia 3D de la punta al centro de la palma.
    True = extendido (punta lejos del centro)."""
    cpx, cpy, cpz = centro_palma
    dist = math.sqrt((tip.x - cpx)**2 + (tip.y - cpy)**2 + (tip.z - cpz)**2)
    return dist > UMBRAL_CERCANIA_PALMA * escala


def _criterio_avance(mcp, pip, tip):
    """Señal 3: proyección de (PIP→TIP) sobre el eje del dedo (MCP→PIP).
    True = extendido (la punta avanza más allá del PIP en la dirección del dedo)."""
    # Eje del dedo: vector unitario MCP→PIP
    v1x, v1y, v1z = pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z
    long_v1 = math.sqrt(v1x**2 + v1y**2 + v1z**2) or 1e-9
    ux, uy, uz = v1x / long_v1, v1y / long_v1, v1z / long_v1

    # Proyección de PIP→TIP sobre el eje
    v2x, v2y, v2z = tip.x - pip.x, tip.y - pip.y, tip.z - pip.z
    proyeccion = _dot3(v2x, v2y, v2z, ux, uy, uz)

    # La proyección se normaliza por la longitud del propio tramo MCP→PIP para
    # hacerla independiente del tamaño de la mano en pantalla.
    return (proyeccion / long_v1) > UMBRAL_AVANCE_PUNTA


def _dedo_extendido_largo(mcp_i, pip_i, tip_i, lm, centro_palma, escala):
    """Votación 2-de-3 para dedos largos (índice, medio, anular, meñique).
    Devuelve True si al menos 2 de las 3 señales dicen "extendido"."""
    mcp, pip, tip = lm[mcp_i], lm[pip_i], lm[tip_i]
    s1 = _criterio_direccion(mcp, pip, tip)
    s2 = _criterio_distancia_palma(tip, centro_palma, escala)
    s3 = _criterio_avance(mcp, pip, tip)
    votos_extendido = int(s1) + int(s2) + int(s3)
    return votos_extendido >= 2


def _angulo_ip_pulgar(lm):
    """Ángulo en grados formado en el IP del pulgar (landmark 3), es decir
    el ángulo entre los vectores MCP→IP y IP→TIP.

    Cuando el pulgar está extendido (recto) este ángulo es cercano a 180°.
    Cuando está doblado a 90° o menos el pulgar está cerrado.
    Trabajamos en 3D para que no le importe la orientación de la mano."""
    mcp, ip, tip = lm[2], lm[3], lm[4]

    # Vector MCP→IP (primer tramo)
    v1x, v1y, v1z = ip.x - mcp.x, ip.y - mcp.y, ip.z - mcp.z
    # Vector IP→TIP (segundo tramo)
    v2x, v2y, v2z = tip.x - ip.x,  tip.y - ip.y,  tip.z - ip.z

    long1 = math.sqrt(v1x**2 + v1y**2 + v1z**2) or 1e-9
    long2 = math.sqrt(v2x**2 + v2y**2 + v2z**2) or 1e-9

    cos_ang = _dot3(v1x, v1y, v1z, v2x, v2y, v2z) / (long1 * long2)
    cos_ang = max(-1.0, min(1.0, cos_ang))   # clamp por errores de float
    return math.degrees(math.acos(cos_ang))


# Ángulo mínimo en el IP para considerar el pulgar extendido.
# A 90° el dedo forma una L → cerrado. A 150°+ está casi recto → extendido.
# Se pone en 130° como umbral: tolera una ligera flexión sin marcar como cerrado.
UMBRAL_ANGULO_IP_PULGAR = 130.0

# Factor mínimo: d(TIP, meñique_MCP) / d(IP, meñique_MCP).
# Si la punta está más cerca del meñique que el IP, el pulgar está cruzado/cerrado.
UMBRAL_K_PULGAR = 1.08


def _pulgar_extendido(lm, centro_palma, escala):
    """Criterio para el pulgar basado en las tres señales más confiables,
    identificadas empíricamente:

      P  – distancia 3D punta→centro_palma  (la más confiable)
      K  – ratio d(TIP,meñique) / d(IP,meñique)  (segunda más confiable)
      Â  – ángulo real en el IP del pulgar  (criterio geométrico directo)

    D (dot-product de dirección) y A (avance proyectado) se descartan porque
    el pulgar se mueve lateralmente y esas dos señales casi siempre votan
    "extendido" sin importar la postura real.

    Necesita 2 de 3 para marcar como extendido.
    El ángulo en IP es el árbitro más limpio: si el pulgar forma ~90° entre
    la falange y el nudillo, está cerrado sin importar las otras señales."""

    tip = lm[4]
    ip  = lm[3]

    # ── Señal P: distancia punta → centro palma ──────────────────────────
    cpx, cpy, cpz = centro_palma
    dist_palma = math.sqrt((tip.x - cpx)**2 + (tip.y - cpy)**2 + (tip.z - cpz)**2)
    s_P = dist_palma > 0.52 * escala

    # ── Señal K: ratio distancia al meñique MCP ──────────────────────────
    pinky_mcp = lm[17]
    d_tip = math.hypot(tip.x - pinky_mcp.x, tip.y - pinky_mcp.y)
    d_ip  = math.hypot(ip.x  - pinky_mcp.x, ip.y  - pinky_mcp.y)
    s_K = d_tip > d_ip * UMBRAL_K_PULGAR

    # ── Señal Â: ángulo en el IP (MCP→IP→TIP) ────────────────────────────
    angulo = _angulo_ip_pulgar(lm)
    s_A = angulo > UMBRAL_ANGULO_IP_PULGAR

    votos = int(s_P) + int(s_K) + int(s_A)
    return votos >= 2


def dedos_extendidos(lm):
    """Devuelve lista de 5 booleanos [Pulgar, Índice, Medio, Anular, Meñique].
    True = extendido, False = cerrado.

    Usa votación multi-criterio 2-de-3 (3-de-4 para el pulgar) en espacio 3D,
    lo que lo hace robusto frente a:
      · Mano inclinada o rotada en profundidad.
      · Dedos semiflexionados (se marcan correctamente como cerrados).
      · Punta de dedo que coincide con o queda detrás del nudillo/falange.
      · Puño con los dedos apuntando hacia la cámara.
    """
    escala = _tamanio_palma(lm)
    centro_palma = _centro_palma_3d(lm)

    estados = [_pulgar_extendido(lm, centro_palma, escala)]

    # MCP, PIP, TIP para cada dedo largo
    CADENAS = [
        (5,  6,  8),   # Índice
        (9,  10, 12),  # Medio
        (13, 14, 16),  # Anular
        (17, 18, 20),  # Meñique
    ]
    for mcp_i, pip_i, tip_i in CADENAS:
        estados.append(_dedo_extendido_largo(mcp_i, pip_i, tip_i, lm, centro_palma, escala))

    return estados


def calcular_puntos(lm, ancho, alto):
    """Devuelve, en píxeles: las 5 puntas, los 5 nudillos (MCP), las 5 articulaciones
    intermedias (PIP) y el centro de la palma."""
    puntas_px      = [(int(lm[i].x * ancho), int(lm[i].y * alto)) for i in PUNTAS]
    nudillos_px    = [(int(lm[i].x * ancho), int(lm[i].y * alto)) for i in NUDILLOS]
    intermedias_px = [(int(lm[i].x * ancho), int(lm[i].y * alto)) for i in INTERMEDIAS]

    cx = sum(lm[i].x for i in BASE_PALMA) / len(BASE_PALMA)
    cy = sum(lm[i].y for i in BASE_PALMA) / len(BASE_PALMA)
    centro_px = (int(cx * ancho), int(cy * alto))

    return puntas_px, nudillos_px, intermedias_px, centro_px


def dedos_extendidos_con_votos(lm):
    """Igual que dedos_extendidos() pero devuelve también los votos individuales
    para el overlay de debug.

    Retorna:
        estados : lista de 5 bool (extendido)
        votos   : lista de 5 tuplas.
                  Pulgar → (s_P, s_K, s_Â)      etiquetas: P, K, A
                  Largos → (s_dir, s_dist, s_av) etiquetas: D, P, A
    """
    escala  = _tamanio_palma(lm)
    centro  = _centro_palma_3d(lm)
    estados = []
    votos   = []

    # ── Pulgar ──────────────────────────────────────────────────────────────
    tip = lm[4]
    ip  = lm[3]
    cpx, cpy, cpz = centro
    dist_palma = math.sqrt((tip.x - cpx)**2 + (tip.y - cpy)**2 + (tip.z - cpz)**2)
    s_P = dist_palma > 0.52 * escala
    pinky_mcp = lm[17]
    d_tip = math.hypot(tip.x - pinky_mcp.x, tip.y - pinky_mcp.y)
    d_ip  = math.hypot(ip.x  - pinky_mcp.x, ip.y  - pinky_mcp.y)
    s_K   = d_tip > d_ip * UMBRAL_K_PULGAR
    s_Ang = _angulo_ip_pulgar(lm) > UMBRAL_ANGULO_IP_PULGAR
    v_pulgar = int(s_P) + int(s_K) + int(s_Ang)
    estados.append(v_pulgar >= 2)
    votos.append((s_P, s_K, s_Ang))

    # ── Dedos largos ────────────────────────────────────────────────────────
    CADENAS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    for mcp_i, pip_i, tip_i in CADENAS:
        m, p, t = lm[mcp_i], lm[pip_i], lm[tip_i]
        s1 = _criterio_direccion(m, p, t)
        s2 = _criterio_distancia_palma(t, centro, escala)
        s3 = _criterio_avance(m, p, t)
        v  = int(s1) + int(s2) + int(s3)
        estados.append(v >= 2)
        votos.append((s1, s2, s3))

    return estados, votos


def _dibujar_votos_debug(panel, puntas_px, votos):
    """Superpone los votos de cada dedo encima de su punta en el panel.
    Pulgar: P / K / Â  |  Largos: D / P / A"""
    etiquetas_pulgar = ["P", "K", "^"]   # P=palma, K=meñique, ^=ángulo IP
    etiquetas_largo  = ["D", "P", "A"]   # D=dirección, P=palma, A=avance
    for i, (punta, voto) in enumerate(zip(puntas_px, votos)):
        px, py = punta
        labels = etiquetas_pulgar if i == 0 else etiquetas_largo
        for j, v in enumerate(voto):
            color = (0, 220, 0) if v else (0, 0, 220)
            x0 = px + j * 8
            cv2.rectangle(panel, (x0, py - 22), (x0 + 6, py - 12), color, -1)
            cv2.putText(panel, labels[j], (x0, py - 24),
                        cv2.FONT_HERSHEY_PLAIN, 0.6, color, 1)


# ---------------------------------------------------------------------------
# Variable global: modo debug (toggle con tecla D)
# ---------------------------------------------------------------------------
_debug_votos = False


def dibujar_panel(manos, debug=False):
    """Dibuja el entorno virtual y, sobre él, todas las manos detectadas.

    `manos` es una lista de tuplas:
        (puntas_px, nudillos_px, intermedias_px, centro_px, estados, etiqueta, votos,
         normal_palma, prof_palma)
    donde `votos` puede ser None si no se calculó el modo debug, `normal_palma` es el
    vector normal unitario de la palma (para rotación), y `prof_palma` es el z de
    MediaPipe del centro de palma (para arrastre en Z).
    """
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    todos_los_puntos = []
    info_gestos = []
    for puntas_px, nudillos_px, intermedias_px, centro_px, estados, etiqueta, _, normal_palma, prof_palma in manos:
        puntos_mano = puntas_px + nudillos_px + intermedias_px + [centro_px]
        todos_los_puntos += puntos_mano
        info_gestos.append({
            "etiqueta":    etiqueta,
            "puntos":      puntos_mano,
            "centro":      centro_px,
            "cerrada":     not any(estados),
            "estados":     estados,   # [Pulgar, Índice, Medio, Anular, Meñique] -> gestos de zoom/rotar
            "normal":      normal_palma,
            "profundidad": prof_palma,
        })

    entorno.actualizar(todos_los_puntos)
    entorno.actualizar_gestos(info_gestos)
    renderizar_entorno(panel)

    for puntas_px, nudillos_px, intermedias_px, centro_px, estados, etiqueta, votos, _, __ in manos:
        for nudillo, intermedia, punta, nombre, extendido in zip(
            nudillos_px, intermedias_px, puntas_px, NOMBRES_DEDOS, estados
        ):
            cv2.line(panel, centro_px, nudillo, (0, 200, 0), 2)
            cv2.line(panel, nudillo,   intermedia, (0, 200, 0), 2)
            cv2.line(panel, intermedia, punta, (0, 200, 0), 2)

            color_punta = (0, 255, 0) if extendido else (0, 0, 255)
            cv2.circle(panel, nudillo,    6, (200, 200, 200), -1)
            cv2.circle(panel, intermedia, 6, (0, 165, 255),   -1)
            cv2.circle(panel, punta,      8, color_punta,     -1)
            cv2.putText(panel, nombre[0], (punta[0] - 5, punta[1] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if debug and votos is not None:
            _dibujar_votos_debug(panel, puntas_px, votos)

        cv2.circle(panel, centro_px, 10, (255, 0, 0), -1)
        cv2.putText(panel, etiqueta,
                    (centro_px[0] - 20, centro_px[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Indicador de modo debug en la esquina
    modo_txt = "[D] DEBUG ON  (D=off)" if debug else "[D] debug off (D=on)"
    cv2.putText(panel, modo_txt, (10, PANEL_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(panel, "Entorno virtual", (10, PANEL_H - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return panel


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

# Columnas en X donde se imprime el texto de estado de cada mano sobre la imagen de la
# cámara, para que el texto de la primera y la segunda mano no se superpongan.
COLUMNAS_TEXTO = [20, 340]

# Objeto CLAHE — se instancia una sola vez fuera del bucle (es costoso crear).
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

# Detector de fallback: detección pura sin tracking (se crea una sola vez).
hands_fallback = mp_hands.Hands(
    static_image_mode=True,       # detecta desde cero en cada frame
    max_num_hands=2,
    model_complexity=MP_COMPLEJIDAD,
    min_detection_confidence=MP_CONF_FALLBACK,
    min_tracking_confidence=1.0,  # ignorado en static_image_mode
)

# LUT de corrección gamma precomputada (operación O(1) por frame via cv2.LUT).
_gamma_lut = np.array(
    [(i / 255.0) ** (1.0 / GAMMA_FALLBACK) * 255 for i in range(256)],
    dtype=np.uint8,
)

# Estado de persistencia de landmarks
_ghost_landmarks  = None   # último multi_hand_landmarks válido
_ghost_handedness = None   # último multi_handedness válido
_ghost_contador   = 0      # frames consecutivos sin detección real
_en_ghost         = False  # indicador de si estamos en "ghost mode"

# Tamaño del frame ya con el margen de detección sumado (constante: se calcula
# una sola vez fuera del bucle).
_pad_x = int(MP_ANCHO * MARGEN_DETECCION)
_pad_y = int(MP_ALTO  * MARGEN_DETECCION)
_mp_ancho_pad = MP_ANCHO + 2 * _pad_x
_mp_alto_pad  = MP_ALTO  + 2 * _pad_y

# Para el contador de FPS real (no el nominal de la cámara)
_t_anterior = time.time()

while True:
    frame = captura_hilo.leer()
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)

    # Reducir a la resolución de trabajo de MediaPipe ANTES de cualquier otro
    # procesamiento: esto es lo que baja el costo por frame y sube los FPS
    # reales (ver comentario de MP_ANCHO/MP_ALTO más arriba). El frame a
    # resolución completa (`frame`, CAM_ANCHO x CAM_ALTO) se conserva intacto
    # para el overlay visible, que se ve nítido independientemente de a qué
    # tamaño corrió la inferencia.
    frame_mp_chico = cv2.resize(frame, (MP_ANCHO, MP_ALTO), interpolation=cv2.INTER_AREA)

    # CLAHE: realce de contraste local solo para MediaPipe
    # Convierte a LAB, realza el canal L y vuelve a BGR. El frame original
    # (sin CLAHE) se conserva para el overlay visible → el video se ve natural.
    lab = cv2.cvtColor(frame_mp_chico, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = _clahe.apply(l_ch)
    frame_mp = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # Head tracking (vision3d): se corre sobre frame_mp SIN el padding que se
    # agrega abajo para manos — la cara no lo necesita, y correrlo antes evita
    # que el borde negro le reste resolución útil a la cara. cabeza_cm queda
    # actualizado en la variable global que usa renderizar_entorno().
    rgb_cara = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
    cabeza_cm = rastreador_cabeza.procesar(rgb_cara)

    # Margen de detección: borde negro alrededor de la imagen de trabajo
    # (ver MARGEN_DETECCION más arriba) para que la mano nunca ocupe el
    # frame de punta a punta, aunque esté muy cerca de la cámara.
    frame_mp_pad = cv2.copyMakeBorder(
        frame_mp, _pad_y, _pad_y, _pad_x, _pad_x,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    rgb = cv2.cvtColor(frame_mp_pad, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Fallback: si el tracker principal no encontró nada y llevamos
    # FRAMES_ANTES_FALLBACK frames consecutivos sin detección, intentar con
    # el detector estático + imagen con gamma boost.
    # _ghost_contador guarda cuántos frames SIN detección hubo ANTES de este
    # frame, así que >= FRAMES_ANTES_FALLBACK activa a partir del 2.º miss.
    if not results.multi_hand_landmarks and _ghost_contador >= FRAMES_ANTES_FALLBACK:
        frame_fb_pad = cv2.LUT(frame_mp_pad, _gamma_lut)   # aclarar imagen
        rgb_fb   = cv2.cvtColor(frame_fb_pad, cv2.COLOR_BGR2RGB)
        results_fb = hands_fallback.process(rgb_fb)
        if results_fb.multi_hand_landmarks:
            results = results_fb                           # usar resultado del fallback

    # Deshacer el margen de detección: los landmarks que devuelve MediaPipe
    # están normalizados respecto al frame CON el borde negro sumado. Los
    # reescribimos en lugar para que queden normalizados respecto al frame
    # de trabajo original (sin margen) — así el resto del código (que asume
    # coordenadas normalizadas [0,1] del frame "real") no se entera de que
    # existió ningún padding.
    if results.multi_hand_landmarks:
        for _mano_lm in results.multi_hand_landmarks:
            for _punto in _mano_lm.landmark:
                _punto.x = (_punto.x * _mp_ancho_pad - _pad_x) / MP_ANCHO
                _punto.y = (_punto.y * _mp_alto_pad  - _pad_y) / MP_ALTO

    # Ghost frames: persistencia cuando MediaPipe no devuelve manos
    if results.multi_hand_landmarks:
        _ghost_landmarks  = results.multi_hand_landmarks
        _ghost_handedness = results.multi_handedness
        _ghost_contador   = 0
        _en_ghost         = False
    else:
        _ghost_contador += 1
        if _ghost_contador <= MAX_GHOST_FRAMES and _ghost_landmarks is not None:
            _en_ghost = True
        else:
            _en_ghost         = False
            _ghost_landmarks  = None
            _ghost_handedness = None

    landmarks_ef  = _ghost_landmarks  if _en_ghost else results.multi_hand_landmarks
    handedness_ef = _ghost_handedness if _en_ghost else results.multi_handedness

    # Reducir frame para display (después de MediaPipe)
    frame = cv2.resize(frame, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
    alto, ancho = frame.shape[:2]
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    # Contador de FPS real (no el nominal de CAM_FPS): mide el tiempo entre
    # iteraciones del bucle, así que refleja cuello de botella real (cámara +
    # CLAHE + MediaPipe + dibujo), útil para verificar que las optimizaciones
    # de arriba realmente están subiendo los FPS en este equipo en particular.
    _t_actual = time.time()
    _dt = _t_actual - _t_anterior
    _fps = 1.0 / max(_dt, 1e-6)
    _t_anterior = _t_actual
    cv2.putText(frame, f"FPS: {_fps:4.1f}", (PANEL_W - 130, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Calidad dinámica: se alimenta con el dt real de este frame y, si el
    # nivel cambió, reasigna las mallas de las figuras antes de dibujar
    # (ver sincronizar_calidad_entorno — no decima nada acá, solo elige).
    gestor_calidad.registrar_frame(_dt)
    opt_obj.sincronizar_calidad_entorno(entorno, gestor_calidad)
    cv2.putText(frame, f"Calidad: {gestor_calidad.nivel_actual}", (PANEL_W - 130, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)

    # Indicador visual de ghost mode en la esquina del frame de cámara
    if _en_ghost:
        cv2.circle(frame, (PANEL_W - 18, 18), 8, (0, 140, 255), -1)

    # Consumir figuras recién generadas por la IA y agregarlas al entorno
    while _cola_figuras:
        nombre, datos, pedido_usuario, malla_info = _cola_figuras.pop(0)
        ubicado = ubicacion.ubicar_y_registrar(nombre, datos, pedido_usuario)
        if malla_info:
            # HIT de biblioteca: se dibuja con la Malla real, no con el bbox
            # placeholder ("esfera") que ubicacion.py usó solo para calcular
            # dónde va (ver objetos.py::_figura_placeholder_desde_malla).
            centro_relativo = ubicado["_ubicacion"]["centro"]
            malla_lod_baja = malla_module.Malla.from_dict(malla_info["lod_bajo"])
            malla_lod_alta = (malla_module.Malla.from_dict(malla_info["lod_alto"])
                               if malla_info.get("lod_alto") else None)
            entorno.agregar_figura_desde_malla(
                centro_relativo, malla_lod_baja, malla_lod_alta, nombre=nombre,
                radio_bounding_relativo=malla_info.get("radio_bounding"),
            )
        else:
            entorno.agregar_figura(ubicado["puntos"], ubicado["conexiones"], primitivas_relativas=ubicado["primitivas"], nombre=nombre)

    # Consumir propiedades físicas que llegaron en el paso 2 y asignarlas a la figura
    while _cola_propiedades:
        nombre, propiedades = _cola_propiedades.pop(0)
        entorno.asignar_propiedades_a_figura(nombre, propiedades)

    # Consumir mallas de IA que terminaron en background (sin HIT de biblioteca
    # al crear el objeto) y reemplazar la figura primitiva por la malla real.

    if landmarks_ef:
        etiquetas_handedness = []
        if handedness_ef:
            etiquetas_handedness = [h.classification[0].label for h in handedness_ef]

        manos_panel = []
        for idx, hand_landmarks in enumerate(landmarks_ef):
            lm = hand_landmarks.landmark

            etiqueta = etiquetas_handedness[idx] if idx < len(etiquetas_handedness) else f"Mano {idx + 1}"
            etiqueta = {"Left": "Izquierda", "Right": "Derecha"}.get(etiqueta, etiqueta)

            # En modo debug calculamos los votos individuales para mostrarlos;
            # en modo normal usamos la versión más liviana sin overhead extra.
            if _debug_votos:
                estados, votos = dedos_extendidos_con_votos(lm)
            else:
                estados = dedos_extendidos(lm)
                votos   = None

            col_x = COLUMNAS_TEXTO[idx] if idx < len(COLUMNAS_TEXTO) else COLUMNAS_TEXTO[-1]
            cv2.putText(frame, etiqueta, (col_x, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            for i, (nombre, extendido) in enumerate(zip(NOMBRES_DEDOS, estados)):
                texto = f"{nombre}: {'LEVANTADO' if extendido else 'CERRADO'}"
                color = (0, 255, 0) if extendido else (0, 0, 255)
                cv2.putText(frame, texto, (col_x, 50 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            puntas_px, nudillos_px, intermedias_px, centro_px = calcular_puntos(lm, ancho, alto)
            for p, extendido in zip(puntas_px, estados):
                color = (0, 255, 0) if extendido else (0, 0, 255)
                cv2.circle(frame, p, 6, color, -1)
            for p in nudillos_px:
                cv2.circle(frame, p, 4, (200, 200, 200), -1)
            for p in intermedias_px:
                cv2.circle(frame, p, 4, (0, 165, 255), -1)
            cv2.circle(frame, centro_px, 8, (255, 0, 0), -1)

            puntas_panel, nudillos_panel, intermedias_panel, centro_panel = \
                calcular_puntos(lm, PANEL_W, PANEL_H)

            # Normal de la palma en espacio de cámara (para el gesto de rotación).
            # Se calcula con el producto cruz de dos vectores del esqueleto de la mano:
            #   v1 = lm[5]  - lm[0]   (muñeca → nudillo índice)
            #   v2 = lm[17] - lm[0]   (muñeca → nudillo meñique)
            # El cruce v1×v2 apunta hacia/desde la cámara dependiendo de qué cara
            # de la mano mira; se normaliza y se pasa al entorno para detectar cuánto
            # giró la palma entre frames.
            _lm0  = lm[0];  _lm5  = lm[5];  _lm17 = lm[17]
            _v1 = (_lm5.x  - _lm0.x, _lm5.y  - _lm0.y, _lm5.z  - _lm0.z)
            _v2 = (_lm17.x - _lm0.x, _lm17.y - _lm0.y, _lm17.z - _lm0.z)
            _nx = _v1[1]*_v2[2] - _v1[2]*_v2[1]
            _ny = _v1[2]*_v2[0] - _v1[0]*_v2[2]
            _nz = _v1[0]*_v2[1] - _v1[1]*_v2[0]
            _nn = math.sqrt(_nx*_nx + _ny*_ny + _nz*_nz)
            if _nn > 1e-9:
                _nx /= _nn; _ny /= _nn; _nz /= _nn
            normal_palma = (_nx, _ny, _nz)

            # Profundidad de palma: z de MediaPipe del punto 9 (MCP dedo medio),
            # el más estable del centro de la palma. Escala relativa; solo importan
            # las diferencias entre frames para el arrastre en Z.
            prof_palma = lm[9].z

            manos_panel.append(
                (puntas_panel, nudillos_panel, intermedias_panel,
                 centro_panel, estados, etiqueta, votos, normal_palma, prof_palma)
            )

        panel = dibujar_panel(manos_panel, debug=_debug_votos)
    else:
        entorno.actualizar([])
        entorno.actualizar_gestos([])
        renderizar_entorno(panel)
        modo_txt = "[D] DEBUG ON  (D=off)" if _debug_votos else "[D] debug off (D=on)"
        cv2.putText(panel, modo_txt, (10, PANEL_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(panel, "Entorno virtual", (10, PANEL_H - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if panel.shape[0] != frame.shape[0]:
        panel = cv2.resize(panel, (PANEL_W, frame.shape[0]))

    combinado = cv2.hconcat([frame, panel])
    cv2.imshow("Mano", combinado)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:        # ESC → salir
        break
    elif key == ord('d') or key == ord('D'):
        _debug_votos = not _debug_votos
        print(f"[debug] modo votos {'ON' if _debug_votos else 'OFF'}")
    elif key == ord('e') or key == ord('E'):
        _modo_vision3d = None if _modo_vision3d == vision3d.MODO_ANAGLIFO else vision3d.MODO_ANAGLIFO
        print(f"[vision3d] modo anaglifo {'ON' if _modo_vision3d == vision3d.MODO_ANAGLIFO else 'OFF'}")
    elif key == ord('p') or key == ord('P'):
        _modo_vision3d = None if _modo_vision3d == vision3d.MODO_OFFAXIS else vision3d.MODO_OFFAXIS
        print(f"[vision3d] modo off-axis {'ON' if _modo_vision3d == vision3d.MODO_OFFAXIS else 'OFF'}")

captura_hilo.detener()
rastreador_cabeza.cerrar()
cv2.destroyAllWindows()