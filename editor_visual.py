"""
editor_visual.py — Interfaz visual futurista para editar propiedades, tamaño y
color de los objetos del catálogo (objetos.py / objetos_db.json).

Qué hace:
    - Lista los objetos guardados. Al hacer clic en uno, muestra:
        * una previsualización 3D en vivo (wireframe, orbitable con el mouse)
          armada a partir de la MISMA geometría que ya usa el entorno virtual
          (figura["puntos"]/["conexiones"]/["primitivas"]);
        * al lado, un panel con la ficha de propiedades físicas (objetos.CAMPOS,
          las mismas 14 claves que ya genera la IA) y una sección de apariencia
          (color, escala por eje) para editar todo desde un solo lugar.
    - Guarda los cambios de propiedades físicas con objetos.guardar_objeto()
      (mismo mecanismo que usa objetos.crear_objeto), sin tocar el esquema existente.
    - Guarda color/escala en una capa separada (editor_visual_estilos.json) para
      NO modificar objetos_db.json ni el prompt que le pide la ficha física al
      modelo — así este módulo se agrega sin tocar un solo archivo existente.
    - Reutiliza las fábricas de primitivas de malla.py (malla_cubo,
      malla_esfera, etc. — las mismas que usa render_malla.py en el
      entorno real) para que la previsualización de las primitivas
      heredadas coincida con lo que se dibuja en la escena.
    - Si el objeto ya tiene una Malla real archivada (biblioteca_mallas.py
      / malla_ia_async.py, ver PLAN_RECONSTRUCCION_MALLAS.md sección 5),
      la previsualización usa esa geometría (LOD bajo) en vez del
      wireframe heredado de puntos/primitivas.

Standalone:
    python editor_visual.py

Integración opcional (no aplicada acá, un solo import + una línea):
    import editor_visual
    editor_visual.abrir_editor_visual(root)   # abre como Toplevel desde otro panel tkinter

Requiere lo mismo que objetos.py (Ollama corriendo, el modelo configurado) SOLO
para "Actualizar con IA…"; para editar y previsualizar a mano no hace falta.
"""

import json
import math
import os
import threading
import tkinter as tk
from tkinter import colorchooser, messagebox, ttk

import objetos as obj
from ui_thread import en_hilo_ui
import malla as malla_mod


# ---------------------------------------------------------------------------
# Paleta futurista — índigo / cian / naranja sobre fondo casi negro
# ---------------------------------------------------------------------------
BG_FONDO      = "#05060c"
BG_PANEL      = "#0b0f1e"
BG_PANEL_2    = "#10182e"
BG_CAMPO      = "#0d1426"
BORDE         = "#1c2a4a"
BORDE_SUAVE   = "#141d38"

INDIGO        = "#6366f1"
INDIGO_SUAVE  = "#4338ca"
INDIGO_OSCURO = "#1e1b4b"
CIAN          = "#22d3ee"
CIAN_SUAVE    = "#0e7490"
NARANJA       = "#fb923c"
NARANJA_SUAVE = "#c2410c"

TEXTO         = "#e7e9f5"
TEXTO_MUTED   = "#7c85a3"
TEXTO_TENUE   = "#4b5470"

FUENTE          = ("Segoe UI", 9)
FUENTE_ETQ      = ("Segoe UI", 9)
FUENTE_TITULO   = ("Segoe UI", 14, "bold")
FUENTE_SUBTIT   = ("Segoe UI", 9)
FUENTE_SECCION  = ("Segoe UI", 10, "bold")
FUENTE_MONO     = ("Consolas", 9)
FUENTE_MONO_B   = ("Consolas", 10, "bold")


# ---------------------------------------------------------------------------
# Utilidades de color
# ---------------------------------------------------------------------------

def _hex_a_rgb(hexcolor: str) -> tuple:
    hexcolor = hexcolor.lstrip("#")
    if len(hexcolor) != 6:
        hexcolor = "22D3EE"
    return tuple(int(hexcolor[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_a_hex(rgb: tuple) -> str:
    r, g, b = (max(0, min(255, int(c))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def _ajustar_brillo(hexcolor: str, factor: float) -> str:
    """factor < 1 oscurece, > 1 aclara (con tope en 0..255). Se usa para el
    halo del efecto 'glow' y para los estados hover de los botones."""
    r, g, b = _hex_a_rgb(hexcolor)
    return _rgb_a_hex((r * factor, g * factor, b * factor))


def color_bgr_para_opencv(nombre: str) -> tuple:
    """Devuelve el color guardado para `nombre` como tupla BGR (uint8-friendly),
    lista para usarse como color_normal de una Figura3D (figura.py).
    Punto de integración opcional: no se usa desde acá adentro."""
    estilo = obtener_estilo(nombre)
    r, g, b = _hex_a_rgb(estilo.get("color_hex", "#22D3EE"))
    return (b, g, r)


# ---------------------------------------------------------------------------
# Persistencia del estilo visual (color + escala) — capa separada, no toca
# objetos_db.json. Se indexa por el mismo nombre que usa objetos.py.
# ---------------------------------------------------------------------------

ESTILO_PATH = "editor_visual_estilos.json"

ESTILO_DEFECTO = {
    "color_hex": "#22D3EE",
    "escala_x": 1.0,
    "escala_y": 1.0,
    "escala_z": 1.0,
}


def _cargar_estilos() -> dict:
    if not os.path.exists(ESTILO_PATH):
        return {}
    try:
        with open(ESTILO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[editor_visual] Error al leer {ESTILO_PATH}: {e}")
        return {}


def _guardar_estilos(db: dict) -> None:
    try:
        with open(ESTILO_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[editor_visual] Error al guardar {ESTILO_PATH}: {e}")


def obtener_estilo(nombre: str) -> dict:
    db = _cargar_estilos()
    estilo = dict(ESTILO_DEFECTO)
    estilo.update(db.get(nombre, {}))
    return estilo


def guardar_estilo(nombre: str, estilo: dict) -> None:
    db = _cargar_estilos()
    db[nombre] = {
        "color_hex": estilo.get("color_hex", ESTILO_DEFECTO["color_hex"]),
        "escala_x": float(estilo.get("escala_x", 1.0)),
        "escala_y": float(estilo.get("escala_y", 1.0)),
        "escala_z": float(estilo.get("escala_z", 1.0)),
    }
    _guardar_estilos(db)


def eliminar_estilo(nombre: str) -> None:
    db = _cargar_estilos()
    if nombre in db:
        del db[nombre]
        _guardar_estilos(db)


# ---------------------------------------------------------------------------
# Geometría de previsualización: reduce cualquier figura (puntos+conexiones,
# primitivas heredadas y/o Malla real de biblioteca/IA) a UNA sola lista de
# vértices + aristas, centrada en el origen y normalizada a una caja de ±1,
# lista para orbitar en el canvas.
# Reusa las mismas fábricas de malla.py que usa el entorno real (vía
# render_malla.py) para que la previsualización de primitivas heredadas sea
# fiel a lo que se ve en la escena. Si el objeto ya tiene una Malla real
# archivada (biblioteca_mallas.py / malla_ia_async.py, formato de
# optimizacion_malla.serializar_json), se usa esa en vez del wireframe
# heredado — ver PLAN_RECONSTRUCCION_MALLAS.md, sección 5.
# ---------------------------------------------------------------------------

def _figura_a_puntos_aristas(figura: dict, malla_real: dict = None) -> tuple:
    puntos = []
    aristas = []

    for p in figura.get("puntos") or []:
        if len(p) >= 3:
            x, y, z = p[0], p[1], p[2]
        elif len(p) == 2:
            x, y, z = p[0], p[1], 0.5
        else:
            continue
        puntos.append((float(x), float(y), float(z)))

    n_base = len(puntos)
    for c in figura.get("conexiones") or []:
        if len(c) == 2 and 0 <= c[0] < n_base and 0 <= c[1] < n_base:
            aristas.append((int(c[0]), int(c[1])))

    for prim in figura.get("primitivas") or []:
        tipo = prim.get("tipo")
        cz = prim.get("cz", 0.5)
        try:
            if tipo == "circulo":
                cx, cy = prim["cx"], prim["cy"]
                m = malla_mod.malla_anillo(prim["r"], prim["r"])
            elif tipo == "elipse":
                cx, cy = prim["cx"], prim["cy"]
                m = malla_mod.malla_anillo(prim["rx"], prim["ry"])
            elif tipo == "rectangulo":
                cx = prim["x"] + prim["ancho"] / 2.0
                cy = prim["y"] + prim["alto"] / 2.0
                m = malla_mod.malla_rectangulo(prim["ancho"], prim["alto"])
            elif tipo == "esfera":
                cx, cy = prim["cx"], prim["cy"]
                m = malla_mod.malla_esfera(prim["r"])
            elif tipo == "cubo":
                cx, cy = prim["cx"], prim["cy"]
                m = malla_mod.malla_cubo(
                    prim["ancho"], prim["alto"], prim.get("profundo", prim["ancho"])
                )
            elif tipo == "cilindro":
                cx, cy = prim["cx"], prim["cy"]
                m = malla_mod.malla_cilindro(prim["r"], prim["alto"])
            else:
                continue
        except (KeyError, TypeError):
            continue

        offset = len(puntos)
        for lx, ly, lz in m.vertices:
            puntos.append((cx + lx, cy + ly, cz + lz))
        aristas += [(offset + a, offset + b) for a, b in m.aristas]

    # Malla real (biblioteca/IA) — fuente PRINCIPAL de geometría en el
    # entorno actual (ver biblioteca_mallas.py / malla_ia_async.py). Vive
    # en el registro AL LADO de "figura" (registro["malla"]), no adentro
    # — cuando hay HIT de biblioteca, "figura" queda como el bbox
    # placeholder tipo "esfera" que usó ubicacion.py, y la Malla real de
    # verdad es esta. LOD bajo alcanza para orbitar a mano en este canvas
    # 2D y es más liviano que el LOD alto (que en el entorno real solo se
    # usa con la figura agarrada). Formato: el mismo dict de
    # optimizacion_malla.serializar_json() ({"lod_bajo": {"v":[...],
    # "f":[...]}, "lod_alto": ... | None, ...}).
    if malla_real:
        datos_lod = malla_real.get("lod_bajo") or malla_real.get("lod_alto")
        if datos_lod:
            try:
                m = malla_mod.Malla.from_dict(datos_lod)
                offset = len(puntos)
                puntos += list(m.vertices)
                aristas += [(offset + a, offset + b) for a, b in m.aristas]
            except (KeyError, TypeError, ValueError):
                pass

    return puntos, aristas


def _normalizar_para_vista(puntos: list) -> list:
    """Centra en el origen y escala para que la figura entera quepa en una
    caja de radio 1, sin importar en qué unidades venían los puntos."""
    if not puntos:
        return []
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    zs = [p[2] for p in puntos]
    cx, cy, cz = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0
    radio = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-6) / 2.0
    return [((x - cx) / radio, (y - cy) / radio, (z - cz) / radio) for x, y, z in puntos]


# ---------------------------------------------------------------------------
# Widget: previsualización 3D orbitable (canvas puro, sin dependencias extra)
# ---------------------------------------------------------------------------

class VistaPrevia3D(tk.Canvas):
    """Wireframe orbitable a mano (clic + arrastre) con auto-rotación opcional
    y efecto 'glow' (halo difuso detrás de cada arista, estilo HUD)."""

    def __init__(self, parent, ancho=380, alto=380, **kw):
        super().__init__(
            parent, width=ancho, height=alto, bg=BG_PANEL,
            highlightthickness=0, bd=0, **kw
        )
        self.ancho, self.alto = ancho, alto
        self._puntos = []
        self._aristas = []
        self._color = CIAN
        self._escala = (1.0, 1.0, 1.0)
        self._ang_x = -0.45
        self._ang_y = 0.7
        self._zoom = 1.0
        self._arrastrando = False
        self._ultimo_xy = (0, 0)
        self._nombre = ""

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>", self._on_zoom)      # Windows / macOS
        self.bind("<Button-4>", self._on_zoom)         # Linux (scroll up)
        self.bind("<Button-5>", self._on_zoom)         # Linux (scroll down)

        self._dibujar_vacio()

    # -- API pública ---------------------------------------------------------

    def cargar(self, nombre: str, figura: dict, color_hex: str, escala_xyz: tuple,
               malla_real: dict = None):
        self._nombre = nombre
        puntos_crudos, self._aristas = _figura_a_puntos_aristas(figura or {}, malla_real)
        self._puntos = _normalizar_para_vista(puntos_crudos)
        self._color = color_hex
        self._escala = escala_xyz
        self._zoom = 1.0
        self.redibujar()

    def actualizar_apariencia(self, color_hex: str, escala_xyz: tuple):
        self._color = color_hex
        self._escala = escala_xyz
        self.redibujar()

    def limpiar(self):
        self._nombre = ""
        self._puntos, self._aristas = [], []
        self._dibujar_vacio()

    # -- interacción -----------------------------------------------------------

    def _on_press(self, evento):
        self._arrastrando = True
        self._ultimo_xy = (evento.x, evento.y)

    def _on_drag(self, evento):
        if not self._arrastrando:
            return
        dx = evento.x - self._ultimo_xy[0]
        dy = evento.y - self._ultimo_xy[1]
        self._ultimo_xy = (evento.x, evento.y)
        self._ang_y += dx * 0.01
        self._ang_x += dy * 0.01
        self._ang_x = max(-1.5, min(1.5, self._ang_x))
        self.redibujar()

    def _on_release(self, _evento):
        self._arrastrando = False

    def _on_zoom(self, evento):
        delta = getattr(evento, "delta", 0)
        if delta > 0 or getattr(evento, "num", None) == 4:
            self._zoom = min(3.0, self._zoom * 1.08)
        else:
            self._zoom = max(0.35, self._zoom / 1.08)
        self.redibujar()

    def girar_auto(self, paso: float = 0.012):
        if not self._arrastrando:
            self._ang_y += paso
            self.redibujar()

    # -- proyección y dibujo --------------------------------------------------

    def _proyectar(self, punto):
        x, y, z = punto
        ex, ey, ez = self._escala
        x, y, z = x * ex, y * ey, z * ez

        cy_, sy_ = math.cos(self._ang_y), math.sin(self._ang_y)
        x, z = x * cy_ + z * sy_, -x * sy_ + z * cy_

        cx_, sx_ = math.cos(self._ang_x), math.sin(self._ang_x)
        y, z = y * cx_ - z * sx_, y * sx_ + z * cx_

        distancia = 3.4
        factor = distancia / (distancia + z) if (distancia + z) > 0.1 else distancia
        escala_pantalla = min(self.ancho, self.alto) * 0.30 * self._zoom
        sx = self.ancho / 2 + x * escala_pantalla * factor
        sy = self.alto / 2 - y * escala_pantalla * factor
        return sx, sy, z

    def _dibujar_vacio(self):
        self.delete("all")
        self._dibujar_rejilla()
        self.create_text(
            self.ancho / 2, self.alto / 2,
            text="Seleccioná un objeto\npara previsualizarlo",
            fill=TEXTO_MUTED, font=FUENTE, justify="center",
        )

    def _dibujar_rejilla(self):
        paso = 24
        for x in range(0, self.ancho, paso):
            self.create_line(x, 0, x, self.alto, fill=BORDE_SUAVE)
        for y in range(0, self.alto, paso):
            self.create_line(0, y, self.ancho, y, fill=BORDE_SUAVE)
        cx, cy = self.ancho / 2, self.alto / 2
        self.create_line(cx - 14, cy, cx + 14, cy, fill=INDIGO_SUAVE)
        self.create_line(cx, cy - 14, cx, cy + 14, fill=INDIGO_SUAVE)

    def redibujar(self):
        self.delete("all")
        self._dibujar_rejilla()

        if not self._puntos or not self._aristas:
            if self._nombre:
                self.create_text(
                    self.ancho / 2, self.alto / 2,
                    text=f"'{self._nombre}' no tiene\ngeometría para mostrar",
                    fill=TEXTO_MUTED, font=FUENTE, justify="center",
                )
            return

        proyectados = [self._proyectar(p) for p in self._puntos]

        aristas_z = []
        for a, b in self._aristas:
            if a >= len(proyectados) or b >= len(proyectados):
                continue
            pa, pb = proyectados[a], proyectados[b]
            z_prom = (pa[2] + pb[2]) / 2.0
            aristas_z.append((z_prom, pa, pb))
        aristas_z.sort(key=lambda t: t[0])  # atrás primero, adelante al final

        halo = _ajustar_brillo(self._color, 0.35)
        for _z, pa, pb in aristas_z:
            self.create_line(pa[0], pa[1], pb[0], pb[1], fill=halo, width=5,
                              capstyle=tk.ROUND, smooth=False)
        for _z, pa, pb in aristas_z:
            self.create_line(pa[0], pa[1], pb[0], pb[1], fill=self._color, width=2,
                              capstyle=tk.ROUND, smooth=False)

        # vértices como puntos leves (naranja) para dar sensación de "nodos"
        for sx, sy, _sz in proyectados:
            self.create_oval(sx - 2, sy - 2, sx + 2, sy + 2, fill=NARANJA, outline="")


# ---------------------------------------------------------------------------
# Helpers de UI: botón plano con hover, separador con acento de color,
# encabezado de sección tipo HUD.
# ---------------------------------------------------------------------------

def crear_boton(parent, texto, comando, color=INDIGO, color_texto="#05060c",
                 ancho=None, relleno=(14, 8)):
    btn = tk.Label(
        parent, text=texto, bg=color, fg=color_texto, font=FUENTE_SECCION,
        cursor="hand2", padx=relleno[0], pady=relleno[1],
    )
    if ancho:
        btn.config(width=ancho)
    color_hover = _ajustar_brillo(color, 1.25)
    btn.bind("<Enter>", lambda _e: btn.config(bg=color_hover))
    btn.bind("<Leave>", lambda _e: btn.config(bg=color))
    btn.bind("<Button-1>", lambda _e: comando())
    return btn


def crear_boton_fantasma(parent, texto, comando, color=CIAN):
    """Botón secundario: borde de color, fondo transparente (mismo que el panel)."""
    btn = tk.Label(
        parent, text=texto, bg=BG_PANEL_2, fg=color, font=FUENTE_ETQ,
        cursor="hand2", padx=10, pady=6,
        highlightbackground=color, highlightthickness=1,
    )
    btn.bind("<Enter>", lambda _e: btn.config(bg=_ajustar_brillo(BG_PANEL_2, 1.6)))
    btn.bind("<Leave>", lambda _e: btn.config(bg=BG_PANEL_2))
    btn.bind("<Button-1>", lambda _e: comando())
    return btn


def encabezado_seccion(parent, texto, color=CIAN):
    marco = tk.Frame(parent, bg=BG_PANEL)
    marco.pack(fill=tk.X, pady=(14, 6))
    tk.Frame(marco, bg=color, width=3, height=16).pack(side=tk.LEFT, padx=(0, 6))
    tk.Label(
        marco, text=texto.upper(), bg=BG_PANEL, fg=TEXTO,
        font=FUENTE_SECCION,
    ).pack(side=tk.LEFT)
    return marco


def _franja_gradiente(canvas, x0, y0, x1, y1, colores):
    """Dibuja una franja horizontal interpolando entre los colores dados,
    simulando un degradé índigo → cian → naranja con rectángulos finos."""
    pasos = 120
    ancho_paso = (x1 - x0) / pasos
    segmentos = len(colores) - 1
    for i in range(pasos):
        t = i / (pasos - 1)
        seg = min(int(t * segmentos), segmentos - 1)
        t_local = (t * segmentos) - seg
        r1, g1, b1 = _hex_a_rgb(colores[seg])
        r2, g2, b2 = _hex_a_rgb(colores[seg + 1])
        r = r1 + (r2 - r1) * t_local
        g = g1 + (g2 - g1) * t_local
        b = b1 + (b2 - b1) * t_local
        color = _rgb_a_hex((r, g, b))
        px = x0 + i * ancho_paso
        canvas.create_rectangle(px, y0, px + ancho_paso + 1, y1, fill=color, outline="")


# ---------------------------------------------------------------------------
# Diálogo modal simple, con el mismo tema oscuro (para "Nuevo objeto…" y
# "Actualizar con IA…")
# ---------------------------------------------------------------------------

class _DialogoTexto:
    def __init__(self, parent, titulo, etiqueta, on_aceptar, texto_boton="Aceptar"):
        self.on_aceptar = on_aceptar

        self.top = tk.Toplevel(parent, bg=BG_PANEL)
        self.top.title(titulo)
        self.top.configure(bg=BG_PANEL)
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()

        tk.Frame(self.top, bg=CIAN, height=3).pack(fill=tk.X)

        tk.Label(
            self.top, text=etiqueta, wraplength=340, justify=tk.LEFT,
            bg=BG_PANEL, fg=TEXTO, font=FUENTE,
        ).pack(padx=16, pady=(16, 8))

        self.entry = tk.Entry(
            self.top, width=46, font=FUENTE_MONO, bg=BG_CAMPO, fg=TEXTO,
            insertbackground=CIAN, relief=tk.FLAT,
            highlightthickness=1, highlightbackground=BORDE, highlightcolor=CIAN,
        )
        self.entry.pack(padx=16, pady=(0, 12), ipady=5)
        self.entry.focus_set()
        self.entry.bind("<Return>", lambda _e: self._aceptar())
        self.entry.bind("<Escape>", lambda _e: self.top.destroy())

        frame_botones = tk.Frame(self.top, bg=BG_PANEL)
        frame_botones.pack(pady=(0, 16), padx=16, fill=tk.X)
        crear_boton(frame_botones, texto_boton, self._aceptar, color=NARANJA).pack(
            side=tk.RIGHT
        )
        crear_boton_fantasma(frame_botones, "Cancelar", self.top.destroy, color=TEXTO_MUTED).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

    def _aceptar(self):
        texto = self.entry.get()
        self.top.destroy()
        if texto.strip():
            self.on_aceptar(texto)


# ---------------------------------------------------------------------------
# Aplicación principal
# ---------------------------------------------------------------------------

class EditorVisual:
    """Ventana completa: lista de objetos ⟶ previsualización 3D ⟶ ficha de
    características (apariencia + propiedades físicas), todo editable."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Naro Studio — Editor visual de objetos")
        self.root.configure(bg=BG_FONDO)
        self.root.geometry("1180x720")
        self.root.minsize(980, 620)

        self.objeto_actual = None      # nombre del objeto seleccionado
        self.estilo_actual = dict(ESTILO_DEFECTO)
        self.entradas_prop = {}        # clave -> widget Entry/Text (propiedades físicas)
        self._sucio = False            # True si hay cambios de apariencia sin guardar

        self._configurar_estilo_ttk()

        self.var_vincular = tk.BooleanVar(value=True)
        self.var_auto_rotar = tk.BooleanVar(value=True)
        self.var_escala_x = tk.DoubleVar(value=1.0)
        self.var_escala_y = tk.DoubleVar(value=1.0)
        self.var_escala_z = tk.DoubleVar(value=1.0)
        self.var_busqueda = tk.StringVar()
        self.var_busqueda.trace_add("write", lambda *_a: self._refrescar_lista())

        self._construir_ui()
        self._refrescar_lista()
        self._loop_auto_rotar()

    # -- tema ttk (sliders, checkbox, scrollbar) -------------------------------

    def _configurar_estilo_ttk(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Naro.Horizontal.TScale",
            background=BG_PANEL, troughcolor=BG_CAMPO,
            lightcolor=CIAN, darkcolor=CIAN, bordercolor=BG_PANEL,
        )
        style.configure(
            "Naro.TCheckbutton",
            background=BG_PANEL, foreground=TEXTO, font=FUENTE,
            focuscolor=BG_PANEL,
        )
        style.map(
            "Naro.TCheckbutton",
            background=[("active", BG_PANEL)],
            foreground=[("active", CIAN)],
        )
        style.configure(
            "Naro.Vertical.TScrollbar",
            background=BG_PANEL_2, troughcolor=BG_FONDO,
            bordercolor=BG_FONDO, arrowcolor=TEXTO_MUTED, relief=tk.FLAT,
        )
        style.map("Naro.Vertical.TScrollbar", background=[("active", INDIGO)])

    # -- construcción de la UI --------------------------------------------------

    def _construir_ui(self):
        self._construir_encabezado()

        cuerpo = tk.Frame(self.root, bg=BG_FONDO)
        cuerpo.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        self._construir_sidebar(cuerpo)
        self._construir_columna_preview(cuerpo)
        self._construir_columna_caracteristicas(cuerpo)

    def _construir_encabezado(self):
        alto_franja = 3
        franja = tk.Canvas(self.root, height=alto_franja, bg=BG_FONDO, highlightthickness=0)
        franja.pack(fill=tk.X, side=tk.TOP)
        self.root.update_idletasks()
        franja.bind(
            "<Configure>",
            lambda e: (franja.delete("all"),
                       _franja_gradiente(franja, 0, 0, e.width, alto_franja,
                                          [INDIGO, CIAN, NARANJA])),
        )

        header = tk.Frame(self.root, bg=BG_FONDO)
        header.pack(fill=tk.X, padx=16, pady=(10, 8))

        tk.Label(
            header, text="NARO STUDIO", bg=BG_FONDO, fg=TEXTO, font=FUENTE_TITULO,
        ).pack(side=tk.LEFT)
        tk.Label(
            header, text="  ·  editor visual de objetos",
            bg=BG_FONDO, fg=TEXTO_MUTED, font=FUENTE_SUBTIT,
        ).pack(side=tk.LEFT, padx=(2, 0))

        leyenda = tk.Label(
            header,
            text="clic + arrastre para orbitar la previsualización · rueda del mouse para zoom",
            bg=BG_FONDO, fg=TEXTO_TENUE, font=("Segoe UI", 8),
        )
        leyenda.pack(side=tk.RIGHT)

    def _panel(self, parent, **kw):
        marco = tk.Frame(parent, bg=BG_PANEL, highlightbackground=BORDE,
                          highlightthickness=1, **kw)
        return marco

    # -- columna izquierda: lista de objetos ------------------------------------

    def _construir_sidebar(self, parent):
        col = self._panel(parent, width=230)
        col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        col.pack_propagate(False)

        encabezado_seccion(col, "Objetos", color=INDIGO)

        marco_buscar = tk.Frame(col, bg=BG_PANEL)
        marco_buscar.pack(fill=tk.X, padx=12)
        self.entry_busqueda = tk.Entry(
            marco_buscar, textvariable=self.var_busqueda, font=FUENTE_MONO,
            bg=BG_CAMPO, fg=TEXTO, insertbackground=CIAN, relief=tk.FLAT,
            highlightthickness=1, highlightbackground=BORDE, highlightcolor=CIAN,
        )
        self.entry_busqueda.pack(fill=tk.X, ipady=4, pady=(0, 8))
        self._placeholder(self.entry_busqueda, "Buscar…")

        marco_lista = tk.Frame(col, bg=BG_PANEL)
        marco_lista.pack(fill=tk.BOTH, expand=True, padx=12)

        scrollbar = ttk.Scrollbar(marco_lista, orient=tk.VERTICAL,
                                   style="Naro.Vertical.TScrollbar")
        self.lista_box = tk.Listbox(
            marco_lista, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE,
            font=FUENTE_MONO, bg=BG_CAMPO, fg=TEXTO,
            selectbackground=INDIGO, selectforeground="#ffffff",
            activestyle="none", relief=tk.FLAT, highlightthickness=0,
            borderwidth=0,
        )
        scrollbar.config(command=self.lista_box.yview)
        self.lista_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.lista_box.bind("<<ListboxSelect>>", self._on_seleccionar)

        marco_botones = tk.Frame(col, bg=BG_PANEL)
        marco_botones.pack(fill=tk.X, padx=12, pady=12)
        crear_boton(marco_botones, "＋ Nuevo objeto…", self._on_nuevo, color=INDIGO).pack(
            fill=tk.X, pady=(0, 6)
        )
        crear_boton_fantasma(marco_botones, "Eliminar seleccionado", self._on_eliminar,
                              color=NARANJA).pack(fill=tk.X)

    def _placeholder(self, entry, texto):
        entry.insert(0, texto)
        entry.config(fg=TEXTO_TENUE)

        def _foco_in(_e):
            if entry.get() == texto:
                entry.delete(0, tk.END)
                entry.config(fg=TEXTO)

        def _foco_out(_e):
            if not entry.get():
                entry.insert(0, texto)
                entry.config(fg=TEXTO_TENUE)

        entry.bind("<FocusIn>", _foco_in)
        entry.bind("<FocusOut>", _foco_out)

    # -- columna central: previsualización + apariencia -------------------------

    def _construir_columna_preview(self, parent):
        col = self._panel(parent, width=380)
        col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        col.pack_propagate(False)

        encabezado_seccion(col, "Previsualización", color=CIAN)

        self.label_nombre_obj = tk.Label(
            col, text="—", bg=BG_PANEL, fg=TEXTO, font=FUENTE_MONO_B, anchor="w",
        )
        self.label_nombre_obj.pack(fill=tk.X, padx=12)

        marco_canvas = tk.Frame(col, bg=BORDE)
        marco_canvas.pack(padx=12, pady=(8, 6))
        self.vista = VistaPrevia3D(marco_canvas, ancho=352, alto=300)
        self.vista.pack(padx=1, pady=1)

        marco_vista_botones = tk.Frame(col, bg=BG_PANEL)
        marco_vista_botones.pack(fill=tk.X, padx=12, pady=(0, 4))
        ttk.Checkbutton(
            marco_vista_botones, text="Auto-rotar", variable=self.var_auto_rotar,
            style="Naro.TCheckbutton",
        ).pack(side=tk.LEFT)
        crear_boton_fantasma(
            marco_vista_botones, "Restablecer vista", self._reset_vista, color=TEXTO_MUTED
        ).pack(side=tk.RIGHT)

        encabezado_seccion(col, "Apariencia", color=NARANJA)

        # --- color ---
        marco_color = tk.Frame(col, bg=BG_PANEL)
        marco_color.pack(fill=tk.X, padx=12, pady=(0, 8))
        tk.Label(marco_color, text="Color", bg=BG_PANEL, fg=TEXTO_MUTED,
                 font=FUENTE_ETQ).pack(side=tk.LEFT)
        self.swatch_color = tk.Frame(
            marco_color, bg=self.estilo_actual["color_hex"], width=26, height=20,
            highlightbackground=BORDE, highlightthickness=1, cursor="hand2",
        )
        self.swatch_color.pack(side=tk.RIGHT)
        self.swatch_color.bind("<Button-1>", lambda _e: self._elegir_color())
        self.label_color_hex = tk.Label(
            marco_color, text=self.estilo_actual["color_hex"], bg=BG_PANEL, fg=TEXTO,
            font=FUENTE_MONO,
        )
        self.label_color_hex.pack(side=tk.RIGHT, padx=(0, 8))

        # --- escala ---
        marco_escala = tk.Frame(col, bg=BG_PANEL)
        marco_escala.pack(fill=tk.X, padx=12, pady=(0, 4))
        tk.Label(marco_escala, text="Escala", bg=BG_PANEL, fg=TEXTO_MUTED,
                 font=FUENTE_ETQ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            marco_escala, text="vincular ejes", variable=self.var_vincular,
            style="Naro.TCheckbutton",
        ).pack(side=tk.RIGHT)

        self.sliders_escala = {}
        for eje, var, color in (
            ("x", self.var_escala_x, CIAN),
            ("y", self.var_escala_y, INDIGO),
            ("z", self.var_escala_z, NARANJA),
        ):
            fila = tk.Frame(col, bg=BG_PANEL)
            fila.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(fila, text=eje.upper(), bg=BG_PANEL, fg=color, font=FUENTE_MONO_B,
                     width=2).pack(side=tk.LEFT)
            escala_slider = ttk.Scale(
                fila, from_=0.2, to=3.0, orient=tk.HORIZONTAL, variable=var,
                style="Naro.Horizontal.TScale",
                command=lambda v, e=eje: self._on_escala_cambio(e, v),
            )
            escala_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
            label_valor = tk.Label(fila, text="1.00×", bg=BG_PANEL, fg=TEXTO,
                                    font=FUENTE_MONO, width=5, anchor="e")
            label_valor.pack(side=tk.RIGHT)
            self.sliders_escala[eje] = (escala_slider, label_valor)

        self._set_controles_apariencia_habilitados(False)

    # -- columna derecha: propiedades físicas (scrollable) -----------------------


    # -- columna derecha: propiedades físicas (scrollable) -----------------------

    def _construir_columna_caracteristicas(self, parent):
        col = self._panel(parent)
        col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        encabezado_seccion(col, "Propiedades físicas", color=INDIGO)

        self.label_geometria = tk.Label(
            col, text="Geometría: —", bg=BG_PANEL, fg=TEXTO_MUTED,
            font=("Segoe UI", 8), anchor="w",
        )
        self.label_geometria.pack(fill=tk.X, padx=12)

        # --- área con scroll para la ficha de campos ---
        marco_scroll = tk.Frame(col, bg=BG_PANEL)
        marco_scroll.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 6))

        canvas_scroll = tk.Canvas(marco_scroll, bg=BG_PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(marco_scroll, orient=tk.VERTICAL,
                                   command=canvas_scroll.yview,
                                   style="Naro.Vertical.TScrollbar")
        self.frame_form = tk.Frame(canvas_scroll, bg=BG_PANEL)

        self.frame_form.bind(
            "<Configure>",
            lambda _e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
        )
        ventana_id = canvas_scroll.create_window((0, 0), window=self.frame_form, anchor="nw")
        canvas_scroll.bind(
            "<Configure>", lambda e: canvas_scroll.itemconfig(ventana_id, width=e.width)
        )
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        def _rueda(evento):
            delta = getattr(evento, "delta", 0)
            paso = -1 if (delta > 0 or getattr(evento, "num", None) == 4) else 1
            canvas_scroll.yview_scroll(paso, "units")

        canvas_scroll.bind_all("<MouseWheel>", _rueda, add="+")
        canvas_scroll.bind_all("<Button-4>", _rueda, add="+")
        canvas_scroll.bind_all("<Button-5>", _rueda, add="+")

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for clave, etiqueta, unidad, _es_num in obj.CAMPOS:
            texto_etiqueta = f"{etiqueta} ({unidad})" if unidad else etiqueta
            fila = tk.Frame(self.frame_form, bg=BG_PANEL)
            fila.pack(fill=tk.X, pady=3)
            tk.Label(
                fila, text=texto_etiqueta, bg=BG_PANEL, fg=TEXTO_MUTED,
                font=FUENTE_ETQ, anchor="w",
            ).pack(fill=tk.X)

            if clave == "notas":
                widget = tk.Text(
                    fila, height=3, font=FUENTE_MONO, bg=BG_CAMPO, fg=TEXTO,
                    insertbackground=CIAN, relief=tk.FLAT, wrap=tk.WORD,
                    highlightthickness=1, highlightbackground=BORDE, highlightcolor=CIAN,
                )
            else:
                widget = tk.Entry(
                    fila, font=FUENTE_MONO, bg=BG_CAMPO, fg=TEXTO,
                    insertbackground=CIAN, relief=tk.FLAT,
                    highlightthickness=1, highlightbackground=BORDE, highlightcolor=CIAN,
                )
            widget.pack(fill=tk.X, ipady=3, pady=(2, 0))
            self.entradas_prop[clave] = widget

        # --- botones de acción ---
        marco_acciones = tk.Frame(col, bg=BG_PANEL)
        marco_acciones.pack(fill=tk.X, padx=12, pady=(4, 4))
        crear_boton(marco_acciones, "Guardar cambios", self._on_guardar, color=NARANJA).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        crear_boton_fantasma(
            marco_acciones, "Actualizar con IA…", self._on_actualizar_ia, color=CIAN
        ).pack(side=tk.LEFT)

        self.label_estado = tk.Label(
            col, text="Seleccioná un objeto o creá uno nuevo.",
            bg=BG_PANEL, fg=TEXTO_MUTED, font=("Segoe UI", 8),
            anchor="w", justify=tk.LEFT, wraplength=420,
        )
        self.label_estado.pack(fill=tk.X, padx=12, pady=(0, 10))

        self._set_formulario_habilitado(False)

    # -- helpers de estado de formulario -----------------------------------------

    def _set_formulario_habilitado(self, habilitado: bool):
        estado = tk.NORMAL if habilitado else tk.DISABLED
        for widget in self.entradas_prop.values():
            widget.config(state=estado)

    def _set_controles_apariencia_habilitados(self, habilitado: bool):
        estado = tk.NORMAL if habilitado else tk.DISABLED
        for slider, _label in self.sliders_escala.values():
            slider.state(["!disabled"] if habilitado else ["disabled"])
        cursor = "hand2" if habilitado else "arrow"
        self.swatch_color.config(cursor=cursor)

    def _leer_formulario(self) -> dict:
        datos = {}
        for clave, widget in self.entradas_prop.items():
            if isinstance(widget, tk.Text):
                datos[clave] = widget.get("1.0", tk.END).strip()
            else:
                datos[clave] = widget.get().strip()
        return datos

    def _escribir_formulario(self, propiedades: dict):
        for clave, widget in self.entradas_prop.items():
            valor = propiedades.get(clave, "")
            texto = str(valor)
            deshabilitado = str(widget.cget("state")) == "disabled"
            if deshabilitado:
                widget.config(state=tk.NORMAL)
            if isinstance(widget, tk.Text):
                widget.delete("1.0", tk.END)
                widget.insert("1.0", texto)
            else:
                widget.delete(0, tk.END)
                widget.insert(0, texto)
            if deshabilitado:
                widget.config(state=tk.DISABLED)

    def _set_estado(self, texto: str):
        self.label_estado.config(text=texto)

    def _marcar_sucio(self):
        self._sucio = True

    # -- lista de objetos ---------------------------------------------------------

    def _refrescar_lista(self, seleccionar: str = None):
        if not hasattr(self, "lista_box"):
            return  # el buscador puede disparar su trace antes de que exista la lista
        filtro = self.var_busqueda.get().strip().lower()
        if filtro == "buscar…":
            filtro = ""
        self.lista_box.delete(0, tk.END)
        nombres = [n for n in obj.listar_objetos() if filtro in n.lower()]
        for nombre in nombres:
            self.lista_box.insert(tk.END, nombre)
        if seleccionar and seleccionar in nombres:
            idx = nombres.index(seleccionar)
            self.lista_box.selection_set(idx)
            self.lista_box.see(idx)
            self._cargar_objeto(seleccionar)

    def _cargar_objeto(self, nombre: str):
        registro = obj.cargar_objeto(nombre)
        if registro is None:
            return
        self.objeto_actual = nombre
        self._sucio = False

        self._escribir_formulario(registro.get("propiedades", {}))
        self._set_formulario_habilitado(True)

        figura = registro.get("figura", {"puntos": [], "conexiones": [], "primitivas": []})
        malla_real = registro.get("malla")
        if malla_real:
            lod_bajo = malla_real.get("lod_bajo") or {}
            n_vert = len(lod_bajo.get("v", []))
            n_caras = len(lod_bajo.get("f", []))
            origen = malla_real.get("origen", "malla")
            self.label_geometria.config(
                text=f"Geometría: malla real ({origen}) · {n_vert} vértices · {n_caras} caras"
            )
        else:
            self.label_geometria.config(
                text=(
                    f"Geometría: {len(figura.get('puntos', []))} puntos · "
                    f"{len(figura.get('conexiones', []))} conexiones · "
                    f"{len(figura.get('primitivas', []))} primitivas"
                )
            )

        self.estilo_actual = obtener_estilo(nombre)
        self._aplicar_estilo_a_controles(self.estilo_actual)
        self._set_controles_apariencia_habilitados(True)

        self.label_nombre_obj.config(text=nombre)
        self.vista.cargar(
            nombre, figura, self.estilo_actual["color_hex"],
            (self.estilo_actual["escala_x"], self.estilo_actual["escala_y"],
             self.estilo_actual["escala_z"]),
            malla_real,
        )

        self._set_estado(f"'{nombre}' — actualizado {registro.get('actualizado', '?')}.")

    def _aplicar_estilo_a_controles(self, estilo: dict):
        self.swatch_color.config(bg=estilo["color_hex"])
        self.label_color_hex.config(text=estilo["color_hex"])
        for eje, var in (("x", self.var_escala_x), ("y", self.var_escala_y),
                         ("z", self.var_escala_z)):
            valor = estilo[f"escala_{eje}"]
            var.set(valor)
            self.sliders_escala[eje][1].config(text=f"{valor:.2f}×")

    # -- eventos: selección / apariencia -------------------------------------------

    def _on_seleccionar(self, _evento):
        sel = self.lista_box.curselection()
        if not sel:
            return
        nombre = self.lista_box.get(sel[0])
        self._cargar_objeto(nombre)

    def _elegir_color(self):
        if not self.objeto_actual:
            return
        _rgb, hexcolor = colorchooser.askcolor(
            color=self.estilo_actual["color_hex"], title="Elegir color del objeto"
        )
        if not hexcolor:
            return
        hexcolor = hexcolor.upper()
        self.estilo_actual["color_hex"] = hexcolor
        self.swatch_color.config(bg=hexcolor)
        self.label_color_hex.config(text=hexcolor)
        self._marcar_sucio()
        self._refrescar_apariencia_vista()

    def _on_escala_cambio(self, eje: str, valor_str: str):
        if not self.objeto_actual:
            return
        valor = round(float(valor_str), 2)
        if self.var_vincular.get():
            for e, var in (("x", self.var_escala_x), ("y", self.var_escala_y),
                           ("z", self.var_escala_z)):
                var.set(valor)
                self.estilo_actual[f"escala_{e}"] = valor
                self.sliders_escala[e][1].config(text=f"{valor:.2f}×")
        else:
            self.estilo_actual[f"escala_{eje}"] = valor
            self.sliders_escala[eje][1].config(text=f"{valor:.2f}×")
        self._marcar_sucio()
        self._refrescar_apariencia_vista()

    def _refrescar_apariencia_vista(self):
        self.vista.actualizar_apariencia(
            self.estilo_actual["color_hex"],
            (self.estilo_actual["escala_x"], self.estilo_actual["escala_y"],
             self.estilo_actual["escala_z"]),
        )

    def _reset_vista(self):
        self.vista._ang_x, self.vista._ang_y, self.vista._zoom = -0.45, 0.7, 1.0
        self.vista.redibujar()

    def _loop_auto_rotar(self):
        if self.var_auto_rotar.get() and self.objeto_actual:
            self.vista.girar_auto()
        self.root.after(40, self._loop_auto_rotar)

    # -- guardar / actualizar con IA -------------------------------------------

    def _on_guardar(self):
        if not self.objeto_actual:
            return
        datos = self._leer_formulario()
        obj.guardar_objeto(self.objeto_actual, propiedades=datos)
        guardar_estilo(self.objeto_actual, self.estilo_actual)
        self._sucio = False
        self._set_estado(f"'{self.objeto_actual}' guardado (propiedades + apariencia).")
        self._refrescar_lista(seleccionar=self.objeto_actual)

    def _on_actualizar_ia(self):
        if not self.objeto_actual:
            messagebox.showinfo("Actualizar con IA", "Primero seleccioná un objeto.",
                                 parent=self.root)
            return
        _DialogoTexto(
            self.root,
            titulo="Actualizar con IA",
            etiqueta=(
                f"¿Qué querés cambiar o actualizar de '{self.objeto_actual}'?\n"
                "(ej: 'es de aluminio en vez de acero', 'en realidad pesa 3.5kg')"
            ),
            on_aceptar=self._actualizar_objeto_async,
            texto_boton="Actualizar",
        )

    def _actualizar_objeto_async(self, pedido: str):
        pedido = pedido.strip()
        if not pedido or not self.objeto_actual:
            return
        nombre = self.objeto_actual
        propiedades_actuales = self._leer_formulario()
        self._set_estado(f"Actualizando '{nombre}' con la IA…")
        threading.Thread(
            target=self._actualizar_objeto_worker,
            args=(nombre, propiedades_actuales, pedido), daemon=True,
        ).start()

    def _actualizar_objeto_worker(self, nombre, propiedades_actuales, pedido):
        # Corre en un hilo de fondo (esperando a Ollama) — nunca toca los
        # widgets directo: tkinter no es thread-safe. Cada actualización de
        # UI se programa con en_hilo_ui() para que corra en el hilo que
        # ejecutó root.mainloop() (ver ui_thread.py).
        propiedades = obj.actualizar_propiedades(propiedades_actuales, pedido)
        if propiedades is None:
            en_hilo_ui(self.root, self._set_estado, f"No se pudo actualizar '{nombre}'.")
            return
        obj.guardar_objeto(nombre, propiedades=propiedades)
        if self.objeto_actual == nombre:
            en_hilo_ui(self.root, self._escribir_formulario, propiedades)
        en_hilo_ui(self.root, self._set_estado, f"'{nombre}' actualizado con la IA.")

    # -- nuevo / eliminar --------------------------------------------------------

    def _on_nuevo(self):
        _DialogoTexto(
            self.root,
            titulo="Nuevo objeto",
            etiqueta="Describí el objeto (ej: 'silla de madera', 'cable de cobre 2mm'):",
            on_aceptar=self._crear_objeto_async,
            texto_boton="Crear",
        )

    def _crear_objeto_async(self, descripcion: str):
        descripcion = descripcion.strip()
        if not descripcion:
            return
        self._set_estado(f"Dibujando '{descripcion}'…")
        threading.Thread(
            target=self._crear_objeto_worker, args=(descripcion,), daemon=True
        ).start()

    def _crear_objeto_worker(self, descripcion: str):
        # Mismo criterio que _actualizar_objeto_worker: este método (y los
        # callbacks que le pasa a obj.crear_objeto) corre en el hilo de
        # fondo. Ningún widget se toca directo, todo pasa por en_hilo_ui().
        def al_dibujar(nombre, _registro):
            en_hilo_ui(self.root, self._refrescar_lista, nombre)
            en_hilo_ui(self.root, self._set_estado,
                       f"'{nombre}' dibujado. Pidiendo sus características físicas…")

        def al_tener_propiedades(nombre, registro):
            if registro is None:
                en_hilo_ui(
                    self.root, self._set_estado,
                    f"'{nombre}' dibujado, pero no se pudieron generar sus características.",
                )
                return
            if self.objeto_actual == nombre:
                en_hilo_ui(self.root, self._cargar_objeto, nombre)
            en_hilo_ui(self.root, self._set_estado,
                       f"'{nombre}' completo: geometría + características físicas.")

        registro = obj.crear_objeto(
            descripcion, callback_figura=al_dibujar, callback_propiedades=al_tener_propiedades
        )
        if registro is None:
            en_hilo_ui(self.root, self._set_estado,
                       f"No se pudo generar la geometría de '{descripcion}'.")

    def _on_eliminar(self):
        sel = self.lista_box.curselection()
        if not sel:
            return
        nombre = self.lista_box.get(sel[0])
        if not messagebox.askyesno("Eliminar objeto", f"¿Eliminar '{nombre}' del catálogo?",
                                    parent=self.root):
            return
        obj.eliminar_objeto(nombre)
        eliminar_estilo(nombre)
        if self.objeto_actual == nombre:
            self.objeto_actual = None
            self._escribir_formulario({})
            self.label_geometria.config(text="Geometría: —")
            self.label_nombre_obj.config(text="—")
            self.vista.limpiar()
            self._set_formulario_habilitado(False)
            self._set_controles_apariencia_habilitados(False)
        self._refrescar_lista()
        self._set_estado(f"'{nombre}' eliminado.")


# ---------------------------------------------------------------------------
# Lanzadores
# ---------------------------------------------------------------------------

def abrir_editor_visual(parent: tk.Misc = None) -> tk.Toplevel | tk.Tk:
    """Punto de integración opcional: abre el editor como ventana propia desde
    otro programa tkinter ya corriendo (ej: el panel de main.py).

        import editor_visual
        editor_visual.abrir_editor_visual(root)
    """
    ventana = tk.Toplevel(parent) if parent is not None else tk.Tk()
    EditorVisual(ventana)
    return ventana


def _lanzar():
    root = tk.Tk()
    EditorVisual(root)
    root.mainloop()


if __name__ == "__main__":
    _lanzar()