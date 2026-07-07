"""
optimizacion_objetos.py — Optimización de objetos YA CONSTRUIDOS del catálogo
(objetos_db.json): los pasa a estados más livianos cuando hace falta, sin
tocar el LLM ni regenerar geometría.

--------------------------------------------------------------------------
Diagnóstico del origen del problema (0.3 fps)
--------------------------------------------------------------------------
`generar_geometria_parametrica()` (objetos.py) arma la malla final con
`ensamblador.ensamblar_partes()` y la serializa DIRECTO con
`optimizacion_malla.serializar_json(..., lod_alto=None)` — sin pasar nunca
por `optimizacion_malla.decimar()`. `biblioteca_mallas.buscar()` y
`malla_ia_async.py` sí decimaban antes de guardar; el camino paramétrico
(que hoy es el camino PRINCIPAL de geometría, ver
`objetos.py::crear_objeto`) era el único que se salteaba ese paso.

Resultado: un objeto con varias partes (cilindros, tubos, esferas, uniones
booleanas) puede terminar con miles de caras SIN NINGÚN tope. Y
`render_malla.dibujar_malla` hace un `panel.copy()` + `cv2.addWeighted` por
CADA cara (`_fill_poly_alpha`) — eso multiplica el costo de una malla grande
por un blend de panel completo por triángulo. Esa combinación es la que
hunde los fps a niveles como 0.3, no "una malla un poco grande".

Este módulo NO regenera geometría ni le pide nada al LLM: solo re-empaqueta
mallas YA CORRECTAS (contactos exactos, watertight — eso no cambia) en menos
triángulos, misma filosofía que `optimizacion_malla.decimar()` (quadric edge
collapse vía pyfqmr) pero:
  1) aplicada RETROACTIVAMENTE a objetos que ya están guardados en
     objetos_db.json con geometría sin decimar (`optimizar_objeto_construido`
     / `optimizar_todos_los_objetos`), y
  2) con un `GestorCalidadDinamica` que reacciona a los fps REALES del bucle
     principal, bajando (o subiendo) de nivel de detalle en caliente sin
     volver a decimar nada — solo elige entre LODs ya precalculados.

Requiere lo mismo que optimizacion_malla.py (numpy + pyfqmr) para la parte
de decimación; la parte de gestión dinámica de calidad es Python puro, sin
dependencias nuevas.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque

import malla as malla_mod
import objetos
import optimizacion_malla as opt

# Tercer escalón, por debajo de opt.LOD_BAJO (150) — piso de emergencia para
# cuando ni el LOD_BAJO de siempre alcanza. No reemplaza el LOD_BAJO/LOD_ALTO
# de siempre: es una malla EXTRA que se precalcula y guarda al optimizar,
# y que GestorCalidadDinamica solo usa si los fps reales colapsan.
LOD_EMERGENCIA = 40


# ---------------------------------------------------------------------------
# Optimización retroactiva de objetos ya guardados (objetos_db.json)
# ---------------------------------------------------------------------------

def _hash_malla_dict(malla_dict: dict) -> str:
    """Hash corto y estable del contenido de una malla serializada (v/f),
    para saber si esta EXACTA geometría ya se optimizó antes y no repetir
    el trabajo cada vez que se corre optimizar_todos_los_objetos() — mismo
    criterio de caché que el resto del proyecto: nunca recalcular lo que
    no cambió."""
    crudo = json.dumps({"v": malla_dict.get("v", []), "f": malla_dict.get("f", [])},
                        separators=(",", ":"))
    return hashlib.sha1(crudo.encode("utf-8")).hexdigest()[:16]


def _num_caras(malla_dict: dict | None) -> int:
    return len(malla_dict.get("f", [])) if malla_dict else 0


def optimizar_objeto_construido(nombre: str, forzar: bool = False) -> dict | None:
    """Optimiza en el lugar la malla real de `nombre` (objetos_db.json),
    (re)generando lod_bajo / lod_alto / lod_emergencia con
    optimizacion_malla.decimar(). Idempotente: si la malla no cambió desde
    la última vez que se optimizó (mismo hash de la malla fuente), no hace
    nada salvo que `forzar=True`.

    Devuelve {"nombre", "caras_antes", "caras_despues", "ya_estaba_optimizado"}
    o None si el objeto no existe o no tiene malla real (objetos con solo
    wireframe heredado no aplican acá — no hay nada que decimar).
    """
    registro = objetos.cargar_objeto(nombre)
    if not registro or not registro.get("malla"):
        return None

    malla_info = registro["malla"]
    lod_bajo_dict = malla_info.get("lod_bajo")
    if not lod_bajo_dict:
        return None

    # La malla "fuente" para decimar es la más detallada que haya disponible
    # (lod_alto si existe, si no lod_bajo) — nunca decimar sobre algo que ya
    # se decimó antes con un tope más chico, eso degrada la forma sin
    # necesidad ni beneficio.
    fuente_dict = malla_info.get("lod_alto") or lod_bajo_dict
    hash_actual = _hash_malla_dict(fuente_dict)
    if not forzar and malla_info.get("_optimizado_hash") == hash_actual:
        return {"nombre": nombre, "caras_antes": _num_caras(lod_bajo_dict),
                "caras_despues": _num_caras(lod_bajo_dict), "ya_estaba_optimizado": True}

    malla_fuente = malla_mod.Malla.from_dict(fuente_dict)
    caras_antes = malla_fuente.num_caras()

    try:
        nuevo_lod_bajo = opt.decimar(malla_fuente, opt.LOD_BAJO)
        nuevo_lod_alto = opt.decimar(malla_fuente, opt.LOD_ALTO)
        nuevo_lod_emergencia = opt.decimar(malla_fuente, LOD_EMERGENCIA)
    except ImportError as e:
        print(f"[optimizacion_objetos] No se pudo optimizar '{nombre}': {e}")
        return None

    nuevo_malla_info = opt.serializar_json(
        nombre, nuevo_lod_bajo, nuevo_lod_alto,
        origen=malla_info.get("origen", "desconocido"),
        radio_bounding=malla_info.get("radio_bounding"),
    )
    nuevo_malla_info["lod_emergencia"] = nuevo_lod_emergencia.to_dict()
    nuevo_malla_info["_optimizado_hash"] = hash_actual

    objetos.guardar_objeto(nombre, malla=nuevo_malla_info)
    invalidar_cache(nombre)

    caras_despues = nuevo_lod_bajo.num_caras()
    print(f"[optimizacion_objetos] '{nombre}': {caras_antes} → {caras_despues} caras "
          f"(lod_bajo={caras_despues}, lod_alto={nuevo_lod_alto.num_caras()}, "
          f"lod_emergencia={nuevo_lod_emergencia.num_caras()}).")
    return {"nombre": nombre, "caras_antes": caras_antes, "caras_despues": caras_despues,
            "ya_estaba_optimizado": False}


def optimizar_todos_los_objetos(forzar: bool = False, callback_progreso=None) -> list:
    """Barre todo objetos_db.json y optimiza cada objeto con malla real.
    Pensado para correr UNA VEZ sobre un catálogo que ya tiene objetos
    pesados guardados de antes de este módulo — no hace falta borrar ni
    regenerar nada, esto re-empaqueta la geometría existente en el lugar.

        python -c "import optimizacion_objetos as o; o.optimizar_todos_los_objetos()"
    """
    resultados = []
    nombres = objetos.listar_objetos()
    for i, nombre in enumerate(nombres):
        resultado = optimizar_objeto_construido(nombre, forzar=forzar)
        if resultado:
            resultados.append(resultado)
        if callback_progreso:
            callback_progreso(i + 1, len(nombres), nombre)

    optimizados = [r for r in resultados if not r["ya_estaba_optimizado"]]
    if optimizados:
        total_antes = sum(r["caras_antes"] for r in optimizados)
        total_despues = sum(r["caras_despues"] for r in optimizados)
        print(f"[optimizacion_objetos] {len(optimizados)} objeto(s) optimizados: "
              f"{total_antes} → {total_despues} caras totales (lod_bajo).")
    else:
        print("[optimizacion_objetos] Nada para optimizar (todo ya estaba al día).")
    return resultados


# ---------------------------------------------------------------------------
# Gestor de calidad dinámica — reacciona a los fps REALES del bucle
# principal, no a un cálculo teórico de "cuántas caras hay". Mismo patrón de
# histéresis que modos.py::AsesorIA (confirmar N frames seguidos antes de
# cambiar de nivel), para no parpadear entre dos niveles vecinos por un pico
# aislado de un solo frame.
# ---------------------------------------------------------------------------

NIVELES = ("alto", "normal", "bajo", "emergencia")

# Umbrales de fps promedio para BAJAR de nivel.
_UMBRAL_BAJAR = {"normal": 20.0, "bajo": 12.0, "emergencia": 6.0}
# Umbrales para SUBIR — a propósito más altos que el umbral de bajar del
# nivel de arriba, para que cueste más "ganarse" volver a más detalle que
# perderlo (evita oscilar en el límite).
_UMBRAL_SUBIR = {"alto": 27.0, "normal": 18.0, "bajo": 10.0}

# Qué LOD (clave dentro del dict "malla") usar para malla_lod_baja/alta según
# el nivel. En "alto" es el comportamiento de siempre (lod_bajo congelada,
# lod_alto solo si la figura está agarrada); en el resto se degrada.
_CLAVES_POR_NIVEL = {
    "alto":       ("lod_bajo", "lod_alto"),
    "normal":     ("lod_bajo", "lod_bajo"),      # nunca sube a lod_alto ni agarrada
    "bajo":       ("lod_bajo", "lod_bajo"),
    "emergencia": ("lod_emergencia", "lod_emergencia"),
}


class GestorCalidadDinamica:
    """Monitorea los fps reales (ventana móvil) y decide un `nivel_actual`
    de detalle para todas las figuras con malla del entorno. NUNCA decima en
    caliente (sería más lento, no más rápido) — solo elige, para cada
    figura, cuál de sus LOD ya precalculados usar este frame (ver
    `sincronizar_calidad_entorno`)."""

    def __init__(self, ventana: int = 30, frames_confirmacion: int = 15):
        self._fps_ventana: deque = deque(maxlen=ventana)
        self._nivel_actual = "alto"
        self._frames_confirmacion = frames_confirmacion
        self._contador_estable = 0
        self._nivel_candidato = "alto"

    def registrar_frame(self, dt_segundos: float) -> None:
        """Llamar una vez por frame con el dt real del bucle principal
        (ej. el mismo que ya usás para el contador de FPS en main.py)."""
        if dt_segundos <= 0:
            return
        self._fps_ventana.append(1.0 / dt_segundos)
        if len(self._fps_ventana) < self._fps_ventana.maxlen:
            return   # todavía no hay suficiente historia para decidir nada

        fps_prom = sum(self._fps_ventana) / len(self._fps_ventana)
        candidato = self._nivel_por_fps(fps_prom)

        if candidato == self._nivel_candidato:
            self._contador_estable += 1
        else:
            self._nivel_candidato = candidato
            self._contador_estable = 0

        if self._contador_estable >= self._frames_confirmacion and candidato != self._nivel_actual:
            print(f"[optimizacion_objetos] Calidad dinámica: '{self._nivel_actual}' → "
                  f"'{candidato}' (fps promedio {fps_prom:.1f}).")
            self._nivel_actual = candidato

    def _nivel_por_fps(self, fps_prom: float) -> str:
        actual = self._nivel_actual
        idx = NIVELES.index(actual)
        if actual in _UMBRAL_BAJAR and fps_prom < _UMBRAL_BAJAR[actual]:
            return NIVELES[min(idx + 1, len(NIVELES) - 1)]
        if actual in _UMBRAL_SUBIR and fps_prom > _UMBRAL_SUBIR[actual]:
            return NIVELES[max(idx - 1, 0)]
        return actual

    @property
    def nivel_actual(self) -> str:
        return self._nivel_actual


# ---------------------------------------------------------------------------
# Sincronización con el entorno — aplica el nivel del gestor a cada figura
# ---------------------------------------------------------------------------

_cache_malla_info: dict[str, dict] = {}
_ultimo_nivel_por_figura: dict[str, str] = {}


def _malla_info_de(nombre: str) -> dict | None:
    if nombre not in _cache_malla_info:
        registro = objetos.cargar_objeto(nombre)
        _cache_malla_info[nombre] = (registro or {}).get("malla")
    return _cache_malla_info[nombre]


def invalidar_cache(nombre: str | None = None) -> None:
    """Llamar cuando la malla de un objeto cambió en disco (ej. justo
    después de optimizar_objeto_construido(), o de reemplazar la malla a
    mano desde el editor visual) — si no, sincronizar_calidad_entorno()
    sigue viendo la versión vieja cacheada en memoria de proceso."""
    if nombre is None:
        _cache_malla_info.clear()
        _ultimo_nivel_por_figura.clear()
    else:
        _cache_malla_info.pop(nombre, None)
        _ultimo_nivel_por_figura.pop(nombre, None)


def sincronizar_calidad_entorno(entorno, gestor: GestorCalidadDinamica) -> None:
    """Llamar UNA vez por frame, antes de entorno.dibujar(panel) /
    renderizar_entorno(). Para cada figura con Malla real, le asigna la
    malla que corresponde al nivel actual del gestor. Costo: O(1) por
    figura salvo que el nivel haya cambiado desde el frame anterior para
    esa figura puntual, en cuyo caso reconstruye dos Mallas (from_dict) una
    sola vez y no de nuevo hasta el próximo cambio de nivel — nunca decima,
    nunca recorre vértices en el camino caliente.
    """
    nivel = gestor.nivel_actual
    for figura in entorno.figuras:
        nombre = figura.nombre
        if not figura.es_malla() or not nombre:
            continue
        if _ultimo_nivel_por_figura.get(nombre) == nivel:
            continue   # ya está en el nivel correcto, nada que hacer este frame

        malla_info = _malla_info_de(nombre)
        if not malla_info:
            continue

        clave_baja, clave_alta = _CLAVES_POR_NIVEL.get(nivel, ("lod_bajo", "lod_alto"))
        dict_bajo = malla_info.get(clave_baja) or malla_info.get("lod_bajo")
        dict_alto = malla_info.get(clave_alta) or dict_bajo
        if not dict_bajo:
            continue

        figura.malla_lod_baja = malla_mod.Malla.from_dict(dict_bajo)
        figura.malla_lod_alta = malla_mod.Malla.from_dict(dict_alto) if dict_alto else None
        _ultimo_nivel_por_figura[nombre] = nivel


if __name__ == "__main__":
    print("=== Optimizando todo el catálogo (objetos_db.json) ===")
    optimizar_todos_los_objetos()
