"""
biblioteca_mallas.py — Índice + retrieval de mallas ya generadas o curadas.

Reemplaza como fuente PRINCIPAL de geometría al pipeline de coordenadas vía
LLM (ia_interprete.py, pasos -1 a 2b), que baja de rango a fallback de
última instancia (ver PLAN_RECONSTRUCCION_MALLAS.md, sección 0).

La biblioteca es auto-creciente: cada vez que malla_ia_async.py termina de
generar algo, se archiva acá con `registrar_nueva()`, así la próxima vez
que se pida algo parecido resuelve por `buscar()` (milisegundos) en vez de
volver a pasar por el pipeline de generación (segundos a minutos).

Estructura en disco (análoga a `figuras_cache/` que ya existe en el
proyecto):
    biblioteca_mallas/
        _indice.json       # nombre -> {embedding, ruta_json, ruta_stl, origen}
        <slug>.json          # LOD bajo + LOD alto, listo para cargar
        <slug>.stl            # original archivado, NUNCA se carga en runtime

El embedding se calcula con la skill nueva `embeddings_biblioteca` de
modelos_config.json (modelo nomic-embed-text vía Ollama) — reusa la
infraestructura de Ollama que ya está en el proyecto, sin librería de
embeddings nueva. Ver sección 4.3 del plan.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass

import modelos
import optimizacion_malla as opt
from malla import Malla

CARPETA_BIBLIOTECA = "biblioteca_mallas"
RUTA_INDICE = os.path.join(CARPETA_BIBLIOTECA, "_indice.json")

UMBRAL_DEFAULT = 0.75


@dataclass
class ResultadoBiblioteca:
    nombre: str
    radio_bounding: float
    malla_lod_baja: Malla
    malla_lod_alta: "Malla | None"
    origen: str


def _slug(nombre: str) -> str:
    s = nombre.strip().lower()
    s = re.sub(r"[^a-z0-9áéíóúñ]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "objeto"


def _asegurar_carpeta() -> None:
    os.makedirs(CARPETA_BIBLIOTECA, exist_ok=True)


def _cargar_indice() -> dict:
    if not os.path.exists(RUTA_INDICE):
        return {}
    try:
        with open(RUTA_INDICE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[biblioteca_mallas] Error al leer {RUTA_INDICE}: {e}")
        return {}


def _guardar_indice(indice: dict) -> None:
    _asegurar_carpeta()
    try:
        with open(RUTA_INDICE, "w", encoding="utf-8") as f:
            json.dump(indice, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[biblioteca_mallas] Error al guardar {RUTA_INDICE}: {e}")


def _embedding(texto: str) -> "list[float] | None":
    """Embedding del texto vía Ollama (skill 'embeddings_biblioteca',
    modelo nomic-embed-text). None si Ollama/el paquete no están
    disponibles o falla la llamada — quien use esto debe tratar None como
    "no se pudo, seguí con el flujo normal", nunca como un vector vacío
    válido para comparar."""
    try:
        import ollama
    except ImportError:
        print("[biblioteca_mallas] Paquete 'ollama' no disponible; no se puede calcular embedding.")
        return None
    try:
        c = modelos.config("embeddings_biblioteca")
        resp = ollama.embeddings(model=c["modelo"], prompt=texto)
        vector = list(resp.get("embedding") or [])
        return vector or None
    except Exception as e:
        print(f"[biblioteca_mallas] Error al calcular embedding de '{texto}': {e}")
        return None


def _similaridad_coseno(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    prod = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return prod / (na * nb)


def buscar(pedido: str, umbral: float = UMBRAL_DEFAULT) -> "ResultadoBiblioteca | None":
    """Busca en la biblioteca la entrada más parecida a `pedido` por
    similaridad coseno de embeddings. HIT (similaridad >= umbral) -> carga
    esa entrada con optimizacion_malla.cargar_json() y la devuelve. Si no
    hay ninguna por encima del umbral, o no se pudo calcular el embedding
    del pedido, o la biblioteca está vacía, devuelve None — quien llama
    (objetos.py::crear_objeto) sigue con el fallback LLM + generación IA
    en background."""
    indice = _cargar_indice()
    if not indice:
        return None

    emb_pedido = _embedding(pedido)
    if emb_pedido is None:
        return None

    mejor_nombre, mejor_sim = None, -1.0
    for nombre, entrada in indice.items():
        sim = _similaridad_coseno(emb_pedido, entrada.get("embedding", []))
        if sim > mejor_sim:
            mejor_nombre, mejor_sim = nombre, sim

    if mejor_nombre is None or mejor_sim < umbral:
        return None

    entrada = indice[mejor_nombre]
    ruta_json = entrada.get("ruta_json", "")
    if not ruta_json or not os.path.exists(ruta_json):
        print(f"[biblioteca_mallas] Índice apunta a '{ruta_json}' pero no existe; se ignora la entrada.")
        return None

    lod_bajo, lod_alto = opt.cargar_json(ruta_json)
    with open(ruta_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[biblioteca_mallas] HIT '{pedido}' -> '{mejor_nombre}' (similaridad {mejor_sim:.3f})")
    return ResultadoBiblioteca(
        nombre=mejor_nombre,
        radio_bounding=float(metadata.get("radio_bounding") or lod_bajo.radio_bounding()),
        malla_lod_baja=lod_bajo,
        malla_lod_alta=lod_alto,
        origen=metadata.get("origen", "desconocido"),
    )


def registrar_nueva(nombre: str, malla_json: dict, origen: str,
                     ruta_stl_original: "str | None" = None) -> None:
    """Archiva una malla nueva (generada por IA o curada a mano) en la
    biblioteca: guarda `<slug>.json` (+ `<slug>.stl` si se pasa un
    original) y agrega su embedding al índice. `malla_json` ya viene en el
    formato de optimizacion_malla.serializar_json()."""
    _asegurar_carpeta()
    slug = _slug(nombre)
    ruta_json = os.path.join(CARPETA_BIBLIOTECA, f"{slug}.json")

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(malla_json, f, ensure_ascii=False, indent=2)

    ruta_stl = None
    if ruta_stl_original and os.path.exists(ruta_stl_original):
        ruta_stl = os.path.join(CARPETA_BIBLIOTECA, f"{slug}.stl")
        try:
            shutil.copyfile(ruta_stl_original, ruta_stl)
        except OSError as e:
            print(f"[biblioteca_mallas] No se pudo archivar el STL original de '{nombre}': {e}")
            ruta_stl = None

    emb = _embedding(nombre)
    indice = _cargar_indice()
    indice[nombre] = {
        "embedding": emb or [],
        "ruta_json": ruta_json,
        "ruta_stl": ruta_stl,
        "origen": origen,
    }
    _guardar_indice(indice)
    print(f"[biblioteca_mallas] '{nombre}' registrado ({origen}) -> {ruta_json}")


def precargar_stl_curados(carpeta: str) -> None:
    """Recorre `carpeta` buscando .stl (assets curados, típicamente CC0 de
    Kenney/Poly Pizza) y los mete a la biblioteca: cada uno pasa por
    malla.malla_desde_stl() -> optimizacion_malla.decimar() (LOD bajo y
    alto) -> serializar_json() -> registrar_nueva(). No hay atajo para
    contenido "de confianza": todo pasa por el mismo tope duro de caras
    que una malla generada por IA (ver sección 4.3 del plan)."""
    from malla import malla_desde_stl

    if not os.path.isdir(carpeta):
        print(f"[biblioteca_mallas] Carpeta '{carpeta}' no existe.")
        return

    for archivo in sorted(os.listdir(carpeta)):
        if not archivo.lower().endswith(".stl"):
            continue
        ruta = os.path.join(carpeta, archivo)
        nombre = os.path.splitext(archivo)[0].replace("_", " ")
        try:
            cruda = malla_desde_stl(ruta)
            lod_bajo = opt.decimar(cruda, opt.LOD_BAJO)
            lod_alto = opt.decimar(cruda, opt.LOD_ALTO)
            malla_json = opt.serializar_json(nombre, lod_bajo, lod_alto, origen="curada")
            registrar_nueva(nombre, malla_json, origen="curada", ruta_stl_original=ruta)
        except Exception as e:
            print(f"[biblioteca_mallas] Falló precarga de '{archivo}': {e}")


if __name__ == "__main__":
    print("=== Prueba de biblioteca_mallas.py (sin GPU, biblioteca vacía) ===")
    r = buscar("licuadora de mano")
    print("buscar('licuadora de mano') ->", r)
