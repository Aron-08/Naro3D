"""
optimizacion_malla.py — Decimación y serialización de mallas: paso
OBLIGATORIO para cualquier malla que vaya a vivir en el entorno o en la
biblioteca, sea curada a mano o generada por IA.

Por qué existe (textual del pedido de Aron): "1 o 2 STL pesados destruyen
cualquier computadora" — un STL de TripoSR sale con decenas de miles de
caras; un asset curado de Kenney/Poly Pizza puede venir igual de pesado sin
que se note hasta que el entorno intenta dibujarlo a 30fps. Ninguna malla
entra al entorno o a la biblioteca sin pasar por acá primero.

Ver PLAN_RECONSTRUCCION_MALLAS.md, sección 4.2 (esta función) y sección 5
(formato JSON de biblioteca/caché).
"""

from __future__ import annotations

import datetime
import json

from malla import Malla

LOD_BAJO = 150    # caras — figura "congelada" (uso normal, sección 4.6 del plan)
LOD_ALTO = 1500   # caras — solo mientras la figura está "activa" (agarrada)


def decimar(malla: Malla, max_caras: int) -> Malla:
    """Reduce `malla` a un tope duro de `max_caras` triángulos con quadric
    edge collapse (pyfqmr). Si la malla ya tiene menos caras que el tope,
    se devuelve tal cual (no hace falta "decimar hacia arriba").

    Requiere `trimesh`/`numpy`/`pyfqmr`. Si no están instalados, se lanza
    ImportError explícito en vez de devolver la malla cruda sin decimar:
    dejar pasar una malla de decenas de miles de caras "por las dudas"
    es exactamente el escenario que esta función existe para evitar.
    """
    if malla.num_caras() <= max_caras:
        return malla

    try:
        import numpy as np
        import pyfqmr
    except ImportError as e:
        raise ImportError(
            "decimar() requiere 'numpy' y 'pyfqmr' (pip install pyfqmr). "
            "No instalados en este entorno — sin esto no hay tope duro de "
            "caras, así que por seguridad no se decima ni se deja pasar "
            "la malla cruda."
        ) from e

    vertices = np.array(malla.vertices, dtype=np.float64)
    caras = np.array(malla.caras, dtype=np.int64)

    simplificador = pyfqmr.Simplify()
    simplificador.setMesh(vertices, caras)
    simplificador.simplify_mesh(target_count=max_caras, aggressiveness=7,
                                 preserve_border=True, verbose=False)
    v_out, f_out, _ = simplificador.getMesh()

    return Malla(
        vertices=[tuple(float(c) for c in v) for v in v_out],
        caras=[tuple(int(i) for i in f) for f in f_out],
    )


def serializar_json(nombre: str, malla_lod_bajo: Malla, malla_lod_alto: "Malla | None",
                     origen: str, radio_bounding: "float | None" = None,
                     hash_pedido: "str | None" = None) -> dict:
    """Produce el formato de biblioteca (sección 5 del plan):

        {
          "nombre": ..., "origen": "ia_generada"|"curada",
          "hash_pedido": ..., "creado": "YYYY-MM-DD",
          "radio_bounding": float,
          "lod_bajo": {"v":[...], "f":[...]},
          "lod_alto": {"v":[...], "f":[...]} | null
        }

    `radio_bounding` se calcula desde `malla_lod_bajo` si no se pasa
    explícito: el LOD alto, cuando existe, es la misma silueta con más
    detalle, no una malla distinta, así que no debería tener un bounding
    mayor.
    """
    if radio_bounding is None:
        radio_bounding = malla_lod_bajo.radio_bounding()

    return {
        "nombre": nombre,
        "origen": origen,
        "hash_pedido": hash_pedido or "",
        "creado": datetime.date.today().isoformat(),
        "radio_bounding": round(radio_bounding, 4),
        "lod_bajo": malla_lod_bajo.to_dict(),
        "lod_alto": (malla_lod_alto.to_dict() if malla_lod_alto is not None else None),
    }


def cargar_json(ruta: str) -> tuple:
    """Devuelve (malla_lod_bajo, malla_lod_alto) desde un JSON de
    biblioteca. `malla_lod_alto` puede ser None si esa entrada no lo trae.
    El .stl archivado al lado (si existe) NUNCA se toca acá — es solo
    respaldo/trazabilidad, nunca se carga en runtime (sección 5 del plan)."""
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    lod_bajo = Malla.from_dict(datos["lod_bajo"])
    lod_alto_datos = datos.get("lod_alto")
    lod_alto = Malla.from_dict(lod_alto_datos) if lod_alto_datos else None
    return lod_bajo, lod_alto
