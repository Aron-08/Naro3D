"""
malla_ia_async.py — Generación de malla por IA: texto -> imagen (SD-Turbo)
-> malla 3D (TripoSR), en background, con un lock exclusivo de GPU.

Por qué GPU con lock exclusivo: en una GTX 1650 Ti de 4GB, SD-Turbo
(~2.5-3.5GB) y TripoSR (~3-4GB) no entran juntos en VRAM. Se cargan y
descargan secuencialmente, nunca las dos etapas residentes a la vez — mismo
criterio que objetos.py::crear_objeto ya aplica para no tener dos contextos
de Ollama vivos al mismo tiempo (regla de la sección 4.4 del plan), acá
aplicado a la GPU en vez de a RAM/Ollama.

Este módulo NUNCA bloquea la creación de un objeto: la figura ya se dibujó
por biblioteca o por el fallback LLM antes de que esto termine (de ~15-20s
a 1-2 min, sin benchmark real todavía en la GTX 1650 Ti — ver sección 4.4
y 9 del plan). `solicitar()` corre en threading.Thread(daemon=True) y avisa
por callback cuando termina; quien llama (objetos.py) lo reenvía a
`_cola_mallas` en main.py, mismo patrón que `_cola_propiedades`.

Requiere: torch (CUDA), diffusers, transformers, accelerate, y TripoSR
(VAST-AI-Research/TripoSR, clonado aparte — no es paquete de PyPI). Si no
están instalados, `solicitar()` avisa por consola y no hace nada: no rompe
el resto del pipeline, que sigue andando con biblioteca + fallback LLM.
"""

from __future__ import annotations

import os
import tempfile
import threading

import malla as malla_mod
import optimizacion_malla as opt
import biblioteca_mallas as biblioteca

_LOCK_GPU = threading.Lock()

PREFIJO_PROMPT_T2I = "product photo, plain white background, single object, centered"
MC_RESOLUTION = 128  # por debajo del default 256 de TripoSR, para bajar VRAM/tiempo en 4GB


def _dependencias_disponibles() -> bool:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError:
        return False
    return True


def _forzar_ollama_cpu() -> None:
    """Antes de tocar la GPU para SD-Turbo/TripoSR, deja los 4GB libres
    enteros para esta parte — los modelos de Ollama del fallback ya corren
    bien en CPU en el pipeline actual, así que esto no degrada nada más
    (ver sección 4.4 y presupuesto de VRAM, sección 6 del plan)."""
    os.environ["OLLAMA_NUM_GPU"] = "0"


def generar_imagen(pedido: str):
    """Carga SD-Turbo (1 paso, fp16), genera una imagen 512x512 de fondo
    limpio y la devuelve (PIL.Image), descargando el pipeline de VRAM antes
    de salir. Se asume que ya se tiene `_LOCK_GPU` adquirido — esta función
    no lo toma ella misma, para que quien orquesta (`_generar_sync`) pueda
    encadenar generar_imagen + imagen_a_malla bajo un único lock."""
    import torch
    from diffusers import AutoPipelineForText2Image

    prompt = f"{PREFIJO_PROMPT_T2I}, {pedido}"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16",
    )
    pipe.to("cuda")
    try:
        imagen = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    finally:
        del pipe
        torch.cuda.empty_cache()
    return imagen


def imagen_a_malla(imagen) -> malla_mod.Malla:
    """Carga TripoSR (mc-resolution 128), genera la malla cruda a partir de
    `imagen` y la devuelve SIN decimar todavía (eso corre en CPU, fuera de
    `_LOCK_GPU`, ver optimizacion_malla.decimar). Descarga el modelo de
    VRAM antes de salir. Requiere el repo VAST-AI-Research/TripoSR
    clonado e instalado (no está en PyPI, ver sección 7 del plan)."""
    import torch
    try:
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground
    except ImportError as e:
        raise ImportError(
            "imagen_a_malla requiere TripoSR instalado desde "
            "VAST-AI-Research/TripoSR (no está en PyPI, ver sección 7 del plan)."
        ) from e

    modelo = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt",
    )
    modelo.renderer.set_chunk_size(8192)
    modelo.to("cuda")
    try:
        imagen_limpia = resize_foreground(remove_background(imagen), 0.85)
        with torch.no_grad():
            codes = modelo([imagen_limpia], device="cuda")
            malla_tm = modelo.extract_mesh(codes, resolution=MC_RESOLUTION)[0]
    finally:
        del modelo
        torch.cuda.empty_cache()

    centro = malla_tm.bounding_box.centroid
    vertices = [tuple(float(c) for c in (v - centro)) for v in malla_tm.vertices]
    caras = [tuple(int(i) for i in f) for f in malla_tm.faces]
    return malla_mod.Malla(vertices=vertices, caras=caras)


def _generar_sync(pedido: str) -> "dict | None":
    """Flujo completo, síncrono (corre dentro del hilo daemon de
    `solicitar`). Devuelve el dict serializado (formato de biblioteca) o
    None si algo falló (deps faltantes, sin GPU, error de generación,
    etc.) — un fallo acá nunca debe tirar el resto del programa: el objeto
    ya tiene la geometría del fallback LLM (o de biblioteca), en el peor
    caso se queda con esa."""
    if not _dependencias_disponibles():
        print("[malla_ia_async] torch/diffusers no disponibles; se omite generación IA de malla.")
        return None

    _forzar_ollama_cpu()
    ruta_stl_tmp = None

    try:
        with _LOCK_GPU:
            imagen = generar_imagen(pedido)
            malla_cruda = imagen_a_malla(imagen)

        # Fuera del lock: esto es CPU, no compite por GPU.
        if malla_cruda.num_caras() == 0:
            print(f"[malla_ia_async] '{pedido}': malla cruda vacía, se descarta.")
            return None

        malla_lod_baja = opt.decimar(malla_cruda, opt.LOD_BAJO)
        malla_lod_alta = opt.decimar(malla_cruda, opt.LOD_ALTO)

        # Archivo original de respaldo/trazabilidad — nunca se vuelve a
        # cargar en runtime, solo queda archivado al lado del .json.
        try:
            import trimesh
            tm = trimesh.Trimesh(vertices=malla_cruda.vertices, faces=malla_cruda.caras)
            fd, ruta_stl_tmp = tempfile.mkstemp(suffix=".stl")
            os.close(fd)
            tm.export(ruta_stl_tmp)
        except Exception as e:
            print(f"[malla_ia_async] No se pudo archivar STL original de '{pedido}': {e}")
            ruta_stl_tmp = None

        malla_json = opt.serializar_json(
            pedido, malla_lod_baja, malla_lod_alta, origen="ia_generada",
        )
        biblioteca.registrar_nueva(pedido, malla_json, origen="ia_generada",
                                    ruta_stl_original=ruta_stl_tmp)
        return malla_json

    except Exception as e:
        print(f"[malla_ia_async] Falló la generación IA de malla para '{pedido}': {e}")
        return None
    finally:
        if ruta_stl_tmp and os.path.exists(ruta_stl_tmp):
            try:
                os.remove(ruta_stl_tmp)
            except OSError:
                pass


def generar_malla_normalizada_sincrona(pedido: str, radio_cm_objetivo: float = 30.0) -> "malla_mod.Malla | None":
    """Punto de entrada usado por `objetos.py::generar_geometria_parametrica`
    como fallback ESPECÍFICO para el caso 'factible: false' del kernel
    paramétrico (ver horizonte "Reconectar TripoSR"): cuando el modelo de
    composición dice honestamente que el objeto es una silueta orgánica
    irregular (no una combinación razonable de caja/cilindro/esfera/...),
    ese es el momento de invocar el pipeline generativo real en vez de
    forzarlo a una caja genérica de 35cm.

    A diferencia de `solicitar()` (pensada para disparar en background y
    reemplazar una figura ya dibujada más tarde, vía callback), esta
    función es SÍNCRONA/bloqueante: `generar_geometria_parametrica` ya es
    una función bloqueante en sí misma (espera reintentos del LLM de
    composición antes de devolver), así que bloquear acá también encaja
    con el contrato existente — quien llama ya está esperando a que la
    geometría esté lista antes de seguir.

    Corre el mismo flujo completo que `solicitar()` (texto -> imagen
    SD-Turbo -> malla TripoSR -> decimar -> archivar en biblioteca) vía
    `_generar_sync`, y después RE-ESCALA el resultado para que quede en el
    mismo espacio de centímetros reales que usa `ensamblador.py` — el
    kernel paramétrico trabaja en cm reales (ver `malla.PX_POR_CM`),
    mientras que la malla cruda de TripoSR sale en las unidades propias
    del modelo (sin relación con cm). Como no hay forma de saber la escala
    real del objeto a partir de la malla generada, se la normaliza a un
    radio de bounding fijo (`radio_cm_objetivo`, default 30cm — del orden
    de un objeto de escritorio mediano) en vez de asumir 1:1. Es una
    aproximación de escala, igual de razonable que la que ya hace la caja
    genérica de seguridad (35cm fijos) pero con la FORMA real del objeto en
    vez de un cubo.

    Devuelve `None` si las dependencias (torch/diffusers/TripoSR) no están
    instaladas, si la generación falla, o si la malla resultante quedó
    vacía — en todos esos casos, quien llama debe seguir con el siguiente
    escalón de la red de seguridad (plantilla determinística / caja
    genérica), nunca romper el pipeline."""
    resultado = _generar_sync(pedido)
    if resultado is None:
        return None

    try:
        m = malla_mod.Malla.from_dict(resultado["lod_bajo"])
    except (KeyError, TypeError, ValueError) as e:
        print(f"[malla_ia_async] Resultado de TripoSR para '{pedido}' no tiene el formato "
              f"esperado ({e}); se descarta.")
        return None

    radio = m.radio_bounding()
    if m.num_vertices() == 0 or radio <= 0:
        print(f"[malla_ia_async] Malla generada para '{pedido}' vacía o degenerada; se descarta.")
        return None

    factor = radio_cm_objetivo / radio
    vertices_cm = [(x * factor, y * factor, z * factor) for x, y, z in m.vertices]
    return malla_mod.Malla(vertices=vertices_cm, caras=m.caras)


def solicitar(pedido: str, callback_terminado=None) -> None:
    """Encola la generación IA de `pedido` en un hilo daemon, sin bloquear
    a quien llama. `callback_terminado` (si se pasa) se invoca con
    (pedido, malla_json | None) cuando termina — quien llama
    (objetos.py -> main.py) lo usa para alimentar `_cola_mallas` y
    reemplazar la figura primitiva por la malla real en el entorno
    (sección 3, paso 3 del plan)."""
    def _correr():
        resultado = _generar_sync(pedido)
        if callback_terminado:
            callback_terminado(pedido, resultado)

    threading.Thread(target=_correr, daemon=True).start()