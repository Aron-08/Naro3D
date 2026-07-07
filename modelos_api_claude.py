"""
modelos_api_claude.py — PROPUESTA de integración con la API de Claude
(Anthropic), pensada como reemplazo drop-in de `ia_interprete._llamar_modelo`
/ `modelos.llamar()`.

*** ESTADO: NO EJECUTADO, NO TESTEADO ***
La API de Anthropic es de pago por token (no hay capa gratuita permanente
para uso sostenido — solo un crédito de prueba único al crear cuenta). Este
archivo documenta CÓMO se conectaría el pipeline a la API real, con el mismo
contrato que ya usa el proyecto (`modelos.llamar(skill, ...)` leyendo
`modelos_config.json`), pero no fue corrido contra la API real por esa
restricción de presupuesto. Se deja como evidencia de diseño, no como código
en producción.

Por qué esto es un reemplazo "drop-in" y no una reescritura:
    El resto del proyecto (ensamblador.py, calculo_estructural.py,
    termodinamica.py, electrico.py, geometria.py, ubicacion.py, objetos.py)
    llama siempre a `modelos.llamar(skill, messages=... | user_content=...)`.
    Nunca llama a Ollama directo. Eso significa que cambiar el backend de
    inferencia es, en teoría, un cambio en UN solo lugar (`modelos.py`),
    no en cada skill — la separación de capas que ya tenía el proyecto
    (skill 00_skill_filtro_ruido_datos.md: "el LLM decide criterio, Python
    calcula") es justo lo que hace posible este swap sin tocar el resto.

Diferencias reales a resolver si se migra de verdad (no triviales):
    1. Formato JSON nativo: Ollama tiene `format="json"` (usado por la
       skill "materiales" y "composicion_parametrica"/"reparacion_..."/
       "composicion_verificacion"). La API de Claude no tiene un modo JSON
       nativo equivalente; se logra pidiéndolo explícitamente en el prompt
       + parseo tolerante (el proyecto YA tiene esa red de seguridad:
       `ia_interprete._extraer_json`), o usando tool_use con un schema
       obligatorio (más robusto, pero cambia el flujo de parseo).
    2. `think=False` / razonamiento oculto: Ollama expone `think=False`
       para modelos con cadena de razonamiento visible (deepseek-r1). En
       la API de Claude el equivalente es simplemente no pedir
       `extended thinking` (no está activado por default), así que no
       hace falta ningún parámetro especial — pero si en algún momento se
       usa thinking, hay que filtrar los bloques `thinking` de la
       respuesta antes de pasarla a los parsers `_parsear_*` existentes.
    3. Modelo único vs. modelo por skill: hoy cada skill en
       modelos_config.json elige un modelo LOCAL distinto según la tarea
       (qwen3:4b para geometría, phi4-mini para materiales/estructural,
       deepseek-r1:7b para razonamiento libre, nemotron-mini:4b para
       redacción corta). Con la API de Claude, el equivalente natural es
       mapear "tarea de criterio simple, prompt corto" -> Haiku, y "tarea
       con más ambigüedad/razonamiento" -> Sonnet — nunca Opus para estas
       skills, sería pagar de más por una decisión de una línea.
    4. Costo por llamada: como cada objeto nuevo dispara varias llamadas
       encadenadas (concepto -> composición -> verificación -> reparación
       si hace falta -> propiedades -> propiedades extendidas), migrar TODO
       el pipeline a la API tiene un costo por objeto creado, no por sesión
       — importa al elegir qué skills migrar primero si el presupuesto es
       limitado (empezar por las de prompt más corto y salida más chica).

Requiere (si se llegara a ejecutar): pip install anthropic --break-system-packages
y la variable de entorno ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os

try:
    import anthropic
    _ANTHROPIC_DISPONIBLE = True
except ImportError:
    _ANTHROPIC_DISPONIBLE = False


# ---------------------------------------------------------------------------
# Mapeo skill -> modelo de Claude. Mismo espíritu que modelos_config.json:
# tareas de criterio corto y bajo ambiguedad -> Haiku (barato, rápido);
# tareas con más razonamiento/ambigüedad (composición paramétrica, concepto,
# verificación semántica) -> Sonnet. Nunca Opus acá: ninguna de estas skills
# necesita el modelo más caro para decidir una fórmula o clasificar una
# topología en una palabra.
# ---------------------------------------------------------------------------

_MODELO_POR_SKILL_CLAUDE = {
    # Criterio corto, formato de una línea por dato -> Haiku alcanza y sobra
    "ubicacion_espacial":        "claude-haiku-4-5-20251001",
    "geometria":                 "claude-haiku-4-5-20251001",
    "termodinamica":             "claude-haiku-4-5-20251001",
    "calculo_estructural":       "claude-haiku-4-5-20251001",
    "electrico":                 "claude-haiku-4-5-20251001",
    "gesto_color":               "claude-haiku-4-5-20251001",
    "asesor_modos":               "claude-haiku-4-5-20251001",
    "expansor_prompt":            "claude-haiku-4-5-20251001",
    # Razonamiento libre / descomposición estructural más abierta -> Sonnet
    "materiales":                       "claude-sonnet-5",
    "propiedades_fisicas_basicas":       "claude-sonnet-5",
    "propiedades_fisicas_actualizar":    "claude-sonnet-5",
    "dibujo_paso0a_razonamiento":        "claude-sonnet-5",
    "dibujo_paso1_coordenadas":          "claude-sonnet-5",
    "composicion_parametrica":          "claude-sonnet-5",
    "reparacion_composicion_parametrica": "claude-sonnet-5",
    "composicion_concepto":             "claude-sonnet-5",
    "composicion_verificacion":         "claude-haiku-4-5-20251001",  # es solo un chequeo tipo OK/PROBLEMA
}

_cliente = None


def _obtener_cliente():
    global _cliente
    if _cliente is None:
        if not _ANTHROPIC_DISPONIBLE:
            raise ImportError(
                "El paquete 'anthropic' no está instalado. "
                "pip install anthropic --break-system-packages"
            )
        _cliente = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _cliente


def llamar_claude(skill: str, messages: list, temperatura: float = 0.2,
                   num_predict: int = 1024) -> str | None:
    """Mismo contrato de salida que `ia_interprete._llamar_modelo`: devuelve
    el texto de la respuesta, o None si falló (nunca lanza hacia arriba,
    para no romper el patrón de "nunca None en cadena" del resto del
    pipeline — quien llama ya sabe tratar un None como "el modelo no
    respondió, seguir con el fallback").

    `messages`: misma forma que ya usa el proyecto,
    [{"role": "system"|"user", "content": str}, ...] — se traduce acá al
    formato de la API de Claude (system aparte, no como mensaje de rol).
    """
    modelo = _MODELO_POR_SKILL_CLAUDE.get(skill, "claude-haiku-4-5-20251001")

    system_prompt = ""
    mensajes_usuario = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            mensajes_usuario.append({"role": m["role"], "content": m["content"]})

    try:
        cliente = _obtener_cliente()
        respuesta = cliente.messages.create(
            model=modelo,
            max_tokens=num_predict if num_predict and num_predict > 0 else 1024,
            temperature=temperatura,
            system=system_prompt,
            messages=mensajes_usuario,
        )
        # La API de Claude puede devolver varios bloques de contenido
        # (texto, tool_use, etc.) — los parsers existentes del proyecto
        # (_parsear_respuesta_termica, _parsear_respuesta_estructural, ...)
        # esperan un string plano, así que se concatenan solo los bloques
        # de texto, igual criterio que ya usa el proyecto al combinar
        # bloques de respuestas de herramientas en otros contextos.
        return "".join(
            bloque.text for bloque in respuesta.content
            if getattr(bloque, "type", None) == "text"
        )
    except Exception as e:
        print(f"[modelos_api_claude] No se pudo llamar a Claude ('{modelo}', skill '{skill}'): {e}")
        return None


# ---------------------------------------------------------------------------
# Punto de integración con modelos.py — cómo quedaría el swap real
# ---------------------------------------------------------------------------
# En modelos.py, la función `llamar()` termina con:
#
#     from ia_interprete import _llamar_modelo
#     return _llamar_modelo(messages=messages, num_predict=np_,
#                            temperatura=temp, modelo=c["modelo"],
#                            formato=c.get("formato"))
#
# El cambio mínimo (una sola línea, sin tocar ninguna skill individual)
# sería agregar un campo "backend" a cada bloque de modelos_config.json
# ("ollama" | "claude", default "ollama" para no romper nada existente) y
# bifurcar acá:
#
#     if c.get("backend") == "claude":
#         return llamar_claude(skill, messages, temperatura=temp, num_predict=np_)
#     return _llamar_modelo(messages=messages, num_predict=np_,
#                            temperatura=temp, modelo=c["modelo"],
#                            formato=c.get("formato"))
#
# Esto permite migrar skill por skill (probar una, dejar el resto en Ollama)
# sin un big-bang de reescritura — coherente con cómo ya está armado el
# resto del proyecto (una skill a la vez, con su propio bloque de config).


if __name__ == "__main__":
    print("=== Prueba de humo (requiere ANTHROPIC_API_KEY y saldo) ===")
    if not _ANTHROPIC_DISPONIBLE:
        print("Paquete 'anthropic' no instalado; instalar con:")
        print("  pip install anthropic --break-system-packages")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        print("Falta la variable de entorno ANTHROPIC_API_KEY. No se ejecuta la prueba real "
              "(este archivo es una propuesta de diseño, ver docstring de nivel de módulo).")
    else:
        texto = llamar_claude(
            "electrico",
            messages=[
                {"role": "system", "content":
                    "Respondé solo: TOPOLOGIA: serie\nMODO_FALLA: ninguno\nNOTA: prueba."},
                {"role": "user", "content": "prueba de conexión"},
            ],
        )
        print("Respuesta:", texto)
