"""
modelos.py — Punto único de lectura de `modelos_config.json`.

Por qué existe: hasta ahora todo el proyecto llamaba a un solo modelo fijo
(`ia_interprete.MODELO`) para todo — descripción, geometría, ubicación,
materiales, térmico, estructural. En la práctica cada una de esas tareas
tiene un perfil distinto (razonamiento libre vs. clasificación estricta vs.
JSON estructurado), así que ahora cada skill puede usar un modelo chico
especializado en lugar de uno solo generalista.

Este módulo NO llama a Ollama directamente — solo resuelve, para cada skill,
qué modelo/temperatura/num_predict/formato le corresponde, y delega la
llamada real a `ia_interprete._llamar_modelo` (que ya tiene el lock global,
el manejo de `think=False` y el fallback de `options`). Así no se duplica esa
lógica en cada módulo nuevo.

Para cambiar qué modelo usa una skill (por ejemplo porque en una de las dos
PCs sin GPU resulta muy pesado), se edita `modelos_config.json` directamente
— no hace falta tocar ningún archivo .py. El archivo se relee en cada
llamada (`_cargar()` es barato: es un JSON chico), así que un cambio hecho
mientras el proceso ya está corriendo se aplica en la siguiente llamada, sin
reiniciar nada.
"""

import json
import os
import random

_RUTA_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos_config.json")


def _cargar() -> dict:
    with open(_RUTA_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def config(skill: str) -> dict:
    """Devuelve el bloque de configuración de una skill (modelo, temperaturas,
    num_predict, formato, system). Lanza KeyError con nombres válidos si la skill
    no existe en el JSON — mejor fallar fuerte acá que silenciosamente usar un
    modelo por defecto equivocado."""
    datos = _cargar()
    skills = datos.get("skills", {})
    if skill not in skills:
        disponibles = ", ".join(sorted(skills.keys()))
        raise KeyError(f"Skill '{skill}' no está en modelos_config.json. Disponibles: {disponibles}")
    return skills[skill]


def temperatura_para(skill: str) -> float:
    """Elige una temperatura dentro del rango [temperatura_min, temperatura_max]
    definido para la skill. Rango colapsado (min == max) devuelve ese valor fijo."""
    c = config(skill)
    lo, hi = float(c["temperatura_min"]), float(c["temperatura_max"])
    if hi <= lo:
        return lo
    return round(random.uniform(lo, hi), 3)


def llamar(skill: str, messages: list | None = None,
           user_content: str | None = None,
           num_predict: int | None = None,
           temperatura: float | None = None) -> str | None:
    """Llama al modelo configurado para `skill`, con su temperatura/num_predict
    salvo que se pasen explícitos.

    Modos de uso:
    1) Pasando `messages`: se usa tal cual (compatibilidad con código antiguo).
    2) Pasando `user_content`: se construye automáticamente la lista de mensajes
       con el system prompt definido en modelos_config.json para esa skill.
    Si se pasan ambos, se ignora `messages` y se usa `user_content`.

    Devuelve el mismo contrato que ia_interprete._llamar_modelo (texto o None si no respondió).
    """
    # Import diferido: ia_interprete importa módulos de skills (objetos.py lo hace
    # indirectamente), así que importar esto arriba del archivo generaría un
    # import circular. Se resuelve en el momento de la llamada, no al cargar el módulo.
    from ia_interprete import _llamar_modelo

    c = config(skill)

    # Construir mensajes si se proporciona user_content
    if user_content is not None:
        system_prompt = c.get("system")
        if system_prompt is None:
            # Si no hay system definido, se usa un fallback o se lanza error.
            # Para no romper, lo dejamos vacío, pero mejor advertir.
            print(f"[modelos] Advertencia: skill '{skill}' no tiene 'system' definido en modelos_config.json")
            system_prompt = ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    elif messages is None:
        raise ValueError("Debe proporcionar 'messages' o 'user_content'")

    temp = temperatura if temperatura is not None else temperatura_para(skill)
    np_ = num_predict if num_predict is not None else int(c.get("num_predict", -1))

    return _llamar_modelo(
        messages=messages,
        num_predict=np_,
        temperatura=temp,
        modelo=c["modelo"],
        formato=c.get("formato"),
    )