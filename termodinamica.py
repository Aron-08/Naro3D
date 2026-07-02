"""
termodinamica.py — Comportamiento térmico de un objeto ya definido (geometría
+ material), pensado para conectar con objetos.py (ficha de propiedades) y,
más adelante, con la experimentación de propulsión H2/O2 (combustión en
punta de pala) y el LAES criogénico.

Filosofía (ver skill 04_skill_termodinamica.md, y skill 00 — filtro de
ruido): el LLM NUNCA hace la cuenta de punta a punta. Se usa solo para elegir
CRITERIO — qué modelo térmico aplica (concentrado vs. gradiente), si el
proceso es físico o de combustión, y para redactar una advertencia corta en
lenguaje natural. Toda la aritmética (Fourier, calorimetría, estequiometría)
es Python puro con fórmulas de libro. Cualquier número que el modelo intente
meter en su "NOTA" se ignora — la nota es solo texto para mostrarle al
usuario, nunca fuente de verdad numérica.

Unidades: SI en todo el módulo (kg, m, m2, m3, s, K/°C, J, W). La geometría
del resto del proyecto vive en coordenadas de escena normalizadas [0,1]^3
(ver ubicacion.py / geometria.py), que NO son metros — por eso el volumen del
objeto se deriva de masa/densidad (peso_kg / densidad_kg_m3, ya presentes en
la ficha de objetos.py), no del área de la figura en pantalla. Ver
`geometria_termica_desde_ficha()`.

Requiere lo mismo que ia_interprete.py (Ollama corriendo + el modelo
configurado).
"""

import re

from ia_interprete import _llamar_modelo   # reutiliza el wrapper de Ollama ya probado


# ---------------------------------------------------------------------------
# Constantes físicas de referencia (fijas, NUNCA se le preguntan al modelo)
# ---------------------------------------------------------------------------

CERO_ABSOLUTO_C          = -273.15   # °C, límite físico infranqueable
PODER_CALORIFICO_H2_MJ_KG = 120.0    # MJ/kg, poder calorífico inferior del H2
MASA_MOLAR_H2_G_MOL       = 2.016
MASA_MOLAR_O2_G_MOL       = 31.998
TEMP_LLAMA_H2O2_C_RANGO   = (2800.0, 3000.0)   # referencia estequiométrica, solo para contraste

H_CONVECCION_AIRE_LIBRE_W_M2K = 10.0   # convección natural típica en aire quieto
H_CONVECCION_AIRE_FORZADO_W_M2K = 40.0  # con ventilación/movimiento de aire

TEMPERATURA_TERMICA = 0.15   # baja: esta skill decide criterio, no crea contenido

# Plantillas de seguridad si falta un dato de material indispensable (nunca
# se deja continuar con calor_especifico=0: eso rompe cualquier cuenta con
# división, incluida Q = m*c*ΔT despejada para ΔT).
_VALORES_SEGURIDAD_TERMICO = {
    "metal_generico":    {"calor_especifico_j_kgk": 460,  "conductividad_termica_w_mk": 40},
    "plastico_generico": {"calor_especifico_j_kgk": 1500, "conductividad_termica_w_mk": 0.3},
    "mineral_generico":  {"calor_especifico_j_kgk": 900,  "conductividad_termica_w_mk": 1.3},
    "organico_generico": {"calor_especifico_j_kgk": 1800, "conductividad_termica_w_mk": 0.15},
}


# ---------------------------------------------------------------------------
# Geometría térmica: volumen y espesor característico, SIEMPRE en Python
# ---------------------------------------------------------------------------
# Nunca se le pide al LLM que estime volumen de una figura (skill 04). El
# volumen sale de masa/densidad, que ya están resueltos por objetos.py; el
# área expuesta (si se conoce) da un espesor característico más realista
# (aproximación de placa delgada); si no se conoce, se usa la aproximación
# de cuerpo compacto (cubo equivalente).

def geometria_termica_desde_ficha(peso_kg: float, densidad_kg_m3: float,
                                   area_expuesta_m2: float | None = None) -> dict:
    """Devuelve {"volumen_estimado_m3", "espesor_caracteristico_m", "area_expuesta_m2"}.

    - volumen = masa / densidad (exacto, siempre que ambos sean > 0).
    - espesor: si se conoce el área expuesta real, volumen / área (placa
      delgada); si no, volumen**(1/3) (cubo equivalente) como aproximación
      conservadora de cuerpo compacto.
    """
    densidad = densidad_kg_m3 if densidad_kg_m3 and densidad_kg_m3 > 0 else 1000.0
    masa = peso_kg if peso_kg and peso_kg > 0 else 0.001
    volumen = masa / densidad

    if area_expuesta_m2 and area_expuesta_m2 > 0:
        espesor = volumen / area_expuesta_m2
        area = area_expuesta_m2
    else:
        espesor = volumen ** (1.0 / 3.0)
        area = 6.0 * espesor ** 2   # superficie de un cubo equivalente, para h*A

    return {
        "volumen_estimado_m3": volumen,
        "espesor_caracteristico_m": espesor,
        "area_expuesta_m2": area,
    }


# ---------------------------------------------------------------------------
# Saneo de entrada (capa 1 del filtro de ruido — skill 00)
# ---------------------------------------------------------------------------

def _categoria_material(nombre_material: str) -> str:
    """Heurística simple por palabras clave (mismo criterio que objetos.py
    usa para defaults), solo para elegir plantilla de seguridad."""
    texto = (nombre_material or "").lower()
    if any(p in texto for p in ("acero", "aluminio", "cobre", "hierro", "metal", "bronce", "titanio")):
        return "metal_generico"
    if any(p in texto for p in ("plastico", "pvc", "abs", "pla", "nylon", "polimero")):
        return "plastico_generico"
    if any(p in texto for p in ("hormigon", "concreto", "piedra", "ceramico", "vidrio", "mineral")):
        return "mineral_generico"
    if any(p in texto for p in ("madera", "tela", "cuero", "papel", "organico")):
        return "organico_generico"
    return "mineral_generico"


def _sanear_material(material: dict) -> dict:
    """Nunca deja calor_especifico o conductividad en 0/negativo — eso rompe
    cualquier cuenta térmica con división. Rellena con plantilla si falta."""
    m = dict(material or {})
    categoria = _categoria_material(m.get("material", ""))
    plantilla = _VALORES_SEGURIDAD_TERMICO[categoria]

    ce = m.get("calor_especifico_j_kgk")
    if not ce or ce <= 0:
        m["calor_especifico_j_kgk"] = plantilla["calor_especifico_j_kgk"]

    k = m.get("conductividad_termica_w_mk")
    if not k or k <= 0:
        m["conductividad_termica_w_mk"] = plantilla["conductividad_termica_w_mk"]

    if not m.get("densidad_kg_m3") or m["densidad_kg_m3"] <= 0:
        m["densidad_kg_m3"] = 1000.0

    return m


def _sanear_temperatura_c(valor: float, defecto: float = 20.0) -> tuple[float, str | None]:
    """Ninguna temperatura puede estar por debajo del cero absoluto — 0
    tolerancia (ver skill 04, tabla de rangos). Devuelve (valor_saneado, aviso|None)."""
    if valor is None:
        return defecto, None
    if valor < CERO_ABSOLUTO_C:
        return CERO_ABSOLUTO_C, (
            f"  Temperatura {valor}°C por debajo del cero absoluto — clampada a "
            f"{CERO_ABSOLUTO_C}°C (violación grave de entrada)"
        )
    return valor, None


# ---------------------------------------------------------------------------
# Fórmulas de libro (Python puro, ver skill 04)
# ---------------------------------------------------------------------------

def numero_biot(h_w_m2k: float, espesor_caracteristico_m: float, k_w_mk: float) -> float:
    """Bi = h * Lc / k. Bi < 0.1 -> aproximación de cuerpo concentrado válida."""
    if k_w_mk <= 0:
        return float("inf")
    return h_w_m2k * espesor_caracteristico_m / k_w_mk


def constante_tiempo_concentrado(masa_kg: float, calor_especifico_j_kgk: float,
                                  h_w_m2k: float, area_expuesta_m2: float) -> float:
    """tau = m*c / (h*A), en segundos. Cuanto mayor, más lento se estabiliza."""
    denom = h_w_m2k * area_expuesta_m2
    if denom <= 0:
        return float("inf")
    return (masa_kg * calor_especifico_j_kgk) / denom


def temperatura_transitoria_concentrado(temp_ambiente_c: float, temp_inicial_c: float,
                                         tiempo_s: float, tau_s: float) -> float:
    """T(t) = T_amb + (T0 - T_amb) * exp(-t/tau) — cuerpo concentrado (lumped
    capacitance), relajándose hacia la temperatura ambiente por convección."""
    import math
    if tau_s == float("inf") or tau_s <= 0:
        return temp_inicial_c
    return temp_ambiente_c + (temp_inicial_c - temp_ambiente_c) * math.exp(-tiempo_s / tau_s)


def masa_o2_estequiometrica_kg(masa_h2_kg: float) -> float:
    """2 H2 + O2 -> 2 H2O. Relación másica O2:H2 = M(O2) / (2*M(H2))."""
    return masa_h2_kg * (MASA_MOLAR_O2_G_MOL / (2.0 * MASA_MOLAR_H2_G_MOL))


def energia_combustion_h2_j(masa_h2_kg: float) -> float:
    """Q = poder_calorifico_kg * masa_combustible_kg, en Joules."""
    return PODER_CALORIFICO_H2_MJ_KG * 1e6 * masa_h2_kg


def delta_temperatura_combustion(energia_j: float, masa_objeto_kg: float,
                                  calor_especifico_j_kgk: float) -> float:
    """ΔT = Q / (m_objeto * c). Asume toda la energía absorbida por el
    objeto (cota superior conservadora — en la realidad hay pérdidas)."""
    denom = masa_objeto_kg * calor_especifico_j_kgk
    if denom <= 0:
        return 0.0
    return energia_j / denom


# ---------------------------------------------------------------------------
# Razonamiento guiado (única parte con LLM) — elegir criterio, no calcular
# ---------------------------------------------------------------------------

SYSTEM_TERMICO = """Analizás el comportamiento térmico de un objeto ya definido (geometría +
material + condiciones de borde). NO hacés ninguna cuenta numérica: todo el cálculo
(conducción de Fourier, calorimetría) ya lo hace Python con fórmulas exactas. Tu trabajo es
elegir el CRITERIO correcto y, si corresponde, redactar una advertencia corta.

Recibís un JSON con "geometria" (área, volumen, espesor característico), "material"
(conductividad, calor específico, densidad, temperatura máxima de servicio) y "condiciones"
(temperatura ambiente, temperatura de la fuente, tipo de proceso, modo).

Respondé ÚNICAMENTE con este formato, un dato por línea, sin texto adicional:

MODELO: <concentrado|gradiente>
BIOT_ESTIMADO: <bajo|alto>
PROCESO: <fisico|combustion>
RIESGO: <ninguno|excede_temp_servicio|excede_punto_fusion>
NOTA: una frase corta, opcional

Reglas:
  - MODELO "concentrado" (cuerpo a temperatura uniforme) es correcto casi siempre que el
    objeto sea chico o metálico (buena conductividad); "gradiente" solo si el objeto es
    grande y mal conductor Y el modo pedido es explícitamente espacial.
  - BIOT_ESTIMADO "bajo" acompaña a MODELO concentrado, "alto" acompaña a gradiente.
  - PROCESO "combustion" solo si tipo_proceso menciona combustión/H2/O2/quemado; si no,
    "fisico" (conducción/convección simple).
  - RIESGO es tu impresión cualitativa, no un cálculo — Python la recalcula y tiene
    prioridad si difieren. Usá "excede_temp_servicio" si la temperatura de la fuente ya
    supera claramente la temperatura máxima de servicio del material; "excede_punto_fusion"
    solo si es evidente que se derrite; si no ves riesgo claro, "ninguno".
  - NOTA: nunca metas números ahí (temperaturas, energías, tiempos) — se ignoran igual,
    y confunden si contradicen el cálculo real. Usala solo para una frase cualitativa.

=== EJEMPLO 1 ===
entrada: objeto chico de acero (espesor 0.02m), temp_ambiente=20°C, temp_fuente=850°C,
tipo_proceso=fisico, temp_max_servicio=400°C
MODELO: concentrado
BIOT_ESTIMADO: bajo
PROCESO: fisico
RIESGO: excede_temp_servicio
NOTA: la fuente supera la temperatura de servicio del acero, revisar exposición prolongada

=== EJEMPLO 2 ===
entrada: cámara de combustión H2/O2, temp_ambiente=20°C, temp_fuente=850°C,
tipo_proceso=combustion_h2o2, temp_max_servicio=1200°C (piedra, ver skill materiales)
MODELO: concentrado
BIOT_ESTIMADO: bajo
PROCESO: combustion
RIESGO: ninguno
NOTA: dentro de rango si el enfriamiento regenerativo funciona como está previsto
"""


def _parsear_respuesta_termica(texto: str) -> dict:
    resultado = {
        "modelo": "concentrado",
        "biot_estimado": "bajo",
        "proceso": "fisico",
        "riesgo_llm": "ninguno",
        "nota": "",
    }
    if not texto:
        return resultado

    m = re.search(r"^MODELO\s*:\s*(\w+)", texto, re.MULTILINE)
    if m and m.group(1).lower() in ("concentrado", "gradiente"):
        resultado["modelo"] = m.group(1).lower()

    m = re.search(r"^BIOT_ESTIMADO\s*:\s*(\w+)", texto, re.MULTILINE)
    if m and m.group(1).lower() in ("bajo", "alto"):
        resultado["biot_estimado"] = m.group(1).lower()

    m = re.search(r"^PROCESO\s*:\s*(\w+)", texto, re.MULTILINE)
    if m and m.group(1).lower() in ("fisico", "combustion"):
        resultado["proceso"] = m.group(1).lower()

    m = re.search(r"^RIESGO\s*:\s*(\w+)", texto, re.MULTILINE)
    if m and m.group(1).lower() in ("ninguno", "excede_temp_servicio", "excede_punto_fusion"):
        resultado["riesgo_llm"] = m.group(1).lower()

    m = re.search(r"^NOTA\s*:\s*(.+)$", texto, re.MULTILINE)
    if m:
        # Se guarda tal cual para mostrar al usuario, pero NUNCA se usa como fuente
        # numérica (ver validar_y_corregir_termico) — es solo texto cualitativo.
        resultado["nota"] = m.group(1).strip()

    return resultado


def _decidir_criterio(geometria: dict, material: dict, condiciones: dict) -> dict:
    resumen = (
        f'{{"geometria": {{"area_expuesta_m2": {geometria.get("area_expuesta_m2", 0):.5f}, '
        f'"volumen_estimado_m3": {geometria.get("volumen_estimado_m3", 0):.6f}, '
        f'"espesor_caracteristico_m": {geometria.get("espesor_caracteristico_m", 0):.5f}}}, '
        f'"material": {{"conductividad_termica_w_mk": {material.get("conductividad_termica_w_mk", 0)}, '
        f'"calor_especifico_j_kgk": {material.get("calor_especifico_j_kgk", 0)}, '
        f'"densidad_kg_m3": {material.get("densidad_kg_m3", 0)}, '
        f'"temperatura_max_servicio_c": {material.get("temperatura_max_servicio_c", "null")}}}, '
        f'"condiciones": {{"temp_ambiente_c": {condiciones.get("temp_ambiente_c", 20)}, '
        f'"temp_fuente_c": {condiciones.get("temp_fuente_c", 20)}, '
        f'"tipo_proceso": "{condiciones.get("tipo_proceso", "fisico")}", '
        f'"modo": "{condiciones.get("modo", "conduccion_transitoria")}"}}}}'
    )
    texto = _llamar_modelo(
        messages=[
            {"role": "system", "content": SYSTEM_TERMICO},
            {"role": "user", "content": f"entrada: {resumen}"},
        ],
        num_predict=-1,
        temperatura=TEMPERATURA_TERMICA,
    )
    return _parsear_respuesta_termica(texto)


# ---------------------------------------------------------------------------
# Filtro de ruido (capa 3 — skill 00): Python es SIEMPRE la fuente numérica
# ---------------------------------------------------------------------------

def validar_y_corregir_termico(decision_llm: dict, calculo: dict) -> tuple[dict, bool, list[str]]:
    """decision_llm: salida de _parsear_respuesta_termica().
    calculo: resultado numérico ya calculado en Python (ver analizar_termico).
    Devuelve (decision_final, es_valida, advertencias). Nunca es_valida=False
    de forma bloqueante: esta skill es la que menos carga tiene en el LLM de
    todo el pipeline, así que un desacuerdo se loguea pero no frena nada."""
    advertencias: list[str] = []
    resultado = dict(decision_llm)

    riesgo_python = calculo["riesgo"]
    if resultado["riesgo_llm"] != riesgo_python:
        advertencias.append(
            f"  RIESGO del modelo ('{resultado['riesgo_llm']}') no coincide con el cálculo "
            f"Python ('{riesgo_python}') — se usa el de Python, es la fuente de verdad numérica."
        )
    resultado["riesgo"] = riesgo_python   # Python gana siempre

    bi = calculo.get("biot")
    if bi is not None:
        biot_python = "bajo" if bi < 0.1 else "alto"
        if resultado["biot_estimado"] != biot_python:
            advertencias.append(
                f"  BIOT_ESTIMADO del modelo ('{resultado['biot_estimado']}') no coincide con "
                f"Bi={bi:.4f} calculado ('{biot_python}') — se usa el calculado."
            )
        resultado["biot_estimado"] = biot_python
        resultado["modelo"] = "concentrado" if biot_python == "bajo" else "gradiente"

    return resultado, True, advertencias


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

def analizar_termico(geometria: dict, material: dict, condiciones: dict) -> dict:
    """Punto de entrada de la skill. Todo dato numérico de salida es Python
    puro; el LLM solo aportó el criterio (modelo/biot/proceso) y la nota.

    geometria   : {"area_expuesta_m2", "volumen_estimado_m3", "espesor_caracteristico_m"}
                  — normalmente sale de geometria_termica_desde_ficha().
    material    : ficha de objetos.py (+ campos extendidos de la skill 03 si existen).
    condiciones : {"temp_ambiente_c", "temp_fuente_c", "tipo_proceso", "modo",
                   "coef_conveccion_w_m2k" (opcional), "masa_h2_kg" (opcional,
                   solo si tipo_proceso es de combustión), "tiempo_s" (opcional)}

    Devuelve:
        {
          "modelo": "concentrado"|"gradiente",
          "biot": float,
          "proceso": "fisico"|"combustion",
          "temperatura_final_c": float,
          "delta_temperatura_c": float,
          "tiempo_estabilizacion_s": float | None,
          "energia_liberada_j": float | None,      # solo si hubo combustión
          "riesgo": "ninguno"|"excede_temp_servicio"|"excede_punto_fusion",
          "nota": str,
          "advertencias": [str, ...],
        }
    """
    advertencias: list[str] = []
    material = _sanear_material(material)

    temp_amb, aviso = _sanear_temperatura_c(condiciones.get("temp_ambiente_c"), 20.0)
    if aviso:
        advertencias.append(aviso)
    temp_fuente, aviso = _sanear_temperatura_c(condiciones.get("temp_fuente_c"), temp_amb)
    if aviso:
        advertencias.append(aviso)

    h = condiciones.get("coef_conveccion_w_m2k") or H_CONVECCION_AIRE_LIBRE_W_M2K
    area = geometria.get("area_expuesta_m2") or 0.01
    espesor = geometria.get("espesor_caracteristico_m") or 0.01
    volumen = geometria.get("volumen_estimado_m3") or 1e-6
    masa_kg = volumen * material["densidad_kg_m3"]
    k = material["conductividad_termica_w_mk"]
    c = material["calor_especifico_j_kgk"]

    biot = numero_biot(h, espesor, k)
    tau = constante_tiempo_concentrado(masa_kg, c, h, area)

    tipo_proceso = (condiciones.get("tipo_proceso") or "fisico").lower()
    es_combustion = "combusti" in tipo_proceso or "h2" in tipo_proceso

    energia_liberada_j = None
    if es_combustion:
        masa_h2_kg = condiciones.get("masa_h2_kg") or 0.0
        if masa_h2_kg > 0:
            energia_liberada_j = energia_combustion_h2_j(masa_h2_kg)
            delta_t = delta_temperatura_combustion(energia_liberada_j, masa_kg, c)
            temperatura_final_c = temp_amb + delta_t
            tiempo_estabilizacion_s = None
        else:
            advertencias.append(
                "  tipo_proceso de combustión sin 'masa_h2_kg' en condiciones — "
                "no se puede calcular energía liberada; se usa temp_fuente_c como resultado."
            )
            temperatura_final_c = temp_fuente
            delta_t = temp_fuente - temp_amb
            tiempo_estabilizacion_s = None
    else:
        tiempo_s = condiciones.get("tiempo_s")
        if tiempo_s is not None:
            temperatura_final_c = temperatura_transitoria_concentrado(
                temp_amb, temp_fuente, tiempo_s, tau
            )
        else:
            # Sin tiempo explícito: reportar la temperatura de régimen (t->inf
            # con esta convección simple tiende al ambiente; lo relevante acá
            # es tau, que se devuelve para que quien llame decida cuánto esperar).
            temperatura_final_c = temp_fuente
        delta_t = temperatura_final_c - temp_amb
        tiempo_estabilizacion_s = tau if tau != float("inf") else None

    # RIESGO: SIEMPRE calculado en Python contra los datos reales de material.
    temp_max_servicio = material.get("temperatura_max_servicio_c")
    punto_fusion = material.get("punto_fusion_c")
    riesgo = "ninguno"
    if punto_fusion and temperatura_final_c >= punto_fusion:
        riesgo = "excede_punto_fusion"
    elif temp_max_servicio and temperatura_final_c >= temp_max_servicio:
        riesgo = "excede_temp_servicio"

    calculo = {"riesgo": riesgo, "biot": biot}

    # ── LLM: solo criterio + nota cualitativa ──────────────────────────────
    geometria_para_llm = {**geometria, "area_expuesta_m2": area,
                           "volumen_estimado_m3": volumen, "espesor_caracteristico_m": espesor}
    condiciones_para_llm = {**condiciones, "temp_ambiente_c": temp_amb, "temp_fuente_c": temp_fuente}
    decision = _decidir_criterio(geometria_para_llm, material, condiciones_para_llm)
    decision, _, adv_filtro = validar_y_corregir_termico(decision, calculo)
    advertencias.extend(adv_filtro)

    return {
        "modelo": decision["modelo"],
        "biot": biot,
        "proceso": decision["proceso"],
        "temperatura_final_c": temperatura_final_c,
        "delta_temperatura_c": delta_t,
        "tiempo_estabilizacion_s": tiempo_estabilizacion_s,
        "energia_liberada_j": energia_liberada_j,
        "riesgo": riesgo,
        "nota": decision["nota"],
        "advertencias": advertencias,
    }


if __name__ == "__main__":
    print("=== Caso físico: bloque de acero chico cerca de una fuente a 850°C ===")
    geo = geometria_termica_desde_ficha(peso_kg=2.0, densidad_kg_m3=7850.0)
    mat = {
        "material": "acero",
        "conductividad_termica_w_mk": 45.0,
        "calor_especifico_j_kgk": 490.0,
        "densidad_kg_m3": 7850.0,
        "temperatura_max_servicio_c": 400.0,
        "punto_fusion_c": 1450.0,
    }
    cond = {"temp_ambiente_c": 20, "temp_fuente_c": 850, "tipo_proceso": "fisico",
            "modo": "conduccion_transitoria", "tiempo_s": 120}
    try:
        r = analizar_termico(geo, mat, cond)
        print("modelo:", r["modelo"], "| Bi:", round(r["biot"], 4),
              "| T_final:", round(r["temperatura_final_c"], 1), "°C",
              "| riesgo:", r["riesgo"])
        for a in r["advertencias"]:
            print(a)
    except Exception as e:
        print(f"(omitido, requiere Ollama corriendo: {e})")

    print("\n=== Caso combustión H2/O2 en cámara de piedra (sin LLM, solo cálculo) ===")
    geo2 = geometria_termica_desde_ficha(peso_kg=0.5, densidad_kg_m3=2600.0)
    q = energia_combustion_h2_j(masa_h2_kg=0.01)
    dt = delta_temperatura_combustion(q, masa_objeto_kg=0.5, calor_especifico_j_kgk=900.0)
    print(f"Energía liberada: {q/1e6:.2f} MJ | ΔT estimado: {dt:.1f} °C")
    print(f"Masa de O2 estequiométrica para 0.01 kg de H2: {masa_o2_estequiometrica_kg(0.01):.4f} kg")