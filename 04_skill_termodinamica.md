# SKILL: Termodinámica

## Objetivo
Estimar comportamiento térmico de un objeto ya definido (geometría +
material): conducción, convección simple, y — pensando en tu experimentación
con combustión H₂+O₂ para el sistema de propulsión rotativo — liberación de
energía por combustión y temperatura resultante. También sirve como fuente de
condiciones de borde térmicas para `simulador_aerodinamico.py` si en algún
momento agregás un campo de temperatura al LBM (hoy el solver es solo de
fluido/momentum).

## Cuándo se activa
Bajo demanda: cuando el usuario pregunta algo térmico de un objeto ya
existente ("¿cuánto tarda en enfriarse el bloque de cemento?", "¿qué
temperatura alcanza la cámara de combustión?"), o como parte de un chequeo de
`temperatura_max_servicio_c` antes de una simulación de combustión.

## Entrada
Todo ya resuelto por skills anteriores — geometría auditada + ficha de
materiales extendida + condiciones de borde del usuario. Nunca se le pide al
modelo un dato que ya calculaste (regla 4 del filtro base).

```json
{
  "geometria": {"area": 0.0234, "volumen_estimado_m3": 0.0018, "espesor_caracteristico_m": 0.03},
  "material": {
    "conductividad_termica_w_mk": 1.3,
    "calor_especifico_j_kgk": 900,
    "densidad_kg_m3": 2450,
    "temperatura_max_servicio_c": 400
  },
  "condiciones": {
    "temp_ambiente_c": 20,
    "temp_fuente_c": 850,
    "tipo_proceso": "combustion_h2o2",
    "modo": "conduccion_transitoria"
  }
}
```
`geometria.volumen_estimado_m3` y `espesor_caracteristico_m` se calculan en
Python a partir de la figura auditada (skill de geometría) — nunca se le pide
al LLM que estime volumen de una figura, es determinístico.

## Proceso de razonamiento
Igual que las otras skills numéricas: el LLM NO hace la cuenta de punta a
punta (para eso ya tenés Python con fórmulas exactas — conducción de Fourier,
`Q = m·c·ΔT`, etc.). El LLM se usa para las partes que requieren criterio, no
aritmética:

1. **Elegir el modelo térmico apropiado** dado `tipo_proceso` y `modo`: ¿es
   razonable tratarlo como cuerpo concentrado (lumped capacitance, válido si
   Biot < 0.1) o hace falta gradiente espacial? Esto es un criterio, no un
   número — el LLM devuelve la decisión, Python hace el cálculo con la
   fórmula correspondiente.
2. **Identificar qué fórmula de combustión aplica** si `tipo_proceso` es de
   combustión: poder calorífico del H₂ (~120 MJ/kg, dato de referencia fijo,
   no lo inventa el modelo) y relación estequiométrica 2H₂+O₂→2H₂O. El LLM
   solo confirma la relación molar/másica a usar; el cálculo de energía
   liberada es Python puro.
3. **Señalar riesgos** si la temperatura estimada por Python supera
   `temperatura_max_servicio_c` del material — esto es la única parte donde
   vale la pena que el LLM redacte una advertencia en lenguaje natural corta.

## Formato de salida (del LLM, solo decisión + advertencia)
```
MODELO: <concentrado|gradiente>
BIOT_ESTIMADO: <bajo|alto>
PROCESO: <fisico|combustion>
RIESGO: <ninguno|excede_temp_servicio|excede_punto_fusion>
NOTA: una frase corta, opcional
```
Todo lo numérico (temperatura final, tiempo de estabilización, energía
liberada) se calcula en Python con las fórmulas estándar, usando la decisión
de arriba solo para elegir qué fórmula aplicar:
- Concentrado: `T(t) = T_amb + (T0 - T_amb)·exp(-t / (m·c / (h·A)))`
- Gradiente: placeholder para si en algún momento agregás campo térmico al LBM
  (no hace falta resolverlo ahora, dejar la interfaz lista).
- Combustión: `Q = poder_calorifico_kg × masa_combustible_kg`, luego
  `ΔT = Q / (m_objeto × calor_especifico)`.

## Rangos de referencia (anti-alucinación)
| Magnitud | Rango físico válido | Nota |
|---|---|---|
| Temperatura | > -273.15 °C siempre | rechazar cualquier valor por debajo del cero absoluto sin excepción |
| Poder calorífico H₂ | 120 MJ/kg (constante, no se le pregunta al modelo) | usar valor fijo en Python |
| Temperatura de llama H₂/O₂ (estequiométrica) | ~2800–3000 °C de referencia | si el modelo devuelve algo muy distinto en su NOTA, ignorar esa cifra y usar la de Python |
| Biot | adimensional, > 0 siempre | Bi < 0.1 → concentrado; si no, marcar gradiente |

## Filtro de ruido específico
```python
def validar_y_corregir_termico(decision_llm: dict, contexto: dict) -> tuple[dict, bool, list[str]]:
```
- Cualquier número que el LLM intente meter en `NOTA` (temperaturas, energías)
  se **ignora y se recalcula en Python** — el campo `NOTA` es solo texto para
  mostrarle al usuario, nunca fuente de verdad numérica. Esto es la regla más
  importante de esta skill: separa completamente "redacción" de "cálculo".
- Si `RIESGO` dice `excede_punto_fusion` pero el cálculo de Python no lo
  confirma (o viceversa), gana Python siempre; se loguea la discrepancia como
  aviso pero no se reintenta (no hace falta, el número correcto ya lo tenés).

## Notas de integración
- Función sugerida: `termodinamica.py :: analizar_termico(geometria, material, condiciones)`.
- Es la skill donde MENOS carga tiene el LLM de todas las nuevas — casi todo
  es Python con fórmulas de libro (conducción de Fourier, calorimetría). Tiene
  sentido para hardware sin GPU: menos llamadas al modelo, más determinismo.
- Relevante directo para tu experimentación de propulsión H₂/O₂: esta skill
  te da el cálculo de ΔT de combustión sin depender de que el modelo chico
  "sepa química" (que no sabe de forma confiable) — los valores de referencia
  (poder calorífico, estequiometría) quedan fijos en la tabla de arriba.
