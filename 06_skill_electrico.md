# SKILL: Circuito eléctrico

## Objetivo
Estimar corriente, caída de tensión y potencia disipada en una RED de
componentes ya definidos (no un objeto aislado) — pensada para el modo
interactivo "armá un circuito con la mano": tocar dos objetos conductores los
conecta eléctricamente, una pila virtual les impone una tensión, y hay que
saber cuánta corriente circula por cada rama en tiempo real. Se apoya en un
campo que ya existe desde el día 1 en la ficha de `objetos.py`:
`resistencia_electrica_ohm_m`.

Conexión directa con `termodinamica.py`: la potencia disipada por efecto
Joule (`P = I²R`) es una fuente de calor más — un cable con sobrecorriente es
el mismo problema físico que una pieza cerca de una llama, solo cambia de
dónde sale la potencia. El modo de falla térmico de un conductor
(`corriente_maxima_por_calentamiento_a`) reusa el mismo modelo de convección
simple que ya tiene termodinámica, no inventa uno nuevo.

## Cuándo se activa
Bajo demanda, cuando el usuario arma o pregunta sobre un circuito ("¿cuánta
corriente pasa por esta resistencia si conecto la pila acá?"), o como el modo
interactivo "eléctrico" de `modos.py`, actualizándose cada tick mientras la
mano acerca/aleja/toca componentes en la escena.

## Entrada
A diferencia de `04_termodinamica` y `05_calculo_estructural`, acá NO hay una
sola ficha de un objeto — hay una lista de componentes y cómo están
conectados entre sí. La topología sale de geometría pura (contacto de bboxes,
`ubicacion.py`), nunca se le pregunta al LLM ni se infiere de la mano qué
está tocando qué:

```json
{
  "elementos_resistivos": [
    {"nombre": "R1", "a": "V+", "b": "N1", "resistencia_ohm": 100.0},
    {"nombre": "R2", "a": "N1", "b": "GND", "resistencia_ohm": 220.0}
  ],
  "fuentes": [
    {"nombre": "pila", "nodo_pos": "V+", "nodo_neg": "GND", "tension_v": 9.0}
  ]
}
```
La resistencia de cada elemento sale de `resistencia_electrica_ohm_m` (ficha)
+ geometría (masa/densidad → volumen → longitud/área, mismo criterio que
`geometria_termica_desde_ficha`), nunca se le pide al LLM un valor de
resistencia.

## Proceso — la resolución de la red es 100% Python/numpy, nunca LLM
A diferencia de estructural/térmico, acá no hay "criterio" que el modelo
tenga que elegir para el cálculo en sí: resolver una red de resistores +
fuentes de tensión es álgebra lineal exacta (análisis nodal modificado, MNA).
No existe ambigüedad de "qué fórmula aplica" — la misma resolución sirve para
serie, paralelo, o cualquier red general, incluyendo "otros componentes
conectados" en cadena. El LLM se usa solo para:

1. **Clasificar la topología en palabras simples** (serie/paralelo/red
   general/abierto) — cosmético, para el mensaje al usuario. La resolución
   real siempre corre igual sea cual sea la clasificación.
2. **Redactar una advertencia corta** si `resolver_red()` + el chequeo de
   corriente máxima por calentamiento detectan sobrecorriente/cortocircuito/
   circuito abierto.

## Formato de salida (del LLM, solo clasificación + nota)
```
TOPOLOGIA: <serie|paralelo|red_general|abierto>
MODO_FALLA: <ninguno|sobrecorriente|sobretension|sobrecalentamiento|cortocircuito>
NOTA: una frase corta, opcional
```
Todo lo numérico (voltajes de nodo, corriente por elemento, potencia
disipada) lo calcula `resolver_red()` con MNA — la única fuente de verdad.

## Topología por contacto (Python puro, determinístico)
Cada objeto-conductor es un resistor con dos terminales (`<nombre>#A`,
`<nombre>#B`). Cuando dos objetos se tocan en la escena (mismo tipo de test
AABB con margen que ya usa `ubicacion.py` para colisión), sus terminales más
cercanos se fusionan en el mismo nodo eléctrico (Union-Find). Objetos cuya
`resistencia_electrica_ohm_m` esté por encima de
`UMBRAL_RESISTIVIDAD_CONDUCTOR_OHM_M` actúan como aislante — no cortan la
topología, pero en la práctica no dejan pasar corriente apreciable.

La polaridad de una fuente (pila) NO se puede inferir de la geometría —
se pasa explícita (`nodo_pos`/`nodo_neg`), típicamente los terminales del
objeto-fuente.

## Rangos / criterios de aceptación
| Magnitud | Criterio | Acción |
|---|---|---|
| Corriente por elemento | > corriente_maxima_por_calentamiento_a del elemento | `MODO_FALLA=sobrecalentamiento` (régimen permanente, mismo modelo de convección que termodinámica) |
| Resistencia de rama | < 1 Ω entre los dos polos de una fuente | `MODO_FALLA=cortocircuito` |
| Matriz de red singular (nodos aislados, ningún camino cerrado) | — | `circuito_abierto=True`, todas las corrientes en 0 (nunca None, nunca romper el pipeline) |

## Filtro de ruido específico
```python
def validar_y_corregir_electrico(decision_llm: dict, riesgo_python: str) -> tuple[dict, bool, list[str]]:
```
- El `MODO_FALLA` del LLM se **ignora y se reemplaza** por el riesgo
  calculado en Python (corriente real vs. corriente máxima térmica real) —
  igual criterio que termodinámica/estructural: el LLM nunca es fuente
  numérica ni de diagnóstico final, solo de redacción.
- Cualquier número que aparezca en `NOTA` se ignora igual que en las otras
  skills numéricas.

## Notas de integración
- Función sugerida: `electrico.py :: evaluar_circuito(elementos_resistivos, fuentes, corrientes_maximas_a)`
  para consulta puntual (con LLM, una vez). `electrico.py :: resolver_red(...)`
  es la versión sin LLM, pensada para llamarse en cada tick de un modo
  interactivo (`modos.py :: ModoElectrico`) — nunca toca el modelo, es
  numpy puro.
- `electrico.py :: construir_red_desde_contacto(objetos_en_escena, fichas)`
  arma `elementos_resistivos` a partir de qué objetos se están tocando en la
  escena — se re-llama cada tick si la mano puede mover componentes en vivo.
- Se guarda como sub-bloque `"electrico"` en `objetos_db.json` por instancia
  de circuito (no por objeto individual — un mismo resistor puede formar
  parte de circuitos distintos en escenas distintas), mismo criterio que las
  demás skills.
