# SKILL: Ciencia de materiales

## Objetivo
Ya tenés `SYSTEM_PROPIEDADES_FISICAS` en `objetos.py`, que genera la ficha
básica (material, densidad, resistencias, dureza, resistividad). Esta skill la
EXTIENDE con los campos que van a hacer falta para que `termodinamica` y
`calculo_estructural` puedan trabajar sin volver a preguntarle nada al modelo:
coeficiente de dilatación térmica, calor específico, módulo de Poisson y
límite de fatiga. También agrega una capa de chequeo de coherencia
material↔propiedades que hoy `_normalizar_propiedades` no hace (solo rellena
huecos con ceros, no valida que los números tengan sentido entre sí).

No reemplaza tu prompt actual — se agrega como paso 1.5 opcional, solo cuando
`termodinamica` o `calculo_estructural` van a necesitar esos campos extra.

## Cuándo se activa
Después de `generar_propiedades()` (tu función actual), solo si el objeto va a
pasar por cálculo estructural o térmico. Si el objeto es solo decorativo
(render_only), no hace falta — ahorra una llamada al modelo.

## Entrada
La ficha ya generada por tu `SYSTEM_PROPIEDADES_FISICAS` actual, como contexto
(no se le vuelve a preguntar lo que ya se sabe — regla 4 del filtro base):
```json
{
  "material": "Hormigón estructural (concreto armado)",
  "densidad_kg_m3": 2450.0,
  "modulo_elasticidad_gpa": 27.5,
  "resistencia_compresion_mpa": 30.0,
  "dureza": "6 Mohs"
}
```

## Proceso de razonamiento (system prompt nuevo, extiende el actual)
Agregar al prompt existente estas líneas de formato (mismo estilo, mismas
reglas de "no inventar números al azar" que ya tenés):

```
"coef_dilatacion_termica_1_k": number,   // 1/K, ej. acero ≈ 1.2e-5, aluminio ≈ 2.3e-5
"calor_especifico_j_kgk": number,        // J/(kg·K), ej. agua = 4186, acero ≈ 490
"modulo_poisson": number,                // adimensional, 0.0-0.5, ej. acero ≈ 0.30, hormigón ≈ 0.20
"limite_fatiga_mpa": number,             // esfuerzo alternante que soporta indefinidamente, típicamente 0.4-0.5x resistencia_traccion
"temperatura_max_servicio_c": number     // temperatura antes de perder propiedades mecánicas significativamente
```

Chequeo de coherencia que el prompt debe pedir explícitamente (esto es lo que
falta hoy en `SYSTEM_ACTUALIZAR_PROPIEDADES`): si `material` cambia, TODOS los
campos derivados tienen que recalcularse juntos, no solo el nombre — ya lo
tenés escrito para densidad/resistencias/módulo/punto de fusión/resistividad,
agregar a esa misma frase los 5 campos nuevos.

## Formato de salida
Igual que tu `SYSTEM_ACTUALIZAR_PROPIEDADES` actual: JSON con las claves de
arriba, sin texto adicional, se parsea con tu `_extraer_json` existente (no
hace falta un parser nuevo).

## Tabla de rangos de referencia (anti-alucinación, va dentro del prompt)
| Material típico | Poisson | Dilatación (1/K) | Calor esp. (J/kg·K) | Fatiga (fracción de tracción) |
|---|---|---|---|---|
| Acero | 0.27–0.30 | 1.1e-5 – 1.3e-5 | 450–500 | 0.45–0.50 |
| Aluminio | 0.32–0.35 | 2.2e-5 – 2.4e-5 | 890–900 | 0.30–0.40 |
| Hormigón | 0.15–0.22 | 0.9e-5 – 1.2e-5 | 880–1000 | no aplica (frágil, usar resistencia a fatiga a compresión ≈ 0.55x) |
| Plástico (PLA/ABS genérico) | 0.35–0.40 | 6e-5 – 9e-5 | 1200–1900 | 0.20–0.30 |
| Madera | 0.30–0.45 (anisótropo, usar valor medio) | 0.3e-5 – 0.6e-5 (fibra) | 1200–2700 | no aplica de forma simple |

## Filtro de ruido específico
```python
def validar_y_corregir_material(propiedades: dict) -> tuple[dict, bool, list[str]]:
```
- **Poisson fuera de [0, 0.5]**: violación grave (viola termodinámica del
  sólido elástico, no es negociable) → clamp a 0.3 (valor típico genérico) y
  aviso.
- **Límite de fatiga > resistencia a la tracción**: imposible físicamente →
  clamp a `0.45 * resistencia_traccion_mpa`.
- **Calor específico ≤ 0**: violación grave → valor de seguridad por
  categoría (ver tabla de plantillas abajo).
- **Coherencia cruzada con la ficha ya existente**: si `densidad_kg_m3` de la
  ficha original es muy baja (<100) pero `material` no menciona espuma/gas,
  esto ya es sospechoso desde el paso actual — vale la pena agregar este
  chequeo también a `_normalizar_propiedades` existente, no solo acá.

### Plantillas de seguridad por categoría (`_VALORES_SEGURIDAD_MATERIAL`)
```python
{
  "metal_generico":    {"poisson": 0.30, "calor_especifico_j_kgk": 460, "dilatacion_1_k": 1.2e-5},
  "plastico_generico":  {"poisson": 0.38, "calor_especifico_j_kgk": 1500, "dilatacion_1_k": 7e-5},
  "mineral_generico":   {"poisson": 0.20, "calor_especifico_j_kgk": 900, "dilatacion_1_k": 1.0e-5},
  "organico_generico":  {"poisson": 0.35, "calor_especifico_j_kgk": 1800, "dilatacion_1_k": 0.5e-5},
}
```
Selección de plantilla: heurística simple por palabras clave en `material`
(igual criterio que ya usás en objetos.py para decidir default), nunca se deja
el campo en 0 silencioso como hace hoy `_normalizar_propiedades` (un 0 en
calor específico rompe cualquier cálculo térmico después, division por cero
incluida).

## Notas de integración
- Función sugerida: `objetos.py :: generar_propiedades_extendidas(nombre, propiedades_base)`,
  hermana de tu `generar_propiedades()` actual, no reemplazo.
- Se guarda como sub-bloque `"propiedades_extendidas"` dentro del mismo
  registro de `objetos_db.json`, para no romper el schema que ya tenés
  (`propiedades` sigue siendo la ficha básica tal cual está hoy).
- Cachear igual que las demás fichas: si el material no cambió, no
  recalcular esto de nuevo.
