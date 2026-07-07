# SKILL: Cálculo estructural

## Objetivo
Dado un objeto con geometría auditada, material extendido y un caso de carga,
estimar si el objeto aguanta: tensión resultante, factor de seguridad,
deflexión aproximada. Pensada para conectar con `MechanismRegistry` de
`cad2d.py` (engranajes, poleas, tornillos) — ahí ya tenés la geometría
mecánica, esta skill le agrega el chequeo de si esa pieza resiste la carga que
se le pide transmitir.

## Cuándo se activa
Cuando el usuario define o edita un mecanismo con carga esperada ("este
engranaje va a transmitir 5 Nm"), o pregunta directamente ("¿aguanta el
bloque de cemento si le pongo 200 kg encima?").

## Entrada
Todo dato geométrico y de material ya validado por las skills anteriores —
esta skill nunca genera geometría ni propiedades de material, solo las
consume:

```json
{
  "geometria": {
    "area_seccion_m2": 0.0234,
    "espesor_caracteristico_m": 0.03,
    "longitud_m": 0.6,
    "tipo_apoyo": "simple",
    "forma_seccion": "rectangular"
  },
  "material": {
    "resistencia_traccion_mpa": 3.5,
    "resistencia_compresion_mpa": 30.0,
    "modulo_elasticidad_gpa": 27.5,
    "limite_fatiga_mpa": 15.0
  },
  "carga": {
    "tipo": "puntual|distribuida|torsion|axial",
    "magnitud_n": 1962.0,
    "posicion_relativa": 0.5
  }
}
```

## Proceso de razonamiento
Mismo principio que termodinámica: las fórmulas de resistencia de materiales
son deterministas (viga simple, esfuerzo axial, torsión de eje circular) — se
calculan en Python, no se le piden al modelo. El LLM se usa solo para:

1. **Clasificar el caso de carga** a la fórmula correcta según
   `tipo`+`tipo_apoyo`+`forma_seccion` (viga simplemente apoyada con carga
   puntual al centro, columna en compresión, eje en torsión, etc.) — esto es
   selección de fórmula, no cálculo.
2. **Detectar modo de falla dominante**: para materiales frágiles (hormigón,
   cerámicos, vidrio) la compresión gobierna y la tracción es el punto débil;
   para materiales dúctiles (acero, aluminio) suele gobernar fluencia o
   fatiga si la carga es cíclica. El LLM decide qué resistencia usar como
   referencia (`resistencia_traccion_mpa` vs `resistencia_compresion_mpa` vs
   `limite_fatiga_mpa`), Python hace la división.
3. **Redactar la advertencia** si el factor de seguridad calculado por Python
   da bajo.

## Formato de salida (del LLM)
```
FORMULA: <viga_simple|viga_empotrada|columna_compresion|eje_torsion|axial_simple>
RESISTENCIA_REFERENCIA: <traccion|compresion|fatiga|fluencia>
MODO_FALLA: <fractura_fragil|fluencia_ductil|fatiga|pandeo>
NOTA: una frase corta, opcional
```

Cálculo en Python según `FORMULA` (fórmulas de referencia, no se le piden al
modelo):
- Viga simple, carga puntual centrada: `σ = M·c / I`, con `M = F·L/4`.
- Columna en compresión: `σ = F / A`, chequear además pandeo de Euler si
  `longitud/espesor > 10`: `F_crit = π²·E·I / L²`.
- Eje en torsión: `τ = T·r / J`.
- Axial simple: `σ = F / A`.
- Factor de seguridad: `FS = resistencia_referencia / σ_calculada`.
- Deflexión (viga simple, carga puntual centrada): `δ = F·L³ / (48·E·I)`.

## Rangos / criterios de aceptación
| Magnitud | Criterio | Acción |
|---|---|---|
| Factor de seguridad | FS ≥ 1.0 siempre para reportar "aguanta"; FS < 1.5 → advertencia aunque "aguante" (margen bajo) | nunca redondear FS hacia arriba en el mensaje al usuario |
| Esbeltez (L/espesor) | > 10 → chequear pandeo de Euler, no solo compresión simple | agregar chequeo automático, no depende del LLM |
| Material frágil + tracción | si `RESISTENCIA_REFERENCIA` es "traccion" para un material con `resistencia_traccion_mpa` muy por debajo de `resistencia_compresion_mpa` (relación típica < 0.15, como el hormigón), advertir que la pieza es mucho más débil a tracción — punto de falla más probable |
| Carga cíclica | si el pedido del usuario menciona "repetido", "vibración", "cíclico" → usar `limite_fatiga_mpa` como referencia, nunca `resistencia_traccion_mpa` | esto ya lo tenés como campo en la skill de materiales extendida |

## Filtro de ruido específico
```python
def validar_y_corregir_estructural(decision_llm: dict, contexto: dict) -> tuple[dict, bool, list[str]]:
```
- Si `FORMULA` elegida por el LLM no matchea con `tipo_apoyo`/`tipo` de la
  entrada (ej. eligió "eje_torsion" para una carga puntual), se ignora la
  elección y se aplica la regla determinística de mapeo
  `tipo_carga → formula` que vive en Python (tabla fija, no ambigua) —
  la clasificación del LLM es una ayuda, no la única fuente de verdad cuando
  el caso es simple y mapea 1 a 1.
- Cualquier número de tensión, deflexión o FS que aparezca en `NOTA` se
  ignora — igual que en termodinámica, el LLM nunca es fuente numérica.
- Si `resistencia_referencia` es `null` o inconsistente, usar automáticamente
  la más conservadora (menor de las dos resistencias disponibles) — nunca
  fallar en silencio con resistencia 0.

## Notas de integración
- Función sugerida: `calculo_estructural.py :: evaluar_carga(geometria, material, carga)`.
- Se conecta natural con `MechanismRegistry` de `cad2d.py`: cuando se define
  un engranaje/eje/tornillo con un torque o fuerza de diseño, esta skill le
  agrega el campo `"chequeo_estructural"` al registro del mecanismo.
- El resultado (`FS`, `deflexion_m`, `modo_falla`) se guarda como sub-bloque
  `"estructural"` en `objetos_db.json`, mismo criterio que las demás skills.
