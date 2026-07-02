# SKILL: Ubicación espacial

## Objetivo
Decidir DÓNDE va un objeto (ya generado por el pipeline de geometría existente)
dentro de la escena 3D compartida, en relación a otros objetos ya colocados y a
la mano detectada por MediaPipe. Es la skill que falta entre "ya sé qué forma
tiene el objeto" (paso 1/2 actual) y "dónde lo pongo sin que choque con nada".

No reemplaza `SYSTEM_FIGURA_RAZONAMIENTO` (esa sigue decidiendo la forma
interna del objeto). Esta skill decide la pose global: posición del centro,
rotación sobre el eje Z (yaw), y si apoya sobre otro objeto o sobre el "piso"
virtual.

## Cuándo se activa
Después de paso 2/2b (figura ya validada) y antes de agregar el objeto a
`entorno_virtual.py`. También se re-activa cuando el usuario pide reubicar un
objeto ya existente ("poné el cubo sobre el bloque").

## Entrada (JSON saneado, capa 1)
```json
{
  "objeto_nuevo": {
    "nombre": "cubo de plastico",
    "bbox_local": {"x_min": 0.25, "x_max": 0.75, "y_min": 0.42, "y_max": 0.80, "z_min": 0.45, "z_max": 0.55}
  },
  "objetos_en_escena": [
    {"nombre": "Bloque de cemento", "centro": [0.5, 0.55, 0.5], "bbox": {...}, "apoyado_sobre": "piso"}
  ],
  "mano": {"presente": true, "centro": [0.6, 0.3, 0.5], "contacto_con": null},
  "pedido_usuario": "al lado del bloque de cemento, sin tocarlo"
}
```
Reglas de saneo: `objetos_en_escena` nunca vacío como `null` — mandar `[]`.
`pedido_usuario` puede ser `""` si no hubo instrucción explícita (se usa
colocación por defecto: piso, no solapado).

## Proceso de razonamiento (system prompt, checklist interno)
El modelo debe resolver, en orden, SOLO estos 4 puntos, sin explicar nada más:

1. **Plano de apoyo**: ¿el objeto va sobre el piso virtual (z fijo, "abajo" de
   la escena) o sobre otro objeto (su `y_max`/`z_max` es la base del nuevo)?
2. **Punto de referencia**: si hay `pedido_usuario`, identificar contra qué
   objeto es relativo ("al lado de X", "arriba de X", "entre X e Y"). Si no
   hay pedido, usar el hueco libre más cercano al centro de la escena.
3. **Offset sin colisión**: calcular el desplazamiento mínimo desde el objeto
   de referencia para que las bounding boxes NO se solapen (dejar margen fijo
   de separación, ver tabla de tolerancias).
4. **Rotación**: 0 salvo que el pedido mencione orientación ("de canto",
   "apoyado de lado").

## Formato de salida exacto
```
CENTRO: cx,cy,cz
APOYA: <nombre_objeto_referencia | piso>
ROT_Z: grados
MARGEN: distancia_usada
```
Sin texto adicional. Un dato por línea, igual criterio que el resto del
pipeline.

### Ejemplo
```
CENTRO: 0.68,0.55,0.50
APOYA: piso
ROT_Z: 0
MARGEN: 0.06
```

## Rangos / tolerancias válidas (anti-alucinación)
| Concepto | Valor por defecto | Rango aceptable |
|---|---|---|
| Margen mínimo entre bboxes | 0.04 (unidades de escena normalizada) | 0.02 – 0.15 |
| Rotación Z | 0° | -180° a 180° |
| Altura sobre "piso" | z del piso de la escena (constante del proyecto) | ± 0.02 tolerancia de asentado |

## Filtro de ruido específico (capa 3, Python)
```python
def validar_y_corregir_ubicacion(datos: dict, contexto: dict) -> tuple[dict, bool, list[str]]:
```
- **Colisión real**: recalcular en Python (no confiar en el LLM) si el bbox
  trasladado al `CENTRO` propuesto se solapa con algún objeto de
  `objetos_en_escena`. Esto es puramente geométrico (AABB overlap test), no
  hace falta el modelo para esto — es más rápido y confiable que reintentar.
  Si hay colisión: empujar el objeto en la dirección de menor solapamiento
  hasta despegarlo (algoritmo de resolución de colisión por eje mínimo, tipo
  el que ya usás para `BlockEntity` en `cad2d.py`).
- **Apoyo inconsistente**: si `APOYA` referencia un objeto que no existe en
  `objetos_en_escena`, violación grave → reintento con la lista de nombres
  válidos explicitada en el prompt de reintento.
- **Objeto fuera de cámara**: si `CENTRO` cae fuera de [0,1]³ con margen de
  cámara, clamp silencioso (violación leve).
- Igual que en bbox: nunca devolver `None`, plantilla de seguridad = "piso,
  centro de escena, sin rotación" si todo falla.

## Notas de integración
- Función sugerida: `ubicacion.py :: calcular_ubicacion(objeto_nuevo, objetos_en_escena, mano, pedido_usuario)`.
- La resolución de colisión (AABB) va en Python puro, no en el LLM — el LLM
  solo decide la intención ("al lado de", "arriba de"), Python decide el
  número exacto. Esto reduce carga al modelo chico y elimina una fuente
  entera de alucinación numérica.
- Se guarda en `objetos_db.json` como bloque nuevo `"espacial"` por instancia
  de objeto en escena (no por definición de objeto — un mismo "cubo de
  plástico" puede estar en dos posiciones distintas en la escena).
