# SKILL: Geometría y topología

## Objetivo
Tu pipeline actual (`SYSTEM_FIGURA_RAZONAMIENTO` + `_parsear_formato_compacto`)
ya genera puntos, líneas y primitivas. Lo que falta es una capa que verifique
que esa geometría sea **topológicamente usable** antes de exportarla a
`cad2d.py` (BlockEntity) o al simulador aerodinámico (LBM). Esta skill no
genera figuras nuevas: audita y corrige las que ya salieron del paso 2.

Motivo: ya tuviste bugs de geometría (polígonos desordenados, contornos que no
cierran, doble escalado) que causaron los 7 bugs que arreglaste en el
pipeline CAD→CFD. Esta skill formaliza esa auditoría para que no dependa de
que vos la encuentres a mano cada vez.

## Cuándo se activa
Después de paso 2b (bbox ya validado), antes de:
- exportar el objeto a `cad2d.py` como `BlockEntity`,
- exportar el contorno a `simulador_aerodinamico.py` como geometría CFD,
- calcular propiedades derivadas para `calculo_estructural` (área, perímetro,
  centroide) — esas SIEMPRE se calculan en Python, nunca se le piden al LLM.

## Entrada
La figura ya validada (`puntos`, `conexiones`, `primitivas`), sin pasar por el
LLM en la mayor parte de esta skill — es sobre todo determinística. El LLM
solo se usa para el punto 4 (clasificación semántica), que es lo único que no
se puede resolver con matemática pura.

```json
{
  "puntos": [[0.48,0.72],[0.52,0.68],[0.52,0.32],[0.48,0.36]],
  "conexiones": [[0,1],[1,2],[2,3],[3,0]],
  "primitivas": [],
  "uso_destino": "cfd" 
}
```
`uso_destino` ∈ {"cfd", "mecanismo", "render_only"} — cambia qué tan estricta
es la validación (CFD no tolera huecos ni auto-intersecciones; render_only sí).

## Proceso — la mayoría es Python puro, no LLM

1. **Contorno cerrado** (Python, determinístico): cada punto referenciado en
   `conexiones` debe tener grado ≥ 2 si el contorno pretende ser cerrado.
   Detectar puntos "colgantes" (grado 1) → esto ya te pasó como "polígono
   desordenado".
2. **Orden y orientación** (Python, determinístico): recalcular el signo del
   área con shoelace formula. Si es negativo y el destino es CFD, invertir el
   orden de los puntos (esto es literalmente el bug de "eje Y invertido" que
   ya arreglaste — conviene que quede como chequeo automático permanente, no
   como fix puntual).
3. **Auto-intersección** (Python, determinístico): test de segmentos cruzados
   O(n²) sobre las aristas del contorno (n es chico, siempre <30 puntos en tus
   figuras, así que el costo no importa). Si hay cruce y destino es "cfd" o
   "mecanismo" → violación grave.
4. **Clasificación semántica** (única parte que sí usa el LLM): dado el
   contorno ya validado, pedirle al modelo que etiquete qué tipo de elemento
   mecánico es, SOLO si `uso_destino == "mecanismo"` — para saber si hay que
   registrarlo en `MechanismRegistry` como engranaje/polea/tornillo/genérico.
   Formato de salida:
   ```
   TIPO: <engranaje|polea|tornillo|husillo|generico>
   EJE: cx,cy
   ```

## Formato de salida (reporte de auditoría, Python)
```python
{
  "cerrado": bool,
  "orientacion": "ccw" | "cw",
  "auto_interseccion": bool,
  "area": float,          # shoelace, ya en unidades de escena normalizada
  "perimetro": float,
  "centroide": [cx, cy],
  "apto_para_destino": bool,
  "correcciones_aplicadas": [str, ...],
}
```

## Reglas de tolerancia
| Chequeo | Tolerancia | Acción si falla |
|---|---|---|
| Punto colgante (grado 1) | 0 tolerancia si `uso_destino=cfd` | cerrar con el punto más cercano, o marcar inválido si la distancia > 0.05 |
| Auto-intersección | 0 tolerancia si `cfd`/`mecanismo` | violación grave → devolver figura al paso 1 con aviso de qué segmentos cruzan |
| Orientación | siempre corregible | invertir orden de puntos, sin reintento (es determinístico, no hace falta el LLM) |
| Área mínima | > 0.0005 (evita "geometría degenerada", ej. figura colapsada en una línea) | violación grave |

## Filtro de ruido específico
```python
def validar_y_corregir_geometria(figura: dict, uso_destino: str) -> tuple[dict, bool, list[str]]:
```
- Todo lo determinístico (1, 2, 3 de arriba) se corrige o rechaza SIN llamar al
  modelo — es más rápido y 100% confiable, cosa que el LLM chico no puede
  garantizar en geometría exacta.
- Solo si `apto_para_destino=False` por auto-intersección se dispara reintento
  del paso 1 (`SYSTEM_FIGURA_RAZONAMIENTO`), no de esta skill — esta skill no
  regenera geometría, solo la valida y, cuando puede, la reordena.

## Notas de integración
- Función sugerida: `geometria.py :: auditar_geometria(figura, uso_destino)`.
- Se llama automáticamente antes de cualquier `exportar_a_cad()` o
  `exportar_a_cfd()` que ya tengas en `cad2d.py` / `simulador_aerodinamico.py`
  — conviene que sea un paso obligatorio de esas funciones, no opcional.
- El resultado (`area`, `perimetro`, `centroide`) se pasa como contexto de
  entrada saneado a `calculo_estructural` (regla 4 de la capa 1 del filtro
  base) — esa skill nunca vuelve a calcular geometría, la recibe ya resuelta.
