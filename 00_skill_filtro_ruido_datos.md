# SKILL: Filtro de ruido y datos (capa base común)

Esta skill NO es un prompt para el modelo. Es el **contrato de arquitectura** que
todas las skills nuevas (`ubicacion_espacial`, `geometria`, `ciencia_materiales`,
`termodinamica`, `calculo_estructural`) tienen que respetar para enchufarse al
pipeline existente de `ia_interprete.py` / `objetos.py` sin romper el patrón
paso 0 → 0b → 1 → 2 → 2b que ya usás.

Objetivo: el modelo chico (Qwen3.5-4B local) alucina valores, mezcla formatos y
a veces corta la respuesta. Esta capa evita que ese ruido llegue a la simulación
física (LBM, CAD, mecanismos) sin filtrar.

---

## Las 3 capas de toda skill

```
ENTRADA SANEADA  →  RAZONAMIENTO GUIADO (system prompt)  →  SALIDA VALIDADA
   (Python)              (LLM, formato fijo)                  (Python, clamp/retry)
```

### Capa 1 — Entrada saneada (antes de llamar al modelo)

Reglas que aplican a **toda** skill nueva:

1. Nunca mandar al modelo un dict de Python crudo (`dict.__repr__`). Siempre
   `json.dumps(datos, ensure_ascii=False)` con claves fijas y tipos primitivos
   (float, str, bool, list). El modelo local es más confiable con JSON de
   entrada que con texto libre.
2. Si un campo numérico puede faltar, mandar `null` explícito, nunca omitir la
   clave. El modelo infiere mejor "esto no lo sé" que "esto no existe".
3. Inyectar SIEMPRE unidades en las claves (`peso_kg`, no `peso`). Ya lo hacés
   así en `objetos.py` — mantenerlo en las skills nuevas.
4. Si la skill depende de una skill anterior (ej. cálculo estructural depende
   de geometría + materiales), mandar el resultado YA VALIDADO de esa skill
   anterior, nunca el crudo del LLM. Esto es lo que ya hacés al pasar `dims`
   del paso 0b al paso 1: nunca se re-pregunta al modelo algo que Python ya
   puede calcular determinísticamente.
5. Nunca pedir en la misma llamada dos cosas heterogéneas (geometría + F=ma).
   Un LLM chico razona peor cuando mezcla dominios. Separar en llamadas
   secuenciales, como ya hacés con geometría → propiedades.

### Capa 2 — Razonamiento guiado (system prompt de cada skill)

Todo prompt nuevo tiene que tener estas 4 secciones, en este orden (mismo
esqueleto que `SYSTEM_FIGURA_RAZONAMIENTO` y `SYSTEM_PROPIEDADES_FISICAS`):

1. **Rol + qué NO hacer** en una frase.
2. **Formato de salida exacto**, con el token de cada línea (`P0:`, `M:`,
   `T:`, `S:`, etc. — ver cada skill). Nunca JSON libre para el modelo chico:
   siempre formato de una línea por dato, igual que ya descubriste que
   funciona mejor que pedirle JSON directo.
3. **Rangos físicos válidos** como tabla de referencia dentro del prompt
   (ej. "el acero ronda 200 GPa, no inventes 5000 GPa"). Esto es anti-alucinación
   preventiva, igual que ya hacés en `SYSTEM_PROPIEDADES_FISICAS`.
4. **2-3 ejemplos few-shot** completos, cortos, sin texto de sobra.

Reglas de estilo (para que el modelo chico no se desvíe):
- Nada de texto explicativo antes/después del bloque de datos.
- Un dato por línea, nunca todo en una sola línea (ya viste que el modelo
  a veces intenta comprimir todo y eso rompe el parser).
- Temperatura baja (0.1–0.2) para estas skills: son de cálculo, no creativas.
  Reservar temperatura más alta solo para nombres/descripciones en lenguaje
  llano (paso 0 tipo `SYSTEM_DESCRIPCION_FISICA`).

### Capa 3 — Salida validada (Python, después de la respuesta del modelo)

Toda skill nueva expone la misma firma de función que `validar_y_corregir_bboxes`:

```python
def validar_y_corregir_<skill>(datos: dict, contexto: dict) -> tuple[dict, bool, list[str]]:
    """
    datos     : lo que parseó _parsear_formato_<skill>() de la respuesta cruda
    contexto  : lo que ya se sabe con certeza (geometría validada, material, etc.)
    devuelve  : (datos_corregidos, es_valido, advertencias)
    """
```

Política de corrección, igual a la que ya usás para bbox:
- **Violación leve** (dentro de tolerancia): clampar en silencio, listar en
  `advertencias`, `es_valido=True`.
- **Violación grave** (fuera de tolerancia, ej. densidad negativa, tensión
  admisible mayor a la de rotura, temperatura por debajo del cero absoluto):
  `es_valido=False` → dispara reintento con prompt reforzado (mismo patrón que
  el paso 2b: se le repite el pedido explicitando qué violó).
- **Sin dato parseable**: usar plantilla de seguridad por categoría de material
  (ya tenés la idea con `plantilla_seguridad` en `generar_figura`). Cada skill
  nueva debería tener su propio diccionario `_VALORES_SEGURIDAD_<skill>` con
  3-4 entradas típicas (ej. genérico metal / genérico plástico / genérico
  madera / genérico compuesto) para no devolver `None` nunca en cadena.

### Reintento — mismo patrón que paso 2b

```
intento 1 (temperatura baja) → validar
  ├─ ok / violación leve   → usar (con clamp)
  └─ violación grave       → intento 2 con prompt reforzado
                              explicitando el valor que violó y su límite
                              → validar de nuevo
                                ├─ ok           → usar
                                └─ sigue mal    → usar clampado del intento 1
                                                   (nunca None, nunca bloquear
                                                   el pipeline)
```

### Caché

Mismo criterio que `figuras_cache/`: cachear por `(nombre_objeto, skill)`,
no por objeto entero. Así si cambiás el material de un objeto no tenés que
recalcular su geometría, y si cambiás la geometría no perdés el cálculo
termodinámico ya hecho. Carpeta sugerida: `<skill>_cache/`, mismo formato de
archivo de texto plano (`clave: valor` por línea) que ya usás — es más
tolerante a errores de parseo que JSON crudo si el archivo se corrompe a mitad.

### Orden de dependencia entre skills (para la próxima etapa de conexión)

```
descripción física (paso 0, ya existe)
        │
        ▼
geometría (paso 1/2, ya existe)  ──────────────┐
        │                                       │
        ▼                                       ▼
ubicación espacial (nueva)          ciencia de materiales (nueva,
  usa la geometría validada           reemplaza/extiende
  para posicionar el objeto           SYSTEM_PROPIEDADES_FISICAS)
  dentro de la escena 3D                       │
        │                            ┌─────────┴─────────┐
        │                            ▼                   ▼
        │                  termodinámica (nueva)   cálculo estructural (nueva)
        │                   usa material + geom.    usa material + geom.
        │                   + condiciones borde      + caso de carga
        └──────────────┬─────────────────────┬────────────┘
                        ▼                     ▼
              todo se guarda junto en objetos_db.json,
              un bloque nuevo por objeto: "espacial", "termico", "estructural"
```

Ninguna skill nueva le pide al modelo algo que ya calculó otra skill: se pasa
el resultado validado como contexto de entrada (regla 4 de la Capa 1).
