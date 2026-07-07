# Naro Studio — Simulador de Entorno Virtual Controlado por Gestos

> Simulador 3D de ingeniería/física controlado por reconocimiento de manos (MediaPipe),
> con generación de geometría y propiedades físicas asistida por LLMs locales,
> pensado para correr en hardware sin GPU dedicada de gama alta (validado en GTX 1650 Ti 4GB).

---

## Tabla de contenidos

1. [Descripción del proyecto](#descripción-del-proyecto)
2. [Arquitectura y filosofía de diseño](#arquitectura-y-filosofía-de-diseño)
3. [Requisitos](#requisitos)
4. [Instalación](#instalación)
5. [Ejecución](#ejecución)
6. [Configuración de modelos de IA (LLMs)](#configuración-de-modelos-de-ia-llms)
7. [Migrar de Ollama a la API de Claude](#migrar-de-ollama-a-la-api-de-claude)
8. [Estructura del repositorio](#estructura-del-repositorio)
9. [Testing](#testing)
10. [Rendimiento y calidad dinámica](#rendimiento-y-calidad-dinámica)
11. [Limitaciones actuales](#limitaciones-actuales)
12. [Hoja de ruta](#hoja-de-ruta)
13. [Licencia y créditos](#licencia-y-créditos)

---

## Descripción del proyecto

Naro Studio es un entorno virtual 3D en el que los objetos se crean por descripción en
lenguaje natural ("silla de madera", "auto rojo") y se manipulan con la mano a través de
la cámara web (sin guantes ni marcadores). Cada objeto no es solo geometría: tiene una
ficha física completa (material, densidad, resistencias mecánicas, resistividad eléctrica,
conductividad térmica, etc.) que alimenta seis módulos de simulación —ubicación espacial,
geometría, ciencia de materiales, termodinámica, cálculo estructural y circuitos
eléctricos— para que el objeto se comporte de forma físicamente consistente dentro de la
escena.

El caso de uso objetivo es la enseñanza y experimentación en ingeniería/física: armar una
viga y cargarla con la mano para ver si aguanta, acercar dos objetos conductores y ver
cuánta corriente circula, o exponer una pieza a una fuente de calor y observar su
temperatura de régimen — todo en tiempo real, sin necesidad de un software CAD/CAE
tradicional ni GPU de gama alta.

### Características principales

- **Control por gestos**: detección de manos con MediaPipe (multi-criterio, robusta a
  rotación e inclinación), gestos de cámara (dolly, rotación), arrastre de objetos en 3D.
- **Generación de geometría determinística**: un *kernel paramétrico* (`ensamblador.py`)
  donde el LLM solo decide **qué partes tiene el objeto** (forma + dimensiones + relaciones
  de contacto) y Python resuelve la geometría exacta — nunca coordenadas de escena
  escritas a mano por el modelo.
- **Física real, no cosmética**: seis skills numéricas (ubicación, geometría, materiales,
  termodinámica, cálculo estructural, circuitos) donde el LLM decide *criterio* y Python
  hace *todo* el cálculo con fórmulas de ingeniería estándar.
- **Modos interactivos en tiempo real**: estructural, térmico y eléctrico, con
  recálculo a 30 fps sin tocar el modelo de lenguaje en el loop.
- **Fallback generativo**: para objetos con siluetas orgánicas irregulares que el
  catálogo paramétrico no puede aproximar razonablemente, se dispara un pipeline
  texto → imagen (SD-Turbo) → malla 3D (TripoSR).
- **Optimizado para hardware modesto**: decimación de mallas, LOD dinámico según fps
  real, serialización de un solo modelo de Ollama vivo a la vez.
- **Editor visual**: panel gráfico standalone (Tkinter) para inspeccionar y editar
  cada objeto del catálogo (propiedades físicas, color, escala, previsualización 3D).

---

## Arquitectura y filosofía de diseño

El principio rector de todo el proyecto, aplicado sin excepciones en los seis módulos de
física y en el kernel de geometría, es:

> **El LLM decide el criterio. Python calcula todo lo demás, siempre con fórmulas
> exactas de libro.**

Esto no es una preferencia estética: es la respuesta a un problema concreto observado
durante el desarrollo. Los modelos pequeños (3–7B) que corren en hardware sin GPU
dedicada son razonablemente confiables para tomar una **decisión discreta** ("esta viga
está simplemente apoyada", "el material predominante es acero", "esta pieza cilíndrica
es una rueda, va acostada") pero son estructuralmente poco fiables para **aritmética
exacta o para escribir coordenadas de escena a mano** — la versión anterior del pipeline,
donde el LLM escribía puntos (x, y, z) directamente, producía geometría con huecos
microscópicos, bordes que "casi" coincidían y auto-intersecciones, que había que corregir
sistemáticamente aguas abajo.

La arquitectura actual separa estrictamente ambas responsabilidades en cada capa:

| Capa | Qué decide el LLM | Qué calcula Python |
|---|---|---|
| **Geometría** (`ensamblador.py`) | Forma (catálogo cerrado de 8 primitivas), dimensiones, relación de contacto entre partes | Resolución algebraica exacta de cada contacto (`_anclar_contacto`), orden topológico, booleanas (unión/resta) |
| **Ubicación espacial** (`ubicacion.py`) | Intención del pedido ("al lado de X", "arriba de Y") | Resolución de colisión AABB, vector de separación mínima, clamps a cámara |
| **Auditoría geométrica** (`geometria.py`) | Clasificación semántica de mecanismos (engranaje/polea/tornillo) | Cierre de contorno, shoelace (área), test de auto-intersección, centroide |
| **Ciencia de materiales** (skill 03) | Valores realistas por material | Validación de rangos físicos (Poisson ∈ [0, 0.5], fatiga ≤ tracción, etc.) |
| **Termodinámica** (`termodinamica.py`) | Modelo térmico aplicable (concentrado/gradiente), riesgo cualitativo | Fourier, calorimetría, estequiometría de combustión — Python siempre gana si hay discrepancia |
| **Cálculo estructural** (`calculo_estructural.py`) | Fórmula aplicable, resistencia de referencia, modo de falla | Resistencia de materiales completa (viga, columna, torsión, pandeo de Euler) |
| **Circuitos eléctricos** (`electrico.py`) | Solo clasificación cosmética de topología | Análisis nodal modificado (MNA) vía NumPy — ni siquiera el criterio hace falta acá |

Cada skill nueva expone además una función `validar_y_corregir_<skill>()` (capa 3 del
contrato común, ver `00_skill_filtro_ruido_datos.md`) que nunca deja pasar un valor
imposible en silencio: clampa violaciones leves, dispara reintentos dirigidos ante
violaciones graves, y si todo falla recurre a una plantilla de seguridad por categoría —
el pipeline **nunca devuelve `None` en cadena**.

### Cadena de creación de un objeto

```
descripción en lenguaje natural
        │
        ▼
¿existe en biblioteca_mallas/? ── sí ──► Malla real, sin tocar el LLM (milisegundos)
        │ no
        ▼
Kernel paramétrico (objetos.py :: generar_geometria_parametrica)
  1. Agente "concepto"       — desglosa el objeto en partes estructurales
  2. Agente "composición"    — asigna forma + dims_cm + contacto (catálogo cerrado)
  3. Agente "verificación"   — audita sentido físico (ruedas horizontales, etc.)
  4. Validación Pydantic + resolución determinística (ensamblador.py)
        │
        ├─ factible=false (silueta orgánica) ──► TripoSR (texto→imagen→malla 3D)
        └─ error puntual ──► reparación dirigida (LLM), nunca "probar de nuevo a ciegas"
        │
        ▼
Decimación (LOD bajo/alto) + archivado en biblioteca_mallas/
        │
        ▼
Ficha de propiedades físicas (paso 2, secuencial — nunca en paralelo con la geometría)
```

---

## Requisitos

### Hardware
- Webcam.
- CPU moderna (4+ núcleos recomendado para MediaPipe en tiempo real).
- GPU **opcional**: solo se usa para el fallback generativo TripoSR/SD-Turbo. El
  proyecto está validado en una **GTX 1650 Ti de 4GB** y funciona íntegramente en CPU
  si no hay GPU disponible (Ollama corre en CPU sin problema con los modelos elegidos).

### Software
- Python 3.11+
- [Ollama](https://ollama.com) instalado y corriendo localmente (para los LLMs locales)
- Opcional: CUDA + PyTorch si se quiere usar el fallback generativo TripoSR

### Dependencias de Python

```bash
pip install opencv-python numpy mediapipe ollama trimesh pyfqmr manifold3d pydantic
```

| Paquete | Uso |
|---|---|
| `opencv-python` | Captura de cámara, dibujo del panel, proyección |
| `mediapipe` | Detección de manos y rostro (head tracking) |
| `numpy` | Álgebra lineal (resolución de circuitos, transformaciones) |
| `ollama` | Cliente Python para los LLMs locales |
| `trimesh` | Operaciones booleanas, exportación STL, chequeo watertight/manifold |
| `pyfqmr` | Decimación de mallas (quadric edge collapse) |
| `manifold3d` | Backend de operaciones booleanas robustas (unión/resta) |
| `pydantic` | Validación estricta de la salida JSON del LLM en las skills críticas |

Dependencias **opcionales** (solo para el fallback generativo de mallas orgánicas):

```bash
pip install torch diffusers transformers accelerate --break-system-packages
# + TripoSR clonado manualmente desde VAST-AI-Research/TripoSR (no está en PyPI)
```

Si estos paquetes no están instalados, el proyecto sigue funcionando con normalidad:
simplemente cae a la red de seguridad determinística (plantillas paramétricas o caja
genérica) en vez de generar la malla orgánica.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd naro-studio

# 2. Crear un entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt   # o el listado manual de la sección anterior

# 4. Instalar y arrancar Ollama
#    https://ollama.com/download
ollama serve &

# 5. Descargar los modelos que usa el proyecto (ver modelos_config.json)
ollama pull gemma4:e4b
ollama pull qwen3:4b
ollama pull phi4-mini:3.8b
ollama pull deepseek-r1:7b
ollama pull nemotron-mini:4b
ollama pull nomic-embed-text
```

> Los tags exactos pueden variar según lo disponible en el registro de Ollama al momento
> de instalar; usar `ollama list` para confirmar los nombres reales y ajustar
> `modelos_config.json` si difieren.

---

## Ejecución

### Entorno principal (cámara + gestos + simulación 3D)

```bash
python main.py
```

Se abren dos ventanas: el feed de la cámara con el overlay de detección de manos, y el
panel de control (Tkinter) donde se escriben las descripciones de los objetos a crear.

**Atajos de teclado** (ventana de OpenCV):
- `ESC` — salir
- `D` — alternar overlay de debug de detección de dedos
- `E` — alternar modo estéreo anaglifo (requiere gafas rojo/cian)
- `P` — alternar modo de proyección fuera de eje (paralaje de movimiento con head tracking)

### Editor visual standalone

```bash
python objetos.py
# o, equivalentemente:
python editor_visual.py
```

Permite crear, editar, previsualizar (wireframe orbitable) y eliminar objetos del
catálogo sin necesidad de la cámara ni los gestos.

### Optimizar retroactivamente el catálogo existente

Si `objetos_db.json` tiene objetos guardados antes de que se implementara la
decimación automática (mallas con miles de caras sin tope), correr una vez:

```bash
python -c "import optimizacion_objetos as o; o.optimizar_todos_los_objetos()"
```

---

## Configuración de modelos de IA (LLMs)

Toda la configuración de modelos vive en **un único archivo**, `modelos_config.json`.
No hace falta tocar ningún `.py` para cambiar qué modelo usa cada parte del pipeline —
el archivo se relee en cada llamada.

### Cambiar el modelo de una skill puntual

Cada bloque dentro de `"skills"` controla una tarea específica:

```json
"calculo_estructural": {
  "modelo": "phi4-mini:3.8b",
  "temperatura_min": 0.1,
  "temperatura_max": 0.2,
  "num_predict": -1,
  "formato": null
}
```

Basta con editar `"modelo"` con el tag exacto que devuelve `ollama list`. El resto de
los campos:

| Campo | Significado |
|---|---|
| `temperatura_min` / `temperatura_max` | Rango del que se elige la temperatura en cada llamada (colapsado si son iguales) |
| `num_predict` | Tokens máximos de salida (`-1` = sin límite) |
| `formato` | `"json"` activa el modo JSON nativo de Ollama; `null` usa el formato de una línea por dato (`P0:`, `CENTRO:`, `TIPO:`, etc.) |
| `system` | Prompt de sistema (usado automáticamente al llamar con `modelos.llamar(skill, user_content=...)`) |

### Qué modelo usa cada skill (configuración por defecto)

| Skill | Modelo | Motivo |
|---|---|---|
| Composición paramétrica / concepto | `gemma4:e4b`, `deepseek-r1:7b` | Descomposición estructural, requiere más razonamiento |
| Reparación dirigida / verificación | `qwen3:4b` | Tarea acotada, no necesita razonamiento libre |
| Materiales (JSON estructurado) | `phi4-mini:3.8b` | Buen seguimiento de esquema JSON estricto |
| Cálculo estructural / geometría | `qwen3:4b`, `phi4-mini:3.8b` | Clasificación de criterio, formato de una línea |
| Termodinámica | `deepseek-r1:7b` | Razonamiento sobre modelo térmico aplicable |
| Redacción corta (colores, advertencias, asesor) | `nemotron-mini:4b` | Modelo liviano, tarea de baja complejidad |
| Embeddings de biblioteca | `nomic-embed-text` | Búsqueda semántica de mallas ya generadas |

### Punto de entrada único: `modelos.py`

Ningún módulo del proyecto llama a Ollama directamente (excepto `ia_interprete.py`, que
implementa el wrapper de bajo nivel). Todos pasan por:

```python
import modelos
texto = modelos.llamar("nombre_de_la_skill", user_content="...")
```

Esto es lo que hace posible cambiar de backend (ver sección siguiente) tocando un solo
archivo, sin modificar ninguna de las skills individuales.

---

## Migrar de Ollama a la API de Claude

El proyecto está diseñado para migrar de modelos locales a la API de Anthropic sin
reescribir ninguna skill, gracias a la separación de capas descrita arriba. El trabajo
de diseño ya está hecho en `modelos_api_claude.py` y documentado en
`NOTA_integracion_claude_api.md`.

> **Estado actual**: la integración con Claude está **documentada pero no activada por
> defecto ni probada contra la API real** (requiere una `ANTHROPIC_API_KEY` de pago,
> fuera del presupuesto de desarrollo de este proyecto). El código en
> `modelos_api_claude.py` es funcionalmente correcto pero debe tratarse como
> **no verificado en producción** hasta la primera corrida real.

### Pasos para activarla

1. Instalar el SDK oficial:

   ```bash
   pip install anthropic --break-system-packages
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

2. En `modelos.py`, bifurcar la llamada según un campo `"backend"` nuevo en
   `modelos_config.json` (por defecto `"ollama"` para no romper nada existente):

   ```python
   def llamar(skill, messages=None, user_content=None, ...):
       c = config(skill)
       ...
       if c.get("backend") == "claude":
           from modelos_api_claude import llamar_claude
           return llamar_claude(skill, messages, temperatura=temp, num_predict=np_)
       from ia_interprete import _llamar_modelo
       return _llamar_modelo(messages=messages, num_predict=np_,
                              temperatura=temp, modelo=c["modelo"], formato=c.get("formato"))
   ```

3. Agregar `"backend": "claude"` al bloque de la skill que se quiera migrar en
   `modelos_config.json`. La migración puede hacerse **skill por skill** — no es
   necesario un big-bang de una sola vez.

### Mapeo de modelos sugerido (ya definido en `modelos_api_claude.py`)

| Tipo de tarea | Modelo de Claude sugerido | Skills |
|---|---|---|
| Criterio corto, formato de una línea, bajo ambigüedad | `claude-haiku-4-5` | ubicación, geometría, termodinámica, estructural, eléctrico, gesto/color, verificación |
| Razonamiento/descomposición más abierta | `claude-sonnet-5` | composición paramétrica, concepto, propiedades físicas, dibujo (pasos 0a/1) |

No se recomienda usar un modelo de nivel Opus para ninguna de estas skills: son
decisiones de una línea (elegir fórmula, clasificar topología), no tareas que requieran
el modelo más capaz de la línea.

### Diferencias a resolver en una migración real

- **Modo JSON nativo**: Ollama expone `format="json"`; la API de Claude no tiene un
  equivalente directo — se resuelve pidiéndolo explícitamente en el prompt más el
  parseo tolerante que el proyecto ya usa (`ia_interprete._extraer_json`), o con
  `tool_use` y un schema obligatorio.
- **Costo por objeto**: crear un objeto nuevo dispara varias llamadas encadenadas
  (concepto → composición → verificación → reparación si hace falta → propiedades →
  propiedades extendidas). Conviene migrar primero las skills de prompt más corto y
  salida más chica si el presupuesto es limitado.

---

## Estructura del repositorio

```
├── main.py                      # Punto de entrada: cámara, gestos, loop principal
├── objetos.py                   # Catálogo de objetos: orquesta geometría + propiedades
├── ensamblador.py                # Kernel paramétrico determinístico (geometría exacta)
├── malla.py                      # Representación de malla + fábricas de primitivas
├── entorno_virtual.py            # Composición de cámara + figuras + gestos
├── camara.py                     # Proyección de perspectiva + gestos de cámara
├── figura.py                     # Figura3D: wireframe heredado o Malla real
├── render_malla.py               # Camino único de dibujo de cualquier Malla
├── editor_visual.py              # Editor gráfico standalone (Tkinter)
│
├── ia_interprete.py              # Wrapper de bajo nivel a Ollama + pipeline heredado de figuras
├── modelos.py                    # Punto único de lectura de modelos_config.json
├── modelos_config.json           # Configuración de modelo/temperatura por skill
├── modelos_api_claude.py         # Adaptador (no activado) para la API de Claude
│
├── ubicacion.py                  # Skill 01 — pose 3D del objeto en la escena
├── geometria.py                  # Skill 02 — auditoría topológica
├── termodinamica.py              # Skill 04 — conducción, combustión H2/O2
├── calculo_estructural.py        # Skill 05 — tensión, factor de seguridad, pandeo
├── electrico.py                  # Skill 06 — análisis nodal de circuitos (MNA)
├── modos.py                      # Modos interactivos en tiempo real (30 fps, sin LLM)
│
├── biblioteca_mallas.py          # Índice + retrieval de mallas por similitud semántica
├── malla_ia_async.py             # Fallback generativo: SD-Turbo + TripoSR
├── optimizacion_malla.py         # Decimación (pyfqmr) + serialización de mallas
├── optimizacion_objetos.py       # Optimización retroactiva + calidad dinámica (LOD)
├── vision3d.py                   # Head tracking, proyección fuera de eje, anaglifo
├── geo_utils.py                  # Primitivas geométricas 2D compartidas
├── ui_thread.py                  # Helper de threading seguro para Tkinter
│
├── objetos_db.json               # Catálogo persistido (geometría + propiedades)
├── biblioteca_mallas/             # Mallas archivadas (JSON + STL de respaldo)
│
├── 00_skill_filtro_ruido_datos.md   # Contrato de arquitectura común a todas las skills
├── 0{1..6}_skill_*.md               # Especificación de cada skill numérica
│
└── test_*.py                     # Suite de tests (no requiere Ollama corriendo)
```

---

## Testing

La suite de tests cubre toda la lógica determinística (kernel paramétrico, geometría,
malla) y **no requiere Ollama corriendo** — el LLM está completamente ausente del
camino que se testea.

```bash
pip install pytest
pytest -v
```

| Archivo | Cobertura |
|---|---|
| `test_malla.py` | Fábricas de primitivas geométricas |
| `test_ensamblador.py` | Resolución de contactos, orden topológico, ensamble end-to-end |
| `test_extension_primitivas.py` | Primitivas nuevas (cono truncado, cápsula, prisma N-lados), suavizado cosmético |
| `test_fase5_y_seguridad.py` | Operaciones booleanas (resta), red de seguridad determinística, plantillas |

Los tests que dependen de un backend booleano externo (`trimesh` + `manifold3d`) se
saltean automáticamente (`pytest.mark.skipif`) si el backend no está instalado, en vez
de fallar.

---

## Rendimiento y calidad dinámica

El proyecto incluye un `GestorCalidadDinamica` (`optimizacion_objetos.py`) que ajusta
en caliente el nivel de detalle de todas las figuras con malla real según los fps reales
del bucle principal (histéresis de 15 frames para evitar oscilación), sin volver a
decimar nada: solo elige entre LODs ya precalculados (`alto` → `normal` → `bajo` →
`emergencia`).

Puntos de optimización activos:
- Resolución de captura de cámara desacoplada de la resolución de inferencia de
  MediaPipe (más nitidez visual sin costo de fps).
- Un solo overlay-blend por malla completa en vez de uno por triángulo
  (`render_malla.py`), corrigiendo una regresión que llegó a colapsar el framerate a
  ~0.3 fps con objetos sin decimar.
- Serialización estricta de las llamadas a Ollama (`_LOCK_OLLAMA` global) para evitar
  que dos contextos de modelo convivan en RAM/VRAM limitada.

---

## Limitaciones actuales

Esta sección se mantiene deliberadamente honesta sobre el estado real del proyecto.

- **Integración con la API de Claude no verificada**: el código en
  `modelos_api_claude.py` nunca corrió contra la API real (ver sección de migración).
  Tratarlo como una propuesta de diseño, no como una ruta probada.
- **`perfil_extruido` no expuesto al LLM**: la fábrica de mallas para secciones no
  circulares (perfiles en L, en T) existe en `malla.py` pero el esquema Pydantic actual
  de `Parte.dims_cm` es un dict plano de escalares, incompatible con una lista de
  puntos arbitraria. Requiere extender el esquema antes de exponerlo.
- **Sin campo de fidelidad explícito**: cuando un objeto se resuelve por biblioteca
  (coincidencia por embedding) o cae a una plantilla/caja genérica, esa información
  existe en tiempo de generación pero no se persiste como un campo consultable
  (`"completa" | "plantilla" | "generica"`) en `objetos_db.json`. Un fallback silencioso
  es indistinguible de una generación exitosa en los logs actuales.
- **Coherencia de embeddings solo textual**: `biblioteca_mallas.buscar()` hace
  similitud coseno sobre el **nombre/descripción** del objeto, no sobre su geometría —
  dos objetos con nombres parecidos pero formas muy distintas pueden dar un HIT
  indebido.
- **Propiedades físicas sin trazabilidad a bases de datos reales**: los valores de
  resistencia, densidad, módulo elástico, etc. son estimaciones de un LLM pequeño sin
  verificación contra fuentes de ingeniería (MatWeb, ASM). Las fórmulas que operan
  sobre esos valores son matemáticamente exactas, pero los datos de entrada no tienen
  garantía de precisión — esto es aceptable para fines educativos/exploratorios, **no
  para diseño de ingeniería real**.
- **`03_skill_ciencia_materiales` con actualizaciones pendientes**: quedó
  explícitamente diferida en las últimas sesiones de desarrollo, junto con la extensión
  del campo de fidelidad mencionado arriba.
- **Geometría 2D/CFD legada convive con el kernel paramétrico**: `ia_interprete.py`
  conserva el pipeline viejo de coordenadas por texto (`generar_figura`) para el caso
  de figuras 2D orientadas a exportación CAD/CFD; el kernel paramétrico 3D
  (`ensamblador.py`) es el camino principal para objetos nuevos del catálogo, pero
  ambos coexisten en el código.
- **Sin soporte multiplataforma probado**: el proyecto asume una webcam UVC estándar y
  fue desarrollado y calibrado en Windows/Linux con GPU NVIDIA; no se ha validado en
  macOS ni con cámaras no estándar.
- **Calibración de visión 3D (paralaje/anaglifo) es manual por equipo**: `vision3d.py`
  requiere medir a mano el tamaño físico del monitor y el offset de la cámara
  (`CalibracionPantalla`) para que el efecto de profundidad "cierre" correctamente; no
  hay autocalibración.
- **Sin persistencia de sesiones de simulación**: los modos interactivos
  (`modos.py`) no guardan historial de una sesión de prueba de carga/térmica/eléctrica
  — el estado vive solo mientras el modo está activo.

---

## Hoja de ruta

- Exponer `perfil_extruido` al LLM extendiendo el esquema Pydantic de `dims_cm`.
- Agregar el campo `fidelidad` (`"completa" | "plantilla" | "generica"`) a cada
  registro de `objetos_db.json`.
- Completar la actualización pendiente de `03_skill_ciencia_materiales`.
- Primera corrida real contra la API de Claude y ajuste del mapeo de modelos según
  resultados empíricos de calidad/costo.
- Embeddings geométricos (no solo textuales) para `biblioteca_mallas.buscar()`.

---

## Licencia y créditos

Proyecto personal de investigación/aprendizaje en simulación física interactiva.
Construido sobre:

- [Ollama](https://ollama.com) — servido de modelos de lenguaje locales
- [MediaPipe](https://developers.google.com/mediapipe) — detección de manos y rostro
- [trimesh](https://trimesh.org/) / [manifold3d](https://github.com/elalish/manifold) — geometría y booleanas 3D
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) — generación de malla desde imagen
- [OpenCV](https://opencv.org/) — captura y render 2D
