# Índice de skills nuevas — Naro Studio

Set de skills pensadas para enchufarse a tu pipeline actual
(`ia_interprete.py` + `objetos.py`), siguiendo el mismo patrón de 3 capas que
ya usás en el paso 0b/2b (dims → generar → clamp/reintento). No tocan nada de
lo que ya tenés; se agregan como pasos nuevos, opcionales, después de que la
geometría y las propiedades básicas ya están validadas.

## Archivos

| Archivo | Qué resuelve | Usa LLM para | Usa Python puro para |
|---|---|---|---|
| `00_skill_filtro_ruido_datos.md` | Arquitectura común (no es una skill en sí, es el contrato que siguen todas) | — | — |
| `01_skill_ubicacion_espacial.md` | Dónde va cada objeto en la escena 3D compartida, sin colisionar | interpretar el pedido ("al lado de X") | detección/resolución de colisión AABB |
| `02_skill_geometria.md` | Auditar que la figura ya generada sea topológicamente válida (cerrada, sin auto-intersección, orientación correcta) antes de exportar a CAD/CFD | clasificar tipo de mecanismo (solo si aplica) | cierre de contorno, shoelace, test de intersección de segmentos |
| `03_skill_ciencia_materiales.md` | Extender la ficha de materiales (Poisson, dilatación térmica, calor específico, fatiga) con chequeo de coherencia | asignar valores realistas por material | validación de rangos (Poisson, fatiga vs. tracción) |
| `04_skill_termodinamica.md` | Conducción, combustión H₂/O₂, riesgo térmico | elegir modelo térmico y modo de falla | TODA la aritmética (Fourier, calorimetría) |
| `05_skill_calculo_estructural.md` | Tensión, factor de seguridad, deflexión, pandeo | elegir fórmula y modo de falla dominante | TODA la aritmética (resistencia de materiales) |
| `06_skill_electrico.md` | Corriente, tensión y potencia en una RED de componentes conectados por contacto | clasificar topología (serie/paralelo/red) y redactar advertencia | TODA el álgebra (análisis nodal, ley de Ohm, I²R) — ni siquiera el criterio hace falta acá |

## Modos de uso en tiempo real (`modos.py`)

Capa de orquestación por encima de las 4 skills numéricas (03 a 06), pensada
para testeo interactivo: un `Modo` (`ModoEstructural`/`ModoTermico`/
`ModoElectrico`) se activa sobre un objeto, y cada tick recalcula con la
entrada de la mano (peso aplicado, cercanía a una fuente de calor, contacto
entre componentes) sin tocar el LLM — cada skill numérica ya expone (o se le
agregó) una función de recálculo puro-Python para esto:
`calculo_estructural.recalcular_carga()`, `termodinamica.paso_integrador_concentrado()`,
`electrico.resolver_red()`. El LLM solo interviene una vez al entrar al modo
(elegir criterio) y, de forma asíncrona con histéresis + cooldown + caché
(`AsesorIA`), para comentar cuando el estado cruza un umbral de verdad — nunca
en el loop de 30 fps. Ver el docstring de `modos.py` para el detalle.

## Orden de dependencia

```
paso 0/0b/1/2/2b (YA EXISTE: descripción física → dims → figura → JSON → validar bbox)
        │
        ▼
02_geometria  (auditoría topológica: cerrado, orientación, sin auto-intersección)
        │
        ├──────────────────────────┐
        ▼                          ▼
01_ubicacion_espacial      03_ciencia_materiales (extiende SYSTEM_PROPIEDADES_FISICAS)
  (pose en la escena)                │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                04_termodinamica        05_calculo_estructural
                (conducción, combustión)  (tensión, FS, deflexión)
```

## Regla transversal (vale para las 5 skills)

En ninguna skill nueva el modelo local hace aritmética de punta a punta. El
LLM decide **criterio** (qué fórmula, qué modo de falla, qué relación
espacial) y Python hace **el cálculo exacto** con esa decisión. Esto es una
extensión directa de algo que ya hacés bien en tu pipeline: el paso 0
(`SYSTEM_DESCRIPCION_FISICA`) separa "pensar la forma" de "poner números"
(paso 1) precisamente porque el modelo chico es mejor decidiendo que
calculando. Estas skills nuevas llevan esa misma separación un paso más allá:
ni siquiera "poner números físicos" se le pide al modelo cuando existe una
fórmula cerrada — solo se le pide elegir cuál aplica.

## Próximo paso

Cuando quieras conectarlas, lo lógico es armar un módulo por skill
(`ubicacion.py`, `geometria.py`, `termodinamica.py`, `calculo_estructural.py`,
y extender `objetos.py` para materiales), cada uno con:
- el `SYSTEM_...` prompt (si esa skill usa LLM),
- el parser del formato de salida (regex, mismo estilo que `_parsear_formato_compacto`),
- la función `validar_y_corregir_<skill>()`.

Todo esto ya está detallado en cada archivo individual, listo para
implementar en el orden que prefieras.