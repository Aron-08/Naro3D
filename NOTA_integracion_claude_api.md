# Nota sobre integración con la API de Claude

Todo el pipeline de LLMs de este proyecto corre hoy sobre modelos locales
servidos con Ollama (ver `modelos_config.json` y `ia_interprete._llamar_modelo`),
elegidos específicamente para funcionar en hardware sin GPU dedicada.

La arquitectura ya está preparada para cambiar de backend sin tocar las
skills individuales: cada módulo (`termodinamica.py`, `calculo_estructural.py`,
`electrico.py`, `geometria.py`, `ubicacion.py`, `ensamblador.py` vía `objetos.py`)
llama siempre a `modelos.llamar(skill, ...)`, nunca a Ollama directo — es
`modelos.py` el único punto que sabría de la existencia de un backend
distinto. Eso es intencional: viene de la misma separación de capas del
proyecto ("el LLM decide criterio, Python calcula" — ver
`00_skill_filtro_ruido_datos.md`), que hace que cambiar *cómo* se llama al
modelo sea un cambio acotado en un solo archivo.

No se migró ni se probó contra la API real de Anthropic (Claude) porque
la API se cobra por token de forma sostenida (no hay una capa gratuita
permanente para uso continuo, solo un crédito de prueba único al crear
cuenta) y no está dentro de mi presupuesto actual mantener eso corriendo
mientras itero sobre el proyecto.

En su lugar, dejo documentado en `modelos_api_claude.py` cómo sería esa
integración: mapeo de qué skill iría a qué modelo de Claude (Haiku para
criterio corto, Sonnet para las tareas de composición/razonamiento más
abiertas — nunca Opus, sería pagar de más por decisiones de una línea),
el adaptador que reemplazaría a `_llamar_modelo` respetando el mismo
contrato de entrada/salida, y el cambio de una línea en `modelos.py` que
haría el swap real, skill por skill, sin necesidad de una migración de
una sola vez. Ese archivo no fue ejecutado contra la API real — es una
propuesta de diseño, no un resultado probado.
