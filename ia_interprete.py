"""
Configuración del llamado a un modelo de IA local (servido por Ollama) para el entorno virtual.

El modelo NO ve imágenes ni video en ningún momento: tanto para crear figuras como para
interpretar gestos, se le manda solamente texto/JSON con datos ya procesados (puntos relativos,
nombres de dedos, booleanos de contacto, etc.). Esto es justamente lo que lo hace eficiente:
no tiene que "entender" una foto, solo razonar sobre unos pocos datos estructurados.

Requisitos para usar este módulo:
    1) Tener Ollama instalado y corriendo (https://ollama.com)
    2) Haber descargado el modelo una vez con:
           ollama pull kwangsuklee/Qwen3.5-4B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2
    3) Instalar el cliente de Python:
           pip install ollama
"""

import json
import math
import os
import re
import threading
import time
import unicodedata

import ollama

from geo_utils import segmentos_cruzan as _segmentos_se_cruzan  # test de cruce: única
                                                                  # implementación, ver geo_utils.py

# ---------------------------------------------------------------------------
# Caché en disco
# ---------------------------------------------------------------------------

CACHE_DIR = "figuras_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _nombre_archivo_cache(descripcion: str) -> str:
    nombre = descripcion.lower().strip()
    nombre = "".join(
        c for c in unicodedata.normalize("NFD", nombre)
        if unicodedata.category(c) != "Mn"
    )
    nombre = re.sub(r"[^a-z0-9 ]+", "", nombre)
    nombre = re.sub(r"\s+", "_", nombre)
    return os.path.join(CACHE_DIR, nombre[:60] + ".txt")


def _guardar_cache(descripcion: str, datos: dict) -> None:
    ruta = _nombre_archivo_cache(descripcion)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(f"descripcion: {descripcion}\n")
        f.write(f"puntos: {json.dumps(datos['puntos'])}\n")
        f.write(f"conexiones: {json.dumps(datos['conexiones'])}\n")
        f.write(f"primitivas: {json.dumps(datos.get('primitivas', []))}\n")
        f.write(f"generado: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"[caché] Guardado en {ruta}")


def _cargar_cache(descripcion: str) -> dict | None:
    ruta = _nombre_archivo_cache(descripcion)
    if not os.path.exists(ruta):
        return None
    try:
        datos = {}
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                if linea.startswith("puntos:"):
                    datos["puntos"] = json.loads(linea.split(":", 1)[1].strip())
                elif linea.startswith("conexiones:"):
                    datos["conexiones"] = json.loads(linea.split(":", 1)[1].strip())
                elif linea.startswith("primitivas:"):
                    datos["primitivas"] = json.loads(linea.split(":", 1)[1].strip())
        if "puntos" in datos and "conexiones" in datos:
            if "primitivas" not in datos:
                datos["primitivas"] = []
            print(f"[caché] '{descripcion}' cargado desde {ruta}")
            return datos
    except Exception as e:
        print(f"[caché] Error al leer: {e}")
    return None


def figuras_en_cache() -> list[str]:
    """Lista todas las descripciones guardadas en caché."""
    resultado = []
    if not os.path.exists(CACHE_DIR):
        return resultado
    for arch in os.listdir(CACHE_DIR):
        if not arch.endswith(".txt"):
            continue
        ruta = os.path.join(CACHE_DIR, arch)
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                primera = f.readline()
            if primera.startswith("descripcion:"):
                resultado.append(primera.split(":", 1)[1].strip())
        except Exception:
            pass
    return resultado

MODELO = "kwangsuklee/Qwen3.5-4B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2"

# ---------------------------------------------------------------------------
# Catálogo de figuras base (2D y 3D) — hardcodeado en figuras_base.json
#
# Por qué hace falta: dejar que el modelo "invente" las coordenadas de un cubo
# punto por punto es exactamente lo que generaba el bug (una figura como si
# fuera un cierre relámpago en vez de un cubo: 6 puntos conectados en cadena,
# porque el modelo no sabe cerrar bien un contorno rectangular a mano).
#
# La solución es sacarle esa responsabilidad al modelo para las formas más
# comunes: este catálogo define, de una vez y en código Python (no en el LLM),
# la geometría "unitaria" (centrada en el origen, tamaño 1) de cada figura
# base — tanto 2D (círculo, cuadrado, triángulo, estrella, ...) como 3D
# (cubo, esfera, cilindro, pirámide, ...). El modelo SOLO tiene que decidir
# QUÉ figura base usar para cada parte del objeto (algo que hace bien, es
# clasificación) y en qué bounding-box va (eso ya lo resuelve el paso 0b).
# Con esos dos datos, `_ensamblar_figura_por_catalogo()` hace el trabajo
# geométrico: escala la plantilla unitaria al tamaño del bbox y la traslada
# a su posición — sin que el LLM tenga que calcular ni un solo número.
# ---------------------------------------------------------------------------

CATALOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras_base.json")

_CATALOGO_CACHE: dict | None = None


def _cargar_catalogo() -> dict:
    """Carga (y cachea en memoria) el catálogo de figuras base desde el JSON."""
    global _CATALOGO_CACHE
    if _CATALOGO_CACHE is not None:
        return _CATALOGO_CACHE
    try:
        with open(CATALOGO_PATH, "r", encoding="utf-8") as f:
            _CATALOGO_CACHE = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[catálogo] No se pudo cargar {CATALOGO_PATH}: {e}")
        _CATALOGO_CACHE = {"2D": {}, "3D": {}}
    return _CATALOGO_CACHE


def _normalizar_texto(s: str) -> str:
    """minúsculas, sin acentos, sin espacios de sobra — para comparar nombres de forma
    tolerante a como los escriba el modelo o el usuario."""
    s = s.lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", "_", s)


def _resolver_forma_base(nombre_forma: str) -> tuple[str, str, dict] | None:
    """Busca `nombre_forma` (tal como lo escribió el modelo) en el catálogo, por
    clave exacta o por alias. Devuelve (clave_canonica, categoria, definicion) o
    None si no está en el catálogo (en ese caso, se recurre al LLM como antes)."""
    if not nombre_forma:
        return None
    objetivo = _normalizar_texto(nombre_forma)
    catalogo = _cargar_catalogo()
    for categoria in ("3D", "2D"):
        for clave, defin in catalogo.get(categoria, {}).items():
            candidatos = {_normalizar_texto(clave)} | {
                _normalizar_texto(a) for a in defin.get("aliases", [])
            }
            if objetivo in candidatos:
                return clave, categoria, defin
    return None


def _texto_catalogo_para_prompt() -> str:
    """Serializa el catálogo a texto legible para inyectarlo en el prompt del
    descriptor físico (paso 0a), como lista cerrada de opciones válidas."""
    catalogo = _cargar_catalogo()
    lineas = []
    for categoria, etiqueta in (("3D", "Formas base 3D (objetos con volumen)"),
                                 ("2D", "Formas base 2D (siluetas planas)")):
        lineas.append(f"{etiqueta}:")
        for clave, defin in catalogo.get(categoria, {}).items():
            alias = defin.get("aliases", [])
            alias_txt = f" (alias: {', '.join(alias)})" if alias else ""
            lineas.append(f"  - {clave}{alias_txt}: {defin.get('descripcion', '')}")
    return "\n".join(lineas)

# ----------------------------------------------------------------------------------
# Paso 0: descripción física en lenguaje llano, SIN coordenadas.
#
# Por qué hace falta: pedirle al modelo que piense "qué forma es cada parte, qué tamaño
# tiene relativo a las demás y dónde se ubica" es un problema mucho más fácil que pedirle
# directamente coordenadas numéricas. Resolviendo primero esto, el paso 1 (que sí tiene que
# convertir todo a números) parte de una lista de partes ya pensada en vez de tener que
# inventar la forma del objeto Y las coordenadas al mismo tiempo. Esto reduce errores como
# mezclar líneas sueltas con primitivas o ubicar partes en lugares incoherentes (oreja en el
# medio de la cara, hocico arriba de los ojos, etc.).
# ----------------------------------------------------------------------------------

SYSTEM_DESCRIPCION_FISICA_BASE = """Describís objetos del mundo real en términos de sus partes geométricas básicas.
NO usés coordenadas ni números de posición: solo lenguaje descriptivo.
NO hacés esto por gusto: todo lo que escribas acá es la ÚNICA fuente de verdad que va a usar el
siguiente paso para poner números. Si acá una unión queda ambigua, en el dibujo final se va a
notar como un hueco o un cruce entre partes. Precisión acá = precisión en el resultado final.

Para cada parte del objeto, indicá en una línea con este formato EXACTO (separado por " | "):

  - NombreParte: forma_base=<clave_del_catalogo> | volumen=2D o 3D | tamaño=grande/mediano/chico/muy_chico | ubicacion=<frase corta> | contacto=<contrato_de_union>

Reglas para cada campo:
  - forma_base: tenés que elegir EXACTAMENTE una clave del catálogo de abajo (o uno de sus alias).
    Es una lista CERRADA: no inventes formas que no estén en el catálogo. Si el objeto real no
    encaja perfecto en ninguna, elegí la forma base más parecida (ej: una lata de gaseosa es
    "cilindro", una pantalla de TV es "rectangulo" o "cubo" según si te interesa el volumen).
  - volumen: "3D" si esa parte tiene un volumen real que importa mostrar (una caja, una pelota,
    un tubo), "2D" si es más bien una silueta plana (una ventana, un ojo, un logo).
  - tamaño: relativo a las demás partes del MISMO objeto. Si dos partes forman un par simétrico
    (oreja izq/der, rueda izq/der, ojo izq/der), las dos llevan EXACTAMENTE la misma palabra de
    tamaño — nunca "chico" en una y "muy_chico" en la otra, eso después se traduce en un par
    desparejo en el dibujo.
  - ubicacion: en una frase corta, dónde va relativa al resto (arriba/abajo/centro/izquierda/
    derecha/dentro de otra parte).
  - contacto: describe la unión física con otras partes usando SIEMPRE uno de estos 3 formatos
    exactos, sin inventar variantes propias:
      ninguna
          → la parte no toca a ninguna otra (puede solaparse levemente o flotar cerca, pero no
            comparte un borde/cara que tenga que coincidir pixel a pixel).
      toca:<lado_propio>=<lado_de_la_otra>:<NombreOtraParte>
          → declara qué borde/cara PROPIO coincide EXACTAMENTE (mismo valor numérico, sin
            aproximar) con qué borde/cara de qué otra parte ya descripta antes. <lado> es
            siempre uno de: arriba, abajo, izquierda, derecha, centro.
            Ejemplos: "toca:abajo=arriba:Cuerpo" (mi borde de abajo = borde de arriba del
            Cuerpo). "toca:arriba=abajo:Carroceria" (mi borde de arriba = borde de abajo de
            la Carroceria).
      simetrica_a:<NombreOtraParte>
          → esta parte es el espejo exacto de otra ya descripta antes en la lista: mismo
            tamaño y forma, posición reflejada sobre el eje vertical central del objeto.
    "contacto" es la ÚNICA fuente de verdad sobre qué partes se tocan. El paso siguiente usa
    literalmente este campo para decidir qué números tienen que repetirse exactos entre dos
    partes, así que nunca lo dejes en "ninguna" si en el objeto real esas partes se tocan.

Reglas generales:
  - Listá TODAS las partes visualmente distinguibles del objeto real, en orden lógico: de
    afuera hacia adentro, o de arriba hacia abajo. Antes de responder, repasá el objeto real
    de punta a punta (silueta exterior → partes que sobresalen → detalles internos → detalles
    chicos simétricos) y confirmá que ninguna quedó afuera de la lista. Es preferible listar una
    parte de más (marcada chica) que olvidar una parte que el objeto realmente tiene.
  - Un objeto simple (una caja, una pelota, un cubo) puede tener UNA SOLA parte: "Cuerpo".
    No inventes partes de más si el objeto real no las tiene.
  - Pensá en proporciones realistas del objeto (ej: las orejas de un oso son chicas comparadas
    con la cabeza; el techo de una casa es tan ancho como el cuerpo, no más angosto).
  - Todo par simétrico (izquierda/derecha) se describe con la segunda parte del par usando
    "contacto=simetrica_a:<primera_parte>", nunca con ubicacion/tamaño inventados por separado
    que puedan terminar en dos tamaños distintos.
  - No expliches nada extra, no agregues introducción ni conclusión, no markdown.

=== ROLES CON POSICIÓN AUTOMÁTICA (usalos tal cual si tu parte cumple ese rol) ===
Si una parte de tu objeto cumple literalmente uno de estos roles, nombrala EXACTAMENTE con una
de estas palabras clave en NombreParte. Un sistema determinístico (no el modelo) le calcula
coordenadas exactas a estos roles antes de dibujar, así que usarlos elimina por completo el
riesgo de que esa parte quede desalineada — es la forma más confiable de garantizar un encastre
perfecto:
  cuerpo / carroceria / torso / base / casco / fuselaje  → estructura central del objeto
  techo / copa                                            → arriba del cuerpo, mismo ancho que él
  cabeza                                                   → arriba del cuerpo/torso, centrada
  puerta / entrada                                         → centrada, apoyada en el piso del cuerpo
  ventana / ojo / ojos                                     → mitad derecha, cuarto superior del cuerpo
  tronco / pie / pierna / patas                            → debajo del cuerpo, centrado
  rueda / neumatico / llanta                                → debajo del cuerpo, a los costados
Si tu parte no encaja en ninguno de estos roles no pasa nada, usá igual el campo "contacto" para
que quede unida a su vecina sin huecos.

=== CATÁLOGO DE FORMAS BASE (usar SOLO estas claves en forma_base) ===
{catalogo}

=== EJEMPLOS ===

cubo de plastico:
- Cuerpo: forma_base=cubo | volumen=3D | tamaño=grande | ubicacion=ocupa todo el objeto | contacto=ninguna

pelota de futbol:
- Cuerpo: forma_base=esfera | volumen=3D | tamaño=grande | ubicacion=ocupa todo el objeto | contacto=ninguna

lata de gaseosa:
- Cuerpo: forma_base=cilindro | volumen=3D | tamaño=grande | ubicacion=ocupa todo el objeto | contacto=ninguna

oso:
- Cuerpo: forma_base=circulo | volumen=2D | tamaño=grande | ubicacion=parte inferior del objeto, base de todo | contacto=ninguna
- Cabeza: forma_base=circulo | volumen=2D | tamaño=grande, un poco mas chica que el cuerpo | ubicacion=arriba del cuerpo | contacto=toca:abajo=arriba:Cuerpo
- Oreja izquierda: forma_base=circulo | volumen=2D | tamaño=chico | ubicacion=arriba a la izquierda de la cabeza | contacto=toca:abajo=arriba:Cabeza
- Oreja derecha: forma_base=circulo | volumen=2D | tamaño=chico | ubicacion=arriba a la derecha de la cabeza | contacto=simetrica_a:Oreja izquierda
- Ojo izquierdo: forma_base=circulo | volumen=2D | tamaño=muy_chico | ubicacion=dentro de la cabeza, a media altura, a la izquierda del centro | contacto=ninguna
- Ojo derecho: forma_base=circulo | volumen=2D | tamaño=muy_chico | ubicacion=dentro de la cabeza, a media altura, a la derecha del centro | contacto=simetrica_a:Ojo izquierdo
- Hocico: forma_base=circulo | volumen=2D | tamaño=chico | ubicacion=dentro de la cabeza, debajo de los ojos, centrado | contacto=ninguna

casa:
- Cuerpo: forma_base=cubo | volumen=3D | tamaño=grande | ubicacion=mitad inferior del objeto | contacto=ninguna
- Techo: forma_base=techo_a_dos_aguas | volumen=3D | tamaño=mismo ancho que el cuerpo | ubicacion=arriba del cuerpo | contacto=toca:abajo=arriba:Cuerpo
- Puerta: forma_base=rectangulo | volumen=2D | tamaño=chico | ubicacion=dentro del cuerpo, centrada horizontalmente, tocando el borde inferior | contacto=toca:abajo=abajo:Cuerpo
- Ventana: forma_base=circulo | volumen=2D | tamaño=chico | ubicacion=dentro del cuerpo, mitad superior | contacto=ninguna

auto:
- Carroceria: forma_base=rectangulo | volumen=2D | tamaño=grande | ubicacion=centro del objeto | contacto=ninguna
- Rueda izquierda: forma_base=circulo | volumen=2D | tamaño=chico | ubicacion=abajo a la izquierda de la carroceria | contacto=toca:arriba=abajo:Carroceria
- Rueda derecha: forma_base=circulo | volumen=2D | tamaño=chico | ubicacion=abajo a la derecha de la carroceria | contacto=simetrica_a:Rueda izquierda
"""


def _system_descripcion_fisica() -> str:
    """Arma el system prompt del paso 0a con el catálogo de formas base ya
    embebido, para que la lista de opciones válidas se mantenga sincronizada
    con figuras_base.json sin tener que tocar este prompt a mano."""
    return SYSTEM_DESCRIPCION_FISICA_BASE.format(catalogo=_texto_catalogo_para_prompt())


# ----------------------------------------------------------------------------------
# Paso 1: razonamiento compacto.
# Formato fijo de una línea por punto y una línea de conexiones.
# Cuanto más corto y estructurado, menos tokens necesita y menos probable que se corte.
# ----------------------------------------------------------------------------------

SYSTEM_FIGURA_RAZONAMIENTO = """Dibujás objetos 3D en un panel. Coordenadas X,Y: (0,0)=arriba izq, (1,1)=abajo der, centrá en (0.5,0.5). Coordenada Z: 0=frente cámara, 1=fondo, 0.5=plano neutro (igual que 2D anterior).

=== REGLA FUNDAMENTAL ===
Elegí UNO de estos dos enfoques y aplicalo COMPLETO. NUNCA mezcles:
  A) SOLO PRIMITIVAS: escribí "L:" vacío y listá C:/R:/E:/S:/K:/Y: para todo.
  B) SOLO LÍNEAS: definí todos los puntos Px: y conectalos TODOS en L:

Usá A para objetos con partes redondas (animales, caras, vehículos con ruedas, esferas).
Usá B para siluetas angulares puras (estrella, cohete, casa solo techo+paredes).
Para objetos mixtos (casa con ventana redonda, auto con ruedas), usá A: líneas en L: + primitivas.

=== REGLA OBLIGATORIA PARA CAJAS Y CUBOS ===
Si el objeto (o una parte) es una caja, cubo, bloque, dado o cualquier prisma rectangular:
PROHIBIDO representarlo con puntos Px: y líneas L:. Un modelo chico casi siempre arma mal el
cierre del contorno y termina dibujando un zigzag en vez de un cubo. Usá SIEMPRE la primitiva
K: cx,cy,cz,ancho,alto,profundo (ancho=alto=profundo si es un cubo perfecto). Lo mismo aplica
a un cuadrado o rectángulo chato (sin volumen): usá SIEMPRE R:, nunca 4 puntos sueltos.

=== REGLA DE UNIÓN EXACTA ENTRE PARTES (esquinas y bordes SIEMPRE coinciden, cero tolerancia) ===
Recibís, para cada parte, un campo "contacto" (toca:<lado_propio>=<lado_otra>:<Otra>,
simetrica_a:<Otra>, o ninguna) y, si aplica, un bloque [DIMENSIONES ASIGNADAS] con
bounding-boxes exactas. Estas dos cosas son la ÚNICA verdad geométrica: nunca "estimes a ojo"
dónde cae el borde de una parte que toca a otra, siempre CALCULÁ o COPIÁ el número exacto de
la parte con la que hace contacto. Un borde que "casi" coincide (0.40 contra 0.41) se ve en el
render como una grieta o un escalón — es un error tan grave como que la parte quede flotando.
Aplicá estas reglas en orden de prioridad:
  1. Si hay [DIMENSIONES ASIGNADAS] para una parte, sus puntos o su primitiva tienen que caer
     DENTRO de ese bbox, y si el bbox de esa parte comparte x_min/x_max/y_min/y_max con el de
     otra parte vecina, usá EXACTAMENTE esos mismos decimales — no los redondees ni los ajustes
     "para que se vea mejor a ojo".
  2. Partes de puntos (Px:/L:) que según "contacto" comparten un borde: NO definas el punto de
     unión dos veces con números parecidos-pero-distintos. Definí ese punto UNA sola vez (en la
     primera parte que lo necesita) y REUTILIZÁ el mismo índice Pn en las conexiones L: de la
     otra parte también. Dos puntos "casi iguales" (0.22,0.45 y 0.221,0.449) dejan un hueco
     microscópico que rompe el contorno cerrado; el mismo índice reutilizado lo hace
     matemáticamente imposible. Mirá el ejemplo "casa" abajo: P1 y P2 son a la vez las esquinas
     de arriba de las paredes Y la base del techo — un solo punto, dos usos.
  3. Una parte-primitiva (K:/R:/C:/E:/S:/Y:) cuyo "contacto" apunta a otra parte: calculá tu
     coordenada de contacto a partir del centro y tamaño YA USADOS por esa otra parte, con la
     misma cantidad de decimales. Ejemplo: el cuerpo es K: 0.50,0.55,0.50,0.40,0.30,0.30 → su
     cara de arriba está en y = cy - alto/2 = 0.55 - 0.15 = 0.40. Si el techo tiene
     "contacto=toca:abajo=arriba:Cuerpo", el techo apoya exactamente en y=0.40 — nunca en 0.39
     ni en 0.41. Mismo criterio para x_min/x_max cuando el contacto es "izquierda"/"derecha".
  4. "contacto=simetrica_a:X": mismo tamaño EXACTO que X (mismo radio/ancho/alto/profundo), y
     posición reflejada sobre el eje x=0.5 (si X tiene cx=0.32, la simétrica va en
     cx=1.0-0.32=0.68 — misma distancia al centro, lado opuesto, mismo cy y cz que X).
  5. "contacto=ninguna" no exime de coherencia: la parte igual tiene que quedar dentro de la
     escena (rango [0,1] en x,y,z) y no atravesar otra parte salvo que la ubicación lo pida
     explícitamente (ej. "se solapa con").

=== CHECKLIST ANTI-OLVIDO (repasar mentalmente antes de responder; NO escribir esto en la salida) ===
  - ¿Cada parte de la descripción física tiene su línea Pn/L o su primitiva en tu salida? Ninguna
    parte descripta puede faltar en la figura final.
  - ¿Cada parte con "contacto=toca:..." comparte el número exacto con su vecina (regla 2 o 3)?
  - ¿Cada parte con "contacto=simetrica_a:..." tiene el mismo tamaño y la posición reflejada?
  - ¿Usaste K: para TODA caja/cubo y R: para TODO rectángulo/cuadrado chato, sin excepción?

=== FORMATO ===
Px: x,y[,z]      ← punto 3D (z opcional; si omitís z se usa 0.5)
L: 0-1, 1-2, ... ← conexiones (escribí "L:" aunque esté vacío)
C: cx,cy,r[,cz]  ← círculo (cz opcional)
R: x,y,w,h[,cz]  ← rectángulo esquina sup-izq (cz opcional)
E: cx,cy,rx,ry[,cz] ← elipse (cz opcional)
S: cx,cy,cz,r    ← esfera 3D
K: cx,cy,cz,w,h,d ← cubo 3D (ancho, alto, profundo)
Y: cx,cy,cz,r,h  ← cilindro 3D

Sin texto extra, sin markdown, sin títulos. Solo el formato.

=== EJEMPLOS ===

cubo de plastico:
L:
K: 0.50,0.50,0.50,0.35,0.35,0.35

caja de carton:
L:
K: 0.50,0.55,0.50,0.40,0.30,0.30

torre de dos cubos:
L:
K: 0.50,0.65,0.50,0.30,0.30,0.30
K: 0.50,0.35,0.50,0.22,0.30,0.22

oso:
L:
C: 0.50,0.45,0.28
C: 0.32,0.22,0.10
C: 0.68,0.22,0.10
C: 0.50,0.60,0.09
C: 0.38,0.42,0.05
C: 0.62,0.42,0.05

gato:
L:
C: 0.50,0.52,0.26
C: 0.50,0.26,0.14
C: 0.33,0.16,0.07
C: 0.67,0.16,0.07
C: 0.43,0.22,0.04
C: 0.57,0.22,0.04
C: 0.50,0.30,0.03

auto:
P0: 0.12,0.62
P1: 0.22,0.42
P2: 0.38,0.30
P3: 0.62,0.30
P4: 0.78,0.42
P5: 0.88,0.62
L: 0-1, 1-2, 2-3, 3-4, 4-5, 5-0
C: 0.27,0.67,0.08
C: 0.73,0.67,0.08

casa:
P0: 0.50,0.18
P1: 0.22,0.45
P2: 0.78,0.45
P3: 0.22,0.82
P4: 0.78,0.82
L: 0-1, 0-2, 1-3, 2-4, 3-4
R: 0.40,0.57,0.20,0.25

persona:
P0: 0.50,0.36
P1: 0.28,0.42
P2: 0.72,0.42
P3: 0.50,0.62
P4: 0.34,0.82
P5: 0.66,0.82
L: 0-1, 0-2, 0-3, 3-4, 3-5
C: 0.50,0.22,0.13

estrella:
P0: 0.50,0.12
P1: 0.61,0.40
P2: 0.88,0.40
P3: 0.67,0.57
P4: 0.76,0.85
P5: 0.50,0.68
P6: 0.24,0.85
P7: 0.33,0.57
P8: 0.12,0.40
P9: 0.39,0.40
L: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-0

avion:
P0: 0.10,0.52
P1: 0.30,0.44
P2: 0.75,0.44
P3: 0.88,0.52
P4: 0.75,0.60
P5: 0.25,0.60
P6: 0.38,0.52
P7: 0.55,0.52
P8: 0.58,0.78
P9: 0.36,0.78
P10: 0.72,0.44
P11: 0.84,0.44
P12: 0.80,0.24
L: 0-1, 1-2, 2-3, 3-4, 4-5, 5-0, 6-7, 7-8, 8-9, 9-6, 10-11, 11-12, 12-10

sol:
L:
C: 0.50,0.50,0.20
C: 0.50,0.18,0.05
C: 0.50,0.82,0.05
C: 0.18,0.50,0.05
C: 0.82,0.50,0.05
C: 0.27,0.27,0.05
C: 0.73,0.27,0.05
C: 0.27,0.73,0.05
C: 0.73,0.73,0.05

arbol:
P0: 0.50,0.15
P1: 0.70,0.30
P2: 0.75,0.50
P3: 0.50,0.58
P4: 0.25,0.50
P5: 0.30,0.30
P6: 0.44,0.58
P7: 0.56,0.58
P8: 0.56,0.88
P9: 0.44,0.88
L: 0-1, 1-2, 2-3, 3-4, 4-5, 5-0, 6-7, 7-8, 8-9, 9-6

corazon:
P0: 0.50,0.78
P1: 0.18,0.38
P2: 0.50,0.20
P3: 0.82,0.38
L: 0-1, 0-3
C: 0.34,0.30,0.17
C: 0.66,0.30,0.17
"""

# ----------------------------------------------------------------------------------
# Paso 2: convierte el formato compacto a JSON.
# Tarea puramente mecánica: copiar números, no razonar.
# ----------------------------------------------------------------------------------

SYSTEM_FIGURA_JSON = """Convertí la lista de puntos y lineas al siguiente JSON.
Respondé SOLO con el JSON, sin texto adicional ni markdown.

Formato:
{"puntos": [[x0,y0],[x1,y1],...], "conexiones": [[i,j],[i,k],...]}

Regla: cada "Pi: x,y" es un punto. Cada par "i-j" en la linea L es una conexion [i,j].
Esta es una tarea de COPIADO exacto, no de redondeo: transcribí cada número tal cual aparece,
con todos sus decimales. Si dos puntos distintos tienen el mismo valor de x o de y (un borde
compartido entre dos partes), ese valor tiene que llegar idéntico a los dos en el JSON — no
los "limpies" ni los acerques a un número redondo.
"""

SYSTEM_GESTO = """Recibís datos ya procesados de una mano (qué dedos están levantados o cerrados) y si
la mano está tocando un objeto en un entorno virtual 2D. No recibís imágenes, solo estos datos.
Respondé ÚNICAMENTE con un JSON válido, sin texto adicional, sin explicaciones, sin markdown.
Formato exacto:
{"color": [B, G, R]}

Donde B, G, R son enteros entre 0 y 255 (formato de color de OpenCV) que indican qué color debería
tomar el objeto según el gesto detectado. Si "contacto" es false, sugerí el color de reposo (algo neutro).
Si "contacto" es true, el color puede variar según qué gesto esté haciendo la mano al tocar el objeto
(por ejemplo, un puño cerrado podría sugerir un color distinto a una mano abierta).
"""


# ---------------------------------------------------------------------------
# Prompts del pipeline de 4 pasos
# ---------------------------------------------------------------------------

# Paso 1 — igual que SYSTEM_FIGURA_RAZONAMIENTO (ya definido arriba), se reutiliza.


def _extraer_json(texto):
    """Limpia la respuesta del modelo y la convierte en un dict de Python.

    Estrategias en orden de preferencia:
    1. Parse directo tras limpiar <think> y markdown
    2. Buscar el primer { ... } completo contando llaves de apertura/cierre
    3. Reparación mínima: comillas simples -> dobles, claves sin comillas -> con comillas
    """
    if not texto:
        return None

    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    texto = re.sub(r"```(?:json)?\s*", "", texto).strip()
    texto = re.sub(r"```\s*$", "", texto).strip()

    # Intento 1: parse directo
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    # Intento 2: extraer el primer objeto JSON completo contando llaves
    inicio = texto.find("{")
    if inicio != -1:
        profundidad = 0
        for i, c in enumerate(texto[inicio:], start=inicio):
            if c == "{":
                profundidad += 1
            elif c == "}":
                profundidad -= 1
                if profundidad == 0:
                    candidato = texto[inicio: i + 1]
                    try:
                        return json.loads(candidato)
                    except json.JSONDecodeError:
                        break

    # Intento 3: reparación mínima
    if inicio != -1:
        fragmento = texto[inicio:]
        fragmento = fragmento.replace("'", '"')
        fragmento = re.sub(r'(\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', fragmento)
        try:
            return json.loads(fragmento)
        except json.JSONDecodeError:
            pass

    return None


def _limpiar_markdown(texto: str) -> str:
    """Elimina bloques markdown (```...```), títulos (#) y espacios extra
    que el modelo agrega aunque se le pida no hacerlo."""
    if not texto:
        return texto
    # Quitar bloques ```...```
    texto = re.sub(r"```[a-zA-Z]*\n?", "", texto)
    texto = re.sub(r"```", "", texto)
    # Quitar líneas que son solo títulos # o ##
    texto = re.sub(r"^\s*#+.*$", "", texto, flags=re.MULTILINE)
    # Quitar líneas de comentario //
    texto = re.sub(r"^\s*//.*$", "", texto, flags=re.MULTILINE)
    return texto.strip()


def _parsear_formato_compacto(texto):
    """Parser robusto: extrae puntos, conexiones y primitivas del formato compacto.

    Acepta tres casos:
      a) Solo primitivas (C/R/E), sin puntos ni líneas.
      b) Puntos + líneas, sin primitivas.
      c) Mezcla de ambos.
    Devuelve None solo si no hay absolutamente nada parseable.
    """
    if not texto:
        return None

    texto = _limpiar_markdown(texto)
    if not texto:
        return None

    # --- Parsear primitivas geométricas (2D heredadas + nuevas 3D) primero ---
    # Se hace antes del bloque de puntos para que el return None de "sin puntos"
    # no bloquee figuras que son puro primitivas.
    primitivas = []

    # C: cx,cy,r[,cz]
    for m in re.finditer(
        r"^C\s*:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?",
        texto, re.MULTILINE,
    ):
        cx, cy, r = float(m.group(1)), float(m.group(2)), float(m.group(3))
        p = {"tipo": "circulo", "cx": cx, "cy": cy, "r": r}
        if m.group(4) is not None:
            p["cz"] = float(m.group(4))
        primitivas.append(p)

    # R: x,y,w,h[,cz]
    for m in re.finditer(
        r"^R\s*:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?",
        texto, re.MULTILINE,
    ):
        x, y, ancho, alto = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        p = {"tipo": "rectangulo", "x": x, "y": y, "ancho": ancho, "alto": alto}
        if m.group(5) is not None:
            p["cz"] = float(m.group(5))
        primitivas.append(p)

    # E: cx,cy,rx,ry[,cz]
    for m in re.finditer(
        r"^E\s*:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?",
        texto, re.MULTILINE,
    ):
        cx, cy, rx, ry = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        p = {"tipo": "elipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry}
        if m.group(5) is not None:
            p["cz"] = float(m.group(5))
        primitivas.append(p)

    # S: cx,cy,cz,r  ← esfera 3D
    for m in re.finditer(
        r"^S\s*:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)",
        texto, re.MULTILINE,
    ):
        primitivas.append({
            "tipo": "esfera",
            "cx": float(m.group(1)), "cy": float(m.group(2)),
            "cz": float(m.group(3)), "r":  float(m.group(4)),
        })

    # K: cx,cy,cz,w,h,d  ← cubo 3D
    for m in re.finditer(
        r"^K\s*:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)",
        texto, re.MULTILINE,
    ):
        primitivas.append({
            "tipo": "cubo",
            "cx": float(m.group(1)), "cy": float(m.group(2)), "cz": float(m.group(3)),
            "ancho": float(m.group(4)), "alto": float(m.group(5)), "profundo": float(m.group(6)),
        })

    # Y: cx,cy,cz,r,h  ← cilindro 3D
    for m in re.finditer(
        r"^Y\s*:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)",
        texto, re.MULTILINE,
    ):
        primitivas.append({
            "tipo": "cilindro",
            "cx": float(m.group(1)), "cy": float(m.group(2)), "cz": float(m.group(3)),
            "r": float(m.group(4)), "alto": float(m.group(5)),
        })

    # --- Parsear puntos: "P0: 0.50,0.22[,0.50]" o "P0: (0.50, 0.22)" ---
    puntos_raw = {}
    for m in re.finditer(
        r"P(\d+)\s*:\s*\(?\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)?", texto
    ):
        idx = int(m.group(1))
        x = min(max(float(m.group(2)), 0.0), 1.0)
        y = min(max(float(m.group(3)), 0.0), 1.0)
        if m.group(4) is not None:
            z = min(max(float(m.group(4)), 0.0), 1.0)
            puntos_raw[idx] = (x, y, z)
        else:
            puntos_raw[idx] = (x, y)

    # --- Fallback: el modelo a veces ignora el formato "P0:, P1:, ..." y en su lugar
    # escribe todos los puntos en una sola línea sin índice, del estilo:
    #   "Px: 0.25,0.45; 0.75,0.45; 0.38,0.69"
    # o incluso "P:" o "Puntos:". En ese caso, se toman los pares x,y en el orden en que
    # aparecen y se les asigna índice secuencial 0,1,2,... según el orden de aparición.
    if not puntos_raw:
        for linea in texto.splitlines():
            m_intro = re.match(r"^P\w*\s*:\s*(.+)$", linea.strip())
            if not m_intro:
                continue
            segmentos = re.split(r"[;]\s*", m_intro.group(1))
            pares_encontrados = []
            for seg in segmentos:
                m_xy = re.match(r"\(?\s*([\d.]+)\s*,\s*([\d.]+)\s*\)?", seg.strip())
                if m_xy:
                    x, y = float(m_xy.group(1)), float(m_xy.group(2))
                    pares_encontrados.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)))
            if pares_encontrados:
                for idx, par in enumerate(pares_encontrados):
                    puntos_raw[idx] = par
                break  # ya encontramos la línea con todos los puntos, no seguir buscando

    # Si no hay puntos NI primitivas → nada parseable
    if not puntos_raw and not primitivas:
        return None

    # Si solo hay primitivas → devolver sin puntos ni conexiones
    if not puntos_raw:
        return {"puntos": [], "conexiones": [], "primitivas": primitivas}

    # --- Reindexar por si hay huecos ---
    indices_ordenados = sorted(puntos_raw.keys())
    puntos_lista = [puntos_raw[i] for i in indices_ordenados]
    reindexado = {viejo: nuevo for nuevo, viejo in enumerate(indices_ordenados)}

    # --- Parsear conexiones ---
    conexiones = []
    placeholders_a = []
    placeholders_b_count = 0

    linea_l = re.search(r"L\s*:\s*(.*)", texto)
    fuente_conexiones = linea_l.group(1) if linea_l else texto

    pairs = re.findall(r"(\d+|\?)\s*-\s*(\d+|\?)", fuente_conexiones)
    for a, b in pairs:
        if a.isdigit() and b.isdigit():
            i, j = int(a), int(b)
            if i in reindexado and j in reindexado:
                conexiones.append([reindexado[i], reindexado[j]])
        elif a.isdigit() and b == "?":
            i = int(a)
            if i in reindexado:
                placeholders_a.append(reindexado[i])
        elif a == "?" and b.isdigit():
            j = int(b)
            if j in reindexado:
                placeholders_a.append(reindexado[j])
        elif a == "?" and b == "?":
            placeholders_b_count += 1

    n = len(puntos_lista)

    # Si no hay conexiones, conectar secuencialmente
    if not conexiones:
        if n >= 3:
            conexiones = [[i, i + 1] for i in range(n - 1)] + [[n - 1, 0]]
        elif n == 2:
            conexiones = [[0, 1]]

    def dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # Resolver placeholders i-?
    for i in placeholders_a:
        vecinos = {v for u, v in conexiones if u == i} | {u for u, v in conexiones if v == i}
        best_j, min_d = -1, float('inf')
        for j in range(n):
            if j != i and j not in vecinos:
                d = dist(puntos_lista[i], puntos_lista[j])
                if d < min_d:
                    min_d, best_j = d, j
        if best_j != -1:
            conexiones.append([i, best_j])

    # Resolver placeholders ?-?
    for _ in range(placeholders_b_count):
        best_pair, min_d = None, float('inf')
        for i in range(n):
            vecinos_i = {v for u, v in conexiones if u == i} | {u for u, v in conexiones if v == i}
            for j in range(i + 1, n):
                if j not in vecinos_i:
                    d = dist(puntos_lista[i], puntos_lista[j])
                    if d < min_d:
                        min_d, best_pair = d, (i, j)
        if best_pair:
            conexiones.append([best_pair[0], best_pair[1]])

    # Garantizar conectividad (unir componentes disconexas)
    if n > 1:
        while True:
            adj = {i: [] for i in range(n)}
            for u, v in conexiones:
                adj[u].append(v)
                adj[v].append(u)
            visitados, componentes = set(), []
            for i in range(n):
                if i not in visitados:
                    comp, pila = [], [i]
                    visitados.add(i)
                    while pila:
                        u = pila.pop()
                        comp.append(u)
                        for v in adj[u]:
                            if v not in visitados:
                                visitados.add(v)
                                pila.append(v)
                    componentes.append(comp)
            if len(componentes) <= 1:
                break
            min_d, best_edge = float('inf'), None
            for idx_a in range(len(componentes)):
                for idx_b in range(idx_a + 1, len(componentes)):
                    for u in componentes[idx_a]:
                        for v in componentes[idx_b]:
                            d = dist(puntos_lista[u], puntos_lista[v])
                            if d < min_d:
                                min_d, best_edge = d, (u, v)
            if best_edge:
                conexiones.append([best_edge[0], best_edge[1]])
            else:
                break

    return {"puntos": list(puntos_lista), "conexiones": conexiones, "primitivas": primitivas}


def _llamar_modelo(messages, num_predict=-1, temperatura=0.2):
    """Hace una sola consulta al modelo."""
    try:
        try:
            respuesta = ollama.chat(
                model=MODELO,
                messages=messages,
                think=False,
                options={
                    "temperature": temperatura,
                    "num_predict": num_predict,
                    "num_ctx": 8192
                },
            )
        except TypeError:
            respuesta = ollama.chat(
                model=MODELO,
                messages=messages,
                options={
                    "temperature": temperatura,
                    "num_predict": num_predict,
                    "num_ctx": 8192
                },
            )
        return respuesta["message"]["content"]
    except Exception as e:
        print(f"[IA] No se pudo llamar al modelo: {e}")
        return None


def _generar_descripcion_fisica(descripcion):
    """Paso 0a: le pide al modelo una descripción física de las partes del objeto,
    en lenguaje llano y sin coordenadas. Esta descripción se usa después como contexto
    para el paso 1 (generador de esquema), para que ya tenga decidido qué partes tiene
    el objeto, de qué forma y cómo se relacionan entre sí, y solo le quede la tarea
    mecánica de pasar eso a números.

    Devuelve el texto de la descripción, o None si el modelo no respondió (en ese caso
    el pipeline sigue funcionando igual, solo que sin este contexto extra).
    """
    texto = _llamar_modelo(
        messages=[
            {"role": "system", "content": _system_descripcion_fisica()},
            {"role": "user", "content": descripcion},
        ],
        num_predict=-1,
        temperatura=0.3,
    )
    if not texto:
        return None
    return _limpiar_markdown(texto)


# ---------------------------------------------------------------------------
# Paso 0b — Dimensionamiento explícito
#
# Por qué hace falta: incluso con una buena descripción física, el modelo local
# tiende a ignorar proporciones relativas a la hora de asignar coordenadas.
# El techo termina más ancho que el cuerpo, la puerta flota fuera del cuerpo, etc.
# Este módulo convierte la descripción textual en bounding-boxes normalizadas
# (x_min, x_max, y_min, y_max en [0,1]) para cada parte, y las inyecta en el
# prompt del paso 1 como restricciones concretas. El LLM solo tiene que "dibujar
# dentro" de los límites dados, sin inventar proporciones.
# ---------------------------------------------------------------------------

# Reglas de dimensionamiento para las partes más comunes.
# Cada entrada es un dict con:
#   "rol"     : palabra clave que aparece en la descripción física de esa parte
#   "calc"    : función (dims_ya_calculadas) -> bbox dict {x_min,x_max,y_min,y_max}
#               recibe el dict de bboxes de las partes calculadas HASTA ESE MOMENTO
#               para que las partes dependientes (techo, puerta) puedan anclarse
#               al cuerpo principal ya calculado.
# Las partes se procesan en orden de lista, así el cuerpo siempre se calcula primero.

_MARGEN   = 0.05
_CX       = 0.50     # centro horizontal global
_CUERPO_W = 0.50     # ancho del cuerpo principal (normalizado)
_CUERPO_H = 0.38     # alto del cuerpo principal
_CUERPO_Y = 0.42     # y_min del cuerpo (empuja el objeto hacia la mitad del panel)

_REGLAS_DIMENSIONAMIENTO = [
    # ── Cuerpo / carrocería / torso: parte estructural central ──────────────
    {
        "roles": ["cuerpo", "carroceria", "torso", "base", "casco", "fuselaje"],
        "calc": lambda d: {
            "x_min": _CX - _CUERPO_W / 2,
            "x_max": _CX + _CUERPO_W / 2,
            "y_min": _CUERPO_Y,
            "y_max": _CUERPO_Y + _CUERPO_H,
        },
    },
    # ── Techo: mismo ancho X que el cuerpo, encima ──────────────────────────
    {
        "roles": ["techo", "copa"],
        "calc": lambda d: {
            "x_min": d["cuerpo"]["x_min"] if "cuerpo" in d else _CX - _CUERPO_W / 2,
            "x_max": d["cuerpo"]["x_max"] if "cuerpo" in d else _CX + _CUERPO_W / 2,
            "y_min": (d["cuerpo"]["y_min"] if "cuerpo" in d else _CUERPO_Y) - 0.22,
            "y_max": d["cuerpo"]["y_min"] if "cuerpo" in d else _CUERPO_Y,
        },
    },
    # ── Cabeza: arriba del cuerpo/torso, centrada ───────────────────────────
    {
        "roles": ["cabeza"],
        "calc": lambda d: {
            "x_min": _CX - 0.14,
            "x_max": _CX + 0.14,
            "y_min": (d["cuerpo"]["y_min"] if "cuerpo" in d else _CUERPO_Y) - 0.20,
            "y_max": (d["cuerpo"]["y_min"] if "cuerpo" in d else _CUERPO_Y) - 0.02,
        },
    },
    # ── Puerta: centro del cuerpo, base tocando el fondo ────────────────────
    {
        "roles": ["puerta", "entrada"],
        "calc": lambda d: {
            "x_min": _CX - 0.06,
            "x_max": _CX + 0.06,
            "y_min": (d["cuerpo"]["y_max"] if "cuerpo" in d else _CUERPO_Y + _CUERPO_H) - 0.18,
            "y_max": d["cuerpo"]["y_max"] if "cuerpo" in d else _CUERPO_Y + _CUERPO_H,
        },
    },
    # ── Ventana: mitad derecha del cuerpo, cuarto superior ──────────────────
    {
        "roles": ["ventana", "ojo", "ojos"],
        "calc": lambda d: {
            "x_min": _CX + 0.05,
            "x_max": _CX + 0.18,
            "y_min": (d["cuerpo"]["y_min"] if "cuerpo" in d else _CUERPO_Y) + 0.05,
            "y_max": (d["cuerpo"]["y_min"] if "cuerpo" in d else _CUERPO_Y) + 0.16,
        },
    },
    # ── Tronco: parte inferior de árboles/plantas ───────────────────────────
    {
        "roles": ["tronco", "pie", "pierna", "patas"],
        "calc": lambda d: {
            "x_min": _CX - 0.06,
            "x_max": _CX + 0.06,
            "y_min": d["cuerpo"]["y_max"] if "cuerpo" in d else _CUERPO_Y + _CUERPO_H,
            "y_max": (d["cuerpo"]["y_max"] if "cuerpo" in d else _CUERPO_Y + _CUERPO_H) + 0.18,
        },
    },
    # ── Rueda / neumático: debajo del cuerpo, a los costados ─────────────────
    {
        "roles": ["rueda", "neumatico", "llanta"],
        "calc": lambda d: {
            "x_min": _MARGEN,
            "x_max": 1.0 - _MARGEN,
            "y_min": d["cuerpo"]["y_max"] if "cuerpo" in d else _CUERPO_Y + _CUERPO_H,
            "y_max": (d["cuerpo"]["y_max"] if "cuerpo" in d else _CUERPO_Y + _CUERPO_H) + 0.12,
        },
    },
]


def _rol_de_parte(nombre_parte: str) -> str | None:
    """Mapea el nombre de una parte (como aparece en la descripción física)
    al nombre de rol canónico usado como clave en el dict de bboxes.
    Devuelve el primer rol que matchee o None si no hay coincidencia."""
    nombre = nombre_parte.lower().strip()
    for regla in _REGLAS_DIMENSIONAMIENTO:
        for palabra_clave in regla["roles"]:
            if palabra_clave in nombre:
                return regla["roles"][0]   # el primer rol es el canónico
    return None


def _calcular_dimensiones(descripcion_fisica: str) -> dict:
    """Paso 0b: parsea la descripción física línea por línea y asigna bounding-boxes
    a cada parte reconocida.

    Devuelve un dict  { nombre_rol: {x_min, x_max, y_min, y_max} }
    con todas las partes para las que existe una regla de dimensionamiento.
    Las partes desconocidas se ignoran (el LLM las ubica libremente).
    """
    dims: dict = {}
    if not descripcion_fisica:
        return dims

    for linea in descripcion_fisica.splitlines():
        # Las líneas de descripción física tienen formato  "- NombreParte: ..."
        # Nos interesa solo el nombre antes de los dos puntos.
        linea = linea.strip().lstrip("-").strip()
        if not linea or ":" not in linea:
            continue
        nombre_parte = linea.split(":")[0].strip()
        rol = _rol_de_parte(nombre_parte)
        if rol and rol not in dims:
            # Buscar la regla correspondiente y calcular el bbox
            for regla in _REGLAS_DIMENSIONAMIENTO:
                if rol in regla["roles"]:
                    try:
                        dims[rol] = regla["calc"](dims)
                    except Exception:
                        pass   # si falla (ej. cuerpo no calculado aún), se ignora
                    break

    return dims


def _dims_a_texto(dims: dict) -> str:
    """Serializa las bounding-boxes para inyectar en el prompt del paso 1."""
    if not dims:
        return ""
    lineas = ["[DIMENSIONES ASIGNADAS — respetarlas estrictamente]"]
    for nombre, bb in dims.items():
        lineas.append(
            f"  {nombre}: "
            f"x=[{bb['x_min']:.2f}, {bb['x_max']:.2f}]  "
            f"y=[{bb['y_min']:.2f}, {bb['y_max']:.2f}]"
        )
    lineas.append(
        "Cada parte debe quedar COMPLETAMENTE dentro de su rango x e y asignado, sin salirse "
        "ni un decimal. Si el campo \"contacto\" de una parte dice \"toca:abajo=arriba:X\", el "
        "y_max de esa parte tiene que ser exactamente igual (mismos decimales, sin redondear) "
        "al y_min de X — no un valor parecido. Lo mismo para \"izquierda\"/\"derecha\" con x_min/"
        "x_max. Estos bounding-boxes ya vienen calculados con precisión; tu trabajo es dibujar "
        "DENTRO de ellos y COPIAR el número de contacto tal cual, nunca inventar uno nuevo."
    )
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Ensamblado determinístico por catálogo — el corazón del fix del bug del cubo.
#
# En vez de pedirle al LLM que "razone" coordenadas para partes cuya forma ya
# está resuelta en el catálogo (cubo, esfera, cilindro, etc.), esta función:
#   1) Lee de la descripción física qué forma_base eligió el modelo para cada parte.
#   2) Si esa forma_base está en el catálogo Y la parte tiene un bbox asignado
#      (paso 0b), construye la primitiva o el conjunto de puntos/líneas
#      DIRECTAMENTE en Python, escalando la plantilla unitaria al tamaño del bbox.
#   3) Si TODAS las partes se pudieron resolver así, la figura queda 100% armada
#      sin haber llamado al LLM para geometría ni una sola vez — cero riesgo de
#      que invente un cubo mal cerrado.
#   4) Si alguna parte no está en el catálogo (forma rara, objeto no contemplado)
#      se devuelve None y el pipeline sigue con el LLM como antes (paso 1),
#      que igual ahora tiene el catálogo mencionado en su propio prompt.
# ---------------------------------------------------------------------------

_PATRON_PARTE_FISICA = re.compile(
    r"^-?\s*([^:]+):\s*forma_base\s*=\s*([a-záéíóúñ_]+)", re.IGNORECASE
)


def _extraer_partes_fisicas(descripcion_fisica: str) -> list[dict]:
    """Parsea las líneas del paso 0a con el formato
    '- NombreParte: forma_base=X | volumen=... | ...' y devuelve
    [{"nombre": "Cuerpo", "forma_base": "cubo"}, ...]. Ignora líneas que no
    matcheen ese formato (ej. si el modelo no siguió el formato pedido)."""
    partes = []
    if not descripcion_fisica:
        return partes
    for linea in descripcion_fisica.splitlines():
        m = _PATRON_PARTE_FISICA.match(linea.strip())
        if m:
            partes.append({
                "nombre": m.group(1).strip(),
                "forma_base": m.group(2).strip(),
            })
    return partes


def _construir_pieza_catalogo(defin: dict, categoria: str, bbox: dict) -> dict:
    """Escala la plantilla unitaria de una forma base al tamaño y posición del
    bbox asignado, y arma la primitiva o el bloque de puntos+líneas en el mismo
    formato que produce el parser del LLM (ver `_parsear_formato_compacto`).

    Devuelve un dict con al menos una de estas dos formas:
        {"primitiva": {...}}                          — para C/R/E/S/Y/K
        {"puntos": [...], "conexiones": [...]}         — para PUNTOS2D/PUNTOS3D
    """
    cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
    cy = (bbox["y_min"] + bbox["y_max"]) / 2.0
    ancho = bbox["x_max"] - bbox["x_min"]
    alto = bbox["y_max"] - bbox["y_min"]
    # Sin info de profundidad explícita: se usa el promedio de ancho/alto,
    # que da un volumen proporcionado en vez de aplanado o estirado al azar.
    profundo = (ancho + alto) / 2.0
    cz = 0.5

    primitiva_tipo = defin.get("primitiva")

    if primitiva_tipo == "K":  # cubo / caja
        if defin.get("cara_cuadrada"):
            lado = min(ancho, alto)
            ancho = alto = lado
        return {"primitiva": {
            "tipo": "cubo", "cx": cx, "cy": cy, "cz": cz,
            "ancho": ancho, "alto": alto, "profundo": profundo,
        }}

    if primitiva_tipo == "S":  # esfera
        r = min(ancho, alto) / 2.0
        return {"primitiva": {"tipo": "esfera", "cx": cx, "cy": cy, "cz": cz, "r": r}}

    if primitiva_tipo == "Y":  # cilindro
        r = min(ancho, alto) / 2.0 if ancho < alto else ancho / 2.0
        return {"primitiva": {"tipo": "cilindro", "cx": cx, "cy": cy, "cz": cz, "r": r, "alto": alto}}

    if primitiva_tipo == "C":  # círculo
        r = min(ancho, alto) / 2.0
        return {"primitiva": {"tipo": "circulo", "cx": cx, "cy": cy, "r": r, "cz": cz}}

    if primitiva_tipo == "E":  # elipse
        return {"primitiva": {"tipo": "elipse", "cx": cx, "cy": cy, "rx": ancho/2.0, "ry": alto/2.0, "cz": cz}}

    if primitiva_tipo == "R":  # rectángulo / cuadrado
        if defin.get("cara_cuadrada"):
            lado = min(ancho, alto)
            ancho = alto = lado
        return {"primitiva": {
            "tipo": "rectangulo", "x": cx - ancho/2.0, "y": cy - alto/2.0,
            "ancho": ancho, "alto": alto, "cz": cz,
        }}

    if primitiva_tipo in ("PUNTOS2D", "PUNTOS3D"):
        plantilla = defin.get("plantilla_puntos", [])
        conexiones = [list(c) for c in defin.get("plantilla_conexiones", [])]
        puntos = []
        for p in plantilla:
            px = cx + p[0] * ancho
            py = cy + p[1] * alto
            if primitiva_tipo == "PUNTOS3D" and len(p) > 2:
                pz = cz + p[2] * profundo
                puntos.append((
                    min(max(px, 0.0), 1.0),
                    min(max(py, 0.0), 1.0),
                    min(max(pz, 0.0), 1.0),
                ))
            else:
                puntos.append((min(max(px, 0.0), 1.0), min(max(py, 0.0), 1.0)))
        return {"puntos": puntos, "conexiones": conexiones}

    return {}


def _ensamblar_figura_por_catalogo(descripcion_fisica: str, dims: dict) -> dict | None:
    """Intenta armar la figura COMPLETA usando solo el catálogo (sin LLM para
    geometría). Devuelve None si alguna parte no se pudo resolver (forma_base
    desconocida o sin bbox asignado), para que el pipeline caiga al flujo con
    LLM como red de seguridad."""
    partes = _extraer_partes_fisicas(descripcion_fisica)
    if not partes:
        return None

    puntos_totales: list = []
    conexiones_totales: list = []
    primitivas_totales: list = []

    for parte in partes:
        rol = _rol_de_parte(parte["nombre"])
        bbox = dims.get(rol) if rol else None
        if bbox is None:
            # Parte sin bbox asignado (rol desconocido para el paso 0b):
            # no hay forma confiable de posicionarla en código, así que se
            # aborta el ensamblado por catálogo y se cae al LLM para TODA
            # la figura (más seguro que mezclar piezas ya ubicadas con
            # piezas "libres").
            return None

        resuelto = _resolver_forma_base(parte["forma_base"])
        if resuelto is None:
            return None
        _clave, categoria, defin = resuelto

        pieza = _construir_pieza_catalogo(defin, categoria, bbox)
        if not pieza:
            return None

        if "primitiva" in pieza:
            primitivas_totales.append(pieza["primitiva"])
        elif "puntos" in pieza:
            offset = len(puntos_totales)
            puntos_totales.extend(pieza["puntos"])
            for i, j in pieza["conexiones"]:
                conexiones_totales.append([i + offset, j + offset])
        else:
            return None

    if not puntos_totales and not primitivas_totales:
        return None

    return {
        "puntos": puntos_totales,
        "conexiones": conexiones_totales,
        "primitivas": primitivas_totales,
    }


# ---------------------------------------------------------------------------
# Paso 2b — Validación de bboxes
#
# Después de que el modelo genera los puntos y primitivas, esta función verifica
# que cada elemento quede dentro del bounding-box asignado para su parte.
# No rechaza la figura completa si hay una violación menor (el modelo local es
# impreciso), sino que:
#   1) Reporta qué partes están fuera de rango.
#   2) Intenta corregir automáticamente clampando las coordenadas al bbox.
#   3) Si la violación es grave (> TOLERANCIA_BBOX fuera del rango), marca la
#      figura para reintentar el paso 1 con un prompt más estricto.
# ---------------------------------------------------------------------------

TOLERANCIA_BBOX = 0.08   # tolerancia máxima permitida antes de considerar la figura inválida


def _asignar_bbox_a_primitiva(prim: dict, dims: dict) -> str | None:
    """Intenta mapear una primitiva a su rol de parte para poder validar su bbox.
    Devuelve el rol canónico o None si no se puede determinar."""
    # Heurística: la primitiva se asocia al rol cuyo bbox la contiene mejor.
    tipo = prim.get("tipo")
    if tipo == "rectangulo":
        cx = prim.get("x", 0) + prim.get("ancho", 0) / 2
        cy = prim.get("y", 0) + prim.get("alto", 0) / 2
    elif tipo == "circulo":
        cx, cy = prim.get("cx", 0.5), prim.get("cy", 0.5)
    elif tipo == "elipse":
        cx, cy = prim.get("cx", 0.5), prim.get("cy", 0.5)
    else:
        return None

    mejor_rol, menor_dist = None, float("inf")
    for rol, bb in dims.items():
        centro_x = (bb["x_min"] + bb["x_max"]) / 2
        centro_y = (bb["y_min"] + bb["y_max"]) / 2
        dist = math.hypot(cx - centro_x, cy - centro_y)
        if dist < menor_dist:
            menor_dist, mejor_rol = dist, rol
    return mejor_rol


def _punto_en_bbox(x: float, y: float, bb: dict, tol: float = 0.0) -> bool:
    return (bb["x_min"] - tol <= x <= bb["x_max"] + tol and
            bb["y_min"] - tol <= y <= bb["y_max"] + tol)


def _clampar_a_bbox(val: float, lo: float, hi: float) -> float:
    return min(max(val, lo), hi)


def validar_y_corregir_bboxes(figura: dict, dims: dict) -> tuple[dict, bool, list[str]]:
    """Paso 2b: valida y corrige en lo posible las coordenadas de la figura
    contra los bounding-boxes asignados en el paso 0b.

    Parámetros
    ----------
    figura : dict con 'puntos', 'conexiones', 'primitivas'
    dims   : dict de bboxes { rol: {x_min,x_max,y_min,y_max} } del paso 0b

    Devuelve
    --------
    figura_corregida : dict igual que `figura` pero con coordenadas clampadas
    valida           : True si no hubo violaciones graves (> TOLERANCIA_BBOX)
    advertencias     : lista de strings describiendo qué se corrigió o qué viola
    """
    if not dims:
        # Sin dimensiones asignadas, no hay nada que validar
        return figura, True, []

    advertencias: list[str] = []
    violacion_grave = False

    puntos_corregidos = list(figura.get("puntos", []))
    primitivas_corregidas = list(figura.get("primitivas", []))

    # ── Validar puntos: se asignan al bbox "cuerpo" por defecto si no hay otro ──
    # (los puntos individuales no tienen nombre de parte, así que solo verificamos
    #  que no se salgan del espacio total esperado [0,1] x [0,1] con tolerancia)
    for idx, (x, y) in enumerate(puntos_corregidos):
        fuera_x = max(0.0, -x, x - 1.0)
        fuera_y = max(0.0, -y, y - 1.0)
        if fuera_x > TOLERANCIA_BBOX or fuera_y > TOLERANCIA_BBOX:
            violacion_grave = True
            advertencias.append(
                f"  P{idx} ({x:.3f},{y:.3f}) fuera del panel [0,1] — violación grave"
            )
        if fuera_x > 0 or fuera_y > 0:
            xc = _clampar_a_bbox(x, 0.0, 1.0)
            yc = _clampar_a_bbox(y, 0.0, 1.0)
            puntos_corregidos[idx] = (xc, yc)
            advertencias.append(
                f"  P{idx} ({x:.3f},{y:.3f}) clampado a ({xc:.3f},{yc:.3f})"
            )

    # ── Validar primitivas contra su bbox de rol ─────────────────────────────
    for idx, prim in enumerate(primitivas_corregidas):
        rol = _asignar_bbox_a_primitiva(prim, dims)
        if rol is None or rol not in dims:
            continue
        bb = dims[rol]
        tipo = prim.get("tipo")
        prim_mod = dict(prim)

        if tipo == "circulo":
            cx, cy, r = prim["cx"], prim["cy"], prim["r"]
            # El centro debe caer dentro del bbox (con tolerancia)
            if not _punto_en_bbox(cx, cy, bb, TOLERANCIA_BBOX):
                violacion_grave = True
                advertencias.append(
                    f"  C{idx} centro ({cx:.3f},{cy:.3f}) fuera del bbox '{rol}' "
                    f"x=[{bb['x_min']:.2f},{bb['x_max']:.2f}] "
                    f"y=[{bb['y_min']:.2f},{bb['y_max']:.2f}] — violación grave"
                )
            else:
                # Clampar el centro si se fue un poco
                nuevo_cx = _clampar_a_bbox(cx, bb["x_min"], bb["x_max"])
                nuevo_cy = _clampar_a_bbox(cy, bb["y_min"], bb["y_max"])
                # El radio no puede hacer que el círculo se salga demasiado del bbox
                r_max_x = min(nuevo_cx - bb["x_min"], bb["x_max"] - nuevo_cx) + TOLERANCIA_BBOX
                r_max_y = min(nuevo_cy - bb["y_min"], bb["y_max"] - nuevo_cy) + TOLERANCIA_BBOX
                r_max   = min(r_max_x, r_max_y)
                if r > r_max + TOLERANCIA_BBOX:
                    advertencias.append(
                        f"  C{idx} radio {r:.3f} excede bbox '{rol}' (max≈{r_max:.3f}), "
                        f"clampado"
                    )
                    prim_mod["r"] = max(0.01, r_max)
                if nuevo_cx != cx or nuevo_cy != cy:
                    advertencias.append(
                        f"  C{idx} centro clampado ({cx:.3f},{cy:.3f})→({nuevo_cx:.3f},{nuevo_cy:.3f})"
                    )
                    prim_mod["cx"] = nuevo_cx
                    prim_mod["cy"] = nuevo_cy

        elif tipo == "rectangulo":
            rx, ry = prim["x"], prim["y"]
            rw, rh = prim["ancho"], prim["alto"]
            cx_r, cy_r = rx + rw / 2, ry + rh / 2
            if not _punto_en_bbox(cx_r, cy_r, bb, TOLERANCIA_BBOX):
                violacion_grave = True
                advertencias.append(
                    f"  R{idx} centro ({cx_r:.3f},{cy_r:.3f}) fuera del bbox '{rol}' — "
                    f"violación grave"
                )
            else:
                nuevo_x  = _clampar_a_bbox(rx, bb["x_min"], bb["x_max"] - rw)
                nuevo_y  = _clampar_a_bbox(ry, bb["y_min"], bb["y_max"] - rh)
                if nuevo_x != rx or nuevo_y != ry:
                    advertencias.append(
                        f"  R{idx} origen clampado ({rx:.3f},{ry:.3f})→({nuevo_x:.3f},{nuevo_y:.3f})"
                    )
                    prim_mod["x"] = nuevo_x
                    prim_mod["y"] = nuevo_y
                # Ancho/alto no pueden exceder el bbox
                ancho_max = bb["x_max"] - prim_mod["x"]
                alto_max  = bb["y_max"] - prim_mod["y"]
                if rw > ancho_max + TOLERANCIA_BBOX:
                    advertencias.append(f"  R{idx} ancho {rw:.3f} clampado a {ancho_max:.3f}")
                    prim_mod["ancho"] = max(0.01, ancho_max)
                if rh > alto_max + TOLERANCIA_BBOX:
                    advertencias.append(f"  R{idx} alto {rh:.3f} clampado a {alto_max:.3f}")
                    prim_mod["alto"] = max(0.01, alto_max)

        elif tipo == "elipse":
            cx, cy = prim["cx"], prim["cy"]
            if not _punto_en_bbox(cx, cy, bb, TOLERANCIA_BBOX):
                violacion_grave = True
                advertencias.append(
                    f"  E{idx} centro ({cx:.3f},{cy:.3f}) fuera del bbox '{rol}' — "
                    f"violación grave"
                )
            else:
                nuevo_cx = _clampar_a_bbox(cx, bb["x_min"], bb["x_max"])
                nuevo_cy = _clampar_a_bbox(cy, bb["y_min"], bb["y_max"])
                if nuevo_cx != cx or nuevo_cy != cy:
                    advertencias.append(
                        f"  E{idx} centro clampado ({cx:.3f},{cy:.3f})→({nuevo_cx:.3f},{nuevo_cy:.3f})"
                    )
                    prim_mod["cx"] = nuevo_cx
                    prim_mod["cy"] = nuevo_cy

        primitivas_corregidas[idx] = prim_mod

    figura_corregida = {
        "puntos":     puntos_corregidos,
        "conexiones": figura.get("conexiones", []),
        "primitivas": primitivas_corregidas,
    }
    return figura_corregida, not violacion_grave, advertencias


TEMPLATES = {
    "cubo": {
        "puntos": [],
        "conexiones": [],
        "primitivas": [
            {"tipo": "cubo", "cx": 0.50, "cy": 0.50, "cz": 0.5,
             "ancho": 0.35, "alto": 0.35, "profundo": 0.35}
        ],
    },
    "caja": "cubo",
    "dado": "cubo",
    "bloque": "cubo",
    "casa": {
        "puntos": [
            [0.50, 0.20], [0.25, 0.45], [0.75, 0.45], [0.25, 0.80], [0.75, 0.80]
        ],
        "conexiones": [
            [0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4]
        ]
    },
    "hogar": "casa",
    "vivienda": "casa",
    "auto": {
        "puntos": [
            # Carrocería (8 puntos): perfil tipo hatchback, contorno cerrado
            [0.12, 0.65], [0.12, 0.52], [0.22, 0.40], [0.38, 0.30], [0.62, 0.30],
            [0.78, 0.40], [0.88, 0.52], [0.88, 0.65],
            # Rueda izquierda (8 puntos, octágono cerrado en vez de una muesca cuadrada)
            [0.36, 0.68], [0.34, 0.74], [0.28, 0.76], [0.22, 0.74],
            [0.20, 0.68], [0.22, 0.62], [0.28, 0.60], [0.34, 0.62],
            # Rueda derecha (8 puntos, mismo octágono, espejado)
            [0.80, 0.68], [0.78, 0.74], [0.72, 0.76], [0.66, 0.74],
            [0.64, 0.68], [0.66, 0.62], [0.72, 0.60], [0.78, 0.62],
        ],
        "conexiones": [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0],
            [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 8],
            [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 16]
        ]
    },
    "carro": "auto",
    "coche": "auto",
    "vehiculo": "auto",
    "avion": {
        "puntos": [
            # Fuselaje (6 puntos), contorno cerrado
            [0.10, 0.55], [0.30, 0.45], [0.75, 0.45], [0.85, 0.52], [0.70, 0.60], [0.25, 0.60],
            # Ala (4 puntos), contorno cerrado, separado del fuselaje
            [0.40, 0.55], [0.55, 0.55], [0.58, 0.80], [0.38, 0.80],
            # Aleta de cola (3 puntos), contorno cerrado, separado del fuselaje
            [0.72, 0.45], [0.83, 0.45], [0.80, 0.25],
        ],
        "conexiones": [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],
            [6, 7], [7, 8], [8, 9], [9, 6],
            [10, 11], [11, 12], [12, 10]
        ]
    },
    "aeroplano": "avion",
    "persona": {
        "puntos": [
            [0.50, 0.15], [0.45, 0.25], [0.55, 0.25], [0.50, 0.35], [0.30, 0.40],
            [0.70, 0.40], [0.50, 0.60], [0.35, 0.80], [0.65, 0.80]
        ],
        "conexiones": [
            [0, 1], [1, 3], [3, 2], [2, 0], [3, 4], [3, 5], [3, 6], [6, 7], [6, 8]
        ]
    },
    "monigote": "persona",
    "humano": "persona",
    "arbol": {
        "puntos": [
            # Copa (6 puntos), contorno cerrado, más redondeada que un triángulo
            [0.50, 0.18], [0.68, 0.30], [0.72, 0.48], [0.50, 0.55], [0.28, 0.48], [0.32, 0.30],
            # Tronco (4 puntos), contorno cerrado (antes le faltaba el lado de arriba)
            [0.45, 0.55], [0.55, 0.55], [0.55, 0.85], [0.45, 0.85],
        ],
        "conexiones": [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],
            [6, 7], [7, 8], [8, 9], [9, 6]
        ]
    },
    "planta": "arbol",
    "estrella": {
        "puntos": [
            [0.50, 0.15], [0.60, 0.40], [0.85, 0.40], [0.65, 0.55], [0.75, 0.85],
            [0.50, 0.70], [0.25, 0.85], [0.35, 0.55], [0.15, 0.40], [0.40, 0.40]
        ],
        "conexiones": [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]
        ]
    },
    "cohete": {
        "puntos": [
            [0.50, 0.15], [0.35, 0.45], [0.65, 0.45], [0.35, 0.75], [0.65, 0.75],
            [0.25, 0.85], [0.75, 0.85]
        ],
        "conexiones": [
            [0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4], [3, 5], [5, 6], [6, 4]
        ]
    },
    "nave": "cohete"
}

def _buscar_plantilla(descripcion):
    desc_normalizada = descripcion.lower().strip()
    # Eliminar acentos para mayor tolerancia
    desc_normalizada = "".join(
        c for c in unicodedata.normalize("NFD", desc_normalizada)
        if unicodedata.category(c) != "Mn"
    )
    
    # Buscar coincidencia de palabra exacta
    for key, value in TEMPLATES.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, desc_normalizada):
            if isinstance(value, str):
                return TEMPLATES[value]
            return value
    return None


# ---------------------------------------------------------------------------
# 1) Crear figuras — dos pasos: formato compacto -> JSON
# ---------------------------------------------------------------------------

def _validar_figura(datos) -> bool:
    """Verifica estructura mínima: debe tener al menos líneas o primitivas."""
    if not datos:
        return False
    puntos = datos.get("puntos", [])
    conexiones = datos.get("conexiones", [])
    primitivas = datos.get("primitivas", [])

    # Tiene que haber algo que dibujar
    tiene_lineas = len(puntos) >= 2 and len(conexiones) >= 1
    tiene_primitivas = len(primitivas) >= 1
    if not (tiene_lineas or tiene_primitivas):
        return False

    # Validar índices de conexiones
    n = len(puntos)
    for c in conexiones:
        if len(c) != 2 or not (0 <= c[0] < n and 0 <= c[1] < n):
            return False
    return True


# (test de orientación / cruce de segmentos: ver geo_utils.py, importado arriba
# como _segmentos_se_cruzan -- antes había una implementación propia acá, más
# simple y sin tolerancia numérica; se unificó con la de geometria.py)


def _eliminar_diagonales_espurias(puntos, conexiones):
    """Red de seguridad geométrica: el modelo local, al "cerrar" el contorno de una figura,
    a veces conecta mal el último punto con el primero y termina dibujando una línea que
    cruza la figura de punta a punta en vez de unir vecinos reales (esto es lo que generaba
    esas diagonales largas atravesando casas y otras figuras). Esta función detecta
    conexiones que son mucho más largas que el resto Y que además cruzan a otra conexión
    de la misma figura, y las elimina -- pero solo si al sacarlas ningún punto se queda
    totalmente suelto (sin ninguna otra línea que lo sostenga).
    """
    if len(conexiones) < 4:
        return conexiones  # muy pocas conexiones: no hay margen seguro para podar nada

    def _largo(c):
        i, j = c
        return math.hypot(puntos[i][0] - puntos[j][0], puntos[i][1] - puntos[j][1])

    largos = sorted(_largo(c) for c in conexiones)
    mediana = largos[len(largos) // 2]
    if mediana == 0:
        return conexiones

    grados = {}
    for i, j in conexiones:
        grados[i] = grados.get(i, 0) + 1
        grados[j] = grados.get(j, 0) + 1

    indices_a_quitar = set()
    for idx_a, ca in enumerate(conexiones):
        if _largo(ca) <= mediana * 1.3:
            continue  # longitud normal, no es sospechosa
        i, j = ca
        if grados[i] <= 1 or grados[j] <= 1:
            continue  # quitarla dejaría un punto sin ninguna línea: mejor no tocarla
        for idx_b, cb in enumerate(conexiones):
            if idx_a == idx_b or set(ca) & set(cb):
                continue  # comparten un punto: no es un cruce real, es solo que se tocan
            if _segmentos_se_cruzan(puntos[ca[0]], puntos[ca[1]], puntos[cb[0]], puntos[cb[1]]):
                indices_a_quitar.add(idx_a)
                break

    if not indices_a_quitar:
        return conexiones
    return [c for idx, c in enumerate(conexiones) if idx not in indices_a_quitar]


def _normalizar_figura(datos) -> dict:
    """Clampea coordenadas a [0,1], filtra conexiones inválidas y normaliza primitivas.
    Acepta puntos 2D (x,y) o 3D (x,y,z); si falta z se preserva la ausencia."""
    puntos_raw = datos.get("puntos", [])
    puntos = []
    for p in puntos_raw:
        x = min(max(float(p[0]), 0.0), 1.0)
        y = min(max(float(p[1]), 0.0), 1.0)
        if len(p) > 2:
            z = min(max(float(p[2]), 0.0), 1.0)
            puntos.append((x, y, z))
        else:
            puntos.append((x, y))

    n = len(puntos)
    conexiones = [(int(i), int(j)) for i, j in datos.get("conexiones", [])
                  if 0 <= int(i) < n and 0 <= int(j) < n]
    # Eliminar diagonales espurias usa solo x,y (los primeros dos valores)
    puntos_2d = [(p[0], p[1]) for p in puntos]
    conexiones = _eliminar_diagonales_espurias(puntos_2d, conexiones)

    primitivas = []
    for prim in datos.get("primitivas", []):
        tipo = prim.get("tipo")
        try:
            base = {}
            if "cz" in prim:
                base["cz"] = min(max(float(prim["cz"]), 0.0), 1.0)

            if tipo == "circulo":
                primitivas.append({
                    "tipo": "circulo",
                    "cx": min(max(float(prim["cx"]), 0.0), 1.0),
                    "cy": min(max(float(prim["cy"]), 0.0), 1.0),
                    "r":  min(max(float(prim["r"]),  0.0), 1.0),
                    **base,
                })
            elif tipo == "rectangulo":
                primitivas.append({
                    "tipo":  "rectangulo",
                    "x":     min(max(float(prim["x"]),     0.0), 1.0),
                    "y":     min(max(float(prim["y"]),     0.0), 1.0),
                    "ancho": min(max(float(prim["ancho"]), 0.0), 1.0),
                    "alto":  min(max(float(prim["alto"]),  0.0), 1.0),
                    **base,
                })
            elif tipo == "elipse":
                primitivas.append({
                    "tipo": "elipse",
                    "cx": min(max(float(prim["cx"]), 0.0), 1.0),
                    "cy": min(max(float(prim["cy"]), 0.0), 1.0),
                    "rx": min(max(float(prim["rx"]), 0.0), 1.0),
                    "ry": min(max(float(prim["ry"]), 0.0), 1.0),
                    **base,
                })
            elif tipo == "esfera":
                primitivas.append({
                    "tipo": "esfera",
                    "cx": min(max(float(prim["cx"]), 0.0), 1.0),
                    "cy": min(max(float(prim["cy"]), 0.0), 1.0),
                    "cz": min(max(float(prim.get("cz", 0.5)), 0.0), 1.0),
                    "r":  min(max(float(prim["r"]),  0.0), 1.0),
                })
            elif tipo == "cubo":
                primitivas.append({
                    "tipo":    "cubo",
                    "cx":      min(max(float(prim["cx"]),      0.0), 1.0),
                    "cy":      min(max(float(prim["cy"]),      0.0), 1.0),
                    "cz":      min(max(float(prim.get("cz", 0.5)), 0.0), 1.0),
                    "ancho":   min(max(float(prim["ancho"]),   0.0), 1.0),
                    "alto":    min(max(float(prim["alto"]),    0.0), 1.0),
                    "profundo":min(max(float(prim.get("profundo", prim["ancho"])), 0.0), 1.0),
                })
            elif tipo == "cilindro":
                primitivas.append({
                    "tipo": "cilindro",
                    "cx":   min(max(float(prim["cx"]),   0.0), 1.0),
                    "cy":   min(max(float(prim["cy"]),   0.0), 1.0),
                    "cz":   min(max(float(prim.get("cz", 0.5)), 0.0), 1.0),
                    "r":    min(max(float(prim["r"]),    0.0), 1.0),
                    "alto": min(max(float(prim["alto"]), 0.0), 1.0),
                })
        except (KeyError, ValueError, TypeError):
            pass  # primitiva malformada, se descarta

    return {"puntos": puntos, "conexiones": conexiones, "primitivas": primitivas}


def generar_figura(descripcion):
    """Pipeline para generar una figura a partir de su descripción.

    Paso 0a — DESCRIPTOR:    convierte la descripción a una lista de partes físicas
                              (forma, tamaño relativo, ubicación) SIN coordenadas.
    Paso 0b — DIMENSIONADOR: calcula bounding-boxes normalizadas para cada parte
                              reconocida y las inyecta en el prompt del paso 1.
                              Esto le da al modelo restricciones concretas de posición
                              y tamaño, en vez de dejarle inventar proporciones libres.
    Paso 1  — GENERADOR:     produce el esquema compacto (P/L/C/R/E) con las
                              dimensiones del paso 0b como restricción explícita.
    Paso 2a — CORRECTOR:     si el paso 1 generó puntos sueltos con primitivas
                              (señal de confusión), pide un segundo intento enfocado.
    Paso 2b — VALIDADOR BBOX: verifica que cada elemento respete su bounding-box.
                              Clampa automáticamente las violaciones menores; si hay
                              violaciones graves reintenta el paso 1 con un prompt
                              más estricto que incluye las dimensiones obligatorias.

    Caché: si la descripción ya existe en disco, se carga directamente.
    Red de seguridad: si todo falla, se usa la plantilla predefinida (si existe).

    Devuelve {"puntos": [...], "conexiones": [...], "primitivas": [...]} o None.
    """

    # ── Caché ──────────────────────────────────────────────────────────────
    datos_cache = _cargar_cache(descripcion)
    if datos_cache:
        return datos_cache

    plantilla_seguridad = _buscar_plantilla(descripcion)

    print(f"\n{'='*60}")
    print(f"[IA] Iniciando pipeline para: '{descripcion}'")
    print(f"{'='*60}")

    figura = None

    # ── PASO 0a: Descriptor físico ─────────────────────────────────────────
    print("[paso 0a] Describiendo partes físicas del objeto...")
    descripcion_fisica = _generar_descripcion_fisica(descripcion)
    if descripcion_fisica:
        print(f"[paso 0a] Descripción física recibida:\n{descripcion_fisica}\n")
    else:
        print("[paso 0a] El modelo no respondió; se continúa sin descripción física.")

    # ── PASO 0b: Dimensionamiento explícito ────────────────────────────────
    dims = {}
    texto_dims = ""
    if descripcion_fisica:
        dims = _calcular_dimensiones(descripcion_fisica)
        if dims:
            texto_dims = _dims_a_texto(dims)
            roles_encontrados = ", ".join(dims.keys())
            print(f"[paso 0b] Bounding-boxes asignadas para: {roles_encontrados}")
            print(f"[paso 0b] {texto_dims}\n")
        else:
            print("[paso 0b] No se reconocieron partes con reglas de dimensionamiento.")

    # ── PASO 1-CATÁLOGO: ensamblado determinístico (sin LLM) ───────────────
    # Si todas las partes de la descripción física resuelven a una forma del
    # catálogo con bbox asignado, la geometría se arma en código y nos
    # ahorramos por completo el paso 1 del LLM (y con él, el bug del cubo).
    if descripcion_fisica and dims:
        figura_catalogo = _ensamblar_figura_por_catalogo(descripcion_fisica, dims)
        if figura_catalogo and _validar_figura(figura_catalogo):
            n_p  = len(figura_catalogo['puntos'])
            n_c  = len(figura_catalogo['conexiones'])
            n_pr = len(figura_catalogo.get('primitivas', []))
            print(
                f"[paso 1-catálogo] ✓ Figura ensamblada 100% por catálogo "
                f"(sin LLM): {n_p} puntos, {n_c} conexiones, {n_pr} primitivas."
            )
            print(f"[IA] ✓ Pipeline exitoso (catálogo): {n_p} puntos, {n_c} conexiones, {n_pr} primitivas.")
            print(f"{'='*60}\n")
            _guardar_cache(descripcion, figura_catalogo)
            return figura_catalogo
        else:
            print("[paso 1-catálogo] Ensamblado incompleto (forma o parte fuera del catálogo); usando LLM.")

    # ── Construcción del prompt para el paso 1 ─────────────────────────────
    if descripcion_fisica:
        contenido_paso1 = (
            f"{descripcion}\n\n"
            f"Descripción física de referencia (partes, forma, tamaño y ubicación relativa):\n"
            f"{descripcion_fisica}\n\n"
        )
        if texto_dims:
            contenido_paso1 += (
                f"{texto_dims}\n\n"
            )
        contenido_paso1 += (
            f"Convertí esta descripción al formato compacto, respetando las formas, tamaños "
            f"relativos, ubicaciones indicadas Y los rangos x/y asignados arriba. "
            f"Cada línea de la descripción física trae un campo 'contacto': si dice "
            f"'toca:<lado_propio>=<lado_otra>:<Parte>', el borde propio y el de esa otra parte "
            f"tienen que quedar en el MISMO número exacto (no un valor parecido) — si ambas son "
            f"partes de puntos, usá el MISMO índice Pi para ese punto compartido en vez de crear "
            f"uno nuevo; si alguna es una primitiva (K:/R:/C:/E:/S:/Y:), calculá su coordenada de "
            f"contacto a partir de la otra parte ya definida. Si dice 'simetrica_a:<Parte>', "
            f"copiá el mismo tamaño exacto y reflejá la posición sobre x=0.5. No agregues puntos "
            f"de más: cada parte angular usa exactamente sus esquinas/puntas mínimas (un "
            f"triángulo 3, un rectángulo 4), ni una más.\n\n"
            f"RECORDÁ EL FORMATO EXACTO: un punto por línea como 'P0: x,y' y 'P1: x,y' cada "
            f"uno en su propia línea, NUNCA todos los puntos juntos en una sola línea ni con 'Px:'."
        )
    else:
        contenido_paso1 = descripcion

    # ── PASO 1: Generador ─────────────────────────────────────────────────
    print("[paso 1] Generando esquema...")
    esquema = _llamar_modelo(
        messages=[
            {"role": "system", "content": SYSTEM_FIGURA_RAZONAMIENTO},
            {"role": "user", "content": contenido_paso1},
        ],
        num_predict=-1,
        temperatura=0.25,
    )

    if not esquema:
        print("[paso 1] El modelo no respondió.")
    else:
        print(f"[paso 1] Esquema recibido:\n{esquema}\n")
        datos = _parsear_formato_compacto(esquema)
        if datos:
            try:
                figura = _normalizar_figura(datos)
                if not _validar_figura(figura):
                    figura = None
                    print("[paso 1] Figura vacía o inválida tras normalizar.")
                else:
                    n_p  = len(figura['puntos'])
                    n_c  = len(figura['conexiones'])
                    n_pr = len(figura.get('primitivas', []))
                    print(f"[paso 1] → {n_p} puntos, {n_c} conexiones, {n_pr} primitivas.")
            except (ValueError, TypeError) as e:
                print(f"[paso 1] Error al normalizar: {e}")
                figura = None
        else:
            print("[paso 1] No se pudo parsear el esquema.")

    if not figura:
        print("[IA] Paso 1 falló. Usando plantilla de seguridad.")
        if plantilla_seguridad:
            return plantilla_seguridad
        return None

    # ── PASO 2a: Corrector de confusión puntos+primitivas ─────────────────
    # Detecta si el modelo mezcló puntos sueltos con primitivas sin terminar
    # ninguno de los dos sistemas. En ese caso pide un segundo intento más enfocado.
    n_p  = len(figura['puntos'])
    n_c  = len(figura['conexiones'])
    n_pr = len(figura.get('primitivas', []))

    # "Confuso" = hay primitivas Y puntos, pero los puntos no están bien conectados
    # (conexiones < puntos - 1 significa que hay puntos sin conectar a nada)
    figura_confusa = n_pr > 0 and n_p > 0 and n_c < n_p - 1

    if figura_confusa:
        print(f"[paso 2a] Figura confusa ({n_p} puntos, {n_c} conexiones, {n_pr} primitivas). Reintentando...")
        contexto_fisico = (
            f"\n\nRecordá la descripción física de referencia:\n{descripcion_fisica}"
            if descripcion_fisica else ""
        )
        contexto_dims = (
            f"\n\n{texto_dims}"
            if texto_dims else ""
        )
        esquema2 = _llamar_modelo(
            messages=[
                {"role": "system", "content": SYSTEM_FIGURA_RAZONAMIENTO},
                {
                    "role": "user",
                    "content": (
                        f"{descripcion}\n\n"
                        f"IMPORTANTE: el primer intento mezcló puntos sueltos con círculos y quedó mal.\n"
                        f"Esta vez elegí UNO de los dos enfoques y aplicalo completo:\n"
                        f"  OPCIÓN A (recomendada para objetos redondos): escribí L: vacío y usá solo C:/R:/E:\n"
                        f"  OPCIÓN B (para siluetas angulares): definí todos los puntos y conectalos con L:\n"
                        f"NO mezcles puntos sueltos sin conexión con primitivas."
                        f"{contexto_fisico}"
                        f"{contexto_dims}"
                    ),
                },
            ],
            num_predict=-1,
            temperatura=0.15,
        )

        if esquema2:
            print(f"[paso 2a] Esquema corregido:\n{esquema2}\n")
            datos2 = _parsear_formato_compacto(esquema2)
            if datos2:
                try:
                    figura2 = _normalizar_figura(datos2)
                    if _validar_figura(figura2):
                        figura = figura2
                        n_p2  = len(figura['puntos'])
                        n_c2  = len(figura['conexiones'])
                        n_pr2 = len(figura.get('primitivas', []))
                        print(f"[paso 2a] → {n_p2} puntos, {n_c2} conexiones, {n_pr2} primitivas.")
                    else:
                        print("[paso 2a] Corrección inválida; manteniendo paso 1.")
                except (ValueError, TypeError) as e:
                    print(f"[paso 2a] Error: {e}; manteniendo paso 1.")
            else:
                print("[paso 2a] No se pudo parsear; manteniendo paso 1.")
        else:
            print("[paso 2a] Modelo sin respuesta; manteniendo paso 1.")
    else:
        print("[paso 2a] Figura coherente.")

    # ── PASO 2b: Validación de bounding-boxes ─────────────────────────────
    # Solo se ejecuta si el paso 0b calculó al menos una bbox.
    if dims:
        print("[paso 2b] Validando bounding-boxes...")
        figura_corr, bbox_ok, advertencias = validar_y_corregir_bboxes(figura, dims)

        if advertencias:
            for aviso in advertencias:
                print(f"[paso 2b]{aviso}")

        if bbox_ok:
            figura = figura_corr
            print("[paso 2b] ✓ Bboxes respetadas (o violaciones menores corregidas).")
        else:
            # Violación grave: reintentar el paso 1 con instrucciones más estrictas
            print("[paso 2b] ✗ Violación grave de bbox. Reintentando con restricciones explícitas...")
            contenido_reintento = (
                f"{descripcion}\n\n"
                f"El intento anterior NO respetó las dimensiones asignadas. "
                f"Este intento es OBLIGATORIO que las respete:\n\n"
                f"{texto_dims}\n\n"
                f"Cada punto Pi y cada primitiva C/R/E deben quedar DENTRO de los rangos "
                f"x e y indicados para su parte. "
                f"Un punto que esté 0.01 fuera del rango ya es un error.\n\n"
            )
            if descripcion_fisica:
                contenido_reintento += (
                    f"Descripción física:\n{descripcion_fisica}\n\n"
                )
            contenido_reintento += (
                f"RECORDÁ EL FORMATO: un punto por línea 'P0: x,y', 'P1: x,y', etc. "
                f"NUNCA todos juntos en una línea."
            )

            esquema_r = _llamar_modelo(
                messages=[
                    {"role": "system", "content": SYSTEM_FIGURA_RAZONAMIENTO},
                    {"role": "user", "content": contenido_reintento},
                ],
                num_predict=-1,
                temperatura=0.10,   # temperatura muy baja: queremos precisión, no creatividad
            )

            if esquema_r:
                print(f"[paso 2b] Esquema reintentado:\n{esquema_r}\n")
                datos_r = _parsear_formato_compacto(esquema_r)
                if datos_r:
                    try:
                        figura_r = _normalizar_figura(datos_r)
                        if _validar_figura(figura_r):
                            # Validar las bboxes del reintento también
                            figura_r2, bbox_ok2, adv2 = validar_y_corregir_bboxes(figura_r, dims)
                            if adv2:
                                for aviso in adv2:
                                    print(f"[paso 2b-r]{aviso}")
                            figura = figura_r2   # usar reintento (corregido) pase o no pase bbox
                            estado = "✓" if bbox_ok2 else "⚠ aún con violaciones"
                            n_pr2  = len(figura['puntos'])
                            n_cr2  = len(figura['conexiones'])
                            n_prr2 = len(figura.get('primitivas', []))
                            print(
                                f"[paso 2b] Reintento {estado}: "
                                f"{n_pr2} puntos, {n_cr2} conexiones, {n_prr2} primitivas."
                            )
                        else:
                            print("[paso 2b] Reintento inválido; usando figura con correcciones clampadas.")
                            figura = figura_corr
                    except (ValueError, TypeError) as e:
                        print(f"[paso 2b] Error en reintento: {e}; usando figura clampada.")
                        figura = figura_corr
                else:
                    print("[paso 2b] No se pudo parsear reintento; usando figura clampada.")
                    figura = figura_corr
            else:
                print("[paso 2b] Sin respuesta en reintento; usando figura clampada.")
                figura = figura_corr
    else:
        print("[paso 2b] Sin dims de paso 0b; omitiendo validación bbox.")

    # ── Resultado final ────────────────────────────────────────────────────
    if not _validar_figura(figura):
        print("[IA] Figura final inválida. Usando plantilla de seguridad.")
        if plantilla_seguridad:
            return plantilla_seguridad
        return None

    n_p  = len(figura['puntos'])
    n_c  = len(figura['conexiones'])
    n_pr = len(figura.get('primitivas', []))
    print(f"[IA] ✓ Pipeline exitoso: {n_p} puntos, {n_c} conexiones, {n_pr} primitivas.")
    print(f"{'='*60}\n")

    _guardar_cache(descripcion, figura)
    return figura


# ----------------------------------------------------------------------------------
# 2) Interpretar el gesto de la mano en contacto con un objeto, en segundo plano
# ----------------------------------------------------------------------------------

class InterpreteGestos:
    """Consulta al modelo qué color debería tomar un objeto según el gesto de la mano y si lo
    está tocando. Pensado para correr en paralelo al video, SIN frenarlo."""

    def __init__(self, intervalo_minimo=1.0):
        self.intervalo_minimo = intervalo_minimo
        self._ultimo_envio = 0.0
        self._ultimo_estado = None
        self._color_sugerido = None
        self._lock = threading.Lock()

    def actualizar(self, nombres_dedos, estados_dedos, contacto):
        """Llamar una vez por frame. Internamente decide si hace falta o no consultar al modelo."""
        estado_actual = (tuple(estados_dedos), bool(contacto))
        ahora = time.time()

        if estado_actual == self._ultimo_estado:
            return
        if ahora - self._ultimo_envio < self.intervalo_minimo:
            return

        self._ultimo_estado = estado_actual
        self._ultimo_envio = ahora

        datos = {
            "dedos": {nombre: bool(ext) for nombre, ext in zip(nombres_dedos, estados_dedos)},
            "contacto": bool(contacto),
        }
        threading.Thread(target=self._consultar, args=(datos,), daemon=True).start()

    def _consultar(self, datos):
        contenido = _llamar_modelo(
            messages=[
                {"role": "system", "content": SYSTEM_GESTO},
                {"role": "user", "content": json.dumps(datos, ensure_ascii=False)},
            ],
            num_predict=-1,
        )

        resultado = _extraer_json(contenido)
        if not resultado or "color" not in resultado:
            return

        color = resultado["color"]
        if isinstance(color, list) and len(color) == 3:
            try:
                color_validado = tuple(max(0, min(255, int(c))) for c in color)
            except (ValueError, TypeError):
                return
            with self._lock:
                self._color_sugerido = color_validado

    def obtener_color(self):
        """Devuelve el último color sugerido por el modelo, o None si todavía no contestó nada."""
        with self._lock:
            return self._color_sugerido


if __name__ == "__main__":
    print("Pidiendo una figura al modelo...")
    figura = generar_figura("una casa")
    print("Resultado final:", figura)

    print("\nPidiendo la interpretación de un gesto...")
    interprete = InterpreteGestos(intervalo_minimo=0)
    interprete.actualizar(
        nombres_dedos=["Pulgar", "Índice", "Medio", "Anular", "Meñique"],
        estados_dedos=[False, False, False, False, False],
        contacto=True,
    )
    time.sleep(5)
    print(interprete.obtener_color())