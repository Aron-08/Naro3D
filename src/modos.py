"""
modos.py — Modos de uso en tiempo real: estructural, térmico, eléctrico. Uno
solo activo a la vez, actualizándose cada frame con la entrada de la mano
(peso aplicado, cercanía a una fuente de calor, tensión de una pila virtual),
dibujando un gauge común y pidiéndole un comentario a la IA solo cuando el
estado cambia de verdad — no cada frame.

Por qué una interfaz común (ver charla de diseño): sin esto, en dos meses hay
tres copias de "aplicar entrada → calcular → dibujar gauge → decidir si
avisar", una por dominio. Con `Modo` + `ResultadoModo` normalizado, las tres
implementaciones concretas de abajo (ModoEstructural, ModoTermico,
ModoElectrico) son delgadas: cada una solo traduce SU física a los mismos 5
campos, y GestorDeModos/dibujar_gauge/AsesorIA no necesitan saber de qué
dominio viene el número.

Regla dura de todo el módulo (la misma de las skills 04/05/06): CERO llamadas
al LLM dentro de tick(). Las tres físicas ya exponen una función de recálculo
puro-Python que no toca el modelo:
    - estructural : calculo_estructural.recalcular_carga()
    - térmico     : termodinamica.paso_integrador_concentrado()
    - eléctrico   : electrico.resolver_red()               (nunca llamó al LLM)
El LLM solo se usa UNA VEZ al entrar al modo (para elegir fórmula/resistencia
de referencia, modelo térmico, etc. — la parte de "criterio" de cada skill) y
después, de forma asíncrona y con histéresis, para comentar cuando el estado
cambia (ver AsesorIA). Nunca en el loop de 30 fps.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import calculo_estructural as est
import termodinamica as term
import electrico as elec
import objetos


# ---------------------------------------------------------------------------
# Resultado normalizado — igual forma para los 3 dominios
# ---------------------------------------------------------------------------

@dataclass
class ResultadoModo:
    valor_principal: float      # σ_MPa, T_°C, I_A — lo que corresponda al modo
    unidad: str                 # "MPa", "°C", "A" — para el HUD
    margen_seguridad: float     # 0..1 normalizado (0 = falló, 1 = sobra margen)
    estado: str                 # "ok" | "alerta" | "falla"
    modo_falla: str             # "pandeo", "excede_temp_servicio", "sobrecorriente", ...
    detalle: dict = field(default_factory=dict)   # lo que cada modo quiera mostrar extra


def _estado_desde_margen(margen: float) -> str:
    if margen <= 0.0:
        return "falla"
    if margen < 0.35:
        return "alerta"
    return "ok"


# ---------------------------------------------------------------------------
# Interfaz común
# ---------------------------------------------------------------------------

class Modo(ABC):
    nombre: str = "modo"

    @abstractmethod
    def entrar(self, objeto: str, contexto: dict) -> None:
        """Se llama UNA vez al activar el modo. Acá (y solo acá) puede haber
        una llamada al LLM — decidir criterio/fórmula una sola vez y
        cachearlo para que tick() sea 100% aritmética."""

    @abstractmethod
    def tick(self, dt: float, entrada: dict) -> ResultadoModo:
        """Se llama cada frame. PROHIBIDO tocar el LLM acá."""

    def salir(self) -> None:
        """Hook opcional para liberar estado al cambiar de modo."""
        pass


# ---------------------------------------------------------------------------
# Modo estructural
# ---------------------------------------------------------------------------

class ModoEstructural(Modo):
    nombre = "estructural"

    def entrar(self, objeto: str, contexto: dict) -> None:
        """contexto: {"escala_m_por_unidad", "tipo_apoyo", "longitud_m"?,
        "carga_inicial": {"tipo", "magnitud_n", "posicion_relativa", "descripcion"?}}."""
        self._objeto = objeto
        registro = objetos.cargar_objeto(objeto)
        if not registro or not registro.get("figura") or not registro.get("propiedades"):
            raise ValueError(f"'{objeto}' no tiene figura/propiedades — no se puede entrar al modo estructural.")

        propiedades_extendidas = objetos._propiedades_extendidas_actualizadas(objeto, registro)
        self._material = {**registro["propiedades"], **propiedades_extendidas}
        self._material.pop("_material_cacheado", None)

        self._geometria = est.geometria_estructural_desde_figura(
            registro["figura"], contexto["escala_m_por_unidad"],
            longitud_m=contexto.get("longitud_m"),
            tipo_apoyo=contexto.get("tipo_apoyo", "simple"),
        )

        # Única llamada al LLM del modo: decide fórmula + resistencia de
        # referencia UNA vez, con la carga inicial. Después, tick() solo
        # cambia la magnitud de la carga y recalcula con la misma fórmula.
        carga_inicial = contexto.get("carga_inicial", {"tipo": "puntual", "magnitud_n": 0.0, "posicion_relativa": 0.5})
        resultado_inicial = est.evaluar_carga(self._geometria, self._material, carga_inicial)
        self._formula = resultado_inicial["formula"]
        self._resistencia_referencia_mpa = resultado_inicial["resistencia_referencia_mpa"]
        self._tipo_carga = carga_inicial.get("tipo", "puntual")
        self._posicion_relativa = carga_inicial.get("posicion_relativa", 0.5)

    def tick(self, dt: float, entrada: dict) -> ResultadoModo:
        """entrada: {"magnitud_n": float} — ej. peso actual mapeado desde un
        gesto de la mano (pinch → kg → N)."""
        carga = {
            "tipo": self._tipo_carga,
            "magnitud_n": entrada.get("magnitud_n", 0.0),
            "posicion_relativa": self._posicion_relativa,
        }
        r = est.recalcular_carga(self._formula, self._resistencia_referencia_mpa,
                                  self._geometria, self._material, carga)

        fs = r["factor_seguridad"]
        # margen_seguridad normalizado: FS=1 -> 0 (falla), FS>=2.5 -> 1 (sobra).
        # FS<1 ya es "más allá de falla" -> clampeado a 0 (no negativo).
        margen = 0.0 if fs <= 1.0 else min(1.0, (fs - 1.0) / 1.5)

        modo_falla = "ninguno"
        if fs <= 1.0:
            modo_falla = "pandeo" if r.get("pandea") else "fractura_fragil_o_fluencia"
        elif r.get("margen_bajo"):
            modo_falla = "margen_bajo"

        return ResultadoModo(
            valor_principal=r["sigma_mpa"],
            unidad="MPa",
            margen_seguridad=margen,
            estado=_estado_desde_margen(margen),
            modo_falla=modo_falla,
            detalle={"factor_seguridad": fs, "deflexion_m": r.get("deflexion_m"), "formula": self._formula},
        )


# ---------------------------------------------------------------------------
# Modo térmico
# ---------------------------------------------------------------------------

class ModoTermico(Modo):
    nombre = "termico"

    def entrar(self, objeto: str, contexto: dict) -> None:
        """contexto: {"temp_ambiente_c", "temp_inicial_c"?, "h_w_m2k"?}."""
        self._objeto = objeto
        registro = objetos.cargar_objeto(objeto)
        if not registro or not registro.get("propiedades"):
            raise ValueError(f"'{objeto}' no tiene propiedades — no se puede entrar al modo térmico.")

        propiedades_extendidas = objetos._propiedades_extendidas_actualizadas(objeto, registro)
        self._material = {**registro["propiedades"], **propiedades_extendidas}
        self._material.pop("_material_cacheado", None)

        geometria = term.geometria_termica_desde_ficha(
            peso_kg=self._material.get("peso_kg", 0.0),
            densidad_kg_m3=self._material.get("densidad_kg_m3", 0.0),
        )
        masa_kg = geometria["volumen_estimado_m3"] * self._material.get("densidad_kg_m3", 1000.0)
        h = contexto.get("h_w_m2k", term.H_CONVECCION_AIRE_LIBRE_W_M2K)
        self._tau_s = term.constante_tiempo_concentrado(
            masa_kg, self._material.get("calor_especifico_j_kgk", 900.0), h, geometria["area_expuesta_m2"]
        )
        self._temp_ambiente_c = contexto.get("temp_ambiente_c", 20.0)
        self._temp_actual_c = contexto.get("temp_inicial_c", self._temp_ambiente_c)
        self._temp_max_servicio_c = self._material.get("temperatura_max_servicio_c")
        self._punto_fusion_c = self._material.get("punto_fusion_c")

    def tick(self, dt: float, entrada: dict) -> ResultadoModo:
        """entrada: {"temp_fuente_c": float | None, "distancia_normalizada": float
        0..1 (0 = tocando la fuente, 1 = lejos)}. Si no hay fuente cerca,
        el objetivo es la temperatura ambiente (se enfría/vuelve a estabilizar)."""
        temp_fuente = entrada.get("temp_fuente_c")
        distancia = entrada.get("distancia_normalizada", 1.0)
        if temp_fuente is None or distancia >= 1.0:
            temp_objetivo = self._temp_ambiente_c
        else:
            # Interpolación simple fuente<->ambiente según cercanía — el
            # integrador (paso_integrador_concentrado) es el que de verdad
            # mueve la temperatura en el tiempo; esto solo define hacia dónde
            # relaja en este instante, y puede cambiar de un tick a otro sin
            # perder estabilidad (ver docstring de la función).
            temp_objetivo = self._temp_ambiente_c + (temp_fuente - self._temp_ambiente_c) * (1.0 - distancia)

        self._temp_actual_c = term.paso_integrador_concentrado(
            self._temp_actual_c, temp_objetivo, dt, self._tau_s
        )

        riesgo = "ninguno"
        if self._punto_fusion_c and self._temp_actual_c >= self._punto_fusion_c:
            riesgo = "excede_punto_fusion"
        elif self._temp_max_servicio_c and self._temp_actual_c >= self._temp_max_servicio_c:
            riesgo = "excede_temp_servicio"

        if self._temp_max_servicio_c and self._temp_max_servicio_c > self._temp_ambiente_c:
            margen = 1.0 - (self._temp_actual_c - self._temp_ambiente_c) / \
                (self._temp_max_servicio_c - self._temp_ambiente_c)
            margen = max(0.0, min(1.0, margen))
        else:
            margen = 1.0

        return ResultadoModo(
            valor_principal=self._temp_actual_c,
            unidad="°C",
            margen_seguridad=margen,
            estado=_estado_desde_margen(margen) if riesgo == "ninguno" else "falla",
            modo_falla=riesgo,
            detalle={"tau_s": self._tau_s, "temp_objetivo_c": temp_objetivo},
        )


# ---------------------------------------------------------------------------
# Modo eléctrico
# ---------------------------------------------------------------------------

class ModoElectrico(Modo):
    nombre = "electrico"

    def entrar(self, objeto: str, contexto: dict) -> None:
        """objeto acá es el nombre de la FUENTE (ej. "pila"); contexto:
        {"objetos_circuito": [nombres...], "tension_v", "margen_contacto"?}.
        No hace falta llamar al LLM para entrar a este modo — la topología
        es geometría (contacto) y la resolución es álgebra lineal exacta,
        ninguna de las dos necesita "criterio" del modelo (a diferencia de
        estructural/térmico)."""
        import ubicacion as ubi

        self._objeto = objeto
        nombres = contexto["objetos_circuito"]
        self._fichas = {}
        for nombre in nombres:
            registro = objetos.cargar_objeto(nombre)
            if not registro or not registro.get("propiedades"):
                raise ValueError(f"'{nombre}' no tiene propiedades — no se puede armar el circuito.")
            self._fichas[nombre] = registro["propiedades"]

        self._nombres = nombres
        self._margen_contacto = contexto.get("margen_contacto", 0.02)
        self._tension_v = contexto.get("tension_v", 9.0)
        self._h_w_m2k = contexto.get("h_w_m2k", elec.H_CONVECCION_AIRE_LIBRE_W_M2K)
        self._temp_ambiente_c = contexto.get("temp_ambiente_c", 20.0)
        self._ubi = ubi

    def tick(self, dt: float, entrada: dict) -> ResultadoModo:
        """entrada: {} — la topología se re-lee de la escena cada tick (la
        mano puede estar acercando/separando componentes en vivo), pero
        resolver_red() es puro numpy, es barato llamarlo cada frame."""
        objetos_escena = [o for o in self._ubi.objetos_en_escena_actual() if o["nombre"] in self._fichas]
        elementos, mapa = elec.construir_red_desde_contacto(objetos_escena, self._fichas, self._margen_contacto)

        nodo_pos = mapa.get(f"{self._objeto}#A", f"{self._objeto}#A")
        nodo_neg = mapa.get(f"{self._objeto}#B", f"{self._objeto}#B")
        fuentes = [{"nombre": self._objeto, "nodo_pos": nodo_pos, "nodo_neg": nodo_neg, "tension_v": self._tension_v}]

        calculo = elec.resolver_red(elementos, fuentes, nodo_referencia=nodo_neg)

        if calculo["circuito_abierto"] or not calculo["corrientes_elemento"]:
            return ResultadoModo(0.0, "A", 1.0, "ok", "abierto", {"elementos": []})

        # El elemento más crítico: mayor corriente relativa a su propio límite
        # térmico (ver corriente_maxima_por_calentamiento_a). Si no hay ficha
        # de área/temp de servicio para un elemento, se lo ignora para el
        # riesgo (pero sigue apareciendo en detalle).
        peor_margen, peor_nombre, peor_i, peor_falla = 1.0, None, 0.0, "ninguno"
        for e in elementos:
            i = abs(calculo["corrientes_elemento"].get(e["nombre"], 0.0))
            ficha = self._fichas.get(e["nombre"], {})
            temp_max = ficha.get("temperatura_max_servicio_c")
            if not temp_max:
                continue
            geo = elec.geometria_electrica_desde_ficha(
                ficha.get("peso_kg", 0.0), ficha.get("densidad_kg_m3", 0.0),
                ficha.get("resistencia_electrica_ohm_m", 1e-8),
            )
            i_max = elec.corriente_maxima_por_calentamiento_a(
                geo["resistencia_ohm"], 6.0 * geo["longitud_m"] ** 2,
                self._temp_ambiente_c, temp_max, self._h_w_m2k,
            )
            if i_max <= 0:
                continue
            margen = max(0.0, min(1.0, 1.0 - i / i_max))
            if margen < peor_margen:
                peor_margen, peor_nombre, peor_i = margen, e["nombre"], i
                peor_falla = "sobrecorriente" if margen <= 0.0 else "ninguno"

        return ResultadoModo(
            valor_principal=peor_i,
            unidad="A",
            margen_seguridad=peor_margen,
            estado=_estado_desde_margen(peor_margen),
            modo_falla=peor_falla,
            detalle={"elemento_critico": peor_nombre, "corrientes": calculo["corrientes_elemento"],
                     "voltajes_nodo": calculo["voltajes_nodo"]},
        )


# ---------------------------------------------------------------------------
# Asesor IA — comenta el estado, con histéresis + cooldown + caché
# ---------------------------------------------------------------------------
# Mismo patrón que el consultor asíncrono que ya existe en ia_interprete.py
# (Lock + hilo daemon + no bloquear el llamador): acá se le suma histéresis
# (confirmar N ticks seguidos en el nuevo estado antes de comentar, para no
# reaccionar a un pico de un solo frame de ruido del gesto) y caché por firma
# de estado (si el usuario oscila entre "alerta" y "ok" en el mismo umbral,
# se reusa el comentario ya generado en vez de volver a llamar al modelo).

SYSTEM_ASESOR_MODOS = """Comentás en UNA frase corta y en lenguaje llano el cambio de estado de una
simulación física (estructural, térmica o eléctrica) que le está mostrando a un usuario en
tiempo real. Nunca inventes números — los que te pasan ya están calculados. No repitas el
valor exacto si ya se ve en el HUD; agregá contexto o una sugerencia breve. Una sola frase,
sin texto adicional."""


class AsesorIA:
    def __init__(self, frames_confirmacion: int = 6, cooldown_s: float = 4.0):
        self._frames_confirmacion = frames_confirmacion
        self._cooldown_s = cooldown_s
        self._lock = threading.Lock()
        self._cache: dict[tuple, str] = {}
        self._ultimo_estado_confirmado: str | None = None
        self._contador_frames = 0
        self._ultimo_comentario_ts = 0.0
        self._comentario_actual = ""
        self._generando = False

    def obtener_comentario(self) -> str:
        with self._lock:
            return self._comentario_actual

    def observar(self, objeto: str, resultado: ResultadoModo) -> None:
        firma = (objeto, resultado.modo_falla, resultado.estado)

        if resultado.estado == self._ultimo_estado_confirmado:
            self._contador_frames = 0
            return

        self._contador_frames += 1
        if self._contador_frames < self._frames_confirmacion:
            return   # todavía no se sostuvo suficientes frames seguidos

        self._contador_frames = 0
        self._ultimo_estado_confirmado = resultado.estado

        if firma in self._cache:
            with self._lock:
                self._comentario_actual = self._cache[firma]
            return

        ahora = time.monotonic()
        if ahora - self._ultimo_comentario_ts < self._cooldown_s:
            return
        if self._generando:
            return

        self._ultimo_comentario_ts = ahora
        self._generando = True
        threading.Thread(target=self._generar, args=(firma, objeto, resultado), daemon=True).start()

    def _generar(self, firma: tuple, objeto: str, resultado: ResultadoModo) -> None:
        import json
        import modelos   # modelo/temperatura de este paso vienen de modelos_config.json ("asesor_modos")
        resumen = {
            "objeto": objeto,
            "estado": resultado.estado,
            "modo_falla": resultado.modo_falla,
            "valor_principal": round(resultado.valor_principal, 3),
            "unidad": resultado.unidad,
            "margen_seguridad": round(resultado.margen_seguridad, 3),
        }
        texto = modelos.llamar(
            "asesor_modos",
            messages=[
                {"role": "system", "content": SYSTEM_ASESOR_MODOS},
                {"role": "user", "content": json.dumps(resumen, ensure_ascii=False)},
            ],
        ) or ""
        texto = texto.strip()
        with self._lock:
            self._cache[firma] = texto
            self._comentario_actual = texto
            self._generando = False


# ---------------------------------------------------------------------------
# Gestor de modos — uno activo a la vez, orquesta tick + asesor + HUD
# ---------------------------------------------------------------------------

class GestorDeModos:
    def __init__(self):
        self._modos: dict[str, Modo] = {
            "estructural": ModoEstructural(),
            "termico": ModoTermico(),
            "electrico": ModoElectrico(),
        }
        self._activo: Modo | None = None
        self._objeto_activo: str | None = None
        self.asesor = AsesorIA()
        self.ultimo_resultado: ResultadoModo | None = None

    def activar(self, nombre_modo: str, objeto: str, contexto: dict) -> None:
        if self._activo is not None:
            self._activo.salir()
        modo = self._modos[nombre_modo]
        modo.entrar(objeto, contexto)   # única llamada al LLM del ciclo (si el modo la necesita)
        self._activo = modo
        self._objeto_activo = objeto
        self.ultimo_resultado = None

    def desactivar(self) -> None:
        if self._activo is not None:
            self._activo.salir()
        self._activo = None
        self._objeto_activo = None

    def tick(self, dt: float, entrada: dict) -> ResultadoModo | None:
        if self._activo is None:
            return None
        resultado = self._activo.tick(dt, entrada)   # 100% Python, sin LLM
        self.ultimo_resultado = resultado
        self.asesor.observar(self._objeto_activo, resultado)   # dispara comentario async si corresponde
        return resultado


# ---------------------------------------------------------------------------
# HUD — gauge común (OpenCV), igual para los 3 dominios
# ---------------------------------------------------------------------------
# Mismo criterio de threading que el resto del proyecto (ui_thread.py): esta
# función asume que la llama el hilo que dibuja el frame (el mismo que ya
# hace cv2.putText en main.py), nunca un hilo de fondo directo.

_COLOR_OK = (80, 200, 80)
_COLOR_ALERTA = (0, 200, 255)
_COLOR_FALLA = (0, 0, 255)


def dibujar_gauge(frame, resultado: ResultadoModo, comentario: str = "",
                   x: int = 20, y: int = 20, ancho: int = 260) -> None:
    import cv2

    color = {"ok": _COLOR_OK, "alerta": _COLOR_ALERTA, "falla": _COLOR_FALLA}[resultado.estado]

    cv2.rectangle(frame, (x, y), (x + ancho, y + 18), (60, 60, 60), -1)
    relleno = int(ancho * resultado.margen_seguridad)
    cv2.rectangle(frame, (x, y), (x + relleno, y + 18), color, -1)
    cv2.rectangle(frame, (x, y), (x + ancho, y + 18), (255, 255, 255), 1)

    texto_valor = f"{resultado.valor_principal:.2f} {resultado.unidad}  ({resultado.estado})"
    cv2.putText(frame, texto_valor, (x, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if resultado.modo_falla != "ninguno":
        cv2.putText(frame, f"riesgo: {resultado.modo_falla}", (x, y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_FALLA, 1, cv2.LINE_AA)

    if comentario:
        cv2.putText(frame, comentario, (x, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)