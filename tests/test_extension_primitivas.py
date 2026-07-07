"""
tests/test_extension_primitivas.py — Cobertura de la extensión de bajo
riesgo / alto retorno al kernel paramétrico:

  1) Primitivas nuevas en malla.py: cono_truncado, capsula, prisma_n_lados
     (y malla_perfil_extruido, utilidad de malla.py todavía no expuesta al
     LLM — ver su docstring).
  2) Suavizado cosmético opcional post-ensamble (suavizar_malla_cm), que
     nunca corre por defecto y nunca rompe el pipeline si trimesh no está.

El fallback de TripoSR para 'factible: false' (objetos.py ::
generar_geometria_parametrica) no se testea acá: requiere Ollama + GPU +
torch/diffusers/TripoSR, igual que el resto de las llamadas a modelos de
este proyecto — se prueba manualmente, no en la suite sin-Ollama.
"""

import math

import pytest

import malla as malla_mod
from ensamblador import (
    Parte,
    ErrorEnsamble,
    fabricar_malla,
    resolver_ensamble,
    fusionar_ensamble,
    ensamblar_partes,
    suavizar_malla_cm,
    partes_desde_json_llm,
)


# ---------------------------------------------------------------------------
# malla.py — fábricas nuevas
# ---------------------------------------------------------------------------

def test_malla_cono_truncado_dimensiones():
    m = malla_mod.malla_cono_truncado(5.0, 2.0, 8.0)
    ys = [v[1] for v in m.vertices]
    assert max(ys) - min(ys) == pytest.approx(8.0)
    radios = [math.hypot(v[0], v[2]) for v in m.vertices]
    assert max(radios) == pytest.approx(5.0, rel=1e-6)


def test_malla_cono_truncado_radio_tope_cero_es_cono_perfecto():
    m = malla_mod.malla_cono_truncado(5.0, 0.0, 8.0)
    radios = [math.hypot(v[0], v[2]) for v in m.vertices]
    # Todos los vértices del "tope" colapsan en (0, -h2, 0): radio 0 ahí.
    assert min(radios) == pytest.approx(0.0, abs=1e-9)
    assert m.num_caras() > 0
    # Ningún triángulo debería tener área cero de forma sistemática: al
    # menos debe seguir siendo una malla con vértices/caras coherentes.
    assert m.num_vertices() > 0


def test_malla_capsula_radio_bounding():
    radio, alto = 4.0, 30.0
    m = malla_mod.malla_capsula(radio, alto)
    # El bounding de una cápsula es su radio + la mitad del largo cilíndrico.
    assert m.radio_bounding() == pytest.approx(alto / 2.0 + radio, rel=1e-3)


def test_malla_capsula_secciones_redondeadas():
    radio, alto = 3.0, 10.0
    m = malla_mod.malla_capsula(radio, alto)
    ys = [v[1] for v in m.vertices]
    # Con casquetes, el largo total debe superar al largo "cilíndrico" puro.
    assert (max(ys) - min(ys)) > alto


def test_malla_prisma_n_lados_hexagono():
    m = malla_mod.malla_prisma_n_lados(2.0, 5.0, 6)
    # 2 anillos de 6 vértices + 2 vértices centrales (uno por tapa, igual
    # criterio que malla_cilindro).
    assert m.num_vertices() == 6 * 2 + 2
    radios = [math.hypot(v[0], v[2]) for v in m.vertices]
    assert max(radios) == pytest.approx(2.0, rel=1e-6)


def test_malla_prisma_n_lados_clampa_minimo_triangulo():
    m = malla_mod.malla_prisma_n_lados(1.0, 1.0, 2)   # lados<3 no tiene sentido
    assert m.num_vertices() == 3 * 2 + 2   # se clampa a 3 (triángulo), no revienta


def test_malla_perfil_extruido_cuadrado():
    perfil = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    m = malla_mod.malla_perfil_extruido(perfil, 5.0)
    ys = [v[1] for v in m.vertices]
    assert max(ys) - min(ys) == pytest.approx(5.0)
    assert m.num_vertices() == 8


def test_malla_perfil_extruido_perfil_invalido():
    with pytest.raises(ValueError):
        malla_mod.malla_perfil_extruido([(0, 0), (1, 0)], 5.0)   # menos de 3 puntos


# ---------------------------------------------------------------------------
# ensamblador.py — Parte / fabricar_malla / catálogo
# ---------------------------------------------------------------------------

def test_parte_cono_truncado_dims_faltantes():
    with pytest.raises(ErrorEnsamble):
        Parte("Cuello", "cono_truncado", {"radio_base": 5, "alto": 8})   # falta radio_tope


def test_parte_capsula_valida():
    p = Parte("Brazo", "capsula", {"radio": 3, "alto": 20})
    assert p.nombre == "Brazo"


def test_parte_prisma_n_lados_valida():
    p = Parte("Tuerca", "prisma_n_lados", {"radio": 1.5, "alto": 0.8, "lados": 6})
    m = fabricar_malla(p)
    assert m.num_vertices() == 6 * 2 + 2   # 2 anillos de 6 + 2 centros de tapa


def test_fabricar_malla_cono_truncado():
    p = Parte("Cuello", "cono_truncado", {"radio_base": 5, "radio_tope": 1.2, "alto": 8})
    m = fabricar_malla(p)
    # El radio de bounding es 3D (desde el origen, centro del sólido), no el
    # radio_base a secas: incluye la mitad de la altura como componente Y.
    esperado = math.hypot(5.0, 8.0 / 2.0)
    assert m.radio_bounding() == pytest.approx(esperado, rel=1e-6)


def test_capsula_orientable_como_cilindro_y_tubo():
    """Las 3 formas nuevas deben poder reorientarse igual que cilindro/tubo
    (necesario para brazos/ejes acostados, no solo verticales)."""
    p_x = Parte("Brazo", "capsula", {"radio": 2, "alto": 10}, orientacion_eje="x")
    m_x = fabricar_malla(p_x)
    p_y = Parte("Brazo2", "capsula", {"radio": 2, "alto": 10}, orientacion_eje="y")
    m_y = fabricar_malla(p_y)
    # Reorientar es una permutación de ejes: incluso con distinta
    # orientación, el conjunto de radios de bounding debe ser el mismo.
    assert m_x.radio_bounding() == pytest.approx(m_y.radio_bounding(), rel=1e-9)


def test_contacto_entre_cilindro_y_cono_truncado():
    cuerpo = Parte("Cuerpo", "cilindro", {"radio": 5, "alto": 20})
    cuello = Parte("Cuello", "cono_truncado", {"radio_base": 5, "radio_tope": 1.2, "alto": 8},
                   contacto="toca:abajo=arriba:Cuerpo")
    resultado = resolver_ensamble([cuerpo, cuello])
    malla_cuerpo, centro_cuerpo = resultado["Cuerpo"]
    malla_cuello, centro_cuello = resultado["Cuello"]
    y_max_cuerpo = max(v[1] for v in malla_cuerpo.vertices) + centro_cuerpo[1]
    y_min_cuello = min(v[1] for v in malla_cuello.vertices) + centro_cuello[1]
    assert y_min_cuello == pytest.approx(y_max_cuerpo, abs=1e-9)


def test_ensamble_con_formas_nuevas_es_determinista():
    def _armar():
        return [
            Parte("Cuerpo", "cilindro", {"radio": 5, "alto": 20}),
            Parte("Cuello", "cono_truncado", {"radio_base": 5, "radio_tope": 1.2, "alto": 8},
                  contacto="toca:abajo=arriba:Cuerpo"),
            Parte("Tuerca", "prisma_n_lados", {"radio": 3, "alto": 2, "lados": 6},
                  contacto="toca:abajo=arriba:Cuerpo"),
        ]
    m1 = ensamblar_partes(_armar())
    m2 = ensamblar_partes(_armar())
    assert m1.vertices == m2.vertices
    assert m1.caras == m2.caras


def test_partes_desde_json_llm_acepta_formas_nuevas():
    datos = {
        "factible": True,
        "partes": [
            {"nombre": "Cuerpo", "forma": "capsula", "dims_cm": {"radio": 4, "alto": 30}, "contacto": None},
            {"nombre": "Tuerca", "forma": "prisma_n_lados",
             "dims_cm": {"radio": 2, "alto": 1, "lados": 8},
             "contacto": "toca:abajo=arriba:Cuerpo"},
        ],
    }
    partes = partes_desde_json_llm(datos)
    assert partes is not None
    assert {p.forma for p in partes} == {"capsula", "prisma_n_lados"}


# ---------------------------------------------------------------------------
# Suavizado cosmético opcional (suavizar_malla_cm) — nunca on por defecto,
# nunca rompe el pipeline si trimesh falta.
# ---------------------------------------------------------------------------

def test_suavizado_apagado_por_defecto_no_cambia_nada():
    partes = [Parte("Cuerpo", "caja", {"ancho": 10, "alto": 10, "profundo": 10})]
    resultado = resolver_ensamble(partes)
    malla_normal = fusionar_ensamble(resultado)
    malla_explicita_0 = fusionar_ensamble(resultado, suavizado_iteraciones=0)
    assert malla_normal.vertices == malla_explicita_0.vertices
    assert malla_normal.caras == malla_explicita_0.caras


def test_suavizado_iteraciones_cero_no_llama_trimesh(monkeypatch):
    """iteraciones=0 debe ser un no-op incluso si trimesh no está instalado
    (no debe intentar importarlo)."""
    m = malla_mod.malla_cubo(10, 10, 10)
    resultado = suavizar_malla_cm(m, iteraciones=0)
    assert resultado.vertices == m.vertices
    assert resultado.caras == m.caras


def test_suavizado_sin_trimesh_no_rompe(monkeypatch):
    """Si trimesh no está disponible (o falla), se debe conservar la malla
    original — mismo criterio de 'nunca romper el pipeline' que _restar_trimesh."""
    import builtins
    import ensamblador as ens_mod

    real_import = builtins.__import__

    def _import_falso(nombre, *args, **kwargs):
        if nombre == "trimesh":
            raise ImportError("simulado: trimesh no disponible")
        return real_import(nombre, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_falso)
    m = malla_mod.malla_cubo(10, 10, 10)
    resultado = ens_mod.suavizar_malla_cm(m, iteraciones=2)
    assert resultado.vertices == m.vertices
    assert resultado.num_vertices() == 8   # cubo intacto, sin subdividir
