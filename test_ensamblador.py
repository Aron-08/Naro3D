"""
tests/test_ensamblador.py — Tests del kernel de resolución de restricciones
(sección 9.2, 9.3, 9.4 del plan de kernel paramétrico). Ninguno de estos
tests necesita Ollama corriendo: el LLM está completamente ausente del
camino que se testea acá (esa es la ventaja central del kernel).
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
    _anclar_contacto,
    _orden_topologico,
    partes_desde_json_llm,
)


# ---------------------------------------------------------------------------
# Parte — validación de construcción
# ---------------------------------------------------------------------------

def test_parte_forma_invalida():
    with pytest.raises(ErrorEnsamble):
        Parte("X", "forma_inventada", {"ancho": 1})


def test_parte_dims_faltantes():
    with pytest.raises(ErrorEnsamble):
        Parte("Cuerpo", "caja", {"ancho": 10, "alto": 10})   # falta 'profundo'


def test_parte_valida_no_lanza():
    p = Parte("Cuerpo", "caja", {"ancho": 10, "alto": 10, "profundo": 10})
    assert p.nombre == "Cuerpo"


# ---------------------------------------------------------------------------
# fabricar_malla — dispatcher
# ---------------------------------------------------------------------------

def test_fabricar_malla_caja():
    p = Parte("C", "caja", {"ancho": 10, "alto": 20, "profundo": 30})
    m = fabricar_malla(p)
    xs = [v[0] for v in m.vertices]
    assert max(xs) - min(xs) == pytest.approx(10)


def test_fabricar_malla_esfera():
    p = Parte("E", "esfera", {"radio": 5})
    m = fabricar_malla(p)
    assert m.radio_bounding() == pytest.approx(5, rel=1e-6)


# ---------------------------------------------------------------------------
# 9.2 — _anclar_contacto (el más crítico)
# ---------------------------------------------------------------------------

def test_contacto_toca_abajo_arriba():
    cuerpo = fabricar_malla(Parte("Cuerpo", "caja", {"ancho": 40, "alto": 30, "profundo": 30}))
    centro_cuerpo = (0.0, 0.0, 0.0)
    techo = fabricar_malla(Parte("Techo", "prisma_triangular", {"ancho": 40, "alto": 15, "profundo": 30}))

    centro_techo = _anclar_contacto(techo, cuerpo, centro_cuerpo, "abajo", "arriba")

    # el punto más bajo del techo (ya trasladado) debe coincidir EXACTO
    # con el punto más alto del cuerpo (ya trasladado) — el prisma tiene su
    # base en y=0 local, así que "abajo" de un prisma es su y mínima (0.0).
    y_min_techo = min(v[1] for v in techo.vertices) + centro_techo[1]
    y_max_cuerpo = max(v[1] for v in cuerpo.vertices) + centro_cuerpo[1]
    assert y_min_techo == pytest.approx(y_max_cuerpo, abs=1e-9)


def test_contacto_toca_derecha_izquierda():
    a = fabricar_malla(Parte("A", "caja", {"ancho": 10, "alto": 10, "profundo": 10}))
    centro_a = (5.0, 2.0, -3.0)
    b = fabricar_malla(Parte("B", "caja", {"ancho": 6, "alto": 6, "profundo": 6}))

    centro_b = _anclar_contacto(b, a, centro_a, "izquierda", "derecha")

    x_max_a = max(v[0] for v in a.vertices) + centro_a[0]
    x_min_b = min(v[0] for v in b.vertices) + centro_b[0]
    assert x_min_b == pytest.approx(x_max_a, abs=1e-9)
    # los otros dos ejes quedan centrados con la parte de referencia
    assert centro_b[1] == pytest.approx(centro_a[1])
    assert centro_b[2] == pytest.approx(centro_a[2])


def test_anclar_contacto_lado_invalido():
    a = fabricar_malla(Parte("A", "caja", {"ancho": 10, "alto": 10, "profundo": 10}))
    b = fabricar_malla(Parte("B", "caja", {"ancho": 6, "alto": 6, "profundo": 6}))
    with pytest.raises(ErrorEnsamble):
        _anclar_contacto(b, a, (0, 0, 0), "arribaX", "arriba")


# ---------------------------------------------------------------------------
# _orden_topologico
# ---------------------------------------------------------------------------

def test_orden_topologico_respeta_dependencias():
    partes = [
        Parte("Techo", "prisma_triangular", {"ancho": 10, "alto": 5, "profundo": 10},
              contacto="toca:abajo=arriba:Cuerpo"),
        Parte("Cuerpo", "caja", {"ancho": 10, "alto": 10, "profundo": 10}),
    ]
    orden = _orden_topologico(partes)
    nombres = [p.nombre for p in orden]
    assert nombres.index("Cuerpo") < nombres.index("Techo")


def test_orden_topologico_referencia_inexistente():
    partes = [
        Parte("X", "caja", {"ancho": 1, "alto": 1, "profundo": 1},
              contacto="toca:abajo=arriba:NoExiste"),
    ]
    with pytest.raises(ErrorEnsamble):
        _orden_topologico(partes)


def test_orden_topologico_ciclo():
    partes = [
        Parte("A", "caja", {"ancho": 1, "alto": 1, "profundo": 1}, contacto="toca:abajo=arriba:B"),
        Parte("B", "caja", {"ancho": 1, "alto": 1, "profundo": 1}, contacto="toca:abajo=arriba:A"),
    ]
    with pytest.raises(ErrorEnsamble):
        _orden_topologico(partes)


def test_orden_topologico_nombres_duplicados():
    partes = [
        Parte("A", "caja", {"ancho": 1, "alto": 1, "profundo": 1}),
        Parte("A", "esfera", {"radio": 1}),
    ]
    with pytest.raises(ErrorEnsamble):
        _orden_topologico(partes)


# ---------------------------------------------------------------------------
# 9.3 — resolver_ensamble end-to-end
# ---------------------------------------------------------------------------

def test_resolver_ensamble_silla_sin_solapamiento_indeseado():
    partes = [
        Parte("Asiento", "caja", {"ancho": 45, "alto": 5, "profundo": 45}),
        Parte("Respaldo", "caja", {"ancho": 45, "alto": 45, "profundo": 5},
              contacto="toca:abajo=arriba:Asiento"),
        Parte("Pata_izq", "cilindro", {"radio": 2, "alto": 45},
              contacto="toca:arriba=abajo:Asiento"),
        Parte("Pata_der", "cilindro", {"radio": 2, "alto": 45},
              contacto="simetrica_a:Pata_izq"),
    ]
    resultado = resolver_ensamble(partes)
    assert set(resultado.keys()) == {"Asiento", "Respaldo", "Pata_izq", "Pata_der"}

    # Respaldo apoya exacto sobre Asiento (cero hueco, cero solape)
    malla_asiento, centro_asiento = resultado["Asiento"]
    malla_respaldo, centro_respaldo = resultado["Respaldo"]
    y_max_asiento = max(v[1] for v in malla_asiento.vertices) + centro_asiento[1]
    y_min_respaldo = min(v[1] for v in malla_respaldo.vertices) + centro_respaldo[1]
    assert y_min_respaldo == pytest.approx(y_max_asiento, abs=1e-9)

    # Pata_der es el reflejo exacto en X de Pata_izq
    _malla_izq, centro_izq = resultado["Pata_izq"]
    _malla_der, centro_der = resultado["Pata_der"]
    assert centro_der[0] == pytest.approx(-centro_izq[0])
    assert centro_der[1] == pytest.approx(centro_izq[1])
    assert centro_der[2] == pytest.approx(centro_izq[2])


def test_resolver_ensamble_contacto_a_parte_inexistente():
    partes = [
        Parte("X", "caja", {"ancho": 1, "alto": 1, "profundo": 1},
              contacto="toca:abajo=arriba:NoExiste"),
    ]
    with pytest.raises(ErrorEnsamble):
        resolver_ensamble(partes)


def test_resolver_ensamble_ciclo():
    partes = [
        Parte("A", "caja", {"ancho": 1, "alto": 1, "profundo": 1}, contacto="toca:abajo=arriba:B"),
        Parte("B", "caja", {"ancho": 1, "alto": 1, "profundo": 1}, contacto="toca:abajo=arriba:A"),
    ]
    with pytest.raises(ErrorEnsamble):
        resolver_ensamble(partes)


def test_resolver_ensamble_lista_vacia():
    with pytest.raises(ErrorEnsamble):
        resolver_ensamble([])


def test_fusionar_ensamble_centra_en_origen():
    partes = [
        Parte("Cuerpo", "caja", {"ancho": 10, "alto": 10, "profundo": 10}),
        Parte("Techo", "prisma_triangular", {"ancho": 10, "alto": 5, "profundo": 10},
              contacto="toca:abajo=arriba:Cuerpo"),
    ]
    resultado = resolver_ensamble(partes)
    malla_final = fusionar_ensamble(resultado)

    xs = [v[0] for v in malla_final.vertices]
    ys = [v[1] for v in malla_final.vertices]
    zs = [v[2] for v in malla_final.vertices]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(0.0, abs=1e-9)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(0.0, abs=1e-9)
    assert (min(zs) + max(zs)) / 2 == pytest.approx(0.0, abs=1e-9)


def test_ensamblar_partes_es_determinista():
    def _armar():
        return [
            Parte("Cuerpo", "caja", {"ancho": 10, "alto": 10, "profundo": 10}),
            Parte("Techo", "prisma_triangular", {"ancho": 10, "alto": 5, "profundo": 10},
                  contacto="toca:abajo=arriba:Cuerpo"),
        ]
    m1 = ensamblar_partes(_armar())
    m2 = ensamblar_partes(_armar())
    assert m1.vertices == m2.vertices
    assert m1.caras == m2.caras


# ---------------------------------------------------------------------------
# 9.4 — Integración Pydantic (mockeando la respuesta del LLM, sin Ollama)
# ---------------------------------------------------------------------------

def test_partes_desde_json_llm_valido():
    datos = {
        "factible": True,
        "partes": [
            {"nombre": "Cuerpo", "forma": "caja",
             "dims_cm": {"ancho": 10, "alto": 10, "profundo": 10}, "contacto": None},
            {"nombre": "Techo", "forma": "prisma_triangular",
             "dims_cm": {"ancho": 10, "alto": 5, "profundo": 10},
             "contacto": "toca:abajo=arriba:Cuerpo"},
        ],
    }
    partes = partes_desde_json_llm(datos)
    assert partes is not None
    assert len(partes) == 2
    assert all(isinstance(p, Parte) for p in partes)


def test_partes_desde_json_llm_no_factible():
    datos = {"factible": False, "partes": []}
    assert partes_desde_json_llm(datos) is None


def test_partes_desde_json_llm_forma_invalida():
    datos = {
        "factible": True,
        "partes": [
            {"nombre": "X", "forma": "forma_rara", "dims_cm": {"a": 1}, "contacto": None},
        ],
    }
    assert partes_desde_json_llm(datos) is None


def test_partes_desde_json_llm_json_malformado():
    assert partes_desde_json_llm({"algo": "que no matchea el esquema"}) is None


def test_partes_desde_json_llm_dims_no_numericas():
    datos = {
        "factible": True,
        "partes": [
            {"nombre": "X", "forma": "caja",
             "dims_cm": {"ancho": "no-es-un-numero", "alto": 1, "profundo": 1}, "contacto": None},
        ],
    }
    assert partes_desde_json_llm(datos) is None
