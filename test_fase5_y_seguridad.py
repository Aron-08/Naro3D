"""
tests/test_fase5_y_seguridad.py — Fase 5 (operaciones booleanas reales) y
red de seguridad determinística del kernel paramétrico (plantillas +
caja genérica), que reemplazan la caída al pipeline viejo.

Los tests de booleanas requieren trimesh + un backend de boolean ops
("manifold3d"); si no están instalados se saltean en vez de fallar — la
degradación (loguear y conservar el sólido sin el hueco) ya está probada
funcionalmente por el hecho de que `resolver_ensamble` nunca lanza en ese
caso, cosa que sí testeamos siempre.
"""

import math

import pytest

from ensamblador import (
    Parte,
    ErrorEnsamble,
    resolver_ensamble,
    fusionar_ensamble,
    buscar_plantilla_parametrica,
    plantilla_generica_de_piso,
    diagnosticar_malla_cm,
    PLANTILLAS_PARAMETRICAS,
)


def _backend_booleano_disponible() -> bool:
    try:
        import trimesh
        a = trimesh.creation.box(extents=[2, 2, 2])
        b = trimesh.creation.box(extents=[1, 1, 1])
        trimesh.boolean.difference([a, b], engine="manifold")
        return True
    except Exception:
        return False


BOOLEANO_OK = _backend_booleano_disponible()


# ---------------------------------------------------------------------------
# Fase 5 — Parte con operacion="resta"
# ---------------------------------------------------------------------------

def test_parte_resta_requiere_contacto_toca():
    with pytest.raises(ErrorEnsamble):
        Parte("Hueco", "caja", {"ancho": 1, "alto": 1, "profundo": 1}, operacion="resta")


def test_parte_resta_con_simetrica_a_invalido():
    with pytest.raises(ErrorEnsamble):
        Parte("Hueco", "caja", {"ancho": 1, "alto": 1, "profundo": 1},
              contacto="simetrica_a:X", operacion="resta")


def test_parte_operacion_invalida():
    with pytest.raises(ErrorEnsamble):
        Parte("X", "caja", {"ancho": 1, "alto": 1, "profundo": 1}, operacion="multiplicar")


@pytest.mark.skipif(not BOOLEANO_OK, reason="trimesh/backend manifold no disponible en este entorno")
def test_resta_booleana_recorta_volumen_exacto():
    partes = [
        Parte("Pared", "caja", {"ancho": 200, "alto": 200, "profundo": 10}),
        Parte("Ventana", "caja", {"ancho": 60, "alto": 60, "profundo": 20},
              contacto="toca:atras=adelante:Pared", operacion="resta"),
    ]
    resultado = resolver_ensamble(partes)
    assert "Ventana" not in resultado   # el hueco nunca se fusiona como sólido
    assert "Pared" in resultado

    import trimesh
    malla_pared, centro = resultado["Pared"]
    tm = trimesh.Trimesh(vertices=malla_pared.vertices, faces=malla_pared.caras, process=True)
    assert tm.is_watertight
    volumen_esperado = 200 * 200 * 10 - 60 * 60 * 10
    assert tm.volume == pytest.approx(volumen_esperado, rel=1e-6)


@pytest.mark.skipif(not BOOLEANO_OK, reason="trimesh/backend manifold no disponible en este entorno")
def test_resta_booleana_mas_vertices_que_solido_sin_hueco():
    partes = [
        Parte("Pared", "caja", {"ancho": 200, "alto": 200, "profundo": 10}),
        Parte("Ventana", "caja", {"ancho": 60, "alto": 60, "profundo": 20},
              contacto="toca:atras=adelante:Pared", operacion="resta"),
    ]
    resultado = resolver_ensamble(partes)
    malla_pared, _ = resultado["Pared"]
    assert malla_pared.num_vertices() > 8   # un cubo sin hueco tiene 8


def test_resta_sin_backend_no_rompe_el_ensamble(monkeypatch):
    """Si trimesh no está disponible (o falla), el ensamble no debe lanzar:
    se conserva el sólido original sin el hueco."""
    import ensamblador as ens_mod
    monkeypatch.setattr(ens_mod, "_restar_trimesh", lambda *a, **k: None)

    partes = [
        Parte("Pared", "caja", {"ancho": 200, "alto": 200, "profundo": 10}),
        Parte("Ventana", "caja", {"ancho": 60, "alto": 60, "profundo": 20},
              contacto="toca:atras=adelante:Pared", operacion="resta"),
    ]
    resultado = resolver_ensamble(partes)
    assert "Ventana" not in resultado
    malla_pared, _ = resultado["Pared"]
    assert malla_pared.num_vertices() == 8   # sólido intacto, sin el hueco


# ---------------------------------------------------------------------------
# Red de seguridad determinística — reemplaza la caída al pipeline viejo
# ---------------------------------------------------------------------------

def test_buscar_plantilla_casa():
    partes = buscar_plantilla_parametrica("una casa grande de dos pisos")
    assert partes is not None
    nombres = [p.nombre for p in partes]
    assert "Cuerpo" in nombres and "Techo" in nombres


def test_buscar_plantilla_silla_por_sinonimo():
    partes = buscar_plantilla_parametrica("necesito un banco para la cocina")
    assert partes is not None
    assert any(p.nombre == "Asiento" for p in partes)


def test_buscar_plantilla_no_encontrada():
    assert buscar_plantilla_parametrica("xyzzy objeto inexistente 12345") is None


def test_todas_las_plantillas_resuelven_sin_error():
    """Cada entry de PLANTILLAS_PARAMETRICAS debe producir un ensamble
    válido — es la red de seguridad, no puede fallar nunca."""
    vistos = set()
    for clave, fabrica in PLANTILLAS_PARAMETRICAS.items():
        if fabrica in vistos:
            continue
        vistos.add(fabrica)
        partes = fabrica()
        malla = fusionar_ensamble(resolver_ensamble(partes))
        assert malla.num_vertices() > 0, f"plantilla de '{clave}' produjo malla vacía"


def test_plantilla_generica_de_piso_nunca_falla():
    partes = plantilla_generica_de_piso()
    malla = fusionar_ensamble(resolver_ensamble(partes))
    assert malla.num_vertices() == 8   # una caja simple


def test_plantillas_no_comparten_dims_mutables():
    """Cada llamada a la misma plantilla debe devolver Partes independientes
    (dims_cm no compartido) — si no, mutar una `Parte` de un objeto
    afectaría a otro objeto ya creado con la misma plantilla."""
    from ensamblador import PLANTILLAS_PARAMETRICAS as PP
    fabrica = PP["casa"]
    p1 = fabrica()
    p2 = fabrica()
    p1[0].dims_cm["ancho"] = 999999
    assert p2[0].dims_cm["ancho"] != 999999


# ---------------------------------------------------------------------------
# Diagnóstico de malla (sección 8.1 del plan, aplicado a la Malla 3D real)
# ---------------------------------------------------------------------------

def test_diagnostico_malla_vacia():
    from malla import Malla
    avisos = diagnosticar_malla_cm(Malla())
    assert avisos and "vacía" in avisos[0]


def test_diagnostico_malla_valida_sin_avisos_graves():
    partes = [Parte("Cuerpo", "caja", {"ancho": 10, "alto": 10, "profundo": 10})]
    malla = fusionar_ensamble(resolver_ensamble(partes))
    avisos = diagnosticar_malla_cm(malla)
    # Puede haber a lo sumo el aviso de "trimesh no disponible"; nunca debería
    # haber un aviso de watertight/winding para un cubo simple bien formado.
    assert not any("watertight" in a or "winding" in a for a in avisos)
