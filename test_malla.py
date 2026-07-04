"""
tests/test_malla.py — Tests unitarios del catálogo de fábricas de malla.py
(sección 9.1 del plan de kernel paramétrico). No requieren Ollama corriendo:
son geometría pura.
"""

import math

import pytest

import malla as malla_mod


# ---------------------------------------------------------------------------
# Caja / cubo
# ---------------------------------------------------------------------------

def test_malla_cubo_dimensiones():
    m = malla_mod.malla_cubo(10, 20, 30)
    xs = [v[0] for v in m.vertices]
    ys = [v[1] for v in m.vertices]
    zs = [v[2] for v in m.vertices]
    assert max(xs) - min(xs) == pytest.approx(10)
    assert max(ys) - min(ys) == pytest.approx(20)
    assert max(zs) - min(zs) == pytest.approx(30)


def test_malla_cubo_centrada_en_origen():
    m = malla_mod.malla_cubo(10, 20, 30)
    xs = [v[0] for v in m.vertices]
    ys = [v[1] for v in m.vertices]
    zs = [v[2] for v in m.vertices]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(0.0)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(0.0)
    assert (min(zs) + max(zs)) / 2 == pytest.approx(0.0)


def test_malla_cubo_radio_bounding():
    m = malla_mod.malla_cubo(10, 10, 10)
    # radio de la esfera mínima que contiene el cubo = mitad de la diagonal
    assert m.radio_bounding() == pytest.approx(math.sqrt(3 * 25))


def test_malla_cubo_numero_vertices_caras():
    m = malla_mod.malla_cubo(1, 1, 1)
    assert m.num_vertices() == 8
    assert m.num_caras() == 12   # 6 caras cuadradas x 2 triángulos


# ---------------------------------------------------------------------------
# Esfera
# ---------------------------------------------------------------------------

def test_malla_esfera_radio_bounding():
    m = malla_mod.malla_esfera(5.0)
    assert m.radio_bounding() == pytest.approx(5.0, rel=1e-6)


def test_malla_esfera_vertices_en_la_superficie():
    r = 7.5
    m = malla_mod.malla_esfera(r)
    for x, y, z in m.vertices:
        assert math.sqrt(x*x + y*y + z*z) == pytest.approx(r, rel=1e-6)


# ---------------------------------------------------------------------------
# Cilindro
# ---------------------------------------------------------------------------

def test_malla_cilindro_dimensiones():
    m = malla_mod.malla_cilindro(3.0, 12.0)
    ys = [v[1] for v in m.vertices]
    assert max(ys) - min(ys) == pytest.approx(12.0)
    radios = [math.hypot(v[0], v[2]) for v in m.vertices]
    assert max(radios) == pytest.approx(3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Prisma triangular (techo a dos aguas)
# ---------------------------------------------------------------------------

def test_malla_prisma_triangular_dimensiones():
    m = malla_mod.malla_prisma_triangular(40, 15, 30)
    xs = [v[0] for v in m.vertices]
    ys = [v[1] for v in m.vertices]
    zs = [v[2] for v in m.vertices]
    assert max(xs) - min(xs) == pytest.approx(40)
    assert max(ys) - min(ys) == pytest.approx(15)
    assert max(zs) - min(zs) == pytest.approx(30)


def test_malla_prisma_triangular_base_en_y_cero():
    m = malla_mod.malla_prisma_triangular(10, 5, 10)
    ys = [v[1] for v in m.vertices]
    assert min(ys) == pytest.approx(0.0)
    assert max(ys) == pytest.approx(5.0)


def test_malla_prisma_triangular_num_caras():
    m = malla_mod.malla_prisma_triangular(10, 5, 10)
    assert m.num_vertices() == 6
    assert m.num_caras() == 8   # 2 base + 6 laterales


# ---------------------------------------------------------------------------
# Tubo (cilindro hueco)
# ---------------------------------------------------------------------------

def test_malla_tubo_radios():
    m = malla_mod.malla_tubo(r_ext=5.0, r_int=3.0, alto=10.0)
    radios = [math.hypot(v[0], v[2]) for v in m.vertices]
    assert max(radios) == pytest.approx(5.0, rel=1e-6)
    assert min(radios) == pytest.approx(3.0, rel=1e-6)


def test_malla_tubo_altura():
    m = malla_mod.malla_tubo(r_ext=5.0, r_int=3.0, alto=20.0)
    ys = [v[1] for v in m.vertices]
    assert max(ys) - min(ys) == pytest.approx(20.0)


def test_malla_tubo_radio_interno_invalido_se_clampa():
    # r_int >= r_ext no debe degenerar en un cilindro macizo con NaN/crash
    m = malla_mod.malla_tubo(r_ext=5.0, r_int=5.0, alto=10.0)
    radios = [math.hypot(v[0], v[2]) for v in m.vertices]
    assert min(radios) < 5.0
    assert m.num_caras() > 0


# ---------------------------------------------------------------------------
# PX_POR_CM — constante de escala única (sección 5.2 del plan)
# ---------------------------------------------------------------------------

def test_px_por_cm_definida_y_positiva():
    assert malla_mod.PX_POR_CM > 0
