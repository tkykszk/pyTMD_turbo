"""
test_constituents.py
Tests for pyTMD_turbo.constituents module

Adapted from pyTMD tests by Tyler Sutterley
"""
import json
import numpy as np
import pytest
from pathlib import Path

import pyTMD_turbo.constituents as constituents


def test_coefficients_table():
    """
    Test retrieval of Doodson coefficients for various constituents
    """
    # Major constituents to test
    cindex = ['m2', 's2', 'n2', 'k1', 'o1', 'p1', 'k2', 'q1', 'mf', 'mm', 'ssa']

    for c in cindex:
        coef = constituents.coefficients_table(c)
        assert coef.shape == (7, 1)
        # Test that the coefficients are numeric
        assert not np.isnan(coef).any()


def test_coefficients_table_multiple():
    """
    Test retrieval of Doodson coefficients for multiple constituents
    """
    clist = ['m2', 's2', 'n2', 'k1', 'o1']
    coef = constituents.coefficients_table(clist)
    assert coef.shape == (7, 5)
    # Verify each column is different
    for i in range(5):
        for j in range(i+1, 5):
            assert not np.allclose(coef[:, i], coef[:, j])


def test_coefficients_unsupported():
    """
    Test that unsupported constituents raise ValueError
    """
    with pytest.raises(ValueError):
        constituents.coefficients_table('invalid_constituent')


def test_frequency():
    """
    Test calculation of angular frequencies
    """
    # Test individual frequencies
    omega_m2 = constituents.frequency('m2')[0]
    omega_s2 = constituents.frequency('s2')[0]
    omega_k1 = constituents.frequency('k1')[0]
    omega_o1 = constituents.frequency('o1')[0]

    # All frequencies should be positive and in reasonable range
    # Semidiurnal: ~1.4e-4 to 1.6e-4 rad/s (10-13 hour periods)
    assert 1.3e-4 < omega_m2 < 1.6e-4
    assert 1.3e-4 < omega_s2 < 1.6e-4

    # Diurnal: ~7e-5 to 8e-5 rad/s (22-26 hour periods)
    assert 6.5e-5 < omega_k1 < 8.5e-5
    assert 6.0e-5 < omega_o1 < 7.5e-5

    # Verify diurnal < semidiurnal
    assert omega_k1 < omega_m2
    assert omega_o1 < omega_m2


def test_frequency_multiple():
    """
    Test frequency calculation for multiple constituents
    """
    clist = ['m2', 's2', 'n2', 'k1', 'o1', 'p1']
    omega = constituents.frequency(clist)
    assert len(omega) == 6

    # All frequencies should be positive
    assert (omega > 0).all()

    # Diurnal constituents should have lower frequencies
    assert omega[3] < omega[0]  # k1 < m2
    assert omega[4] < omega[0]  # o1 < m2
    assert omega[5] < omega[0]  # p1 < m2


def test_mean_longitudes_cartwright():
    """
    Test mean longitude calculations with Cartwright method
    """
    MJD = 55414.0  # Reference MJD

    s, h, p, n, pp = constituents.mean_longitudes(MJD, method='Cartwright')

    # Verify all outputs are scalars
    assert np.isscalar(s) or len(s) == 1
    assert np.isscalar(h) or len(h) == 1
    assert np.isscalar(p) or len(p) == 1
    assert np.isscalar(n) or len(n) == 1
    assert np.isscalar(pp) or len(pp) == 1


def test_mean_longitudes_meeus():
    """
    Test mean longitude calculations with Meeus method
    """
    MJD = 55414.0

    s, h, p, n, pp = constituents.mean_longitudes(MJD, method='Meeus')

    # Verify output ranges are reasonable (degrees)
    # Mean longitudes should cycle between 0-360
    assert not np.isnan(s).any()
    assert not np.isnan(h).any()
    assert not np.isnan(p).any()
    assert not np.isnan(n).any()
    assert not np.isnan(pp).any()


def test_mean_longitudes_astro5():
    """
    Test mean longitude calculations with ASTRO5 method
    """
    MJD = 55414.0

    s1, h1, p1, n1, pp1 = constituents.mean_longitudes(MJD, method='Meeus')
    s2, h2, p2, n2, pp2 = constituents.mean_longitudes(MJD, method='ASTRO5')

    # Meeus and ASTRO5 should give similar results (modulo 360)
    assert np.allclose(s1 % 360, s2 % 360, atol=1.0)  # Within 1 degree
    assert np.allclose(h1 % 360, h2 % 360, atol=1.0)
    assert np.allclose(p1 % 360, p2 % 360, atol=1.0)
    assert np.allclose(n1 % 360, n2 % 360, atol=1.0)


def test_mean_longitudes_array():
    """
    Test mean longitude calculations with array input
    """
    MJD = np.array([55414.0, 55415.0, 55416.0])

    s, h, p, n, pp = constituents.mean_longitudes(MJD, method='Cartwright')

    # Verify array output
    assert len(s) == 3
    assert len(h) == 3
    assert len(p) == 3
    assert len(n) == 3
    assert len(pp) == 3

    # Mean longitude of moon increases by ~13.2 degrees per day
    ds = (s[1] - s[0]) % 360
    assert 12.0 < ds < 15.0


def test_nodal_modulation():
    """
    Test nodal modulation calculations
    """
    n = np.array([125.0, 130.0, 135.0])  # Lunar node longitude (degrees)
    p = np.array([83.0, 84.0, 85.0])     # Lunar perigee longitude (degrees)

    u, f = constituents.nodal_modulation(n, p, ['m2', 's2', 'k1'], corrections='OTIS')

    # u shape: (n_times, n_constituents)
    assert u.shape == (3, 3)
    assert f.shape == (3, 3)

    # f should be close to 1.0
    assert (f > 0.8).all()
    assert (f < 1.2).all()

    # u should be small (in radians)
    assert np.abs(u).max() < 0.5


def test_arguments():
    """
    Test arguments calculation for nodal corrections
    """
    MJD = np.array([55414.0, 55415.0])
    clist = ['m2', 's2', 'n2', 'k1', 'o1']

    pu, pf, G = constituents.arguments(MJD, clist, corrections='OTIS')

    # Check shapes
    assert pu.shape == (2, 5)
    assert pf.shape == (2, 5)
    assert G.shape == (2, 5)

    # pf should be close to 1.0
    assert (pf > 0.8).all()
    assert (pf < 1.2).all()


@pytest.mark.parametrize("corrections", ['OTIS', 'GOT'])
def test_arguments_corrections(corrections):
    """
    Test arguments with different correction types
    """
    MJD = np.array([55414.0, 55415.0, 55416.0])
    clist = ['m2', 's2', 'k1', 'o1']

    pu, pf, G = constituents.arguments(MJD, clist, corrections=corrections)

    # Verify output shapes
    assert pu.shape == (3, 4)
    assert pf.shape == (3, 4)
    assert G.shape == (3, 4)

    # All values should be finite
    assert np.isfinite(pu).all()
    assert np.isfinite(pf).all()
    assert np.isfinite(G).all()


def test_polynomial_sum():
    """
    Test polynomial evaluation
    """
    coef = np.array([1.0, 2.0, 3.0])  # 1 + 2t + 3t^2
    t = np.array([0.0, 1.0, 2.0])

    result = constituents.polynomial_sum(coef, t)

    expected = np.array([1.0, 6.0, 17.0])  # 1, 1+2+3, 1+4+12
    assert np.allclose(result, expected)


def test_doodson_json_loaded():
    """
    Test that doodson.json is properly loaded
    """
    data_path = Path(__file__).parent.parent / "pyTMD_turbo" / "data" / "doodson.json"

    with open(data_path, 'r', encoding='utf-8') as f:
        coefficients = json.load(f)

    # Major constituents should be present
    major = ['m2', 's2', 'n2', 'k1', 'o1', 'p1', 'k2', 'q1', 'mf', 'mm']
    for c in major:
        assert c in coefficients, f"Missing constituent: {c}"
        # Each constituent should have 7 coefficients
        assert len(coefficients[c]) == 7, f"Wrong coefficient count for {c}"
