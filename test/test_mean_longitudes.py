"""
test_mean_longitudes.py
Tests for astronomical mean longitude calculations in pyTMD_turbo.constituents

Adapted from pyTMD test_astro.py by Tyler Sutterley
"""
import numpy as np
import pytest

import pyTMD_turbo.constituents as constituents


def test_mean_longitudes():
    """
    Test that mean longitudes match between Meeus and ASTRO5 methods
    """
    MJD = 55414.0

    # Meeus method from Astronomical Algorithms
    s1, h1, p1, N1, PP1 = constituents.mean_longitudes(MJD, method='Meeus')

    # ASTRO5 methods as implemented by pyTMD
    s2, h2, p2, N2, PP2 = constituents.mean_longitudes(MJD, method='ASTRO5')

    # Methods should give similar results (within a degree, modulo 360)
    assert np.allclose(s1 % 360, s2 % 360, atol=1.0)
    assert np.allclose(h1 % 360, h2 % 360, atol=1.0)
    assert np.allclose(p1 % 360, p2 % 360, atol=1.0)
    assert np.allclose(N1 % 360, N2 % 360, atol=1.0)
    assert np.allclose(PP1 % 360, PP2 % 360, atol=0.5)


def test_cartwright_method():
    """
    Test Cartwright (default) method for mean longitudes
    """
    MJD = 55414.0

    # Cartwright method
    s, h, p, n, pp = constituents.mean_longitudes(MJD, method='Cartwright')

    # Verify output types
    assert not np.isnan(s).any()
    assert not np.isnan(h).any()
    assert not np.isnan(p).any()
    assert not np.isnan(n).any()
    assert not np.isnan(pp).any()

    # Compare with ASTRO5
    s2, h2, p2, n2, pp2 = constituents.mean_longitudes(MJD, method='ASTRO5')

    # Should be relatively close
    assert np.allclose(s % 360, s2 % 360, atol=2.0)
    assert np.allclose(h % 360, h2 % 360, atol=2.0)


def test_mean_longitudes_array():
    """
    Test mean longitude calculations with array inputs
    """
    MJD = np.array([55414.0, 55415.0, 55416.0])

    s, h, p, n, pp = constituents.mean_longitudes(MJD, method='Cartwright')

    # Should return arrays
    assert len(s) == 3
    assert len(h) == 3
    assert len(p) == 3
    assert len(n) == 3
    assert len(pp) == 3


def test_lunar_motion():
    """
    Test that mean longitude of moon increases correctly
    Moon completes one orbit in ~27.3 days
    Daily motion is ~13.2 degrees
    """
    MJD = np.array([55414.0, 55415.0])

    s1, _, _, _, _ = constituents.mean_longitudes(MJD[0:1], method='Cartwright')
    s2, _, _, _, _ = constituents.mean_longitudes(MJD[1:2], method='Cartwright')

    # Daily motion should be ~13.2 degrees
    ds = (s2[0] - s1[0]) % 360
    assert np.allclose(ds, 13.17, atol=0.1)


def test_solar_motion():
    """
    Test that mean longitude of sun increases correctly
    Sun completes one orbit in ~365.25 days
    Daily motion is ~0.99 degrees
    """
    MJD = np.array([55414.0, 55415.0])

    _, h1, _, _, _ = constituents.mean_longitudes(MJD[0:1], method='Cartwright')
    _, h2, _, _, _ = constituents.mean_longitudes(MJD[1:2], method='Cartwright')

    # Daily motion should be ~0.99 degrees
    dh = (h2[0] - h1[0]) % 360
    assert np.allclose(dh, 0.986, atol=0.01)


def test_lunar_perigee_motion():
    """
    Test that mean longitude of lunar perigee increases correctly
    Lunar perigee completes one cycle in ~8.85 years
    Daily motion is ~0.111 degrees
    """
    MJD = np.array([55414.0, 55415.0])

    _, _, p1, _, _ = constituents.mean_longitudes(MJD[0:1], method='Cartwright')
    _, _, p2, _, _ = constituents.mean_longitudes(MJD[1:2], method='Cartwright')

    # Daily motion should be ~0.111 degrees
    dp = (p2[0] - p1[0]) % 360
    assert np.allclose(dp, 0.111, atol=0.01)


def test_lunar_node_motion():
    """
    Test that mean longitude of lunar node decreases correctly
    Lunar node completes one cycle in ~18.61 years (retrograde)
    Daily motion is ~-0.053 degrees
    """
    MJD = np.array([55414.0, 55415.0])

    _, _, _, n1, _ = constituents.mean_longitudes(MJD[0:1], method='Cartwright')
    _, _, _, n2, _ = constituents.mean_longitudes(MJD[1:2], method='Cartwright')

    # Daily motion should be ~-0.053 degrees (retrograde)
    dn = n2[0] - n1[0]
    assert np.allclose(dn, -0.053, atol=0.01)


def test_polynomial_sum():
    """
    Test polynomial evaluation function
    """
    # Test: 1 + 2x + 3x^2 at x = 0, 1, 2
    coef = np.array([1.0, 2.0, 3.0])
    t = np.array([0.0, 1.0, 2.0])

    result = constituents.polynomial_sum(coef, t)

    # Expected: 1, 1+2+3=6, 1+4+12=17
    expected = np.array([1.0, 6.0, 17.0])
    assert np.allclose(result, expected)


def test_polynomial_sum_higher_order():
    """
    Test polynomial evaluation with higher order terms
    """
    # Test: 1 + x + x^2 + x^3 + x^4 at x = 1, 2
    coef = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    t = np.array([1.0, 2.0])

    result = constituents.polynomial_sum(coef, t)

    # Expected: 5, 1+2+4+8+16=31
    expected = np.array([5.0, 31.0])
    assert np.allclose(result, expected)


@pytest.mark.parametrize("method", ['Cartwright', 'Meeus', 'ASTRO5'])
def test_all_methods(method):
    """
    Test all mean longitude calculation methods
    """
    MJD = 55414.0

    s, h, p, n, pp = constituents.mean_longitudes(MJD, method=method)

    # All values should be finite
    assert np.isfinite(s).all()
    assert np.isfinite(h).all()
    assert np.isfinite(p).all()
    assert np.isfinite(n).all()
    assert np.isfinite(pp).all()


def test_reference_epoch():
    """
    Test mean longitudes at J2000 epoch
    """
    # J2000.0 = 2000-01-01T12:00:00 = MJD 51544.5
    MJD = 51544.5

    s, h, p, n, pp = constituents.mean_longitudes(MJD, method='Cartwright')

    # At J2000, using Cartwright coefficients:
    # These are the epoch values
    assert np.allclose(s % 360, 218.3164, atol=0.01)
    assert np.allclose(h % 360, 280.4661, atol=0.01)
    assert np.allclose(p % 360, 83.3535, atol=0.01)
    assert np.allclose(n % 360, 125.0445, atol=0.01)


def test_long_time_range():
    """
    Test mean longitudes over long time range
    """
    # Test over 100 years
    MJD = np.array([51544.5, 51544.5 + 365.25 * 100])

    s, h, p, n, pp = constituents.mean_longitudes(MJD, method='Cartwright')

    # Values should still be finite
    assert np.isfinite(s).all()
    assert np.isfinite(h).all()
    assert np.isfinite(p).all()
    assert np.isfinite(n).all()
    assert np.isfinite(pp).all()
