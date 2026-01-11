"""
test_arguments.py
Tests for nodal corrections and arguments calculations in pyTMD_turbo.constituents

Adapted from pyTMD tests by Tyler Sutterley
"""
import pytest
import numpy as np

import pyTMD_turbo.constituents as constituents


@pytest.mark.parametrize("corrections", ['OTIS', 'GOT'])
def test_arguments(corrections):
    """
    Tests the calculation of nodal corrections for tidal constituents

    Parameters
    ----------
    corrections: str
        Use nodal corrections from OTIS or GOT models
    """
    # Use a random set of modified Julian days
    MJD = np.random.randint(58000, 61000, size=10).astype(float)

    # Set method for astronomical longitudes based on correction type
    if corrections in ('OTIS', 'ATLAS', 'TMD3', 'netcdf'):
        method = 'Cartwright'
    else:
        method = 'ASTRO5'

    # Convert from Modified Julian Dates into Ephemeris Time
    s, h, p, n, pp = constituents.mean_longitudes(MJD, method=method)

    # Number of temporal values
    nt = len(np.atleast_1d(MJD))

    # Initial time conversions
    hour = 24.0 * np.mod(MJD, 1)
    # Convert from hours into degrees
    t1 = 15.0 * hour
    t2 = 30.0 * hour
    # Convert from hours solar time into mean lunar time in degrees
    tau = 15.0 * hour - s + h

    # Major tidal constituents to test
    cindex = ['m2', 's2', 'n2', 'k1', 'o1', 'k2', 'p1', 'q1', 'mf', 'mm']

    # Get nodal corrections
    pu, pf, G = constituents.arguments(MJD, cindex, corrections=corrections)

    # Verify shapes
    assert pu.shape == (nt, len(cindex))
    assert pf.shape == (nt, len(cindex))
    assert G.shape == (nt, len(cindex))

    # Verify amplitude correction factors are reasonable
    # f should be close to 1.0 (typically 0.8-1.2)
    assert np.all(pf > 0.5)
    assert np.all(pf < 1.5)

    # Verify phase corrections are finite
    assert np.isfinite(pu).all()
    assert np.isfinite(G).all()


def test_nodal_m2():
    """
    Test nodal corrections for M2 constituent specifically
    """
    MJD = np.array([55414.0, 55500.0, 55600.0])

    pu, pf, G = constituents.arguments(MJD, ['m2'], corrections='OTIS')

    # M2 amplitude correction should be close to 1.0
    assert np.allclose(pf, 1.0, atol=0.05)

    # M2 phase correction should be small
    assert np.abs(pu).max() < 0.1  # radians


def test_nodal_k1():
    """
    Test nodal corrections for K1 constituent
    """
    MJD = np.array([55414.0, 55500.0, 55600.0])

    pu, pf, G = constituents.arguments(MJD, ['k1'], corrections='OTIS')

    # K1 should have slight amplitude modulation
    assert np.all(pf > 0.9)
    assert np.all(pf < 1.2)


def test_nodal_s2():
    """
    Test nodal corrections for S2 constituent
    S2 should have no nodal corrections (f=1, u=0)
    """
    MJD = np.array([55414.0, 55500.0, 55600.0])

    pu, pf, G = constituents.arguments(MJD, ['s2'], corrections='OTIS')

    # S2 should have no amplitude correction
    assert np.allclose(pf, 1.0)

    # S2 should have no phase correction
    assert np.allclose(pu, 0.0)


def test_table():
    """
    Test that Doodson coefficients match expected values for major constituents
    """
    # Expected Doodson coefficients for M2 (Doodson number 255.555)
    # tau, s, h, p, n, pp, k = 2, 0, 0, 0, 0, 0, 0
    coef_m2 = constituents.coefficients_table('m2')
    assert np.allclose(coef_m2[0], 2)  # tau
    assert np.allclose(coef_m2[1], 0)  # s
    assert np.allclose(coef_m2[2], 0)  # h

    # Expected for S2 (Doodson number 273.555): tau=2, s=2, h=-2
    coef_s2 = constituents.coefficients_table('s2')
    assert np.allclose(coef_s2[0], 2)  # tau
    assert np.allclose(coef_s2[1], 2)  # s
    assert np.allclose(coef_s2[2], -2)  # h

    # Expected for K1 (Doodson number 165.555): tau=1, s=1, h=0, k=1
    coef_k1 = constituents.coefficients_table('k1')
    assert np.allclose(coef_k1[0], 1)  # tau
    assert np.allclose(coef_k1[1], 1)  # s (not 0!)
    assert np.allclose(coef_k1[2], 0)  # h
    assert np.allclose(coef_k1[6], 1)  # k (90 degree phase)


def test_parameters():
    """
    Test that frequencies are computed correctly and are physically reasonable
    """
    # Get frequencies for major constituents
    omega_m2 = constituents.frequency('m2')[0]
    omega_s2 = constituents.frequency('s2')[0]
    omega_k1 = constituents.frequency('k1')[0]
    omega_o1 = constituents.frequency('o1')[0]

    # All frequencies should be positive
    assert omega_m2 > 0
    assert omega_s2 > 0
    assert omega_k1 > 0
    assert omega_o1 > 0

    # Semidiurnal constituents should be approximately twice diurnal
    assert omega_m2 > 1.5 * omega_k1
    assert omega_s2 > 1.5 * omega_k1

    # M2 and S2 should be similar (both semidiurnal)
    assert np.allclose(omega_m2, omega_s2, rtol=0.1)

    # K1 and O1 should be similar (both diurnal)
    assert np.allclose(omega_k1, omega_o1, rtol=0.15)


@pytest.mark.parametrize("constituent", ['m2', 's2', 'n2', 'k1', 'o1', 'p1', 'k2', 'q1'])
def test_major_constituents(constituent):
    """
    Test that major constituents can be processed without errors
    """
    MJD = np.array([55414.0, 55415.0, 55416.0])

    # Should not raise any errors
    pu, pf, G = constituents.arguments(MJD, [constituent], corrections='OTIS')

    # All outputs should be finite
    assert np.isfinite(pu).all()
    assert np.isfinite(pf).all()
    assert np.isfinite(G).all()

    # Amplitude factor should be positive
    assert np.all(pf > 0)


def test_time_variation():
    """
    Test that nodal corrections vary over 18.6 year cycle
    """
    # Create MJD spanning multiple years
    MJD = np.arange(55000, 62000, 100).astype(float)

    pu, pf, G = constituents.arguments(MJD, ['m2', 'k1', 'o1'], corrections='OTIS')

    # pf for M2 should vary over time
    pf_m2 = pf[:, 0]
    assert pf_m2.max() - pf_m2.min() > 0.01  # Should have some variation

    # pf for K1 should have more variation
    pf_k1 = pf[:, 1]
    assert pf_k1.max() - pf_k1.min() > 0.1  # More variation expected

    # pf for O1 should have variation
    pf_o1 = pf[:, 2]
    assert pf_o1.max() - pf_o1.min() > 0.05


def test_equilibrium_phases():
    """
    Test that equilibrium phase (G) increases over time
    """
    MJD = np.array([55414.0, 55414.5, 55415.0, 55415.5])

    _, _, G = constituents.arguments(MJD, ['m2'], corrections='OTIS')

    # Phase should increase (or wrap) over time
    # M2 phase increases by ~29 degrees per hour
    G_m2 = G[:, 0]

    # Check phase differences (accounting for wrapping)
    dG = np.diff(G_m2)
    # Phases should change significantly over 12 hours
    assert np.abs(dG).max() > 100  # Degrees


def test_long_period_constituents():
    """
    Test long-period constituents (Mf, Mm)
    """
    MJD = np.array([55414.0, 55500.0, 55600.0])

    pu, pf, G = constituents.arguments(MJD, ['mf', 'mm'], corrections='OTIS')

    # Mf and Mm should have nodal corrections
    assert np.all(pf > 0.5)
    assert np.all(pf < 1.5)

    # Phases should be finite
    assert np.isfinite(G).all()


def test_consistency_over_time():
    """
    Test that arguments function gives consistent results
    """
    MJD = 55414.0

    # Single value
    pu1, pf1, G1 = constituents.arguments(np.array([MJD]), ['m2', 's2'], corrections='OTIS')

    # Same value in array
    pu2, pf2, G2 = constituents.arguments(np.array([MJD, MJD + 1]), ['m2', 's2'], corrections='OTIS')

    # First values should match
    assert np.allclose(pu1[0], pu2[0])
    assert np.allclose(pf1[0], pf2[0])
    assert np.allclose(G1[0], G2[0])
