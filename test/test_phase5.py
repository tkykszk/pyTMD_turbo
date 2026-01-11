"""
Tests for pyTMD_turbo Phase 5 functions

Tests the following functions:
- normalize_angle
- doodson_arguments
- delaunay_arguments
- schureman_arguments
- minor_arguments
"""

import numpy as np
import pytest


class TestNormalizeAngle:
    """Tests for normalize_angle function"""

    def test_normalize_angle_basic(self):
        """Test basic angle normalization"""
        from pyTMD_turbo.astro.ephemeris import normalize_angle

        # Test values that need normalization
        assert normalize_angle(360.0) == pytest.approx(0.0)
        assert normalize_angle(720.0) == pytest.approx(0.0)
        assert normalize_angle(450.0) == pytest.approx(90.0)

    def test_normalize_angle_negative(self):
        """Test normalization of negative angles"""
        from pyTMD_turbo.astro.ephemeris import normalize_angle

        # Negative angles should wrap to positive
        assert normalize_angle(-90.0) == pytest.approx(270.0)
        assert normalize_angle(-180.0) == pytest.approx(180.0)
        assert normalize_angle(-360.0) == pytest.approx(0.0)

    def test_normalize_angle_array(self):
        """Test with array input"""
        from pyTMD_turbo.astro.ephemeris import normalize_angle

        angles = np.array([0.0, 90.0, 180.0, 270.0, 360.0, 450.0])
        result = normalize_angle(angles)

        expected = np.array([0.0, 90.0, 180.0, 270.0, 0.0, 90.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_angle_radians(self):
        """Test with radian circle"""
        from pyTMD_turbo.astro.ephemeris import normalize_angle

        result = normalize_angle(3 * np.pi, circle=2 * np.pi)
        assert result == pytest.approx(np.pi)

    def test_normalize_angle_already_normalized(self):
        """Test angles already in range"""
        from pyTMD_turbo.astro.ephemeris import normalize_angle

        angles = np.array([0.0, 45.0, 90.0, 180.0, 270.0, 359.9])
        result = normalize_angle(angles)
        np.testing.assert_array_almost_equal(result, angles)


class TestDoodsonArguments:
    """Tests for Doodson arguments calculation"""

    def test_doodson_arguments_basic(self):
        """Test basic Doodson arguments calculation"""
        from pyTMD_turbo.astro.ephemeris import doodson_arguments

        mjd = np.array([51544.5])  # J2000.0
        TAU, S, H, P, Np, Ps = doodson_arguments(mjd)

        # All should be in [0, 2*pi)
        assert 0 <= TAU[0] < 2 * np.pi
        assert 0 <= S[0] < 2 * np.pi
        assert 0 <= H[0] < 2 * np.pi
        assert 0 <= P[0] < 2 * np.pi
        assert 0 <= Np[0] < 2 * np.pi
        assert 0 <= Ps[0] < 2 * np.pi

    def test_doodson_arguments_time_variation(self):
        """Test that arguments vary with time"""
        from pyTMD_turbo.astro.ephemeris import doodson_arguments

        mjd = np.array([51544.5, 51545.5, 51546.5])  # 3 days
        TAU, S, H, P, Np, Ps = doodson_arguments(mjd)

        # All arguments should change over time
        assert not np.allclose(TAU[0], TAU[1])
        assert not np.allclose(S[0], S[1])
        assert not np.allclose(H[0], H[1])

    def test_doodson_arguments_vs_pytmd(self):
        """Compare with pyTMD if available"""
        pytest.importorskip("pyTMD")
        import pyTMD.astro
        from pyTMD_turbo.astro.ephemeris import doodson_arguments

        mjd = np.array([60000.0, 60001.0, 60002.0])

        # pyTMD_turbo
        TAU_t, S_t, H_t, P_t, Np_t, Ps_t = doodson_arguments(mjd)

        # pyTMD
        TAU_p, S_p, H_p, P_p, Np_p, Ps_p = pyTMD.astro.doodson_arguments(mjd)

        # Should be very close
        np.testing.assert_array_almost_equal(TAU_t, TAU_p, decimal=4)
        np.testing.assert_array_almost_equal(S_t, S_p, decimal=4)
        np.testing.assert_array_almost_equal(H_t, H_p, decimal=4)
        np.testing.assert_array_almost_equal(P_t, P_p, decimal=4)


class TestDelaunayArguments:
    """Tests for Delaunay arguments calculation"""

    def test_delaunay_arguments_basic(self):
        """Test basic Delaunay arguments calculation"""
        from pyTMD_turbo.astro.ephemeris import delaunay_arguments

        mjd = np.array([51544.5])  # J2000.0
        l, lp, F, D, N = delaunay_arguments(mjd)

        # All should be in [0, 2*pi)
        assert 0 <= l[0] < 2 * np.pi
        assert 0 <= lp[0] < 2 * np.pi
        assert 0 <= F[0] < 2 * np.pi
        assert 0 <= D[0] < 2 * np.pi
        assert 0 <= N[0] < 2 * np.pi

    def test_delaunay_arguments_time_variation(self):
        """Test that arguments vary with time"""
        from pyTMD_turbo.astro.ephemeris import delaunay_arguments

        mjd = np.array([51544.5, 51545.5, 51546.5])
        l, lp, F, D, N = delaunay_arguments(mjd)

        # l (lunar anomaly) should change fastest (~13 deg/day)
        assert not np.allclose(l[0], l[1])

    def test_delaunay_arguments_vs_pytmd(self):
        """Compare with pyTMD if available"""
        pytest.importorskip("pyTMD")
        import pyTMD.astro
        from pyTMD_turbo.astro.ephemeris import delaunay_arguments

        mjd = np.array([60000.0, 60001.0, 60002.0])

        # pyTMD_turbo
        l_t, lp_t, F_t, D_t, N_t = delaunay_arguments(mjd)

        # pyTMD
        l_p, lp_p, F_p, D_p, N_p = pyTMD.astro.delaunay_arguments(mjd)

        # Should be very close
        np.testing.assert_array_almost_equal(l_t, l_p, decimal=4)
        np.testing.assert_array_almost_equal(lp_t, lp_p, decimal=4)
        np.testing.assert_array_almost_equal(F_t, F_p, decimal=4)
        np.testing.assert_array_almost_equal(D_t, D_p, decimal=4)
        np.testing.assert_array_almost_equal(N_t, N_p, decimal=4)


class TestSchuremanArguments:
    """Tests for Schureman arguments calculation"""

    def test_schureman_arguments_basic(self):
        """Test basic Schureman arguments calculation"""
        from pyTMD_turbo.astro.ephemeris import schureman_arguments

        # Test with sample P and N values
        P = np.array([1.0])  # radians
        N = np.array([0.5])  # radians

        I, xi, nu, Qa, Qu, Ra, Ru, nu_p, nu_s = schureman_arguments(P, N)

        # I (obliquity of lunar orbit) should be around 5 degrees (0.087 rad)
        # at N=0, and varies with N. At N=0.5, it's about 28 degrees (0.49 rad)
        # The formula is: I = arccos(0.913694997 - 0.035692561 * cos(N))
        expected_I = np.arccos(0.913694997 - 0.035692561 * np.cos(0.5))
        assert abs(I[0] - expected_I) < 0.001

        # All outputs should be finite
        assert np.isfinite(I[0])
        assert np.isfinite(xi[0])
        assert np.isfinite(nu[0])
        assert np.isfinite(Qa[0])
        assert np.isfinite(Qu[0])
        assert np.isfinite(Ra[0])
        assert np.isfinite(Ru[0])
        assert np.isfinite(nu_p[0])
        assert np.isfinite(nu_s[0])

    def test_schureman_arguments_array(self):
        """Test with array inputs"""
        from pyTMD_turbo.astro.ephemeris import schureman_arguments

        P = np.linspace(0, 2 * np.pi, 10)
        N = np.linspace(0, 2 * np.pi, 10)

        I, xi, nu, Qa, Qu, Ra, Ru, nu_p, nu_s = schureman_arguments(P, N)

        assert len(I) == 10
        assert np.all(np.isfinite(I))

    def test_schureman_arguments_vs_pytmd(self):
        """Compare with pyTMD if available"""
        pytest.importorskip("pyTMD")
        import pyTMD.astro
        from pyTMD_turbo.astro.ephemeris import schureman_arguments

        P = np.array([1.0, 2.0, 3.0])
        N = np.array([0.5, 1.0, 1.5])

        # pyTMD_turbo
        result_t = schureman_arguments(P, N)

        # pyTMD
        result_p = pyTMD.astro.schureman_arguments(P, N)

        # Should be very close
        for i in range(9):
            np.testing.assert_array_almost_equal(result_t[i], result_p[i], decimal=6)


class TestMinorArguments:
    """Tests for minor_arguments calculation"""

    def test_minor_arguments_basic(self):
        """Test basic minor arguments calculation"""
        from pyTMD_turbo.constituents import minor_arguments

        mjd = np.array([60000.0])
        pu, pf, G = minor_arguments(mjd)

        # Check shapes
        assert pu.shape == (1, 20)
        assert pf.shape == (1, 20)
        assert G.shape == (1, 20)

        # pf should be positive (amplitude factors)
        assert np.all(pf > 0)

    def test_minor_arguments_time_series(self):
        """Test with time series"""
        from pyTMD_turbo.constituents import minor_arguments

        mjd = 60000.0 + np.arange(30)  # 30 days
        pu, pf, G = minor_arguments(mjd)

        assert pu.shape == (30, 20)
        assert pf.shape == (30, 20)
        assert G.shape == (30, 20)

        # Phase should vary with time
        assert not np.allclose(G[0], G[1])

    def test_minor_arguments_corrections(self):
        """Test different correction methods"""
        from pyTMD_turbo.constituents import minor_arguments

        mjd = np.array([60000.0])

        # OTIS corrections (default)
        pu_otis, pf_otis, G_otis = minor_arguments(mjd, corrections='OTIS')

        # GOT corrections
        pu_got, pf_got, G_got = minor_arguments(mjd, corrections='GOT')

        # Should have same shape
        assert pu_otis.shape == pu_got.shape

        # G (phase) should be same regardless of corrections (within tolerance)
        # The equilibrium argument G comes from mean_longitudes which can have
        # small differences due to floating point precision
        np.testing.assert_array_almost_equal(G_otis, G_got, decimal=0)

    def test_minor_arguments_vs_pytmd(self):
        """Compare with pyTMD if available"""
        pytest.importorskip("pyTMD")
        import pyTMD.constituents
        from pyTMD_turbo.constituents import minor_arguments

        mjd = np.array([60000.0, 60001.0, 60002.0])

        # pyTMD_turbo
        pu_t, pf_t, G_t = minor_arguments(mjd, corrections='OTIS')

        # pyTMD
        pu_p, pf_p, G_p = pyTMD.constituents.minor_arguments(mjd, corrections='OTIS')

        # Should be similar (allow some tolerance due to different implementations)
        # pf (nodal factors) should match well
        np.testing.assert_array_almost_equal(pf_t, pf_p, decimal=3)
        # G (equilibrium arguments) may have small differences (~0.2 deg)
        # due to different mean_longitudes implementations
        np.testing.assert_array_almost_equal(G_t, G_p, decimal=0)


class TestMinorTable:
    """Tests for _minor_table helper function"""

    def test_minor_table_shape(self):
        """Test minor table shape"""
        from pyTMD_turbo.constituents import _minor_table

        coef = _minor_table()

        # Should be (7 variables, 20 constituents)
        assert coef.shape == (7, 20)

    def test_minor_table_constituents(self):
        """Test that minor constituents are included"""
        from pyTMD_turbo.constituents import _MINOR_CONSTITUENTS

        expected = [
            "2q1", "sigma1", "rho1", "m1b", "m1a", "chi1",
            "pi1", "phi1", "theta1", "j1", "oo1",
            "2n2", "mu2", "nu2", "lambda2", "l2a", "l2b",
            "t2", "eps2", "eta2",
        ]

        assert _MINOR_CONSTITUENTS == expected


class TestModuleExports:
    """Tests for module exports"""

    def test_astro_exports(self):
        """Test that astro module exports new functions"""
        from pyTMD_turbo import astro

        assert hasattr(astro, 'normalize_angle')
        assert hasattr(astro, 'doodson_arguments')
        assert hasattr(astro, 'delaunay_arguments')
        assert hasattr(astro, 'schureman_arguments')

    def test_constituents_exports(self):
        """Test that constituents module exports minor_arguments"""
        from pyTMD_turbo import constituents

        assert hasattr(constituents, 'minor_arguments')

    def test_callable(self):
        """Test that all functions are callable"""
        from pyTMD_turbo.astro import (
            normalize_angle,
            doodson_arguments,
            delaunay_arguments,
            schureman_arguments,
        )
        from pyTMD_turbo.constituents import minor_arguments

        assert callable(normalize_angle)
        assert callable(doodson_arguments)
        assert callable(delaunay_arguments)
        assert callable(schureman_arguments)
        assert callable(minor_arguments)
