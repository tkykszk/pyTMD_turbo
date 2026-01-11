"""
Tests for pyTMD_turbo Phase 4 functions

Tests the following functions:
- equilibrium_tide / LPET_elevations
- tide_masks
- extrapolate / bilinear
"""

import numpy as np
import pytest
from datetime import datetime, timezone


class TestEquilibriumTide:
    """Tests for equilibrium tide calculations"""

    def test_equilibrium_tide_basic(self):
        """Test basic equilibrium tide calculation"""
        from pyTMD_turbo.predict.equilibrium import equilibrium_tide

        # Days since 1992-01-01
        t = np.array([0.0])
        lat = np.array([45.0])

        lpet = equilibrium_tide(t, lat)

        assert lpet.shape == (1,)
        # Equilibrium tide is typically a few cm
        assert np.abs(lpet[0]) < 0.5  # Less than 50 cm

    def test_equilibrium_tide_time_series(self):
        """Test equilibrium tide with time series"""
        from pyTMD_turbo.predict.equilibrium import equilibrium_tide

        # 30 days
        t = np.arange(0, 30, 1.0)
        lat = np.array([35.0])

        lpet = equilibrium_tide(t, lat)

        assert lpet.shape == (30,)
        # Should have long-period variation (small but non-zero)
        assert np.std(lpet) > 1e-5

    def test_equilibrium_tide_multiple_points(self):
        """Test equilibrium tide with multiple locations"""
        from pyTMD_turbo.predict.equilibrium import equilibrium_tide

        t = np.array([0.0])
        lat = np.array([0.0, 30.0, 60.0, 90.0])

        lpet = equilibrium_tide(t, lat)

        assert lpet.shape == (4, 1)
        # Polar and equatorial should differ
        assert not np.isclose(lpet[0, 0], lpet[3, 0])

    def test_equilibrium_tide_latitude_dependence(self):
        """Test latitude-dependent amplitude"""
        from pyTMD_turbo.predict.equilibrium import equilibrium_tide

        t = np.array([0.0])
        lat_equator = np.array([0.0])
        lat_45 = np.array([45.0])
        lat_pole = np.array([90.0])

        lpet_eq = equilibrium_tide(t, lat_equator)
        lpet_45 = equilibrium_tide(t, lat_45)
        lpet_pole = equilibrium_tide(t, lat_pole)

        # Amplitude should vary with latitude due to P_2^0 function
        # P_2^0(0) = -0.5, P_2^0(45) = 0.25, P_2^0(90) = 1.0
        assert np.abs(lpet_pole[0]) > np.abs(lpet_eq[0])

    def test_equilibrium_tide_mjd_input(self):
        """Test equilibrium tide with MJD input"""
        from pyTMD_turbo.predict.equilibrium import equilibrium_tide

        # MJD values (> 40000)
        mjd = np.array([60000.0, 60001.0, 60002.0])
        lat = np.array([35.0])

        lpet = equilibrium_tide(mjd, lat)

        assert lpet.shape == (3,)


class TestLPETElevations:
    """Tests for LPET_elevations high-level function"""

    def test_lpet_elevations_basic(self):
        """Test basic LPET_elevations call"""
        from pyTMD_turbo.compute import LPET_elevations

        x = np.array([140.0])
        y = np.array([35.0])
        times = np.array(['2024-01-01'], dtype='datetime64')

        lpet = LPET_elevations(x, y, times)

        assert lpet.shape == (1,)

    def test_lpet_elevations_time_series(self):
        """Test LPET_elevations with time series"""
        from pyTMD_turbo.compute import LPET_elevations

        x = np.array([140.0])
        y = np.array([35.0])
        times = np.arange('2024-01-01', '2024-02-01', dtype='datetime64[D]')

        lpet = LPET_elevations(x, y, times)

        assert lpet.shape == (31,)
        # Should have variation over a month (small but non-zero)
        assert np.std(lpet) > 1e-5

    def test_lpet_elevations_datetime_input(self):
        """Test LPET_elevations with datetime input"""
        from pyTMD_turbo.compute import LPET_elevations

        x = np.array([0.0])
        y = np.array([45.0])
        times = np.array([
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 15, tzinfo=timezone.utc),
        ])

        lpet = LPET_elevations(x, y, times)

        assert lpet.shape == (2,)


class TestMeanLongitudes:
    """Tests for mean longitude calculations"""

    def test_mean_longitudes_basic(self):
        """Test mean longitude calculation"""
        from pyTMD_turbo.predict.equilibrium import mean_longitudes

        mjd = np.array([51544.5])  # J2000.0

        longs = mean_longitudes(mjd)

        assert 's' in longs
        assert 'h' in longs
        assert 'p' in longs
        assert 'N' in longs
        assert 'pp' in longs

        # All values should be in [0, 360)
        for key in longs:
            assert 0 <= longs[key][0] < 360

    def test_mean_longitudes_time_variation(self):
        """Test that mean longitudes vary with time"""
        from pyTMD_turbo.predict.equilibrium import mean_longitudes

        mjd = np.array([51544.5, 51545.5, 51546.5])  # 3 days

        longs = mean_longitudes(mjd)

        # Moon's mean longitude should change ~13 deg/day
        s_rate = np.diff(longs['s'])
        assert np.all(np.abs(s_rate) > 10)  # At least 10 deg/day


class TestLegendrePolynomial:
    """Tests for Legendre polynomial calculations"""

    def test_legendre_P20_equator(self):
        """Test P_2^0 at equator"""
        from pyTMD_turbo.predict.equilibrium import legendre_polynomial

        P = legendre_polynomial(np.array([0.0]), l=2, m=0)

        # At equator (theta=90), P_2^0 = -0.5 (unnormalized)
        # Normalized: sqrt(5/4pi) * (-0.5)
        assert P.shape == (1,)

    def test_legendre_P20_pole(self):
        """Test P_2^0 at pole"""
        from pyTMD_turbo.predict.equilibrium import legendre_polynomial

        P = legendre_polynomial(np.array([90.0]), l=2, m=0)

        # At pole (theta=0), P_2^0 = 1 (unnormalized)
        # Normalized: sqrt(5/4pi) * 1
        assert P.shape == (1,)
        assert P[0] > 0

    def test_legendre_P20_symmetry(self):
        """Test P_2^0 hemispheric symmetry"""
        from pyTMD_turbo.predict.equilibrium import legendre_polynomial

        P_north = legendre_polynomial(np.array([45.0]), l=2, m=0)
        P_south = legendre_polynomial(np.array([-45.0]), l=2, m=0)

        # P_2^0 is symmetric about equator
        np.testing.assert_almost_equal(P_north, P_south)


class TestTideMasks:
    """Tests for tide_masks function"""

    def test_tide_masks_import(self):
        """Test that tide_masks can be imported"""
        from pyTMD_turbo.compute import tide_masks

        assert callable(tide_masks)

    def test_tide_masks_signature(self):
        """Test tide_masks function signature"""
        from pyTMD_turbo.compute import tide_masks
        import inspect

        sig = inspect.signature(tide_masks)
        params = list(sig.parameters.keys())

        assert 'x' in params
        assert 'y' in params
        assert 'model' in params


class TestExtrapolate:
    """Tests for extrapolation function"""

    def test_extrapolate_basic(self):
        """Test basic extrapolation"""
        from pyTMD_turbo.interpolate import extrapolate

        # Create simple source grid
        xs = np.linspace(0, 10, 11)
        ys = np.linspace(0, 10, 11)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        zs = np.sin(xs_grid) + np.cos(ys_grid)

        # Output points (inside grid)
        X = np.array([5.0])
        Y = np.array([5.0])

        result = extrapolate(xs, ys, zs, X, Y, is_geographic=False)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_extrapolate_outside_grid(self):
        """Test extrapolation to points outside grid"""
        from pyTMD_turbo.interpolate import extrapolate

        xs = np.linspace(0, 10, 11)
        ys = np.linspace(0, 10, 11)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        zs = np.sin(xs_grid) + np.cos(ys_grid)

        # Output point outside grid
        X = np.array([15.0])
        Y = np.array([5.0])

        result = extrapolate(xs, ys, zs, X, Y, is_geographic=False)

        assert result.shape == (1,)
        # Should extrapolate (nearest neighbor)
        assert np.isfinite(result[0])

    def test_extrapolate_with_cutoff(self):
        """Test extrapolation with distance cutoff"""
        from pyTMD_turbo.interpolate import extrapolate

        xs = np.linspace(0, 10, 11)
        ys = np.linspace(0, 10, 11)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        zs = np.sin(xs_grid) + np.cos(ys_grid)

        # Output point far outside grid
        X = np.array([100.0])
        Y = np.array([5.0])

        # With strict cutoff
        result = extrapolate(xs, ys, zs, X, Y, cutoff=5.0, is_geographic=False)

        # Should be masked (beyond cutoff)
        assert result.mask[0] == True

    def test_extrapolate_masked_source(self):
        """Test extrapolation with masked source data"""
        from pyTMD_turbo.interpolate import extrapolate

        xs = np.linspace(0, 10, 11)
        ys = np.linspace(0, 10, 11)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        zs = np.sin(xs_grid) + np.cos(ys_grid)

        # Mask some source points
        mask = np.zeros_like(zs, dtype=bool)
        mask[5:, 5:] = True
        zs_masked = np.ma.array(zs, mask=mask)

        X = np.array([2.0])
        Y = np.array([2.0])

        result = extrapolate(xs, ys, zs_masked, X, Y, is_geographic=False)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_extrapolate_geographic(self):
        """Test extrapolation with geographic coordinates"""
        from pyTMD_turbo.interpolate import extrapolate

        # Global grid
        xs = np.linspace(-180, 180, 37)
        ys = np.linspace(-90, 90, 19)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        zs = np.cos(np.radians(ys_grid)) * np.sin(np.radians(xs_grid))

        X = np.array([0.0, 180.0])
        Y = np.array([0.0, 0.0])

        result = extrapolate(xs, ys, zs, X, Y, is_geographic=True)

        assert result.shape == (2,)


class TestBilinear:
    """Tests for bilinear interpolation"""

    def test_bilinear_basic(self):
        """Test basic bilinear interpolation"""
        from pyTMD_turbo.interpolate import bilinear

        xs = np.linspace(0, 10, 11)
        ys = np.linspace(0, 10, 11)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        zs = xs_grid + ys_grid

        X = np.array([5.5])
        Y = np.array([5.5])

        result = bilinear(xs, ys, zs, X, Y)

        # Linear function should be exactly interpolated
        np.testing.assert_almost_equal(result[0], 11.0, decimal=5)

    def test_bilinear_multiple_points(self):
        """Test bilinear with multiple output points"""
        from pyTMD_turbo.interpolate import bilinear

        xs = np.linspace(0, 10, 11)
        ys = np.linspace(0, 10, 11)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        zs = xs_grid + ys_grid

        X = np.array([1.5, 5.5, 8.5])
        Y = np.array([1.5, 5.5, 8.5])

        result = bilinear(xs, ys, zs, X, Y)

        assert result.shape == (3,)
        np.testing.assert_almost_equal(result, [3.0, 11.0, 17.0], decimal=5)

    def test_bilinear_outside_grid(self):
        """Test bilinear with points outside grid"""
        from pyTMD_turbo.interpolate import bilinear

        xs = np.linspace(0, 10, 11)
        ys = np.linspace(0, 10, 11)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        zs = xs_grid + ys_grid

        X = np.array([15.0])
        Y = np.array([5.0])

        result = bilinear(xs, ys, zs, X, Y, fill_value=-999.0)

        # Outside grid should get fill_value
        assert result[0] == -999.0


class TestModuleExports:
    """Tests for module exports"""

    def test_predict_module_exports(self):
        """Test that equilibrium functions are exported from predict"""
        from pyTMD_turbo import predict

        assert hasattr(predict, 'equilibrium_tide')
        assert hasattr(predict, 'LPET_elevations')
        assert hasattr(predict, 'mean_longitudes')
        assert hasattr(predict, 'legendre_polynomial')

    def test_compute_module_exports(self):
        """Test that Phase 4 functions are in compute"""
        from pyTMD_turbo import compute

        assert hasattr(compute, 'LPET_elevations')
        assert hasattr(compute, 'tide_masks')

    def test_interpolate_module_exports(self):
        """Test interpolate module exports"""
        from pyTMD_turbo import interpolate

        assert hasattr(interpolate, 'extrapolate')
        assert hasattr(interpolate, 'bilinear')

    def test_main_module_exports(self):
        """Test main module exports"""
        import pyTMD_turbo

        assert hasattr(pyTMD_turbo, 'LPET_elevations')
        assert hasattr(pyTMD_turbo, 'equilibrium_tide')
        assert hasattr(pyTMD_turbo, 'tide_masks')
        assert hasattr(pyTMD_turbo, 'extrapolate')
        assert hasattr(pyTMD_turbo, 'bilinear')
