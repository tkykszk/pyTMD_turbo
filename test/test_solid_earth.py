"""
Tests for pyTMD_turbo solid Earth tide functions

Tests the solid Earth tide displacement calculations including:
- solid_earth_tide(): Core ECEF displacement calculation
- SET_displacements(): High-level wrapper with coordinate conversion
- body_tide(): Spectral/catalog-based method
- Love number functions
"""

import numpy as np
import pytest
from datetime import datetime, timezone


class TestLoveNumbers:
    """Tests for Love number calculations"""

    def test_love_numbers_default(self):
        """Test frequency-dependent Love numbers with default PREM model"""
        from pyTMD_turbo.predict.solid_earth import love_numbers

        # Test semi-diurnal frequency (M2 ~ 1.9 cpd ~ 1.4e-4 rad/s)
        omega_sd = np.array([1.4e-4])
        h2, k2, l2 = love_numbers(omega_sd)

        assert h2.shape == (1,)
        assert 0.60 < h2[0] < 0.62  # Expected range for semi-diurnal
        assert 0.30 < k2[0] < 0.31
        assert 0.084 < l2[0] < 0.086

    def test_love_numbers_diurnal(self):
        """Test Love numbers in diurnal band"""
        from pyTMD_turbo.predict.solid_earth import love_numbers

        # Diurnal frequency (K1 ~ 1.0 cpd ~ 7.3e-5 rad/s)
        omega_d = np.array([7.3e-5])
        h2, k2, l2 = love_numbers(omega_d)

        assert h2.shape == (1,)
        # Diurnal band has resonance effects
        assert 0.55 < h2[0] < 0.65

    def test_love_numbers_long_period(self):
        """Test Love numbers for long-period constituents"""
        from pyTMD_turbo.predict.solid_earth import love_numbers

        # Long-period frequency (Mf ~ 0.07 cpd ~ 5e-6 rad/s)
        omega_lp = np.array([5e-6])
        h2, k2, l2 = love_numbers(omega_lp)

        assert h2.shape == (1,)
        assert 0.60 < h2[0] < 0.62  # Long-period values
        assert 0.29 < k2[0] < 0.30

    def test_complex_love_numbers(self):
        """Test complex Love numbers with anelasticity"""
        from pyTMD_turbo.predict.solid_earth import complex_love_numbers

        omega = np.array([1.4e-4])  # Semi-diurnal
        h2, k2, l2 = complex_love_numbers(omega)

        assert np.iscomplexobj(h2)
        assert h2.real[0] > 0.6  # Real part
        assert h2.imag[0] < 0  # Imaginary part (out-of-phase)


class TestSolidEarthTide:
    """Tests for solid_earth_tide function"""

    def test_solid_earth_tide_basic(self):
        """Test basic solid Earth tide calculation"""
        from pyTMD_turbo.predict.solid_earth import solid_earth_tide

        # Single point, single time
        t = np.array([0.0])  # Days since 1992-01-01
        xyz = np.array([[6378137.0, 0.0, 0.0]])  # On equator
        sun_xyz = np.array([[1.5e11, 0.0, 0.0]])  # Sun on x-axis
        moon_xyz = np.array([[3.84e8, 0.0, 0.0]])  # Moon on x-axis

        dx, dy, dz = solid_earth_tide(t, xyz, sun_xyz, moon_xyz)

        assert dx.shape == (1, 1)
        assert dy.shape == (1, 1)
        assert dz.shape == (1, 1)

        # Displacements should be in reasonable range (cm to dm)
        total_disp = np.sqrt(dx**2 + dy**2 + dz**2)
        assert total_disp[0, 0] < 1.0  # Less than 1 meter

    def test_solid_earth_tide_multiple_times(self):
        """Test solid Earth tide with multiple time steps"""
        from pyTMD_turbo.predict.solid_earth import solid_earth_tide

        t = np.linspace(0, 1, 24)  # 1 day, hourly
        xyz = np.array([[6378137.0, 0.0, 0.0]])

        # Create time-varying ephemeris
        angles = np.linspace(0, 2*np.pi, 24)
        sun_xyz = np.column_stack([
            1.5e11 * np.cos(angles),
            1.5e11 * np.sin(angles),
            np.zeros(24)
        ])
        moon_xyz = np.column_stack([
            3.84e8 * np.cos(angles * 13.4),  # Moon moves faster
            3.84e8 * np.sin(angles * 13.4),
            np.zeros(24)
        ])

        dx, dy, dz = solid_earth_tide(t, xyz, sun_xyz, moon_xyz)

        assert dx.shape == (1, 24)
        assert dy.shape == (1, 24)
        assert dz.shape == (1, 24)

    def test_solid_earth_tide_multiple_points(self):
        """Test solid Earth tide with multiple locations"""
        from pyTMD_turbo.predict.solid_earth import solid_earth_tide

        t = np.array([0.0])

        # Multiple points on equator at different longitudes
        angles = np.linspace(0, np.pi/2, 4)
        xyz = np.column_stack([
            6378137.0 * np.cos(angles),
            6378137.0 * np.sin(angles),
            np.zeros(4)
        ])

        sun_xyz = np.array([[1.5e11, 0.0, 0.0]])
        moon_xyz = np.array([[3.84e8, 0.0, 0.0]])

        dx, dy, dz = solid_earth_tide(t, xyz, sun_xyz, moon_xyz)

        assert dx.shape == (4, 1)

    def test_solid_earth_tide_mean_tide(self):
        """Test mean tide system correction"""
        from pyTMD_turbo.predict.solid_earth import solid_earth_tide

        t = np.array([0.0])
        xyz = np.array([[0.0, 0.0, 6356752.0]])  # At pole
        sun_xyz = np.array([[1.5e11, 0.0, 0.0]])
        moon_xyz = np.array([[3.84e8, 0.0, 0.0]])

        dx_free, dy_free, dz_free = solid_earth_tide(
            t, xyz, sun_xyz, moon_xyz, tide_system='tide_free'
        )
        dx_mean, dy_mean, dz_mean = solid_earth_tide(
            t, xyz, sun_xyz, moon_xyz, tide_system='mean_tide'
        )

        # Mean tide should differ from tide-free
        assert not np.allclose(dz_free, dz_mean)


class TestSETDisplacements:
    """Tests for SET_displacements high-level function"""

    def test_set_displacements_basic(self):
        """Test basic SET_displacements call"""
        from pyTMD_turbo.compute import SET_displacements

        x = np.array([140.0])  # Tokyo longitude
        y = np.array([35.0])   # Tokyo latitude
        times = np.array(['2024-01-01T12:00:00'], dtype='datetime64')

        dn, de, du = SET_displacements(x, y, times)

        assert dn.shape == (1,)
        assert de.shape == (1,)
        assert du.shape == (1,)

        # Displacements should be in reasonable range
        assert np.abs(dn[0]) < 0.5  # Less than 50 cm
        assert np.abs(de[0]) < 0.5
        assert np.abs(du[0]) < 0.5

    def test_set_displacements_time_series(self):
        """Test SET_displacements with time series"""
        from pyTMD_turbo.compute import SET_displacements

        x = np.array([140.0])
        y = np.array([35.0])
        times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')

        dn, de, du = SET_displacements(x, y, times)

        assert dn.shape == (24,)
        assert de.shape == (24,)
        assert du.shape == (24,)

        # Should have variation over the day
        assert np.std(du) > 0.001  # Some variation expected

    def test_set_displacements_multiple_points(self):
        """Test SET_displacements with multiple locations"""
        from pyTMD_turbo.compute import SET_displacements

        x = np.array([140.0, 0.0, -120.0])
        y = np.array([35.0, 51.0, 34.0])
        times = np.array(['2024-01-01T12:00:00'], dtype='datetime64')

        dn, de, du = SET_displacements(x, y, times)

        assert dn.shape == (3, 1)
        assert de.shape == (3, 1)
        assert du.shape == (3, 1)

    def test_set_displacements_cartesian(self):
        """Test SET_displacements with Cartesian output"""
        from pyTMD_turbo.compute import SET_displacements

        x = np.array([0.0])
        y = np.array([0.0])
        times = np.array(['2024-01-01T12:00:00'], dtype='datetime64')

        dx, dy, dz = SET_displacements(
            x, y, times, coordinate_system='cartesian'
        )

        assert dx.shape == (1,)
        assert dy.shape == (1,)
        assert dz.shape == (1,)

    def test_set_displacements_with_height(self):
        """Test SET_displacements with height specification"""
        from pyTMD_turbo.compute import SET_displacements

        x = np.array([140.0])
        y = np.array([35.0])
        h = np.array([1000.0])  # 1 km altitude
        times = np.array(['2024-01-01T12:00:00'], dtype='datetime64')

        dn, de, du = SET_displacements(x, y, times, h=h)

        assert dn.shape == (1,)
        assert de.shape == (1,)
        assert du.shape == (1,)

    def test_set_displacements_datetime_input(self):
        """Test SET_displacements with datetime input"""
        from pyTMD_turbo.compute import SET_displacements

        x = np.array([0.0])
        y = np.array([45.0])
        times = np.array([datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)])

        dn, de, du = SET_displacements(x, y, times)

        assert dn.shape == (1,)


class TestBodyTide:
    """Tests for body_tide spectral method"""

    def test_body_tide_basic(self):
        """Test basic body_tide calculation"""
        from pyTMD_turbo.predict.solid_earth import body_tide

        lat = np.array([35.0])
        lon = np.array([140.0])
        mjd = np.array([60000.0])

        dn, de, du = body_tide(lat, lon, mjd)

        assert dn.shape == (1,)
        assert de.shape == (1,)
        assert du.shape == (1,)

        # Check reasonable magnitude
        assert np.abs(du[0]) < 0.5  # Less than 50 cm

    def test_body_tide_time_series(self):
        """Test body_tide with time series"""
        from pyTMD_turbo.predict.solid_earth import body_tide

        lat = np.array([45.0])
        lon = np.array([0.0])
        mjd = np.linspace(60000.0, 60001.0, 24)  # 1 day hourly

        dn, de, du = body_tide(lat, lon, mjd)

        assert dn.shape == (24,)
        assert de.shape == (24,)
        assert du.shape == (24,)

        # Should have semi-diurnal variation
        assert np.std(du) > 0.001

    def test_body_tide_multiple_points(self):
        """Test body_tide with multiple locations"""
        from pyTMD_turbo.predict.solid_earth import body_tide

        lat = np.array([0.0, 45.0, 90.0])
        lon = np.array([0.0, 0.0, 0.0])
        mjd = np.array([60000.0])

        dn, de, du = body_tide(lat, lon, mjd)

        assert dn.shape == (3, 1)
        assert de.shape == (3, 1)
        assert du.shape == (3, 1)

        # Polar tide should differ from equatorial
        assert not np.isclose(du[0, 0], du[2, 0])

    def test_body_tide_specific_constituents(self):
        """Test body_tide with specific constituents"""
        from pyTMD_turbo.predict.solid_earth import body_tide

        lat = np.array([35.0])
        lon = np.array([140.0])
        mjd = np.linspace(60000.0, 60001.0, 24)

        # M2 only
        dn_m2, de_m2, du_m2 = body_tide(lat, lon, mjd, constituents=['m2'])

        # All constituents
        dn_all, de_all, du_all = body_tide(lat, lon, mjd)

        # M2 should be dominant but all should have more
        assert np.max(np.abs(du_all)) >= np.max(np.abs(du_m2))

    def test_body_tide_catalog(self):
        """Test that body tide catalog has expected constituents"""
        from pyTMD_turbo.predict.solid_earth import _BODY_TIDE_CATALOG

        expected = ['m2', 's2', 'n2', 'k2', 'k1', 'o1', 'p1', 'q1', 'mf', 'mm', 'ssa']
        for const in expected:
            assert const in _BODY_TIDE_CATALOG


class TestECEFToENURotation:
    """Tests for ECEF to ENU rotation matrix"""

    def test_rotation_equator(self):
        """Test rotation at equator"""
        from pyTMD_turbo.predict.solid_earth import ecef_to_enu_rotation

        R = ecef_to_enu_rotation(0.0, 0.0)

        assert R.shape == (3, 3)
        # Should be orthogonal
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_rotation_pole(self):
        """Test rotation at pole"""
        from pyTMD_turbo.predict.solid_earth import ecef_to_enu_rotation

        R = ecef_to_enu_rotation(np.pi/2, 0.0)  # North pole

        assert R.shape == (3, 3)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestModuleExports:
    """Tests for module exports"""

    def test_predict_module_exports(self):
        """Test that solid earth functions are exported from predict module"""
        from pyTMD_turbo import predict

        assert hasattr(predict, 'solid_earth_tide')
        assert hasattr(predict, 'body_tide')
        assert hasattr(predict, 'love_numbers')
        assert hasattr(predict, 'complex_love_numbers')

    def test_compute_module_exports(self):
        """Test that SET_displacements is in compute module"""
        from pyTMD_turbo import compute

        assert hasattr(compute, 'SET_displacements')


class TestConsistency:
    """Tests for consistency between methods"""

    def test_solid_earth_vs_body_tide_order_of_magnitude(self):
        """Test that solid_earth_tide and body_tide give similar orders of magnitude"""
        from pyTMD_turbo.predict.solid_earth import solid_earth_tide, body_tide
        from pyTMD_turbo.spatial import to_cartesian
        from pyTMD_turbo.astro.ephemeris import solar_ecef, lunar_ecef

        # Common parameters
        lat = 35.0
        lon = 140.0
        mjd = 60000.0

        # body_tide
        dn_b, de_b, du_b = body_tide(
            np.array([lat]), np.array([lon]), np.array([mjd])
        )

        # solid_earth_tide
        xyz_x, xyz_y, xyz_z = to_cartesian(
            np.array([lon]), np.array([lat]), np.array([0.0])
        )
        xyz = np.column_stack([xyz_x, xyz_y, xyz_z])

        sun_x, sun_y, sun_z = solar_ecef(np.array([mjd]))
        moon_x, moon_y, moon_z = lunar_ecef(np.array([mjd]))
        sun_xyz = np.column_stack([sun_x, sun_y, sun_z])
        moon_xyz = np.column_stack([moon_x, moon_y, moon_z])

        MJD_1992 = 48622.0
        t = np.array([mjd - MJD_1992])

        dx, dy, dz = solid_earth_tide(t, xyz, sun_xyz, moon_xyz)
        total_disp_set = np.sqrt(dx[0, 0]**2 + dy[0, 0]**2 + dz[0, 0]**2)
        total_disp_bt = np.sqrt(dn_b[0]**2 + de_b[0]**2 + du_b[0]**2)

        # Both should give displacements in cm range
        assert total_disp_set < 1.0  # Less than 1 m
        assert total_disp_bt < 1.0   # Less than 1 m

        # Both should be non-zero
        assert total_disp_set > 0.001
        assert total_disp_bt > 0.001
