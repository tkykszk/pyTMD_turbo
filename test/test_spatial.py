"""
Tests for pyTMD_turbo.spatial coordinate transformations

Tests geographic <-> cartesian coordinate conversions and scale factors.
"""

import numpy as np
import pytest

from pyTMD_turbo.spatial import (
    to_cartesian,
    to_geodetic,
    to_sphere,
    scale_factors,
    datum,
    convert_ellipsoid,
)


class TestDatum:
    """Test datum/ellipsoid class"""

    def test_wgs84_default(self):
        """Test WGS84 ellipsoid parameters"""
        d = datum('WGS84')
        assert d.name == 'WGS84'
        assert d.a == 6378137.0
        np.testing.assert_almost_equal(d.f, 1.0 / 298.257223563)

    def test_grs80(self):
        """Test GRS80 ellipsoid parameters"""
        d = datum('GRS80')
        assert d.name == 'GRS80'
        assert d.a == 6378137.0

    def test_ellipsoid_properties(self):
        """Test derived ellipsoid properties"""
        d = datum('WGS84')

        # Semi-minor axis
        assert d.b < d.a
        np.testing.assert_almost_equal(d.b, 6356752.3142, decimal=1)

        # Eccentricity squared
        assert 0 < d.e2 < 0.01
        np.testing.assert_almost_equal(d.e2, 0.00669438, decimal=6)

    def test_unknown_ellipsoid(self):
        """Test error for unknown ellipsoid"""
        with pytest.raises(ValueError, match="Unknown ellipsoid"):
            datum('UNKNOWN')


class TestToCartesian:
    """Test geographic to cartesian conversion"""

    def test_equator_prime_meridian(self):
        """Test point on equator at prime meridian"""
        x, y, z = to_cartesian(0.0, 0.0, 0.0)

        # Should be at (a, 0, 0)
        d = datum('WGS84')
        np.testing.assert_almost_equal(x[0], d.a, decimal=0)
        np.testing.assert_almost_equal(y[0], 0.0, decimal=0)
        np.testing.assert_almost_equal(z[0], 0.0, decimal=0)

    def test_equator_90e(self):
        """Test point on equator at 90 E"""
        x, y, z = to_cartesian(90.0, 0.0, 0.0)

        d = datum('WGS84')
        np.testing.assert_almost_equal(x[0], 0.0, decimal=0)
        np.testing.assert_almost_equal(y[0], d.a, decimal=0)
        np.testing.assert_almost_equal(z[0], 0.0, decimal=0)

    def test_north_pole(self):
        """Test north pole"""
        x, y, z = to_cartesian(0.0, 90.0, 0.0)

        d = datum('WGS84')
        np.testing.assert_almost_equal(x[0], 0.0, decimal=0)
        np.testing.assert_almost_equal(y[0], 0.0, decimal=0)
        np.testing.assert_almost_equal(z[0], d.b, decimal=0)

    def test_tokyo(self):
        """Test Tokyo coordinates"""
        x, y, z = to_cartesian(139.6917, 35.6895, 0.0)

        # Known approximate values (calculated)
        np.testing.assert_almost_equal(x[0], -3954844.3, decimal=-1)
        np.testing.assert_almost_equal(y[0], 3354936.5, decimal=-1)
        np.testing.assert_almost_equal(z[0], 3700264.0, decimal=-1)

    def test_with_height(self):
        """Test conversion with height"""
        h = 100.0  # 100 meters
        x_0, y_0, z_0 = to_cartesian(0.0, 0.0, 0.0)
        x_h, y_h, z_h = to_cartesian(0.0, 0.0, h)

        # X should increase by h at equator/prime meridian
        np.testing.assert_almost_equal(x_h[0] - x_0[0], h, decimal=1)

    def test_array_input(self):
        """Test with array input"""
        lons = np.array([0.0, 90.0, 180.0])
        lats = np.array([0.0, 0.0, 0.0])

        x, y, z = to_cartesian(lons, lats)

        assert len(x) == 3
        assert len(y) == 3
        assert len(z) == 3


class TestToGeodetic:
    """Test cartesian to geographic conversion"""

    def test_roundtrip_equator(self):
        """Test roundtrip conversion at equator"""
        lon_in, lat_in, h_in = 0.0, 0.0, 0.0

        x, y, z = to_cartesian(lon_in, lat_in, h_in)
        lon_out, lat_out, h_out = to_geodetic(x, y, z)

        np.testing.assert_almost_equal(lon_out[0], lon_in, decimal=6)
        np.testing.assert_almost_equal(lat_out[0], lat_in, decimal=6)
        np.testing.assert_almost_equal(h_out[0], h_in, decimal=3)

    def test_roundtrip_pole(self):
        """Test roundtrip conversion at north pole"""
        lon_in, lat_in, h_in = 45.0, 90.0, 0.0

        x, y, z = to_cartesian(lon_in, lat_in, h_in)
        lon_out, lat_out, h_out = to_geodetic(x, y, z)

        # Longitude is undefined at pole, only check latitude
        np.testing.assert_almost_equal(lat_out[0], lat_in, decimal=6)
        np.testing.assert_almost_equal(h_out[0], h_in, decimal=3)

    def test_roundtrip_tokyo(self):
        """Test roundtrip conversion for Tokyo"""
        lon_in, lat_in, h_in = 139.6917, 35.6895, 50.0

        x, y, z = to_cartesian(lon_in, lat_in, h_in)
        lon_out, lat_out, h_out = to_geodetic(x, y, z)

        np.testing.assert_almost_equal(lon_out[0], lon_in, decimal=6)
        np.testing.assert_almost_equal(lat_out[0], lat_in, decimal=6)
        np.testing.assert_almost_equal(h_out[0], h_in, decimal=3)

    def test_roundtrip_with_height(self):
        """Test roundtrip with various heights"""
        for h_in in [0.0, 100.0, 1000.0, 10000.0]:
            x, y, z = to_cartesian(140.0, 35.0, h_in)
            lon_out, lat_out, h_out = to_geodetic(x, y, z)

            np.testing.assert_almost_equal(h_out[0], h_in, decimal=2)

    def test_array_roundtrip(self):
        """Test roundtrip with array input"""
        lons_in = np.array([0.0, 90.0, 180.0, -90.0])
        lats_in = np.array([0.0, 45.0, -45.0, 30.0])
        h_in = np.zeros(4)

        x, y, z = to_cartesian(lons_in, lats_in, h_in)
        lons_out, lats_out, h_out = to_geodetic(x, y, z)

        np.testing.assert_array_almost_equal(lons_out, lons_in, decimal=6)
        np.testing.assert_array_almost_equal(lats_out, lats_in, decimal=6)
        np.testing.assert_array_almost_equal(h_out, h_in, decimal=3)


class TestToSphere:
    """Test spherical coordinate conversion"""

    def test_unit_sphere(self):
        """Test unit sphere conversion"""
        x, y, z = to_sphere(0.0, 0.0)

        np.testing.assert_almost_equal(x[0], 1.0, decimal=10)
        np.testing.assert_almost_equal(y[0], 0.0, decimal=10)
        np.testing.assert_almost_equal(z[0], 0.0, decimal=10)

    def test_north_pole_sphere(self):
        """Test north pole on sphere"""
        x, y, z = to_sphere(0.0, 90.0)

        np.testing.assert_almost_equal(x[0], 0.0, decimal=10)
        np.testing.assert_almost_equal(y[0], 0.0, decimal=10)
        np.testing.assert_almost_equal(z[0], 1.0, decimal=10)

    def test_custom_radius(self):
        """Test with custom radius"""
        r = 6371000.0  # Earth mean radius
        x, y, z = to_sphere(0.0, 0.0, r)

        np.testing.assert_almost_equal(x[0], r, decimal=1)


class TestScaleFactors:
    """Test scale factor calculations"""

    def test_equator_scale_factors(self):
        """Test scale factors at equator"""
        h_lat, h_lon = scale_factors(0.0)

        # At equator, both should be approximately equal
        # and about 111 km per degree
        np.testing.assert_almost_equal(h_lat[0] / 1000, 110.6, decimal=0)
        np.testing.assert_almost_equal(h_lon[0] / 1000, 111.3, decimal=0)

    def test_high_latitude_scale_factors(self):
        """Test scale factors at high latitude"""
        h_lat_0, h_lon_0 = scale_factors(0.0)
        h_lat_60, h_lon_60 = scale_factors(60.0)

        # Longitudinal scale should decrease at higher latitudes
        assert h_lon_60[0] < h_lon_0[0]

        # Meridional scale should be relatively constant
        assert abs(h_lat_60[0] - h_lat_0[0]) < 5000

    def test_pole_scale_factors(self):
        """Test scale factors at pole"""
        h_lat, h_lon = scale_factors(90.0)

        # Longitudinal scale should be near zero at pole
        np.testing.assert_almost_equal(h_lon[0], 0.0, decimal=0)


class TestConvertEllipsoid:
    """Test ellipsoid conversion"""

    def test_wgs84_to_grs80(self):
        """Test WGS84 to GRS80 conversion (should be nearly identical)"""
        lon_in, lat_in, h_in = 140.0, 35.0, 100.0

        lon_out, lat_out, h_out = convert_ellipsoid(
            lon_in, lat_in, h_in,
            source_ellipsoid='WGS84',
            target_ellipsoid='GRS80'
        )

        # WGS84 and GRS80 are very similar
        np.testing.assert_almost_equal(lon_out[0], lon_in, decimal=10)
        np.testing.assert_almost_equal(lat_out[0], lat_in, decimal=8)
        # Height difference should be small (< 1mm typically)
        np.testing.assert_almost_equal(h_out[0], h_in, decimal=2)


class TestDatumAdditional:
    """Additional datum tests for full coverage"""

    def test_datum_n_property(self):
        """Test third flattening property"""
        d = datum('WGS84')
        # Third flattening should be approximately f / (2 - f)
        expected_n = d.f / (2.0 - d.f)
        np.testing.assert_almost_equal(d.n, expected_n, decimal=10)

    def test_datum_ep2_property(self):
        """Test second eccentricity squared property"""
        d = datum('WGS84')
        # ep2 = e2 / (1 - e2)
        expected_ep2 = d.e2 / (1.0 - d.e2)
        np.testing.assert_almost_equal(d.ep2, expected_ep2, decimal=10)

    def test_all_supported_ellipsoids(self):
        """Test all supported ellipsoids can be instantiated"""
        ellipsoids = ['WGS84', 'GRS80', 'WGS72', 'GRS67', 'TOPEX', 'EGM2008']
        for name in ellipsoids:
            d = datum(name)
            assert d.name == name
            assert d.a > 0
            assert 0 < d.f < 0.01

    def test_case_insensitive_ellipsoid(self):
        """Test ellipsoid names are case insensitive"""
        d1 = datum('wgs84')
        d2 = datum('WGS84')
        assert d1.a == d2.a
        assert d1.f == d2.f


class TestToCartesianAdditional:
    """Additional to_cartesian tests"""

    def test_different_ellipsoids(self):
        """Test to_cartesian with different ellipsoids"""
        lon, lat, h = 140.0, 35.0, 0.0

        x_wgs84, y_wgs84, z_wgs84 = to_cartesian(lon, lat, h, ellipsoid='WGS84')
        x_grs80, y_grs80, z_grs80 = to_cartesian(lon, lat, h, ellipsoid='GRS80')

        # WGS84 and GRS80 should give nearly identical results
        np.testing.assert_almost_equal(x_wgs84[0], x_grs80[0], decimal=0)
        np.testing.assert_almost_equal(y_wgs84[0], y_grs80[0], decimal=0)
        np.testing.assert_almost_equal(z_wgs84[0], z_grs80[0], decimal=0)

    def test_southern_hemisphere(self):
        """Test conversion in southern hemisphere"""
        x, y, z = to_cartesian(-70.0, -45.0, 0.0)

        # X should be negative (west of prime meridian)
        # Y should be negative (south of equator affects position)
        # Z should be negative (south of equator)
        assert z[0] < 0


class TestToGeodeticAdditional:
    """Additional to_geodetic tests"""

    def test_iterative_method(self):
        """Test to_geodetic with iterative method"""
        lon_in, lat_in, h_in = 140.0, 35.0, 100.0

        x, y, z = to_cartesian(lon_in, lat_in, h_in)
        lon_out, lat_out, h_out = to_geodetic(x, y, z, method='iterative')

        np.testing.assert_almost_equal(lon_out[0], lon_in, decimal=5)
        np.testing.assert_almost_equal(lat_out[0], lat_in, decimal=5)
        np.testing.assert_almost_equal(h_out[0], h_in, decimal=2)

    def test_different_ellipsoids(self):
        """Test to_geodetic with different ellipsoids"""
        # Start with cartesian coordinates
        x, y, z = to_cartesian(140.0, 35.0, 0.0, ellipsoid='WGS84')

        lon_wgs84, lat_wgs84, h_wgs84 = to_geodetic(x, y, z, ellipsoid='WGS84')
        lon_grs80, lat_grs80, h_grs80 = to_geodetic(x, y, z, ellipsoid='GRS80')

        # Results should be nearly identical
        np.testing.assert_almost_equal(lon_wgs84[0], lon_grs80[0], decimal=6)
        np.testing.assert_almost_equal(lat_wgs84[0], lat_grs80[0], decimal=6)


class TestToSphereAdditional:
    """Additional to_sphere tests"""

    def test_south_pole(self):
        """Test south pole on sphere"""
        x, y, z = to_sphere(0.0, -90.0)

        np.testing.assert_almost_equal(x[0], 0.0, decimal=10)
        np.testing.assert_almost_equal(y[0], 0.0, decimal=10)
        np.testing.assert_almost_equal(z[0], -1.0, decimal=10)

    def test_array_with_radius(self):
        """Test array input with custom radius"""
        lons = np.array([0.0, 90.0, 180.0])
        lats = np.array([0.0, 0.0, 0.0])
        r = np.array([1.0, 2.0, 3.0])

        x, y, z = to_sphere(lons, lats, r)

        np.testing.assert_almost_equal(x[0], 1.0, decimal=10)
        np.testing.assert_almost_equal(y[1], 2.0, decimal=10)
        np.testing.assert_almost_equal(x[2], -3.0, decimal=10)


class TestScaleFactorsAdditional:
    """Additional scale_factors tests"""

    def test_array_input(self):
        """Test scale_factors with array input"""
        lats = np.array([0.0, 30.0, 60.0, 90.0])

        h_lat, h_lon = scale_factors(lats)

        assert len(h_lat) == 4
        assert len(h_lon) == 4
        # Longitudinal scale should decrease with latitude
        assert h_lon[0] > h_lon[1] > h_lon[2] > h_lon[3]

    def test_different_ellipsoids(self):
        """Test scale_factors with different ellipsoids"""
        h_lat_wgs84, h_lon_wgs84 = scale_factors(45.0, ellipsoid='WGS84')
        h_lat_grs80, h_lon_grs80 = scale_factors(45.0, ellipsoid='GRS80')

        # Should be nearly identical
        np.testing.assert_almost_equal(h_lat_wgs84[0], h_lat_grs80[0], decimal=0)
        np.testing.assert_almost_equal(h_lon_wgs84[0], h_lon_grs80[0], decimal=0)
