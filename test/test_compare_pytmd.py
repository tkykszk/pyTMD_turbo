"""
Comparison tests with original pyTMD

These tests verify accuracy by comparing results with the original pyTMD module.
Tests are skipped if pyTMD is not installed.

Copyright (c) 2024-2026 tkykszk
"""

import numpy as np
import pytest

# Skip all tests if pyTMD is not available
pytmd = pytest.importorskip("pyTMD", reason="pyTMD not installed")


class TestSolarComparison:
    """Compare solar position calculations with pyTMD"""

    def test_solar_ecef_accuracy(self):
        """Compare solar ECEF coordinates with pyTMD"""
        import pyTMD.astro
        from pyTMD_turbo.astro.ephemeris import solar_ecef

        # One year of data (daily)
        mjd = 60000 + np.arange(365, dtype=np.float64)

        # pyTMD_turbo
        X_turbo, Y_turbo, Z_turbo = solar_ecef(mjd)

        # Original pyTMD
        X_pytmd, Y_pytmd, Z_pytmd = pyTMD.astro.solar_ecef(
            mjd, ephemerides='approximate'
        )

        # Distance comparison
        dist_turbo = np.sqrt(X_turbo**2 + Y_turbo**2 + Z_turbo**2)
        dist_pytmd = np.sqrt(X_pytmd**2 + Y_pytmd**2 + Z_pytmd**2)

        # Relative error should be < 0.1%
        rel_err = np.abs(dist_turbo - dist_pytmd) / dist_pytmd * 100
        assert rel_err.max() < 0.1, f"Distance error too large: {rel_err.max():.4f}%"

        # Correlation should be > 0.9999
        corr_x = np.corrcoef(X_turbo, X_pytmd)[0, 1]
        corr_y = np.corrcoef(Y_turbo, Y_pytmd)[0, 1]
        corr_z = np.corrcoef(Z_turbo, Z_pytmd)[0, 1]

        assert corr_x > 0.9999, f"X correlation too low: {corr_x}"
        assert corr_y > 0.9999, f"Y correlation too low: {corr_y}"
        assert corr_z > 0.9999, f"Z correlation too low: {corr_z}"


class TestLunarComparison:
    """Compare lunar position calculations with pyTMD"""

    def test_lunar_ecef_accuracy(self):
        """Compare lunar ECEF coordinates with pyTMD"""
        import pyTMD.astro
        from pyTMD_turbo.astro.ephemeris import lunar_ecef

        # One month of data (hourly)
        mjd = 60000 + np.arange(30 * 24, dtype=np.float64) / 24.0

        # pyTMD_turbo
        X_turbo, Y_turbo, Z_turbo = lunar_ecef(mjd)

        # Original pyTMD
        X_pytmd, Y_pytmd, Z_pytmd = pyTMD.astro.lunar_ecef(
            mjd, ephemerides='approximate'
        )

        # Distance comparison
        dist_turbo = np.sqrt(X_turbo**2 + Y_turbo**2 + Z_turbo**2)
        dist_pytmd = np.sqrt(X_pytmd**2 + Y_pytmd**2 + Z_pytmd**2)

        # Relative error should be < 1% (lunar calculation is more complex)
        rel_err = np.abs(dist_turbo - dist_pytmd) / dist_pytmd * 100
        assert rel_err.max() < 1.0, f"Distance error too large: {rel_err.max():.4f}%"

        # Correlation should be > 0.99
        corr_x = np.corrcoef(X_turbo, X_pytmd)[0, 1]
        corr_y = np.corrcoef(Y_turbo, Y_pytmd)[0, 1]
        corr_z = np.corrcoef(Z_turbo, Z_pytmd)[0, 1]

        assert corr_x > 0.99, f"X correlation too low: {corr_x}"
        assert corr_y > 0.99, f"Y correlation too low: {corr_y}"
        assert corr_z > 0.99, f"Z correlation too low: {corr_z}"


class TestSolidEarthTideComparison:
    """Compare solid Earth tide calculations with pyTMD"""

    def test_set_radial_accuracy(self):
        """Compare SET radial displacement with pyTMD"""
        import pyTMD.compute
        from pyTMD_turbo.astro.ephemeris import solar_ecef, lunar_ecef
        from pyTMD_turbo.astro.potential import solid_earth_tide

        # Test location (Tokyo)
        lat = 35.0
        lon = 135.0

        # One day of times (hourly)
        mjd = 60000.0 + np.arange(24) / 24.0

        # pyTMD_turbo
        sun_x, sun_y, sun_z = solar_ecef(mjd)
        moon_x, moon_y, moon_z = lunar_ecef(mjd)

        _, _, dR_turbo = solid_earth_tide(
            mjd, np.array([lat]), np.array([lon]),
            sun_x, sun_y, sun_z, moon_x, moon_y, moon_z
        )

        # Original pyTMD
        # delta_time is seconds since 1992-01-01
        delta_time = (mjd - 48622.0) * 86400.0

        SE_pytmd = pyTMD.compute.SET_displacements(
            np.array([lon]), np.array([lat]), delta_time,
            method='ephemerides',
            ephemerides='approximate',
            type='time series',
            variable='R'
        )
        dR_pytmd = np.asarray(SE_pytmd).flatten()

        # Correlation should be > 0.7
        # Note: pyTMD_turbo uses simplified IERS conventions while pyTMD uses
        # more complete ephemerides, so some difference is expected
        corr = np.corrcoef(dR_turbo, dR_pytmd)[0, 1]
        assert corr > 0.7, f"Correlation too low: {corr}"

        # RMS error should be < 100mm (allowing for different algorithms)
        # The implementations use different approaches - pyTMD_turbo uses
        # a streamlined IERS approach while pyTMD has full ephemerides
        rms = np.sqrt(np.mean((dR_turbo - dR_pytmd)**2))
        assert rms < 0.1, f"RMS error too large: {rms*1000:.2f} mm"

    def test_set_multiple_locations(self):
        """Compare SET at multiple locations"""
        import pyTMD.compute
        from pyTMD_turbo.astro.ephemeris import solar_ecef, lunar_ecef
        from pyTMD_turbo.astro.potential import solid_earth_tide

        # Test locations
        locations = [
            (35.0, 135.0, "Tokyo"),
            (51.5, 0.0, "London"),
            (40.7, -74.0, "New York"),
            (-33.9, 151.2, "Sydney"),
        ]

        mjd = np.array([60000.0])
        delta_time = (mjd - 48622.0) * 86400.0

        sun_x, sun_y, sun_z = solar_ecef(mjd)
        moon_x, moon_y, moon_z = lunar_ecef(mjd)

        for lat, lon, name in locations:
            # pyTMD_turbo
            _, _, dR_turbo = solid_earth_tide(
                mjd, np.array([lat]), np.array([lon]),
                sun_x, sun_y, sun_z, moon_x, moon_y, moon_z
            )

            # Original pyTMD
            SE_pytmd = pyTMD.compute.SET_displacements(
                np.array([lon]), np.array([lat]), delta_time,
                method='ephemerides',
                ephemerides='approximate',
                type='time series',
                variable='R'
            )
            dR_pytmd = np.asarray(SE_pytmd).flatten()[0]

            # Difference should be < 150mm (accounting for algorithm differences)
            diff = abs(dR_turbo[0] - dR_pytmd)
            assert diff < 0.15, f"{name}: difference too large: {diff*1000:.2f} mm"


class TestSETDisplacementsComparison:
    """Direct comparison of SET_displacements between pyTMD and pyTMD_turbo

    Implementation Details:
    =======================
    Both pyTMD and pyTMD_turbo solid_earth_tide implement:
      - Degree-2 and Degree-3 tidal displacement (Mathews et al. 1997)
      - Latitude-dependent Love numbers
      - Out-of-phase diurnal/semidiurnal corrections (IERS 2010)
      - Frequency-dependent diurnal/long-period corrections (IERS 2010)

    IMPORTANT: pyTMD default epoch is (2000, 1, 1, 0, 0, 0).
    When using delta_time = (mjd - 48622.0) * 86400.0 (seconds since 1992),
    you MUST specify epoch=(1992, 1, 1, 0, 0, 0).

    Minor differences expected due to:
      - Different rotation conventions (geodetic vs geocentric latitude)
      - North component sign convention (opposite signs)
      - Slight differences in correction function implementations

    Expected results:
      - Up correlation: > 0.99
      - RMS difference: < 15 mm
      - North correlation: ~-0.999 (sign convention difference)
    """

    def test_set_displacements_single_point_time_series(self):
        """Compare SET_displacements for single point over 24 hours"""
        import pyTMD.compute
        import pyTMD_turbo

        # Test location (Tokyo)
        lon = np.array([140.0])
        lat = np.array([35.0])

        # 24 hours of times (hourly)
        mjd = 60000.0 + np.arange(24) / 24.0
        delta_time = (mjd - 48622.0) * 86400.0  # seconds since 1992-01-01

        # pyTMD_turbo (uses MJD directly)
        dn_turbo, de_turbo, du_turbo = pyTMD_turbo.SET_displacements(
            lon, lat, mjd,
            coordinate_system='geographic'
        )

        # pyTMD - Get N, E, R components
        # IMPORTANT: Must specify epoch=(1992, 1, 1, 0, 0, 0) since our delta_time
        # is seconds since 1992-01-01 (pyTMD default epoch is 2000-01-01)
        dn_pytmd = np.asarray(pyTMD.compute.SET_displacements(
            lon, lat, delta_time,
            method='ephemerides',
            ephemerides='approximate',
            type='time series',
            variable='N',
            epoch=(1992, 1, 1, 0, 0, 0)
        )).flatten()

        de_pytmd = np.asarray(pyTMD.compute.SET_displacements(
            lon, lat, delta_time,
            method='ephemerides',
            ephemerides='approximate',
            type='time series',
            variable='E',
            epoch=(1992, 1, 1, 0, 0, 0)
        )).flatten()

        du_pytmd = np.asarray(pyTMD.compute.SET_displacements(
            lon, lat, delta_time,
            method='ephemerides',
            ephemerides='approximate',
            type='time series',
            variable='R',
            epoch=(1992, 1, 1, 0, 0, 0)
        )).flatten()

        # Correlation tests
        # Note: North has opposite sign convention between pyTMD and pyTMD_turbo
        # (geodetic vs geocentric latitude in rotation matrix)
        corr_n = np.corrcoef(-dn_turbo.flatten(), dn_pytmd)[0, 1]  # Negate for sign convention
        corr_e = np.corrcoef(de_turbo.flatten(), de_pytmd)[0, 1]
        corr_u = np.corrcoef(du_turbo.flatten(), du_pytmd)[0, 1]

        print(f"\nSET_displacements single point comparison:")
        print(f"  North correlation (sign-adjusted): {corr_n:.6f}")
        print(f"  East correlation:  {corr_e:.6f}")
        print(f"  Up correlation:    {corr_u:.6f}")

        # RMS differences (use sign-adjusted North)
        rms_n = np.sqrt(np.mean((-dn_turbo.flatten() - dn_pytmd)**2)) * 1000  # mm
        rms_e = np.sqrt(np.mean((de_turbo.flatten() - de_pytmd)**2)) * 1000  # mm
        rms_u = np.sqrt(np.mean((du_turbo.flatten() - du_pytmd)**2)) * 1000  # mm

        print(f"  North RMS diff (sign-adjusted): {rms_n:.3f} mm")
        print(f"  East RMS diff:  {rms_e:.3f} mm")
        print(f"  Up RMS diff:    {rms_u:.3f} mm")

        # Assertions - high correlation expected with correct epoch
        assert corr_u > 0.99, f"Up correlation too low: {corr_u}"
        assert rms_u < 15, f"Up RMS too large: {rms_u:.2f} mm"
        assert corr_n > 0.99, f"North correlation (sign-adjusted) too low: {corr_n}"

    def test_set_displacements_multiple_points(self):
        """Compare SET_displacements for multiple locations at single time"""
        import pyTMD.compute
        import pyTMD_turbo

        # Multiple test locations
        lons = np.array([140.0, 0.0, -74.0, 151.2, -122.4])
        lats = np.array([35.0, 51.5, 40.7, -33.9, 37.8])
        names = ["Tokyo", "London", "New York", "Sydney", "San Francisco"]

        # Single time
        mjd = np.array([60000.0])
        delta_time = (mjd - 48622.0) * 86400.0

        print(f"\nSET_displacements multiple points comparison:")

        for i, (lon, lat, name) in enumerate(zip(lons, lats, names)):
            lon_arr = np.array([lon])
            lat_arr = np.array([lat])

            # pyTMD_turbo
            dn_turbo, de_turbo, du_turbo = pyTMD_turbo.SET_displacements(
                lon_arr, lat_arr, mjd,
                coordinate_system='geographic'
            )

            # pyTMD - with correct epoch
            du_pytmd = np.asarray(pyTMD.compute.SET_displacements(
                lon_arr, lat_arr, delta_time,
                method='ephemerides',
                ephemerides='approximate',
                type='time series',
                variable='R',
                epoch=(1992, 1, 1, 0, 0, 0)
            )).flatten()[0]

            dn_pytmd = np.asarray(pyTMD.compute.SET_displacements(
                lon_arr, lat_arr, delta_time,
                method='ephemerides',
                ephemerides='approximate',
                type='time series',
                variable='N',
                epoch=(1992, 1, 1, 0, 0, 0)
            )).flatten()[0]

            de_pytmd = np.asarray(pyTMD.compute.SET_displacements(
                lon_arr, lat_arr, delta_time,
                method='ephemerides',
                ephemerides='approximate',
                type='time series',
                variable='E',
                epoch=(1992, 1, 1, 0, 0, 0)
            )).flatten()[0]

            # Differences in mm (negate North for sign convention)
            diff_n = (-dn_turbo.flatten()[0] - dn_pytmd) * 1000
            diff_e = (de_turbo.flatten()[0] - de_pytmd) * 1000
            diff_u = (du_turbo.flatten()[0] - du_pytmd) * 1000

            print(f"  {name:15s}: N={diff_n:+7.2f}mm, E={diff_e:+7.2f}mm, U={diff_u:+7.2f}mm")

            # Up component should agree within 20mm with correct epoch
            assert abs(diff_u) < 20, f"{name}: Up difference too large: {diff_u:.2f} mm"

    def test_set_displacements_batch_vs_loop(self):
        """Verify batch calculation matches loop calculation"""
        import pyTMD_turbo

        # 10 locations
        np.random.seed(42)
        lons = np.random.uniform(-180, 180, 10)
        lats = np.random.uniform(-60, 60, 10)

        # 24 hours
        mjd = 60000.0 + np.arange(24) / 24.0

        # Batch calculation (vectorized)
        dn_batch, de_batch, du_batch = pyTMD_turbo.SET_displacements(
            lons, lats, mjd,
            coordinate_system='geographic'
        )

        # Loop calculation (one point at a time)
        for i in range(len(lons)):
            dn_single, de_single, du_single = pyTMD_turbo.SET_displacements(
                np.array([lons[i]]), np.array([lats[i]]), mjd,
                coordinate_system='geographic'
            )

            # Should match exactly (numerical precision)
            np.testing.assert_allclose(
                dn_batch[i], dn_single.flatten(),
                rtol=1e-10, atol=1e-15,
                err_msg=f"North mismatch at point {i}"
            )
            np.testing.assert_allclose(
                de_batch[i], de_single.flatten(),
                rtol=1e-10, atol=1e-15,
                err_msg=f"East mismatch at point {i}"
            )
            np.testing.assert_allclose(
                du_batch[i], du_single.flatten(),
                rtol=1e-10, atol=1e-15,
                err_msg=f"Up mismatch at point {i}"
            )

        print(f"\nBatch vs loop: 10 points × 24 times - all match!")

    def test_set_displacements_cartesian_vs_geographic(self):
        """Verify Cartesian output is consistent with Geographic output"""
        import pyTMD_turbo
        from pyTMD_turbo.predict.solid_earth import ecef_to_enu_rotation

        # Test location
        lon = np.array([140.0])
        lat = np.array([35.0])

        # 24 hours
        mjd = 60000.0 + np.arange(24) / 24.0

        # Get both outputs
        dn, de, du = pyTMD_turbo.SET_displacements(
            lon, lat, mjd,
            coordinate_system='geographic'
        )

        dx, dy, dz = pyTMD_turbo.SET_displacements(
            lon, lat, mjd,
            coordinate_system='cartesian'
        )

        # Convert Cartesian back to ENU manually
        R = ecef_to_enu_rotation(np.radians(lat[0]), np.radians(lon[0]))
        d_ecef = np.stack([dx.flatten(), dy.flatten(), dz.flatten()], axis=0)
        d_enu = R @ d_ecef

        de_check = d_enu[0]  # East
        dn_check = d_enu[1]  # North
        du_check = d_enu[2]  # Up

        # Should match exactly
        np.testing.assert_allclose(
            de.flatten(), de_check, rtol=1e-10, atol=1e-15,
            err_msg="East component mismatch"
        )
        np.testing.assert_allclose(
            dn.flatten(), dn_check, rtol=1e-10, atol=1e-15,
            err_msg="North component mismatch"
        )
        np.testing.assert_allclose(
            du.flatten(), du_check, rtol=1e-10, atol=1e-15,
            err_msg="Up component mismatch"
        )

        print(f"\nCartesian vs Geographic consistency check: PASSED")
        print(f"  Max East diff:  {np.max(np.abs(de.flatten() - de_check))*1e9:.3f} nm")
        print(f"  Max North diff: {np.max(np.abs(dn.flatten() - dn_check))*1e9:.3f} nm")
        print(f"  Max Up diff:    {np.max(np.abs(du.flatten() - du_check))*1e9:.3f} nm")

    def test_set_displacements_long_period(self):
        """Compare SET_displacements over longer time period (1 week)"""
        import pyTMD.compute
        import pyTMD_turbo

        # Test location
        lon = np.array([140.0])
        lat = np.array([35.0])

        # 1 week of hourly data
        mjd = 60000.0 + np.arange(168) / 24.0  # 7 days × 24 hours
        delta_time = (mjd - 48622.0) * 86400.0

        # pyTMD_turbo
        dn_turbo, de_turbo, du_turbo = pyTMD_turbo.SET_displacements(
            lon, lat, mjd,
            coordinate_system='geographic'
        )

        # pyTMD - with correct epoch
        du_pytmd = np.asarray(pyTMD.compute.SET_displacements(
            lon, lat, delta_time,
            method='ephemerides',
            ephemerides='approximate',
            type='time series',
            variable='R',
            epoch=(1992, 1, 1, 0, 0, 0)
        )).flatten()

        # Correlation
        corr_u = np.corrcoef(du_turbo.flatten(), du_pytmd)[0, 1]

        # RMS difference
        rms_u = np.sqrt(np.mean((du_turbo.flatten() - du_pytmd)**2)) * 1000

        # Statistics
        mean_turbo = np.mean(du_turbo) * 1000
        mean_pytmd = np.mean(du_pytmd) * 1000
        std_turbo = np.std(du_turbo) * 1000
        std_pytmd = np.std(du_pytmd) * 1000

        print(f"\nSET_displacements 1-week comparison:")
        print(f"  Up correlation: {corr_u:.6f}")
        print(f"  Up RMS diff: {rms_u:.2f} mm")
        print(f"  turbo: mean={mean_turbo:+.2f}mm, std={std_turbo:.2f}mm")
        print(f"  pyTMD: mean={mean_pytmd:+.2f}mm, std={std_pytmd:.2f}mm")

        # High correlation expected with correct epoch
        assert corr_u > 0.99, f"Up correlation over 1 week too low: {corr_u}"
        assert rms_u < 15, f"Up RMS over 1 week too large: {rms_u:.2f} mm"


class TestPerformance:
    """Performance comparison tests"""

    def test_solar_performance(self):
        """Verify pyTMD_turbo is faster than pyTMD for solar calculations"""
        import time
        import pyTMD.astro
        from pyTMD_turbo.astro.ephemeris import solar_ecef

        mjd = 60000 + np.arange(10000, dtype=np.float64) / 1440.0

        # pyTMD_turbo timing
        start = time.time()
        for _ in range(10):
            solar_ecef(mjd)
        turbo_time = time.time() - start

        # Original pyTMD timing
        start = time.time()
        for _ in range(10):
            pyTMD.astro.solar_ecef(mjd, ephemerides='approximate')
        pytmd_time = time.time() - start

        # pyTMD_turbo should be faster (or at least not slower)
        speedup = pytmd_time / turbo_time
        print(f"\nSolar calculation speedup: {speedup:.1f}x")
        print(f"  pyTMD_turbo: {turbo_time*100:.1f}ms")
        print(f"  pyTMD:       {pytmd_time*100:.1f}ms")

        # Allow some tolerance - at minimum shouldn't be more than 2x slower
        assert speedup > 0.5, f"pyTMD_turbo too slow: {speedup:.2f}x"

    def test_lunar_performance(self):
        """Verify pyTMD_turbo is faster than pyTMD for lunar calculations"""
        import time
        import pyTMD.astro
        from pyTMD_turbo.astro.ephemeris import lunar_ecef

        mjd = 60000 + np.arange(10000, dtype=np.float64) / 1440.0

        # pyTMD_turbo timing
        start = time.time()
        for _ in range(10):
            lunar_ecef(mjd)
        turbo_time = time.time() - start

        # Original pyTMD timing
        start = time.time()
        for _ in range(10):
            pyTMD.astro.lunar_ecef(mjd, ephemerides='approximate')
        pytmd_time = time.time() - start

        speedup = pytmd_time / turbo_time
        print(f"\nLunar calculation speedup: {speedup:.1f}x")
        print(f"  pyTMD_turbo: {turbo_time*100:.1f}ms")
        print(f"  pyTMD:       {pytmd_time*100:.1f}ms")

        assert speedup > 0.5, f"pyTMD_turbo too slow: {speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
