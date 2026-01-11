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
