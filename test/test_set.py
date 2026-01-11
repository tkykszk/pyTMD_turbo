"""
Solid Earth Tide (SET) calculation tests

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

import numpy as np
import pytest

from pyTMD_turbo.astro.ephemeris import solar_ecef, lunar_ecef
from pyTMD_turbo.astro.potential import solid_earth_tide, solid_earth_tide_radial


def test_set_single_point():
    """Single point solid Earth tide test"""
    # Test conditions
    lat = 35.0  # Near Tokyo
    lon = 135.0

    # One day of times (hourly intervals)
    mjd = 60000.0 + np.arange(24) / 24.0

    # Calculate celestial body positions
    sun_x, sun_y, sun_z = solar_ecef(mjd)
    moon_x, moon_y, moon_z = lunar_ecef(mjd)

    dN, dE, dR = solid_earth_tide(
        mjd, np.array([lat]), np.array([lon]),
        sun_x, sun_y, sun_z, moon_x, moon_y, moon_z
    )

    # Solid Earth tide should be in the range of mm to cm
    assert dR.max() - dR.min() > 0.01, "Variation too small"
    assert np.abs(dR).max() < 1.0, "Values too large"


def test_set_multiple_points():
    """Multiple points solid Earth tide test"""
    # Test conditions (various locations in Japan)
    lats = np.array([35.0, 43.0, 26.0, 33.0])
    lons = np.array([135.0, 141.0, 127.0, 130.0])

    # Single time
    mjd = np.array([60000.0])

    sun_x, sun_y, sun_z = solar_ecef(mjd)
    moon_x, moon_y, moon_z = lunar_ecef(mjd)

    for lat, lon in zip(lats, lons):
        lat_arr = np.array([lat])
        lon_arr = np.array([lon])

        dN, dE, dR = solid_earth_tide(
            mjd, lat_arr, lon_arr, sun_x, sun_y, sun_z, moon_x, moon_y, moon_z
        )

        # Values should be reasonable
        assert np.abs(dR[0]) < 1.0, f"Value too large at ({lat}, {lon})"


def test_set_radial_only():
    """Radial component only test"""
    lat = 35.0
    lon = 135.0
    mjd = 60000.0 + np.arange(24) / 24.0

    sun_x, sun_y, sun_z = solar_ecef(mjd)
    moon_x, moon_y, moon_z = lunar_ecef(mjd)

    dR = solid_earth_tide_radial(
        mjd, np.array([lat]), np.array([lon]),
        sun_x, sun_y, sun_z, moon_x, moon_y, moon_z
    )

    # Compare with full calculation
    dN, dE, dR_full = solid_earth_tide(
        mjd, np.array([lat]), np.array([lon]),
        sun_x, sun_y, sun_z, moon_x, moon_y, moon_z
    )

    np.testing.assert_allclose(dR, dR_full, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
