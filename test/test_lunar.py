"""
Lunar position calculation tests

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

import numpy as np
import pytest

from pyTMD_turbo.astro.ephemeris import lunar_ecef, lunar_distance


def test_lunar_ecef_single_time():
    """Single time lunar ECEF coordinate test"""
    mjd = np.array([60000.0])  # Around 25 February 2023

    X, Y, Z = lunar_ecef(mjd)

    # Check that values are reasonable
    # Lunar distance should be around 384,400 km = 3.844e8 m
    dist = np.sqrt(X[0]**2 + Y[0]**2 + Z[0]**2)
    assert 3.5e8 < dist < 4.1e8, f"Lunar distance out of range: {dist}"


def test_lunar_ecef_multiple_times():
    """Multiple times lunar ECEF coordinate test"""
    # One month of data (hourly intervals)
    mjd = 60000 + np.arange(30 * 24, dtype=np.float64) / 24.0

    X, Y, Z = lunar_ecef(mjd)

    # Distance comparison
    dist = np.sqrt(X**2 + Y**2 + Z**2)

    assert len(dist) == 30 * 24
    # Moon distance varies between perigee (~356,500 km) and apogee (~406,700 km)
    assert dist.min() > 3.5e8
    assert dist.max() < 4.1e8


def test_lunar_distance():
    """Lunar distance calculation test"""
    mjd = np.array([60000.0])

    dist = lunar_distance(mjd)

    # Should be around 384,400 km
    assert 3.5e8 < dist[0] < 4.1e8


def test_lunar_ecef_shape():
    """Output shape test"""
    mjd = np.arange(100, dtype=np.float64)

    X, Y, Z = lunar_ecef(mjd)

    assert X.shape == (100,)
    assert Y.shape == (100,)
    assert Z.shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
