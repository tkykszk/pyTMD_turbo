"""
Solar position calculation tests

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

import numpy as np
import pytest

from pyTMD_turbo.astro.ephemeris import solar_ecef, solar_distance


def test_solar_ecef_single_time():
    """Single time solar ECEF coordinate test"""
    mjd = np.array([60000.0])  # Around 25 February 2023

    X, Y, Z = solar_ecef(mjd)

    # Check that values are reasonable
    # Solar distance should be around 1 AU = 1.496e11 m
    dist = np.sqrt(X[0]**2 + Y[0]**2 + Z[0]**2)
    assert 1.4e11 < dist < 1.6e11, f"Solar distance out of range: {dist}"


def test_solar_ecef_multiple_times():
    """Multiple times solar ECEF coordinate test"""
    # One year of data (daily intervals)
    mjd = 60000 + np.arange(365, dtype=np.float64)

    X, Y, Z = solar_ecef(mjd)

    # Distance should vary between perihelion and aphelion
    dist = np.sqrt(X**2 + Y**2 + Z**2)

    assert len(dist) == 365
    assert dist.min() > 1.4e11
    assert dist.max() < 1.6e11
    # Variation between perihelion and aphelion
    assert dist.max() - dist.min() > 4e9


def test_solar_distance():
    """Solar distance calculation test"""
    mjd = np.array([60000.0])

    dist = solar_distance(mjd)

    # Should be around 1 AU
    assert 1.4e11 < dist[0] < 1.6e11


def test_solar_ecef_shape():
    """Output shape test"""
    mjd = np.arange(100, dtype=np.float64)

    X, Y, Z = solar_ecef(mjd)

    assert X.shape == (100,)
    assert Y.shape == (100,)
    assert Z.shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
