"""
pyTMD_turbo.astro - Astronomical calculation module

Provides functions for:
- Solar and lunar ephemerides
- Astronomical phase arguments
- Greenwich sidereal time

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

from .ephemeris import (
    solar_ecef,
    lunar_ecef,
    solar_distance,
    lunar_distance,
    greenwich_mean_sidereal_time,
    greenwich_hour_angle,
    polynomial_sum,
    normalize_angle,
    doodson_arguments,
    delaunay_arguments,
    schureman_arguments,
)

__all__ = [
    'solar_ecef',
    'lunar_ecef',
    'solar_distance',
    'lunar_distance',
    'greenwich_mean_sidereal_time',
    'greenwich_hour_angle',
    'polynomial_sum',
    'normalize_angle',
    'doodson_arguments',
    'delaunay_arguments',
    'schureman_arguments',
]
