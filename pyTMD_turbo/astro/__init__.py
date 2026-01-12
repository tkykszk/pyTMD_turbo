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
    delaunay_arguments,
    doodson_arguments,
    greenwich_hour_angle,
    greenwich_mean_sidereal_time,
    lunar_distance,
    lunar_ecef,
    normalize_angle,
    polynomial_sum,
    schureman_arguments,
    solar_distance,
    solar_ecef,
)

__all__ = [
    'delaunay_arguments',
    'doodson_arguments',
    'greenwich_hour_angle',
    'greenwich_mean_sidereal_time',
    'lunar_distance',
    'lunar_ecef',
    'normalize_angle',
    'polynomial_sum',
    'schureman_arguments',
    'solar_distance',
    'solar_ecef',
]
