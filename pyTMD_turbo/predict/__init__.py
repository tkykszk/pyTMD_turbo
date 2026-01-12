"""
pyTMD_turbo.predict - Tide prediction module

Provides functions for:
- Ocean tide prediction (harmonic synthesis)
- Solid Earth tide prediction
- Minor constituent inference

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

from .equilibrium import (
    LPET_elevations,
    equilibrium_tide,
    legendre_polynomial,
    mean_longitudes,
)
from .infer_minor import (
    DIURNAL_MINORS,
    MINOR_CONSTITUENTS,
    SEMI_DIURNAL_MINORS,
    infer_diurnal,
    infer_minor,
    infer_semi_diurnal,
)
from .solid_earth import (
    body_tide,
    complex_love_numbers,
    latitude_dependence,
    love_numbers,
    out_of_phase_diurnal,
    out_of_phase_semidiurnal,
    solid_earth_tide,
)

__all__ = [
    'DIURNAL_MINORS',
    'MINOR_CONSTITUENTS',
    'SEMI_DIURNAL_MINORS',
    'LPET_elevations',
    'body_tide',
    'complex_love_numbers',
    # Equilibrium tide
    'equilibrium_tide',
    'infer_diurnal',
    # Minor constituent inference
    'infer_minor',
    'infer_semi_diurnal',
    'latitude_dependence',
    'legendre_polynomial',
    'love_numbers',
    'mean_longitudes',
    'out_of_phase_diurnal',
    'out_of_phase_semidiurnal',
    # Solid Earth tide
    'solid_earth_tide',
]
