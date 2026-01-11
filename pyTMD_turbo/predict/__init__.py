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

from .infer_minor import (
    infer_minor,
    infer_diurnal,
    infer_semi_diurnal,
    MINOR_CONSTITUENTS,
    DIURNAL_MINORS,
    SEMI_DIURNAL_MINORS,
)
from .solid_earth import (
    solid_earth_tide,
    body_tide,
    love_numbers,
    complex_love_numbers,
    out_of_phase_diurnal,
    out_of_phase_semidiurnal,
    latitude_dependence,
)
from .equilibrium import (
    equilibrium_tide,
    LPET_elevations,
    mean_longitudes,
    legendre_polynomial,
)

__all__ = [
    # Minor constituent inference
    'infer_minor',
    'infer_diurnal',
    'infer_semi_diurnal',
    'MINOR_CONSTITUENTS',
    'DIURNAL_MINORS',
    'SEMI_DIURNAL_MINORS',
    # Solid Earth tide
    'solid_earth_tide',
    'body_tide',
    'love_numbers',
    'complex_love_numbers',
    'out_of_phase_diurnal',
    'out_of_phase_semidiurnal',
    'latitude_dependence',
    # Equilibrium tide
    'equilibrium_tide',
    'LPET_elevations',
    'mean_longitudes',
    'legendre_polynomial',
]
