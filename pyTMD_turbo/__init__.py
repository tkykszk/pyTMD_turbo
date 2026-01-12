"""
pyTMD_turbo - High-performance tidal prediction module

Optimized for batch processing: ~700x faster for many locations/times.
Maintains >99% correlation with pyTMD results.

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.

Usage:
    import numpy as np
    import pyTMD_turbo

    # Time as Modified Julian Day (MJD)
    mjd = 60310.0 + np.arange(24) / 24.0

    # Single point prediction
    tide = pyTMD_turbo.predict_single(
        lat=35.0, lon=140.0, mjd=mjd,
        model='GOT5.5', directory='/path/to/models'
    )

    # Batch prediction (many points, many times)
    tide = pyTMD_turbo.predict_batch(
        lats, lons, mjd,
        model='GOT5.5', directory='/path/to/models'
    )
"""

from . import compute, interpolate, predict, spatial
from .cache import (
    # Context managers
    cache_disabled,
    cache_disabled_for,
    clear_all_cache,
    clear_cache,
    disable_cache,
    disable_cache_for,
    disable_temp_cache,
    # Enable/disable
    enable_cache,
    enable_cache_for,
    # Temp cache
    enable_temp_cache,
    get_cache_info,
    is_cache_enabled,
    is_cache_enabled_for,
    rebuild_all_cache,
    # Cache operations
    rebuild_cache,
    # Status
    show_cache_status,
)
from .compute import (
    LPET_elevations,
    SET_displacements,
    datetime_to_mjd,
    init_model,
    predict_batch,
    predict_single,
    tide_currents,
    tide_elevations,
    tide_masks,
)
from .interpolate import (
    bilinear,
    extrapolate,
)
from .predict import (
    body_tide,
    equilibrium_tide,
    infer_diurnal,
    infer_minor,
    infer_semi_diurnal,
    solid_earth_tide,
)
from .spatial import (
    datum,
    scale_factors,
    to_cartesian,
    to_geodetic,
    to_sphere,
)

__version__ = '0.2.1'
__all__ = [
    # Equilibrium tide
    'LPET_elevations',
    # Solid Earth tide
    'SET_displacements',
    'bilinear',
    'body_tide',
    'cache_disabled',
    'cache_disabled_for',
    'clear_all_cache',
    'clear_cache',
    'compute',
    'datetime_to_mjd',
    'datum',
    'disable_cache',
    'disable_cache_for',
    'disable_temp_cache',
    # Cache control
    'enable_cache',
    'enable_cache_for',
    'enable_temp_cache',
    'equilibrium_tide',
    # Interpolation
    'extrapolate',
    'get_cache_info',
    'infer_diurnal',
    # Prediction
    'infer_minor',
    'infer_semi_diurnal',
    'init_model',
    'interpolate',
    'is_cache_enabled',
    'is_cache_enabled_for',
    'predict',
    'predict_batch',
    'predict_single',
    'rebuild_all_cache',
    'rebuild_cache',
    'scale_factors',
    'show_cache_status',
    'solid_earth_tide',
    'spatial',
    'tide_currents',
    # Ocean tide
    'tide_elevations',
    # Masks
    'tide_masks',
    # Spatial
    'to_cartesian',
    'to_geodetic',
    'to_sphere',
]
