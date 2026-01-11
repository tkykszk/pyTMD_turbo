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

from . import compute
from . import spatial
from . import predict
from . import interpolate
from .compute import (
    tide_elevations,
    tide_currents,
    predict_batch,
    predict_single,
    init_model,
    datetime_to_mjd,
    SET_displacements,
    LPET_elevations,
    tide_masks,
)
from .spatial import (
    to_cartesian,
    to_geodetic,
    to_sphere,
    scale_factors,
    datum,
)
from .predict import (
    infer_minor,
    infer_diurnal,
    infer_semi_diurnal,
    solid_earth_tide,
    body_tide,
    equilibrium_tide,
)
from .interpolate import (
    extrapolate,
    bilinear,
)

__version__ = '0.1.0'
__all__ = [
    'compute',
    'spatial',
    'predict',
    'interpolate',
    # Ocean tide
    'tide_elevations',
    'tide_currents',
    'predict_batch',
    'predict_single',
    'init_model',
    'datetime_to_mjd',
    # Solid Earth tide
    'SET_displacements',
    'solid_earth_tide',
    'body_tide',
    # Equilibrium tide
    'LPET_elevations',
    'equilibrium_tide',
    # Masks
    'tide_masks',
    # Spatial
    'to_cartesian',
    'to_geodetic',
    'to_sphere',
    'scale_factors',
    'datum',
    # Prediction
    'infer_minor',
    'infer_diurnal',
    'infer_semi_diurnal',
    # Interpolation
    'extrapolate',
    'bilinear',
]
