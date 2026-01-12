"""
pyTMD_turbo.io - I/O modules for tidal model data

This module provides readers for various tidal model formats:
- GOT: Goddard Ocean Tide models (NetCDF)
- OTIS: Oregon State University OTIS format (binary)
- ATLAS: ATLAS compact format
- FES: Finite Element Solution models (NetCDF)

Also provides the tmd accessor for xarray Datasets.

Copyright (c) 2024-2026 tkykszk
Derived from pyTMD by Tyler Sutterley (MIT License)
"""

from . import ATLAS, FES, OTIS
from .ATLAS import (
    open_atlas_elevation,
    open_atlas_grid,
    open_atlas_transport,
)
from .dataset import TMDAccessor, register_accessor
from .FES import (
    open_fes_elevation,
    open_fes_transport,
)
from .model import load_database, model
from .OTIS import (
    open_dataset,
    open_mfdataset,
    open_otis_elevation,
    open_otis_grid,
    open_otis_transport,
)

__all__ = [
    'ATLAS',
    'FES',
    'OTIS',
    'TMDAccessor',
    'load_database',
    'model',
    'open_atlas_elevation',
    # ATLAS functions
    'open_atlas_grid',
    'open_atlas_transport',
    # OTIS functions
    'open_dataset',
    # FES functions
    'open_fes_elevation',
    'open_fes_transport',
    'open_mfdataset',
    'open_otis_elevation',
    'open_otis_grid',
    'open_otis_transport',
    'register_accessor',
]
