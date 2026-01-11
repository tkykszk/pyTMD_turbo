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

from .model import model, load_database
from .dataset import TMDAccessor, register_accessor
from . import OTIS
from . import ATLAS
from . import FES
from .OTIS import (
    open_dataset,
    open_otis_grid,
    open_otis_elevation,
    open_otis_transport,
    open_mfdataset,
)
from .ATLAS import (
    open_atlas_grid,
    open_atlas_elevation,
    open_atlas_transport,
)
from .FES import (
    open_fes_elevation,
    open_fes_transport,
)

__all__ = [
    'model',
    'load_database',
    'OTIS',
    'ATLAS',
    'FES',
    'TMDAccessor',
    'register_accessor',
    # OTIS functions
    'open_dataset',
    'open_otis_grid',
    'open_otis_elevation',
    'open_otis_transport',
    'open_mfdataset',
    # ATLAS functions
    'open_atlas_grid',
    'open_atlas_elevation',
    'open_atlas_transport',
    # FES functions
    'open_fes_elevation',
    'open_fes_transport',
]
