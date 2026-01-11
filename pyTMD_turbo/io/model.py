"""
pyTMD_turbo.io.model - Tide model loading and management

Provides model class for loading and accessing tidal model data.
Supports GOT, OTIS, ATLAS, and FES formats.

Copyright (c) 2024-2026 tkykszk
Derived from pyTMD by Tyler Sutterley (MIT License)
"""

from __future__ import annotations

import json
import pathlib
import warnings
import numpy as np
from typing import Optional, Dict, List, Any

from .. import cache as cache_module

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import netCDF4
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

__all__ = ["DataBase", "load_database", "model"]

# Constituent name aliases (short name -> canonical name)
_CONSTITUENT_ALIASES = {
    'sig1': 'sigma1',
    '2n2': '2n2',
    'oo1': 'oo1',
    'm3': 'm3',
    "m3'": 'm3',  # GOT5.6 uses m3' for third-degree
}

# Path to data files
_data_path = pathlib.Path(__file__).parent.parent / "data"
_database_file = _data_path / "database.json"

# Default cache directory
def _get_cache_path() -> pathlib.Path:
    """Get default cache path for tide models"""
    cache_dir = pathlib.Path.home() / ".cache" / "pyTMD"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class DataBase:
    """Model database with attribute access"""

    def __init__(self, d: dict):
        self.__dict__ = d

    def update(self, d: dict):
        self.__dict__.update(d)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return str(self.__dict__)


def load_database(extra_databases: list = []) -> DataBase:
    """
    Load the JSON database of model files

    Parameters
    ----------
    extra_databases : list
        Additional database files or dictionaries to merge

    Returns
    -------
    DataBase
        Model database object
    """
    # Load default database
    with open(_database_file, 'r', encoding='utf-8') as f:
        database = json.load(f)

    # Merge extra databases
    for extra in extra_databases:
        if isinstance(extra, dict):
            database.update(extra)
        elif isinstance(extra, (str, pathlib.Path)):
            with open(extra, 'r', encoding='utf-8') as f:
                database.update(json.load(f))

    return DataBase(database)


class model:
    """
    Tide model class for loading and accessing model data

    Parameters
    ----------
    directory : str or pathlib.Path, optional
        Working data directory for tide models

    Attributes
    ----------
    name : str
        Model name
    format : str
        Model format (GOT, OTIS, ATLAS, FES, etc.)
    constituents : list
        List of tidal constituents
    """

    def __init__(self, directory: Optional[str] = None, **kwargs):
        # Set directory
        if directory is None:
            self.directory = _get_cache_path()
        else:
            self.directory = pathlib.Path(directory).expanduser().absolute()

        # Initialize attributes
        self.name = None
        self.format = None
        self.model_file = None
        self.grid_file = None
        self.constituents = []
        self.corrections = 'OTIS'
        self.scale = 1.0
        self.variable = 'tide_ocean'
        self.projection = None
        self._database = None

    def from_database(self, model_name: str, group: str = 'z') -> 'model':
        """
        Load model parameters from database

        Parameters
        ----------
        model_name : str
            Name of the tide model
        group : str, default 'z'
            Component group to load:
            - 'z': tide elevation
            - 'u': zonal (east-west) current component
            - 'v': meridional (north-south) current component

        Returns
        -------
        model
            Self with loaded parameters
        """
        # Load database if not already loaded
        if self._database is None:
            self._database = load_database()

        # Check if model exists
        if not hasattr(self._database, model_name):
            available = list(self._database.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available[:10]}...")

        # Get model parameters
        params = self._database[model_name]

        # Set attributes
        self.name = model_name
        self.format = params.get('format', 'OTIS')
        self.projection = params.get('projection')
        self.group = group

        # Set file paths based on group
        group_params = params.get(group, {})
        if not group_params and group != 'z':
            raise ValueError(f"Model '{model_name}' does not have '{group}' component")

        if group_params:
            model_file = group_params.get('model_file')
            grid_file = group_params.get('grid_file')

            if model_file:
                # Handle both single file (string) and multiple files (list)
                if isinstance(model_file, list):
                    self.model_file = [self.directory / f for f in model_file]
                else:
                    self.model_file = self.directory / model_file
            if grid_file:
                self.grid_file = self.directory / grid_file

            self.variable = group_params.get('variable', 'tide_ocean')
            self.scale = group_params.get('scale', 1.0)
            self.units = group_params.get('units', 'm')

        # Set corrections based on format
        if self.format in ('GOT-ascii', 'GOT-netcdf'):
            self.corrections = 'GOT'
        elif self.format in ('FES-ascii', 'FES-netcdf'):
            self.corrections = 'FES'
        else:
            self.corrections = 'OTIS'

        return self

    def has_currents(self) -> bool:
        """
        Check if model has current (u, v) components

        Returns
        -------
        bool
            True if model has u and v components
        """
        if self._database is None:
            self._database = load_database()

        if self.name is None:
            return False

        params = self._database.get(self.name, {})
        return 'u' in params and 'v' in params

    def open_dataset(self, use_cache: bool = True, **kwargs) -> 'xr.Dataset':
        """
        Open model as xarray Dataset

        Parameters
        ----------
        use_cache : bool, default True
            If True, try to load from cache first. If cache doesn't exist,
            load from files and save to cache for future use.

        Returns
        -------
        xr.Dataset
            Model data as xarray Dataset
        """
        if not HAS_XARRAY:
            raise ImportError("xarray is required for open_dataset")

        # Try to load from cache if enabled
        if use_cache and self.name and cache_module.is_cache_enabled_for(self.name):
            cached_ds = self._load_from_cache()
            if cached_ds is not None:
                return cached_ds

        # Load from original files
        if self.format in ('GOT-netcdf',):
            ds = self._open_got_netcdf(**kwargs)
        elif self.format in ('GOT-ascii',):
            ds = self._open_got_ascii(**kwargs)
        elif self.format in ('OTIS', 'ATLAS', 'TMD3'):
            ds = self._open_otis(**kwargs)
        elif self.format in ('FES-netcdf',):
            ds = self._open_fes_netcdf(**kwargs)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        # Save to cache if enabled
        if use_cache and self.name and cache_module.is_cache_enabled_for(self.name):
            self._save_to_cache(ds)

        return ds

    def _get_cache_path(self) -> pathlib.Path:
        """Get cache file path for this model"""
        return cache_module.get_cache_path(self.name, self.directory)

    def _get_source_files(self) -> List[pathlib.Path]:
        """Get list of source files for cache validation"""
        files = []
        if self.model_file:
            if isinstance(self.model_file, list):
                files.extend([pathlib.Path(f) for f in self.model_file])
            else:
                files.append(pathlib.Path(self.model_file))
        if self.grid_file:
            files.append(pathlib.Path(self.grid_file))
        return files

    def _load_from_cache(self) -> Optional['xr.Dataset']:
        """Load model data from cache"""
        cache_path = self._get_cache_path()
        source_files = self._get_source_files()

        data = cache_module.load_cache(self.name, cache_path, source_files)
        if data is None:
            return None

        try:
            # Reconstruct xarray Dataset from cached numpy arrays
            coords = {}
            data_vars = {}

            # Extract coordinates
            if 'x' in data:
                coords['x'] = data['x']
            if 'y' in data:
                coords['y'] = data['y']
            if 'constituent' in data:
                # Convert bytes to strings if needed
                constituents = data['constituent']
                if constituents.dtype.kind == 'S':
                    constituents = [c.decode('utf-8') for c in constituents]
                else:
                    constituents = list(constituents)
                coords['constituent'] = constituents
                self.constituents = constituents

            # Extract data variables
            for key in data:
                if key in ('x', 'y', 'constituent'):
                    continue

                arr = data[key]
                if arr.ndim == 3:
                    data_vars[key] = (['constituent', 'y', 'x'], arr)
                elif arr.ndim == 2:
                    data_vars[key] = (['y', 'x'], arr)
                else:
                    data_vars[key] = arr

            ds = xr.Dataset(data_vars, coords=coords)
            return ds

        except Exception:
            return None

    def _save_to_cache(self, ds: 'xr.Dataset') -> None:
        """Save model data to cache"""
        cache_path = self._get_cache_path()
        source_files = self._get_source_files()

        try:
            # Convert xarray Dataset to dict of numpy arrays
            data = {}

            # Save coordinates
            if 'x' in ds.coords:
                data['x'] = ds.coords['x'].values
            if 'y' in ds.coords:
                data['y'] = ds.coords['y'].values
            if 'constituent' in ds.coords:
                # Convert strings to bytes for storage
                constituents = ds.coords['constituent'].values
                if hasattr(constituents[0], 'encode'):
                    data['constituent'] = np.array([c.encode('utf-8') for c in constituents])
                else:
                    data['constituent'] = constituents

            # Save data variables
            for name, var in ds.data_vars.items():
                data[name] = var.values

            cache_module.save_cache(self.name, cache_path, data, source_files)

        except Exception:
            # Silently ignore cache save errors
            pass

    def _open_got_netcdf(self, **kwargs) -> 'xr.Dataset':
        """Open GOT NetCDF format model"""
        # Determine constituent files
        if isinstance(self.model_file, list):
            # Use the pre-defined list of files
            nc_files = self.model_file
        else:
            # Find files by globbing directory
            model_dir = self.model_file.parent if self.model_file else self.directory
            nc_files = sorted(model_dir.glob("*.nc"))

        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found")

        # Load all constituents
        datasets = []
        constituents = []

        # Unit scale factor (convert cm to m for GOT models)
        scale = 0.01 if getattr(self, 'units', 'm') == 'cm' else 1.0

        # Get coordinates from first file
        first_ds = xr.open_dataset(nc_files[0])
        if 'longitude' in first_ds.data_vars:
            lon = first_ds['longitude'].values
            lat = first_ds['latitude'].values
        elif 'lon' in first_ds.coords:
            lon = first_ds['lon'].values
            lat = first_ds['lat'].values
        else:
            # Fallback: assume dimension names are coordinates
            lon = first_ds['lon'].values if 'lon' in first_ds.dims else np.arange(first_ds.dims['lon'])
            lat = first_ds['lat'].values if 'lat' in first_ds.dims else np.arange(first_ds.dims['lat'])
        first_ds.close()

        for nc_file in nc_files:
            ds = xr.open_dataset(nc_file)

            # Extract constituent name from filename
            nc_path = pathlib.Path(nc_file)
            const_name = nc_path.stem.lower()
            # Handle special cases like 'k1_5.5D' -> 'k1'
            if '_' in const_name:
                const_name = const_name.split('_')[0]
            # Apply aliases for canonical constituent names
            const_name = _CONSTITUENT_ALIASES.get(const_name, const_name)

            # Apply scale factor and create complex amplitude
            # GOT files have 'amplitude' and 'phase' variables
            # Note: Use negative phase to match pyTMD convention (exp(-i*phase))
            if 'amplitude' in ds.data_vars and 'phase' in ds.data_vars:
                amplitude = ds['amplitude'].values * scale
                phase = np.radians(ds['phase'].values)
                hc = amplitude * np.exp(-1j * phase)

                # Create DataArray with proper coordinates
                da = xr.DataArray(
                    hc,
                    dims=['y', 'x'],
                    coords={'y': lat, 'x': lon},
                    name=const_name
                )
                datasets.append(da.to_dataset())
            else:
                # Fallback for non-standard format
                ds = ds.assign_coords(x=lon, y=lat)
                datasets.append(ds[[const_name]])

            constituents.append(const_name)
            ds.close()

        # Merge all constituents into a single dataset
        combined = xr.merge(datasets)

        self.constituents = constituents
        return combined

    def _open_got_ascii(self, **kwargs) -> 'xr.Dataset':
        """Open GOT ASCII format model"""
        # GOT ASCII format: similar to NetCDF but text-based
        raise NotImplementedError("GOT ASCII format not yet implemented")

    def _open_otis(self, **kwargs) -> 'xr.Dataset':
        """Open OTIS/ATLAS format model"""
        from . import OTIS

        # Read grid file
        lon, lat, hz, mask = OTIS.read_grid(self.grid_file)

        # Read elevation file
        hc, constituents = OTIS.read_elevation(self.model_file)

        self.constituents = constituents

        # Create xarray dataset
        ds = xr.Dataset(
            {
                'hc': (['constituent', 'y', 'x'], hc),
                'hz': (['y', 'x'], hz),
                'mask': (['y', 'x'], mask),
            },
            coords={
                'x': lon,
                'y': lat,
                'constituent': constituents,
            }
        )

        return ds

    def _open_fes_netcdf(self, **kwargs) -> 'xr.Dataset':
        """Open FES NetCDF format model"""
        # FES models have one file per constituent
        model_dir = self.model_file.parent if self.model_file else self.directory

        nc_files = list(model_dir.glob("*.nc"))

        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {model_dir}")

        datasets = []
        constituents = []

        for nc_file in sorted(nc_files):
            ds = xr.open_dataset(nc_file)
            # Extract constituent from filename
            const_name = nc_file.stem.lower()

            # Standardize coordinate names
            if 'longitude' in ds.dims:
                ds = ds.rename({'longitude': 'x', 'latitude': 'y'})
            elif 'lon' in ds.dims:
                ds = ds.rename({'lon': 'x', 'lat': 'y'})

            datasets.append(ds)
            constituents.append(const_name)

        combined = xr.concat(datasets, dim='constituent')
        combined = combined.assign_coords(constituent=constituents)

        self.constituents = constituents
        return combined

    def __repr__(self):
        return f"model(name={self.name}, format={self.format})"
