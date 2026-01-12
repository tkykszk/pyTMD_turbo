"""
pyTMD_turbo.io.FES - FES format tidal model reader

Reads FES (Finite Element Solution) tidal models in NetCDF format.
FES models are produced by LEGOS/CNES and provide global ocean tide solutions.

Supports:
- FES2012, FES2014, FES2022 and similar models
- Single constituent per file (NetCDF)
- Combined NetCDF files
- xarray Dataset output

Copyright (c) 2024-2026 tkykszk
Derived from pyTMD by Tyler Sutterley (MIT License)
"""

from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

__all__ = [
    'open_dataset',
    'open_fes_elevation',
    'open_fes_transport',
    'open_mfdataset',
    'read_constituent',
    'read_netcdf',
]


def read_netcdf(
    input_file: str | pathlib.Path,
    variable: str | None = None,
) -> dict:
    """
    Read FES NetCDF file

    Parameters
    ----------
    input_file : str or pathlib.Path
        Input NetCDF file
    variable : str, optional
        Variable name to read. If None, will auto-detect.

    Returns
    -------
    data : dict
        Dictionary containing:
        - amplitude : amplitude values
        - phase : phase values (degrees)
        - x : longitude coordinates
        - y : latitude coordinates
        - constituent : constituent name (if available)
    """
    import xarray as xr

    input_file = pathlib.Path(input_file).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Open dataset
    ds = xr.open_dataset(input_file)

    data = {}

    # Get coordinate names (handle different naming conventions)
    x_names = ['lon', 'longitude', 'x', 'Longitude']
    y_names = ['lat', 'latitude', 'y', 'Latitude']

    x_coord = None
    y_coord = None
    for name in x_names:
        if name in ds.coords or name in ds.dims:
            x_coord = name
            break
    for name in y_names:
        if name in ds.coords or name in ds.dims:
            y_coord = name
            break

    if x_coord is None or y_coord is None:
        raise ValueError("Could not find longitude/latitude coordinates")

    data['x'] = ds[x_coord].values
    data['y'] = ds[y_coord].values

    # Get amplitude and phase
    amp_names = ['amplitude', 'amp', 'Ha', 'h_amp', 'u_amp', 'v_amp']
    phase_names = ['phase', 'ph', 'Hg', 'h_phase', 'u_phase', 'v_phase']

    amplitude = None
    phase = None

    for name in amp_names:
        if name in ds.data_vars:
            amplitude = ds[name].values
            break

    for name in phase_names:
        if name in ds.data_vars:
            phase = ds[name].values
            break

    if amplitude is None or phase is None:
        # Try to find any 2D variables
        for var in ds.data_vars:
            if 'amp' in var.lower() and amplitude is None:
                amplitude = ds[var].values
            elif ('phase' in var.lower() or 'ph' in var.lower()) and phase is None:
                phase = ds[var].values

    if amplitude is None or phase is None:
        raise ValueError("Could not find amplitude and phase variables")

    data['amplitude'] = amplitude
    data['phase'] = phase

    # Try to get constituent name from attributes or filename
    constituent = None
    for attr in ['constituent', 'CONSTITUENT', 'cons']:
        if attr in ds.attrs:
            constituent = ds.attrs[attr]
            break

    if constituent is None:
        # Try to extract from filename
        stem = input_file.stem.lower()
        known_constituents = [
            'm2', 's2', 'n2', 'k2', 'k1', 'o1', 'p1', 'q1',
            '2n2', 'eps2', 'j1', 'l2', 'la2', 'lambda2',
            'm4', 'ms4', 'mu2', 'nu2', 'r2', 's1', 'sa', 'ssa',
            'mf', 'mm', 'msf', 'mtm', 't2', 'z0'
        ]
        for c in known_constituents:
            if c in stem:
                constituent = c
                break

    data['constituent'] = constituent

    ds.close()

    return data


def read_constituent(
    input_file: str | pathlib.Path,
) -> dict:
    """
    Read single constituent from FES NetCDF file

    Parameters
    ----------
    input_file : str or pathlib.Path
        Input NetCDF file

    Returns
    -------
    data : dict
        Dictionary with constituent data including complex harmonic constants
    """
    data = read_netcdf(input_file)

    # Convert amplitude/phase to complex
    phase_rad = np.radians(data['phase'])
    hc = data['amplitude'] * np.exp(-1j * phase_rad)
    data['hc'] = hc

    return data


def open_fes_elevation(
    input_files: str | pathlib.Path | list[str | pathlib.Path],
    apply_mask: bool = True,
) -> xr.Dataset:
    """
    Read FES elevation files and return as xarray Dataset

    Parameters
    ----------
    input_files : str, Path, or list
        Input FES elevation file(s). Can be single file or list of files
        (one per constituent).
    apply_mask : bool, default True
        Apply mask (NaN) to invalid data

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing tidal elevation constituents as complex values
    """
    import xarray as xr

    if isinstance(input_files, (str, pathlib.Path)):
        input_files = [input_files]

    all_data = {}
    x_coord = None
    y_coord = None
    constituents = []

    for f in input_files:
        f = pathlib.Path(f).expanduser().resolve()
        data = read_constituent(f)

        constituent = data['constituent']
        if constituent is None:
            # Use filename as fallback
            warnings.warn(
                f"No constituent name found in '{f}'. Using filename '{f.stem}' as constituent name. "
                "This may indicate a non-standard file format.",
                RuntimeWarning,
                stacklevel=2
            )
            constituent = f.stem.lower()

        constituents.append(constituent)
        all_data[constituent] = data['hc']

        if x_coord is None:
            x_coord = data['x']
            y_coord = data['y']

    # Create Dataset
    data_vars = {}
    for const in constituents:
        hc = all_data[const]
        if apply_mask:
            # Mask invalid values
            hc = np.where(np.isfinite(hc), hc, np.nan + 0j)

        data_vars[const] = (['y', 'x'], hc, {
            'standard_name': f'sea_surface_height_amplitude_due_to_{const}_tidal_constituent',
            'units': 'm',
            'long_name': f'{const.upper()} tidal elevation',
        })

    ds = xr.Dataset(
        data_vars,
        coords={
            'x': ('x', x_coord, {'standard_name': 'longitude', 'units': 'degrees_east'}),
            'y': ('y', y_coord, {'standard_name': 'latitude', 'units': 'degrees_north'}),
        },
        attrs={
            'title': 'FES Tidal Elevation Model',
            'format': 'FES',
            'constituents': constituents,
        }
    )

    return ds


def open_fes_transport(
    u_files: str | pathlib.Path | list[str | pathlib.Path],
    v_files: str | pathlib.Path | list[str | pathlib.Path],
    apply_mask: bool = True,
) -> xr.Dataset:
    """
    Read FES transport files and return as xarray Dataset

    Parameters
    ----------
    u_files : str, Path, or list
        Input FES U-component file(s)
    v_files : str, Path, or list
        Input FES V-component file(s)
    apply_mask : bool, default True
        Apply mask to invalid data

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing U and V transport/velocity constituents
    """
    import xarray as xr

    if isinstance(u_files, (str, pathlib.Path)):
        u_files = [u_files]
    if isinstance(v_files, (str, pathlib.Path)):
        v_files = [v_files]

    all_u = {}
    all_v = {}
    x_coord = None
    y_coord = None
    constituents = []

    # Read U components
    for f in u_files:
        f = pathlib.Path(f).expanduser().resolve()
        data = read_constituent(f)

        constituent = data['constituent']
        if constituent is None:
            constituent = f.stem.lower().replace('u_', '').replace('_u', '')

        if constituent not in constituents:
            constituents.append(constituent)
        all_u[constituent] = data['hc']

        if x_coord is None:
            x_coord = data['x']
            y_coord = data['y']

    # Read V components
    for f in v_files:
        f = pathlib.Path(f).expanduser().resolve()
        data = read_constituent(f)

        constituent = data['constituent']
        if constituent is None:
            constituent = f.stem.lower().replace('v_', '').replace('_v', '')

        all_v[constituent] = data['hc']

    # Create Dataset
    data_vars = {}
    for const in constituents:
        if const in all_u:
            uc = all_u[const]
            if apply_mask:
                uc = np.where(np.isfinite(uc), uc, np.nan + 0j)
            data_vars[f'u_{const}'] = (['y', 'x'], uc, {
                'standard_name': f'eastward_sea_water_velocity_due_to_{const}',
                'units': 'm/s',
                'long_name': f'{const.upper()} U velocity',
            })

        if const in all_v:
            vc = all_v[const]
            if apply_mask:
                vc = np.where(np.isfinite(vc), vc, np.nan + 0j)
            data_vars[f'v_{const}'] = (['y', 'x'], vc, {
                'standard_name': f'northward_sea_water_velocity_due_to_{const}',
                'units': 'm/s',
                'long_name': f'{const.upper()} V velocity',
            })

    ds = xr.Dataset(
        data_vars,
        coords={
            'x': ('x', x_coord, {'standard_name': 'longitude', 'units': 'degrees_east'}),
            'y': ('y', y_coord, {'standard_name': 'latitude', 'units': 'degrees_north'}),
        },
        attrs={
            'title': 'FES Tidal Velocity Model',
            'format': 'FES',
            'constituents': constituents,
        }
    )

    return ds


def open_dataset(
    model_files: str | pathlib.Path | list[str | pathlib.Path],
    group: str = 'z',
    apply_mask: bool = True,
) -> xr.Dataset:
    """
    Open FES tidal model as xarray Dataset

    Parameters
    ----------
    model_files : str, Path, or list
        Path to model file(s)
    group : str, default 'z'
        Variable group: 'z' (elevation), 'u', or 'v'
    apply_mask : bool, default True
        Apply mask to invalid data

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing tidal constituents
    """
    import xarray as xr

    if isinstance(model_files, (str, pathlib.Path)):
        model_files = [model_files]

    if group.lower() == 'z':
        return open_fes_elevation(model_files, apply_mask=apply_mask)
    elif group.lower() in ['u', 'v']:
        # For U or V, just read the files as single component
        ds = open_fes_elevation(model_files, apply_mask=apply_mask)

        # Rename variables to indicate component
        new_vars = {}
        for var in ds.data_vars:
            new_vars[f'{group.lower()}_{var}'] = ds[var]

        filtered_ds = xr.Dataset(
            new_vars,
            coords=ds.coords,
            attrs={
                'title': f'FES Tidal Velocity Model ({group.upper()} component)',
                'format': 'FES',
                'constituents': ds.attrs.get('constituents', []),
                'component': group.lower(),
            }
        )
        return filtered_ds
    else:
        raise ValueError(f"Unknown group: {group}. Must be 'z', 'u', or 'v'")


def open_mfdataset(
    model_files: list[str | pathlib.Path],
    group: str = 'z',
    apply_mask: bool = True,
) -> xr.Dataset:
    """
    Open multiple FES files and merge into single Dataset

    Parameters
    ----------
    model_files : list
        List of model file paths
    group : str, default 'z'
        Variable group ('z', 'u', or 'v')
    apply_mask : bool, default True
        Apply mask to invalid data

    Returns
    -------
    ds : xarray.Dataset
        Merged dataset containing all constituents
    """
    return open_dataset(model_files, group=group, apply_mask=apply_mask)
