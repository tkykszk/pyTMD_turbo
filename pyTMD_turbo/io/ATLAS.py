"""
pyTMD_turbo.io.ATLAS - ATLAS format tidal model reader

Reads ATLAS-compact format tidal solutions from TPXO9 and similar models.
ATLAS format combines global and local models with variable resolution.

Supports:
- ATLAS-compact binary format (big-endian)
- Global + local model merging
- Memory-mapped file reading for large files
- xarray Dataset output

Copyright (c) 2024-2026 tkykszk
Derived from pyTMD by Tyler Sutterley (MIT License)
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Tuple, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

__all__ = [
    'read_atlas_grid',
    'read_atlas_elevation',
    'read_atlas_transport',
    'open_atlas_grid',
    'open_atlas_elevation',
    'open_atlas_transport',
    'open_dataset',
]


def _read_binary(
    path: Union[str, pathlib.Path],
    dtype: Union[np.dtype, str],
    shape: tuple,
    offset: int = 0,
    use_mmap: bool = False,
) -> np.ndarray:
    """Read binary data from file"""
    path = pathlib.Path(path).expanduser().resolve()
    dtype = np.dtype(dtype)
    count = int(np.prod(shape))

    if use_mmap:
        var = np.memmap(
            path, dtype=dtype, mode='r',
            offset=offset, shape=shape
        )
        var = np.array(var)
    else:
        with open(path, mode="rb") as fid:
            var = np.fromfile(fid, dtype=dtype, offset=offset, count=count)
            var = var.reshape(shape, order='C')
    return var


def read_atlas_grid(
    input_file: Union[str, pathlib.Path],
    use_mmap: bool = False,
) -> dict:
    """
    Read ATLAS-compact model grid file

    ATLAS-compact format contains global grid with optional local refinements.

    Parameters
    ----------
    input_file : str or pathlib.Path
        Input ATLAS grid file
    use_mmap : bool, default False
        Use memory-mapped file reading

    Returns
    -------
    grid : dict
        Dictionary containing:
        - x : longitude coordinates (global)
        - y : latitude coordinates (global)
        - hz : bathymetry (global)
        - mz : mask (global)
        - local : list of local grid dictionaries (if any)
    """
    input_file = pathlib.Path(input_file).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    grid = {}
    offset = 4  # Skip record marker

    # Read global grid dimensions
    nx, ny = _read_binary(input_file, '>i4', (2,), offset=offset, use_mmap=use_mmap)
    nx, ny = int(nx), int(ny)
    offset += 2 * 4

    # Read coordinate limits
    ylim = _read_binary(input_file, '>f4', (2,), offset=offset, use_mmap=use_mmap)
    offset += 2 * 4

    xlim = _read_binary(input_file, '>f4', (2,), offset=offset, use_mmap=use_mmap)
    offset += 2 * 4

    # Read dt (time step or grid spacing indicator)
    (dt,) = _read_binary(input_file, '>f4', (1,), offset=offset, use_mmap=use_mmap)
    offset += 4

    # Convert longitudinal limits
    if (xlim[0] < 0) & (xlim[1] < 0) & (dt > 0):
        xlim = xlim + 360.0

    # Calculate grid spacing and coordinates
    dx = (xlim[1] - xlim[0]) / nx
    dy = (ylim[1] - ylim[0]) / ny
    x = np.linspace(xlim[0] + dx / 2.0, xlim[1] - dx / 2.0, nx)
    y = np.linspace(ylim[0] + dy / 2.0, ylim[1] - dy / 2.0, ny)

    # Read number of open boundary points
    (nob,) = _read_binary(input_file, '>i4', (1,), offset=offset, use_mmap=use_mmap)
    nob = int(nob)
    offset += 4

    # Skip padding
    offset += 8

    # Skip open boundary indices
    if nob == 0:
        offset += 4
    else:
        offset += 2 * 4 * nob

    # Skip padding
    offset += 8

    # Read bathymetry
    hz = _read_binary(input_file, '>f4', (ny, nx), offset=offset, use_mmap=use_mmap)
    offset += 4 * nx * ny

    # Skip padding
    offset += 8

    # Read mask
    mz = _read_binary(input_file, '>i4', (ny, nx), offset=offset, use_mmap=use_mmap)
    offset += 4 * nx * ny

    # Update mask for cases where bathymetry is zero or negative
    mz = np.minimum(mz, (hz > 0).astype(np.int32))

    grid['x'] = x
    grid['y'] = y
    grid['hz'] = hz
    grid['mz'] = mz
    grid['xlim'] = xlim
    grid['ylim'] = ylim
    grid['local'] = []

    # Try to read local grids (if file has more data)
    try:
        # Skip padding
        offset += 8

        # Read number of local models
        (n_local,) = _read_binary(input_file, '>i4', (1,), offset=offset, use_mmap=use_mmap)
        n_local = int(n_local)
        offset += 4

        # Read each local model
        for _ in range(n_local):
            local_grid = _read_local_grid(input_file, offset, use_mmap)
            grid['local'].append(local_grid)
            offset = local_grid['_next_offset']

    except (ValueError, IndexError):
        # No local grids or end of file
        pass

    return grid


def _read_local_grid(
    input_file: pathlib.Path,
    offset: int,
    use_mmap: bool = False,
) -> dict:
    """Read a single local grid from ATLAS file"""
    local = {}

    # Skip padding
    offset += 8

    # Read local grid name (20 characters)
    name_bytes = _read_binary(input_file, 'S20', (1,), offset=offset, use_mmap=use_mmap)
    local['name'] = name_bytes[0].decode('utf-8').strip()
    offset += 20

    # Read local grid dimensions
    lnx, lny = _read_binary(input_file, '>i4', (2,), offset=offset, use_mmap=use_mmap)
    lnx, lny = int(lnx), int(lny)
    offset += 2 * 4

    # Read coordinate limits
    lylim = _read_binary(input_file, '>f8', (2,), offset=offset, use_mmap=use_mmap)
    offset += 2 * 8

    lxlim = _read_binary(input_file, '>f8', (2,), offset=offset, use_mmap=use_mmap)
    offset += 2 * 8

    # Calculate local grid spacing and coordinates
    ldx = (lxlim[1] - lxlim[0]) / lnx
    ldy = (lylim[1] - lylim[0]) / lny
    local['x'] = np.linspace(lxlim[0] + ldx / 2.0, lxlim[1] - ldx / 2.0, lnx)
    local['y'] = np.linspace(lylim[0] + ldy / 2.0, lylim[1] - ldy / 2.0, lny)
    local['xlim'] = lxlim
    local['ylim'] = lylim

    # Skip padding
    offset += 8

    # Read bathymetry
    local['hz'] = _read_binary(input_file, '>f4', (lny, lnx), offset=offset, use_mmap=use_mmap)
    offset += 4 * lnx * lny

    # Skip padding
    offset += 8

    # Read mask
    local['mz'] = _read_binary(input_file, '>i4', (lny, lnx), offset=offset, use_mmap=use_mmap)
    offset += 4 * lnx * lny

    local['_next_offset'] = offset

    return local


def read_atlas_elevation(
    input_file: Union[str, pathlib.Path],
    use_mmap: bool = False,
) -> dict:
    """
    Read ATLAS-compact model elevation file

    Parameters
    ----------
    input_file : str or pathlib.Path
        Input ATLAS elevation file
    use_mmap : bool, default False
        Use memory-mapped file reading

    Returns
    -------
    data : dict
        Dictionary containing:
        - hc : complex harmonic constants (nc, ny, nx) or sparse
        - constituents : list of constituent names
        - local : list of local elevation dictionaries (if any)
    """
    input_file = pathlib.Path(input_file).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    data = {}
    offset = 4  # Skip record marker

    # Read dimensions
    header = _read_binary(input_file, '>i4', (4,), offset=offset, use_mmap=use_mmap)
    ll, nx, ny, nc = [int(h) for h in header]
    offset += 4 * 4

    # Read coordinate limits
    ylim = _read_binary(input_file, '>f4', (2,), offset=offset, use_mmap=use_mmap)
    offset += 2 * 4

    xlim = _read_binary(input_file, '>f4', (2,), offset=offset, use_mmap=use_mmap)
    offset += 2 * 4

    # Skip padding
    offset += 8

    # Read constituent names
    const_bytes = _read_binary(input_file, 'S4', (nc,), offset=offset, use_mmap=use_mmap)
    constituents = [c.decode('utf-8').strip().lower() for c in const_bytes]
    data['constituents'] = constituents
    offset += 4 * nc

    # Skip padding
    offset += 8

    # Check if using sparse or dense format based on ll
    if ll == 0:
        # Dense format - read full grid for each constituent
        hc = np.zeros((nc, ny, nx), dtype=np.complex64)
        for i in range(nc):
            real_data = _read_binary(input_file, '>f4', (ny, nx), offset=offset, use_mmap=use_mmap)
            offset += 4 * nx * ny
            offset += 8
            imag_data = _read_binary(input_file, '>f4', (ny, nx), offset=offset, use_mmap=use_mmap)
            offset += 4 * nx * ny
            offset += 8
            hc[i] = real_data + 1j * imag_data
        data['hc'] = hc
        data['sparse'] = False
    else:
        # Sparse format - read valid indices and data
        # Read number of valid points
        (n_valid,) = _read_binary(input_file, '>i4', (1,), offset=offset, use_mmap=use_mmap)
        n_valid = int(n_valid)
        offset += 4

        # Read valid indices (linear indices into grid)
        valid_indices = _read_binary(input_file, '>i4', (n_valid,), offset=offset, use_mmap=use_mmap)
        offset += 4 * n_valid
        offset += 8

        # Read sparse data for each constituent
        hc_sparse = []
        for i in range(nc):
            real_data = _read_binary(input_file, '>f4', (n_valid,), offset=offset, use_mmap=use_mmap)
            offset += 4 * n_valid
            offset += 8
            imag_data = _read_binary(input_file, '>f4', (n_valid,), offset=offset, use_mmap=use_mmap)
            offset += 4 * n_valid
            offset += 8
            hc_sparse.append(real_data + 1j * imag_data)

        data['hc_sparse'] = np.array(hc_sparse)
        data['valid_indices'] = valid_indices
        data['sparse'] = True
        data['shape'] = (ny, nx)

    data['xlim'] = xlim
    data['ylim'] = ylim
    data['local'] = []

    return data


def read_atlas_transport(
    input_file: Union[str, pathlib.Path],
    use_mmap: bool = False,
) -> dict:
    """
    Read ATLAS-compact model transport file

    Parameters
    ----------
    input_file : str or pathlib.Path
        Input ATLAS transport file
    use_mmap : bool, default False
        Use memory-mapped file reading

    Returns
    -------
    data : dict
        Dictionary containing:
        - uc, vc : complex transport constants
        - constituents : list of constituent names
        - local : list of local transport dictionaries (if any)
    """
    input_file = pathlib.Path(input_file).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    data = {}
    offset = 4

    # Read dimensions
    header = _read_binary(input_file, '>i4', (4,), offset=offset, use_mmap=use_mmap)
    ll, nx, ny, nc = [int(h) for h in header]
    offset += 4 * 4

    # Read coordinate limits
    ylim = _read_binary(input_file, '>f4', (2,), offset=offset, use_mmap=use_mmap)
    offset += 2 * 4

    xlim = _read_binary(input_file, '>f4', (2,), offset=offset, use_mmap=use_mmap)
    offset += 2 * 4

    # Skip padding
    offset += 8

    # Read constituent names
    const_bytes = _read_binary(input_file, 'S4', (nc,), offset=offset, use_mmap=use_mmap)
    constituents = [c.decode('utf-8').strip().lower() for c in const_bytes]
    data['constituents'] = constituents
    offset += 4 * nc

    # Skip padding
    offset += 8

    # Dense format - read full grid for each constituent
    uc = np.zeros((nc, ny, nx), dtype=np.complex64)
    vc = np.zeros((nc, ny, nx), dtype=np.complex64)

    for i in range(nc):
        # U component
        u_real = _read_binary(input_file, '>f4', (ny, nx), offset=offset, use_mmap=use_mmap)
        offset += 4 * nx * ny
        offset += 8
        u_imag = _read_binary(input_file, '>f4', (ny, nx), offset=offset, use_mmap=use_mmap)
        offset += 4 * nx * ny
        offset += 8
        uc[i] = u_real + 1j * u_imag

        # V component
        v_real = _read_binary(input_file, '>f4', (ny, nx), offset=offset, use_mmap=use_mmap)
        offset += 4 * nx * ny
        offset += 8
        v_imag = _read_binary(input_file, '>f4', (ny, nx), offset=offset, use_mmap=use_mmap)
        offset += 4 * nx * ny
        offset += 8
        vc[i] = v_real + 1j * v_imag

    data['uc'] = uc
    data['vc'] = vc
    data['xlim'] = xlim
    data['ylim'] = ylim
    data['local'] = []

    return data


def open_atlas_grid(
    grid_file: Union[str, pathlib.Path],
    use_mmap: bool = False,
) -> 'xr.Dataset':
    """
    Read ATLAS grid file and return as xarray Dataset

    Parameters
    ----------
    grid_file : str or pathlib.Path
        Path to ATLAS grid file
    use_mmap : bool, default False
        Use memory-mapped file reading

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing bathymetry and mask
    """
    import xarray as xr

    grid = read_atlas_grid(grid_file, use_mmap=use_mmap)

    ds = xr.Dataset(
        {
            'bathymetry': (['y', 'x'], grid['hz'], {
                'standard_name': 'sea_floor_depth_below_geoid',
                'units': 'm',
                'long_name': 'Bathymetry',
            }),
            'mask': (['y', 'x'], grid['mz'], {
                'standard_name': 'sea_binary_mask',
                'long_name': 'Land-sea mask (1=wet, 0=dry)',
            }),
        },
        coords={
            'x': ('x', grid['x'], {'standard_name': 'longitude', 'units': 'degrees_east'}),
            'y': ('y', grid['y'], {'standard_name': 'latitude', 'units': 'degrees_north'}),
        },
        attrs={
            'title': 'ATLAS Grid',
            'source': str(grid_file),
            'format': 'ATLAS',
            'n_local_grids': len(grid['local']),
        }
    )

    return ds


def open_atlas_elevation(
    elevation_file: Union[str, pathlib.Path],
    grid_file: Optional[Union[str, pathlib.Path]] = None,
    use_mmap: bool = False,
    apply_mask: bool = True,
) -> 'xr.Dataset':
    """
    Read ATLAS elevation file and return as xarray Dataset

    Parameters
    ----------
    elevation_file : str or pathlib.Path
        Path to ATLAS elevation file
    grid_file : str or pathlib.Path, optional
        Path to grid file for coordinates and masking
    use_mmap : bool, default False
        Use memory-mapped file reading
    apply_mask : bool, default True
        Apply land-sea mask to data

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing tidal elevation constituents as complex values
    """
    import xarray as xr

    data = read_atlas_elevation(elevation_file, use_mmap=use_mmap)

    if data['sparse']:
        # Convert sparse to dense
        ny, nx = data['shape']
        hc = np.full((len(data['constituents']), ny, nx), np.nan + 0j, dtype=np.complex64)
        for i, hc_sparse in enumerate(data['hc_sparse']):
            # Unravel indices and fill
            flat_indices = data['valid_indices']
            iy, ix = np.unravel_index(flat_indices, (ny, nx))
            hc[i, iy, ix] = hc_sparse
    else:
        hc = data['hc']

    nc, ny, nx = hc.shape

    # Get grid coordinates if available
    if grid_file is not None:
        grid = read_atlas_grid(grid_file, use_mmap=use_mmap)
        x, y = grid['x'], grid['y']
        if apply_mask:
            mz = grid['mz']
            for i in range(nc):
                hc[i, mz == 0] = np.nan + 0j
    else:
        # Create coordinates from limits
        xlim, ylim = data['xlim'], data['ylim']
        dx = (xlim[1] - xlim[0]) / nx
        dy = (ylim[1] - ylim[0]) / ny
        x = np.linspace(xlim[0] + dx / 2.0, xlim[1] - dx / 2.0, nx)
        y = np.linspace(ylim[0] + dy / 2.0, ylim[1] - dy / 2.0, ny)

    # Create Dataset
    data_vars = {}
    for i, const in enumerate(data['constituents']):
        data_vars[const] = (['y', 'x'], hc[i], {
            'standard_name': f'sea_surface_height_amplitude_due_to_{const}_tidal_constituent',
            'units': 'm',
            'long_name': f'{const.upper()} tidal elevation',
        })

    ds = xr.Dataset(
        data_vars,
        coords={
            'x': ('x', x, {'standard_name': 'longitude', 'units': 'degrees_east'}),
            'y': ('y', y, {'standard_name': 'latitude', 'units': 'degrees_north'}),
        },
        attrs={
            'title': 'ATLAS Tidal Elevation Model',
            'source': str(elevation_file),
            'format': 'ATLAS',
            'constituents': data['constituents'],
        }
    )

    return ds


def open_atlas_transport(
    transport_file: Union[str, pathlib.Path],
    grid_file: Optional[Union[str, pathlib.Path]] = None,
    use_mmap: bool = False,
    apply_mask: bool = True,
    convert_to_velocity: bool = False,
) -> 'xr.Dataset':
    """
    Read ATLAS transport file and return as xarray Dataset

    Parameters
    ----------
    transport_file : str or pathlib.Path
        Path to ATLAS transport file
    grid_file : str or pathlib.Path, optional
        Path to grid file for coordinates and masking
    use_mmap : bool, default False
        Use memory-mapped file reading
    apply_mask : bool, default True
        Apply land-sea mask to data
    convert_to_velocity : bool, default False
        Convert transport (m^2/s) to velocity (m/s)

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing U and V transport/velocity constituents
    """
    import xarray as xr

    data = read_atlas_transport(transport_file, use_mmap=use_mmap)
    uc, vc = data['uc'], data['vc']
    nc, ny, nx = uc.shape

    # Get grid data if available
    hz = None
    if grid_file is not None:
        grid = read_atlas_grid(grid_file, use_mmap=use_mmap)
        x, y = grid['x'], grid['y']
        hz = grid['hz']
        if apply_mask:
            mz = grid['mz']
            for i in range(nc):
                uc[i, mz == 0] = np.nan + 0j
                vc[i, mz == 0] = np.nan + 0j

        if convert_to_velocity and hz is not None:
            hz_safe = np.where(hz > 0, hz, np.nan)
            for i in range(nc):
                uc[i] = uc[i] / hz_safe
                vc[i] = vc[i] / hz_safe
    else:
        xlim, ylim = data['xlim'], data['ylim']
        dx = (xlim[1] - xlim[0]) / nx
        dy = (ylim[1] - ylim[0]) / ny
        x = np.linspace(xlim[0] + dx / 2.0, xlim[1] - dx / 2.0, nx)
        y = np.linspace(ylim[0] + dy / 2.0, ylim[1] - dy / 2.0, ny)

    # Determine units
    if convert_to_velocity:
        units = 'm/s'
        name = 'velocity'
    else:
        units = 'm^2/s'
        name = 'transport'

    # Create Dataset
    data_vars = {}
    for i, const in enumerate(data['constituents']):
        data_vars[f'u_{const}'] = (['y', 'x'], uc[i], {
            'standard_name': f'eastward_sea_water_{name}_due_to_{const}',
            'units': units,
            'long_name': f'{const.upper()} U {name}',
        })
        data_vars[f'v_{const}'] = (['y', 'x'], vc[i], {
            'standard_name': f'northward_sea_water_{name}_due_to_{const}',
            'units': units,
            'long_name': f'{const.upper()} V {name}',
        })

    ds = xr.Dataset(
        data_vars,
        coords={
            'x': ('x', x, {'standard_name': 'longitude', 'units': 'degrees_east'}),
            'y': ('y', y, {'standard_name': 'latitude', 'units': 'degrees_north'}),
        },
        attrs={
            'title': f'ATLAS Tidal {name.capitalize()} Model',
            'source': str(transport_file),
            'format': 'ATLAS',
            'constituents': data['constituents'],
        }
    )

    return ds


def open_dataset(
    model_file: Union[str, pathlib.Path],
    grid_file: Optional[Union[str, pathlib.Path]] = None,
    group: str = 'z',
    use_mmap: bool = False,
    apply_mask: bool = True,
    convert_to_velocity: bool = False,
) -> 'xr.Dataset':
    """
    Open ATLAS tidal model as xarray Dataset

    Parameters
    ----------
    model_file : str or pathlib.Path
        Path to model file
    grid_file : str or pathlib.Path, optional
        Path to grid file
    group : str, default 'z'
        Variable group: 'z' (elevation), 'u', or 'v'
    use_mmap : bool, default False
        Use memory-mapped file reading
    apply_mask : bool, default True
        Apply land-sea mask
    convert_to_velocity : bool, default False
        Convert transport to velocity

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing tidal constituents
    """
    import xarray as xr

    model_file = pathlib.Path(model_file).expanduser().resolve()

    # Auto-detect grid file
    if grid_file is None:
        model_dir = model_file.parent
        for pattern in ['grid*', 'Grid*', 'GRID*']:
            matches = list(model_dir.glob(pattern))
            if matches:
                grid_file = matches[0]
                break

    if group.lower() == 'z':
        return open_atlas_elevation(
            model_file, grid_file=grid_file,
            use_mmap=use_mmap, apply_mask=apply_mask
        )
    elif group.lower() in ['u', 'v']:
        ds = open_atlas_transport(
            model_file, grid_file=grid_file,
            use_mmap=use_mmap, apply_mask=apply_mask,
            convert_to_velocity=convert_to_velocity
        )

        # Filter to requested component
        constituents = ds.attrs['constituents']
        prefix = group.lower() + '_'

        data_vars = {}
        for const in constituents:
            var_name = f'{prefix}{const}'
            if var_name in ds.data_vars:
                data_vars[const] = ds[var_name]

        filtered_ds = xr.Dataset(
            data_vars,
            coords=ds.coords,
            attrs={
                'title': f'ATLAS Tidal {"Velocity" if convert_to_velocity else "Transport"} Model ({group.upper()} component)',
                'source': str(model_file),
                'format': 'ATLAS',
                'constituents': constituents,
                'component': group.lower(),
            }
        )
        return filtered_ds
    else:
        raise ValueError(f"Unknown group: {group}. Must be 'z', 'u', or 'v'")
