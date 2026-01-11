"""
pyTMD_turbo.io.OTIS - OTIS format tidal model reader

Reads OTIS format tidal solutions provided by Oregon State University and ESR
    http://volkov.oce.orst.edu/tides/region.html
    https://www.esr.org/research/polar-tide-models/list-of-polar-tide-models/

Supports:
- OTIS binary format (big-endian)
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
    'read_raw_binary',
    'read_grid',
    'read_elevation',
    'read_transport',
    'open_dataset',
    'open_otis_grid',
    'open_otis_elevation',
    'open_otis_transport',
    'open_mfdataset',
]


def read_raw_binary(
    path: Union[str, pathlib.Path],
    dtype: Union[np.dtype, str],
    shape: tuple,
    offset: int = 0,
    order: str = "C",
    use_mmap: bool = False,
) -> np.ndarray:
    """
    Read a variable from a raw binary file

    Parameters
    ----------
    path : str or pathlib.Path
        Path to input file
    dtype : numpy.dtype or str
        Variable data type
    shape : tuple
        Shape of the data
    offset : int, default 0
        Offset to apply on read
    order : str, default 'C'
        Memory layout of array
    use_mmap : bool, default False
        Use memory-mapped file reading for large files

    Returns
    -------
    var : numpy.ndarray
        Data variable
    """
    path = pathlib.Path(path).expanduser().resolve()
    dtype = np.dtype(dtype)
    count = int(np.prod(shape))

    if use_mmap:
        # Memory-mapped reading
        var = np.memmap(
            path,
            dtype=dtype,
            mode='r',
            offset=offset,
            shape=shape,
            order=order,
        )
        # Convert to regular array to avoid file handle issues
        var = np.array(var)
    else:
        with open(path, mode="rb") as fid:
            var = np.fromfile(
                fid, dtype=dtype, offset=offset, count=count
            )
            var = var.reshape(shape, order=order)
    return var


def read_grid(
    input_file: Union[str, pathlib.Path],
    use_mmap: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read OTIS model grid file

    Parameters
    ----------
    input_file : str or pathlib.Path
        Input OTIS grid file
    use_mmap : bool, default False
        Use memory-mapped file reading

    Returns
    -------
    x : numpy.ndarray
        Longitude coordinates
    y : numpy.ndarray
        Latitude coordinates
    hz : numpy.ndarray
        Bathymetry (water depth)
    mz : numpy.ndarray
        Mask (1: wet, 0: dry)
    """
    input_file = pathlib.Path(input_file).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Initial offset (skip 4 bytes)
    offset = 4

    # Read data as big endian
    # Get model dimensions
    nx, ny = read_raw_binary(
        input_file,
        dtype=np.dtype(">i4"),
        shape=(2,),
        offset=offset,
        use_mmap=use_mmap,
    )
    offset += 2 * 4

    # Extract x and y limits
    ylim = read_raw_binary(
        input_file,
        dtype=np.dtype(">f4"),
        shape=(2,),
        offset=offset,
        use_mmap=use_mmap,
    )
    offset += 2 * 4

    xlim = read_raw_binary(
        input_file,
        dtype=np.dtype(">f4"),
        shape=(2,),
        offset=offset,
        use_mmap=use_mmap,
    )
    offset += 2 * 4

    # Read dt from file
    (dt,) = read_raw_binary(
        input_file,
        dtype=np.dtype(">f4"),
        shape=(1,),
        offset=offset,
        use_mmap=use_mmap,
    )
    offset += 4

    # Convert longitudinal limits (if x == longitude)
    if (xlim[0] < 0) & (xlim[1] < 0) & (dt > 0):
        xlim = xlim + 360.0

    # X and y coordinate spacing
    dx = (xlim[1] - xlim[0]) / nx
    dy = (ylim[1] - ylim[0]) / ny

    # Create x and y arrays
    x = np.linspace(xlim[0] + dx / 2.0, xlim[1] - dx / 2.0, int(nx))
    y = np.linspace(ylim[0] + dy / 2.0, ylim[1] - dy / 2.0, int(ny))

    # Read nob from file
    (nob,) = read_raw_binary(
        input_file,
        dtype=np.dtype(">i4"),
        shape=(1,),
        offset=offset,
        use_mmap=use_mmap,
    )
    offset += 4

    # Skip 8 bytes
    offset += 8

    # Read iob from file
    if nob == 0:
        offset += 4
    else:
        offset += 2 * 4 * int(nob)

    # Skip 8 bytes
    offset += 8

    # Read hz matrix (bathymetry)
    hz = read_raw_binary(
        input_file,
        dtype=">f4",
        shape=(int(ny), int(nx)),
        offset=offset,
        order="C",
        use_mmap=use_mmap,
    )
    offset += 4 * int(nx) * int(ny)

    # Skip 8 bytes
    offset += 8

    # Read mz matrix (1: wet point, 0: dry point)
    mz = read_raw_binary(
        input_file,
        dtype=">i4",
        shape=(int(ny), int(nx)),
        offset=offset,
        order="C",
        use_mmap=use_mmap,
    )

    # Update mask for cases where bathymetry is zero or negative
    mz = np.minimum(mz, (hz > 0).astype(np.int32))

    return x, y, hz, mz


def read_elevation(
    input_file: Union[str, pathlib.Path],
    grid_file: Optional[Union[str, pathlib.Path]] = None,
    use_mmap: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Read OTIS model elevation file

    Parameters
    ----------
    input_file : str or pathlib.Path
        Input OTIS elevation file (h_*)
    grid_file : str or pathlib.Path, optional
        Grid file for dimensions (if not provided, read from elevation file)
    use_mmap : bool, default False
        Use memory-mapped file reading

    Returns
    -------
    hc : numpy.ndarray
        Complex harmonic constants, shape (nc, ny, nx)
    constituents : list
        List of constituent names
    """
    input_file = pathlib.Path(input_file).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Initial offset
    offset = 4

    # Read header
    header = read_raw_binary(
        input_file,
        dtype=np.dtype(">i4"),
        shape=(3,),
        offset=offset,
        use_mmap=use_mmap,
    )
    nx, ny, nc = header
    nx, ny, nc = int(nx), int(ny), int(nc)
    offset += 3 * 4

    # Skip 8 bytes
    offset += 8

    # Read constituent names
    const_bytes = read_raw_binary(
        input_file,
        dtype="S4",
        shape=(nc,),
        offset=offset,
        use_mmap=use_mmap,
    )
    constituents = [c.decode('utf-8').strip().lower() for c in const_bytes]
    offset += 4 * nc

    # Skip 8 bytes
    offset += 8

    # Read complex harmonic constants
    hc = np.zeros((nc, ny, nx), dtype=np.complex64)

    for i in range(nc):
        # Read real and imaginary parts interleaved
        data = read_raw_binary(
            input_file,
            dtype=">f4",
            shape=(ny, nx * 2),
            offset=offset,
            order="C",
            use_mmap=use_mmap,
        )
        # Convert to complex
        hc[i, :, :] = data[:, 0::2] + 1j * data[:, 1::2]
        offset += 4 * nx * ny * 2
        # Skip 8 bytes between constituents
        offset += 8

    return hc, constituents


def read_transport(
    input_file: Union[str, pathlib.Path],
    use_mmap: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read OTIS model transport file

    Parameters
    ----------
    input_file : str or pathlib.Path
        Input OTIS transport file (UV_*)
    use_mmap : bool, default False
        Use memory-mapped file reading

    Returns
    -------
    uc : numpy.ndarray
        Complex U transport, shape (nc, ny, nx)
    vc : numpy.ndarray
        Complex V transport, shape (nc, ny, nx)
    constituents : list
        List of constituent names
    """
    input_file = pathlib.Path(input_file).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Initial offset
    offset = 4

    # Read header
    header = read_raw_binary(
        input_file,
        dtype=np.dtype(">i4"),
        shape=(3,),
        offset=offset,
        use_mmap=use_mmap,
    )
    nx, ny, nc = header
    nx, ny, nc = int(nx), int(ny), int(nc)
    offset += 3 * 4

    # Skip 8 bytes
    offset += 8

    # Read constituent names
    const_bytes = read_raw_binary(
        input_file,
        dtype="S4",
        shape=(nc,),
        offset=offset,
        use_mmap=use_mmap,
    )
    constituents = [c.decode('utf-8').strip().lower() for c in const_bytes]
    offset += 4 * nc

    # Skip 8 bytes
    offset += 8

    # Read complex transport constants
    uc = np.zeros((nc, ny, nx), dtype=np.complex64)
    vc = np.zeros((nc, ny, nx), dtype=np.complex64)

    for i in range(nc):
        # Read U component
        data_u = read_raw_binary(
            input_file,
            dtype=">f4",
            shape=(ny, nx * 2),
            offset=offset,
            order="C",
            use_mmap=use_mmap,
        )
        uc[i, :, :] = data_u[:, 0::2] + 1j * data_u[:, 1::2]
        offset += 4 * nx * ny * 2
        offset += 8

        # Read V component
        data_v = read_raw_binary(
            input_file,
            dtype=">f4",
            shape=(ny, nx * 2),
            offset=offset,
            order="C",
            use_mmap=use_mmap,
        )
        vc[i, :, :] = data_v[:, 0::2] + 1j * data_v[:, 1::2]
        offset += 4 * nx * ny * 2
        offset += 8

    return uc, vc, constituents


def open_otis_grid(
    grid_file: Union[str, pathlib.Path],
    use_mmap: bool = False,
) -> 'xr.Dataset':
    """
    Read OTIS grid file and return as xarray Dataset

    Parameters
    ----------
    grid_file : str or pathlib.Path
        Path to OTIS grid file
    use_mmap : bool, default False
        Use memory-mapped file reading

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing bathymetry and mask
    """
    import xarray as xr

    x, y, hz, mz = read_grid(grid_file, use_mmap=use_mmap)

    ds = xr.Dataset(
        {
            'bathymetry': (['y', 'x'], hz, {
                'standard_name': 'sea_floor_depth_below_geoid',
                'units': 'm',
                'long_name': 'Bathymetry',
            }),
            'mask': (['y', 'x'], mz, {
                'standard_name': 'sea_binary_mask',
                'long_name': 'Land-sea mask (1=wet, 0=dry)',
            }),
        },
        coords={
            'x': ('x', x, {'standard_name': 'longitude', 'units': 'degrees_east'}),
            'y': ('y', y, {'standard_name': 'latitude', 'units': 'degrees_north'}),
        },
        attrs={
            'title': 'OTIS Grid',
            'source': str(grid_file),
            'format': 'OTIS',
        }
    )

    return ds


def open_otis_elevation(
    elevation_file: Union[str, pathlib.Path],
    grid_file: Optional[Union[str, pathlib.Path]] = None,
    use_mmap: bool = False,
    apply_mask: bool = True,
) -> 'xr.Dataset':
    """
    Read OTIS elevation file and return as xarray Dataset

    Parameters
    ----------
    elevation_file : str or pathlib.Path
        Path to OTIS elevation file (h_*)
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

    hc, constituents = read_elevation(elevation_file, use_mmap=use_mmap)
    nc, ny, nx = hc.shape

    # Get grid coordinates if available
    if grid_file is not None:
        x, y, hz, mz = read_grid(grid_file, use_mmap=use_mmap)
        if apply_mask:
            # Apply mask to data (set land points to NaN)
            for i in range(nc):
                hc[i, mz == 0] = np.nan + 0j
    else:
        # Create default coordinates
        x = np.arange(nx, dtype=np.float64)
        y = np.arange(ny, dtype=np.float64)
        mz = None

    # Create Dataset with each constituent as a variable
    data_vars = {}
    for i, const in enumerate(constituents):
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
            'title': 'OTIS Tidal Elevation Model',
            'source': str(elevation_file),
            'format': 'OTIS',
            'constituents': constituents,
        }
    )

    return ds


def open_otis_transport(
    transport_file: Union[str, pathlib.Path],
    grid_file: Optional[Union[str, pathlib.Path]] = None,
    use_mmap: bool = False,
    apply_mask: bool = True,
    convert_to_velocity: bool = False,
) -> 'xr.Dataset':
    """
    Read OTIS transport file and return as xarray Dataset

    Parameters
    ----------
    transport_file : str or pathlib.Path
        Path to OTIS transport file (UV_*)
    grid_file : str or pathlib.Path, optional
        Path to grid file for coordinates and masking
    use_mmap : bool, default False
        Use memory-mapped file reading
    apply_mask : bool, default True
        Apply land-sea mask to data
    convert_to_velocity : bool, default False
        Convert transport (m^2/s) to velocity (m/s) by dividing by bathymetry

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing U and V transport/velocity constituents
    """
    import xarray as xr

    uc, vc, constituents = read_transport(transport_file, use_mmap=use_mmap)
    nc, ny, nx = uc.shape

    # Get grid data if available
    hz = None
    if grid_file is not None:
        x, y, hz, mz = read_grid(grid_file, use_mmap=use_mmap)
        if apply_mask:
            for i in range(nc):
                uc[i, mz == 0] = np.nan + 0j
                vc[i, mz == 0] = np.nan + 0j

        if convert_to_velocity and hz is not None:
            # Convert transport to velocity
            hz_safe = np.where(hz > 0, hz, np.nan)
            for i in range(nc):
                uc[i] = uc[i] / hz_safe
                vc[i] = vc[i] / hz_safe
    else:
        x = np.arange(nx, dtype=np.float64)
        y = np.arange(ny, dtype=np.float64)

    # Determine units based on conversion
    if convert_to_velocity:
        u_units = 'm/s'
        v_units = 'm/s'
        u_name = 'velocity'
        v_name = 'velocity'
    else:
        u_units = 'm^2/s'
        v_units = 'm^2/s'
        u_name = 'transport'
        v_name = 'transport'

    # Create separate datasets for U and V
    data_vars = {}
    for i, const in enumerate(constituents):
        data_vars[f'u_{const}'] = (['y', 'x'], uc[i], {
            'standard_name': f'eastward_sea_water_{u_name}_due_to_{const}',
            'units': u_units,
            'long_name': f'{const.upper()} U {u_name}',
        })
        data_vars[f'v_{const}'] = (['y', 'x'], vc[i], {
            'standard_name': f'northward_sea_water_{v_name}_due_to_{const}',
            'units': v_units,
            'long_name': f'{const.upper()} V {v_name}',
        })

    ds = xr.Dataset(
        data_vars,
        coords={
            'x': ('x', x, {'standard_name': 'longitude', 'units': 'degrees_east'}),
            'y': ('y', y, {'standard_name': 'latitude', 'units': 'degrees_north'}),
        },
        attrs={
            'title': f'OTIS Tidal {u_name.capitalize()} Model',
            'source': str(transport_file),
            'format': 'OTIS',
            'constituents': constituents,
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
    Open OTIS tidal model as xarray Dataset

    Universal opener for OTIS binary format files.

    Parameters
    ----------
    model_file : str or pathlib.Path
        Path to model file (elevation h_* or transport UV_*)
    grid_file : str or pathlib.Path, optional
        Path to grid file. If None, will look for grid file in same directory
    group : str, default 'z'
        Variable group to read:
        - 'z': tidal elevation
        - 'u' or 'U': zonal (eastward) component
        - 'v' or 'V': meridional (northward) component
    use_mmap : bool, default False
        Use memory-mapped file reading for large files
    apply_mask : bool, default True
        Apply land-sea mask to data
    convert_to_velocity : bool, default False
        Convert transport to velocity (only for u/v groups)

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing tidal constituents
    """
    import xarray as xr

    model_file = pathlib.Path(model_file).expanduser().resolve()

    # Auto-detect grid file if not provided
    if grid_file is None:
        # Look for common grid file names
        model_dir = model_file.parent
        grid_patterns = ['grid*', 'Grid*', 'GRID*', '*.grid']
        for pattern in grid_patterns:
            matches = list(model_dir.glob(pattern))
            if matches:
                grid_file = matches[0]
                break

    if group.lower() == 'z':
        return open_otis_elevation(
            model_file,
            grid_file=grid_file,
            use_mmap=use_mmap,
            apply_mask=apply_mask,
        )
    elif group.lower() in ['u', 'v']:
        ds = open_otis_transport(
            model_file,
            grid_file=grid_file,
            use_mmap=use_mmap,
            apply_mask=apply_mask,
            convert_to_velocity=convert_to_velocity,
        )

        # Filter to only requested component
        constituents = ds.attrs['constituents']
        prefix = group.lower() + '_'

        # Create new dataset with only requested component
        data_vars = {}
        for const in constituents:
            var_name = f'{prefix}{const}'
            if var_name in ds.data_vars:
                # Rename to just constituent name for consistency
                data_vars[const] = ds[var_name]

        filtered_ds = xr.Dataset(
            data_vars,
            coords=ds.coords,
            attrs={
                'title': f'OTIS Tidal {"Velocity" if convert_to_velocity else "Transport"} Model ({group.upper()} component)',
                'source': str(model_file),
                'format': 'OTIS',
                'constituents': constituents,
                'component': group.lower(),
            }
        )
        return filtered_ds
    else:
        raise ValueError(f"Unknown group: {group}. Must be 'z', 'u', or 'v'")


def open_mfdataset(
    model_files: List[Union[str, pathlib.Path]],
    grid_file: Optional[Union[str, pathlib.Path]] = None,
    group: str = 'z',
    use_mmap: bool = False,
    apply_mask: bool = True,
    parallel: bool = False,
) -> 'xr.Dataset':
    """
    Open multiple OTIS files and merge into single Dataset

    Parameters
    ----------
    model_files : list
        List of model file paths (one constituent per file)
    grid_file : str or pathlib.Path, optional
        Path to grid file
    group : str, default 'z'
        Variable group to read ('z', 'u', or 'v')
    use_mmap : bool, default False
        Use memory-mapped file reading
    apply_mask : bool, default True
        Apply land-sea mask
    parallel : bool, default False
        Use parallel reading (requires dask)

    Returns
    -------
    ds : xarray.Dataset
        Merged dataset containing all constituents
    """
    import xarray as xr

    if parallel:
        try:
            from dask import delayed
            from dask.diagnostics import ProgressBar

            @delayed
            def _load_file(f):
                return open_dataset(
                    f, grid_file=grid_file, group=group,
                    use_mmap=use_mmap, apply_mask=apply_mask
                )

            delayed_datasets = [_load_file(f) for f in model_files]

            with ProgressBar():
                datasets = [ds.compute() for ds in delayed_datasets]
        except ImportError:
            # Fall back to sequential reading
            parallel = False

    if not parallel:
        datasets = []
        for f in model_files:
            ds = open_dataset(
                f, grid_file=grid_file, group=group,
                use_mmap=use_mmap, apply_mask=apply_mask
            )
            datasets.append(ds)

    # Merge all datasets
    merged = xr.merge(datasets)

    # Update attributes
    all_constituents = []
    for ds in datasets:
        all_constituents.extend(ds.attrs.get('constituents', []))

    merged.attrs['constituents'] = list(set(all_constituents))
    merged.attrs['title'] = f'OTIS Tidal Model ({group.upper()} component)'
    merged.attrs['format'] = 'OTIS'

    return merged
