"""
pyTMD_turbo.interpolate - Spatial interpolation and extrapolation

Provides interpolation and extrapolation functions for tidal model data.

Functions:
    extrapolate: Nearest-neighbor extrapolation using k-d tree
    bilinear: Bilinear interpolation
    spline: Spline interpolation

Copyright (c) 2024-2026 tkykszk
Derived from pyTMD by Tyler Sutterley (MIT License)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple

import numpy as np

__all__ = [
    'extrapolate',
    'bilinear',
]

# Earth semi-major axis (km) for geographic distance calculations
_EARTH_RADIUS_KM = 6378.137


def _to_cartesian_3d(
    lon: np.ndarray,
    lat: np.ndarray,
    radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert geographic coordinates to 3D Cartesian on a sphere

    Parameters
    ----------
    lon : numpy.ndarray
        Longitude in degrees
    lat : numpy.ndarray
        Latitude in degrees
    radius : float
        Sphere radius (default 1.0 for unit sphere)

    Returns
    -------
    x, y, z : numpy.ndarray
        Cartesian coordinates
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    return x, y, z


def extrapolate(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    fill_value: float = np.nan,
    cutoff: float = np.inf,
    is_geographic: bool = True,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Nearest-neighbor extrapolation using k-d tree

    Extrapolates valid model data to output points using spatial
    nearest-neighbor search with optional distance cutoff.

    Parameters
    ----------
    xs : numpy.ndarray
        x-coordinates of source grid (longitude for geographic)
    ys : numpy.ndarray
        y-coordinates of source grid (latitude for geographic)
    zs : numpy.ndarray
        Source data values (may contain NaN or masked values)
    X : numpy.ndarray
        x-coordinates of output points
    Y : numpy.ndarray
        y-coordinates of output points
    fill_value : float, default np.nan
        Value to use for invalid/masked data
    cutoff : float, default np.inf
        Maximum distance for extrapolation in kilometers.
        Points beyond this distance are filled with fill_value.
    is_geographic : bool, default True
        If True, treat coordinates as geographic (lon/lat) and use
        3D Cartesian distance on sphere. If False, use 2D Euclidean.
    dtype : numpy.dtype, optional
        Output data type (default: same as zs)

    Returns
    -------
    data : numpy.ndarray
        Extrapolated data at output points. Masked array if zs is masked.

    Notes
    -----
    The extrapolation uses scipy.spatial.cKDTree for efficient
    nearest-neighbor search. For geographic coordinates, points are
    converted to 3D Cartesian coordinates on a sphere before distance
    calculation.

    Examples
    --------
    >>> import numpy as np
    >>> # Source grid
    >>> xs = np.linspace(0, 10, 11)
    >>> ys = np.linspace(0, 10, 11)
    >>> zs = np.random.rand(11, 11)
    >>> # Output points
    >>> X = np.array([5.5, 12.0])  # 12.0 is outside grid
    >>> Y = np.array([5.5, 5.0])
    >>> result = extrapolate(xs, ys, zs, X, Y, is_geographic=False)
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError("scipy is required for extrapolate function")

    # Ensure arrays
    xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
    ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
    X = np.atleast_1d(np.asarray(X, dtype=np.float64))
    Y = np.atleast_1d(np.asarray(Y, dtype=np.float64))

    if dtype is None:
        dtype = zs.dtype

    # Handle masked arrays
    if hasattr(zs, 'mask'):
        zs_data = np.asarray(zs.data, dtype=np.float64)
        zs_mask = np.asarray(zs.mask, dtype=bool)
    else:
        zs_data = np.asarray(zs, dtype=np.float64)
        zs_mask = np.zeros(zs_data.shape, dtype=bool)

    # Flatten source data
    zs_flat = zs_data.ravel()
    mask_flat = zs_mask.ravel()

    # Create meshgrid for source coordinates if needed
    if zs_data.ndim == 2 and len(xs) != zs_data.size:
        # Assume xs and ys are 1D coordinate arrays
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        xs_flat = xs_grid.ravel()
        ys_flat = ys_grid.ravel()
    else:
        xs_flat = xs.ravel()
        ys_flat = ys.ravel()

    # Identify valid source points
    valid_mask = ~mask_flat & np.isfinite(zs_flat)

    if not np.any(valid_mask):
        # No valid source points
        output = np.full(len(X), fill_value, dtype=dtype)
        return np.ma.array(output, mask=np.ones(len(X), dtype=bool))

    # Extract valid source data
    xs_valid = xs_flat[valid_mask]
    ys_valid = ys_flat[valid_mask]
    zs_valid = zs_flat[valid_mask]

    if is_geographic:
        # Convert to 3D Cartesian on unit sphere
        src_x, src_y, src_z = _to_cartesian_3d(xs_valid, ys_valid, radius=_EARTH_RADIUS_KM)
        out_x, out_y, out_z = _to_cartesian_3d(X, Y, radius=_EARTH_RADIUS_KM)

        # Build k-d tree from valid source points
        source_points = np.column_stack([src_x, src_y, src_z])
        output_points = np.column_stack([out_x, out_y, out_z])
    else:
        # Use 2D coordinates directly
        source_points = np.column_stack([xs_valid, ys_valid])
        output_points = np.column_stack([X, Y])

    # Build k-d tree
    tree = cKDTree(source_points)

    # Query nearest neighbors
    distances, indices = tree.query(output_points, k=1)

    # Initialize output
    output = np.full(len(X), fill_value, dtype=dtype)
    output_mask = np.ones(len(X), dtype=bool)

    # Apply cutoff
    if is_geographic:
        # For 3D Cartesian, convert cutoff to chord distance
        # Chord distance ≈ 2 * R * sin(d / 2R) where d is arc distance
        # For small angles: chord ≈ arc, so cutoff_3d ≈ cutoff
        cutoff_dist = cutoff
    else:
        cutoff_dist = cutoff

    # Fill valid points
    valid_output = distances <= cutoff_dist
    if np.any(valid_output):
        output[valid_output] = zs_valid[indices[valid_output]]
        output_mask[valid_output] = False

    return np.ma.array(output, mask=output_mask, fill_value=fill_value)


def bilinear(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Bilinear interpolation on regular grid

    Parameters
    ----------
    xs : numpy.ndarray
        x-coordinates of source grid (1D, monotonic)
    ys : numpy.ndarray
        y-coordinates of source grid (1D, monotonic)
    zs : numpy.ndarray
        Source data values, shape (len(ys), len(xs))
    X : numpy.ndarray
        x-coordinates of output points
    Y : numpy.ndarray
        y-coordinates of output points
    fill_value : float, default np.nan
        Value for points outside grid

    Returns
    -------
    data : numpy.ndarray
        Interpolated data at output points

    Examples
    --------
    >>> import numpy as np
    >>> xs = np.linspace(0, 10, 11)
    >>> ys = np.linspace(0, 10, 11)
    >>> zs = np.random.rand(11, 11)
    >>> X = np.array([5.5])
    >>> Y = np.array([5.5])
    >>> result = bilinear(xs, ys, zs, X, Y)
    """
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        raise ImportError("scipy is required for bilinear interpolation")

    xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
    ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
    X = np.atleast_1d(np.asarray(X, dtype=np.float64))
    Y = np.atleast_1d(np.asarray(Y, dtype=np.float64))
    zs = np.asarray(zs, dtype=np.float64)

    # Handle masked arrays
    if hasattr(zs, 'mask'):
        zs_data = np.asarray(zs.data, dtype=np.float64)
        zs_data[zs.mask] = np.nan
    else:
        zs_data = zs

    # Create interpolator
    interp = RegularGridInterpolator(
        (ys, xs), zs_data,
        method='linear',
        bounds_error=False,
        fill_value=fill_value
    )

    # Interpolate
    points = np.column_stack([Y, X])
    result = interp(points)

    return result
