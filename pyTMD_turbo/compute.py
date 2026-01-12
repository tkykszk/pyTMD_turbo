"""
pytmd_turbo.compute - PyTMD-compatible tidal prediction API

Provides the same interface as pyTMD.compute.tide_elevations,
whilst using high-speed batch processing internally.

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .predict.cache_optimized import OceanTideCacheOptimized

# Global cache (model is loaded only once)
_cache: Optional[OceanTideCacheOptimized] = None
_loaded_models: set = set()

# Default model directory
_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "LargeFiles"


def datetime_to_mjd(dt: datetime) -> float:
    """Convert datetime to Modified Julian Day (MJD)"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    unix_timestamp = dt.timestamp()
    return unix_timestamp / 86400.0 + 40587.0


def init_model(model: str = 'GOT5.5', directory: Optional[str] = None) -> None:
    """
    Initialise model (for explicit invocation)

    Parameters
    ----------
    model : str
        Model name ('GOT5.5', 'GOT5.6', etc.)
    directory : str, optional
        Model data directory
    """
    global _cache, _loaded_models

    if directory is None:
        directory = str(_DEFAULT_MODEL_DIR)

    if _cache is None:
        _cache = OceanTideCacheOptimized()

    if model not in _loaded_models:
        _cache.load_model(model, directory)
        _loaded_models.add(model)


def _ensure_model_loaded(model: str, directory: Optional[str] = None) -> None:
    """Load model if not already loaded"""
    if model not in _loaded_models:
        init_model(model, directory)


def tide_elevations(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    model: str = 'GOT5.5',
    directory: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    PyTMD-compatible tide elevation calculation API

    Parameters
    ----------
    x : np.ndarray
        Array of longitudes (degrees)
    y : np.ndarray
        Array of latitudes (degrees)
    times : np.ndarray
        Array of times (datetime64 or datetime)
    model : str
        Model name (default: 'GOT5.5')
    directory : str, optional
        Model data directory
    **kwargs
        Dummy arguments for PyTMD compatibility (ignored)

    Returns
    -------
    np.ndarray
        Tide elevation (metres)

    Notes
    -----
    Differences from PyTMD:
    - Uses batch processing internally for speed
    - Return value is a 1D array (tide elevation at each time)

    Examples
    --------
    >>> import fast_pytmd as pyTMD
    >>> import numpy as np
    >>> x = np.array([140.0, 150.0])
    >>> y = np.array([35.0, 30.0])
    >>> times = np.array(['2020-01-01T00:00:00', '2020-01-01T01:00:00'], dtype='datetime64')
    >>> tide = pyTMD.compute.tide_elevations(x, y, times)
    """
    if kwargs:
        warnings.warn(
            f"pyTMD_turbo.tide_elevations: ignoring unsupported kwargs: {list(kwargs.keys())}. "
            "These parameters are accepted for PyTMD compatibility but have no effect.",
            UserWarning,
            stacklevel=2
        )
    _ensure_model_loaded(model, directory)

    # Convert times to MJD
    if isinstance(times[0], np.datetime64):
        # numpy datetime64 -> MJD
        # Convert datetime64 to Unix timestamp
        times_unix = times.astype('datetime64[s]').astype('float64')
        mjd = times_unix / 86400.0 + 40587.0
    elif isinstance(times[0], datetime):
        mjd = np.array([datetime_to_mjd(t) for t in times])
    else:
        # Assume already MJD
        mjd = np.asarray(times)

    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))
    mjd = np.atleast_1d(mjd)

    # Determine input format
    # PyTMD type='drift' equivalent: calculate for each (x[i], y[i], times[i])
    if len(x) == len(mjd) and len(x) > 1:
        # Drift format: each point has a different time
        # Group by unique coordinates and calculate
        unique_coords = {}
        for i, (lon, lat) in enumerate(zip(x, y)):
            key = (lat, lon)
            if key not in unique_coords:
                unique_coords[key] = []
            unique_coords[key].append(i)

        result = np.zeros(len(x))
        for (lat, lon), indices in unique_coords.items():
            mjd_subset = mjd[indices]
            tide = _cache.predict_single(model, lat, lon, mjd_subset)
            result[indices] = tide

        return result
    else:
        # Grid format: all coordinates × all times
        tide = _cache.predict_batch(model, y, x, mjd)
        # If single location, return 1D array
        if len(x) == 1:
            return tide[0]
        return tide


def predict_batch(
    lats: np.ndarray,
    lons: np.ndarray,
    mjd: np.ndarray,
    model: str = 'GOT5.5',
    directory: Optional[str] = None,
    apply_nodal: bool = True
) -> np.ndarray:
    """
    High-speed batch prediction API

    Parameters
    ----------
    lats : np.ndarray
        Array of latitudes (degrees)
    lons : np.ndarray
        Array of longitudes (degrees)
    mjd : np.ndarray
        Array of Modified Julian Days
    model : str
        Model name
    directory : str, optional
        Model data directory
    apply_nodal : bool
        Whether to apply nodal corrections

    Returns
    -------
    np.ndarray
        Tide elevation (metres), shape (n_points, n_times)

    Examples
    --------
    >>> import fast_pytmd as pyTMD
    >>> import numpy as np
    >>> lats = np.array([35.0, 30.0, 25.0])
    >>> lons = np.array([140.0, 150.0, 160.0])
    >>> mjd = np.arange(59000.0, 59001.0, 1/24)  # 1 day
    >>> tide = pyTMD.predict_batch(lats, lons, mjd)
    >>> tide.shape
    (3, 24)
    """
    _ensure_model_loaded(model, directory)
    return _cache.predict_batch(model, lats, lons, mjd, apply_nodal)


def predict_single(
    lat: float,
    lon: float,
    mjd: np.ndarray,
    model: str = 'GOT5.5',
    directory: Optional[str] = None,
    apply_nodal: bool = True
) -> np.ndarray:
    """
    Tide prediction for a single location

    Parameters
    ----------
    lat : float
        Latitude (degrees)
    lon : float
        Longitude (degrees)
    mjd : np.ndarray
        Array of Modified Julian Days
    model : str
        Model name
    directory : str, optional
        Model data directory
    apply_nodal : bool
        Whether to apply nodal corrections

    Returns
    -------
    np.ndarray
        Tide elevation (metres), shape (n_times,)
    """
    _ensure_model_loaded(model, directory)
    return _cache.predict_single(model, lat, lon, mjd, apply_nodal)


# Separate caches for u and v components
_cache_u: Optional[OceanTideCacheOptimized] = None
_cache_v: Optional[OceanTideCacheOptimized] = None
_loaded_models_u: set = set()
_loaded_models_v: set = set()


def _ensure_current_model_loaded(model: str, directory: Optional[str] = None) -> None:
    """Load u and v components of model if not already loaded"""
    global _cache_u, _cache_v, _loaded_models_u, _loaded_models_v

    if directory is None:
        directory = str(_DEFAULT_MODEL_DIR)

    # Load u component
    if model not in _loaded_models_u:
        if _cache_u is None:
            _cache_u = OceanTideCacheOptimized()
        try:
            _cache_u.load_model(model, directory, group='u')
            _loaded_models_u.add(model)
        except (ValueError, FileNotFoundError) as e:
            raise ValueError(f"Model '{model}' does not have current (u) data: {e}") from e

    # Load v component
    if model not in _loaded_models_v:
        if _cache_v is None:
            _cache_v = OceanTideCacheOptimized()
        try:
            _cache_v.load_model(model, directory, group='v')
            _loaded_models_v.add(model)
        except (ValueError, FileNotFoundError) as e:
            raise ValueError(f"Model '{model}' does not have current (v) data: {e}") from e


def SET_displacements(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    h: Optional[np.ndarray] = None,
    ellipsoid: str = 'WGS84',
    tide_system: str = 'tide_free',
    coordinate_system: str = 'geographic',
    **kwargs
) -> tuple:
    """
    Calculate Solid Earth Tide (SET) displacements

    Computes the solid Earth body tide displacement following
    IERS Conventions 2010.

    Parameters
    ----------
    x : np.ndarray
        Array of longitudes (degrees)
    y : np.ndarray
        Array of latitudes (degrees)
    times : np.ndarray
        Array of times (datetime64, datetime, or MJD)
    h : np.ndarray, optional
        Height above ellipsoid (meters), default is 0
    ellipsoid : str, default 'WGS84'
        Reference ellipsoid name
    tide_system : str, default 'tide_free'
        Tide system: 'tide_free' or 'mean_tide'
    coordinate_system : str, default 'geographic'
        Output coordinate system:
        - 'geographic': returns (North, East, Up) displacements
        - 'cartesian': returns (dX, dY, dZ) ECEF displacements

    Returns
    -------
    If coordinate_system == 'geographic':
        dn : np.ndarray
            North displacement (meters)
        de : np.ndarray
            East displacement (meters)
        du : np.ndarray
            Up displacement (meters)
    If coordinate_system == 'cartesian':
        dx : np.ndarray
            X displacement (meters)
        dy : np.ndarray
            Y displacement (meters)
        dz : np.ndarray
            Z displacement (meters)

    Notes
    -----
    - Uses the solid_earth_tide() function from predict.solid_earth module
    - Computes solar and lunar ephemeris internally
    - Output shape is (n_points, n_times) for multiple points/times,
      or (n_times,) for single point

    Examples
    --------
    >>> import pyTMD_turbo as pytmd
    >>> import numpy as np
    >>> x = np.array([140.0])  # Tokyo longitude
    >>> y = np.array([35.0])   # Tokyo latitude
    >>> times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')
    >>> dn, de, du = pytmd.SET_displacements(x, y, times)
    """
    from .astro.ephemeris import lunar_ecef, solar_ecef
    from .predict.solid_earth import ecef_to_enu_rotation, solid_earth_tide
    from .spatial import to_cartesian

    # Convert times to MJD
    if isinstance(times[0], np.datetime64):
        times_unix = times.astype('datetime64[s]').astype('float64')
        mjd = times_unix / 86400.0 + 40587.0
    elif isinstance(times[0], datetime):
        mjd = np.array([datetime_to_mjd(t) for t in times])
    else:
        mjd = np.asarray(times)

    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    mjd = np.atleast_1d(mjd)

    if h is None:
        h = np.zeros_like(x)
    else:
        h = np.atleast_1d(np.asarray(h, dtype=np.float64))

    n_points = len(x)
    n_times = len(mjd)

    # Convert geographic to ECEF coordinates
    xyz_x, xyz_y, xyz_z = to_cartesian(x, y, h, ellipsoid=ellipsoid)
    xyz = np.column_stack([xyz_x, xyz_y, xyz_z])  # Shape: (n_points, 3)

    # Compute solar and lunar ECEF coordinates
    sun_x, sun_y, sun_z = solar_ecef(mjd)
    moon_x, moon_y, moon_z = lunar_ecef(mjd)
    sun_xyz = np.column_stack([sun_x, sun_y, sun_z])   # Shape: (n_times, 3)
    moon_xyz = np.column_stack([moon_x, moon_y, moon_z])  # Shape: (n_times, 3)

    # Reference time: MJD 48622.0 = 1992-01-01T00:00:00
    MJD_1992 = 48622.0
    t = mjd - MJD_1992  # Days since 1992-01-01

    # Compute solid Earth tide displacement in ECEF
    dx, dy, dz = solid_earth_tide(
        t, xyz, sun_xyz, moon_xyz,
        tide_system=tide_system,
    )

    if coordinate_system.lower() == 'cartesian':
        # Return ECEF displacements
        if n_points == 1:
            return dx[0], dy[0], dz[0]
        return dx, dy, dz

    # Convert to local (North, East, Up) coordinates using vectorized approach
    lat_rad = np.radians(y)
    lon_rad = np.radians(x)

    # Optimized path for single point (most common case in earthquake analysis)
    if n_points == 1:
        R = ecef_to_enu_rotation(lat_rad[0], lon_rad[0])
        # Stack all times: (3, n_times)
        d_ecef = np.stack([dx[0, :], dy[0, :], dz[0, :]], axis=0)
        # Matrix multiply: (3, 3) @ (3, n_times) = (3, n_times)
        d_enu = R @ d_ecef
        return d_enu[1], d_enu[0], d_enu[2]  # North, East, Up

    # Multiple points: vectorized ECEF->ENU conversion
    dn = np.zeros((n_points, n_times))
    de = np.zeros((n_points, n_times))
    du = np.zeros((n_points, n_times))

    for i_p in range(n_points):
        R = ecef_to_enu_rotation(lat_rad[i_p], lon_rad[i_p])
        # Process all times at once: (3, n_times)
        d_ecef = np.stack([dx[i_p, :], dy[i_p, :], dz[i_p, :]], axis=0)
        d_enu = R @ d_ecef  # (3, n_times)
        de[i_p, :] = d_enu[0]  # East
        dn[i_p, :] = d_enu[1]  # North
        du[i_p, :] = d_enu[2]  # Up

    return dn, de, du


def tide_currents(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    model: str = 'TPXO9-atlas-v5',
    directory: Optional[str] = None,
    **kwargs
) -> tuple:
    """
    Calculate tidal currents (u, v velocity components)

    Parameters
    ----------
    x : np.ndarray
        Array of longitudes (degrees)
    y : np.ndarray
        Array of latitudes (degrees)
    times : np.ndarray
        Array of times (datetime64 or datetime)
    model : str
        Model name (must have u/v components, e.g., 'TPXO9-atlas-v5')
    directory : str, optional
        Model data directory
    **kwargs
        Dummy arguments for compatibility (ignored)

    Returns
    -------
    u : np.ndarray
        Zonal (east-west) tidal current velocity (m/s or m²/s depending on model)
    v : np.ndarray
        Meridional (north-south) tidal current velocity (m/s or m²/s)

    Notes
    -----
    - Most OTIS/ATLAS models store transport (m²/s). Divide by depth to get velocity.
    - GOT models typically don't include current data.

    Examples
    --------
    >>> import pyTMD_turbo as pytmd
    >>> import numpy as np
    >>> x = np.array([140.0])
    >>> y = np.array([35.0])
    >>> times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')
    >>> u, v = pytmd.tide_currents(x, y, times, model='TPXO9-atlas-v5')
    """
    if kwargs:
        warnings.warn(
            f"pyTMD_turbo.tide_currents: ignoring unsupported kwargs: {list(kwargs.keys())}. "
            "These parameters are accepted for compatibility but have no effect.",
            UserWarning,
            stacklevel=2
        )
    _ensure_current_model_loaded(model, directory)

    # Convert times to MJD
    if isinstance(times[0], np.datetime64):
        times_unix = times.astype('datetime64[s]').astype('float64')
        mjd = times_unix / 86400.0 + 40587.0
    elif isinstance(times[0], datetime):
        mjd = np.array([datetime_to_mjd(t) for t in times])
    else:
        mjd = np.asarray(times)

    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))
    mjd = np.atleast_1d(mjd)

    # Determine input format
    if len(x) == len(mjd) and len(x) > 1:
        # Drift format: each point has a different time
        unique_coords = {}
        for i, (lon, lat) in enumerate(zip(x, y)):
            key = (lat, lon)
            if key not in unique_coords:
                unique_coords[key] = []
            unique_coords[key].append(i)

        result_u = np.zeros(len(x))
        result_v = np.zeros(len(x))

        for (lat, lon), indices in unique_coords.items():
            mjd_subset = mjd[indices]
            u_tide = _cache_u.predict_single(model, lat, lon, mjd_subset)
            v_tide = _cache_v.predict_single(model, lat, lon, mjd_subset)
            result_u[indices] = u_tide
            result_v[indices] = v_tide

        return result_u, result_v
    else:
        # Grid format
        u_tide = _cache_u.predict_batch(model, y, x, mjd)
        v_tide = _cache_v.predict_batch(model, y, x, mjd)

        if len(x) == 1:
            return u_tide[0], v_tide[0]
        return u_tide, v_tide


def LPET_elevations(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Calculate Long-Period Equilibrium Tide elevations

    Computes the quasi-static equilibrium response of the ocean to
    the tidal potential of the moon and sun using Cartwright-Tayler-Edden
    tables.

    Parameters
    ----------
    x : np.ndarray
        Longitudes in degrees
    y : np.ndarray
        Latitudes in degrees
    times : np.ndarray
        Times (datetime64, datetime, or MJD)
    **kwargs
        Additional arguments:
        - deltat : float - TT-UT1 correction in days
        - h2 : float - Degree-2 Love number (default 0.606)
        - k2 : float - Degree-2 potential Love number (default 0.299)

    Returns
    -------
    lpet : np.ndarray
        Long-period equilibrium tide elevation in meters

    Examples
    --------
    >>> import pyTMD_turbo as pytmd
    >>> import numpy as np
    >>> x = np.array([140.0])
    >>> y = np.array([35.0])
    >>> times = np.arange('2024-01-01', '2024-01-31', dtype='datetime64[D]')
    >>> lpet = pytmd.LPET_elevations(x, y, times)
    """
    from .predict.equilibrium import LPET_elevations as _lpet
    return _lpet(x, y, times, **kwargs)


def tide_masks(
    x: np.ndarray,
    y: np.ndarray,
    model: str = 'GOT5.5',
    directory: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    Determine valid/invalid regions for tide model

    Returns a boolean mask indicating which points are within the
    valid domain of the tide model.

    Parameters
    ----------
    x : np.ndarray
        Longitudes in degrees
    y : np.ndarray
        Latitudes in degrees
    model : str
        Model name (e.g., 'GOT5.5', 'TPXO9-atlas-v5', 'FES2014')
    directory : str, optional
        Model data directory
    **kwargs
        Additional arguments for model loading

    Returns
    -------
    mask : np.ndarray
        Boolean mask (True = valid, False = invalid/outside domain)

    Notes
    -----
    This function checks the data availability of the tide model
    at the specified coordinates. Points where the model has valid
    (non-NaN) data are marked as True.

    Examples
    --------
    >>> import pyTMD_turbo as pytmd
    >>> import numpy as np
    >>> x = np.linspace(-180, 180, 361)
    >>> y = np.linspace(-90, 90, 181)
    >>> mask = pytmd.tide_masks(x, y, model='GOT5.5')
    """
    if kwargs:
        warnings.warn(
            f"pyTMD_turbo.tide_masks: ignoring unsupported kwargs: {list(kwargs.keys())}. "
            "These parameters are accepted for compatibility but have no effect.",
            UserWarning,
            stacklevel=2
        )
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))

    # Ensure model is loaded
    _ensure_model_loaded(model, directory)

    # Get interpolated constituent data for first constituent
    # The mask is determined by checking for valid (non-NaN) data
    try:
        # Use a reference time (doesn't matter for mask)
        mjd = np.array([60000.0])

        # Get tide values - NaN indicates invalid region
        tide = _cache.predict_batch(model, y, x, mjd, apply_nodal=False)

        # Create mask: valid where not NaN
        mask = np.isfinite(tide[:, 0])

    except Exception as e:
        # If prediction fails, return all False
        warnings.warn(
            f"tide_masks prediction failed for model '{model}': {e}. "
            "Returning all-False mask. This may indicate a model loading or data issue.",
            RuntimeWarning,
            stacklevel=2
        )
        mask = np.zeros(len(x), dtype=bool)

    return mask
