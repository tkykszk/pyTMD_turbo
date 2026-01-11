"""
pyTMD_turbo.io.dataset - xarray Dataset accessor for tidal models

Provides the tmd accessor for xarray Datasets, compatible with pyTMD.

Usage:
    import xarray as xr
    from pyTMD_turbo.io import register_accessor

    ds = model.open_dataset()

    # Interpolate to coordinates
    hc = ds.tmd.interp(x=lons, y=lats)

    # Predict tide heights
    tide = ds.tmd.predict(times)

Copyright (c) 2024-2026 tkykszk
Derived from pyTMD by Tyler Sutterley (MIT License)
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional, Union, List, TYPE_CHECKING

from scipy import ndimage

if TYPE_CHECKING:
    import xarray as xr
    import pyproj as pyproj_types
    CRSType = pyproj_types.CRS
else:
    CRSType = object

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    xr = None

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    pyproj = None

__all__ = ['TMDAccessor', 'register_accessor']


class TMDAccessor:
    """
    xarray Dataset accessor for tidal model data

    Provides methods compatible with pyTMD's dataset accessor:
    - interp: interpolate model to coordinates
    - predict: predict tide heights at given times
    - transform_as: coordinate transformation
    - constituents: list of tidal constituents
    - crs: coordinate reference system

    Parameters
    ----------
    xarray_obj : xr.Dataset
        xarray Dataset containing tidal model data

    Examples
    --------
    >>> ds = model.open_dataset()
    >>> # Interpolate to coordinates
    >>> hc = ds.tmd.interp(x=lons, y=lats)
    >>> # Predict tide heights
    >>> tide = ds.tmd.predict(times)
    """

    # Reference time (MJD of 1992-01-01)
    _MJD_TIDE = 48622.0

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._crs = None
        self._constituents = None

    @property
    def constituents(self) -> List[str]:
        """
        List of tidal constituents in the dataset

        Returns
        -------
        list
            List of constituent names (lowercase)
        """
        if self._constituents is not None:
            return self._constituents

        ds = self._obj

        # Check for constituent dimension
        if 'constituent' in ds.dims:
            self._constituents = list(ds.coords['constituent'].values)
        else:
            # Constituents are separate variables
            # Filter out coordinate variables
            coord_vars = {'x', 'y', 'lon', 'lat', 'longitude', 'latitude',
                          'hz', 'mask', 'depth', 'bathymetry'}
            self._constituents = [
                str(v).lower() for v in ds.data_vars
                if str(v).lower() not in coord_vars
            ]

        return self._constituents

    @property
    def crs(self) -> Optional[CRSType]:
        """
        Coordinate reference system of the dataset

        Returns
        -------
        pyproj.CRS or None
            CRS object if pyproj is available
        """
        if not HAS_PYPROJ:
            return None

        if self._crs is not None:
            return self._crs

        # Try to determine CRS from attributes
        ds = self._obj

        # Check for crs attribute
        if 'crs' in ds.attrs:
            crs_str = ds.attrs['crs']
            try:
                self._crs = pyproj.CRS(crs_str)
            except Exception:
                pass
        elif 'projection' in ds.attrs:
            try:
                self._crs = pyproj.CRS(ds.attrs['projection'])
            except Exception:
                pass

        # Default to WGS84 geographic
        if self._crs is None:
            self._crs = pyproj.CRS.from_epsg(4326)

        return self._crs

    @property
    def is_global(self) -> bool:
        """
        Check if model covers the globe

        Returns
        -------
        bool
            True if model appears to be global
        """
        ds = self._obj
        x = ds.coords['x'].values

        # Check if longitude spans ~360 degrees
        lon_range = float(x.max() - x.min())
        return lon_range > 350.0

    def transform_as(
        self,
        x: np.ndarray,
        y: np.ndarray,
        crs: Optional[Union[str, CRSType]] = None,
    ) -> tuple:
        """
        Transform coordinates to the dataset's CRS

        Parameters
        ----------
        x : np.ndarray
            x coordinates (longitude or easting)
        y : np.ndarray
            y coordinates (latitude or northing)
        crs : str or pyproj.CRS, optional
            Source coordinate reference system
            Default is WGS84 geographic (EPSG:4326)

        Returns
        -------
        tuple
            Transformed (x, y) coordinates in dataset CRS
        """
        if not HAS_PYPROJ:
            # No transformation, return as-is
            return np.asarray(x), np.asarray(y)

        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))

        # Source CRS
        if crs is None:
            source_crs = pyproj.CRS.from_epsg(4326)
        elif isinstance(crs, str):
            source_crs = pyproj.CRS(crs)
        else:
            source_crs = crs

        # Target CRS (dataset CRS)
        target_crs = self.crs

        # Transform if CRS differ
        if source_crs != target_crs:
            transformer = pyproj.Transformer.from_crs(
                source_crs, target_crs, always_xy=True
            )
            x, y = transformer.transform(x, y)

        return x, y

    def coords_as(
        self,
        crs: Optional[Union[str, CRSType]] = None,
    ) -> tuple:
        """
        Get dataset coordinates in specified CRS

        Parameters
        ----------
        crs : str or pyproj.CRS, optional
            Target coordinate reference system
            Default is WGS84 geographic (EPSG:4326)

        Returns
        -------
        tuple
            (x, y) coordinate arrays in target CRS
        """
        ds = self._obj
        x = ds.coords['x'].values
        y = ds.coords['y'].values

        if not HAS_PYPROJ or crs is None:
            return x, y

        # Target CRS
        if isinstance(crs, str):
            target_crs = pyproj.CRS(crs)
        else:
            target_crs = crs

        # Source CRS (dataset CRS)
        source_crs = self.crs

        # Transform if CRS differ
        if source_crs != target_crs:
            # Create meshgrid for transformation
            xx, yy = np.meshgrid(x, y)
            transformer = pyproj.Transformer.from_crs(
                source_crs, target_crs, always_xy=True
            )
            xx, yy = transformer.transform(xx, yy)
            return xx, yy

        return x, y

    def interp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'bilinear',
        extrapolate: bool = False,
    ) -> 'xr.Dataset':
        """
        Interpolate model to specified coordinates

        Parameters
        ----------
        x : np.ndarray
            x coordinates (longitude)
        y : np.ndarray
            y coordinates (latitude)
        method : str, default 'bilinear'
            Interpolation method ('bilinear', 'nearest', 'spline')
        extrapolate : bool, default False
            Whether to extrapolate outside model domain

        Returns
        -------
        xr.Dataset
            Interpolated dataset with harmonic constants
        """
        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))

        ds = self._obj

        # Get grid coordinates
        lon = ds.coords['x'].values
        lat = ds.coords['y'].values
        dlon = float(lon[1] - lon[0])
        dlat = float(lat[1] - lat[0])

        # Normalize longitude to 0-360
        x_norm = x % 360.0

        # Calculate continuous indices
        lon_idx = (x_norm - lon[0]) / dlon
        lat_idx = (y - lat[0]) / dlat

        # Coordinate array for scipy
        coords = np.array([lat_idx, lon_idx])

        # Interpolation order
        if method == 'nearest':
            order = 0
        elif method == 'bilinear':
            order = 1
        elif method == 'spline':
            order = 3
        else:
            order = 1

        # Interpolate each constituent
        interp_data = {}
        constituents = self.constituents

        for const_name in constituents:
            # Get data for this constituent
            if 'constituent' in ds.dims:
                if 'hc' in ds.data_vars:
                    data = ds['hc'].sel(constituent=const_name).values
                else:
                    var_name = list(ds.data_vars)[0]
                    data = ds[var_name].sel(constituent=const_name).values
            else:
                data = ds[const_name].values

            # Handle complex data
            if np.iscomplexobj(data):
                real_interp = ndimage.map_coordinates(
                    data.real, coords, order=order, mode='nearest'
                )
                imag_interp = ndimage.map_coordinates(
                    data.imag, coords, order=order, mode='nearest'
                )
                interp_data[const_name] = real_interp + 1j * imag_interp
            else:
                interp_data[const_name] = ndimage.map_coordinates(
                    data, coords, order=order, mode='nearest'
                )

        # Create output dataset
        n_points = len(x)
        data_vars = {
            name: (['point'], values)
            for name, values in interp_data.items()
        }

        result = xr.Dataset(
            data_vars,
            coords={
                'x': (['point'], x),
                'y': (['point'], y),
            }
        )

        return result

    def predict(
        self,
        t: np.ndarray,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        corrections: str = 'GOT',
        method: str = 'bilinear',
    ) -> np.ndarray:
        """
        Predict tide heights at specified times

        Parameters
        ----------
        t : np.ndarray
            Times as numpy datetime64, datetime, or MJD
        x : np.ndarray, optional
            x coordinates for interpolation
        y : np.ndarray, optional
            y coordinates for interpolation
        corrections : str, default 'GOT'
            Nodal correction type ('GOT', 'OTIS', 'FES')
        method : str, default 'bilinear'
            Interpolation method if x, y provided

        Returns
        -------
        np.ndarray
            Predicted tide heights (metres)
            Shape: (n_points, n_times) if x, y provided
            Shape: (n_lat, n_lon, n_times) otherwise
        """
        from pyTMD_turbo import constituents as _constituents

        # Convert times to MJD
        mjd = self._to_mjd(t)
        n_times = len(mjd)

        # Get dataset (interpolated or full grid)
        if x is not None and y is not None:
            ds = self.interp(x, y, method=method)
            is_point = True
        else:
            ds = self._obj
            is_point = False

        constituents = self.constituents
        n_const = len(constituents)

        # Grid dimensions (will be set for grid mode)
        n_lat: int = 0
        n_lon: int = 0

        # Get harmonic constants
        if is_point:
            # Point data: shape (n_points,) per constituent
            n_points = len(ds.coords.get('point', ds.coords['x']))
            hc = np.zeros((n_points, n_const), dtype=np.complex128)
            for i, c in enumerate(constituents):
                hc[:, i] = ds[c].values
        else:
            # Grid data: shape (n_lat, n_lon) per constituent
            lon = ds.coords['x'].values
            lat = ds.coords['y'].values
            n_lat, n_lon = len(lat), len(lon)

            if 'constituent' in ds.dims and 'hc' in ds.data_vars:
                hc = ds['hc'].values  # (n_const, n_lat, n_lon)
                hc = hc.reshape(n_const, -1).T  # (n_lat*n_lon, n_const)
            else:
                hc = np.zeros((n_lat * n_lon, n_const), dtype=np.complex128)
                for i, c in enumerate(constituents):
                    hc[:, i] = ds[c].values.ravel()

        # Get nodal corrections
        pu, pf, G = _constituents.arguments(mjd, constituents, corrections=corrections)

        # Calculate phase angles
        if corrections in ('OTIS', 'ATLAS', 'TMD3'):
            # OTIS type: omega * t * 86400 + phase_0 + pu
            omega = _constituents.frequency(constituents, corrections=corrections)
            t_days = mjd - self._MJD_TIDE
            theta = (np.outer(t_days, omega) * 86400.0 + pu)
        else:
            # GOT/FES type: radians(G) + pu
            theta = np.radians(G) + pu

        # Harmonic synthesis
        hc_real = hc.real
        hc_imag = hc.imag

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        f_cos = pf * cos_theta  # (n_times, n_const)
        f_sin = pf * sin_theta

        # tide[p, t] = sum_c (hc_real[p,c] * f_cos[t,c] - hc_imag[p,c] * f_sin[t,c])
        tide = hc_real @ f_cos.T - hc_imag @ f_sin.T

        if not is_point:
            # Reshape back to grid
            tide = tide.reshape(n_lat, n_lon, n_times)

        return tide

    def infer(
        self,
        t: np.ndarray,
        method: str = 'linear',
        corrections: str = 'GOT',
        deltat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Infer minor tidal constituents

        Uses major constituents to infer contributions from minor constituents
        based on equilibrium tidal theory.

        Parameters
        ----------
        t : np.ndarray
            Times as numpy datetime64, datetime, or MJD
        method : str, default 'linear'
            Inference method:
            - 'linear': Linear interpolation in frequency domain
            - 'admittance': Equilibrium theory admittance functions
        corrections : str, default 'GOT'
            Nodal correction type ('GOT', 'OTIS', 'FES')
        deltat : np.ndarray, optional
            Time correction for converting to TT

        Returns
        -------
        np.ndarray
            Minor constituent tide prediction (metres)

        Examples
        --------
        >>> ds = model.open_dataset()
        >>> local = ds.tmd.interp(x=140.0, y=35.0)
        >>> times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')
        >>> minor = local.tmd.infer(times, method='admittance')
        """
        from pyTMD_turbo.predict.infer_minor import infer_minor

        mjd = self._to_mjd(t)
        return infer_minor(mjd, self._obj, method=method, corrections=corrections,
                          deltat=deltat)

    def _to_mjd(self, t: np.ndarray) -> np.ndarray:
        """
        Convert times to Modified Julian Day

        Parameters
        ----------
        t : np.ndarray
            Times as numpy datetime64, datetime objects, or MJD floats

        Returns
        -------
        np.ndarray
            Modified Julian Day values
        """
        from datetime import datetime, timezone

        t = np.atleast_1d(t)

        if isinstance(t[0], np.datetime64):
            # numpy datetime64 -> MJD
            times_unix = t.astype('datetime64[s]').astype('float64')
            mjd = times_unix / 86400.0 + 40587.0
        elif isinstance(t[0], datetime):
            # datetime -> MJD
            mjd = np.zeros(len(t))
            for i, dt in enumerate(t):
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                mjd[i] = dt.timestamp() / 86400.0 + 40587.0
        else:
            # Assume already MJD
            mjd = np.asarray(t, dtype=np.float64)

        return mjd


def register_accessor():
    """
    Register the tmd accessor with xarray

    After calling this function, any xarray Dataset will have
    a `tmd` accessor available.

    Examples
    --------
    >>> from pyTMD_turbo.io import register_accessor
    >>> register_accessor()
    >>> ds = xr.open_dataset('model.nc')
    >>> ds.tmd.constituents
    ['m2', 's2', 'n2', ...]
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required for the tmd accessor")

    # Check if already registered
    if hasattr(xr.Dataset, 'tmd'):
        return

    @xr.register_dataset_accessor('tmd')
    class _TMDAccessor(TMDAccessor):
        pass


# Auto-register if xarray is available
if HAS_XARRAY:
    try:
        register_accessor()
    except Exception:
        pass
