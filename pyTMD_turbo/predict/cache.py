"""
Cache class for OCEAN tide prediction

Optimises PyTMD's NetCDF I/O and caches harmonic constants
for faster repeated calculations.

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TidalConstant:
    """Harmonic constant for a single constituent"""
    name: str
    hc_real: float        # Real part of complex amplitude
    hc_imag: float        # Imaginary part of complex amplitude
    omega: float          # Angular frequency (rad/day * 86400 = rad/day_to_sec)
    phase_0: float        # Phase-0 (rad)


@dataclass
class CachedModel:
    """Cached tidal model"""
    name: str
    format: str
    constituents: List[str]

    # Grid data (global)
    lon: np.ndarray
    lat: np.ndarray
    hc_real: Dict[str, np.ndarray]  # Constituent name -> real part grid
    hc_imag: Dict[str, np.ndarray]  # Constituent name -> imaginary part grid

    # Constituent parameters
    omega: Dict[str, float]     # Angular frequency (rad/day * 86400)
    phase_0: Dict[str, float]   # Phase-0 (rad)

    # Interpolated data (specific locations)
    interpolated: Dict[Tuple[float, float], Dict[str, TidalConstant]] = field(default_factory=dict)


class OceanTideCache:
    """
    OCEAN tide prediction cache class

    Usage:
    ```python
    cache = OceanTideCache()
    cache.load_model('GOT5.5', directory='~/.cache/pytmd')

    # Get harmonic constants (cached)
    hc = cache.get_constants(lat=35.0, lon=135.0)

    # Predict tide
    tide = cache.predict(lat=35.0, lon=135.0, mjd=mjd_array)
    ```
    """

    # PyTMD reference time (MJD of 1992-01-01)
    MJD_TIDE = 48622.0

    def __init__(self):
        self.models: Dict[str, CachedModel] = {}
        self._pytmd_available = self._check_pytmd()

    def _check_pytmd(self) -> bool:
        """Check if PyTMD is available"""
        try:
            import pyTMD.io
            import pyTMD.constituents
            return True
        except ImportError:
            return False

    def load_model(self, model_name: str, directory: str = '~/.cache/pytmd') -> None:
        """
        Load tidal model and cache

        Parameters
        ----------
        model_name : str
            Model name (e.g., 'GOT5.5', 'GOT5.6', 'FES2014')
        directory : str
            Model data directory
        """
        if not self._pytmd_available:
            raise ImportError("PyTMD is required")

        import pyTMD.io
        import pyTMD.constituents

        # Create model object
        m = pyTMD.io.model(directory=directory).from_database(model_name)

        # Load dataset
        ds = m.open_dataset()

        # Get grid coordinates
        lon = ds.coords['x'].values
        lat = ds.coords['y'].values

        # Extract real/imaginary parts and constituent parameters
        hc_real = {}
        hc_imag = {}
        omega = {}
        phase_0 = {}
        constituents = []

        for var_name in ds.data_vars:
            # Normalise constituent name
            const_name = var_name.lower().replace("'", "")
            constituents.append(const_name)

            # Get real/imaginary parts from complex data
            data = ds[var_name].values
            if np.iscomplexobj(data):
                hc_real[const_name] = data.real.copy()
                hc_imag[const_name] = data.imag.copy()
            else:
                hc_real[const_name] = data.copy()
                hc_imag[const_name] = np.zeros_like(data)

            # Get constituent parameters
            try:
                amp, ph, om, alpha, species = pyTMD.constituents._constituent_parameters(const_name)
                omega[const_name] = om  # rad/day * 86400
                phase_0[const_name] = ph  # rad
            except Exception as e:
                warnings.warn(
                    f"Unknown constituent '{const_name}' in model '{model_name}': {e}. "
                    "Using omega=0.0 which will result in incorrect predictions for this constituent.",
                    RuntimeWarning,
                    stacklevel=2
                )
                omega[const_name] = 0.0
                phase_0[const_name] = 0.0

        # Save to cache
        self.models[model_name] = CachedModel(
            name=model_name,
            format=m.format,
            constituents=constituents,
            lon=lon,
            lat=lat,
            hc_real=hc_real,
            hc_imag=hc_imag,
            omega=omega,
            phase_0=phase_0
        )

    def get_constants(self, model_name: str, lat: float, lon: float,
                      method: str = 'bilinear') -> Dict[str, TidalConstant]:
        """
        Get harmonic constants for a specified location (returns cached if available)

        Parameters
        ----------
        model_name : str
            Model name
        lat : float
            Latitude (degrees)
        lon : float
            Longitude (degrees)
        method : str
            Interpolation method ('bilinear' or 'nearest')

        Returns
        -------
        Dict[str, TidalConstant]
            Mapping of constituent name to harmonic constant
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' has not been loaded")

        model = self.models[model_name]
        key = (lat, lon)

        # Return if cached
        if key in model.interpolated:
            return model.interpolated[key]

        # Interpolate
        constants = {}
        for const_name in model.constituents:
            real = self._interpolate(
                model.lon, model.lat, model.hc_real[const_name],
                lon, lat, method
            )
            imag = self._interpolate(
                model.lon, model.lat, model.hc_imag[const_name],
                lon, lat, method
            )

            constants[const_name] = TidalConstant(
                name=const_name,
                hc_real=real,
                hc_imag=imag,
                omega=model.omega[const_name],
                phase_0=model.phase_0[const_name]
            )

        # Save to cache
        model.interpolated[key] = constants

        return constants

    def _interpolate(self, grid_lon: np.ndarray, grid_lat: np.ndarray,
                     data: np.ndarray, lon: float, lat: float,
                     method: str = 'bilinear') -> float:
        """
        Interpolate grid data to a specified location
        """
        # Normalise longitude to 0-360
        lon = lon % 360.0

        # Calculate grid indices
        dlon = grid_lon[1] - grid_lon[0]
        dlat = grid_lat[1] - grid_lat[0]

        i = (lon - grid_lon[0]) / dlon
        j = (lat - grid_lat[0]) / dlat

        if method == 'nearest':
            i = int(round(i))
            j = int(round(j))
            i = max(0, min(i, len(grid_lon) - 1))
            j = max(0, min(j, len(grid_lat) - 1))
            return data[j, i]

        elif method == 'bilinear':
            i0 = int(np.floor(i))
            j0 = int(np.floor(j))
            i1 = i0 + 1
            j1 = j0 + 1

            # Boundary check
            i0 = max(0, min(i0, len(grid_lon) - 1))
            i1 = max(0, min(i1, len(grid_lon) - 1))
            j0 = max(0, min(j0, len(grid_lat) - 1))
            j1 = max(0, min(j1, len(grid_lat) - 1))

            # Weights
            wi = i - int(np.floor(i))
            wj = j - int(np.floor(j))

            # Bilinear interpolation
            v00 = data[j0, i0]
            v01 = data[j0, i1]
            v10 = data[j1, i0]
            v11 = data[j1, i1]

            # NaN handling
            if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
                return np.nan

            return (v00 * (1-wi) * (1-wj) +
                    v01 * wi * (1-wj) +
                    v10 * (1-wi) * wj +
                    v11 * wi * wj)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _get_nodal_corrections(self, mjd: np.ndarray, constituents: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get nodal corrections

        Returns
        -------
        pu : np.ndarray
            Phase correction (rad), shape (n_times, n_constituents)
        pf : np.ndarray
            Amplitude correction factor, shape (n_times, n_constituents)
        """
        import pyTMD.constituents

        pu, pf, G = pyTMD.constituents.arguments(mjd, constituents, corrections='OTIS')
        return pu, pf

    def predict(self, model_name: str, lat: float, lon: float,
                mjd: np.ndarray, apply_nodal: bool = True) -> np.ndarray:
        """
        Predict tide

        Parameters
        ----------
        model_name : str
            Model name
        lat : float
            Latitude (degrees)
        lon : float
            Longitude (degrees)
        mjd : np.ndarray
            Modified Julian Day array
        apply_nodal : bool
            Whether to apply nodal corrections

        Returns
        -------
        np.ndarray
            Tide height (metres)
        """
        constants = self.get_constants(model_name, lat, lon)
        model = self.models[model_name]

        # Days since 1992-01-01
        t_days = mjd - self.MJD_TIDE

        # Nodal corrections
        if apply_nodal:
            pu, pf = self._get_nodal_corrections(mjd, model.constituents)
        else:
            pu = np.zeros((len(mjd), len(model.constituents)))
            pf = np.ones((len(mjd), len(model.constituents)))

        # Harmonic synthesis
        # PyTMD formula:
        # theta = omega * t * 86400.0 + phase_0 + pu
        # tpred = hc_real * f * cos(theta) - hc_imag * f * sin(theta)

        tide = np.zeros_like(t_days)

        for i, const_name in enumerate(model.constituents):
            const = constants[const_name]

            if const.omega == 0 or np.isnan(const.hc_real):
                continue

            # Phase angle
            theta = const.omega * t_days * 86400.0 + const.phase_0 + pu[:, i]

            # Amplitude correction
            f = pf[:, i]

            # Tide contribution
            tide += f * (const.hc_real * np.cos(theta) - const.hc_imag * np.sin(theta))

        return tide

    def predict_fast(self, model_name: str, lat: float, lon: float,
                     mjd: np.ndarray) -> np.ndarray:
        """
        Fast tide prediction (without nodal corrections)

        Faster by omitting nodal corrections.
        Accuracy is slightly reduced, but trends are captured.
        """
        return self.predict(model_name, lat, lon, mjd, apply_nodal=False)

    def predict_batch(self, model_name: str, lats: np.ndarray, lons: np.ndarray,
                      mjd: np.ndarray, apply_nodal: bool = True) -> np.ndarray:
        """
        Batch prediction for multiple locations

        Parameters
        ----------
        model_name : str
            Model name
        lats : np.ndarray
            Latitude array (degrees)
        lons : np.ndarray
            Longitude array (degrees)
        mjd : np.ndarray
            Modified Julian Day array
        apply_nodal : bool
            Whether to apply nodal corrections

        Returns
        -------
        np.ndarray
            Tide height (metres), shape (len(lats), len(mjd))
        """
        n_points = len(lats)
        n_times = len(mjd)

        result = np.zeros((n_points, n_times))

        for i, (lat, lon) in enumerate(zip(lats, lons)):
            result[i, :] = self.predict(model_name, lat, lon, mjd, apply_nodal)

        return result


def create_cache(model_name: str = 'GOT5.5',
                 directory: str = '~/.cache/pytmd') -> OceanTideCache:
    """
    Convenience function to create and return a cache
    """
    cache = OceanTideCache()
    cache.load_model(model_name, directory)
    return cache
