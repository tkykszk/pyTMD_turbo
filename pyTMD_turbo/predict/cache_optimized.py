"""
Optimised cache class for OCEAN tide prediction

Uses scipy.ndimage.map_coordinates for fast interpolation and
NumPy vectorisation for high-speed batch processing.

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Import from pyTMD_turbo modules (standalone)
from pyTMD_turbo import constituents as _constituents
from pyTMD_turbo.io.model import model as _Model


@dataclass
class CachedModelOptimized:
    """Optimised cached tidal model"""
    name: str
    format: str
    constituents: list[str]
    corrections: str       # Correction type ('GOT' or 'OTIS')

    # Grid data (global)
    lon: np.ndarray
    lat: np.ndarray
    dlon: float
    dlat: float

    # Complex grid (per constituent)
    hc_real: dict[str, np.ndarray]
    hc_imag: dict[str, np.ndarray]

    # Constituent parameters (array format)
    omega: np.ndarray      # Angular frequency, shape (n_constituents,)
    phase_0: np.ndarray    # Phase-0, shape (n_constituents,)


class OceanTideCacheOptimized:
    """
    Optimised OCEAN tide prediction cache class

    Features:
    - Fast batch interpolation via scipy.ndimage.map_coordinates
    - Fast harmonic synthesis via NumPy vectorisation
    - Reduced overhead through pre-computed parameters
    - Standalone operation without pyTMD dependency

    Usage:
    ```python
    cache = OceanTideCacheOptimized()
    cache.load_model('GOT5.5', directory='/path/to/models')

    # Batch prediction (recommended)
    tide = cache.predict_batch('GOT5.5', lats, lons, mjd)
    ```
    """

    # PyTMD reference time (MJD of 1992-01-01)
    MJD_TIDE = 48622.0

    def __init__(self):
        self.models: dict[str, CachedModelOptimized] = {}
        self._model_dirs: dict[str, str] = {}

    def load_model(self, model_name: str, directory: str = '~/.cache/pytmd',
                   group: str = 'z') -> None:
        """
        Load tidal model and cache

        Parameters
        ----------
        model_name : str
            Name of the tide model (e.g., 'GOT5.5', 'TPXO9-atlas-v5')
        directory : str
            Directory containing the model files
        group : str, default 'z'
            Component group to load ('z' for elevation, 'u' for zonal current,
            'v' for meridional current)
        """
        # Create model object using pyTMD_turbo.io
        m = _Model(directory=directory).from_database(model_name, group=group)
        ds = m.open_dataset()

        # Get grid coordinates
        lon = ds.coords['x'].values
        lat = ds.coords['y'].values
        dlon = float(lon[1] - lon[0])
        dlat = float(lat[1] - lat[0])

        # Extract data and constituent parameters
        hc_real = {}
        hc_imag = {}
        omega_list = []
        phase_0_list = []
        constituents_list = []

        # Check if data has 'constituent' dimension or separate variables
        if 'constituent' in ds.dims:
            # Data organized by constituent dimension
            constituents_list = list(ds.coords['constituent'].values)
            for const_name in constituents_list:
                const_name_lower = str(const_name).lower()

                # Select constituent data
                if 'hc' in ds.data_vars:
                    data = ds['hc'].sel(constituent=const_name).values
                else:
                    # Try to find elevation variable
                    for var in ds.data_vars:
                        if 'elevation' in var.lower() or 'tide' in var.lower():
                            data = ds[var].sel(constituent=const_name).values
                            break
                    else:
                        # Use first data variable
                        var_name = next(iter(ds.data_vars))
                        data = ds[var_name].sel(constituent=const_name).values

                if np.iscomplexobj(data):
                    hc_real[const_name_lower] = data.real.astype(np.float64)
                    hc_imag[const_name_lower] = data.imag.astype(np.float64)
                else:
                    hc_real[const_name_lower] = data.astype(np.float64)
                    hc_imag[const_name_lower] = np.zeros_like(data, dtype=np.float64)

                # Get constituent frequency
                try:
                    omega = _constituents.frequency([const_name_lower])[0]
                    omega_list.append(omega)
                    phase_0_list.append(0.0)  # Phase offset is handled in nodal corrections
                except (ValueError, KeyError):
                    warnings.warn(
                        f"Unknown constituent '{const_name_lower}' in model '{model_name}'. "
                        "Using omega=0.0 which will result in incorrect predictions for this constituent.",
                        RuntimeWarning,
                        stacklevel=2
                    )
                    omega_list.append(0.0)
                    phase_0_list.append(0.0)

                constituents_list.append(const_name_lower)
        else:
            # Data organized as separate variables per constituent
            for var_name in ds.data_vars:
                const_name = var_name.lower()
                constituents_list.append(const_name)

                data = ds[var_name].values
                if np.iscomplexobj(data):
                    hc_real[const_name] = data.real.astype(np.float64)
                    hc_imag[const_name] = data.imag.astype(np.float64)
                else:
                    hc_real[const_name] = data.astype(np.float64)
                    hc_imag[const_name] = np.zeros_like(data, dtype=np.float64)

                # Get constituent frequency
                try:
                    omega = _constituents.frequency([const_name])[0]
                    omega_list.append(omega)
                    phase_0_list.append(0.0)
                except (ValueError, KeyError):
                    warnings.warn(
                        f"Unknown constituent '{const_name}' in model '{model_name}'. "
                        "Using omega=0.0 which will result in incorrect predictions for this constituent.",
                        RuntimeWarning,
                        stacklevel=2
                    )
                    omega_list.append(0.0)
                    phase_0_list.append(0.0)

        # Save directory
        self._model_dirs[model_name] = directory

        # Save to cache (with model-specific corrections type)
        self.models[model_name] = CachedModelOptimized(
            name=model_name,
            format=m.format,
            constituents=constituents_list,
            corrections=m.corrections,
            lon=lon,
            lat=lat,
            dlon=dlon,
            dlat=dlat,
            hc_real=hc_real,
            hc_imag=hc_imag,
            omega=np.array(omega_list),
            phase_0=np.array(phase_0_list)
        )

    def interpolate_batch(self, model_name: str,
                          lats: np.ndarray,
                          lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch interpolation of harmonic constants for multiple locations

        Parameters
        ----------
        model_name : str
            Model name
        lats : np.ndarray
            Latitude array (degrees)
        lons : np.ndarray
            Longitude array (degrees)

        Returns
        -------
        hc_real : np.ndarray
            Real part, shape (n_points, n_constituents)
        hc_imag : np.ndarray
            Imaginary part, shape (n_points, n_constituents)
        """
        model = self.models[model_name]

        # Normalise longitude to 0-360
        lons = lons % 360.0

        # Calculate continuous indices
        lon_idx = (lons - model.lon[0]) / model.dlon
        lat_idx = (lats - model.lat[0]) / model.dlat

        # Coordinate array (for scipy)
        coords = np.array([lat_idx, lon_idx])

        # Batch interpolation for all constituents
        hc_real_list = []
        hc_imag_list = []

        for const_name in model.constituents:
            real = ndimage.map_coordinates(
                model.hc_real[const_name], coords, order=1, mode='nearest'
            )
            imag = ndimage.map_coordinates(
                model.hc_imag[const_name], coords, order=1, mode='nearest'
            )
            hc_real_list.append(real)
            hc_imag_list.append(imag)

        return np.array(hc_real_list).T, np.array(hc_imag_list).T

    def get_nodal_corrections(self, mjd: np.ndarray,
                              constituents: list[str],
                              corrections: str = 'OTIS') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get nodal corrections

        Parameters
        ----------
        mjd : np.ndarray
            Modified Julian Day array
        constituents : List[str]
            List of tidal constituents
        corrections : str
            Correction type ('GOT' or 'OTIS')

        Returns
        -------
        pu : np.ndarray
            Phase correction
        pf : np.ndarray
            Amplitude correction
        G : np.ndarray
            Greenwich phase angle (degrees)
        """
        # Use pyTMD_turbo.constituents
        pu, pf, G = _constituents.arguments(mjd, constituents, corrections=corrections)
        return pu, pf, G

    def predict_batch(self, model_name: str,
                      lats: np.ndarray,
                      lons: np.ndarray,
                      mjd: np.ndarray,
                      apply_nodal: bool = True,
                      infer_minor: bool = False) -> np.ndarray:
        """
        Batch prediction for multiple locations (optimised version)

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
        infer_minor : bool
            Whether to infer and add minor tidal constituents
            (Not yet implemented in standalone mode)

        Returns
        -------
        np.ndarray
            Tide height (metres), shape (n_points, n_times)
        """
        model = self.models[model_name]
        n_times = len(mjd)
        n_const = len(model.constituents)

        # 1. Batch interpolation
        hc_real, hc_imag = self.interpolate_batch(model_name, lats, lons)

        # 2. Time conversion
        t_days = mjd - self.MJD_TIDE

        # 3. Nodal corrections (using model-specific correction type)
        if apply_nodal:
            pu, pf, G = self.get_nodal_corrections(mjd, model.constituents, model.corrections)
        else:
            pu = np.zeros((n_times, n_const))
            pf = np.ones((n_times, n_const))
            G = np.zeros((n_times, n_const))

        # 4. Phase angle calculation (different methods depending on corrections type)
        if model.corrections in ('OTIS', 'ATLAS', 'TMD3'):
            # OTIS type: omega * t * 86400 + phase_0 + pu
            theta = (np.outer(t_days, model.omega) * 86400.0 +
                     model.phase_0[np.newaxis, :] +
                     pu)
        else:
            # GOT type: radians(G) + pu (uses Greenwich phase angle)
            theta = np.radians(G) + pu

        # 5. cos/sin calculation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 6. Amplitude correction
        f_cos = pf * cos_theta  # (n_times, n_const)
        f_sin = pf * sin_theta  # (n_times, n_const)

        # 7. Harmonic synthesis via matrix multiplication
        # tide[p, t] = sum_c (hc_real[p,c] * f_cos[t,c] - hc_imag[p,c] * f_sin[t,c])
        tide = hc_real @ f_cos.T - hc_imag @ f_sin.T

        # 8. Minor constituent inference (not yet implemented in standalone mode)
        if infer_minor:
            # TODO: Implement standalone minor constituent inference
            pass

        return tide

    def predict_single(self, model_name: str,
                       lat: float,
                       lon: float,
                       mjd: np.ndarray,
                       apply_nodal: bool = True,
                       infer_minor: bool = False) -> np.ndarray:
        """
        Predict tide for a single location
        """
        return self.predict_batch(
            model_name,
            np.array([lat]),
            np.array([lon]),
            mjd,
            apply_nodal,
            infer_minor
        )[0]


def create_optimized_cache(model_name: str = 'GOT5.5',
                           directory: str = '~/.cache/pytmd') -> OceanTideCacheOptimized:
    """
    Convenience function to create and return an optimised cache
    """
    cache = OceanTideCacheOptimized()
    cache.load_model(model_name, directory)
    return cache
