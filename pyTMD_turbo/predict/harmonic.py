"""
Fast harmonic synthesis implementation

Uses NumPy vectorisation for high-speed tidal prediction
across multiple locations and times.

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HarmonicConstants:
    """
    Harmonic constants for multiple locations (for vectorisation)

    Attributes
    ----------
    constituents : List[str]
        List of constituent names
    hc_real : np.ndarray
        Real part, shape (n_points, n_constituents)
    hc_imag : np.ndarray
        Imaginary part, shape (n_points, n_constituents)
    omega : np.ndarray
        Angular frequency, shape (n_constituents,)
    phase_0 : np.ndarray
        Phase-0, shape (n_constituents,)
    """
    constituents: List[str]
    hc_real: np.ndarray
    hc_imag: np.ndarray
    omega: np.ndarray
    phase_0: np.ndarray


def predict_vectorized(hc: HarmonicConstants,
                       t_days: np.ndarray,
                       pu: Optional[np.ndarray] = None,
                       pf: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Vectorised tide prediction

    Parameters
    ----------
    hc : HarmonicConstants
        Harmonic constants
    t_days : np.ndarray
        Days since 1992-01-01, shape (n_times,)
    pu : np.ndarray, optional
        Phase correction, shape (n_times, n_constituents)
    pf : np.ndarray, optional
        Amplitude correction, shape (n_times, n_constituents)

    Returns
    -------
    np.ndarray
        Tide height, shape (n_points, n_times)
    """
    n_points = hc.hc_real.shape[0]
    n_times = len(t_days)
    n_const = len(hc.constituents)

    # Default nodal corrections
    if pu is None:
        pu = np.zeros((n_times, n_const))
    if pf is None:
        pf = np.ones((n_times, n_const))

    # Phase angle calculation
    # theta[t, c] = omega[c] * t_days[t] * 86400 + phase_0[c] + pu[t, c]
    theta = (np.outer(t_days, hc.omega) * 86400.0 +
             hc.phase_0[np.newaxis, :] +
             pu)  # (n_times, n_const)

    # cos(theta), sin(theta)
    cos_theta = np.cos(theta)  # (n_times, n_const)
    sin_theta = np.sin(theta)  # (n_times, n_const)

    # Tide calculation
    # tide[p, t] = sum_c f[t,c] * (hc_real[p,c] * cos_theta[t,c] - hc_imag[p,c] * sin_theta[t,c])

    # Use matrix operations for efficient calculation
    # (n_points, n_const) @ (n_const, n_times) -> (n_points, n_times)

    # Pre-compute f * cos_theta and f * sin_theta
    f_cos = pf * cos_theta  # (n_times, n_const)
    f_sin = pf * sin_theta  # (n_times, n_const)

    # Matrix product
    tide = (hc.hc_real @ f_cos.T - hc.hc_imag @ f_sin.T)  # (n_points, n_times)

    return tide


def predict_single_point(hc_real: np.ndarray,
                         hc_imag: np.ndarray,
                         omega: np.ndarray,
                         phase_0: np.ndarray,
                         t_days: np.ndarray,
                         pu: Optional[np.ndarray] = None,
                         pf: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Tide prediction for a single location (optimised version)

    Parameters
    ----------
    hc_real : np.ndarray
        Real part, shape (n_constituents,)
    hc_imag : np.ndarray
        Imaginary part, shape (n_constituents,)
    omega : np.ndarray
        Angular frequency, shape (n_constituents,)
    phase_0 : np.ndarray
        Phase-0, shape (n_constituents,)
    t_days : np.ndarray
        Days since 1992-01-01, shape (n_times,)
    pu : np.ndarray, optional
        Phase correction, shape (n_times, n_constituents)
    pf : np.ndarray, optional
        Amplitude correction, shape (n_times, n_constituents)

    Returns
    -------
    np.ndarray
        Tide height, shape (n_times,)
    """
    n_times = len(t_days)
    n_const = len(omega)

    # Default nodal corrections
    if pu is None:
        pu = np.zeros((n_times, n_const))
    if pf is None:
        pf = np.ones((n_times, n_const))

    # Phase angle calculation
    theta = (np.outer(t_days, omega) * 86400.0 +
             phase_0[np.newaxis, :] +
             pu)  # (n_times, n_const)

    # Tide calculation
    tide = np.sum(pf * (hc_real * np.cos(theta) - hc_imag * np.sin(theta)), axis=1)

    return tide


def get_nodal_corrections(mjd: np.ndarray,
                          constituents: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get nodal corrections

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day
    constituents : List[str]
        List of constituent names

    Returns
    -------
    pu : np.ndarray
        Phase correction (rad), shape (n_times, n_constituents)
    pf : np.ndarray
        Amplitude correction factor, shape (n_times, n_constituents)
    """
    from pyTMD_turbo import constituents as _constituents

    pu, pf, G = _constituents.arguments(mjd, constituents, corrections='OTIS')
    return pu, pf


class FastHarmonicPredictor:
    """
    Fast harmonic synthesis predictor

    Holds harmonic constants for multiple locations and performs
    high-speed predictions using vectorised calculations.
    """

    # PyTMD reference time (MJD of 1992-01-01)
    MJD_TIDE = 48622.0

    def __init__(self):
        self.hc: Optional[HarmonicConstants] = None
        self.lats: Optional[np.ndarray] = None
        self.lons: Optional[np.ndarray] = None

    def set_constants(self,
                      constituents: List[str],
                      hc_real: np.ndarray,
                      hc_imag: np.ndarray,
                      omega: np.ndarray,
                      phase_0: np.ndarray,
                      lats: np.ndarray,
                      lons: np.ndarray) -> None:
        """
        Set harmonic constants

        Parameters
        ----------
        constituents : List[str]
            List of constituent names
        hc_real : np.ndarray
            Real part, shape (n_points, n_constituents)
        hc_imag : np.ndarray
            Imaginary part, shape (n_points, n_constituents)
        omega : np.ndarray
            Angular frequency, shape (n_constituents,)
        phase_0 : np.ndarray
            Phase-0, shape (n_constituents,)
        lats : np.ndarray
            Latitudes, shape (n_points,)
        lons : np.ndarray
            Longitudes, shape (n_points,)
        """
        self.hc = HarmonicConstants(
            constituents=constituents,
            hc_real=hc_real,
            hc_imag=hc_imag,
            omega=omega,
            phase_0=phase_0
        )
        self.lats = lats
        self.lons = lons

    def predict(self, mjd: np.ndarray, apply_nodal: bool = True) -> np.ndarray:
        """
        Predict tide

        Parameters
        ----------
        mjd : np.ndarray
            Modified Julian Day
        apply_nodal : bool
            Whether to apply nodal corrections

        Returns
        -------
        np.ndarray
            Tide height, shape (n_points, n_times)
        """
        if self.hc is None:
            raise ValueError("Harmonic constants have not been set")

        t_days = mjd - self.MJD_TIDE

        if apply_nodal:
            pu, pf = get_nodal_corrections(mjd, self.hc.constituents)
        else:
            pu = None
            pf = None

        return predict_vectorized(self.hc, t_days, pu, pf)
