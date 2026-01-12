"""
pyTMD_turbo.predict.infer_minor - Minor constituent inference

Infers tidal heights from minor constituents using major constituent
relationships based on equilibrium tidal theory.

References:
    R. Ray, "A Global Ocean Tide Model From TOPEX/POSEIDON Altimetry: GOT99.2",
        NASA Tech. Memo., 1999.
    M. G. G. Foreman, "Manual of Harmonic Analysis and Prediction of Tides", 1977.
    A. T. Doodson, "Harmonic development of tide-generating potential", 1921.

Copyright (c) 2024-2026 tkykszk
Derived from pyTMD by Tyler Sutterley (MIT License)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

from pyTMD_turbo import constituents as _constituents

__all__ = [
    'DIURNAL_MINORS',
    'MINOR_CONSTITUENTS',
    'SEMI_DIURNAL_MINORS',
    'infer_diurnal',
    'infer_minor',
    'infer_semi_diurnal',
]

# Minor constituent groups
DIURNAL_MINORS = ['2q1', 'sigma1', 'rho1', 'm1', 'chi1', 'pi1', 'phi1',
                  'theta1', 'j1', 'oo1', 'ups1']
SEMI_DIURNAL_MINORS = ['eps2', 'lambda2', 'l2', 't2', 'r2', 'eta2']
MINOR_CONSTITUENTS = DIURNAL_MINORS + SEMI_DIURNAL_MINORS

# MJD of 1992-01-01 (pyTMD reference time)
_MJD_TIDE = 48622.0


def _get_constituent_amplitude(
    ds: xr.Dataset,
    constituent: str,
) -> np.ndarray:
    """
    Get harmonic constant for a constituent from dataset

    Parameters
    ----------
    ds : xr.Dataset
        Interpolated model dataset
    constituent : str
        Constituent name

    Returns
    -------
    np.ndarray
        Complex harmonic constant
    """
    const_lower = constituent.lower()

    # Check if constituent exists
    if 'constituent' in ds.dims:
        constituents_in_ds = [str(c).lower() for c in ds.coords['constituent'].values]
        if const_lower in constituents_in_ds and 'hc' in ds.data_vars:
            return ds['hc'].sel(constituent=constituent).values
    elif const_lower in ds.data_vars:
        return ds[const_lower].values

    # Constituent not found - return zeros
    return np.zeros(1, dtype=np.complex128)


def infer_diurnal(
    mjd: np.ndarray,
    ds: xr.Dataset,
    method: str = 'linear',
    corrections: str = 'GOT',
    deltat: np.ndarray | None = None,
) -> np.ndarray:
    """
    Infer minor diurnal tidal constituents

    Uses O1 and K1 to infer: 2q1, sigma1, rho1, m1, chi1, pi1, phi1, theta1, j1, oo1

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day
    ds : xr.Dataset
        Interpolated model dataset with harmonic constants
    method : str, default 'linear'
        Inference method ('linear' or 'admittance')
    corrections : str, default 'GOT'
        Nodal correction type ('GOT', 'OTIS', 'FES')
    deltat : np.ndarray, optional
        Time correction for converting to TT

    Returns
    -------
    np.ndarray
        Minor constituent tide prediction (metres)
    """
    mjd = np.atleast_1d(np.asarray(mjd, dtype=np.float64))
    n_times = len(mjd)

    if deltat is None:
        deltat = np.zeros_like(mjd)

    # Get major constituent harmonic constants
    o1 = _get_constituent_amplitude(ds, 'o1')
    q1 = _get_constituent_amplitude(ds, 'q1')
    k1 = _get_constituent_amplitude(ds, 'k1')
    p1 = _get_constituent_amplitude(ds, 'p1')

    n_points = len(o1) if np.ndim(o1) > 0 else 1
    o1 = np.atleast_1d(o1)
    q1 = np.atleast_1d(q1)
    k1 = np.atleast_1d(k1)
    p1 = np.atleast_1d(p1)

    # Minor constituents to infer
    minors = ['2q1', 'sigma1', 'rho1', 'm1', 'chi1', 'pi1', 'phi1',
              'theta1', 'j1', 'oo1']

    # Get nodal corrections for minor constituents
    pu, pf, G = _constituents.arguments(mjd + deltat, minors, corrections=corrections)

    # Calculate phase angles
    if corrections in ('OTIS', 'ATLAS', 'TMD3'):
        omega = _constituents.frequency(minors, corrections=corrections)
        t_days = mjd - _MJD_TIDE
        theta = np.outer(t_days, omega) * 86400.0 + pu
    else:
        theta = np.radians(G) + pu

    # Inference ratios based on equilibrium theory
    # These are empirical relationships between minor and major constituents
    if method == 'admittance':
        # Admittance-based inference (more accurate)
        # Ratios from equilibrium tidal theory
        ratios = {
            '2q1': (0.263, 'q1'),   # 2Q1 from Q1
            'sigma1': (0.297, 'q1'),  # sigma1 from Q1
            'rho1': (0.164, 'q1'),    # rho1 from Q1
            'm1': (0.092, 'o1'),      # M1 from O1
            'chi1': (0.048, 'k1'),    # chi1 from K1
            'pi1': (0.098, 'k1'),     # pi1 from K1
            'phi1': (0.090, 'k1'),    # phi1 from K1
            'theta1': (0.061, 'k1'),  # theta1 from K1
            'j1': (0.187, 'k1'),      # J1 from K1
            'oo1': (0.117, 'k1'),     # OO1 from K1
        }
    else:
        # Linear interpolation (default)
        # Using frequency ratios for interpolation
        ratios = {
            '2q1': (0.235, 'q1'),
            'sigma1': (0.263, 'q1'),
            'rho1': (0.195, 'q1'),
            'm1': (0.071, 'o1'),
            'chi1': (0.054, 'k1'),
            'pi1': (0.107, 'k1'),
            'phi1': (0.073, 'k1'),
            'theta1': (0.054, 'k1'),
            'j1': (0.168, 'k1'),
            'oo1': (0.108, 'k1'),
        }

    # Map source constituents to data
    sources = {'o1': o1, 'q1': q1, 'k1': k1, 'p1': p1}

    # Initialize output
    tide = np.zeros((n_points, n_times))

    # Calculate contribution from each minor constituent
    for i, minor in enumerate(minors):
        ratio, source = ratios[minor]
        hc = sources[source] * ratio

        # Harmonic synthesis
        for t in range(n_times):
            tide[:, t] += pf[t, i] * (
                hc.real * np.cos(theta[t, i]) - hc.imag * np.sin(theta[t, i])
            )

    return tide


def infer_semi_diurnal(
    mjd: np.ndarray,
    ds: xr.Dataset,
    method: str = 'linear',
    corrections: str = 'GOT',
    deltat: np.ndarray | None = None,
) -> np.ndarray:
    """
    Infer minor semi-diurnal tidal constituents

    Uses M2, S2, N2, K2 to infer: eps2, lambda2, l2, t2, r2, eta2

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day
    ds : xr.Dataset
        Interpolated model dataset with harmonic constants
    method : str, default 'linear'
        Inference method ('linear' or 'admittance')
    corrections : str, default 'GOT'
        Nodal correction type ('GOT', 'OTIS', 'FES')
    deltat : np.ndarray, optional
        Time correction for converting to TT

    Returns
    -------
    np.ndarray
        Minor constituent tide prediction (metres)
    """
    mjd = np.atleast_1d(np.asarray(mjd, dtype=np.float64))
    n_times = len(mjd)

    if deltat is None:
        deltat = np.zeros_like(mjd)

    # Get major constituent harmonic constants
    m2 = _get_constituent_amplitude(ds, 'm2')
    s2 = _get_constituent_amplitude(ds, 's2')
    n2 = _get_constituent_amplitude(ds, 'n2')
    k2 = _get_constituent_amplitude(ds, 'k2')

    n_points = len(m2) if np.ndim(m2) > 0 else 1
    m2 = np.atleast_1d(m2)
    s2 = np.atleast_1d(s2)
    n2 = np.atleast_1d(n2)
    k2 = np.atleast_1d(k2)

    # Minor constituents to infer
    minors = ['eps2', 'lambda2', 'l2', 't2', 'eta2']

    # Get nodal corrections for minor constituents
    pu, pf, G = _constituents.arguments(mjd + deltat, minors, corrections=corrections)

    # Calculate phase angles
    if corrections in ('OTIS', 'ATLAS', 'TMD3'):
        omega = _constituents.frequency(minors, corrections=corrections)
        t_days = mjd - _MJD_TIDE
        theta = np.outer(t_days, omega) * 86400.0 + pu
    else:
        theta = np.radians(G) + pu

    # Inference ratios based on equilibrium theory
    if method == 'admittance':
        ratios = {
            'eps2': (0.062, 'n2'),    # eps2 from N2
            'lambda2': (0.084, 'n2'),  # lambda2 from N2
            'l2': (0.085, 'm2'),       # L2 from M2
            't2': (0.112, 's2'),       # T2 from S2
            'eta2': (0.048, 'k2'),     # eta2 from K2
        }
    else:
        # Linear interpolation
        ratios = {
            'eps2': (0.057, 'n2'),
            'lambda2': (0.071, 'n2'),
            'l2': (0.079, 'm2'),
            't2': (0.094, 's2'),
            'eta2': (0.043, 'k2'),
        }

    # Map source constituents to data
    sources = {'m2': m2, 's2': s2, 'n2': n2, 'k2': k2}

    # Initialize output
    tide = np.zeros((n_points, n_times))

    # Calculate contribution from each minor constituent
    for i, minor in enumerate(minors):
        ratio, source = ratios[minor]
        hc = sources[source] * ratio

        # Harmonic synthesis
        for t in range(n_times):
            tide[:, t] += pf[t, i] * (
                hc.real * np.cos(theta[t, i]) - hc.imag * np.sin(theta[t, i])
            )

    return tide


def infer_minor(
    mjd: np.ndarray,
    ds: xr.Dataset,
    method: str = 'linear',
    corrections: str = 'GOT',
    deltat: np.ndarray | None = None,
) -> np.ndarray:
    """
    Infer all minor tidal constituents

    Calculates tidal heights from minor constituents using empirical
    relationships with major constituents.

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day
    ds : xr.Dataset
        Interpolated model dataset with harmonic constants
    method : str, default 'linear'
        Inference method:
        - 'linear': Linear interpolation in frequency domain
        - 'admittance': Equilibrium theory admittance functions
    corrections : str, default 'GOT'
        Nodal correction type ('GOT', 'OTIS', 'FES')
    deltat : np.ndarray, optional
        Time correction for converting to TT (Terrestrial Time)

    Returns
    -------
    np.ndarray
        Minor constituent tide prediction (metres)
        Shape: (n_points, n_times)

    Examples
    --------
    >>> ds = model.open_dataset()
    >>> local = ds.tmd.interp(x=140.0, y=35.0)
    >>> mjd = 60310.0 + np.arange(24) / 24.0
    >>> minor = infer_minor(mjd, local, method='admittance')
    """
    diurnal = infer_diurnal(mjd, ds, method=method, corrections=corrections,
                            deltat=deltat)
    semi_diurnal = infer_semi_diurnal(mjd, ds, method=method, corrections=corrections,
                                       deltat=deltat)

    return diurnal + semi_diurnal
