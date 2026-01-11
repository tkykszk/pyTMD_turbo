"""
pyTMD_turbo.predict.equilibrium - Long-Period Equilibrium Tide

Computes long-period equilibrium tides (LPET) using the summation of
fifteen tidal spectral lines from Cartwright-Tayler-Edden tables.

References:
    Cartwright & Tayler (1971) - Geophysical Journal Int.
    Cartwright & Edden (1973) - Geophysical Journal Int.
    Wahr (1981) - Geophysical Journal Int.

Copyright (c) 2024-2026 tkykszk
Derived from pyTMD by Tyler Sutterley (MIT License)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Optional, Union, List

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

__all__ = [
    'equilibrium_tide',
    'LPET_elevations',
    'mean_longitudes',
    'legendre_polynomial',
]

# Constants
_MJD_TIDE_EPOCH = 48622.0  # MJD of 1992-01-01T00:00:00
_MJD_J2000 = 51544.5       # MJD of J2000.0
_JULIAN_CENTURY = 36525.0  # Days per Julian century

# Long-period constituent Doodson coefficients (7 x 15)
# Format: [tau, s, h, p, N', pp, k] for each of 15 constituents
# tau is always 0 for long-period constituents
_DOODSON_COEFFICIENTS = np.array([
    # tau, s,  h,  p, N', pp, k  - Constituent name
    [0,   0,  0,  0,  1,  0,  0],  # node
    [0,   0,  1,  0,  0, -1,  0],  # sa (solar annual)
    [0,   0,  2,  0,  0,  0,  0],  # ssa (solar semiannual)
    [0,   1,  0, -1,  0,  0,  0],  # mm (monthly)
    [0,   1,  0,  1,  0,  0,  0],  # mm (variation)
    [0,   2, -2,  0,  0,  0,  0],  # mf (fortnightly)
    [0,   2,  0,  0,  0,  0,  0],  # mf (variation)
    [0,   2,  0,  0, -1,  0,  0],  # mf nodal
    [0,   3, -2, -1,  0,  0,  0],  # mt (monthly tertiary)
    [0,   3,  0, -1,  0,  0,  0],  # mt variation
    [0,   3,  0,  1,  0,  0,  0],  # mt variation 2
    [0,   4, -2,  0,  0,  0,  0],  # msf
    [0,   4,  0,  0,  0,  0,  0],  # msf variation
    [0,   4,  0,  0, -1,  0,  0],  # msf nodal
    [0,   4,  0,  0, -2,  0,  0],  # msf double nodal
], dtype=np.float64)

# Cartwright-Edden amplitudes (centimeters)
# These are the tidal potential amplitudes for degree-2 terms
_CTE_AMPLITUDES = np.array([
    2.7929,   # node
    -0.4922,  # sa
    -3.0988,  # ssa
    -3.5184,  # mm
    0.1054,   # mm variation
    -6.6607,  # mf
    1.3174,   # mf variation
    -0.1527,  # mf nodal
    -0.4138,  # mt
    -0.2965,  # mt variation
    0.0890,   # mt variation 2
    -0.6040,  # msf
    0.2027,   # msf variation
    -0.0235,  # msf nodal
    0.0020,   # msf double nodal
], dtype=np.float64)

# Constituent names for reference
_CONSTITUENT_NAMES = [
    'node', 'sa', 'ssa', 'mm', 'mm2', 'mf', 'mf2', 'mf_nodal',
    'mt', 'mt2', 'mt3', 'msf', 'msf2', 'msf_nodal', 'msf_nodal2'
]

# Mean longitude polynomial coefficients (Meeus, Astronomical Algorithms)
# From J2000.0, in degrees
_MEAN_LONGITUDE_COEFFICIENTS = {
    # s: Mean longitude of Moon
    's': np.array([218.3164477, 481267.88123421, -1.5786e-3, 1.855835e-6, -1.53388e-8]),
    # h: Mean longitude of Sun
    'h': np.array([280.4664567, 360007.6982779, 3.0323e-4, -2.0e-8]),
    # p: Mean longitude of lunar perigee
    'p': np.array([83.3532465, 4069.0137287, -1.032e-2, -1.249e-5]),
    # N: Mean longitude of ascending lunar node
    'N': np.array([125.04452, -1934.136261, 2.0708e-3, 2.22222e-6]),
    # pp: Mean longitude of solar perigee (perihelion)
    'pp': np.array([282.93768, 1.7195269, 3.086e-4]),
}


def mean_longitudes(mjd: np.ndarray) -> dict:
    """
    Calculate mean astronomical longitudes

    Parameters
    ----------
    mjd : numpy.ndarray
        Modified Julian Day

    Returns
    -------
    longitudes : dict
        Dictionary with keys 's', 'h', 'p', 'N', 'pp' containing
        mean longitudes in degrees (0-360)
    """
    mjd = np.atleast_1d(mjd)

    # Julian centuries from J2000.0
    T = (mjd - _MJD_J2000) / _JULIAN_CENTURY

    longitudes = {}
    for key, coeffs in _MEAN_LONGITUDE_COEFFICIENTS.items():
        # Evaluate polynomial
        value = np.zeros_like(T)
        for i, c in enumerate(coeffs):
            value = value + c * np.power(T, i)
        # Normalize to 0-360 degrees
        longitudes[key] = np.mod(value, 360.0)

    return longitudes


def legendre_polynomial(lat: np.ndarray, l: int = 2, m: int = 0) -> np.ndarray:
    """
    Calculate normalized associated Legendre polynomial

    Parameters
    ----------
    lat : numpy.ndarray
        Latitude in degrees
    l : int, default 2
        Degree
    m : int, default 0
        Order

    Returns
    -------
    P : numpy.ndarray
        Normalized Legendre polynomial value
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    theta = np.radians(90.0 - lat)  # Colatitude in radians

    if l == 2 and m == 0:
        # P_2^0(cos(theta)) = (3*cos^2(theta) - 1) / 2
        cos_theta = np.cos(theta)
        P = (3.0 * cos_theta**2 - 1.0) / 2.0

        # Normalization factor: sqrt((2l+1)/(4*pi))
        norm = np.sqrt(5.0 / (4.0 * np.pi))
        return norm * P

    elif l == 2 and m == 1:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        P = 3.0 * cos_theta * sin_theta
        norm = np.sqrt(5.0 / (24.0 * np.pi))
        return norm * P

    elif l == 2 and m == 2:
        sin_theta = np.sin(theta)
        P = 3.0 * sin_theta**2
        norm = np.sqrt(5.0 / (96.0 * np.pi))
        return norm * P

    else:
        raise ValueError(f"Legendre polynomial P_{l}^{m} not implemented")


def equilibrium_tide(
    t: np.ndarray,
    lat: np.ndarray,
    lon: Optional[np.ndarray] = None,
    deltat: Union[float, np.ndarray] = 0.0,
    constituents: Optional[List[int]] = None,
    h2: float = 0.606,
    k2: float = 0.299,
) -> np.ndarray:
    """
    Calculate long-period equilibrium tide

    Computes the long-period equilibrium tidal elevation using the
    summation of fifteen spectral lines from Cartwright-Tayler-Edden
    tables.

    Parameters
    ----------
    t : numpy.ndarray
        Time in days relative to 1992-01-01T00:00:00 (tide epoch)
        or Modified Julian Day if values > 40000
    lat : numpy.ndarray
        Latitude in degrees
    lon : numpy.ndarray, optional
        Longitude in degrees (not used for degree-2 order-0 terms)
    deltat : float or array, default 0.0
        TT - UT1 time correction in days
    constituents : list, optional
        Indices of constituents to include (0-14).
        Default is all 15 constituents.
    h2 : float, default 0.606
        Degree-2 Love number (vertical displacement)
    k2 : float, default 0.299
        Degree-2 potential Love number

    Returns
    -------
    lpet : numpy.ndarray
        Long-period equilibrium tide elevation in meters.
        Shape is (n_points, n_times) or (n_times,) for single point.

    Notes
    -----
    The equilibrium tide is calculated as:

        LPET = P_2^0(lat) × Σ[A_i × γ_2 × cos(G_i)]

    where:
        - P_2^0 is the degree-2, order-0 normalized Legendre polynomial
        - A_i is the Cartwright-Edden amplitude for constituent i
        - γ_2 = 1 + k_2 - h_2 is the tilt factor
        - G_i is the phase argument from Doodson coefficients

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0.0])  # 1992-01-01
    >>> lat = np.array([45.0])
    >>> lpet = equilibrium_tide(t, lat)
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))

    # Determine if t is relative to tide epoch or MJD
    if np.mean(t) > 40000:
        # Assume MJD
        mjd = t
    else:
        # Assume days since 1992-01-01
        mjd = t + _MJD_TIDE_EPOCH

    # Apply time correction
    mjd = mjd - deltat

    n_points = len(lat)
    n_times = len(mjd)

    # Select constituents
    if constituents is None:
        constituents = list(range(15))

    # Calculate tilt factor
    gamma_2 = 1.0 + k2 - h2

    # Calculate spatial response (Legendre polynomial)
    P20 = legendre_polynomial(lat, l=2, m=0)  # Shape: (n_points,)

    # Calculate mean longitudes for all times
    longs = mean_longitudes(mjd)
    s = longs['s']   # Mean longitude of Moon
    h = longs['h']   # Mean longitude of Sun
    p = longs['p']   # Mean longitude of lunar perigee
    N = longs['N']   # Mean longitude of ascending node
    pp = longs['pp'] # Mean longitude of solar perigee

    # N' = 360 - N (negative of ascending node longitude)
    Np = 360.0 - N

    # Initialize output
    lpet = np.zeros((n_points, n_times))

    # Sum over selected constituents
    for i in constituents:
        # Get Doodson coefficients for this constituent
        coef = _DOODSON_COEFFICIENTS[i]
        amp = _CTE_AMPLITUDES[i]

        # Calculate phase argument G
        # G = tau*τ + s*s + h*h + p*p + N'*Np + pp*pp + k*90
        # For long-period, tau coefficient is always 0
        G = (coef[1] * s + coef[2] * h + coef[3] * p +
             coef[4] * Np + coef[5] * pp + coef[6] * 90.0)

        # Convert to radians
        G_rad = np.radians(G)

        # Add contribution: A * gamma_2 * cos(G)
        # Broadcast: P20 is (n_points,), cos(G) is (n_times,)
        for ip in range(n_points):
            lpet[ip, :] += P20[ip] * amp * gamma_2 * np.cos(G_rad)

    # Convert from centimeters to meters
    lpet = lpet / 100.0

    # Return scalar array for single point
    if n_points == 1:
        return lpet[0]

    return lpet


def LPET_elevations(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Calculate Long-Period Equilibrium Tide elevations

    High-level wrapper compatible with pyTMD.compute.LPET_elevations.

    Parameters
    ----------
    x : numpy.ndarray
        Longitudes in degrees
    y : numpy.ndarray
        Latitudes in degrees
    times : numpy.ndarray
        Times (datetime64, datetime, or MJD)
    **kwargs
        Additional arguments passed to equilibrium_tide()

    Returns
    -------
    lpet : numpy.ndarray
        Long-period equilibrium tide elevation in meters

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([140.0])
    >>> y = np.array([35.0])
    >>> times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')
    >>> lpet = LPET_elevations(x, y, times)
    """
    from datetime import datetime

    # Convert times to MJD
    times = np.atleast_1d(times)
    if isinstance(times[0], np.datetime64):
        times_unix = times.astype('datetime64[s]').astype('float64')
        mjd = times_unix / 86400.0 + 40587.0
    elif isinstance(times[0], datetime):
        mjd = np.array([
            t.timestamp() / 86400.0 + 40587.0
            for t in times
        ])
    else:
        # Assume already MJD
        mjd = np.asarray(times, dtype=np.float64)

    # Convert to days since tide epoch
    t = mjd - _MJD_TIDE_EPOCH

    return equilibrium_tide(t, y, lon=x, **kwargs)
