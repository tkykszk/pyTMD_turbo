"""
pyTMD_turbo.constituents - Tidal constituent calculations

Calculates constituent parameters and nodal arguments for tidal prediction.

References:
    A. T. Doodson and H. Warburg, "Admiralty Manual of Tides", HMSO, (1941).
    P. Schureman, "Manual of Harmonic Analysis and Prediction of Tides"
        US Coast and Geodetic Survey, Special Publication, 98, (1958).
    M. G. G. Foreman and R. F. Henry, "The harmonic analysis of tidal model
        time series", Advances in Water Resources, 12, (1989).

Copyright (c) 2024-2026 tkykszk
Derived from pyTMD by Tyler Sutterley (MIT License)
"""

from __future__ import annotations

import json
import pathlib
import numpy as np
from typing import List, Tuple, Optional

__all__ = [
    'arguments',
    'frequency',
    'coefficients_table',
    'nodal_modulation',
    'mean_longitudes',
    'minor_arguments',
]

# Path to data files
_data_path = pathlib.Path(__file__).parent / "data"
_coefficients_table = _data_path / "doodson.json"

# MJD of J2000.0 (2000-01-01T12:00:00)
_mjd_j2000 = 51544.5
# Days per Julian century
_century = 36525.0


def polynomial_sum(coefficients: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Evaluate polynomial with given coefficients

    Parameters
    ----------
    coefficients : numpy.ndarray
        Polynomial coefficients (c0, c1, c2, ...)
    t : numpy.ndarray
        Time values

    Returns
    -------
    numpy.ndarray
        Polynomial sum
    """
    result = np.zeros_like(t, dtype=np.float64)
    for i, c in enumerate(coefficients):
        result += c * np.power(t, i)
    return result


def mean_longitudes(
    MJD: np.ndarray,
    method: str = "Cartwright"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the basic astronomical mean longitudes: S, H, P, N and Ps

    Parameters
    ----------
    MJD : numpy.ndarray
        Modified Julian Day of input date
    method : str, default 'Cartwright'
        Method for calculating mean longitudes
        - 'Cartwright': coefficients from David Cartwright
        - 'Meeus': coefficients from Meeus Astronomical Algorithms
        - 'ASTRO5': Meeus coefficients from ASTRO5

    Returns
    -------
    S : numpy.ndarray
        Mean longitude of moon (degrees)
    H : numpy.ndarray
        Mean longitude of sun (degrees)
    P : numpy.ndarray
        Mean longitude of lunar perigee (degrees)
    N : numpy.ndarray
        Mean longitude of ascending lunar node (degrees)
    Ps : numpy.ndarray
        Longitude of solar perigee (degrees)
    """
    MJD = np.atleast_1d(MJD).astype(np.float64)

    if method.title() == "Meeus":
        # Convert from MJD to days relative to 2000-01-01T12:00:00
        T = MJD - _mjd_j2000

        # Mean longitude of moon
        lunar_longitude = np.array([
            218.3164591, 13.17639647754579, -9.9454632e-13,
            3.8086292e-20, -8.6184958e-27
        ])
        S = polynomial_sum(lunar_longitude, T)

        # Mean longitude of sun
        solar_longitude = np.array([280.46645, 0.985647360164271, 2.2727347e-13])
        H = polynomial_sum(solar_longitude, T)

        # Mean longitude of lunar perigee
        lunar_perigee = np.array([
            83.3532430, 0.11140352391786447, -7.7385418e-12,
            -2.5636086e-19, 2.95738836e-26
        ])
        P = polynomial_sum(lunar_perigee, T)

        # Mean longitude of ascending lunar node
        lunar_node = np.array([
            125.0445550, -0.052953762762491446, 1.55628359e-12,
            4.390675353e-20, -9.26940435e-27
        ])
        N = polynomial_sum(lunar_node, T)

        # Mean longitude of solar perigee
        Ps = 282.94 + (1.7192 * T) / _century

    elif method.upper() == "ASTRO5":
        # Convert from MJD to centuries relative to 2000-01-01T12:00:00
        T = (MJD - _mjd_j2000) / _century

        # Mean longitude of moon
        lunar_longitude = np.array([
            218.3164477, 481267.88123421, -1.5786e-3, 1.855835e-6, -1.53388e-8
        ])
        S = polynomial_sum(lunar_longitude, T)

        # Mean longitude of sun
        lunar_elongation = np.array([
            297.8501921, 445267.1114034, -1.8819e-3, 1.83195e-6, -8.8445e-9
        ])
        H = polynomial_sum(lunar_longitude - lunar_elongation, T)

        # Mean longitude of lunar perigee
        lunar_perigee = np.array([83.3532465, 4069.0137287, -1.032e-2, -1.249172e-5])
        P = polynomial_sum(lunar_perigee, T)

        # Mean longitude of ascending lunar node
        lunar_node = np.array([125.04452, -1934.136261, 2.0708e-3, 2.22222e-6])
        N = polynomial_sum(lunar_node, T)

        # Mean longitude of solar perigee
        Ps = 282.94 + 1.7192 * T

    else:  # Cartwright (default)
        # Convert from MJD to days relative to 2000-01-01T12:00:00
        T = MJD - _mjd_j2000

        # Mean longitude of moon (Cartwright coefficients)
        S = 218.3164 + 13.17639648 * T

        # Mean longitude of sun
        H = 280.4661 + 0.98564736 * T

        # Mean longitude of lunar perigee
        P = 83.3535 + 0.11140353 * T

        # Mean longitude of ascending lunar node
        N = 125.0445 - 0.05295377 * T

        # Mean longitude of solar perigee
        Ps = 282.9384 + 0.00000471 * T

    # Normalize to [0, 360) and return mean longitudes (degrees)
    S = np.mod(S, 360.0)
    H = np.mod(H, 360.0)
    P = np.mod(P, 360.0)
    N = np.mod(N, 360.0)
    Ps = np.mod(Ps, 360.0)

    return S, H, P, N, Ps


def coefficients_table(
    constituents: list | tuple | np.ndarray | str,
    corrections: str = 'OTIS',
) -> np.ndarray:
    """
    Doodson table coefficients for tidal constituents

    Parameters
    ----------
    constituents : list, tuple, np.ndarray or str
        Tidal constituent IDs
    corrections : str, default 'OTIS'
        Use coefficients from OTIS, FES or GOT models

    Returns
    -------
    coef : numpy.ndarray
        Doodson coefficients (Cartwright numbers) for each constituent
        Shape: (7, nc) where columns are [tau, s, h, p, n, pp, k]
    """
    # Load coefficients table
    with open(_coefficients_table, 'r', encoding='utf-8') as f:
        coefficients = json.load(f)

    # Set s1 coefficients for OTIS type models
    if corrections in ('OTIS', 'ATLAS', 'TMD3', 'netcdf'):
        coefficients['s1'] = [1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0]

    # Make constituents iterable
    if isinstance(constituents, str):
        constituents = [constituents]

    # Allocate output
    nc = len(constituents)
    coef = np.zeros((7, nc))

    # Get coefficients for each constituent
    for i, c in enumerate(constituents):
        c_lower = c.lower()
        if c_lower in coefficients:
            coef[:, i] = coefficients[c_lower]
        else:
            raise ValueError(f"Unsupported constituent: {c}")

    return coef


def nodal_modulation(
    n: np.ndarray,
    p: np.ndarray,
    constituents: list | tuple | np.ndarray | str,
    corrections: str = 'OTIS',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate nodal corrections for tidal constituents

    Parameters
    ----------
    n : numpy.ndarray
        Mean longitude of ascending lunar node (degrees)
    p : numpy.ndarray
        Mean longitude of lunar perigee (degrees)
    constituents : list, tuple, np.ndarray or str
        Tidal constituent IDs
    corrections : str, default 'OTIS'
        Use nodal corrections from OTIS, FES or GOT models

    Returns
    -------
    u : numpy.ndarray
        Nodal correction angle (radians)
    f : numpy.ndarray
        Nodal modulation factor
    """
    # Make constituents iterable
    if isinstance(constituents, str):
        constituents = [constituents]

    # Convert longitudes to radians
    N = np.radians(n)
    P = np.radians(p)

    # Trigonometric factors
    sinn = np.sin(N)
    cosn = np.cos(N)
    sin2n = np.sin(2.0 * N)
    cos2n = np.cos(2.0 * N)
    sin3n = np.sin(3.0 * N)
    sinp = np.sin(P)
    cosp = np.cos(P)
    sin2p = np.sin(2.0 * P)
    cos2p = np.cos(2.0 * P)

    # Set correction type
    OTIS_TYPE = corrections in ('OTIS', 'ATLAS', 'TMD3', 'netcdf')
    FES_TYPE = corrections in ('FES',)

    # Initialize arrays
    nt = len(np.atleast_1d(n))
    nc = len(constituents)
    f = np.zeros((nt, nc))
    u = np.zeros((nt, nc))

    # Compute nodal corrections for each constituent
    for i, c in enumerate(constituents):
        c_lower = c.lower()

        if not corrections:
            # No corrections
            f[:, i] = 1.0
            u[:, i] = 0.0

        elif c_lower in ('msf', 'tau1', 'p1', 'theta1', 'lambda2', 's2') and OTIS_TYPE:
            f[:, i] = 1.0
            u[:, i] = 0.0

        elif c_lower in ('p1', 's2') and FES_TYPE:
            f[:, i] = 1.0
            u[:, i] = 0.0

        elif c_lower in ('mm', 'msm') and OTIS_TYPE:
            f[:, i] = 1.0 - 0.130 * cosn
            u[:, i] = 0.0

        elif c_lower in ('mf', 'msqm', 'msp', 'mq', 'mtm') and OTIS_TYPE:
            f[:, i] = 1.043 + 0.414 * cosn
            u[:, i] = np.radians(-23.7 * sinn + 2.7 * sin2n - 0.4 * sin3n)

        elif c_lower in ('o1', 'so3', 'op2') and OTIS_TYPE:
            term1 = 0.189 * sinn - 0.0058 * sin2n
            term2 = 1.0 + 0.189 * cosn - 0.0058 * cos2n
            f[:, i] = np.sqrt(term1**2 + term2**2)
            u[:, i] = np.radians(10.8 * sinn - 1.3 * sin2n + 0.2 * sin3n)

        elif c_lower in ('2q1', 'q1', 'rho1', 'sigma1') and OTIS_TYPE:
            f[:, i] = np.sqrt((1.0 + 0.188 * cosn)**2 + (0.188 * sinn)**2)
            u[:, i] = np.arctan(0.189 * sinn / (1.0 + 0.189 * cosn))

        elif c_lower in ('k1', 'sk3', 'ok1') and OTIS_TYPE:
            term1 = -0.1554 * sinn + 0.0029 * sin2n
            term2 = 1.0 + 0.1158 * cosn - 0.0029 * cos2n
            f[:, i] = np.sqrt(term1**2 + term2**2)
            u[:, i] = np.radians(-8.9 * sinn + 0.7 * sin2n + 0.0 * sin3n)

        elif c_lower in ('j1', 'chi1', 'phi1', 'theta1', 'oo1') and OTIS_TYPE:
            term1 = -0.227 * sinn
            term2 = 1.0 + 0.169 * cosn
            f[:, i] = np.sqrt(term1**2 + term2**2)
            u[:, i] = np.arctan(term1 / term2)

        elif c_lower in ('m1',) and OTIS_TYPE:
            # M1 with perth5 coefficients
            term1 = (-0.2294 * sinn - 0.3594 * sin2p
                     - 0.0664 * np.sin(2.0 * P - N))
            term2 = (1.0 - 0.2294 * cosn + 0.3594 * cos2p
                     + 0.0664 * np.cos(2.0 * P - N))
            f[:, i] = np.sqrt(term1**2 + term2**2)
            u[:, i] = np.arctan2(term1, term2)

        elif c_lower in ('n2', 'nu2', '2n2', 'mu2', 'l2') and OTIS_TYPE:
            f[:, i] = np.sqrt((1.0 - 0.0373 * cosn)**2 + (0.0373 * sinn)**2)
            u[:, i] = np.arctan(0.0373 * sinn / (1.0 - 0.0373 * cosn))

        elif c_lower in ('m2', 'ms4', 'm4', 'mn4', 's4', 'mk3', '2mk5', '2ms6', '2sm6') and OTIS_TYPE:
            f[:, i] = np.sqrt((1.0 - 0.0373 * cosn)**2 + (0.0373 * sinn)**2)
            u[:, i] = np.arctan(-0.0373 * sinn / (1.0 - 0.0373 * cosn))

        elif c_lower in ('k2', '2sk5', 'sk4') and OTIS_TYPE:
            term1 = -0.3108 * sinn - 0.0324 * sin2n
            term2 = 1.0 + 0.2852 * cosn + 0.0324 * cos2n
            f[:, i] = np.sqrt(term1**2 + term2**2)
            u[:, i] = np.arctan(term1 / term2)

        else:
            # Default: no correction
            f[:, i] = 1.0
            u[:, i] = 0.0

    return u, f


def arguments(
    MJD: np.ndarray,
    constituents: list | np.ndarray,
    corrections: str = 'OTIS',
    deltat: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the nodal corrections for tidal constituents

    Parameters
    ----------
    MJD : numpy.ndarray
        Modified Julian day of input date
    constituents : list
        Tidal constituent IDs
    corrections : str, default 'OTIS'
        Use nodal corrections from OTIS, FES or GOT models
    deltat : float, default 0.0
        Time correction for converting to Ephemeris Time (days)

    Returns
    -------
    pu : numpy.ndarray
        Nodal correction angle (radians)
    pf : numpy.ndarray
        Nodal modulation factor
    G : numpy.ndarray
        Phase correction (degrees)
    """
    MJD = np.atleast_1d(MJD).astype(np.float64)

    # Set method based on corrections type
    if corrections in ('OTIS', 'ATLAS', 'TMD3', 'netcdf'):
        method = 'Cartwright'
    else:
        method = 'ASTRO5'

    # Get mean longitudes
    s, h, p, n, pp = mean_longitudes(MJD + deltat, method=method)

    # Number of temporal values
    nt = len(MJD)

    # Convert hours to mean lunar time
    hour = 24.0 * np.mod(MJD, 1)
    tau = 15.0 * hour - s + h

    # Variable for multiples of 90 degrees
    k = 90.0 + np.zeros(nt)

    # Calculate equilibrium arguments
    fargs = np.column_stack([tau, s, h, p, n, pp, k])
    coef = coefficients_table(constituents, corrections=corrections)
    G = np.dot(fargs, coef)

    # Calculate nodal modulations
    pu, pf = nodal_modulation(n, p, constituents, corrections=corrections)

    return pu, pf, G


def frequency(
    constituents: list | tuple | np.ndarray | str,
    corrections: str = 'OTIS',
) -> np.ndarray:
    """
    Calculate angular frequency for tidal constituents

    Parameters
    ----------
    constituents : list, tuple, np.ndarray or str
        Tidal constituent IDs
    corrections : str, default 'OTIS'
        Use corrections from OTIS, FES or GOT models

    Returns
    -------
    omega : numpy.ndarray
        Angular frequency (radians per second)
    """
    # Make constituents iterable
    if isinstance(constituents, str):
        constituents = [constituents]

    # Get Doodson coefficients
    coef = coefficients_table(constituents, corrections=corrections)

    # Fundamental frequencies (cycles per day)
    # tau: mean lunar time (1.0350 cycles/day = 1 + 0.0350)
    # s: mean longitude of moon (0.0369 cycles/day)
    # h: mean longitude of sun (0.0027 cycles/day)
    # p: mean longitude of lunar perigee (0.000305 cycles/day)
    # n: mean longitude of lunar node (-0.000145 cycles/day)
    # pp: longitude of solar perigee (0.00000047 cycles/day)

    freq = np.array([
        1.0350,      # tau (lunar day frequency)
        0.0369,      # s
        0.0274,      # h
        0.00031,     # p
        -0.000145,   # n
        4.7e-7,      # pp
        0.0,         # k (phase, not frequency)
    ])

    # Calculate frequency for each constituent (cycles per day)
    cpd = np.dot(freq, coef)

    # Convert to radians per second
    omega = np.abs(cpd * 2.0 * np.pi / 86400.0)

    return omega


# Minor constituents for inference
_MINOR_CONSTITUENTS = [
    "2q1", "sigma1", "rho1", "m1b", "m1a", "chi1",
    "pi1", "phi1", "theta1", "j1", "oo1",
    "2n2", "mu2", "nu2", "lambda2", "l2a", "l2b",
    "t2", "eps2", "eta2",
]


def _minor_table(corrections: str = 'OTIS') -> np.ndarray:
    """
    Get Doodson coefficients table for minor tidal constituents

    Parameters
    ----------
    corrections : str, default 'OTIS'
        Use corrections from OTIS, FES or GOT models

    Returns
    -------
    coef : np.ndarray
        Doodson coefficients (7, 20) for each minor constituent
    """
    return coefficients_table(_MINOR_CONSTITUENTS, corrections=corrections)


def minor_arguments(
    MJD: np.ndarray,
    deltat: float = 0.0,
    corrections: str = 'OTIS',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate nodal corrections for minor tidal constituents

    Used to infer minor constituent values from major constituents.

    Parameters
    ----------
    MJD : np.ndarray
        Modified Julian Day of input date
    deltat : float, default 0.0
        Time correction for converting to Ephemeris Time (days)
    corrections : str, default 'OTIS'
        Use nodal corrections from OTIS, FES or GOT models

    Returns
    -------
    pu : np.ndarray
        Nodal correction angle (radians), shape (nt, 20)
    pf : np.ndarray
        Nodal modulation factor, shape (nt, 20)
    G : np.ndarray
        Phase correction (degrees), shape (nt, 20)

    References
    ----------
    Doodson, A. T. and H. Warburg, "Admiralty Manual of Tides", HMSO, (1941).
    Schureman, P., "Manual of Harmonic Analysis and Prediction of Tides"
        US Coast and Geodetic Survey, Special Publication, 98, (1958).
    """
    MJD = np.atleast_1d(MJD).astype(np.float64)

    # Set method for astronomical longitudes based on correction type
    if corrections.upper() in ('OTIS', 'ATLAS', 'TMD3', 'NETCDF'):
        method = 'Cartwright'
    else:
        method = 'ASTRO5'

    # Get mean longitudes
    s, h, p, n, pp = mean_longitudes(MJD + deltat, method=method)

    # Number of temporal values
    nt = len(MJD)

    # Hour of day and convert to mean lunar time
    hour = 24.0 * np.mod(MJD, 1)
    tau = 15.0 * hour - s + h

    # Variable for multiples of 90 degrees (Ray technical note 2017)
    k = 90.0 + np.zeros(nt)

    # Determine equilibrium arguments
    fargs = np.column_stack([tau, s, h, p, n, pp, k])
    coef = _minor_table(corrections=corrections)
    arg = np.dot(fargs, coef)

    # Convert mean longitudes to radians
    P = np.radians(p)
    N = np.radians(n)

    # Trigonometric terms
    sinn = np.sin(N)
    cosn = np.cos(N)
    sin2n = np.sin(2.0 * N)
    cos2n = np.cos(2.0 * N)

    # Initialize nodal factor corrections (pf) and angle corrections (pu)
    pf = np.ones((nt, 20))
    pu = np.zeros((nt, 20))

    # OTIS-style corrections (default)
    # 2Q1, sigma1, rho1 (indices 0, 1, 2)
    pf[:, 0] = np.sqrt((1.0 + 0.189 * cosn - 0.0058 * cos2n)**2 +
                       (0.189 * sinn - 0.0058 * sin2n)**2)
    pf[:, 1] = pf[:, 0]
    pf[:, 2] = pf[:, 0]

    # M12 (index 3)
    pf[:, 3] = np.sqrt((1.0 + 0.185 * cosn)**2 + (0.185 * sinn)**2)

    # M11 (index 4)
    pf[:, 4] = np.sqrt((1.0 + 0.201 * cosn)**2 + (0.201 * sinn)**2)

    # chi1 (index 5)
    pf[:, 5] = np.sqrt((1.0 + 0.221 * cosn)**2 + (0.221 * sinn)**2)

    # J1 (index 9)
    pf[:, 9] = np.sqrt((1.0 + 0.198 * cosn)**2 + (0.198 * sinn)**2)

    # OO1 (index 10)
    pf[:, 10] = np.sqrt((1.0 + 0.640 * cosn + 0.134 * cos2n)**2 +
                        (0.640 * sinn + 0.134 * sin2n)**2)

    # 2N2, mu2, nu2 (indices 11, 12, 13)
    pf[:, 11] = np.sqrt((1.0 - 0.0373 * cosn)**2 + (0.0373 * sinn)**2)
    pf[:, 12] = pf[:, 11]
    pf[:, 13] = pf[:, 11]

    # L2a (index 15)
    pf[:, 15] = pf[:, 11]

    # L2b (index 16)
    pf[:, 16] = np.sqrt((1.0 + 0.441 * cosn)**2 + (0.441 * sinn)**2)

    # Nodal angle corrections (pu)
    # 2Q1, sigma1, rho1
    pu[:, 0] = np.arctan2(0.189 * sinn - 0.0058 * sin2n,
                          1.0 + 0.189 * cosn - 0.0058 * sin2n)
    pu[:, 1] = pu[:, 0]
    pu[:, 2] = pu[:, 0]

    # M12
    pu[:, 3] = np.arctan2(0.185 * sinn, 1.0 + 0.185 * cosn)

    # M11
    pu[:, 4] = np.arctan2(-0.201 * sinn, 1.0 + 0.201 * cosn)

    # chi1
    pu[:, 5] = np.arctan2(-0.221 * sinn, 1.0 + 0.221 * cosn)

    # J1
    pu[:, 9] = np.arctan2(-0.198 * sinn, 1.0 + 0.198 * cosn)

    # OO1
    pu[:, 10] = np.arctan2(-0.640 * sinn - 0.134 * sin2n,
                           1.0 + 0.640 * cosn + 0.134 * cos2n)

    # 2N2, mu2, nu2
    pu[:, 11] = np.arctan2(-0.0373 * sinn, 1.0 - 0.0373 * cosn)
    pu[:, 12] = pu[:, 11]
    pu[:, 13] = pu[:, 11]

    # L2a, L2b
    pu[:, 15] = pu[:, 11]
    pu[:, 16] = np.arctan2(-0.441 * sinn, 1.0 + 0.441 * cosn)

    # FES-style corrections
    if corrections.upper() == 'FES':
        from .astro.ephemeris import schureman_arguments
        II, xi, nu, Qa, Qu, Ra, Ru, nu_prime, nu_sec = schureman_arguments(P, N)

        # Nodal factors for FES
        pf[:, 0] = np.sin(II) * (np.cos(II / 2.0)**2) / 0.38  # 2Q1
        pf[:, 1] = pf[:, 0]  # sigma1
        pf[:, 2] = pf[:, 0]  # rho1
        pf[:, 3] = pf[:, 0]  # M12
        pf[:, 4] = np.sin(2.0 * II) / 0.7214  # M11
        pf[:, 5] = pf[:, 4]  # chi1
        pf[:, 9] = pf[:, 4]  # J1
        pf[:, 10] = np.sin(II) * np.power(np.sin(II / 2.0), 2.0) / 0.01640  # OO1
        pf[:, 11] = np.power(np.cos(II / 2.0), 4.0) / 0.9154  # 2N2
        pf[:, 12] = pf[:, 11]  # mu2
        pf[:, 13] = pf[:, 11]  # nu2
        pf[:, 14] = pf[:, 11]  # lambda2
        pf[:, 15] = pf[:, 11] / Ra  # L2a
        pf[:, 18] = pf[:, 11]  # eps2
        pf[:, 19] = np.power(np.sin(II), 2.0) / 0.1565  # eta2

        # Nodal angles for FES
        pu[:, 0] = 2.0 * xi - nu  # 2Q1
        pu[:, 1] = pu[:, 0]  # sigma1
        pu[:, 2] = pu[:, 0]  # rho1
        pu[:, 3] = pu[:, 0]  # M12
        pu[:, 4] = -nu  # M11
        pu[:, 5] = pu[:, 4]  # chi1
        pu[:, 9] = pu[:, 4]  # J1
        pu[:, 10] = -2.0 * xi - nu  # OO1
        pu[:, 11] = 2.0 * xi - 2.0 * nu  # 2N2
        pu[:, 12] = pu[:, 11]  # mu2
        pu[:, 13] = pu[:, 11]  # nu2
        pu[:, 14] = 2.0 * xi - 2.0 * nu  # lambda2
        pu[:, 15] = 2.0 * xi - 2.0 * nu - Ru  # L2a
        pu[:, 18] = pu[:, 12]  # eps2
        pu[:, 19] = -2.0 * nu  # eta2

    elif corrections.upper() == 'GOT':
        # GOT-style corrections
        pf[:, 18] = pf[:, 11]  # eps2
        pf[:, 19] = np.sqrt((1.0 + 0.441 * cosn)**2 + (0.441 * sinn)**2)  # eta2
        pu[:, 18] = pu[:, 11]  # eps2
        pu[:, 19] = np.arctan2(-0.441 * sinn, 1.0 + 0.441 * cosn)  # eta2

    # Equilibrium phase argument (G)
    G = arg

    return pu, pf, G
