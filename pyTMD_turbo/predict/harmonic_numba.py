"""
Numba-accelerated harmonic synthesis implementation

Uses JIT compilation and parallelisation to achieve
further performance improvements.

Note: This module requires the optional 'numba' dependency.
Install with: pip install pyTMD_turbo[numba]

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""


import numpy as np

try:
    from numba import float64, jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create dummy decorators for when numba is not installed
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    float64 = None


@jit(nopython=True, parallel=True, fastmath=True)
def predict_numba_parallel(hc_real: np.ndarray,
                           hc_imag: np.ndarray,
                           omega: np.ndarray,
                           phase_0: np.ndarray,
                           t_days: np.ndarray,
                           pu: np.ndarray,
                           pf: np.ndarray) -> np.ndarray:
    """
    Numba parallelised tide prediction

    Parameters
    ----------
    hc_real : np.ndarray
        Real part, shape (n_points, n_constituents)
    hc_imag : np.ndarray
        Imaginary part, shape (n_points, n_constituents)
    omega : np.ndarray
        Angular frequency, shape (n_constituents,)
    phase_0 : np.ndarray
        Phase-0, shape (n_constituents,)
    t_days : np.ndarray
        Days since 1992-01-01, shape (n_times,)
    pu : np.ndarray
        Phase correction, shape (n_times, n_constituents)
    pf : np.ndarray
        Amplitude correction, shape (n_times, n_constituents)

    Returns
    -------
    np.ndarray
        Tide height, shape (n_points, n_times)
    """
    n_points = hc_real.shape[0]
    n_times = len(t_days)
    n_const = len(omega)

    result = np.zeros((n_points, n_times))

    # Parallel loop (per location)
    for p in prange(n_points):
        for t in range(n_times):
            tide = 0.0
            for c in range(n_const):
                # Phase angle
                theta = omega[c] * t_days[t] * 86400.0 + phase_0[c] + pu[t, c]
                # Tide contribution
                tide += pf[t, c] * (hc_real[p, c] * np.cos(theta) - hc_imag[p, c] * np.sin(theta))
            result[p, t] = tide

    return result


@jit(nopython=True, fastmath=True)
def predict_numba_single(hc_real: np.ndarray,
                         hc_imag: np.ndarray,
                         omega: np.ndarray,
                         phase_0: np.ndarray,
                         t_days: np.ndarray,
                         pu: np.ndarray,
                         pf: np.ndarray) -> np.ndarray:
    """
    Numba version of single-location tide prediction

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
    pu : np.ndarray
        Phase correction, shape (n_times, n_constituents)
    pf : np.ndarray
        Amplitude correction, shape (n_times, n_constituents)

    Returns
    -------
    np.ndarray
        Tide height, shape (n_times,)
    """
    n_times = len(t_days)
    n_const = len(omega)

    result = np.zeros(n_times)

    for t in range(n_times):
        tide = 0.0
        for c in range(n_const):
            theta = omega[c] * t_days[t] * 86400.0 + phase_0[c] + pu[t, c]
            tide += pf[t, c] * (hc_real[c] * np.cos(theta) - hc_imag[c] * np.sin(theta))
        result[t] = tide

    return result


@jit(nopython=True, parallel=True, fastmath=True)
def bilinear_interpolate_batch(grid_data: np.ndarray,
                               lon_indices: np.ndarray,
                               lat_indices: np.ndarray,
                               lon_weights: np.ndarray,
                               lat_weights: np.ndarray) -> np.ndarray:
    """
    Batch bilinear interpolation (Numba parallelised version)

    Parameters
    ----------
    grid_data : np.ndarray
        Grid data, shape (n_lat, n_lon)
    lon_indices : np.ndarray
        Longitude indices (integer part), shape (n_points,)
    lat_indices : np.ndarray
        Latitude indices (integer part), shape (n_points,)
    lon_weights : np.ndarray
        Longitude weights (fractional part), shape (n_points,)
    lat_weights : np.ndarray
        Latitude weights (fractional part), shape (n_points,)

    Returns
    -------
    np.ndarray
        Interpolated values, shape (n_points,)
    """
    n_points = len(lon_indices)
    n_lat, n_lon = grid_data.shape

    result = np.zeros(n_points)

    for i in prange(n_points):
        i0 = lon_indices[i]
        j0 = lat_indices[i]
        i1 = min(i0 + 1, n_lon - 1)
        j1 = min(j0 + 1, n_lat - 1)

        wi = lon_weights[i]
        wj = lat_weights[i]

        v00 = grid_data[j0, i0]
        v01 = grid_data[j0, i1]
        v10 = grid_data[j1, i0]
        v11 = grid_data[j1, i1]

        # NaN check
        if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
            result[i] = np.nan
        else:
            result[i] = (v00 * (1-wi) * (1-wj) +
                         v01 * wi * (1-wj) +
                         v10 * (1-wi) * wj +
                         v11 * wi * wj)

    return result


@jit(nopython=True, fastmath=True)
def compute_nodal_m2(mjd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute nodal corrections for M2 constituent (simplified version)

    Computes the 18.61-year nodal correction.

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day

    Returns
    -------
    pu : np.ndarray
        Phase correction (rad)
    pf : np.ndarray
        Amplitude correction factor
    """
    n = len(mjd)
    pu = np.zeros(n)
    pf = np.ones(n)

    # Moon's ascending node longitude (simplified calculation)
    # N = 125.04 - 0.0529539 * d (degrees)
    # d = MJD - 51544.5 (days since J2000)

    for i in range(n):
        d = mjd[i] - 51544.5
        N = np.radians(125.04 - 0.0529539 * d)

        # M2 nodal corrections
        # f = 1.0 - 0.037 * cos(N)
        # u = -2.1 * sin(N) (degrees)
        pf[i] = 1.0 - 0.037 * np.cos(N)
        pu[i] = np.radians(-2.1 * np.sin(N))

    return pu, pf


def warmup():
    """Warm up JIT compilation"""
    # Execute compilation with small data
    hc_real = np.random.randn(10, 5)
    hc_imag = np.random.randn(10, 5)
    omega = np.random.randn(5)
    phase_0 = np.random.randn(5)
    t_days = np.arange(10, dtype=np.float64)
    pu = np.zeros((10, 5))
    pf = np.ones((10, 5))

    # Execute compilation
    _ = predict_numba_parallel(hc_real, hc_imag, omega, phase_0, t_days, pu, pf)
    _ = predict_numba_single(hc_real[0], hc_imag[0], omega, phase_0, t_days, pu, pf)

    # Warm up interpolation
    grid = np.random.randn(100, 100)
    lon_idx = np.zeros(10, dtype=np.int64)
    lat_idx = np.zeros(10, dtype=np.int64)
    lon_w = np.zeros(10)
    lat_w = np.zeros(10)
    _ = bilinear_interpolate_batch(grid, lon_idx, lat_idx, lon_w, lat_w)
