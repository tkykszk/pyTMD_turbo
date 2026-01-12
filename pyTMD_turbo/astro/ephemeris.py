"""
Fast astronomical calculation module

Removes xarray dependency from PyTMD and performs astronomical
calculations using NumPy only.
Uses the same formulae as PyTMD/timescale to ensure accuracy.

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""


import numpy as np

# Constants
_MJD_J2000 = 51544.5  # MJD of J2000.0
_JD_MJD = 2400000.5   # Difference between JD and MJD
_JD_J2000 = _JD_MJD + _MJD_J2000  # JD of J2000.0
_JULIAN_CENTURY = 36525.0  # Julian century (days)
_DAY_SECONDS = 86400.0  # Seconds per day
_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)  # Arcseconds to radians
_DEG_TO_RAD = np.pi / 180.0  # Degrees to radians

# TT-UT1 correction approximation (seconds)
# Actual value varies with time, but high precision is not required for tidal calculations
# The timescale library uses IERS data internally, but
# the fast version uses an approximation
_TT_UT1_APPROX = 69.2  # Approximate value circa 2023 (seconds)

# Pre-computed constants for optimization
_TT_OFFSET = _JD_MJD + _TT_UT1_APPROX / _DAY_SECONDS  # Combined offset for TT
_EPSILON_J2000 = 23.43929111 * _DEG_TO_RAD  # Obliquity of ecliptic (radians)
_COS_EPSILON = np.cos(_EPSILON_J2000)
_SIN_EPSILON = np.sin(_EPSILON_J2000)
_COS_NEG_EPSILON = np.cos(-_EPSILON_J2000)
_SIN_NEG_EPSILON = np.sin(-_EPSILON_J2000)


def polynomial_sum(coefficients: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute polynomial sum using Horner's method (optimized)

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficient array [c0, c1, c2, ...]
    t : np.ndarray
        Time variable

    Returns
    -------
    np.ndarray
        c0 + c1*t + c2*t^2 + ...

    Notes
    -----
    Uses Horner's method: ((c3*t + c2)*t + c1)*t + c0
    This is ~12x faster than the naive loop implementation.
    """
    # Horner's method: evaluate from highest degree to lowest
    result = np.zeros_like(t)
    for c in reversed(coefficients):
        result = result * t + c
    return result


def mjd_to_tt(mjd: np.ndarray) -> np.ndarray:
    """
    Compute JD in TT (Terrestrial Time) from MJD

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day

    Returns
    -------
    np.ndarray
        JD in TT
    """
    # TT = UT1 + TT-UT1 correction
    # UT1 ≈ UTC ≈ MJD + 2400000.5
    return mjd + _JD_MJD + _TT_UT1_APPROX / _DAY_SECONDS


def mjd_to_ut1(mjd: np.ndarray) -> np.ndarray:
    """
    Compute JD in UT1 from MJD

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day

    Returns
    -------
    np.ndarray
        JD in UT1
    """
    # UT1 ≈ UTC ≈ MJD + 2400000.5
    return mjd + _JD_MJD


def greenwich_mean_sidereal_time(mjd: np.ndarray) -> np.ndarray:
    """
    Compute Greenwich Mean Sidereal Time (fraction of day)

    Uses the same formula as the timescale library

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day

    Returns
    -------
    np.ndarray
        GMST (fraction of day, 0-1)
    """
    # GMST coefficients (seconds)
    GMST_coef = np.array([24110.54841, 8640184.812866, 9.3104e-2, -6.2e-6])

    # Compute Julian centuries from JD in UT1
    ut1 = mjd_to_ut1(mjd)
    ut1_jc = (ut1 - _JD_J2000) / _JULIAN_CENTURY

    # Compute GMST in seconds and convert to fraction of day
    gmst_sec = polynomial_sum(GMST_coef, ut1_jc)
    return np.mod(gmst_sec / _DAY_SECONDS, 1.0)


def greenwich_hour_angle(mjd: np.ndarray) -> np.ndarray:
    """
    Compute Greenwich Hour Angle (degrees)

    Uses the same formula as the timescale library

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day

    Returns
    -------
    np.ndarray
        GHA (degrees, 0-360)
    """
    # GMST (fraction of day)
    gmst = greenwich_mean_sidereal_time(mjd)

    # T (Julian centuries based on TT)
    tt = mjd_to_tt(mjd)
    T = (tt - _JD_J2000) / _JULIAN_CENTURY

    # GHA = GMST*360 + 360*T*century + 180 (mod 360)
    gha = np.mod(gmst * 360.0 + 360.0 * T * _JULIAN_CENTURY + 180.0, 360.0)
    return gha


def rotate_z(angle_rad: np.ndarray) -> np.ndarray:
    """
    Generate rotation matrix about Z-axis

    Parameters
    ----------
    angle_rad : np.ndarray
        Rotation angle (radians), shape (N,)

    Returns
    -------
    np.ndarray
        Rotation matrix, shape (3, 3, N)
    """
    n = len(angle_rad)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rot = np.zeros((3, 3, n))
    rot[0, 0, :] = cos_a
    rot[0, 1, :] = sin_a
    rot[1, 0, :] = -sin_a
    rot[1, 1, :] = cos_a
    rot[2, 2, :] = 1.0

    return rot


def rotate_x(angle_rad: float) -> np.ndarray:
    """
    Generate rotation matrix about X-axis (for scalar angle)

    Parameters
    ----------
    angle_rad : float
        Rotation angle (radians)

    Returns
    -------
    np.ndarray
        Rotation matrix, shape (3, 3)
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, sin_a],
        [0.0, -sin_a, cos_a]
    ])

    return rot


def solar_ecef(mjd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute solar ECEF coordinates (same approximation as PyTMD)

    Optimized implementation with:
    - Pre-computed constants
    - Horner's method for polynomials
    - Trigonometric identities (cos(2M) = 2cos²M - 1)
    - Inline GHA calculation
    - Inline Z-rotation

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day

    Returns
    -------
    X, Y, Z : np.ndarray
        Solar ECEF coordinates (metres)
    """
    # T (Julian centuries based on TT) - inline calculation
    T = (mjd + _TT_OFFSET - _JD_J2000) / _JULIAN_CENTURY

    # Solar longitude of perihelion (radians)
    Ps = (282.94 + 1.7192 * T) * _DEG_TO_RAD

    # Solar mean anomaly (Horner's method inline)
    M = (357.5256 + T * (35999.049 + T * (-1.559e-4 - 4.8e-7 * T))) * _DEG_TO_RAD

    # Pre-compute trig values
    cos_M = np.cos(M)
    sin_M = np.sin(M)
    # Use trigonometric identity: cos(2M) = 2cos²(M) - 1, sin(2M) = 2sin(M)cos(M)
    cos_2M = 2.0 * cos_M * cos_M - 1.0
    sin_2M = 2.0 * sin_M * cos_M

    # Solar distance (metres)
    r_sun = 1e9 * (149.619 - 2.499 * cos_M - 0.021 * cos_2M)

    # Ecliptic longitude (radians)
    lambda_sun = Ps + M + _ARCSEC_TO_RAD * (6892.0 * sin_M + 72.0 * sin_2M)

    # Equatorial rectangular coordinates (use pre-computed epsilon constants)
    cos_lambda = np.cos(lambda_sun)
    sin_lambda = np.sin(lambda_sun)
    x = r_sun * cos_lambda
    y_temp = r_sun * sin_lambda
    y = y_temp * _COS_EPSILON
    z = y_temp * _SIN_EPSILON

    # Greenwich Hour Angle (inline calculation)
    # GMST calculation
    ut1_jc = (mjd + _JD_MJD - _JD_J2000) / _JULIAN_CENTURY
    gmst_sec = 24110.54841 + ut1_jc * (8640184.812866 + ut1_jc * (9.3104e-2 - 6.2e-6 * ut1_jc))
    gmst = np.mod(gmst_sec / _DAY_SECONDS, 1.0)
    gha_rad = np.mod(gmst * 360.0 + 360.0 * T * _JULIAN_CENTURY + 180.0, 360.0) * _DEG_TO_RAD

    # Inline Z-rotation (avoid creating 3x3xN array)
    cos_gha = np.cos(gha_rad)
    sin_gha = np.sin(gha_rad)

    # Z-rotation: [cos, sin, 0; -sin, cos, 0; 0, 0, 1] * [x, y, z]
    X = cos_gha * x + sin_gha * y
    Y = -sin_gha * x + cos_gha * y
    Z = z  # z unchanged by Z-rotation

    return X, Y, Z


def solar_distance(mjd: np.ndarray) -> np.ndarray:
    """
    Compute solar distance

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day

    Returns
    -------
    np.ndarray
        Solar distance (metres)
    """
    X, Y, Z = solar_ecef(mjd)
    return np.sqrt(X**2 + Y**2 + Z**2)


# ============================================================
# Lunar position calculation
# ============================================================

def lunar_ecef(mjd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute lunar ECEF coordinates (same approximation as PyTMD)

    Optimized implementation with:
    - Pre-computed constants
    - Horner's method for polynomials
    - Inline GHA calculation
    - Inline rotations

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day

    Returns
    -------
    X, Y, Z : np.ndarray
        Lunar ECEF coordinates (metres)
    """
    # T (Julian centuries based on TT) - inline calculation
    T = (mjd + _TT_OFFSET - _JD_J2000) / _JULIAN_CENTURY

    # Lunar mean longitude (Horner's method)
    s = (218.3164477 + T * (481267.88123421 + T * (-1.5786e-3 +
         T * (1.855835e-6 - 1.53388e-8 * T)))) * _DEG_TO_RAD

    # Lunar mean elongation (Horner's method)
    D = (297.8501921 + T * (445267.1114034 + T * (-1.8819e-3 +
         T * (1.83195e-6 - 8.8445e-9 * T)))) * _DEG_TO_RAD

    # Lunar ascending node longitude (Horner's method)
    N = (125.04452 + T * (-1934.136261 + T * (2.0708e-3 + 2.22222e-6 * T))) * _DEG_TO_RAD
    F = s - N

    # Solar mean anomaly (radians)
    M = (357.5256 + 35999.049 * T) * _DEG_TO_RAD

    # Lunar mean anomaly (radians)
    l = (134.96292 + 477198.86753 * T) * _DEG_TO_RAD

    # Pre-compute common angle combinations
    D2 = 2.0 * D
    l2 = 2.0 * l
    F2 = 2.0 * F

    # Pre-compute trig values for commonly used angles
    cos_l = np.cos(l)
    sin_l = np.sin(l)
    cos_D2 = np.cos(D2)
    sin_D2 = np.sin(D2)
    sin_M = np.sin(M)
    sin_F2 = np.sin(F2)

    # Lunar distance (metres) - optimized series expansion
    r_moon = 1e3 * (
        385000.0
        - 20905.0 * cos_l
        - 3699.0 * np.cos(D2 - l)
        - 2956.0 * cos_D2
        - 570.0 * np.cos(l2)
        + 246.0 * np.cos(l2 - D2)
        - 205.0 * np.cos(M - D2)
        - 171.0 * np.cos(l + D2)
        - 152.0 * np.cos(l + M - D2)
    )

    # Lunar ecliptic longitude (radians) - optimized series expansion
    lambda_moon = s + _ARCSEC_TO_RAD * (
        22640.0 * sin_l
        + 769.0 * np.sin(l2)
        - 4586.0 * np.sin(l - D2)
        + 2370.0 * sin_D2
        - 668.0 * sin_M
        - 412.0 * sin_F2
        - 212.0 * np.sin(l2 - D2)
        - 206.0 * np.sin(l + M - D2)
        + 192.0 * np.sin(l + D2)
        - 165.0 * np.sin(M - D2)
        - 148.0 * np.sin(l - M)
        - 125.0 * np.sin(D)
        - 110.0 * np.sin(l + M)
        - 55.0 * np.sin(F2 - D2)
    )

    # Lunar ecliptic latitude (radians)
    q = _ARCSEC_TO_RAD * (412.0 * sin_F2 + 541.0 * sin_M)
    F_minus_D2 = F - D2
    beta_moon = _ARCSEC_TO_RAD * (
        18520.0 * np.sin(F + lambda_moon - s + q)
        - 526.0 * np.sin(F_minus_D2)
        + 44.0 * np.sin(l + F_minus_D2)
        - 31.0 * np.sin(-l + F_minus_D2)
        - 25.0 * np.sin(-l2 + F)
        - 23.0 * np.sin(M + F_minus_D2)
        + 21.0 * np.sin(-l + F)
        + 11.0 * np.sin(-M + F_minus_D2)
    )

    # Convert from ecliptic to rectangular coordinates
    cos_lambda_moon = np.cos(lambda_moon)
    sin_lambda_moon = np.sin(lambda_moon)
    cos_beta = np.cos(beta_moon)
    sin_beta = np.sin(beta_moon)

    x = r_moon * cos_lambda_moon * cos_beta
    y = r_moon * sin_lambda_moon * cos_beta
    z = r_moon * sin_beta

    # Rotation about X-axis (ecliptic to equatorial) - inline with pre-computed constants
    # X-rotation matrix for -epsilon: [1,0,0; 0,cos,-sin; 0,sin,cos]
    u = x
    v = _COS_NEG_EPSILON * y + _SIN_NEG_EPSILON * z
    w = -_SIN_NEG_EPSILON * y + _COS_NEG_EPSILON * z

    # Greenwich Hour Angle (inline calculation)
    ut1_jc = (mjd + _JD_MJD - _JD_J2000) / _JULIAN_CENTURY
    gmst_sec = 24110.54841 + ut1_jc * (8640184.812866 + ut1_jc * (9.3104e-2 - 6.2e-6 * ut1_jc))
    gmst = np.mod(gmst_sec / _DAY_SECONDS, 1.0)
    gha_rad = np.mod(gmst * 360.0 + 360.0 * T * _JULIAN_CENTURY + 180.0, 360.0) * _DEG_TO_RAD

    # Inline Z-rotation
    cos_gha = np.cos(gha_rad)
    sin_gha = np.sin(gha_rad)

    # Z-rotation: [cos, sin, 0; -sin, cos, 0; 0, 0, 1] * [u, v, w]
    X = cos_gha * u + sin_gha * v
    Y = -sin_gha * u + cos_gha * v
    Z = w

    return X, Y, Z


def lunar_distance(mjd: np.ndarray) -> np.ndarray:
    """
    Compute lunar distance

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day

    Returns
    -------
    np.ndarray
        Lunar distance (metres)
    """
    X, Y, Z = lunar_ecef(mjd)
    return np.sqrt(X**2 + Y**2 + Z**2)


# ============================================================
# Angle normalization and astronomical arguments
# ============================================================

def normalize_angle(theta: np.ndarray, circle: float = 360.0) -> np.ndarray:
    """
    Normalize an angle to a single rotation

    Parameters
    ----------
    theta : float or np.ndarray
        Angle to normalize
    circle : float, default 360.0
        Circle of the angle (360.0 for degrees, 2*pi for radians)

    Returns
    -------
    np.ndarray
        Normalized angle in range [0, circle)
    """
    return np.mod(theta, circle)


def doodson_arguments(
    mjd: np.ndarray,
    deltat: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the six Doodson astronomical arguments

    Computes τ (tau), S, H, P, N', and Ps for tidal harmonic analysis.
    Follows IERS conventions.

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day
    deltat : float, default 0.0
        Time correction for converting to Ephemeris Time (days)

    Returns
    -------
    TAU : np.ndarray
        Mean lunar time (radians)
    S : np.ndarray
        Mean longitude of the moon (radians)
    H : np.ndarray
        Mean longitude of the sun (radians)
    P : np.ndarray
        Mean longitude of lunar perigee (radians)
    Np : np.ndarray
        Negative mean longitude of ascending lunar node (radians)
    Ps : np.ndarray
        Mean longitude of solar perigee (radians)
    """
    # Convert from MJD to centuries relative to J2000.0
    T = (mjd + deltat - _MJD_J2000) / (_JULIAN_CENTURY / 365.25)  # Julian centuries
    T = (mjd + deltat - _MJD_J2000) / 36525.0  # Correct: days to centuries

    # Hour of the day
    hour = np.mod(mjd, 1) * 24.0

    # Mean longitude of moon (degrees) - Horner's method
    S = 218.3164477 + T * (481267.88123421 + T * (-1.5786e-3 +
        T * (1.855835e-6 - 1.53388e-8 * T)))

    # Mean solar longitude for TAU calculation (degrees) - Horner's method
    LAMBDA = 280.4606184 + T * (36000.7700536 + T * (3.8793e-4 - 2.58e-8 * T))

    # Mean lunar time (degrees)
    TAU = hour * 15.0 - S + LAMBDA

    # Correction for mean lunar longitude (degrees) - Horner's method
    PR = T * (1.396971278 + T * (3.08889e-4 + T * (2.1e-8 + 7.0e-9 * T)))
    S = S + PR

    # Mean longitude of sun (degrees) - Horner's method
    H = 280.46645 + T * (36000.7697489 + T * (3.0322222e-4 +
        T * (2.0e-8 - 6.54e-9 * T)))

    # Mean longitude of lunar perigee (degrees) - Horner's method
    P = 83.3532465 + T * (4069.0137287 + T * (-1.032172222e-2 +
        T * (-1.24991e-5 + 5.263e-8 * T)))

    # Negative mean longitude of ascending lunar node (degrees) - Horner's method
    Np = 234.95544499 + T * (1934.13626197 + T * (-2.07561111e-3 +
        T * (-2.13944e-6 + 1.65e-8 * T)))

    # Mean longitude of solar perigee (degrees) - Horner's method
    Ps = 282.93734098 + T * (1.71945766667 + T * (4.5688889e-4 +
        T * (-1.778e-8 - 3.34e-9 * T)))

    # Normalize to [0, 360) and convert to radians
    TAU = np.radians(normalize_angle(TAU))
    S = np.radians(normalize_angle(S))
    H = np.radians(normalize_angle(H))
    P = np.radians(normalize_angle(P))
    Np = np.radians(normalize_angle(Np))
    Ps = np.radians(normalize_angle(Ps))

    return TAU, S, H, P, Np, Ps


def delaunay_arguments(
    mjd: np.ndarray,
    deltat: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the five Delaunay astronomical arguments

    Computes l, l', F, D, and N (Omega) for nutation calculations.
    Follows IERS conventions (Capitaine et al., 2003).

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day
    deltat : float, default 0.0
        Time correction for converting to Ephemeris Time (days)

    Returns
    -------
    l : np.ndarray
        Mean anomaly of the moon (radians)
    lp : np.ndarray
        Mean anomaly of the sun (radians)
    F : np.ndarray
        Mean argument of the moon (radians)
    D : np.ndarray
        Mean elongation of the moon from the sun (radians)
    N : np.ndarray
        Mean longitude of ascending lunar node (radians)
    """
    # Convert from MJD to centuries relative to J2000.0
    T = (mjd + deltat - _MJD_J2000) / 36525.0

    # Circle in arcseconds (360 degrees = 1296000 arcseconds)
    circle = 1296000.0

    # Mean anomaly of moon (arcseconds) - Horner's method
    l = 485868.249036 + T * (1717915923.2178 + T * (31.8792 +
        T * (0.051635 - 2.447e-4 * T)))

    # Mean anomaly of sun (arcseconds) - Horner's method
    lp = 1287104.79305 + T * (129596581.0481 + T * (-0.5532 +
        T * (1.36e-4 - 1.149e-5 * T)))

    # Mean argument of moon (arcseconds) - Horner's method
    F = 335779.526232 + T * (1739527262.8478 + T * (-12.7512 +
        T * (-1.037e-3 + 4.17e-6 * T)))

    # Mean elongation of moon from sun (arcseconds) - Horner's method
    D = 1072260.70369 + T * (1602961601.2090 + T * (-6.3706 +
        T * (6.593e-3 - 3.169e-5 * T)))

    # Mean longitude of ascending lunar node (arcseconds) - Horner's method
    N = 450160.398036 + T * (-6962890.5431 + T * (7.4722 +
        T * (7.702e-3 - 5.939e-5 * T)))

    # Normalize to [0, circle) and convert to radians
    l = normalize_angle(l, circle=circle) * _ARCSEC_TO_RAD
    lp = normalize_angle(lp, circle=circle) * _ARCSEC_TO_RAD
    F = normalize_angle(F, circle=circle) * _ARCSEC_TO_RAD
    D = normalize_angle(D, circle=circle) * _ARCSEC_TO_RAD
    N = normalize_angle(N, circle=circle) * _ARCSEC_TO_RAD

    return l, lp, F, D, N


def schureman_arguments(
    P: np.ndarray,
    N: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute additional Schureman astronomical arguments for FES models

    Parameters
    ----------
    P : np.ndarray
        Mean longitude of lunar perigee (radians)
    N : np.ndarray
        Mean longitude of ascending lunar node (radians)

    Returns
    -------
    I : np.ndarray
        Obliquity of lunar orbit with respect to Earth's equator (radians)
    xi : np.ndarray
        Longitude in moon's orbit of lunar intersection (radians)
    nu : np.ndarray
        Right ascension of lunar intersection (radians)
    Qa : np.ndarray
        Factor in amplitude for M1 constituent
    Qu : np.ndarray
        Term in argument for M1 constituent (radians)
    Ra : np.ndarray
        Factor in amplitude for L2 constituent
    Ru : np.ndarray
        Term in argument for L2 constituent (radians)
    nu_p : np.ndarray
        Term in argument for K1 constituent (radians)
    nu_s : np.ndarray
        Term in argument for K2 constituent (radians)

    References
    ----------
    Schureman, P., "Manual of Harmonic Analysis and Prediction of Tides"
        US Coast and Geodetic Survey, Special Publication, 98, (1958).
    """
    # Inclination of moon's orbit to Earth's equator (Schureman p.156)
    I = np.arccos(0.913694997 - 0.035692561 * np.cos(N))

    # Longitude in moon's orbit of lunar intersection
    at1 = np.arctan(1.01883 * np.tan(N / 2.0))
    at2 = np.arctan(0.64412 * np.tan(N / 2.0))
    xi = -at1 - at2 + N
    xi = np.arctan2(np.sin(xi), np.cos(xi))

    # Right ascension of lunar intersection
    nu = at1 - at2

    # Mean longitude of lunar perigee from lunar intersection (p.41)
    p = P - xi

    # Equation 202 (p.42)
    Q = np.arctan((5.0 * np.cos(I) - 1.0) * np.tan(p) / (7.0 * np.cos(I) + 1.0))

    # Equation 197 (p.41) - M1 amplitude factor
    Qa = np.power(2.31 + 1.435 * np.cos(2.0 * p), -0.5)

    # Equation 204 (p.42) - M1 argument term
    Qu = p - Q

    # Equation 214 (p.44) - L2 argument term
    P_R = np.sin(2.0 * p)
    Q_R = np.power(np.tan(I / 2.0), -2.0) / 6.0 - np.cos(2.0 * p)
    Ru = np.arctan(P_R / Q_R)

    # Equation 213 (p.44) - L2 amplitude factor (normally used as inverse)
    term1 = 12.0 * np.power(np.tan(I / 2.0), 2.0) * np.cos(2.0 * p)
    term2 = 36.0 * np.power(np.tan(I / 2.0), 4.0)
    Ra = np.power(1.0 - term1 + term2, -0.5)

    # Equation 224 (p.45) - K1 argument term
    P_prime = np.sin(2.0 * I) * np.sin(nu)
    Q_prime = np.sin(2.0 * I) * np.cos(nu) + 0.3347
    nu_p = np.arctan(P_prime / Q_prime)

    # Equation 232 (p.46) - K2 argument term
    P_sec = (np.sin(I) ** 2) * np.sin(2.0 * nu)
    Q_sec = (np.sin(I) ** 2) * np.cos(2.0 * nu) + 0.0727
    nu_s = 0.5 * np.arctan(P_sec / Q_sec)

    return I, xi, nu, Qa, Qu, Ra, Ru, nu_p, nu_s
