"""
Fast tidal potential calculation module

Removes xarray dependency from PyTMD and computes solid Earth tide
using NumPy only.
Uses formulae based on IERS Conventions (2010).

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

import numpy as np
from typing import Tuple

# Constants
_A_AXIS = 6378136.3  # Earth's semi-major axis (metres)
_MJD_TIDE = 48622.0  # MJD of 1992-01-01T00:00:00

# Love and Shida numbers (default values)
_H2 = 0.6078  # 2nd degree Love number (vertical displacement)
_L2 = 0.0847  # 2nd degree Shida number (horizontal displacement)
_H3 = 0.292   # 3rd degree Love number
_L3 = 0.015   # 3rd degree Shida number

# Mass ratios
_MASS_RATIO_SOLAR = 332946.0482  # Earth/Sun
_MASS_RATIO_LUNAR = 0.0123000371  # Moon/Earth


def geodetic_to_ecef(lat: np.ndarray, lon: np.ndarray, h: np.ndarray = None,
                     a: float = 6378137.0, f: float = 1/298.257223563) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert geodetic coordinates to ECEF coordinates

    Parameters
    ----------
    lat : np.ndarray
        Latitude (degrees)
    lon : np.ndarray
        Longitude (degrees)
    h : np.ndarray, optional
        Ellipsoidal height (metres), default is 0
    a : float
        Semi-major axis (metres)
    f : float
        Flattening

    Returns
    -------
    X, Y, Z : np.ndarray
        ECEF coordinates (metres)
    """
    if h is None:
        h = np.zeros_like(lat)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Square of eccentricity
    e2 = 2*f - f**2

    # Radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    X = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - e2) + h) * np.sin(lat_rad)

    return X, Y, Z


def solid_earth_tide(
    mjd: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sun_x: np.ndarray,
    sun_y: np.ndarray,
    sun_z: np.ndarray,
    moon_x: np.ndarray,
    moon_y: np.ndarray,
    moon_z: np.ndarray,
    h2: float = _H2,
    l2: float = _L2,
    h3: float = _H3,
    l3: float = _L3,
    tide_system: str = "tide_free",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute solid Earth tide (IERS Conventions 2010)

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day
    lat : np.ndarray
        Latitude (degrees)
    lon : np.ndarray
        Longitude (degrees)
    sun_x, sun_y, sun_z : np.ndarray
        Solar ECEF coordinates (metres)
    moon_x, moon_y, moon_z : np.ndarray
        Lunar ECEF coordinates (metres)
    h2, l2 : float
        2nd degree Love and Shida numbers
    h3, l3 : float
        3rd degree Love and Shida numbers
    tide_system : str
        Tide system ("tide_free" or "mean_tide")

    Returns
    -------
    dN, dE, dR : np.ndarray
        Displacement in north, east, and radial directions (metres)
    """
    # Observer's ECEF coordinates
    X, Y, Z = geodetic_to_ecef(lat, lon)

    # Various distances
    radius = np.sqrt(X**2 + Y**2 + Z**2)
    solar_radius = np.sqrt(sun_x**2 + sun_y**2 + sun_z**2)
    lunar_radius = np.sqrt(moon_x**2 + moon_y**2 + moon_z**2)

    # Scalar products (cos(zenith angle))
    solar_scalar = (X * sun_x + Y * sun_y + Z * sun_z) / (radius * solar_radius)
    lunar_scalar = (X * moon_x + Y * moon_y + Z * moon_z) / (radius * lunar_radius)

    # Latitude-dependent Love number correction (Mathews et al., 1997)
    cosphi = np.sqrt(X**2 + Y**2) / radius
    h2_corr = h2 - 0.0006 * (1.0 - 3.0/2.0 * cosphi**2)
    l2_corr = l2 + 0.0002 * (1.0 - 3.0/2.0 * cosphi**2)

    # P2 term (2nd degree Legendre polynomial related)
    P2_solar = 3.0 * (h2_corr/2.0 - l2_corr) * solar_scalar**2 - h2_corr/2.0
    P2_lunar = 3.0 * (h2_corr/2.0 - l2_corr) * lunar_scalar**2 - h2_corr/2.0

    # P3 term (3rd degree Legendre polynomial related)
    P3_solar = (5.0/2.0 * (h3 - 3.0*l3) * solar_scalar**3
                + 3.0/2.0 * (l3 - h3) * solar_scalar)
    P3_lunar = (5.0/2.0 * (h3 - 3.0*l3) * lunar_scalar**3
                + 3.0/2.0 * (l3 - h3) * lunar_scalar)

    # Solar/lunar direction terms
    X2_solar = 3.0 * l2_corr * solar_scalar
    X2_lunar = 3.0 * l2_corr * lunar_scalar
    X3_solar = 3.0 * l3/2.0 * (5.0 * solar_scalar**2 - 1.0)
    X3_lunar = 3.0 * l3/2.0 * (5.0 * lunar_scalar**2 - 1.0)

    # Coefficients (functions of mass ratio and distance)
    F2_solar = _MASS_RATIO_SOLAR * _A_AXIS * (_A_AXIS / solar_radius)**3
    F2_lunar = _MASS_RATIO_LUNAR * _A_AXIS * (_A_AXIS / lunar_radius)**3
    F3_solar = _MASS_RATIO_SOLAR * _A_AXIS * (_A_AXIS / solar_radius)**4
    F3_lunar = _MASS_RATIO_LUNAR * _A_AXIS * (_A_AXIS / lunar_radius)**4

    # Displacement in ECEF coordinates
    dX = np.zeros_like(mjd)
    dY = np.zeros_like(mjd)
    dZ = np.zeros_like(mjd)

    # Displacement due to Sun (2nd + 3rd degree)
    dX += F2_solar * (X2_solar * sun_x/solar_radius + P2_solar * X/radius)
    dY += F2_solar * (X2_solar * sun_y/solar_radius + P2_solar * Y/radius)
    dZ += F2_solar * (X2_solar * sun_z/solar_radius + P2_solar * Z/radius)

    dX += F3_solar * (X3_solar * sun_x/solar_radius + P3_solar * X/radius)
    dY += F3_solar * (X3_solar * sun_y/solar_radius + P3_solar * Y/radius)
    dZ += F3_solar * (X3_solar * sun_z/solar_radius + P3_solar * Z/radius)

    # Displacement due to Moon (2nd + 3rd degree)
    dX += F2_lunar * (X2_lunar * moon_x/lunar_radius + P2_lunar * X/radius)
    dY += F2_lunar * (X2_lunar * moon_y/lunar_radius + P2_lunar * Y/radius)
    dZ += F2_lunar * (X2_lunar * moon_z/lunar_radius + P2_lunar * Z/radius)

    dX += F3_lunar * (X3_lunar * moon_x/lunar_radius + P3_lunar * X/radius)
    dY += F3_lunar * (X3_lunar * moon_y/lunar_radius + P3_lunar * Y/radius)
    dZ += F3_lunar * (X3_lunar * moon_z/lunar_radius + P3_lunar * Z/radius)

    # Convert ECEF displacement to local coordinate system (N, E, R)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Radial (upward) component
    dR = cos_lat * cos_lon * dX + cos_lat * sin_lon * dY + sin_lat * dZ

    # Northward component
    dN = -sin_lat * cos_lon * dX - sin_lat * sin_lon * dY + cos_lat * dZ

    # Eastward component
    dE = -sin_lon * dX + cos_lon * dY

    return dN, dE, dR


def solid_earth_tide_radial(
    mjd: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sun_x: np.ndarray,
    sun_y: np.ndarray,
    sun_z: np.ndarray,
    moon_x: np.ndarray,
    moon_y: np.ndarray,
    moon_z: np.ndarray,
    h2: float = _H2,
    l2: float = _L2,
    h3: float = _H3,
    l3: float = _L3,
    tide_system: str = "tide_free",
) -> np.ndarray:
    """
    Compute radial component of solid Earth tide only (fast version)

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Day
    lat : np.ndarray
        Latitude (degrees)
    lon : np.ndarray
        Longitude (degrees)
    sun_x, sun_y, sun_z : np.ndarray
        Solar ECEF coordinates (metres)
    moon_x, moon_y, moon_z : np.ndarray
        Lunar ECEF coordinates (metres)
    h2, l2 : float
        2nd degree Love and Shida numbers
    h3, l3 : float
        3rd degree Love and Shida numbers
    tide_system : str
        Tide system ("tide_free" or "mean_tide")

    Returns
    -------
    dR : np.ndarray
        Radial displacement (metres)
    """
    _, _, dR = solid_earth_tide(
        mjd, lat, lon, sun_x, sun_y, sun_z, moon_x, moon_y, moon_z,
        h2, l2, h3, l3, tide_system
    )
    return dR
