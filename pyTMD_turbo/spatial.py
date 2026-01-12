"""
pyTMD_turbo.spatial - Spatial coordinate transformations

Provides coordinate transformation functions compatible with pyTMD.

Functions:
    to_cartesian: Convert geographic to cartesian (ECEF) coordinates
    to_geodetic: Convert cartesian (ECEF) to geographic coordinates
    to_sphere: Convert geographic to spherical coordinates
    scale_factors: Calculate scale factors for coordinate systems
    datum: Ellipsoid parameters

References:
    B. Hofmann-Wellenhof and H. Moritz, "Physical Geodesy", 2005.
    J. Meeus, "Astronomical Algorithms", 1998.

Copyright (c) 2024-2026 tkykszk
Derived from pyTMD by Tyler Sutterley (MIT License)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    'datum',
    'scale_factors',
    'to_cartesian',
    'to_geodetic',
    'to_sphere',
]


@dataclass
class datum:
    """
    Ellipsoid parameters for geodetic calculations

    Parameters
    ----------
    ellipsoid : str, optional
        Ellipsoid name (default: 'WGS84')
        Supported: 'WGS84', 'GRS80', 'WGS72', 'GRS67', 'TOPEX'

    Attributes
    ----------
    a : float
        Semi-major axis (meters)
    f : float
        Flattening factor
    b : float
        Semi-minor axis (meters)
    e2 : float
        First eccentricity squared
    ep2 : float
        Second eccentricity squared

    Examples
    --------
    >>> d = datum('WGS84')
    >>> d.a
    6378137.0
    >>> d.f
    0.0033528106647474805
    """

    # Ellipsoid parameters (a: semi-major axis, f: flattening)
    _ellipsoids = {
        'WGS84': (6378137.0, 1.0 / 298.257223563),
        'GRS80': (6378137.0, 1.0 / 298.257222101),
        'WGS72': (6378135.0, 1.0 / 298.26),
        'GRS67': (6378160.0, 1.0 / 298.247167427),
        'TOPEX': (6378136.3, 1.0 / 298.257),
        'EGM2008': (6378136.3, 1.0 / 298.257222101),
    }

    a: float = 6378137.0  # WGS84 default
    f: float = 1.0 / 298.257223563
    name: str = 'WGS84'

    def __init__(self, ellipsoid: str = 'WGS84'):
        """Initialize datum with ellipsoid parameters"""
        ellipsoid = ellipsoid.upper()
        if ellipsoid not in self._ellipsoids:
            raise ValueError(
                f"Unknown ellipsoid: {ellipsoid}. "
                f"Supported: {list(self._ellipsoids.keys())}"
            )
        self.a, self.f = self._ellipsoids[ellipsoid]
        self.name = ellipsoid

    @property
    def b(self) -> float:
        """Semi-minor axis (meters)"""
        return self.a * (1.0 - self.f)

    @property
    def e2(self) -> float:
        """First eccentricity squared"""
        return self.f * (2.0 - self.f)

    @property
    def ep2(self) -> float:
        """Second eccentricity squared"""
        return self.e2 / (1.0 - self.e2)

    @property
    def n(self) -> float:
        """Third flattening"""
        return self.f / (2.0 - self.f)


def to_cartesian(
    lon: np.ndarray,
    lat: np.ndarray,
    h: np.ndarray | None = None,
    ellipsoid: str = 'WGS84',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert geographic coordinates to cartesian (ECEF)

    Converts geodetic coordinates (longitude, latitude, height) to
    Earth-Centered Earth-Fixed (ECEF) cartesian coordinates.

    Parameters
    ----------
    lon : np.ndarray
        Longitude (degrees)
    lat : np.ndarray
        Latitude (degrees)
    h : np.ndarray, optional
        Height above ellipsoid (meters), default is 0
    ellipsoid : str, default 'WGS84'
        Reference ellipsoid name

    Returns
    -------
    x : np.ndarray
        X coordinate (meters)
    y : np.ndarray
        Y coordinate (meters)
    z : np.ndarray
        Z coordinate (meters)

    Examples
    --------
    >>> x, y, z = to_cartesian(140.0, 35.0, 0.0)
    >>> print(f"X: {x:.1f}, Y: {y:.1f}, Z: {z:.1f}")
    X: -3906851.1, Y: 3381995.5, Z: 3638907.0
    """
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))

    if h is None:
        h = np.zeros_like(lon)
    else:
        h = np.atleast_1d(np.asarray(h, dtype=np.float64))

    # Get ellipsoid parameters
    d = datum(ellipsoid)

    # Convert to radians
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    # Trigonometric functions
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)

    # Radius of curvature in the prime vertical
    N = d.a / np.sqrt(1.0 - d.e2 * sin_lat**2)

    # Cartesian coordinates
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - d.e2) + h) * sin_lat

    return x, y, z


def to_geodetic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ellipsoid: str = 'WGS84',
    method: str = 'bowring',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert cartesian (ECEF) coordinates to geographic

    Converts Earth-Centered Earth-Fixed (ECEF) cartesian coordinates
    to geodetic coordinates (longitude, latitude, height).

    Parameters
    ----------
    x : np.ndarray
        X coordinate (meters)
    y : np.ndarray
        Y coordinate (meters)
    z : np.ndarray
        Z coordinate (meters)
    ellipsoid : str, default 'WGS84'
        Reference ellipsoid name
    method : str, default 'bowring'
        Conversion method ('bowring', 'iterative')

    Returns
    -------
    lon : np.ndarray
        Longitude (degrees)
    lat : np.ndarray
        Latitude (degrees)
    h : np.ndarray
        Height above ellipsoid (meters)

    Examples
    --------
    >>> lon, lat, h = to_geodetic(-3906851.0, 3381995.5, 3638907.0)
    >>> print(f"Lon: {lon:.4f}, Lat: {lat:.4f}, H: {h:.1f}")
    Lon: 139.0000, Lat: 35.0000, H: 0.0
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))

    # Get ellipsoid parameters
    d = datum(ellipsoid)

    # Longitude
    lon = np.degrees(np.arctan2(y, x))

    # Distance from Z-axis
    p = np.sqrt(x**2 + y**2)

    if method == 'bowring':
        # Bowring's method (iterative)
        # Initial estimate using spherical approximation
        lat = np.arctan2(z, p * (1.0 - d.e2))

        for _ in range(10):
            sin_lat = np.sin(lat)
            N = d.a / np.sqrt(1.0 - d.e2 * sin_lat**2)
            lat_new = np.arctan2(z + d.e2 * N * sin_lat, p)

            if np.max(np.abs(lat_new - lat)) < 1e-12:
                break
            lat = lat_new

        lat = np.degrees(lat)

        # Height
        sin_lat = np.sin(np.radians(lat))
        cos_lat = np.cos(np.radians(lat))
        N = d.a / np.sqrt(1.0 - d.e2 * sin_lat**2)
        h = p / cos_lat - N

    else:
        # Simple iterative method
        lat = np.arctan2(z, p)

        for _ in range(10):
            sin_lat = np.sin(lat)
            N = d.a / np.sqrt(1.0 - d.e2 * sin_lat**2)
            lat_new = np.arctan2(z + d.e2 * N * sin_lat, p)

            if np.max(np.abs(lat_new - lat)) < 1e-12:
                break
            lat = lat_new

        lat = np.degrees(lat)

        sin_lat = np.sin(np.radians(lat))
        cos_lat = np.cos(np.radians(lat))
        N = d.a / np.sqrt(1.0 - d.e2 * sin_lat**2)
        h = p / cos_lat - N

    return lon, lat, h


def to_sphere(
    lon: np.ndarray,
    lat: np.ndarray,
    r: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert geographic coordinates to spherical coordinates

    Parameters
    ----------
    lon : np.ndarray
        Longitude (degrees)
    lat : np.ndarray
        Latitude (degrees)
    r : np.ndarray, optional
        Radius (default is 1.0)

    Returns
    -------
    x : np.ndarray
        X coordinate (unit sphere if r=1)
    y : np.ndarray
        Y coordinate
    z : np.ndarray
        Z coordinate
    """
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))

    if r is None:
        r = np.ones_like(lon)
    else:
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))

    # Convert to radians
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    # Spherical to Cartesian
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return x, y, z


def scale_factors(
    lat: np.ndarray,
    ellipsoid: str = 'WGS84',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate scale factors for ellipsoid

    Computes the meridional and transverse scale factors
    at given latitudes.

    Parameters
    ----------
    lat : np.ndarray
        Latitude (degrees)
    ellipsoid : str, default 'WGS84'
        Reference ellipsoid name

    Returns
    -------
    h_lat : np.ndarray
        Meridional scale factor (meters per degree latitude)
    h_lon : np.ndarray
        Transverse scale factor (meters per degree longitude)
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))

    # Get ellipsoid parameters
    d = datum(ellipsoid)

    # Convert to radians
    lat_rad = np.radians(lat)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    # Radius of curvature in meridian
    M = d.a * (1.0 - d.e2) / (1.0 - d.e2 * sin_lat**2)**1.5

    # Radius of curvature in prime vertical
    N = d.a / np.sqrt(1.0 - d.e2 * sin_lat**2)

    # Scale factors (meters per radian)
    h_lat = M
    h_lon = N * cos_lat

    # Convert to meters per degree
    h_lat *= np.pi / 180.0
    h_lon *= np.pi / 180.0

    return h_lat, h_lon


def convert_ellipsoid(
    lon: np.ndarray,
    lat: np.ndarray,
    h: np.ndarray,
    source_ellipsoid: str = 'WGS84',
    target_ellipsoid: str = 'GRS80',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert coordinates between ellipsoids

    Parameters
    ----------
    lon : np.ndarray
        Longitude (degrees)
    lat : np.ndarray
        Latitude (degrees)
    h : np.ndarray
        Height above source ellipsoid (meters)
    source_ellipsoid : str, default 'WGS84'
        Source ellipsoid name
    target_ellipsoid : str, default 'GRS80'
        Target ellipsoid name

    Returns
    -------
    lon : np.ndarray
        Longitude (unchanged)
    lat : np.ndarray
        Latitude in target ellipsoid (degrees)
    h : np.ndarray
        Height above target ellipsoid (meters)
    """
    # Convert to cartesian using source ellipsoid
    x, y, z = to_cartesian(lon, lat, h, ellipsoid=source_ellipsoid)

    # Convert back to geodetic using target ellipsoid
    lon_out, lat_out, h_out = to_geodetic(x, y, z, ellipsoid=target_ellipsoid)

    return lon_out, lat_out, h_out
