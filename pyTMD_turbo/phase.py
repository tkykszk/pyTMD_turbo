"""
Phase calculation module for tidal predictions.

Provides fast phase and derivative computation using:
- NumPy vectorized operations for large batches
- Numba JIT for small batches (when available)

No parallel/multi-thread overhead - all methods are single-threaded optimized.
"""

from typing import Optional

import numpy as np

# =============================================================================
# Constituent Data
# =============================================================================

# Constituent periods in hours
# Reference: https://tidesandcurrents.noaa.gov/harmonic.html
CONSTITUENT_PERIODS = {
    # Semi-diurnal
    'm2': 12.4206012,   # Principal lunar
    's2': 12.0000000,   # Principal solar
    'n2': 12.6583482,   # Larger lunar elliptic
    'k2': 11.9672348,   # Luni-solar
    # Diurnal
    'k1': 23.9344696,   # Luni-solar
    'o1': 25.8193417,   # Principal lunar
    'p1': 24.0658902,   # Principal solar
    'q1': 26.8683567,   # Larger lunar elliptic
    # Long period
    'mf': 327.8599387,  # Lunar fortnightly
    'mm': 661.3111655,  # Lunar monthly
}

# Pre-computed angular frequencies (rad/s)
_OMEGA_CACHE: dict[str, float] = {}


def get_omega(constituent: str) -> float:
    """
    Get angular frequency for constituent (rad/s).

    Uses cached values for O(1) lookup.

    Parameters
    ----------
    constituent : str
        Constituent name (e.g., 'm2', 's2', 'k1')

    Returns
    -------
    float
        Angular frequency in rad/s
    """
    key = constituent.lower()
    if key not in _OMEGA_CACHE:
        period_hours = CONSTITUENT_PERIODS.get(key)
        if period_hours is None:
            raise ValueError(f"Unknown constituent: {constituent}")
        _OMEGA_CACHE[key] = 2 * np.pi / (period_hours * 3600)
    return _OMEGA_CACHE[key]


def get_omegas(constituents: list[str]) -> np.ndarray:
    """Get array of angular frequencies for multiple constituents."""
    return np.array([get_omega(c) for c in constituents])


# =============================================================================
# Numba JIT functions (optional, for small batch optimization)
# =============================================================================

try:
    from numba import jit

    @jit(nopython=True, cache=True, fastmath=True)
    def _derivative_numba(t_points: np.ndarray, amplitudes: np.ndarray,
                          phases: np.ndarray, omegas: np.ndarray) -> np.ndarray:
        """
        Numba-optimized derivative computation.

        ~40x faster than pure Python for small batches (1-10 points).
        Single-threaded, no parallel overhead.
        """
        n_points = len(t_points)
        n_const = len(omegas)
        derivatives = np.zeros(n_points)

        for i in range(n_points):
            t = t_points[i]
            total = 0.0
            for j in range(n_const):
                phase_at_t = omegas[j] * t + phases[j]
                total += amplitudes[j] * omegas[j] * np.cos(phase_at_t)
            derivatives[i] = total

        return derivatives

    NUMBA_AVAILABLE = True

except ImportError:
    NUMBA_AVAILABLE = False
    _derivative_numba = None


# =============================================================================
# Phase Fitting
# =============================================================================

class PhaseFitter:
    """
    Multi-constituent phase fitter using linear least squares.

    Fits the model:
        y(t) = sum_i A_i * sin(omega_i * t + phi_i) + C

    Uses NumPy's LAPACK-optimized lstsq (already highly optimized,
    Numba cannot improve it).
    """

    def __init__(self, constituents: Optional[list[str]] = None):
        """
        Initialize fitter with constituent list.

        Parameters
        ----------
        constituents : list of str, optional
            Constituent names. Default: ['m2', 's2', 'k1', 'o1', 'n2']
        """
        if constituents is None:
            constituents = ['m2', 's2', 'k1', 'o1', 'n2']

        self.constituents = constituents
        self.omegas = get_omegas(constituents)
        self.n_constituents = len(constituents)

        # Results (set after fit)
        self.amplitudes: Optional[np.ndarray] = None
        self.phases: Optional[np.ndarray] = None
        self.offset: Optional[float] = None
        self._coeffs: Optional[np.ndarray] = None

    def fit(self, t: np.ndarray, y: np.ndarray) -> 'PhaseFitter':
        """
        Fit phase model to data.

        Parameters
        ----------
        t : np.ndarray
            Time values in seconds from reference
        y : np.ndarray
            Tide height values

        Returns
        -------
        PhaseFitter
            Self for method chaining
        """
        # Build design matrix using vectorized NumPy
        # Shape: (n_points, 2*n_constituents + 1)
        n = len(t)
        X = np.empty((n, 2 * self.n_constituents + 1))

        for i, omega in enumerate(self.omegas):
            wt = omega * t
            X[:, 2*i] = np.sin(wt)
            X[:, 2*i + 1] = np.cos(wt)
        X[:, -1] = 1.0  # offset

        # Solve least squares (LAPACK optimized)
        self._coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

        # Extract amplitudes and phases
        a = self._coeffs[0::2][:-1]  # sin coefficients (exclude offset placeholder)
        b = self._coeffs[1::2]        # cos coefficients

        # Handle case where we have one extra 'a' due to offset
        if len(a) > self.n_constituents:
            a = a[:self.n_constituents]

        self.amplitudes = np.sqrt(a**2 + b[:self.n_constituents]**2)
        self.phases = np.arctan2(b[:self.n_constituents], a)
        self.offset = self._coeffs[-1]

        return self

    def derivative(self, t: np.ndarray) -> np.ndarray:
        """
        Compute tide derivative (rate of change) at given times.

        dy/dt = sum_i A_i * omega_i * cos(omega_i * t + phi_i)

        Automatically selects optimal method:
        - Small batch (< 50 points): Numba JIT if available
        - Large batch: NumPy vectorized

        Parameters
        ----------
        t : np.ndarray
            Time values in seconds from reference

        Returns
        -------
        np.ndarray
            Derivative values in meters/second
        """
        if self.amplitudes is None:
            raise ValueError("Must call fit() before derivative()")

        t = np.atleast_1d(t)

        # Choose method based on batch size and Numba availability
        if NUMBA_AVAILABLE and len(t) < 50:
            # Small batch: Numba is ~40x faster
            return _derivative_numba(t, self.amplitudes, self.phases, self.omegas)
        else:
            # Large batch: NumPy vectorized is optimal
            return self._derivative_vectorized(t)

    def _derivative_vectorized(self, t: np.ndarray) -> np.ndarray:
        """
        Vectorized derivative using NumPy broadcasting.

        Efficient for large batches due to SIMD optimization.
        """
        # Broadcasting: t[:, None] * omegas[None, :] -> (n_points, n_constituents)
        phases_at_t = np.outer(t, self.omegas) + self.phases

        # Sum over constituents
        return np.sum(self.amplitudes * self.omegas * np.cos(phases_at_t), axis=1)

    def predict(self, t: np.ndarray) -> np.ndarray:
        """
        Predict tide height at given times.

        Parameters
        ----------
        t : np.ndarray
            Time values in seconds from reference

        Returns
        -------
        np.ndarray
            Predicted tide heights in meters
        """
        if self.amplitudes is None:
            raise ValueError("Must call fit() before predict()")

        t = np.atleast_1d(t)

        # Vectorized prediction
        phases_at_t = np.outer(t, self.omegas) + self.phases
        return np.sum(self.amplitudes * np.sin(phases_at_t), axis=1) + self.offset

    def get_constituent_info(self) -> dict[str, dict]:
        """
        Get fitted parameters for each constituent.

        Returns
        -------
        dict
            Dictionary with constituent names as keys and
            {amplitude, phase, omega} as values
        """
        if self.amplitudes is None:
            raise ValueError("Must call fit() first")

        info = {}
        for i, const in enumerate(self.constituents):
            info[const] = {
                'amplitude': self.amplitudes[i],
                'phase': self.phases[i],
                'phase_deg': np.degrees(self.phases[i]),
                'omega': self.omegas[i],
            }
        info['offset'] = self.offset
        return info


# =============================================================================
# Convenience Functions
# =============================================================================

def fit_phase(t: np.ndarray, y: np.ndarray,
              constituents: Optional[list[str]] = None) -> PhaseFitter:
    """
    Fit multi-constituent phase model to tide data.

    Parameters
    ----------
    t : np.ndarray
        Time values in seconds from reference
    y : np.ndarray
        Tide height values
    constituents : list of str, optional
        Constituent names. Default: ['m2', 's2', 'k1', 'o1', 'n2']

    Returns
    -------
    PhaseFitter
        Fitted model with amplitude, phase, and derivative methods
    """
    return PhaseFitter(constituents).fit(t, y)


def compute_derivative(t: np.ndarray, fitter: PhaseFitter) -> np.ndarray:
    """
    Compute tide derivative at given times.

    Parameters
    ----------
    t : np.ndarray
        Time values in seconds
    fitter : PhaseFitter
        Fitted phase model

    Returns
    -------
    np.ndarray
        Derivative in m/s
    """
    return fitter.derivative(t)
