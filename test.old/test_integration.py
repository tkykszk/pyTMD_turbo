"""
OCEAN acceleration integration tests

Basic functionality tests without external dependencies.

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

import numpy as np
import pytest

from pyTMD_turbo.predict.harmonic import HarmonicConstants, predict_vectorized


def test_harmonic_constants_creation():
    """Test HarmonicConstants dataclass creation"""
    hc = HarmonicConstants(
        constituents=['m2', 's2', 'k1', 'o1'],
        hc_real=np.array([[0.5, 0.3, 0.1, 0.05]]),
        hc_imag=np.array([[0.1, 0.05, 0.02, 0.01]]),
        omega=np.array([1.4e-4, 1.45e-4, 7.3e-5, 6.8e-5]),
        phase_0=np.array([0.0, 0.0, 0.0, 0.0])
    )

    assert len(hc.constituents) == 4
    assert hc.hc_real.shape == (1, 4)


def test_predict_vectorized_basic():
    """Test basic vectorized prediction"""
    # Create simple harmonic constants
    hc = HarmonicConstants(
        constituents=['m2'],
        hc_real=np.array([[1.0]]),
        hc_imag=np.array([[0.0]]),
        omega=np.array([1.405189e-4]),  # M2 angular frequency (rad/s)
        phase_0=np.array([0.0])
    )

    # Create time array (one day)
    t_days = np.arange(24) / 24.0

    # Simple nodal corrections (no correction)
    pu = np.zeros((len(t_days), 1))
    pf = np.ones((len(t_days), 1))

    # Predict
    tide = predict_vectorized(hc, t_days, pu, pf)

    # Should have variation (tidal signal)
    assert tide.shape == (1, 24)
    assert tide.max() - tide.min() > 0  # Should have some variation


def test_predict_vectorized_multiple_points():
    """Test vectorized prediction with multiple points"""
    n_points = 5
    n_constituents = 3

    hc = HarmonicConstants(
        constituents=['m2', 's2', 'k1'],
        hc_real=np.random.rand(n_points, n_constituents),
        hc_imag=np.random.rand(n_points, n_constituents) * 0.1,
        omega=np.array([1.405189e-4, 1.454441e-4, 7.292117e-5]),
        phase_0=np.array([0.0, 0.0, 0.0])
    )

    t_days = np.arange(48) / 24.0
    n_times = len(t_days)

    pu = np.zeros((n_times, n_constituents))
    pf = np.ones((n_times, n_constituents))

    tide = predict_vectorized(hc, t_days, pu, pf)

    assert tide.shape == (n_points, n_times)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
