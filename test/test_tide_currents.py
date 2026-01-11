"""
Tests for pyTMD_turbo.compute.tide_currents

Tests tidal current calculation functionality.
"""

import numpy as np
import pytest

import pyTMD_turbo


class TestTideCurrentsBasic:
    """Basic tests for tide_currents function"""

    def test_import(self):
        """Test that tide_currents is importable"""
        assert hasattr(pyTMD_turbo, 'tide_currents')

    def test_function_signature(self):
        """Test function has correct signature"""
        import inspect
        sig = inspect.signature(pyTMD_turbo.tide_currents)
        params = list(sig.parameters.keys())
        assert 'x' in params
        assert 'y' in params
        assert 'times' in params
        assert 'model' in params

    def test_default_model(self):
        """Test default model parameter"""
        import inspect
        sig = inspect.signature(pyTMD_turbo.tide_currents)
        default = sig.parameters['model'].default
        # Default should be an OTIS-type model with current data
        assert default == 'TPXO9-atlas-v5'

    def test_function_returns_tuple(self):
        """Test that function returns a tuple (u, v)"""
        # This test requires model data, so just check function exists
        # and has correct return type annotation if any
        import inspect
        sig = inspect.signature(pyTMD_turbo.tide_currents)
        # Return annotation should be tuple
        assert sig.return_annotation == tuple or sig.return_annotation == inspect.Parameter.empty


class TestTideCurrentsWithModel:
    """Test tide_currents with actual model data (requires model files)"""

    @pytest.fixture
    def model_directory(self):
        """Get model directory from environment"""
        import os
        directory = os.environ.get('PYTMD_RESOURCE')
        if not directory:
            pytest.skip('PYTMD_RESOURCE environment variable not set')
        return directory

    def test_tpxo9_currents(self, model_directory):
        """Test tide_currents with TPXO9-atlas-v5 model"""
        import os

        # Check if TPXO9 data exists
        tpxo_dir = os.path.join(model_directory, 'TPXO9_atlas_v5')
        if not os.path.exists(tpxo_dir):
            pytest.skip('TPXO9-atlas-v5 data not available')

        x = np.array([140.0])
        y = np.array([35.0])
        times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')

        try:
            u, v = pyTMD_turbo.tide_currents(
                x, y, times,
                model='TPXO9-atlas-v5',
                directory=model_directory
            )

            # Check output shape
            assert len(u) == len(times)
            assert len(v) == len(times)

            # Currents should be finite
            assert np.all(np.isfinite(u))
            assert np.all(np.isfinite(v))

            # Transport should be within reasonable range (m²/s)
            # Typical values are < 100 m²/s
            assert np.nanmax(np.abs(u)) < 200
            assert np.nanmax(np.abs(v)) < 200

        except ValueError as e:
            if "does not have current" in str(e):
                pytest.skip(f"Model currents not available: {e}")
            raise

    def test_cats2008_currents(self, model_directory):
        """Test tide_currents with CATS2008 model"""
        import os

        # Check if CATS2008 data exists
        cats_dir = os.path.join(model_directory, 'CATS2008')
        if not os.path.exists(cats_dir):
            pytest.skip('CATS2008 data not available')

        # Antarctic location
        x = np.array([-70.0])
        y = np.array([-75.0])
        times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')

        try:
            u, v = pyTMD_turbo.tide_currents(
                x, y, times,
                model='CATS2008',
                directory=model_directory
            )

            assert len(u) == len(times)
            assert len(v) == len(times)

        except ValueError as e:
            if "does not have current" in str(e):
                pytest.skip(f"Model currents not available: {e}")
            raise

    def test_model_without_currents(self, model_directory):
        """Test that GOT models raise error for currents"""
        x = np.array([140.0])
        y = np.array([35.0])
        times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')

        # GOT models don't have current data
        with pytest.raises(ValueError, match="does not have"):
            pyTMD_turbo.tide_currents(
                x, y, times,
                model='GOT5.5',
                directory=model_directory
            )
