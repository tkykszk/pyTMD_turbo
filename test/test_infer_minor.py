"""
Tests for pyTMD_turbo.predict.infer_minor

Tests minor constituent inference functionality.
"""

import numpy as np
import pytest

pytest.importorskip('xarray')
import xarray as xr

from pyTMD_turbo.predict import (
    infer_minor,
    infer_diurnal,
    infer_semi_diurnal,
    MINOR_CONSTITUENTS,
    DIURNAL_MINORS,
    SEMI_DIURNAL_MINORS,
)


class TestMinorConstituents:
    """Test minor constituent definitions"""

    def test_diurnal_minors(self):
        """Test diurnal minor constituents are defined"""
        assert '2q1' in DIURNAL_MINORS
        assert 'sigma1' in DIURNAL_MINORS
        assert 'j1' in DIURNAL_MINORS

    def test_semi_diurnal_minors(self):
        """Test semi-diurnal minor constituents are defined"""
        assert 'eps2' in SEMI_DIURNAL_MINORS
        assert 'lambda2' in SEMI_DIURNAL_MINORS
        assert 't2' in SEMI_DIURNAL_MINORS

    def test_all_minors(self):
        """Test all minor constituents"""
        assert len(MINOR_CONSTITUENTS) == len(DIURNAL_MINORS) + len(SEMI_DIURNAL_MINORS)


class TestInferMinor:
    """Test minor constituent inference"""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample tidal model dataset with major constituents"""
        # Create sample harmonic constants for major constituents
        # Using realistic amplitudes (in meters)
        np.random.seed(42)

        # Create point data (already interpolated)
        constituents = {
            # Diurnal majors
            'o1': 0.10 * np.exp(-1j * np.pi / 4),
            'k1': 0.12 * np.exp(-1j * np.pi / 3),
            'q1': 0.02 * np.exp(-1j * np.pi / 6),
            'p1': 0.04 * np.exp(-1j * np.pi / 5),
            # Semi-diurnal majors
            'm2': 0.50 * np.exp(-1j * np.pi / 2),
            's2': 0.20 * np.exp(-1j * np.pi / 4),
            'n2': 0.10 * np.exp(-1j * np.pi / 3),
            'k2': 0.05 * np.exp(-1j * np.pi / 6),
        }

        # Single point
        data_vars = {
            name: (['point'], np.atleast_1d(value))
            for name, value in constituents.items()
        }

        ds = xr.Dataset(
            data_vars,
            coords={
                'x': (['point'], [140.0]),
                'y': (['point'], [35.0]),
            }
        )

        return ds

    def test_infer_diurnal_linear(self, sample_dataset):
        """Test diurnal minor inference with linear method"""
        mjd = 60310.0 + np.arange(24) / 24.0  # 1 day hourly

        tide = infer_diurnal(mjd, sample_dataset, method='linear')

        assert tide.shape == (1, 24)
        # Tide should be small (minor constituents)
        assert np.nanmax(np.abs(tide)) < 0.1

    def test_infer_diurnal_admittance(self, sample_dataset):
        """Test diurnal minor inference with admittance method"""
        mjd = 60310.0 + np.arange(24) / 24.0

        tide = infer_diurnal(mjd, sample_dataset, method='admittance')

        assert tide.shape == (1, 24)
        assert np.nanmax(np.abs(tide)) < 0.1

    def test_infer_semi_diurnal_linear(self, sample_dataset):
        """Test semi-diurnal minor inference with linear method"""
        mjd = 60310.0 + np.arange(24) / 24.0

        tide = infer_semi_diurnal(mjd, sample_dataset, method='linear')

        assert tide.shape == (1, 24)
        assert np.nanmax(np.abs(tide)) < 0.1

    def test_infer_semi_diurnal_admittance(self, sample_dataset):
        """Test semi-diurnal minor inference with admittance method"""
        mjd = 60310.0 + np.arange(24) / 24.0

        tide = infer_semi_diurnal(mjd, sample_dataset, method='admittance')

        assert tide.shape == (1, 24)
        assert np.nanmax(np.abs(tide)) < 0.1

    def test_infer_minor_combined(self, sample_dataset):
        """Test combined minor constituent inference"""
        mjd = 60310.0 + np.arange(24) / 24.0

        tide = infer_minor(mjd, sample_dataset, method='linear')

        assert tide.shape == (1, 24)
        # Combined tide should still be small
        assert np.nanmax(np.abs(tide)) < 0.2

    def test_admittance_vs_linear(self, sample_dataset):
        """Test that admittance and linear methods give different results"""
        mjd = 60310.0 + np.arange(24) / 24.0

        tide_linear = infer_minor(mjd, sample_dataset, method='linear')
        tide_admittance = infer_minor(mjd, sample_dataset, method='admittance')

        # Results should be different but correlated
        assert not np.allclose(tide_linear, tide_admittance)
        # Both should have similar magnitude
        assert np.abs(np.nanmean(np.abs(tide_linear)) - np.nanmean(np.abs(tide_admittance))) < 0.05

    def test_corrections_types(self, sample_dataset):
        """Test different correction types"""
        mjd = 60310.0 + np.arange(24) / 24.0

        # Should work with different corrections
        tide_got = infer_minor(mjd, sample_dataset, corrections='GOT')
        tide_otis = infer_minor(mjd, sample_dataset, corrections='OTIS')

        assert tide_got.shape == (1, 24)
        assert tide_otis.shape == (1, 24)

    def test_deltat_parameter(self, sample_dataset):
        """Test deltat time correction parameter"""
        mjd = 60310.0 + np.arange(24) / 24.0
        deltat = np.zeros_like(mjd) + 0.0001  # ~8.6 seconds

        tide_no_dt = infer_minor(mjd, sample_dataset)
        tide_with_dt = infer_minor(mjd, sample_dataset, deltat=deltat)

        # Results should be slightly different
        assert not np.allclose(tide_no_dt, tide_with_dt, atol=1e-10)
        # But very similar
        assert np.allclose(tide_no_dt, tide_with_dt, atol=1e-3)

    def test_multiple_points(self):
        """Test minor inference with multiple points"""
        # Create dataset with multiple points
        n_points = 5
        constituents = {
            'o1': 0.10 * np.exp(-1j * np.random.rand(n_points) * 2 * np.pi),
            'k1': 0.12 * np.exp(-1j * np.random.rand(n_points) * 2 * np.pi),
            'q1': 0.02 * np.exp(-1j * np.random.rand(n_points) * 2 * np.pi),
            'p1': 0.04 * np.exp(-1j * np.random.rand(n_points) * 2 * np.pi),
            'm2': 0.50 * np.exp(-1j * np.random.rand(n_points) * 2 * np.pi),
            's2': 0.20 * np.exp(-1j * np.random.rand(n_points) * 2 * np.pi),
            'n2': 0.10 * np.exp(-1j * np.random.rand(n_points) * 2 * np.pi),
            'k2': 0.05 * np.exp(-1j * np.random.rand(n_points) * 2 * np.pi),
        }

        data_vars = {
            name: (['point'], values)
            for name, values in constituents.items()
        }

        ds = xr.Dataset(
            data_vars,
            coords={
                'x': (['point'], np.linspace(130, 150, n_points)),
                'y': (['point'], np.linspace(30, 40, n_points)),
            }
        )

        mjd = 60310.0 + np.arange(24) / 24.0
        tide = infer_minor(mjd, ds, method='linear')

        assert tide.shape == (n_points, 24)
        assert np.all(np.isfinite(tide))

    def test_missing_diurnal_constituent(self):
        """Test behavior when diurnal constituent is missing"""
        # Create dataset without O1
        constituents = {
            'k1': 0.12 * np.exp(-1j * np.pi / 3),
            'q1': 0.02 * np.exp(-1j * np.pi / 6),
            'm2': 0.50 * np.exp(-1j * np.pi / 2),
            's2': 0.20 * np.exp(-1j * np.pi / 4),
            'n2': 0.10 * np.exp(-1j * np.pi / 3),
            'k2': 0.05 * np.exp(-1j * np.pi / 6),
        }

        data_vars = {
            name: (['point'], np.atleast_1d(value))
            for name, value in constituents.items()
        }

        ds = xr.Dataset(
            data_vars,
            coords={
                'x': (['point'], [140.0]),
                'y': (['point'], [35.0]),
            }
        )

        mjd = 60310.0 + np.arange(24) / 24.0
        # Should still work (missing O1 treated as zero)
        tide = infer_minor(mjd, ds, method='linear')

        assert tide.shape == (1, 24)
        assert np.all(np.isfinite(tide))

    def test_fes_corrections(self, sample_dataset):
        """Test with FES correction type"""
        mjd = 60310.0 + np.arange(24) / 24.0

        tide = infer_minor(mjd, sample_dataset, corrections='FES')

        assert tide.shape == (1, 24)
        assert np.all(np.isfinite(tide))


class TestInferMinorWithModel:
    """Test minor inference with actual model data (requires model files)"""

    @pytest.fixture
    def model_directory(self):
        """Get model directory from environment"""
        import os
        directory = os.environ.get('PYTMD_RESOURCE')
        if not directory:
            pytest.skip('PYTMD_RESOURCE environment variable not set')
        return directory

    def test_got55_minor_inference(self, model_directory):
        """Test minor inference with GOT5.5 model"""
        from pyTMD_turbo.io import model, TMDAccessor

        m = model(directory=model_directory).from_database('GOT5.5')
        ds = m.open_dataset()

        accessor = TMDAccessor(ds)

        # Interpolate to a point
        local = accessor.interp(x=140.0, y=35.0)

        # Test minor inference
        mjd = 60310.0 + np.arange(24) / 24.0
        minor = infer_minor(mjd, local, method='admittance', corrections='GOT')

        assert minor.shape == (1, 24)
        # Minor constituents should have reasonable amplitude
        assert np.nanmax(np.abs(minor)) > 0.001
        assert np.nanmax(np.abs(minor)) < 0.5
