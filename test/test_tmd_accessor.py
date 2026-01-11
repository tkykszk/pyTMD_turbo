"""
Tests for the TMD xarray accessor

Tests the ds.tmd accessor interface for interpolation and prediction.
"""

import numpy as np
import pytest

# Check for required dependencies
pytest.importorskip('xarray')
pytest.importorskip('netCDF4')

import xarray as xr
from pyTMD_turbo.io import TMDAccessor, register_accessor


class TestTMDAccessor:
    """Test TMD accessor functionality"""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample tidal model dataset"""
        # Create a simple grid
        lon = np.arange(0, 360, 1.0)
        lat = np.arange(-90, 90, 1.0)
        n_lon, n_lat = len(lon), len(lat)

        # Create sample harmonic constants for M2 and S2
        np.random.seed(42)
        m2_amp = 0.5 + 0.3 * np.random.rand(n_lat, n_lon)
        m2_phase = 2 * np.pi * np.random.rand(n_lat, n_lon)
        m2 = m2_amp * np.exp(-1j * m2_phase)

        s2_amp = 0.3 + 0.2 * np.random.rand(n_lat, n_lon)
        s2_phase = 2 * np.pi * np.random.rand(n_lat, n_lon)
        s2 = s2_amp * np.exp(-1j * s2_phase)

        ds = xr.Dataset(
            {
                'm2': (['y', 'x'], m2),
                's2': (['y', 'x'], s2),
            },
            coords={
                'x': lon,
                'y': lat,
            }
        )

        return ds

    def test_accessor_registration(self):
        """Test that accessor is properly registered"""
        register_accessor()
        assert hasattr(xr.Dataset, 'tmd')

    def test_constituents_property(self, sample_dataset):
        """Test constituents property"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        constituents = accessor.constituents
        assert 'm2' in constituents
        assert 's2' in constituents
        assert len(constituents) == 2

    def test_is_global_property(self, sample_dataset):
        """Test is_global property for global dataset"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        assert accessor.is_global is True

    def test_is_global_regional(self):
        """Test is_global property for regional dataset"""
        # Create a regional dataset
        lon = np.arange(130, 150, 1.0)
        lat = np.arange(30, 40, 1.0)
        n_lon, n_lat = len(lon), len(lat)

        ds = xr.Dataset(
            {
                'm2': (['y', 'x'], np.random.rand(n_lat, n_lon) + 0j),
            },
            coords={'x': lon, 'y': lat}
        )

        accessor = TMDAccessor(ds)
        assert accessor.is_global is False

    def test_interp_single_point(self, sample_dataset):
        """Test interpolation at a single point"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        result = accessor.interp(x=140.5, y=35.5)

        assert 'm2' in result.data_vars
        assert 's2' in result.data_vars
        assert len(result.coords['x']) == 1

    def test_interp_multiple_points(self, sample_dataset):
        """Test interpolation at multiple points"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        lons = np.array([140.0, 150.0, 160.0])
        lats = np.array([35.0, 30.0, 25.0])

        result = accessor.interp(x=lons, y=lats)

        assert 'm2' in result.data_vars
        assert len(result.coords['x']) == 3
        assert np.allclose(result.coords['x'].values, lons)
        assert np.allclose(result.coords['y'].values, lats)

    def test_interp_preserves_complex(self, sample_dataset):
        """Test that interpolation preserves complex values"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        result = accessor.interp(x=140.0, y=35.0)

        # Check that M2 is complex
        assert np.iscomplexobj(result['m2'].values)

    def test_predict_point(self, sample_dataset):
        """Test prediction at interpolated points"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        # Create time array
        times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')

        tide = accessor.predict(
            t=times,
            x=np.array([140.0]),
            y=np.array([35.0]),
            corrections='GOT'
        )

        assert tide.shape == (1, len(times))
        # Tide should be within reasonable range
        assert np.all(np.abs(tide) < 5.0)

    def test_predict_multiple_points(self, sample_dataset):
        """Test prediction at multiple points"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')
        lons = np.array([140.0, 150.0, 160.0])
        lats = np.array([35.0, 30.0, 25.0])

        tide = accessor.predict(t=times, x=lons, y=lats, corrections='GOT')

        assert tide.shape == (3, len(times))

    def test_transform_as_no_pyproj(self, sample_dataset):
        """Test transform_as returns input when pyproj not available or same CRS"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        x_in = np.array([140.0, 150.0])
        y_in = np.array([35.0, 30.0])

        x_out, y_out = accessor.transform_as(x_in, y_in)

        np.testing.assert_array_equal(x_out, x_in)
        np.testing.assert_array_equal(y_out, y_in)

    def test_coords_as(self, sample_dataset):
        """Test coords_as returns dataset coordinates"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        x, y = accessor.coords_as()

        np.testing.assert_array_equal(x, ds.coords['x'].values)
        np.testing.assert_array_equal(y, ds.coords['y'].values)

    def test_crs_property(self, sample_dataset):
        """Test crs property returns CRS object or None"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        crs = accessor.crs
        # Should return a CRS object (if pyproj available) or None
        # Default should be WGS84 (EPSG:4326)
        if crs is not None:
            assert hasattr(crs, 'to_epsg')

    def test_infer_method(self):
        """Test infer method for minor constituent inference"""
        # Create dataset with major constituents needed for inference
        lon = np.arange(0, 360, 1.0)
        lat = np.arange(-90, 90, 1.0)
        n_lon, n_lat = len(lon), len(lat)

        np.random.seed(42)
        constituents = {}
        for name in ['o1', 'k1', 'q1', 'p1', 'm2', 's2', 'n2', 'k2']:
            amp = 0.1 + 0.2 * np.random.rand(n_lat, n_lon)
            phase = 2 * np.pi * np.random.rand(n_lat, n_lon)
            constituents[name] = amp * np.exp(-1j * phase)

        ds = xr.Dataset(
            {name: (['y', 'x'], data) for name, data in constituents.items()},
            coords={'x': lon, 'y': lat}
        )

        accessor = TMDAccessor(ds)
        local = accessor.interp(x=140.0, y=35.0)

        # Test infer method
        times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')
        local_accessor = TMDAccessor(local)
        minor = local_accessor.infer(times, method='linear', corrections='GOT')

        assert minor.shape[1] == len(times)
        assert np.all(np.isfinite(minor))

    def test_interp_nearest_method(self, sample_dataset):
        """Test interpolation with nearest neighbor method"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        result = accessor.interp(x=140.5, y=35.5, method='nearest')

        assert 'm2' in result.data_vars
        assert np.iscomplexobj(result['m2'].values)

    def test_interp_spline_method(self, sample_dataset):
        """Test interpolation with spline method"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        result = accessor.interp(x=140.5, y=35.5, method='spline')

        assert 'm2' in result.data_vars
        assert np.iscomplexobj(result['m2'].values)

    def test_predict_with_datetime_objects(self, sample_dataset):
        """Test prediction with Python datetime objects"""
        from datetime import datetime, timezone

        ds = sample_dataset
        accessor = TMDAccessor(ds)

        # Create datetime array
        times = [
            datetime(2024, 1, 1, i, 0, 0, tzinfo=timezone.utc)
            for i in range(24)
        ]

        tide = accessor.predict(
            t=np.array(times),
            x=np.array([140.0]),
            y=np.array([35.0]),
            corrections='GOT'
        )

        assert tide.shape == (1, 24)

    def test_predict_with_mjd(self, sample_dataset):
        """Test prediction with MJD values"""
        ds = sample_dataset
        accessor = TMDAccessor(ds)

        # MJD for 2024-01-01
        mjd = 60310.0 + np.arange(24) / 24.0

        tide = accessor.predict(
            t=mjd,
            x=np.array([140.0]),
            y=np.array([35.0]),
            corrections='GOT'
        )

        assert tide.shape == (1, 24)


class TestTMDAccessorWithModel:
    """Test TMD accessor with actual model data (requires model files)"""

    @pytest.fixture
    def model_directory(self):
        """Get model directory from environment"""
        import os
        directory = os.environ.get('PYTMD_RESOURCE')
        if not directory:
            pytest.skip('PYTMD_RESOURCE environment variable not set')
        return directory

    def test_got55_accessor(self, model_directory):
        """Test accessor with GOT5.5 model"""
        from pyTMD_turbo.io import model

        m = model(directory=model_directory).from_database('GOT5.5')
        ds = m.open_dataset()

        accessor = TMDAccessor(ds)

        # Check constituents
        assert len(accessor.constituents) > 0
        assert 'm2' in accessor.constituents

        # Check is_global
        assert accessor.is_global is True

        # Test interpolation
        result = accessor.interp(x=140.0, y=35.0)
        assert 'm2' in result.data_vars

        # Test prediction
        times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')
        tide = accessor.predict(
            t=times,
            x=np.array([140.0]),
            y=np.array([35.0]),
            corrections='GOT'
        )

        assert tide.shape == (1, len(times))
        # Tide amplitude should be reasonable
        assert np.nanmax(np.abs(tide)) > 0.01
        assert np.nanmax(np.abs(tide)) < 10.0
