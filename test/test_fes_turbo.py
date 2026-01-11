"""
Tests for pyTMD_turbo.io.FES

Tests FES NetCDF format reading functionality for pyTMD_turbo.
"""

import os
import tempfile
import numpy as np
import pytest

xr = pytest.importorskip('xarray')
nc = pytest.importorskip('netCDF4')

from pyTMD_turbo.io import FES


class TestFESNetCDF:
    """Test FES NetCDF reading"""

    @pytest.fixture
    def sample_fes_file(self, tmp_path):
        """Create a sample FES NetCDF file for testing"""
        nc_path = tmp_path / "m2_tide.nc"

        # Create sample grid
        lon = np.linspace(0, 360, 361)
        lat = np.linspace(-90, 90, 181)
        nlon, nlat = len(lon), len(lat)

        # Create amplitude and phase with some structure
        np.random.seed(42)
        lon2d, lat2d = np.meshgrid(lon, lat)
        amplitude = 0.5 + 0.3 * np.cos(np.radians(lat2d)) * np.sin(np.radians(lon2d))
        phase = np.mod(lon2d * 2 + lat2d, 360)

        # Create NetCDF file
        with nc.Dataset(nc_path, 'w', format='NETCDF4') as ds:
            # Create dimensions
            ds.createDimension('lon', nlon)
            ds.createDimension('lat', nlat)

            # Create coordinate variables
            lon_var = ds.createVariable('lon', 'f4', ('lon',))
            lon_var[:] = lon
            lon_var.standard_name = 'longitude'
            lon_var.units = 'degrees_east'

            lat_var = ds.createVariable('lat', 'f4', ('lat',))
            lat_var[:] = lat
            lat_var.standard_name = 'latitude'
            lat_var.units = 'degrees_north'

            # Create data variables
            amp_var = ds.createVariable('amplitude', 'f4', ('lat', 'lon'))
            amp_var[:] = amplitude
            amp_var.standard_name = 'sea_surface_height_amplitude'
            amp_var.units = 'm'

            phase_var = ds.createVariable('phase', 'f4', ('lat', 'lon'))
            phase_var[:] = phase
            phase_var.standard_name = 'sea_surface_height_phase'
            phase_var.units = 'degrees'

            # Global attributes
            ds.constituent = 'm2'

        return nc_path

    @pytest.fixture
    def sample_fes_u_file(self, tmp_path):
        """Create a sample FES U velocity file"""
        nc_path = tmp_path / "u_m2.nc"

        lon = np.linspace(0, 360, 361)
        lat = np.linspace(-90, 90, 181)
        nlon, nlat = len(lon), len(lat)

        np.random.seed(43)
        amplitude = 0.1 + 0.05 * np.random.rand(nlat, nlon)
        phase = np.random.rand(nlat, nlon) * 360

        with nc.Dataset(nc_path, 'w', format='NETCDF4') as ds:
            ds.createDimension('lon', nlon)
            ds.createDimension('lat', nlat)

            lon_var = ds.createVariable('lon', 'f4', ('lon',))
            lon_var[:] = lon

            lat_var = ds.createVariable('lat', 'f4', ('lat',))
            lat_var[:] = lat

            amp_var = ds.createVariable('amplitude', 'f4', ('lat', 'lon'))
            amp_var[:] = amplitude

            phase_var = ds.createVariable('phase', 'f4', ('lat', 'lon'))
            phase_var[:] = phase

            ds.constituent = 'm2'

        return nc_path

    @pytest.fixture
    def sample_fes_v_file(self, tmp_path):
        """Create a sample FES V velocity file"""
        nc_path = tmp_path / "v_m2.nc"

        lon = np.linspace(0, 360, 361)
        lat = np.linspace(-90, 90, 181)
        nlon, nlat = len(lon), len(lat)

        np.random.seed(44)
        amplitude = 0.1 + 0.05 * np.random.rand(nlat, nlon)
        phase = np.random.rand(nlat, nlon) * 360

        with nc.Dataset(nc_path, 'w', format='NETCDF4') as ds:
            ds.createDimension('lon', nlon)
            ds.createDimension('lat', nlat)

            lon_var = ds.createVariable('lon', 'f4', ('lon',))
            lon_var[:] = lon

            lat_var = ds.createVariable('lat', 'f4', ('lat',))
            lat_var[:] = lat

            amp_var = ds.createVariable('amplitude', 'f4', ('lat', 'lon'))
            amp_var[:] = amplitude

            phase_var = ds.createVariable('phase', 'f4', ('lat', 'lon'))
            phase_var[:] = phase

            ds.constituent = 'm2'

        return nc_path

    def test_read_netcdf(self, sample_fes_file):
        """Test reading FES NetCDF file"""
        data = FES.read_netcdf(sample_fes_file)

        assert 'amplitude' in data
        assert 'phase' in data
        assert 'x' in data
        assert 'y' in data
        assert len(data['x']) == 361
        assert len(data['y']) == 181
        assert data['amplitude'].shape == (181, 361)
        assert data['constituent'] == 'm2'

    def test_read_constituent(self, sample_fes_file):
        """Test reading constituent with complex conversion"""
        data = FES.read_constituent(sample_fes_file)

        assert 'hc' in data
        assert np.iscomplexobj(data['hc'])
        assert data['hc'].shape == (181, 361)

        # Check amplitude/phase to complex conversion
        expected_amp = data['amplitude']
        computed_amp = np.abs(data['hc'])
        np.testing.assert_array_almost_equal(expected_amp, computed_amp, decimal=5)

    def test_open_fes_elevation_single_file(self, sample_fes_file):
        """Test opening FES elevation as xarray Dataset"""
        ds = FES.open_fes_elevation(sample_fes_file)

        assert 'm2' in ds.data_vars
        assert 'x' in ds.coords
        assert 'y' in ds.coords
        assert ds.attrs['format'] == 'FES'
        assert np.iscomplexobj(ds['m2'].values)

    def test_open_fes_elevation_multiple_files(self, tmp_path):
        """Test opening multiple FES files"""
        # Create multiple constituent files
        files = []
        for const in ['m2', 's2']:
            nc_path = tmp_path / f"{const}_tide.nc"

            lon = np.linspace(0, 360, 37)
            lat = np.linspace(-90, 90, 19)
            nlon, nlat = len(lon), len(lat)

            np.random.seed(42)
            amplitude = 0.5 * np.random.rand(nlat, nlon)
            phase = np.random.rand(nlat, nlon) * 360

            with nc.Dataset(nc_path, 'w', format='NETCDF4') as ds:
                ds.createDimension('lon', nlon)
                ds.createDimension('lat', nlat)

                lon_var = ds.createVariable('lon', 'f4', ('lon',))
                lon_var[:] = lon

                lat_var = ds.createVariable('lat', 'f4', ('lat',))
                lat_var[:] = lat

                amp_var = ds.createVariable('amplitude', 'f4', ('lat', 'lon'))
                amp_var[:] = amplitude

                phase_var = ds.createVariable('phase', 'f4', ('lat', 'lon'))
                phase_var[:] = phase

                ds.constituent = const

            files.append(nc_path)

        ds = FES.open_fes_elevation(files)

        assert 'm2' in ds.data_vars
        assert 's2' in ds.data_vars
        assert len(ds.attrs['constituents']) == 2

    def test_open_fes_transport(self, sample_fes_u_file, sample_fes_v_file):
        """Test opening FES transport files"""
        ds = FES.open_fes_transport([sample_fes_u_file], [sample_fes_v_file])

        assert 'u_m2' in ds.data_vars
        assert 'v_m2' in ds.data_vars
        assert ds.attrs['format'] == 'FES'

    def test_open_dataset_elevation(self, sample_fes_file):
        """Test open_dataset for elevation"""
        ds = FES.open_dataset(sample_fes_file, group='z')

        assert 'm2' in ds.data_vars
        assert ds.attrs['format'] == 'FES'

    def test_open_dataset_u_component(self, sample_fes_u_file):
        """Test open_dataset for U component"""
        ds = FES.open_dataset(sample_fes_u_file, group='u')

        assert 'u_m2' in ds.data_vars
        assert ds.attrs['component'] == 'u'

    def test_open_dataset_invalid_group(self, sample_fes_file):
        """Test open_dataset with invalid group"""
        with pytest.raises(ValueError, match="Unknown group"):
            FES.open_dataset(sample_fes_file, group='invalid')

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError):
            FES.read_netcdf('/nonexistent/path/to/file.nc')


class TestFESWithModelData:
    """Test FES functions with actual model data (requires model files)"""

    @pytest.fixture
    def model_directory(self):
        """Get model directory from environment"""
        directory = os.environ.get('PYTMD_RESOURCE')
        if not directory:
            pytest.skip('PYTMD_RESOURCE environment variable not set')
        return directory

    def test_fes2014_elevation(self, model_directory):
        """Test reading FES2014 elevation"""
        import glob

        fes_dir = os.path.join(model_directory, 'fes2014', 'ocean_tide')
        if not os.path.exists(fes_dir):
            pytest.skip('FES2014 data not found')

        # Find M2 file
        m2_files = glob.glob(os.path.join(fes_dir, '*m2*.nc'))
        if not m2_files:
            pytest.skip('FES2014 M2 file not found')

        data = FES.read_netcdf(m2_files[0])

        assert 'amplitude' in data
        assert 'phase' in data
        assert len(data['x']) > 0
        assert len(data['y']) > 0


class TestAPIConsistency:
    """Test API consistency and exports"""

    def test_module_exports(self):
        """Test that all expected functions are exported"""
        assert hasattr(FES, 'read_netcdf')
        assert hasattr(FES, 'read_constituent')
        assert hasattr(FES, 'open_dataset')
        assert hasattr(FES, 'open_fes_elevation')
        assert hasattr(FES, 'open_fes_transport')
        assert hasattr(FES, 'open_mfdataset')

    def test_io_module_exports(self):
        """Test that FES functions are exported from io module"""
        from pyTMD_turbo import io

        assert hasattr(io, 'FES')
        assert hasattr(io, 'open_fes_elevation')
        assert hasattr(io, 'open_fes_transport')
