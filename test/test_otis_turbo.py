"""
Tests for pyTMD_turbo.io.OTIS

Tests OTIS binary format reading functionality for pyTMD_turbo.
"""

import os
import tempfile
import pathlib
import numpy as np
import pytest

pytest.importorskip('xarray')

from pyTMD_turbo.io import OTIS


class TestReadRawBinary:
    """Test raw binary reading functionality"""

    def test_read_raw_binary_basic(self):
        """Test basic binary file reading"""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            data = np.array([1.0, 2.0, 3.0, 4.0], dtype='>f4')
            data.tofile(f)
            temp_path = f.name

        try:
            result = OTIS.read_raw_binary(temp_path, dtype='>f4', shape=(4,))
            np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])
        finally:
            os.unlink(temp_path)

    def test_read_raw_binary_with_offset(self):
        """Test binary reading with offset"""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            # Write header (4 bytes) + data
            header = np.array([0], dtype='>i4')
            data = np.array([5.0, 6.0, 7.0, 8.0], dtype='>f4')
            header.tofile(f)
            data.tofile(f)
            temp_path = f.name

        try:
            result = OTIS.read_raw_binary(temp_path, dtype='>f4', shape=(4,), offset=4)
            np.testing.assert_array_almost_equal(result, [5.0, 6.0, 7.0, 8.0])
        finally:
            os.unlink(temp_path)

    def test_read_raw_binary_2d_shape(self):
        """Test binary reading with 2D shape"""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype='>f4')
            data.tofile(f)
            temp_path = f.name

        try:
            result = OTIS.read_raw_binary(temp_path, dtype='>f4', shape=(2, 2))
            np.testing.assert_array_almost_equal(result, [[1.0, 2.0], [3.0, 4.0]])
        finally:
            os.unlink(temp_path)

    def test_read_raw_binary_with_mmap(self):
        """Test memory-mapped file reading"""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            data = np.array([1.0, 2.0, 3.0, 4.0], dtype='>f4')
            data.tofile(f)
            temp_path = f.name

        try:
            result = OTIS.read_raw_binary(temp_path, dtype='>f4', shape=(4,), use_mmap=True)
            np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])
        finally:
            os.unlink(temp_path)


class TestOTISBinaryFormat:
    """Test OTIS binary format specific functionality"""

    @pytest.fixture
    def sample_otis_grid_file(self):
        """Create a sample OTIS grid file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.grid', delete=False) as f:
            temp_path = f.name

            nx, ny = 10, 8
            ylim = np.array([30.0, 40.0], dtype='>f4')
            xlim = np.array([130.0, 140.0], dtype='>f4')
            dt = np.array([0.5], dtype='>f4')
            nob = np.array([0], dtype='>i4')

            # Create bathymetry (water depth)
            hz = np.ones((ny, nx), dtype='>f4') * 100.0
            hz[0, :] = 0  # Land at bottom

            # Create mask (1=wet, 0=dry)
            mz = np.ones((ny, nx), dtype='>i4')
            mz[0, :] = 0  # Land at bottom

            # Write file
            with open(temp_path, 'wb') as fid:
                # Record marker
                np.array([36], dtype='>i4').tofile(fid)
                # Header
                np.array([nx, ny], dtype='>i4').tofile(fid)
                ylim.tofile(fid)
                xlim.tofile(fid)
                dt.tofile(fid)
                nob.tofile(fid)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)
                # iob placeholder when nob=0
                np.array([0], dtype='>i4').tofile(fid)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)
                # hz matrix
                hz.tofile(fid)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)
                # mz matrix
                mz.tofile(fid)

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def sample_otis_elevation_file(self):
        """Create a sample OTIS elevation file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.h', delete=False) as f:
            temp_path = f.name

            nx, ny, nc = 10, 8, 2
            constituents = [b'M2  ', b'S2  ']

            # Create complex harmonic constants
            np.random.seed(42)
            hc_real = np.random.rand(nc, ny, nx).astype('>f4') * 0.5
            hc_imag = np.random.rand(nc, ny, nx).astype('>f4') * 0.5

            with open(temp_path, 'wb') as fid:
                # Record marker
                np.array([0], dtype='>i4').tofile(fid)
                # Header
                np.array([nx, ny, nc], dtype='>i4').tofile(fid)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)
                # Constituent names
                for c in constituents:
                    fid.write(c)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)
                # Write complex data interleaved
                for i in range(nc):
                    data = np.zeros((ny, nx * 2), dtype='>f4')
                    data[:, 0::2] = hc_real[i]
                    data[:, 1::2] = hc_imag[i]
                    data.tofile(fid)
                    np.array([0, 0], dtype='>i4').tofile(fid)

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def sample_otis_transport_file(self):
        """Create a sample OTIS transport file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.UV', delete=False) as f:
            temp_path = f.name

            nx, ny, nc = 10, 8, 2
            constituents = [b'M2  ', b'S2  ']

            np.random.seed(42)

            with open(temp_path, 'wb') as fid:
                # Record marker
                np.array([0], dtype='>i4').tofile(fid)
                # Header
                np.array([nx, ny, nc], dtype='>i4').tofile(fid)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)
                # Constituent names
                for c in constituents:
                    fid.write(c)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)

                # Write U and V components for each constituent
                for i in range(nc):
                    # U component
                    u_data = np.zeros((ny, nx * 2), dtype='>f4')
                    u_data[:, 0::2] = np.random.rand(ny, nx).astype('>f4') * 10
                    u_data[:, 1::2] = np.random.rand(ny, nx).astype('>f4') * 10
                    u_data.tofile(fid)
                    np.array([0, 0], dtype='>i4').tofile(fid)

                    # V component
                    v_data = np.zeros((ny, nx * 2), dtype='>f4')
                    v_data[:, 0::2] = np.random.rand(ny, nx).astype('>f4') * 10
                    v_data[:, 1::2] = np.random.rand(ny, nx).astype('>f4') * 10
                    v_data.tofile(fid)
                    np.array([0, 0], dtype='>i4').tofile(fid)

        yield temp_path
        os.unlink(temp_path)

    def test_read_grid(self, sample_otis_grid_file):
        """Test reading OTIS grid file"""
        x, y, hz, mz = OTIS.read_grid(sample_otis_grid_file)

        assert len(x) == 10
        assert len(y) == 8
        assert hz.shape == (8, 10)
        assert mz.shape == (8, 10)

        # Check coordinate ranges
        assert x[0] >= 130.0
        assert x[-1] <= 140.0
        assert y[0] >= 30.0
        assert y[-1] <= 40.0

        # Check mask
        assert np.sum(mz[0, :]) == 0  # Bottom row is land

    def test_read_grid_with_mmap(self, sample_otis_grid_file):
        """Test reading OTIS grid file with memory mapping"""
        x, y, hz, mz = OTIS.read_grid(sample_otis_grid_file, use_mmap=True)

        assert len(x) == 10
        assert len(y) == 8
        assert hz.shape == (8, 10)
        assert mz.shape == (8, 10)

    def test_read_elevation(self, sample_otis_elevation_file):
        """Test reading OTIS elevation file"""
        hc, constituents = OTIS.read_elevation(sample_otis_elevation_file)

        assert hc.shape == (2, 8, 10)
        assert len(constituents) == 2
        assert 'm2' in constituents
        assert 's2' in constituents
        assert np.iscomplexobj(hc)

    def test_read_transport(self, sample_otis_transport_file):
        """Test reading OTIS transport file"""
        uc, vc, constituents = OTIS.read_transport(sample_otis_transport_file)

        assert uc.shape == (2, 8, 10)
        assert vc.shape == (2, 8, 10)
        assert len(constituents) == 2
        assert np.iscomplexobj(uc)
        assert np.iscomplexobj(vc)

    def test_read_grid_file_not_found(self):
        """Test FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError):
            OTIS.read_grid('/nonexistent/path/to/grid')


class TestOpenFunctions:
    """Test xarray Dataset opening functions"""

    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample OTIS files for testing"""
        nx, ny, nc = 10, 8, 2
        ylim = np.array([30.0, 40.0], dtype='>f4')
        xlim = np.array([130.0, 140.0], dtype='>f4')

        np.random.seed(42)

        # Create grid file
        grid_path = tmp_path / "grid"
        with open(grid_path, 'wb') as fid:
            np.array([0], dtype='>i4').tofile(fid)
            np.array([nx, ny], dtype='>i4').tofile(fid)
            ylim.tofile(fid)
            xlim.tofile(fid)
            np.array([0.5], dtype='>f4').tofile(fid)
            np.array([0], dtype='>i4').tofile(fid)
            np.array([0, 0], dtype='>i4').tofile(fid)
            np.array([0], dtype='>i4').tofile(fid)
            np.array([0, 0], dtype='>i4').tofile(fid)
            hz = np.ones((ny, nx), dtype='>f4') * 100.0
            hz.tofile(fid)
            np.array([0, 0], dtype='>i4').tofile(fid)
            mz = np.ones((ny, nx), dtype='>i4')
            mz.tofile(fid)

        # Create elevation file
        elev_path = tmp_path / "h_model"
        with open(elev_path, 'wb') as fid:
            np.array([0], dtype='>i4').tofile(fid)
            np.array([nx, ny, nc], dtype='>i4').tofile(fid)
            np.array([0, 0], dtype='>i4').tofile(fid)
            fid.write(b'M2  S2  ')
            np.array([0, 0], dtype='>i4').tofile(fid)
            for i in range(nc):
                data = np.zeros((ny, nx * 2), dtype='>f4')
                data[:, 0::2] = np.random.rand(ny, nx).astype('>f4') * 0.5
                data[:, 1::2] = np.random.rand(ny, nx).astype('>f4') * 0.5
                data.tofile(fid)
                np.array([0, 0], dtype='>i4').tofile(fid)

        # Create transport file
        trans_path = tmp_path / "UV_model"
        with open(trans_path, 'wb') as fid:
            np.array([0], dtype='>i4').tofile(fid)
            np.array([nx, ny, nc], dtype='>i4').tofile(fid)
            np.array([0, 0], dtype='>i4').tofile(fid)
            fid.write(b'M2  S2  ')
            np.array([0, 0], dtype='>i4').tofile(fid)
            for i in range(nc):
                # U component
                u_data = np.zeros((ny, nx * 2), dtype='>f4')
                u_data[:, 0::2] = np.random.rand(ny, nx).astype('>f4') * 10
                u_data[:, 1::2] = np.random.rand(ny, nx).astype('>f4') * 10
                u_data.tofile(fid)
                np.array([0, 0], dtype='>i4').tofile(fid)
                # V component
                v_data = np.zeros((ny, nx * 2), dtype='>f4')
                v_data[:, 0::2] = np.random.rand(ny, nx).astype('>f4') * 10
                v_data[:, 1::2] = np.random.rand(ny, nx).astype('>f4') * 10
                v_data.tofile(fid)
                np.array([0, 0], dtype='>i4').tofile(fid)

        return {
            'grid': grid_path,
            'elevation': elev_path,
            'transport': trans_path,
        }

    def test_open_otis_grid(self, sample_files):
        """Test opening OTIS grid as xarray Dataset"""
        ds = OTIS.open_otis_grid(sample_files['grid'])

        assert 'bathymetry' in ds.data_vars
        assert 'mask' in ds.data_vars
        assert 'x' in ds.coords
        assert 'y' in ds.coords
        assert ds.attrs['format'] == 'OTIS'

    def test_open_otis_elevation(self, sample_files):
        """Test opening OTIS elevation as xarray Dataset"""
        ds = OTIS.open_otis_elevation(
            sample_files['elevation'],
            grid_file=sample_files['grid']
        )

        assert 'm2' in ds.data_vars
        assert 's2' in ds.data_vars
        assert 'constituents' in ds.attrs
        assert np.iscomplexobj(ds['m2'].values)

    def test_open_otis_elevation_no_grid(self, sample_files):
        """Test opening OTIS elevation without grid file"""
        ds = OTIS.open_otis_elevation(sample_files['elevation'])

        assert 'm2' in ds.data_vars
        assert 's2' in ds.data_vars

    def test_open_otis_transport(self, sample_files):
        """Test opening OTIS transport as xarray Dataset"""
        ds = OTIS.open_otis_transport(
            sample_files['transport'],
            grid_file=sample_files['grid']
        )

        assert 'u_m2' in ds.data_vars
        assert 'v_m2' in ds.data_vars
        assert 'u_s2' in ds.data_vars
        assert 'v_s2' in ds.data_vars

    def test_open_otis_transport_velocity_conversion(self, sample_files):
        """Test opening OTIS transport with velocity conversion"""
        ds = OTIS.open_otis_transport(
            sample_files['transport'],
            grid_file=sample_files['grid'],
            convert_to_velocity=True
        )

        # Check units
        assert ds['u_m2'].attrs['units'] == 'm/s'
        assert ds['v_m2'].attrs['units'] == 'm/s'

    def test_open_dataset_elevation(self, sample_files):
        """Test open_dataset for elevation"""
        ds = OTIS.open_dataset(
            sample_files['elevation'],
            grid_file=sample_files['grid'],
            group='z'
        )

        assert 'm2' in ds.data_vars
        assert 's2' in ds.data_vars

    def test_open_dataset_u_component(self, sample_files):
        """Test open_dataset for U component"""
        ds = OTIS.open_dataset(
            sample_files['transport'],
            grid_file=sample_files['grid'],
            group='u'
        )

        assert 'm2' in ds.data_vars
        assert 's2' in ds.data_vars
        assert ds.attrs['component'] == 'u'

    def test_open_dataset_v_component(self, sample_files):
        """Test open_dataset for V component"""
        ds = OTIS.open_dataset(
            sample_files['transport'],
            grid_file=sample_files['grid'],
            group='v'
        )

        assert 'm2' in ds.data_vars
        assert 's2' in ds.data_vars
        assert ds.attrs['component'] == 'v'

    def test_open_dataset_invalid_group(self, sample_files):
        """Test open_dataset with invalid group"""
        with pytest.raises(ValueError, match="Unknown group"):
            OTIS.open_dataset(
                sample_files['elevation'],
                grid_file=sample_files['grid'],
                group='invalid'
            )

    def test_open_dataset_auto_grid_detection(self, sample_files):
        """Test that open_dataset auto-detects grid file"""
        # Grid file should be auto-detected since it's named 'grid' in same directory
        ds = OTIS.open_dataset(
            sample_files['elevation'],
            group='z'
        )

        assert 'm2' in ds.data_vars
        # Should have proper coordinates from grid file
        assert len(ds.coords['x']) == 10


class TestOTISWithModelData:
    """Test OTIS functions with actual model data (requires model files)"""

    @pytest.fixture
    def model_directory(self):
        """Get model directory from environment"""
        directory = os.environ.get('PYTMD_RESOURCE')
        if not directory:
            pytest.skip('PYTMD_RESOURCE environment variable not set')
        return directory

    def test_cats2008_grid(self, model_directory):
        """Test reading CATS2008 grid"""
        grid_file = os.path.join(model_directory, 'CATS2008', 'grid_CATS2008')

        if not os.path.exists(grid_file):
            pytest.skip('CATS2008 grid file not found')

        x, y, hz, mz = OTIS.read_grid(grid_file)

        assert len(x) > 0
        assert len(y) > 0
        assert hz.shape == (len(y), len(x))
        assert mz.shape == (len(y), len(x))

    def test_cats2008_elevation(self, model_directory):
        """Test reading CATS2008 elevation"""
        elev_file = os.path.join(model_directory, 'CATS2008', 'hf.CATS2008.out')

        if not os.path.exists(elev_file):
            pytest.skip('CATS2008 elevation file not found')

        hc, constituents = OTIS.read_elevation(elev_file)

        assert hc.ndim == 3
        assert len(constituents) > 0
        assert np.iscomplexobj(hc)

    def test_cats2008_dataset(self, model_directory):
        """Test opening CATS2008 as xarray Dataset"""
        elev_file = os.path.join(model_directory, 'CATS2008', 'hf.CATS2008.out')
        grid_file = os.path.join(model_directory, 'CATS2008', 'grid_CATS2008')

        if not os.path.exists(elev_file):
            pytest.skip('CATS2008 elevation file not found')

        ds = OTIS.open_dataset(
            elev_file,
            grid_file=grid_file,
            group='z'
        )

        assert 'm2' in ds.data_vars or 'M2' in ds.data_vars
        assert ds.attrs['format'] == 'OTIS'


class TestMemoryMapping:
    """Test memory-mapped file reading"""

    def test_mmap_vs_regular_reading(self, tmp_path):
        """Test that memory-mapped reading gives same results as regular reading"""
        # Create test file
        test_file = tmp_path / "test.bin"
        np.random.seed(42)
        data = np.random.rand(100, 100).astype('>f4')
        data.tofile(test_file)

        # Read with and without mmap
        result_regular = OTIS.read_raw_binary(
            test_file, dtype='>f4', shape=(100, 100), use_mmap=False
        )
        result_mmap = OTIS.read_raw_binary(
            test_file, dtype='>f4', shape=(100, 100), use_mmap=True
        )

        np.testing.assert_array_equal(result_regular, result_mmap)


class TestAPIConsistency:
    """Test API consistency and exports"""

    def test_module_exports(self):
        """Test that all expected functions are exported"""
        assert hasattr(OTIS, 'read_raw_binary')
        assert hasattr(OTIS, 'read_grid')
        assert hasattr(OTIS, 'read_elevation')
        assert hasattr(OTIS, 'read_transport')
        assert hasattr(OTIS, 'open_dataset')
        assert hasattr(OTIS, 'open_otis_grid')
        assert hasattr(OTIS, 'open_otis_elevation')
        assert hasattr(OTIS, 'open_otis_transport')
        assert hasattr(OTIS, 'open_mfdataset')

    def test_io_module_exports(self):
        """Test that OTIS functions are exported from io module"""
        from pyTMD_turbo import io

        assert hasattr(io, 'open_dataset')
        assert hasattr(io, 'open_otis_grid')
        assert hasattr(io, 'open_otis_elevation')
        assert hasattr(io, 'open_otis_transport')
        assert hasattr(io, 'open_mfdataset')
