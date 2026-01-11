"""
Tests for pyTMD_turbo.io.ATLAS

Tests ATLAS-compact binary format reading functionality for pyTMD_turbo.
"""

import os
import tempfile
import numpy as np
import pytest

pytest.importorskip('xarray')

from pyTMD_turbo.io import ATLAS


class TestATLASBinaryFormat:
    """Test ATLAS binary format reading"""

    @pytest.fixture
    def sample_atlas_grid_file(self):
        """Create a sample ATLAS grid file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.grid', delete=False) as f:
            temp_path = f.name

            nx, ny = 10, 8
            ylim = np.array([30.0, 40.0], dtype='>f4')
            xlim = np.array([130.0, 140.0], dtype='>f4')
            dt = np.array([0.5], dtype='>f4')
            nob = np.array([0], dtype='>i4')

            # Create bathymetry
            hz = np.ones((ny, nx), dtype='>f4') * 100.0
            hz[0, :] = 0  # Land at bottom

            # Create mask
            mz = np.ones((ny, nx), dtype='>i4')
            mz[0, :] = 0

            with open(temp_path, 'wb') as fid:
                # Record marker
                np.array([0], dtype='>i4').tofile(fid)
                # Header
                np.array([nx, ny], dtype='>i4').tofile(fid)
                ylim.tofile(fid)
                xlim.tofile(fid)
                dt.tofile(fid)
                nob.tofile(fid)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)
                # iob placeholder
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
    def sample_atlas_elevation_file(self):
        """Create a sample ATLAS elevation file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.h', delete=False) as f:
            temp_path = f.name

            nx, ny, nc = 10, 8, 2
            ylim = np.array([30.0, 40.0], dtype='>f4')
            xlim = np.array([130.0, 140.0], dtype='>f4')
            constituents = [b'M2  ', b'S2  ']

            np.random.seed(42)

            with open(temp_path, 'wb') as fid:
                # Record marker
                np.array([0], dtype='>i4').tofile(fid)
                # Header: ll, nx, ny, nc
                np.array([0, nx, ny, nc], dtype='>i4').tofile(fid)
                # Coordinate limits
                ylim.tofile(fid)
                xlim.tofile(fid)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)
                # Constituent names
                for c in constituents:
                    fid.write(c)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)

                # Write complex data for each constituent (dense format)
                for i in range(nc):
                    real_data = np.random.rand(ny, nx).astype('>f4') * 0.5
                    imag_data = np.random.rand(ny, nx).astype('>f4') * 0.5
                    real_data.tofile(fid)
                    np.array([0, 0], dtype='>i4').tofile(fid)
                    imag_data.tofile(fid)
                    np.array([0, 0], dtype='>i4').tofile(fid)

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def sample_atlas_transport_file(self):
        """Create a sample ATLAS transport file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.UV', delete=False) as f:
            temp_path = f.name

            nx, ny, nc = 10, 8, 2
            ylim = np.array([30.0, 40.0], dtype='>f4')
            xlim = np.array([130.0, 140.0], dtype='>f4')
            constituents = [b'M2  ', b'S2  ']

            np.random.seed(42)

            with open(temp_path, 'wb') as fid:
                # Record marker
                np.array([0], dtype='>i4').tofile(fid)
                # Header
                np.array([0, nx, ny, nc], dtype='>i4').tofile(fid)
                ylim.tofile(fid)
                xlim.tofile(fid)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)
                # Constituent names
                for c in constituents:
                    fid.write(c)
                # Padding
                np.array([0, 0], dtype='>i4').tofile(fid)

                # Write U and V for each constituent
                for i in range(nc):
                    # U component
                    u_real = np.random.rand(ny, nx).astype('>f4') * 10
                    u_imag = np.random.rand(ny, nx).astype('>f4') * 10
                    u_real.tofile(fid)
                    np.array([0, 0], dtype='>i4').tofile(fid)
                    u_imag.tofile(fid)
                    np.array([0, 0], dtype='>i4').tofile(fid)
                    # V component
                    v_real = np.random.rand(ny, nx).astype('>f4') * 10
                    v_imag = np.random.rand(ny, nx).astype('>f4') * 10
                    v_real.tofile(fid)
                    np.array([0, 0], dtype='>i4').tofile(fid)
                    v_imag.tofile(fid)
                    np.array([0, 0], dtype='>i4').tofile(fid)

        yield temp_path
        os.unlink(temp_path)

    def test_read_atlas_grid(self, sample_atlas_grid_file):
        """Test reading ATLAS grid file"""
        grid = ATLAS.read_atlas_grid(sample_atlas_grid_file)

        assert len(grid['x']) == 10
        assert len(grid['y']) == 8
        assert grid['hz'].shape == (8, 10)
        assert grid['mz'].shape == (8, 10)

        # Check coordinate ranges
        assert grid['x'][0] >= 130.0
        assert grid['x'][-1] <= 140.0
        assert grid['y'][0] >= 30.0
        assert grid['y'][-1] <= 40.0

    def test_read_atlas_elevation(self, sample_atlas_elevation_file):
        """Test reading ATLAS elevation file"""
        data = ATLAS.read_atlas_elevation(sample_atlas_elevation_file)

        assert 'constituents' in data
        assert len(data['constituents']) == 2
        assert 'm2' in data['constituents']
        assert 's2' in data['constituents']

        if not data['sparse']:
            assert data['hc'].shape == (2, 8, 10)
            assert np.iscomplexobj(data['hc'])

    def test_read_atlas_transport(self, sample_atlas_transport_file):
        """Test reading ATLAS transport file"""
        data = ATLAS.read_atlas_transport(sample_atlas_transport_file)

        assert 'constituents' in data
        assert len(data['constituents']) == 2
        assert data['uc'].shape == (2, 8, 10)
        assert data['vc'].shape == (2, 8, 10)
        assert np.iscomplexobj(data['uc'])
        assert np.iscomplexobj(data['vc'])


class TestATLASOpenFunctions:
    """Test xarray Dataset opening functions"""

    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample ATLAS files for testing"""
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
            np.array([0, nx, ny, nc], dtype='>i4').tofile(fid)
            ylim.tofile(fid)
            xlim.tofile(fid)
            np.array([0, 0], dtype='>i4').tofile(fid)
            fid.write(b'M2  S2  ')
            np.array([0, 0], dtype='>i4').tofile(fid)
            for i in range(nc):
                real_data = np.random.rand(ny, nx).astype('>f4') * 0.5
                imag_data = np.random.rand(ny, nx).astype('>f4') * 0.5
                real_data.tofile(fid)
                np.array([0, 0], dtype='>i4').tofile(fid)
                imag_data.tofile(fid)
                np.array([0, 0], dtype='>i4').tofile(fid)

        # Create transport file
        trans_path = tmp_path / "UV_model"
        with open(trans_path, 'wb') as fid:
            np.array([0], dtype='>i4').tofile(fid)
            np.array([0, nx, ny, nc], dtype='>i4').tofile(fid)
            ylim.tofile(fid)
            xlim.tofile(fid)
            np.array([0, 0], dtype='>i4').tofile(fid)
            fid.write(b'M2  S2  ')
            np.array([0, 0], dtype='>i4').tofile(fid)
            for i in range(nc):
                # U
                u_real = np.random.rand(ny, nx).astype('>f4') * 10
                u_imag = np.random.rand(ny, nx).astype('>f4') * 10
                u_real.tofile(fid)
                np.array([0, 0], dtype='>i4').tofile(fid)
                u_imag.tofile(fid)
                np.array([0, 0], dtype='>i4').tofile(fid)
                # V
                v_real = np.random.rand(ny, nx).astype('>f4') * 10
                v_imag = np.random.rand(ny, nx).astype('>f4') * 10
                v_real.tofile(fid)
                np.array([0, 0], dtype='>i4').tofile(fid)
                v_imag.tofile(fid)
                np.array([0, 0], dtype='>i4').tofile(fid)

        return {
            'grid': grid_path,
            'elevation': elev_path,
            'transport': trans_path,
        }

    def test_open_atlas_grid(self, sample_files):
        """Test opening ATLAS grid as xarray Dataset"""
        ds = ATLAS.open_atlas_grid(sample_files['grid'])

        assert 'bathymetry' in ds.data_vars
        assert 'mask' in ds.data_vars
        assert 'x' in ds.coords
        assert 'y' in ds.coords
        assert ds.attrs['format'] == 'ATLAS'

    def test_open_atlas_elevation(self, sample_files):
        """Test opening ATLAS elevation as xarray Dataset"""
        ds = ATLAS.open_atlas_elevation(
            sample_files['elevation'],
            grid_file=sample_files['grid']
        )

        assert 'm2' in ds.data_vars
        assert 's2' in ds.data_vars
        assert 'constituents' in ds.attrs
        assert np.iscomplexobj(ds['m2'].values)

    def test_open_atlas_transport(self, sample_files):
        """Test opening ATLAS transport as xarray Dataset"""
        ds = ATLAS.open_atlas_transport(
            sample_files['transport'],
            grid_file=sample_files['grid']
        )

        assert 'u_m2' in ds.data_vars
        assert 'v_m2' in ds.data_vars
        assert 'u_s2' in ds.data_vars
        assert 'v_s2' in ds.data_vars

    def test_open_dataset_elevation(self, sample_files):
        """Test open_dataset for elevation"""
        ds = ATLAS.open_dataset(
            sample_files['elevation'],
            grid_file=sample_files['grid'],
            group='z'
        )

        assert 'm2' in ds.data_vars
        assert 's2' in ds.data_vars

    def test_open_dataset_u_component(self, sample_files):
        """Test open_dataset for U component"""
        ds = ATLAS.open_dataset(
            sample_files['transport'],
            grid_file=sample_files['grid'],
            group='u'
        )

        assert 'm2' in ds.data_vars
        assert 's2' in ds.data_vars
        assert ds.attrs['component'] == 'u'

    def test_open_dataset_v_component(self, sample_files):
        """Test open_dataset for V component"""
        ds = ATLAS.open_dataset(
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
            ATLAS.open_dataset(
                sample_files['elevation'],
                grid_file=sample_files['grid'],
                group='invalid'
            )


class TestATLASWithModelData:
    """Test ATLAS functions with actual model data (requires model files)"""

    @pytest.fixture
    def model_directory(self):
        """Get model directory from environment"""
        directory = os.environ.get('PYTMD_RESOURCE')
        if not directory:
            pytest.skip('PYTMD_RESOURCE environment variable not set')
        return directory

    def test_tpxo9_atlas_grid(self, model_directory):
        """Test reading TPXO9-atlas grid"""
        grid_file = os.path.join(model_directory, 'TPXO9_atlas_v5', 'grid_tpxo9_atlas_30_v5')

        if not os.path.exists(grid_file):
            pytest.skip('TPXO9 grid file not found')

        grid = ATLAS.read_atlas_grid(grid_file)

        assert len(grid['x']) > 0
        assert len(grid['y']) > 0


class TestAPIConsistency:
    """Test API consistency and exports"""

    def test_module_exports(self):
        """Test that all expected functions are exported"""
        assert hasattr(ATLAS, 'read_atlas_grid')
        assert hasattr(ATLAS, 'read_atlas_elevation')
        assert hasattr(ATLAS, 'read_atlas_transport')
        assert hasattr(ATLAS, 'open_atlas_grid')
        assert hasattr(ATLAS, 'open_atlas_elevation')
        assert hasattr(ATLAS, 'open_atlas_transport')
        assert hasattr(ATLAS, 'open_dataset')

    def test_io_module_exports(self):
        """Test that ATLAS functions are exported from io module"""
        from pyTMD_turbo import io

        assert hasattr(io, 'ATLAS')
        assert hasattr(io, 'open_atlas_grid')
        assert hasattr(io, 'open_atlas_elevation')
        assert hasattr(io, 'open_atlas_transport')
