"""
Tests for pyTMD_turbo cache control functionality

Tests for the cache module that provides zero-config caching with
environment variable and programmatic control.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import pyTMD_turbo
from pyTMD_turbo import cache


class TestCacheEnableDisable:
    """Tests for cache enable/disable functions"""

    def setup_method(self):
        """Reset cache state before each test"""
        cache._state._global_enabled = True
        cache._state._disabled_models.clear()
        cache._state._temp_mode = False

    def test_is_cache_enabled_default(self):
        """Cache should be enabled by default"""
        assert pyTMD_turbo.is_cache_enabled() is True

    def test_disable_cache(self):
        """Test disabling all cache"""
        pyTMD_turbo.disable_cache()
        assert pyTMD_turbo.is_cache_enabled() is False

    def test_enable_cache(self):
        """Test re-enabling cache"""
        pyTMD_turbo.disable_cache()
        pyTMD_turbo.enable_cache()
        assert pyTMD_turbo.is_cache_enabled() is True

    def test_disable_cache_for_model(self):
        """Test disabling cache for specific model"""
        pyTMD_turbo.disable_cache_for('GOT5.5')
        assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is False
        assert pyTMD_turbo.is_cache_enabled_for('FES2014') is True

    def test_disable_cache_for_multiple_models(self):
        """Test disabling cache for multiple models"""
        pyTMD_turbo.disable_cache_for('GOT5.5', 'FES2014')
        assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is False
        assert pyTMD_turbo.is_cache_enabled_for('FES2014') is False
        assert pyTMD_turbo.is_cache_enabled_for('TPXO9') is True

    def test_enable_cache_for_model(self):
        """Test re-enabling cache for specific model"""
        pyTMD_turbo.disable_cache_for('GOT5.5')
        pyTMD_turbo.enable_cache_for('GOT5.5')
        assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is True

    def test_global_disable_overrides_model(self):
        """Global disable should override model-specific settings"""
        pyTMD_turbo.disable_cache()
        assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is False


class TestCacheContextManagers:
    """Tests for cache context managers"""

    def setup_method(self):
        """Reset cache state before each test"""
        cache._state._global_enabled = True
        cache._state._disabled_models.clear()

    def test_cache_disabled_context(self):
        """Test cache_disabled context manager"""
        assert pyTMD_turbo.is_cache_enabled() is True

        with pyTMD_turbo.cache_disabled():
            assert pyTMD_turbo.is_cache_enabled() is False

        assert pyTMD_turbo.is_cache_enabled() is True

    def test_cache_disabled_for_context(self):
        """Test cache_disabled_for context manager"""
        assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is True

        with pyTMD_turbo.cache_disabled_for('GOT5.5'):
            assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is False
            assert pyTMD_turbo.is_cache_enabled_for('FES2014') is True

        assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is True

    def test_cache_disabled_for_multiple(self):
        """Test cache_disabled_for with multiple models"""
        with pyTMD_turbo.cache_disabled_for('GOT5.5', 'FES2014'):
            assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is False
            assert pyTMD_turbo.is_cache_enabled_for('FES2014') is False

        assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is True
        assert pyTMD_turbo.is_cache_enabled_for('FES2014') is True

    def test_nested_context_managers(self):
        """Test nested context managers"""
        with pyTMD_turbo.cache_disabled_for('GOT5.5'):
            assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is False

            with pyTMD_turbo.cache_disabled():
                assert pyTMD_turbo.is_cache_enabled() is False

            # After inner context, global should be restored
            assert pyTMD_turbo.is_cache_enabled() is True
            # But GOT5.5 should still be disabled
            assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is False

        assert pyTMD_turbo.is_cache_enabled_for('GOT5.5') is True


class TestTempCache:
    """Tests for temporary cache mode"""

    def setup_method(self):
        """Reset cache state before each test"""
        cache._state._temp_mode = False
        cache._state._temp_cache_files.clear()

    def test_enable_temp_cache(self):
        """Test enabling temp cache mode"""
        pyTMD_turbo.enable_temp_cache()
        assert cache._state._temp_mode is True

    def test_disable_temp_cache(self):
        """Test disabling temp cache mode"""
        pyTMD_turbo.enable_temp_cache()
        pyTMD_turbo.disable_temp_cache()
        assert cache._state._temp_mode is False


class TestCachePath:
    """Tests for cache path functions"""

    def test_get_cache_path_default(self):
        """Test default cache path (same directory as model)"""
        model_dir = Path('/path/to/models/GOT5.5')
        cache_path = cache.get_cache_path('GOT5.5', model_dir)

        assert cache_path == model_dir / '.pytmd_turbo_cache.npz'

    def test_get_cache_path_custom_dir(self):
        """Test custom cache directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['PYTMD_TURBO_CACHE_DIR'] = tmpdir
            try:
                cache_path = cache.get_cache_path('GOT5.5', '/path/to/models')
                assert cache_path == Path(tmpdir) / 'GOT5.5.npz'
            finally:
                del os.environ['PYTMD_TURBO_CACHE_DIR']


class TestCacheOperations:
    """Tests for cache save/load operations"""

    def setup_method(self):
        """Reset cache state before each test"""
        cache._state._global_enabled = True
        cache._state._disabled_models.clear()
        cache._state._known_caches.clear()

    def test_save_and_load_cache(self):
        """Test saving and loading cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'test.npz'
            data = {
                'amplitude': np.array([1.0, 2.0, 3.0]),
                'phase': np.array([0.1, 0.2, 0.3]),
            }

            # Save
            cache.save_cache('test_model', cache_path, data, [])

            # Load
            loaded = cache.load_cache('test_model', cache_path)

            assert loaded is not None
            np.testing.assert_array_equal(loaded['amplitude'], data['amplitude'])
            np.testing.assert_array_equal(loaded['phase'], data['phase'])

    def test_load_nonexistent_cache(self):
        """Test loading non-existent cache returns None"""
        cache_path = Path('/nonexistent/path/cache.npz')
        loaded = cache.load_cache('test_model', cache_path)
        assert loaded is None

    def test_load_cache_when_disabled(self):
        """Test that load returns None when cache is disabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'test.npz'
            data = {'test': np.array([1, 2, 3])}

            cache.save_cache('test_model', cache_path, data, [])

            pyTMD_turbo.disable_cache_for('test_model')
            loaded = cache.load_cache('test_model', cache_path)
            assert loaded is None

    def test_clear_cache(self):
        """Test clearing a specific cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'test.npz'
            data = {'test': np.array([1, 2, 3])}

            cache.save_cache('test_model', cache_path, data, [])
            assert cache_path.exists()

            result = cache.clear_cache('test_model')
            assert result is True
            assert not cache_path.exists()

    def test_clear_all_cache(self):
        """Test clearing all caches"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple caches
            for i in range(3):
                cache_path = Path(tmpdir) / f'test{i}.npz'
                data = {'test': np.array([i])}
                cache.save_cache(f'model{i}', cache_path, data, [])

            deleted = cache.clear_all_cache()
            assert deleted == 3


class TestCacheStatus:
    """Tests for cache status functions"""

    def setup_method(self):
        """Reset cache state before each test"""
        cache._state._known_caches.clear()

    def test_get_cache_info_empty(self):
        """Test get_cache_info with no caches"""
        info = cache.get_cache_info()
        assert info == []

    def test_get_cache_info_with_cache(self):
        """Test get_cache_info with a cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'test.npz'
            data = {'test': np.array([1, 2, 3])}
            cache.save_cache('test_model', cache_path, data, [])

            info = cache.get_cache_info()
            assert len(info) == 1
            assert info[0]['model'] == 'test_model'
            assert info[0]['exists'] is True
            assert info[0]['size'] is not None

    def test_format_size(self):
        """Test size formatting"""
        assert cache._format_size(None) == '-'
        assert cache._format_size(100) == '100.0 B'
        assert cache._format_size(1024) == '1.0 KB'
        assert cache._format_size(1024 * 1024) == '1.0 MB'
        assert cache._format_size(1024 * 1024 * 1024) == '1.0 GB'


class TestEnvironmentVariables:
    """Tests for environment variable handling"""

    def test_disabled_env_var(self):
        """Test PYTMD_TURBO_DISABLED environment variable"""
        os.environ['PYTMD_TURBO_DISABLED'] = '1'
        try:
            # Create new state to pick up env var
            state = cache._CacheState()
            assert state.global_enabled is False
        finally:
            del os.environ['PYTMD_TURBO_DISABLED']

    def test_disabled_models_env_var(self):
        """Test PYTMD_TURBO_DISABLED_MODELS environment variable"""
        os.environ['PYTMD_TURBO_DISABLED_MODELS'] = 'GOT5.5,FES2014'
        try:
            state = cache._CacheState()
            assert state.is_model_disabled('GOT5.5') is True
            assert state.is_model_disabled('FES2014') is True
            assert state.is_model_disabled('TPXO9') is False
        finally:
            del os.environ['PYTMD_TURBO_DISABLED_MODELS']

    def test_temp_cache_env_var(self):
        """Test PYTMD_TURBO_TEMP_CACHE environment variable"""
        os.environ['PYTMD_TURBO_TEMP_CACHE'] = '1'
        try:
            state = cache._CacheState()
            assert state.temp_mode is True
        finally:
            del os.environ['PYTMD_TURBO_TEMP_CACHE']


class TestModuleExports:
    """Tests for module exports"""

    def test_cache_functions_exported(self):
        """Test that all cache functions are exported from main module"""
        assert hasattr(pyTMD_turbo, 'enable_cache')
        assert hasattr(pyTMD_turbo, 'disable_cache')
        assert hasattr(pyTMD_turbo, 'enable_cache_for')
        assert hasattr(pyTMD_turbo, 'disable_cache_for')
        assert hasattr(pyTMD_turbo, 'is_cache_enabled')
        assert hasattr(pyTMD_turbo, 'is_cache_enabled_for')
        assert hasattr(pyTMD_turbo, 'enable_temp_cache')
        assert hasattr(pyTMD_turbo, 'disable_temp_cache')
        assert hasattr(pyTMD_turbo, 'cache_disabled')
        assert hasattr(pyTMD_turbo, 'cache_disabled_for')
        assert hasattr(pyTMD_turbo, 'rebuild_cache')
        assert hasattr(pyTMD_turbo, 'rebuild_all_cache')
        assert hasattr(pyTMD_turbo, 'clear_cache')
        assert hasattr(pyTMD_turbo, 'clear_all_cache')
        assert hasattr(pyTMD_turbo, 'show_cache_status')
        assert hasattr(pyTMD_turbo, 'get_cache_info')

    def test_functions_callable(self):
        """Test that all cache functions are callable"""
        assert callable(pyTMD_turbo.enable_cache)
        assert callable(pyTMD_turbo.disable_cache)
        assert callable(pyTMD_turbo.cache_disabled)
        assert callable(pyTMD_turbo.show_cache_status)
