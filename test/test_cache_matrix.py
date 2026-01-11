"""
Comprehensive cache behavior matrix tests for pyTMD_turbo

Tests all combinations of:
- Cache file existence (exists / not exists)
- Global cache enabled/disabled
- Per-model cache enabled/disabled
- Operations: save, load, clear

Expected behaviors:
- Cache creation
- Cache deletion
- Cache usage
- Cache ignored
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from pyTMD_turbo import cache


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def cache_state():
    """Reset cache state before and after each test"""
    # Save original state
    orig_global = cache._state._global_enabled
    orig_disabled = cache._state._disabled_models.copy()
    orig_temp = cache._state._temp_mode
    orig_known = cache._state._known_caches.copy()

    # Reset to clean state
    cache._state._global_enabled = True
    cache._state._disabled_models.clear()
    cache._state._temp_mode = False
    cache._state._known_caches.clear()

    yield cache._state

    # Restore original state
    cache._state._global_enabled = orig_global
    cache._state._disabled_models = orig_disabled
    cache._state._temp_mode = orig_temp
    cache._state._known_caches = orig_known


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_test_data():
    """Create test data for cache operations"""
    return {
        'x': np.linspace(-180, 180, 10),
        'y': np.linspace(-90, 90, 10),
        'amplitude': np.random.rand(10, 10),
        'phase': np.random.rand(10, 10),
    }


def create_cache_file(cache_path: Path, data: dict = None):
    """Helper to create a cache file"""
    if data is None:
        data = create_test_data()
    np.savez_compressed(cache_path, **data)
    return data


# =============================================================================
# Test Matrix: save_cache
# =============================================================================

class TestSaveCacheMatrix:
    """Test matrix for save_cache operation"""

    # -------------------------------------------------------------------------
    # Cache doesn't exist initially
    # -------------------------------------------------------------------------

    def test_save_no_cache_global_enabled_model_enabled(self, cache_state, temp_cache_dir):
        """
        Initial: No cache exists
        Global: Enabled
        Model: Enabled
        Expected: Cache CREATED
        """
        cache_path = temp_cache_dir / 'model.npz'
        data = create_test_data()

        assert not cache_path.exists(), "Precondition: cache should not exist"

        cache.save_cache('test_model', cache_path, data, [])

        assert cache_path.exists(), "Cache should be created"
        assert 'test_model' in cache._state._known_caches, "Model should be registered"

    def test_save_no_cache_global_enabled_model_disabled(self, cache_state, temp_cache_dir):
        """
        Initial: No cache exists
        Global: Enabled
        Model: Disabled
        Expected: Cache NOT CREATED
        """
        cache_path = temp_cache_dir / 'model.npz'
        data = create_test_data()
        cache._state._disabled_models.add('test_model')

        assert not cache_path.exists(), "Precondition: cache should not exist"

        cache.save_cache('test_model', cache_path, data, [])

        assert not cache_path.exists(), "Cache should NOT be created when model disabled"

    def test_save_no_cache_global_disabled_model_enabled(self, cache_state, temp_cache_dir):
        """
        Initial: No cache exists
        Global: Disabled
        Model: Enabled (but overridden)
        Expected: Cache NOT CREATED
        """
        cache_path = temp_cache_dir / 'model.npz'
        data = create_test_data()
        cache._state._global_enabled = False

        assert not cache_path.exists(), "Precondition: cache should not exist"

        cache.save_cache('test_model', cache_path, data, [])

        assert not cache_path.exists(), "Cache should NOT be created when globally disabled"

    def test_save_no_cache_global_disabled_model_disabled(self, cache_state, temp_cache_dir):
        """
        Initial: No cache exists
        Global: Disabled
        Model: Disabled
        Expected: Cache NOT CREATED
        """
        cache_path = temp_cache_dir / 'model.npz'
        data = create_test_data()
        cache._state._global_enabled = False
        cache._state._disabled_models.add('test_model')

        cache.save_cache('test_model', cache_path, data, [])

        assert not cache_path.exists(), "Cache should NOT be created"

    # -------------------------------------------------------------------------
    # Cache exists initially
    # -------------------------------------------------------------------------

    def test_save_cache_exists_global_enabled_model_enabled(self, cache_state, temp_cache_dir):
        """
        Initial: Cache exists
        Global: Enabled
        Model: Enabled
        Expected: Cache UPDATED (overwritten)
        """
        cache_path = temp_cache_dir / 'model.npz'
        old_data = create_cache_file(cache_path)
        old_mtime = cache_path.stat().st_mtime

        time.sleep(0.01)  # Ensure different mtime
        new_data = create_test_data()
        new_data['amplitude'] = np.ones((10, 10)) * 999  # Different data

        cache.save_cache('test_model', cache_path, new_data, [])

        assert cache_path.exists(), "Cache should still exist"
        new_mtime = cache_path.stat().st_mtime
        assert new_mtime > old_mtime, "Cache should be updated (newer mtime)"

        # Verify new data
        loaded = np.load(cache_path)
        assert loaded['amplitude'][0, 0] == 999, "Cache should contain new data"

    def test_save_cache_exists_global_enabled_model_disabled(self, cache_state, temp_cache_dir):
        """
        Initial: Cache exists
        Global: Enabled
        Model: Disabled
        Expected: Cache NOT UPDATED (old cache preserved)
        """
        cache_path = temp_cache_dir / 'model.npz'
        old_data = create_cache_file(cache_path)
        old_mtime = cache_path.stat().st_mtime

        cache._state._disabled_models.add('test_model')
        time.sleep(0.01)
        new_data = create_test_data()

        cache.save_cache('test_model', cache_path, new_data, [])

        new_mtime = cache_path.stat().st_mtime
        assert new_mtime == old_mtime, "Cache should NOT be updated when model disabled"


# =============================================================================
# Test Matrix: load_cache
# =============================================================================

class TestLoadCacheMatrix:
    """Test matrix for load_cache operation"""

    # -------------------------------------------------------------------------
    # Cache doesn't exist
    # -------------------------------------------------------------------------

    def test_load_no_cache_global_enabled_model_enabled(self, cache_state, temp_cache_dir):
        """
        Initial: No cache exists
        Global: Enabled
        Model: Enabled
        Expected: Returns None, cache NOT USED
        """
        cache_path = temp_cache_dir / 'model.npz'

        result = cache.load_cache('test_model', cache_path)

        assert result is None, "Should return None when cache doesn't exist"

    def test_load_no_cache_global_enabled_model_disabled(self, cache_state, temp_cache_dir):
        """
        Initial: No cache exists
        Global: Enabled
        Model: Disabled
        Expected: Returns None
        """
        cache_path = temp_cache_dir / 'model.npz'
        cache._state._disabled_models.add('test_model')

        result = cache.load_cache('test_model', cache_path)

        assert result is None, "Should return None"

    def test_load_no_cache_global_disabled_model_enabled(self, cache_state, temp_cache_dir):
        """
        Initial: No cache exists
        Global: Disabled
        Model: Enabled
        Expected: Returns None
        """
        cache_path = temp_cache_dir / 'model.npz'
        cache._state._global_enabled = False

        result = cache.load_cache('test_model', cache_path)

        assert result is None, "Should return None"

    def test_load_no_cache_global_disabled_model_disabled(self, cache_state, temp_cache_dir):
        """
        Initial: No cache exists
        Global: Disabled
        Model: Disabled
        Expected: Returns None
        """
        cache_path = temp_cache_dir / 'model.npz'
        cache._state._global_enabled = False
        cache._state._disabled_models.add('test_model')

        result = cache.load_cache('test_model', cache_path)

        assert result is None, "Should return None"

    # -------------------------------------------------------------------------
    # Cache exists
    # -------------------------------------------------------------------------

    def test_load_cache_exists_global_enabled_model_enabled(self, cache_state, temp_cache_dir):
        """
        Initial: Cache exists
        Global: Enabled
        Model: Enabled
        Expected: Cache USED, returns data
        """
        cache_path = temp_cache_dir / 'model.npz'
        original_data = create_cache_file(cache_path)

        result = cache.load_cache('test_model', cache_path)

        assert result is not None, "Should return cached data"
        np.testing.assert_array_equal(result['x'], original_data['x'])
        np.testing.assert_array_equal(result['amplitude'], original_data['amplitude'])
        assert 'test_model' in cache._state._known_caches, "Model should be registered"

    def test_load_cache_exists_global_enabled_model_disabled(self, cache_state, temp_cache_dir):
        """
        Initial: Cache exists
        Global: Enabled
        Model: Disabled
        Expected: Cache NOT USED, returns None
        """
        cache_path = temp_cache_dir / 'model.npz'
        create_cache_file(cache_path)
        cache._state._disabled_models.add('test_model')

        result = cache.load_cache('test_model', cache_path)

        assert result is None, "Should return None when model disabled"

    def test_load_cache_exists_global_disabled_model_enabled(self, cache_state, temp_cache_dir):
        """
        Initial: Cache exists
        Global: Disabled
        Model: Enabled (but overridden)
        Expected: Cache NOT USED, returns None
        """
        cache_path = temp_cache_dir / 'model.npz'
        create_cache_file(cache_path)
        cache._state._global_enabled = False

        result = cache.load_cache('test_model', cache_path)

        assert result is None, "Should return None when globally disabled"

    def test_load_cache_exists_global_disabled_model_disabled(self, cache_state, temp_cache_dir):
        """
        Initial: Cache exists
        Global: Disabled
        Model: Disabled
        Expected: Cache NOT USED, returns None
        """
        cache_path = temp_cache_dir / 'model.npz'
        create_cache_file(cache_path)
        cache._state._global_enabled = False
        cache._state._disabled_models.add('test_model')

        result = cache.load_cache('test_model', cache_path)

        assert result is None, "Should return None"


# =============================================================================
# Test Matrix: clear_cache
# =============================================================================

class TestClearCacheMatrix:
    """Test matrix for clear_cache operation"""

    def test_clear_cache_exists_in_known(self, cache_state, temp_cache_dir):
        """
        Initial: Cache exists and is registered
        Expected: Cache DELETED
        """
        cache_path = temp_cache_dir / 'model.npz'
        create_cache_file(cache_path)
        cache._state._known_caches['test_model'] = cache_path

        result = cache.clear_cache('test_model')

        assert result is True, "Should return True on success"
        assert not cache_path.exists(), "Cache file should be deleted"
        assert 'test_model' not in cache._state._known_caches, "Model should be unregistered"

    def test_clear_cache_exists_not_in_known(self, cache_state, temp_cache_dir):
        """
        Initial: Cache exists but NOT registered
        Expected: Returns False (not found in known caches)
        """
        cache_path = temp_cache_dir / 'model.npz'
        create_cache_file(cache_path)
        # Note: NOT registering in _known_caches

        result = cache.clear_cache('test_model')

        assert result is False, "Should return False when not in known caches"
        assert cache_path.exists(), "Cache file should still exist"

    def test_clear_cache_not_exists(self, cache_state, temp_cache_dir):
        """
        Initial: Cache doesn't exist
        Expected: Returns False
        """
        result = cache.clear_cache('nonexistent_model')

        assert result is False, "Should return False when cache doesn't exist"

    def test_clear_all_cache(self, cache_state, temp_cache_dir):
        """
        Initial: Multiple caches exist
        Expected: All caches DELETED
        """
        paths = []
        for i in range(3):
            cache_path = temp_cache_dir / f'model{i}.npz'
            create_cache_file(cache_path)
            cache._state._known_caches[f'model{i}'] = cache_path
            paths.append(cache_path)

        result = cache.clear_all_cache()

        assert result == 3, "Should return count of deleted caches"
        for p in paths:
            assert not p.exists(), f"Cache {p} should be deleted"
        assert len(cache._state._known_caches) == 0, "All models should be unregistered"


# =============================================================================
# Test Matrix: Cache Staleness
# =============================================================================

class TestCacheStalenessMatrix:
    """Test matrix for cache staleness detection"""

    def test_load_stale_cache_source_newer(self, cache_state, temp_cache_dir):
        """
        Initial: Cache exists, source file is NEWER than cache
        Expected: Cache INVALIDATED, returns None
        """
        cache_path = temp_cache_dir / 'model.npz'
        source_file = temp_cache_dir / 'source.nc'

        # Create cache first
        data = create_test_data()
        data['_metadata_json'] = np.array([str({'source_mtime': 1000.0})])
        np.savez_compressed(cache_path, **data)

        # Create source file with newer mtime
        source_file.touch()
        time.sleep(0.01)
        source_file.touch()  # Update mtime to be newer

        result = cache.load_cache('test_model', cache_path, source_files=[source_file])

        assert result is None, "Should return None when source is newer (cache stale)"

    def test_load_fresh_cache_source_older(self, cache_state, temp_cache_dir):
        """
        Initial: Cache exists, source file is OLDER than cache
        Expected: Cache USED
        """
        cache_path = temp_cache_dir / 'model.npz'
        source_file = temp_cache_dir / 'source.nc'

        # Create source file first
        source_file.touch()
        source_mtime = source_file.stat().st_mtime

        time.sleep(0.01)

        # Create cache with metadata indicating it's newer
        data = create_test_data()
        data['_metadata_json'] = np.array([str({'source_mtime': source_mtime + 100})])
        np.savez_compressed(cache_path, **data)

        result = cache.load_cache('test_model', cache_path, source_files=[source_file])

        assert result is not None, "Should return cached data when cache is fresh"


# =============================================================================
# Test Matrix: Temp Cache Mode
# =============================================================================

class TestTempCacheModeMatrix:
    """Test matrix for temporary cache mode"""

    def test_save_temp_mode_enabled(self, cache_state, temp_cache_dir):
        """
        Temp Mode: Enabled
        Expected: Cache created AND registered for cleanup
        """
        cache_path = temp_cache_dir / 'model.npz'
        cache._state._temp_mode = True
        data = create_test_data()

        cache.save_cache('test_model', cache_path, data, [])

        assert cache_path.exists(), "Cache should be created"
        assert cache_path in cache._state._temp_cache_files, "Cache should be registered for cleanup"

    def test_save_temp_mode_disabled(self, cache_state, temp_cache_dir):
        """
        Temp Mode: Disabled
        Expected: Cache created, NOT registered for cleanup
        """
        cache_path = temp_cache_dir / 'model.npz'
        cache._state._temp_mode = False
        data = create_test_data()

        cache.save_cache('test_model', cache_path, data, [])

        assert cache_path.exists(), "Cache should be created"
        assert cache_path not in cache._state._temp_cache_files, "Cache should NOT be registered for cleanup"


# =============================================================================
# Test Matrix: Context Managers
# =============================================================================

class TestContextManagerMatrix:
    """Test matrix for context manager behavior"""

    def test_cache_disabled_context_save(self, cache_state, temp_cache_dir):
        """
        Context: cache_disabled()
        Operation: save
        Expected: Cache NOT created within context
        """
        cache_path = temp_cache_dir / 'model.npz'
        data = create_test_data()

        with cache.cache_disabled():
            cache.save_cache('test_model', cache_path, data, [])
            assert not cache_path.exists(), "Cache should NOT be created within disabled context"

        # After context, save should work
        cache.save_cache('test_model', cache_path, data, [])
        assert cache_path.exists(), "Cache should be created after context"

    def test_cache_disabled_context_load(self, cache_state, temp_cache_dir):
        """
        Context: cache_disabled()
        Operation: load
        Expected: Cache NOT used within context
        """
        cache_path = temp_cache_dir / 'model.npz'
        create_cache_file(cache_path)

        with cache.cache_disabled():
            result = cache.load_cache('test_model', cache_path)
            assert result is None, "Cache should NOT be used within disabled context"

        # After context, load should work
        result = cache.load_cache('test_model', cache_path)
        assert result is not None, "Cache should be used after context"

    def test_cache_disabled_for_model_context(self, cache_state, temp_cache_dir):
        """
        Context: cache_disabled_for('model_a')
        Expected: model_a disabled, model_b enabled
        """
        cache_path_a = temp_cache_dir / 'model_a.npz'
        cache_path_b = temp_cache_dir / 'model_b.npz'
        data = create_test_data()

        with cache.cache_disabled_for('model_a'):
            cache.save_cache('model_a', cache_path_a, data, [])
            cache.save_cache('model_b', cache_path_b, data, [])

            assert not cache_path_a.exists(), "model_a cache should NOT be created"
            assert cache_path_b.exists(), "model_b cache SHOULD be created"


# =============================================================================
# Test Matrix: Error Conditions
# =============================================================================

class TestErrorConditionsMatrix:
    """Test matrix for error handling"""

    def test_load_corrupted_cache(self, cache_state, temp_cache_dir):
        """
        Initial: Corrupted cache file
        Expected: Returns None with warning
        """
        cache_path = temp_cache_dir / 'model.npz'
        cache_path.write_text("not a valid npz file")

        with pytest.warns(RuntimeWarning, match="Failed to load cache"):
            result = cache.load_cache('test_model', cache_path)

        assert result is None, "Should return None for corrupted cache"

    def test_save_failure_emits_warning(self, cache_state, temp_cache_dir):
        """
        Initial: Save operation fails
        Expected: Warning emitted, no crash

        Note: Uses mocking to reliably test error handling across all platforms.
        """
        from unittest.mock import patch

        cache_path = temp_cache_dir / 'model.npz'
        data = create_test_data()

        # Mock np.savez_compressed to raise an exception
        with patch('numpy.savez_compressed', side_effect=PermissionError("Permission denied")):
            with pytest.warns(RuntimeWarning, match="Failed to save cache"):
                cache.save_cache('test_model', cache_path, data, [])


# =============================================================================
# Integration Test: Full Workflow
# =============================================================================

class TestCacheWorkflowIntegration:
    """Integration tests for complete cache workflows"""

    def test_full_cache_lifecycle(self, cache_state, temp_cache_dir):
        """
        Test complete cache lifecycle:
        1. No cache -> save -> cache exists
        2. Load cache -> data returned
        3. Clear cache -> cache deleted
        4. Load cache -> None returned
        """
        cache_path = temp_cache_dir / 'lifecycle.npz'
        data = create_test_data()

        # Step 1: Save cache
        assert not cache_path.exists(), "Initial: no cache"
        cache.save_cache('lifecycle', cache_path, data, [])
        assert cache_path.exists(), "After save: cache exists"

        # Step 2: Load cache
        loaded = cache.load_cache('lifecycle', cache_path)
        assert loaded is not None, "Load should return data"
        np.testing.assert_array_equal(loaded['amplitude'], data['amplitude'])

        # Step 3: Clear cache
        result = cache.clear_cache('lifecycle')
        assert result is True, "Clear should succeed"
        assert not cache_path.exists(), "After clear: no cache"

        # Step 4: Load again
        loaded = cache.load_cache('lifecycle', cache_path)
        assert loaded is None, "Load should return None after clear"

    def test_enable_disable_toggle(self, cache_state, temp_cache_dir):
        """
        Test enable/disable toggle behavior
        """
        cache_path = temp_cache_dir / 'toggle.npz'
        data = create_test_data()

        # Enabled: save works
        cache.enable_cache()
        cache.save_cache('toggle', cache_path, data, [])
        assert cache_path.exists(), "Save should work when enabled"

        # Delete for next test
        cache_path.unlink()
        cache._state._known_caches.clear()

        # Disabled: save doesn't work
        cache.disable_cache()
        cache.save_cache('toggle', cache_path, data, [])
        assert not cache_path.exists(), "Save should NOT work when disabled"

        # Re-enable: save works again
        cache.enable_cache()
        cache.save_cache('toggle', cache_path, data, [])
        assert cache_path.exists(), "Save should work when re-enabled"
