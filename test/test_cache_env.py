"""
Environment variable tests for pyTMD_turbo cache system

Tests all environment variables:
- PYTMD_TURBO_DISABLED: Disable cache globally
- PYTMD_TURBO_DISABLED_MODELS: Comma-separated list of models to disable
- PYTMD_TURBO_TEMP_CACHE: Enable temporary cache mode
- PYTMD_TURBO_CACHE_DIR: Custom cache directory
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# =============================================================================
# Test: PYTMD_TURBO_DISABLED
# =============================================================================

class TestEnvDisabled:
    """Test PYTMD_TURBO_DISABLED environment variable"""

    def test_disabled_with_1(self):
        """PYTMD_TURBO_DISABLED=1 should disable cache globally"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED': '1'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.global_enabled is False

    def test_disabled_with_true(self):
        """PYTMD_TURBO_DISABLED=true should disable cache globally"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED': 'true'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.global_enabled is False

    def test_disabled_with_yes(self):
        """PYTMD_TURBO_DISABLED=yes should disable cache globally"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED': 'yes'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.global_enabled is False

    def test_disabled_with_TRUE_uppercase(self):
        """PYTMD_TURBO_DISABLED=TRUE (uppercase) should disable cache"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED': 'TRUE'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.global_enabled is False

    def test_disabled_with_empty(self):
        """PYTMD_TURBO_DISABLED='' should NOT disable cache"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED': ''}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.global_enabled is True

    def test_disabled_with_0(self):
        """PYTMD_TURBO_DISABLED=0 should NOT disable cache"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED': '0'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.global_enabled is True

    def test_disabled_with_false(self):
        """PYTMD_TURBO_DISABLED=false should NOT disable cache"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED': 'false'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.global_enabled is True

    def test_disabled_not_set(self):
        """PYTMD_TURBO_DISABLED not set should leave cache enabled"""
        env = os.environ.copy()
        env.pop('PYTMD_TURBO_DISABLED', None)
        with patch.dict(os.environ, env, clear=True):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.global_enabled is True


# =============================================================================
# Test: PYTMD_TURBO_DISABLED_MODELS
# =============================================================================

class TestEnvDisabledModels:
    """Test PYTMD_TURBO_DISABLED_MODELS environment variable"""

    def test_single_model(self):
        """PYTMD_TURBO_DISABLED_MODELS=GOT5.5 should disable GOT5.5"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED_MODELS': 'GOT5.5'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.is_model_disabled('GOT5.5') is True
            assert state.is_model_disabled('FES2014') is False

    def test_multiple_models(self):
        """PYTMD_TURBO_DISABLED_MODELS=GOT5.5,FES2014 should disable both"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED_MODELS': 'GOT5.5,FES2014'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.is_model_disabled('GOT5.5') is True
            assert state.is_model_disabled('FES2014') is True
            assert state.is_model_disabled('TPXO9') is False

    def test_models_with_spaces(self):
        """PYTMD_TURBO_DISABLED_MODELS='GOT5.5 , FES2014' should handle spaces"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED_MODELS': 'GOT5.5 , FES2014 '}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.is_model_disabled('GOT5.5') is True
            assert state.is_model_disabled('FES2014') is True

    def test_empty_models(self):
        """PYTMD_TURBO_DISABLED_MODELS='' should not disable any model"""
        with patch.dict(os.environ, {'PYTMD_TURBO_DISABLED_MODELS': ''}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.is_model_disabled('GOT5.5') is False
            assert state.is_model_disabled('FES2014') is False

    def test_models_not_set(self):
        """PYTMD_TURBO_DISABLED_MODELS not set should not disable any model"""
        env = os.environ.copy()
        env.pop('PYTMD_TURBO_DISABLED_MODELS', None)
        with patch.dict(os.environ, env, clear=True):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.is_model_disabled('GOT5.5') is False


# =============================================================================
# Test: PYTMD_TURBO_TEMP_CACHE
# =============================================================================

class TestEnvTempCache:
    """Test PYTMD_TURBO_TEMP_CACHE environment variable"""

    def test_temp_cache_with_1(self):
        """PYTMD_TURBO_TEMP_CACHE=1 should enable temp mode"""
        with patch.dict(os.environ, {'PYTMD_TURBO_TEMP_CACHE': '1'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.temp_mode is True

    def test_temp_cache_with_true(self):
        """PYTMD_TURBO_TEMP_CACHE=true should enable temp mode"""
        with patch.dict(os.environ, {'PYTMD_TURBO_TEMP_CACHE': 'true'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.temp_mode is True

    def test_temp_cache_with_yes(self):
        """PYTMD_TURBO_TEMP_CACHE=yes should enable temp mode"""
        with patch.dict(os.environ, {'PYTMD_TURBO_TEMP_CACHE': 'yes'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.temp_mode is True

    def test_temp_cache_with_empty(self):
        """PYTMD_TURBO_TEMP_CACHE='' should NOT enable temp mode"""
        with patch.dict(os.environ, {'PYTMD_TURBO_TEMP_CACHE': ''}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.temp_mode is False

    def test_temp_cache_with_0(self):
        """PYTMD_TURBO_TEMP_CACHE=0 should NOT enable temp mode"""
        with patch.dict(os.environ, {'PYTMD_TURBO_TEMP_CACHE': '0'}):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.temp_mode is False

    def test_temp_cache_not_set(self):
        """PYTMD_TURBO_TEMP_CACHE not set should not enable temp mode"""
        env = os.environ.copy()
        env.pop('PYTMD_TURBO_TEMP_CACHE', None)
        with patch.dict(os.environ, env, clear=True):
            from pyTMD_turbo.cache import _CacheState
            state = _CacheState()
            assert state.temp_mode is False


# =============================================================================
# Test: PYTMD_TURBO_CACHE_DIR
# =============================================================================

class TestEnvCacheDir:
    """Test PYTMD_TURBO_CACHE_DIR environment variable"""

    def test_cache_dir_set(self):
        """PYTMD_TURBO_CACHE_DIR should return custom directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'PYTMD_TURBO_CACHE_DIR': tmpdir}):
                from pyTMD_turbo.cache import get_cache_dir
                result = get_cache_dir()
                assert result == Path(tmpdir)

    def test_cache_dir_not_set(self):
        """PYTMD_TURBO_CACHE_DIR not set should return None"""
        env = os.environ.copy()
        env.pop('PYTMD_TURBO_CACHE_DIR', None)
        with patch.dict(os.environ, env, clear=True):
            from pyTMD_turbo.cache import get_cache_dir
            result = get_cache_dir()
            assert result is None

    def test_cache_path_with_custom_dir(self):
        """get_cache_path should use custom directory when set"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'PYTMD_TURBO_CACHE_DIR': tmpdir}):
                from pyTMD_turbo.cache import get_cache_path
                result = get_cache_path('GOT5.5', '/some/model/dir')
                assert result == Path(tmpdir) / 'GOT5.5.npz'

    def test_cache_path_without_custom_dir(self):
        """get_cache_path should use model directory when no custom dir"""
        env = os.environ.copy()
        env.pop('PYTMD_TURBO_CACHE_DIR', None)
        with patch.dict(os.environ, env, clear=True):
            from pyTMD_turbo.cache import get_cache_path
            result = get_cache_path('GOT5.5', '/some/model/dir')
            assert result == Path('/some/model/dir/.pytmd_turbo_cache.npz')


# =============================================================================
# Integration Test: Subprocess with environment variables
# =============================================================================

class TestEnvIntegration:
    """Integration tests using subprocess to verify environment variables"""

    def test_subprocess_disabled(self):
        """Verify PYTMD_TURBO_DISABLED works in subprocess"""
        code = """
import sys
from pyTMD_turbo import cache
result = cache.is_cache_enabled()
sys.exit(0 if result == False else 1)
"""
        env = os.environ.copy()
        env['PYTMD_TURBO_DISABLED'] = '1'
        result = subprocess.run(
            [sys.executable, '-c', code],
            env=env,
            capture_output=True,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}, stderr: {result.stderr}"

    def test_subprocess_disabled_models(self):
        """Verify PYTMD_TURBO_DISABLED_MODELS works in subprocess"""
        code = """
import sys
from pyTMD_turbo import cache
got55_disabled = not cache.is_cache_enabled_for('GOT5.5')
fes_enabled = cache.is_cache_enabled_for('FES2014')
sys.exit(0 if got55_disabled and fes_enabled else 1)
"""
        env = os.environ.copy()
        env['PYTMD_TURBO_DISABLED_MODELS'] = 'GOT5.5'
        result = subprocess.run(
            [sys.executable, '-c', code],
            env=env,
            capture_output=True,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}, stderr: {result.stderr}"

    def test_subprocess_temp_cache(self):
        """Verify PYTMD_TURBO_TEMP_CACHE works in subprocess"""
        code = """
import sys
from pyTMD_turbo.cache import _state
result = _state.temp_mode
sys.exit(0 if result == True else 1)
"""
        env = os.environ.copy()
        env['PYTMD_TURBO_TEMP_CACHE'] = 'true'
        result = subprocess.run(
            [sys.executable, '-c', code],
            env=env,
            capture_output=True,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}, stderr: {result.stderr}"

    def test_subprocess_cache_dir(self):
        """Verify PYTMD_TURBO_CACHE_DIR works in subprocess"""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = f"""
import sys
from pyTMD_turbo.cache import get_cache_dir
result = get_cache_dir()
expected = "{tmpdir}"
sys.exit(0 if str(result) == expected else 1)
"""
            env = os.environ.copy()
            env['PYTMD_TURBO_CACHE_DIR'] = tmpdir
            result = subprocess.run(
                [sys.executable, '-c', code],
                env=env,
                capture_output=True,
            )
            assert result.returncode == 0, f"stdout: {result.stdout}, stderr: {result.stderr}"


# =============================================================================
# Test: Combined environment variables
# =============================================================================

class TestEnvCombined:
    """Test combinations of environment variables"""

    def test_disabled_overrides_disabled_models(self):
        """PYTMD_TURBO_DISABLED=1 should override PYTMD_TURBO_DISABLED_MODELS"""
        code = """
import sys
from pyTMD_turbo import cache
# Even if GOT5.5 is not in disabled models, global disable should take precedence
result = cache.is_cache_enabled_for('FES2014')
sys.exit(0 if result == False else 1)
"""
        env = os.environ.copy()
        env['PYTMD_TURBO_DISABLED'] = '1'
        env['PYTMD_TURBO_DISABLED_MODELS'] = 'GOT5.5'
        result = subprocess.run(
            [sys.executable, '-c', code],
            env=env,
            capture_output=True,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}, stderr: {result.stderr}"

    def test_temp_cache_with_custom_dir(self):
        """PYTMD_TURBO_TEMP_CACHE and PYTMD_TURBO_CACHE_DIR can be used together"""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = f"""
import sys
from pyTMD_turbo.cache import _state, get_cache_dir
temp_mode = _state.temp_mode
cache_dir = get_cache_dir()
expected_dir = "{tmpdir}"
sys.exit(0 if temp_mode and str(cache_dir) == expected_dir else 1)
"""
            env = os.environ.copy()
            env['PYTMD_TURBO_TEMP_CACHE'] = '1'
            env['PYTMD_TURBO_CACHE_DIR'] = tmpdir
            result = subprocess.run(
                [sys.executable, '-c', code],
                env=env,
                capture_output=True,
            )
            assert result.returncode == 0, f"stdout: {result.stdout}, stderr: {result.stderr}"
