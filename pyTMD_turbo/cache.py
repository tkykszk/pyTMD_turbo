"""
pyTMD_turbo.cache - Automatic caching for tidal model data

Zero-config caching: just import pyTMD_turbo to enable automatic caching.

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley

This software is licensed under the MIT License.
"""

import atexit
import os
import threading
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = [
    # Context managers
    'cache_disabled',
    'cache_disabled_for',
    'clear_all_cache',
    'clear_cache',
    'disable_cache',
    'disable_cache_for',
    'disable_temp_cache',
    # Enable/disable
    'enable_cache',
    'enable_cache_for',
    # Temp cache
    'enable_temp_cache',
    'get_cache_info',
    'get_cache_path',
    # Low-level
    'is_cache_enabled',
    'is_cache_enabled_for',
    'rebuild_all_cache',
    # Cache operations
    'rebuild_cache',
    # Status
    'show_cache_status',
]


# =============================================================================
# Global State
# =============================================================================

class _CacheState:
    """Thread-safe cache state manager."""

    def __init__(self):
        self._lock = threading.Lock()
        self._global_enabled = True
        self._disabled_models: set[str] = set()
        self._temp_mode = False
        self._temp_cache_files: set[Path] = set()
        self._known_caches: dict[str, Path] = {}  # model_name -> cache_path

        # Read environment variables
        self._init_from_env()

        # Register cleanup on exit
        atexit.register(self._cleanup_temp_caches)

    def _init_from_env(self):
        """Initialize state from environment variables."""
        # PYTMD_TURBO_DISABLED
        disabled = os.environ.get('PYTMD_TURBO_DISABLED', '').lower()
        if disabled in ('1', 'true', 'yes'):
            self._global_enabled = False

        # PYTMD_TURBO_DISABLED_MODELS
        disabled_models = os.environ.get('PYTMD_TURBO_DISABLED_MODELS', '')
        if disabled_models:
            self._disabled_models = set(m.strip() for m in disabled_models.split(',') if m.strip())

        # PYTMD_TURBO_TEMP_CACHE
        temp_cache = os.environ.get('PYTMD_TURBO_TEMP_CACHE', '').lower()
        if temp_cache in ('1', 'true', 'yes'):
            self._temp_mode = True

    def _cleanup_temp_caches(self):
        """Clean up temporary cache files on exit."""
        if not self._temp_cache_files:
            return

        for cache_path in self._temp_cache_files:
            try:
                if cache_path.exists():
                    cache_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors

    @property
    def global_enabled(self) -> bool:
        with self._lock:
            return self._global_enabled

    @global_enabled.setter
    def global_enabled(self, value: bool):
        with self._lock:
            self._global_enabled = value

    def is_model_disabled(self, model_name: str) -> bool:
        with self._lock:
            return model_name in self._disabled_models

    def disable_model(self, model_name: str):
        with self._lock:
            self._disabled_models.add(model_name)

    def enable_model(self, model_name: str):
        with self._lock:
            self._disabled_models.discard(model_name)

    @property
    def temp_mode(self) -> bool:
        with self._lock:
            return self._temp_mode

    @temp_mode.setter
    def temp_mode(self, value: bool):
        with self._lock:
            self._temp_mode = value

    def register_temp_cache(self, cache_path: Path):
        with self._lock:
            self._temp_cache_files.add(cache_path)

    def register_cache(self, model_name: str, cache_path: Path):
        with self._lock:
            self._known_caches[model_name] = cache_path

    def get_known_caches(self) -> dict[str, Path]:
        with self._lock:
            return dict(self._known_caches)

    def clear_known_cache(self, model_name: str):
        with self._lock:
            self._known_caches.pop(model_name, None)

    def clear_all_known_caches(self):
        with self._lock:
            self._known_caches.clear()


# Global state instance
_state = _CacheState()


# =============================================================================
# Enable/Disable Functions
# =============================================================================

def enable_cache() -> None:
    """Enable caching for all models."""
    _state.global_enabled = True


def disable_cache() -> None:
    """Disable caching for all models."""
    _state.global_enabled = False


def enable_cache_for(*model_names: str) -> None:
    """Enable caching for specific models.

    Parameters
    ----------
    *model_names : str
        Model names to enable caching for
    """
    for name in model_names:
        _state.enable_model(name)


def disable_cache_for(*model_names: str) -> None:
    """Disable caching for specific models.

    Parameters
    ----------
    *model_names : str
        Model names to disable caching for
    """
    for name in model_names:
        _state.disable_model(name)


def is_cache_enabled() -> bool:
    """Check if caching is globally enabled.

    Returns
    -------
    bool
        True if caching is enabled
    """
    return _state.global_enabled


def is_cache_enabled_for(model_name: str) -> bool:
    """Check if caching is enabled for a specific model.

    Parameters
    ----------
    model_name : str
        Model name to check

    Returns
    -------
    bool
        True if caching is enabled for this model
    """
    if not _state.global_enabled:
        return False
    return not _state.is_model_disabled(model_name)


# =============================================================================
# Temp Cache Functions
# =============================================================================

def enable_temp_cache() -> None:
    """Enable temporary cache mode.

    In this mode, cache files created after this call will be
    automatically deleted when the process exits.
    """
    _state.temp_mode = True


def disable_temp_cache() -> None:
    """Disable temporary cache mode.

    Cache files created after this call will be persistent.
    """
    _state.temp_mode = False


# =============================================================================
# Context Managers
# =============================================================================

@contextmanager
def cache_disabled():
    """Context manager to temporarily disable all caching.

    Example
    -------
    >>> with cache_disabled():
    ...     ds = model.open_dataset()  # No caching
    """
    prev_state = _state.global_enabled
    _state.global_enabled = False
    try:
        yield
    finally:
        _state.global_enabled = prev_state


@contextmanager
def cache_disabled_for(*model_names: str):
    """Context manager to temporarily disable caching for specific models.

    Parameters
    ----------
    *model_names : str
        Model names to disable caching for

    Example
    -------
    >>> with cache_disabled_for('GOT5.5'):
    ...     ds = model.open_dataset()  # No caching for GOT5.5
    """
    # Remember previous state
    prev_disabled = set()
    for name in model_names:
        if _state.is_model_disabled(name):
            prev_disabled.add(name)
        else:
            _state.disable_model(name)

    try:
        yield
    finally:
        # Restore previous state
        for name in model_names:
            if name not in prev_disabled:
                _state.enable_model(name)


# =============================================================================
# Cache Path Functions
# =============================================================================

def get_cache_dir() -> Optional[Path]:
    """Get the custom cache directory from environment variable.

    Returns
    -------
    Path or None
        Custom cache directory, or None if not set
    """
    cache_dir = os.environ.get('PYTMD_TURBO_CACHE_DIR')
    if cache_dir:
        return Path(cache_dir)
    return None


def get_cache_path(model_name: str, model_directory: Union[str, Path]) -> Path:
    """Get the cache file path for a model.

    Parameters
    ----------
    model_name : str
        Name of the model
    model_directory : str or Path
        Directory containing the model data

    Returns
    -------
    Path
        Path to the cache file
    """
    custom_dir = get_cache_dir()

    if custom_dir:
        custom_dir.mkdir(parents=True, exist_ok=True)
        return custom_dir / f"{model_name}.npz"
    else:
        model_dir = Path(model_directory).expanduser()
        return model_dir / ".pytmd_turbo_cache.npz"


def _get_source_mtime(model_directory: Union[str, Path], model_files: list[Path]) -> float:
    """Get the latest modification time of source files.

    Parameters
    ----------
    model_directory : str or Path
        Model directory
    model_files : list of Path
        List of model files

    Returns
    -------
    float
        Latest mtime as timestamp
    """
    latest_mtime = 0.0
    for f in model_files:
        try:
            mtime = f.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
        except OSError:
            pass
    return latest_mtime


# =============================================================================
# Cache Operations
# =============================================================================

def save_cache(
    model_name: str,
    cache_path: Path,
    data: dict[str, np.ndarray],
    source_files: list[Path],
) -> None:
    """Save data to cache file.

    Parameters
    ----------
    model_name : str
        Name of the model
    cache_path : Path
        Path to save cache file
    data : dict
        Dictionary of numpy arrays to cache
    source_files : list of Path
        Source files used to generate the cache
    """
    if not is_cache_enabled_for(model_name):
        return

    # Get source file modification time
    source_mtime = _get_source_mtime(cache_path.parent, source_files)

    # Prepare metadata
    metadata = {
        'model_name': model_name,
        'created_at': datetime.now().isoformat(),
        'source_mtime': source_mtime,
        'version': '1.0',
    }

    # Save with metadata
    cache_data = dict(data)
    cache_data['_metadata_json'] = np.array([str(metadata)])

    try:
        # Ensure parent directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        np.savez_compressed(cache_path, **cache_data)

        # Register cache
        _state.register_cache(model_name, cache_path)

        # Register for temp cleanup if in temp mode
        if _state.temp_mode:
            _state.register_temp_cache(cache_path)

    except Exception as e:
        warnings.warn(
            f"Failed to save cache for model '{model_name}' to '{cache_path}': {e}. "
            "Cache will not be available for subsequent loads.",
            RuntimeWarning,
            stacklevel=2
        )


def load_cache(
    model_name: str,
    cache_path: Path,
    source_files: Optional[list[Path]] = None,
) -> Optional[dict[str, np.ndarray]]:
    """Load data from cache file.

    Parameters
    ----------
    model_name : str
        Name of the model
    cache_path : Path
        Path to cache file
    source_files : list of Path, optional
        Source files to check for updates

    Returns
    -------
    dict or None
        Cached data, or None if cache is invalid/missing
    """
    if not is_cache_enabled_for(model_name):
        return None

    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=True) as npz:
            data = {key: npz[key] for key in npz.files if not key.startswith('_')}

            # Check if source files are newer than cache
            if source_files:
                metadata_arr = npz.get('_metadata_json')
                if metadata_arr is not None:
                    metadata = eval(str(metadata_arr[0]))
                    cached_mtime = metadata.get('source_mtime', 0)
                    current_mtime = _get_source_mtime(cache_path.parent, source_files)

                    if current_mtime > cached_mtime:
                        # Source files are newer, cache is stale
                        return None

            # Register cache
            _state.register_cache(model_name, cache_path)

            return data

    except Exception as e:
        warnings.warn(
            f"Failed to load cache from '{cache_path}': {e}. "
            "Cache will be ignored and data will be loaded from source.",
            RuntimeWarning,
            stacklevel=2
        )
        return None


def clear_cache(model_name: str) -> bool:
    """Delete cache for a specific model.

    Parameters
    ----------
    model_name : str
        Name of the model

    Returns
    -------
    bool
        True if cache was deleted
    """
    caches = _state.get_known_caches()
    cache_path = caches.get(model_name)

    if cache_path and cache_path.exists():
        try:
            cache_path.unlink()
            _state.clear_known_cache(model_name)
            return True
        except OSError:
            return False

    return False


def clear_all_cache() -> int:
    """Delete all known caches.

    Returns
    -------
    int
        Number of caches deleted
    """
    caches = _state.get_known_caches()
    deleted = 0

    for _model_name, cache_path in caches.items():
        if cache_path.exists():
            try:
                cache_path.unlink()
                deleted += 1
            except OSError:
                pass

    _state.clear_all_known_caches()
    return deleted


def rebuild_cache(model_name: str) -> bool:
    """Force rebuild cache for a specific model.

    This deletes the existing cache, forcing it to be regenerated
    on the next model load.

    Parameters
    ----------
    model_name : str
        Name of the model

    Returns
    -------
    bool
        True if cache was deleted (will be rebuilt on next load)
    """
    return clear_cache(model_name)


def rebuild_all_cache() -> int:
    """Force rebuild all caches.

    This deletes all existing caches, forcing them to be regenerated
    on the next model load.

    Returns
    -------
    int
        Number of caches deleted
    """
    return clear_all_cache()


# =============================================================================
# Status Functions
# =============================================================================

def get_cache_info() -> list[dict]:
    """Get information about all known caches.

    Returns
    -------
    list of dict
        List of cache information dictionaries
    """
    caches = _state.get_known_caches()
    info_list = []

    for model_name, cache_path in caches.items():
        info = {
            'model': model_name,
            'path': str(cache_path),
            'exists': cache_path.exists(),
            'size': None,
            'created': None,
            'temp': cache_path in _state._temp_cache_files,
        }

        if cache_path.exists():
            stat = cache_path.stat()
            info['size'] = stat.st_size
            info['created'] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        info_list.append(info)

    return info_list


def _format_size(size_bytes: Optional[int]) -> str:
    """Format size in human readable form."""
    if size_bytes is None:
        return '-'

    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024

    return f"{size:.1f} TB"


def show_cache_status() -> None:
    """Print cache status to stdout."""
    info_list = get_cache_info()

    if not info_list:
        print("No caches found.")
        print(f"\nCache enabled: {is_cache_enabled()}")
        print(f"Temp mode: {_state.temp_mode}")
        return

    # Header
    print(f"{'Model':<20} {'Size':>10} {'Created':<20} {'Temp':<5} Path")
    print("-" * 100)

    for info in info_list:
        size_str = _format_size(info['size'])
        created_str = info['created'][:19] if info['created'] else '-'
        temp_str = 'Yes' if info['temp'] else 'No'
        path_str = info['path']

        # Truncate path if too long
        if len(path_str) > 45:
            path_str = '...' + path_str[-42:]

        print(f"{info['model']:<20} {size_str:>10} {created_str:<20} {temp_str:<5} {path_str}")

    print()
    print(f"Cache enabled: {is_cache_enabled()}")
    print(f"Temp mode: {_state.temp_mode}")
