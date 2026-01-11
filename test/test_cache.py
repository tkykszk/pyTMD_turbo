"""
OceanTideCache tests

Copyright (c) 2024-2026 tkykszk
A derivative work of PyTMD (https://github.com/tsutterley/pyTMD)
Original author: Tyler Sutterley
Original license: MIT License (source code), CC BY 4.0 (content)

This software is licensed under the MIT License.
See LICENSE file for details.
"""

import pytest


def test_import_cache():
    """Test that cache module can be imported"""
    from pyTMD_turbo.predict.cache import OceanTideCache
    assert OceanTideCache is not None


def test_import_cache_optimized():
    """Test that cache_optimized module can be imported"""
    from pyTMD_turbo.predict.cache_optimized import OceanTideCacheOptimized
    assert OceanTideCacheOptimized is not None


def test_import_harmonic():
    """Test that harmonic module can be imported"""
    from pyTMD_turbo.predict.harmonic import HarmonicConstants, predict_vectorized
    assert HarmonicConstants is not None
    assert predict_vectorized is not None


def test_cache_instantiation():
    """Test that cache can be instantiated"""
    from pyTMD_turbo.predict.cache import OceanTideCache
    cache = OceanTideCache()
    assert cache is not None


def test_cache_optimized_instantiation():
    """Test that optimized cache can be instantiated"""
    from pyTMD_turbo.predict.cache_optimized import OceanTideCacheOptimized
    cache = OceanTideCacheOptimized()
    assert cache is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
