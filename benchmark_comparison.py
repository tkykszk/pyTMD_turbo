#!/usr/bin/env python3
"""
Benchmark comparison: pyTMD vs pyTMD_turbo
- pyTMD (original)
- pyTMD_turbo (no cache, first run)
- pyTMD_turbo (with cache, second run)
"""

import numpy as np
import time
from datetime import datetime

# Model directory
MODEL_DIR = "/Users/taka/Documents/python/eq/LargeFiles"
MODEL_NAME = "GOT5.5"

# Test parameters
POINT_COUNTS = [10, 100, 1000]
N_TIMES = 24  # 24 time points (hourly for one day)

def generate_test_data(n_points, n_times):
    """Generate random test coordinates and times."""
    np.random.seed(42)  # Reproducible results
    lons = np.random.uniform(-180, 180, n_points)
    lats = np.random.uniform(-60, 60, n_points)  # Avoid polar regions

    # Generate times (one day, hourly)
    base_time = datetime(2024, 1, 1)
    times = np.array([
        np.datetime64(base_time) + np.timedelta64(i, 'h')
        for i in range(n_times)
    ])

    return lons, lats, times


def generate_test_data_drift(n_points):
    """Generate test data in drift format (each point has one time)."""
    np.random.seed(42)
    lons = np.random.uniform(-180, 180, n_points)
    lats = np.random.uniform(-60, 60, n_points)

    # Each point has a different time (simulating drift buoy data)
    base_time = datetime(2024, 1, 1)
    times = np.array([
        np.datetime64(base_time) + np.timedelta64(i, 'h')
        for i in range(n_points)
    ])

    return lons, lats, times


def benchmark_pytmd(lons, lats, times):
    """Benchmark original pyTMD (drift format: each point has one time)."""
    import pyTMD.compute

    start = time.perf_counter()

    # Use pyTMD's compute.tide_elevations
    tide = pyTMD.compute.tide_elevations(
        lons, lats, times,
        directory=MODEL_DIR,
        model=MODEL_NAME,
        crs=4326,
        type='drift',
        standard='datetime'
    )

    elapsed = time.perf_counter() - start
    return elapsed, tide


def benchmark_pytmd_grid(lons, lats, times):
    """Benchmark original pyTMD (grid format: all points x all times)."""
    import pyTMD.compute

    n_points = len(lons)
    n_times = len(times)

    start = time.perf_counter()

    # For grid computation, we need to compute for each point at each time
    # pyTMD doesn't natively support batch point x time, so we loop
    results = []
    for i, t in enumerate(times):
        # Compute for all points at this single time
        tide = pyTMD.compute.tide_elevations(
            lons, lats, np.full(n_points, t),
            directory=MODEL_DIR,
            model=MODEL_NAME,
            crs=4326,
            type='drift',
            standard='datetime'
        )
        results.append(tide)

    tide_all = np.column_stack(results)  # (n_points, n_times)

    elapsed = time.perf_counter() - start
    return elapsed, tide_all


def benchmark_pytmd_turbo_no_cache(lons, lats, times):
    """Benchmark pyTMD_turbo with cache disabled."""
    import pyTMD_turbo
    from pyTMD_turbo import cache

    # Clear any existing cache and disable caching
    cache.clear_cache(MODEL_NAME)
    cache.disable_cache()

    # Reset global cache to force reload
    import pyTMD_turbo.compute as compute_module
    compute_module._cache = None
    compute_module._loaded_models = set()

    start = time.perf_counter()

    tide = pyTMD_turbo.tide_elevations(
        lons, lats, times,
        model=MODEL_NAME,
        directory=MODEL_DIR
    )

    elapsed = time.perf_counter() - start
    return elapsed, tide


def benchmark_pytmd_turbo_with_cache(lons, lats, times):
    """Benchmark pyTMD_turbo with cache enabled (second run)."""
    import pyTMD_turbo
    from pyTMD_turbo import cache

    # Enable caching and rebuild cache
    cache.enable_cache()

    # Reset global cache to force reload
    import pyTMD_turbo.compute as compute_module
    compute_module._cache = None
    compute_module._loaded_models = set()

    # First run to build cache
    _ = pyTMD_turbo.tide_elevations(
        lons, lats, times,
        model=MODEL_NAME,
        directory=MODEL_DIR
    )

    # Reset again to measure cached load
    compute_module._cache = None
    compute_module._loaded_models = set()

    start = time.perf_counter()

    tide = pyTMD_turbo.tide_elevations(
        lons, lats, times,
        model=MODEL_NAME,
        directory=MODEL_DIR
    )

    elapsed = time.perf_counter() - start
    return elapsed, tide


def main():
    print("=" * 70)
    print("Benchmark: pyTMD vs pyTMD_turbo")
    print(f"Model: {MODEL_NAME}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Time points: {N_TIMES}")
    print("=" * 70)
    print()

    results = []

    for n_points in POINT_COUNTS:
        print(f"\n{'='*70}")
        print(f"Testing with {n_points} points x {N_TIMES} times")
        print("=" * 70)

        lons, lats, times = generate_test_data(n_points, N_TIMES)

        row = {"n_points": n_points}

        # 1. pyTMD (loop over times)
        print(f"\n[1/3] Running pyTMD...")
        try:
            t_pytmd, _ = benchmark_pytmd_grid(lons, lats, times)
            row["pytmd"] = t_pytmd
            print(f"      Time: {t_pytmd:.4f} sec")
        except Exception as e:
            print(f"      Error: {e}")
            row["pytmd"] = None

        # 2. pyTMD_turbo (no cache)
        print(f"\n[2/3] Running pyTMD_turbo (no cache, first run)...")
        try:
            t_turbo_nocache, tide_turbo = benchmark_pytmd_turbo_no_cache(lons, lats, times)
            row["turbo_no_cache"] = t_turbo_nocache
            print(f"      Time: {t_turbo_nocache:.4f} sec")
        except Exception as e:
            print(f"      Error: {e}")
            row["turbo_no_cache"] = None

        # 3. pyTMD_turbo (with cache)
        print(f"\n[3/3] Running pyTMD_turbo (with cache, second run)...")
        try:
            t_turbo_cache, tide_turbo_cache = benchmark_pytmd_turbo_with_cache(lons, lats, times)
            row["turbo_with_cache"] = t_turbo_cache
            print(f"      Time: {t_turbo_cache:.4f} sec")
        except Exception as e:
            print(f"      Error: {e}")
            row["turbo_with_cache"] = None

        results.append(row)

    # Summary table
    print("\n")
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Points':<10} {'pyTMD':<15} {'turbo(no cache)':<18} {'turbo(cached)':<15} {'Speedup':<10}")
    print("-" * 70)

    for row in results:
        n = row["n_points"]
        t1 = row.get("pytmd")
        t2 = row.get("turbo_no_cache")
        t3 = row.get("turbo_with_cache")

        t1_str = f"{t1:.4f}s" if t1 else "N/A"
        t2_str = f"{t2:.4f}s" if t2 else "N/A"
        t3_str = f"{t3:.4f}s" if t3 else "N/A"

        if t1 and t3:
            speedup = f"{t1/t3:.1f}x"
        else:
            speedup = "N/A"

        print(f"{n:<10} {t1_str:<15} {t2_str:<18} {t3_str:<15} {speedup:<10}")

    print()
    print("Speedup = pyTMD time / pyTMD_turbo (cached) time")


if __name__ == "__main__":
    main()
