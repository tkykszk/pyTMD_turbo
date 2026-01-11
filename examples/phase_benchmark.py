"""
Phase Calculation Benchmark

Measures computation time for each step of phase calculation.
Compares single constituent vs multi-constituent fitting.
"""

import sys
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyTMD_turbo.compute import tide_elevations, init_model


# =============================================================================
# Timing Utilities
# =============================================================================

@dataclass
class TimingRecord:
    """Record of computation times"""
    name: str
    times: List[float] = field(default_factory=list)

    def add(self, elapsed: float):
        self.times.append(elapsed)

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def mean(self) -> float:
        return np.mean(self.times) if self.times else 0

    @property
    def count(self) -> int:
        return len(self.times)


class Timer:
    """Context manager for timing"""
    def __init__(self, record: TimingRecord):
        self.record = record
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.record.add(elapsed)


class BenchmarkResults:
    """Collection of timing records"""
    def __init__(self):
        self.records: Dict[str, TimingRecord] = {}

    def get(self, name: str) -> TimingRecord:
        if name not in self.records:
            self.records[name] = TimingRecord(name)
        return self.records[name]

    def timer(self, name: str) -> Timer:
        return Timer(self.get(name))

    def print_summary(self):
        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        print(f"{'Operation':<40} {'Count':>8} {'Total (ms)':>12} {'Mean (ms)':>12}")
        print("-" * 70)

        total_time = 0
        for name, record in sorted(self.records.items()):
            total_ms = record.total * 1000
            mean_ms = record.mean * 1000
            total_time += record.total
            print(f"{name:<40} {record.count:>8} {total_ms:>12.3f} {mean_ms:>12.3f}")

        print("-" * 70)
        print(f"{'TOTAL':<40} {'':<8} {total_time*1000:>12.3f}")
        print("=" * 70)


# =============================================================================
# Constituent Data
# =============================================================================

# Constituent angular frequencies (rad/s)
# Reference: https://tidesandcurrents.noaa.gov/harmonic.html
CONSTITUENT_PERIODS = {
    # Semi-diurnal
    'm2': 12.4206012,   # Principal lunar
    's2': 12.0000000,   # Principal solar
    'n2': 12.6583482,   # Larger lunar elliptic
    'k2': 11.9672348,   # Luni-solar
    # Diurnal
    'k1': 23.9344696,   # Luni-solar
    'o1': 25.8193417,   # Principal lunar
    'p1': 24.0658902,   # Principal solar
    'q1': 26.8683567,   # Larger lunar elliptic
    # Long period
    'mf': 327.8599387,  # Lunar fortnightly
    'mm': 661.3111655,  # Lunar monthly
}


def get_omega(constituent: str) -> float:
    """
    Get angular frequency for constituent (rad/s)

    This is a simple lookup - very fast O(1)
    """
    period_hours = CONSTITUENT_PERIODS.get(constituent.lower())
    if period_hours is None:
        raise ValueError(f"Unknown constituent: {constituent}")
    return 2 * np.pi / (period_hours * 3600)


def get_omega_from_pytmd(constituent: str) -> float:
    """
    Get angular frequency from pyTMD (for comparison)
    """
    import pyTMD.arguments

    # This requires more computation
    omega = pyTMD.arguments.frequency(constituent)
    return omega  # cycles/second, need to convert to rad/s


# =============================================================================
# Phase Fitting Functions
# =============================================================================

def fit_single_constituent(t: np.ndarray, y: np.ndarray, omega: float) -> dict:
    """
    Fit single sinusoid: y = A*sin(wt + phi) + C
    """
    sin_t = np.sin(omega * t)
    cos_t = np.cos(omega * t)
    ones = np.ones_like(t)

    X = np.column_stack([sin_t, cos_t, ones])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    a, b, c = coeffs

    amplitude = np.sqrt(a**2 + b**2)
    phase = np.arctan2(b, a)

    return {
        'amplitude': amplitude,
        'phase': phase,
        'offset': c,
    }


def fit_multi_constituent(t: np.ndarray, y: np.ndarray,
                          constituents: List[str]) -> Dict[str, dict]:
    """
    Fit multiple sinusoids simultaneously:
    y = sum_i A_i * sin(w_i * t + phi_i) + C

    Uses linear least squares with all constituents.
    """
    # Build design matrix
    columns = []
    for const in constituents:
        omega = get_omega(const)
        columns.append(np.sin(omega * t))
        columns.append(np.cos(omega * t))
    columns.append(np.ones_like(t))  # offset

    X = np.column_stack(columns)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    # Extract results
    results = {}
    for i, const in enumerate(constituents):
        a = coeffs[2*i]
        b = coeffs[2*i + 1]
        results[const] = {
            'amplitude': np.sqrt(a**2 + b**2),
            'phase': np.arctan2(b, a),
            'a': a,
            'b': b,
        }
    results['offset'] = coeffs[-1]

    return results


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = 'GOT5.5'
MODEL_DIR = '~/Documents/python/eq/LargeFiles'

LAT = 27.0744
LON = 142.2178

START_DATE = datetime(2026, 1, 1, 0, 0, 0)
END_DATE = datetime(2026, 1, 4, 0, 0, 0)

N_GRID_POINTS = 240
N_RANDOM_POINTS = 3

CONSTITUENTS = ['m2', 's2', 'k1', 'o1', 'n2']


# =============================================================================
# Main Benchmark
# =============================================================================

def datetime_to_mjd(dt: datetime) -> float:
    mjd_epoch = datetime(1858, 11, 17, 0, 0, 0)
    return (dt - mjd_epoch).total_seconds() / 86400.0


def mjd_to_datetime(mjd: float) -> datetime:
    mjd_epoch = datetime(1858, 11, 17, 0, 0, 0)
    return mjd_epoch + timedelta(days=mjd)


def main():
    bench = BenchmarkResults()

    print("=" * 70)
    print("Phase Calculation Benchmark")
    print("=" * 70)
    print(f"Location: {LAT}°N, {LON}°E")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Grid points: {N_GRID_POINTS}")
    print(f"Constituents: {CONSTITUENTS}")
    print()

    # Time range
    mjd_start = datetime_to_mjd(START_DATE)
    mjd_end = datetime_to_mjd(END_DATE)
    mjd_grid = np.linspace(mjd_start, mjd_end, N_GRID_POINTS)
    t_seconds = (mjd_grid - mjd_start) * 86400

    # Random points
    np.random.seed(42)
    random_mjd = np.sort(np.random.uniform(mjd_start, mjd_end, N_RANDOM_POINTS))
    random_t_seconds = (random_mjd - mjd_start) * 86400

    print("Random time points:")
    for i, mjd in enumerate(random_mjd):
        dt = mjd_to_datetime(mjd)
        print(f"  Point {i+1}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # =========================================================================
    # Step 1: Model initialization
    # =========================================================================
    print("Step 1: Model initialization...")
    with bench.timer("1. init_model"):
        init_model(MODEL_NAME, MODEL_DIR)
    print(f"  Done: {bench.get('1. init_model').total*1000:.2f} ms")

    # =========================================================================
    # Step 2: Tide computation (grid)
    # =========================================================================
    print("\nStep 2: Tide computation (grid)...")
    with bench.timer("2. tide_elevations (grid)"):
        tide_grid = tide_elevations(
            np.full(N_GRID_POINTS, LON),
            np.full(N_GRID_POINTS, LAT),
            mjd_grid,
            model=MODEL_NAME
        )
    print(f"  Done: {bench.get('2. tide_elevations (grid)').total*1000:.2f} ms")

    # =========================================================================
    # Step 3: Tide computation (random points)
    # =========================================================================
    print("\nStep 3: Tide computation (random points)...")
    with bench.timer("3. tide_elevations (random)"):
        tide_random = tide_elevations(
            np.full(N_RANDOM_POINTS, LON),
            np.full(N_RANDOM_POINTS, LAT),
            random_mjd,
            model=MODEL_NAME
        )
    print(f"  Done: {bench.get('3. tide_elevations (random)').total*1000:.2f} ms")

    # =========================================================================
    # Step 4: Get omega values
    # =========================================================================
    print("\nStep 4: Get omega values...")

    # 4a: From lookup table (our implementation)
    with bench.timer("4a. get_omega (lookup)"):
        for _ in range(1000):  # 1000 iterations for measurement
            for const in CONSTITUENTS:
                omega = get_omega(const)
    lookup_time = bench.get("4a. get_omega (lookup)").total
    print(f"  Lookup (1000 iter x {len(CONSTITUENTS)} constituents): {lookup_time*1000:.3f} ms")
    print(f"  Per call: {lookup_time/1000/len(CONSTITUENTS)*1e6:.3f} µs")

    # 4b: From pyTMD (if available)
    try:
        import pyTMD.arguments
        with bench.timer("4b. pyTMD.arguments.frequency"):
            for _ in range(1000):
                for const in CONSTITUENTS:
                    omega = pyTMD.arguments.frequency(const)
        pytmd_time = bench.get("4b. pyTMD.arguments.frequency").total
        print(f"  pyTMD (1000 iter x {len(CONSTITUENTS)} constituents): {pytmd_time*1000:.3f} ms")
        print(f"  Per call: {pytmd_time/1000/len(CONSTITUENTS)*1e6:.3f} µs")
        print(f"  Speedup: {pytmd_time/lookup_time:.1f}x faster with lookup")
    except ImportError:
        print("  pyTMD.arguments not available for comparison")

    # =========================================================================
    # Step 5: Single constituent fitting (M2 only)
    # =========================================================================
    print("\nStep 5: Single constituent fitting (M2)...")
    omega_m2 = get_omega('m2')

    with bench.timer("5. fit_single (M2)"):
        for _ in range(100):  # 100 iterations
            result_single = fit_single_constituent(t_seconds, tide_grid, omega_m2)

    single_time = bench.get("5. fit_single (M2)").total
    print(f"  100 iterations: {single_time*1000:.3f} ms")
    print(f"  Per fit: {single_time/100*1000:.3f} ms")
    print(f"  Result: A={result_single['amplitude']:.4f} m, φ={np.degrees(result_single['phase']):.1f}°")

    # =========================================================================
    # Step 6: Multi-constituent fitting
    # =========================================================================
    print(f"\nStep 6: Multi-constituent fitting ({len(CONSTITUENTS)} constituents)...")

    with bench.timer("6. fit_multi"):
        for _ in range(100):  # 100 iterations
            result_multi = fit_multi_constituent(t_seconds, tide_grid, CONSTITUENTS)

    multi_time = bench.get("6. fit_multi").total
    print(f"  100 iterations: {multi_time*1000:.3f} ms")
    print(f"  Per fit: {multi_time/100*1000:.3f} ms")
    print(f"  Overhead vs single: {multi_time/single_time:.2f}x")

    print("\n  Fitted constituents:")
    for const in CONSTITUENTS:
        r = result_multi[const]
        print(f"    {const.upper()}: A={r['amplitude']:.4f} m, φ={np.degrees(r['phase']):.1f}°")

    # =========================================================================
    # Step 7: Phase computation at random points
    # =========================================================================
    print("\nStep 7: Phase computation at random points...")

    with bench.timer("7. phase_at_points"):
        for _ in range(1000):  # 1000 iterations
            for t in random_t_seconds:
                for const in CONSTITUENTS:
                    omega = get_omega(const)
                    phase = omega * t + result_multi[const]['phase']
                    cos_phase = np.cos(phase)

    phase_time = bench.get("7. phase_at_points").total
    n_ops = 1000 * N_RANDOM_POINTS * len(CONSTITUENTS)
    print(f"  {n_ops} phase computations: {phase_time*1000:.3f} ms")
    print(f"  Per computation: {phase_time/n_ops*1e6:.3f} µs")

    # =========================================================================
    # Step 8: Derivative computation
    # =========================================================================
    print("\nStep 8: Derivative computation...")

    with bench.timer("8. derivative"):
        for _ in range(1000):
            for t in random_t_seconds:
                total_deriv = 0.0
                for const in CONSTITUENTS:
                    omega = get_omega(const)
                    r = result_multi[const]
                    phase = omega * t + r['phase']
                    deriv = r['amplitude'] * omega * np.cos(phase)
                    total_deriv += deriv

    deriv_time = bench.get("8. derivative").total
    print(f"  {n_ops} derivative computations: {deriv_time*1000:.3f} ms")
    print(f"  Per computation: {deriv_time/n_ops*1e6:.3f} µs")

    # =========================================================================
    # Summary
    # =========================================================================
    bench.print_summary()

    # =========================================================================
    # Final results for random points
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE AND DERIVATIVE AT RANDOM POINTS (Multi-constituent)")
    print("=" * 70)

    for i, (mjd, t_sec, tide_val) in enumerate(zip(random_mjd, random_t_seconds, tide_random)):
        dt = mjd_to_datetime(mjd)
        print(f"\nPoint {i+1}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Tide height: {tide_val:.4f} m")

        total_deriv = 0.0
        for const in CONSTITUENTS:
            omega = get_omega(const)
            r = result_multi[const]
            phase = omega * t_sec + r['phase']
            phase_norm = phase % (2 * np.pi)
            deriv = r['amplitude'] * omega * np.cos(phase)
            total_deriv += deriv
            print(f"  {const.upper()}: phase={np.degrees(phase_norm):6.1f}°, "
                  f"cos(φ)={np.cos(phase):7.4f}, deriv={deriv*3600:8.4f} m/h")

        print(f"  TOTAL derivative: {total_deriv*3600:.4f} m/h")

    return bench


if __name__ == '__main__':
    main()
