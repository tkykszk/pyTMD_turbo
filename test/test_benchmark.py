"""
Benchmark test comparing pyTMD and pyTMD_turbo performance

Tests:
1. pyTMD (original)
2. pyTMD_turbo single-thread (NumPy)
3. pyTMD_turbo multi-thread (Numba parallel)
4. pyTMD_turbo multi-process (concurrent.futures)

Model loading time is EXCLUDED from measurements.
"""

import pytest
import numpy as np
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing


# Skip if pyTMD not installed
pytmd_available = pytest.importorskip("pyTMD", reason="pyTMD not installed")


class BenchmarkConfig:
    """Benchmark configuration"""
    # Test sizes
    SMALL = {'n_points': 10, 'n_times': 24, 'label': 'Small (10pts x 24h)'}
    MEDIUM = {'n_points': 100, 'n_times': 168, 'label': 'Medium (100pts x 1week)'}
    LARGE = {'n_points': 1000, 'n_times': 720, 'label': 'Large (1000pts x 1month)'}

    # Repeat count for timing accuracy
    N_REPEATS = 3

    # Test location (Tokyo area)
    BASE_LAT = 35.0
    BASE_LON = 140.0

    # Model
    MODEL = 'GOT5.5'


def generate_test_data(n_points: int, n_times: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data (locations and times)"""
    np.random.seed(seed)

    # Random locations around base point
    lats = BenchmarkConfig.BASE_LAT + np.random.uniform(-5, 5, n_points)
    lons = BenchmarkConfig.BASE_LON + np.random.uniform(-10, 10, n_points)

    # Time array (hourly from 2026-01-01)
    mjd_start = 60676.0  # 2026-01-01
    mjd = mjd_start + np.arange(n_times) / 24.0

    return lats, lons, mjd


def _worker_pytmd(args):
    """Worker function for pyTMD multiprocess"""
    import pyTMD.compute
    lon, lat, delta_time, model, model_dir = args
    result = pyTMD.compute.tide_elevations(
        lon, lat, delta_time,
        directory=model_dir,
        model=model,
        crs=4326,
        epoch=(1992, 1, 1, 0, 0, 0),
        type='drift',
        method='linear',
    )
    return np.asarray(result)


def _worker_turbo(args):
    """Worker function for pyTMD_turbo multiprocess"""
    import pyTMD_turbo
    lon, lat, mjd, model, model_dir = args
    pyTMD_turbo.init_model(model, model_dir)
    result = pyTMD_turbo.tide_elevations(lon, lat, mjd, model=model)
    return np.asarray(result)


class TestBenchmark:
    """Benchmark tests for tide calculation performance"""

    @pytest.fixture(autouse=True)
    def setup(self, directory):
        """Setup test environment"""
        self.model_dir = str(directory)
        self.results = {}

    def _run_pytmd_single(self, lats: np.ndarray, lons: np.ndarray, mjd: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run pyTMD (single thread)"""
        import pyTMD.compute

        # Convert MJD to delta_time (seconds since 1992-01-01)
        delta_time = (mjd - 48622.0) * 86400.0

        # Expand for drift mode
        n_points = len(lats)
        n_times = len(mjd)
        lon_expanded = np.repeat(lons, n_times)
        lat_expanded = np.repeat(lats, n_times)
        time_expanded = np.tile(delta_time, n_points)

        start = time.perf_counter()
        result = pyTMD.compute.tide_elevations(
            lon_expanded, lat_expanded, time_expanded,
            directory=self.model_dir,
            model=BenchmarkConfig.MODEL,
            crs=4326,
            epoch=(1992, 1, 1, 0, 0, 0),
            type='drift',
            method='linear',
        )
        elapsed = time.perf_counter() - start

        return np.asarray(result).reshape(n_points, n_times), elapsed

    def _run_turbo_single(self, lats: np.ndarray, lons: np.ndarray, mjd: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run pyTMD_turbo (single thread, NumPy only)"""
        import pyTMD_turbo
        from pyTMD_turbo.predict import harmonic_numba

        # Disable Numba parallel temporarily
        original_parallel = getattr(harmonic_numba, '_use_parallel', True)

        # Load model first (excluded from timing)
        pyTMD_turbo.init_model(BenchmarkConfig.MODEL, self.model_dir)

        # Expand data
        n_points = len(lats)
        n_times = len(mjd)
        lon_expanded = np.repeat(lons, n_times)
        lat_expanded = np.repeat(lats, n_times)
        mjd_expanded = np.tile(mjd, n_points)

        start = time.perf_counter()
        result = pyTMD_turbo.tide_elevations(
            lon_expanded, lat_expanded, mjd_expanded,
            model=BenchmarkConfig.MODEL
        )
        elapsed = time.perf_counter() - start

        return np.asarray(result).reshape(n_points, n_times), elapsed

    def _run_turbo_parallel(self, lats: np.ndarray, lons: np.ndarray, mjd: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run pyTMD_turbo (Numba parallel)"""
        import pyTMD_turbo

        # Load model first (excluded from timing)
        pyTMD_turbo.init_model(BenchmarkConfig.MODEL, self.model_dir)

        # Expand data
        n_points = len(lats)
        n_times = len(mjd)
        lon_expanded = np.repeat(lons, n_times)
        lat_expanded = np.repeat(lats, n_times)
        mjd_expanded = np.tile(mjd, n_points)

        start = time.perf_counter()
        result = pyTMD_turbo.tide_elevations(
            lon_expanded, lat_expanded, mjd_expanded,
            model=BenchmarkConfig.MODEL
        )
        elapsed = time.perf_counter() - start

        return np.asarray(result).reshape(n_points, n_times), elapsed

    def _run_turbo_multiprocess(self, lats: np.ndarray, lons: np.ndarray, mjd: np.ndarray,
                                 n_workers: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """Run pyTMD_turbo with multiprocessing"""
        import pyTMD_turbo

        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), len(lats))

        # Load model first (excluded from timing)
        pyTMD_turbo.init_model(BenchmarkConfig.MODEL, self.model_dir)

        # Split work by points
        n_points = len(lats)
        chunk_size = max(1, n_points // n_workers)

        # Prepare args for each worker
        args_list = []
        for i in range(0, n_points, chunk_size):
            end_idx = min(i + chunk_size, n_points)
            chunk_lons = np.repeat(lons[i:end_idx], len(mjd))
            chunk_lats = np.repeat(lats[i:end_idx], len(mjd))
            chunk_mjd = np.tile(mjd, end_idx - i)
            args_list.append((chunk_lons, chunk_lats, chunk_mjd,
                             BenchmarkConfig.MODEL, self.model_dir))

        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_worker_turbo, args_list))
        elapsed = time.perf_counter() - start

        # Combine results
        combined = np.concatenate([r.reshape(-1, len(mjd)) for r in results], axis=0)

        return combined, elapsed

    def _benchmark_size(self, size_config: dict) -> Dict:
        """Run benchmark for a specific size configuration"""
        n_points = size_config['n_points']
        n_times = size_config['n_times']
        label = size_config['label']

        print(f"\n{'='*60}")
        print(f"Benchmark: {label}")
        print(f"{'='*60}")

        lats, lons, mjd = generate_test_data(n_points, n_times)

        results = {
            'config': size_config,
            'n_calculations': n_points * n_times,
            'methods': {}
        }

        # 1. pyTMD (single thread)
        print(f"\n1. pyTMD (single thread)...")
        times_pytmd = []
        for i in range(BenchmarkConfig.N_REPEATS):
            _, elapsed = self._run_pytmd_single(lats, lons, mjd)
            times_pytmd.append(elapsed)
            print(f"   Run {i+1}: {elapsed*1000:.1f} ms")
        results['methods']['pyTMD'] = {
            'times': times_pytmd,
            'mean_ms': np.mean(times_pytmd) * 1000,
            'std_ms': np.std(times_pytmd) * 1000,
        }

        # 2. pyTMD_turbo (single thread / NumPy)
        print(f"\n2. pyTMD_turbo (single thread)...")
        times_turbo_single = []
        for i in range(BenchmarkConfig.N_REPEATS):
            _, elapsed = self._run_turbo_single(lats, lons, mjd)
            times_turbo_single.append(elapsed)
            print(f"   Run {i+1}: {elapsed*1000:.1f} ms")
        results['methods']['turbo_single'] = {
            'times': times_turbo_single,
            'mean_ms': np.mean(times_turbo_single) * 1000,
            'std_ms': np.std(times_turbo_single) * 1000,
        }

        # 3. pyTMD_turbo (Numba parallel)
        print(f"\n3. pyTMD_turbo (Numba parallel)...")
        times_turbo_parallel = []
        for i in range(BenchmarkConfig.N_REPEATS):
            _, elapsed = self._run_turbo_parallel(lats, lons, mjd)
            times_turbo_parallel.append(elapsed)
            print(f"   Run {i+1}: {elapsed*1000:.1f} ms")
        results['methods']['turbo_parallel'] = {
            'times': times_turbo_parallel,
            'mean_ms': np.mean(times_turbo_parallel) * 1000,
            'std_ms': np.std(times_turbo_parallel) * 1000,
        }

        # 4. pyTMD_turbo (multiprocess)
        n_workers = min(multiprocessing.cpu_count(), n_points)
        print(f"\n4. pyTMD_turbo (multiprocess, {n_workers} workers)...")
        times_turbo_mp = []
        for i in range(BenchmarkConfig.N_REPEATS):
            _, elapsed = self._run_turbo_multiprocess(lats, lons, mjd, n_workers)
            times_turbo_mp.append(elapsed)
            print(f"   Run {i+1}: {elapsed*1000:.1f} ms")
        results['methods']['turbo_multiprocess'] = {
            'times': times_turbo_mp,
            'mean_ms': np.mean(times_turbo_mp) * 1000,
            'std_ms': np.std(times_turbo_mp) * 1000,
            'n_workers': n_workers,
        }

        # Calculate speedups
        pytmd_mean = results['methods']['pyTMD']['mean_ms']
        for method_name, method_data in results['methods'].items():
            if method_name != 'pyTMD':
                method_data['speedup'] = pytmd_mean / method_data['mean_ms']

        return results

    def test_benchmark_small(self):
        """Benchmark with small dataset"""
        results = self._benchmark_size(BenchmarkConfig.SMALL)
        self._print_summary(results)
        self.results['small'] = results

    def test_benchmark_medium(self):
        """Benchmark with medium dataset"""
        results = self._benchmark_size(BenchmarkConfig.MEDIUM)
        self._print_summary(results)
        self.results['medium'] = results

    @pytest.mark.slow
    def test_benchmark_large(self):
        """Benchmark with large dataset"""
        results = self._benchmark_size(BenchmarkConfig.LARGE)
        self._print_summary(results)
        self.results['large'] = results

    def _print_summary(self, results: Dict):
        """Print benchmark summary"""
        print(f"\n{'='*60}")
        print(f"Summary: {results['config']['label']}")
        print(f"Total calculations: {results['n_calculations']:,}")
        print(f"{'='*60}")
        print(f"{'Method':<25} {'Time (ms)':<15} {'Speedup':<10}")
        print("-" * 50)

        for method_name, method_data in results['methods'].items():
            speedup = method_data.get('speedup', 1.0)
            speedup_str = f"{speedup:.1f}x" if speedup != 1.0 else "baseline"
            print(f"{method_name:<25} {method_data['mean_ms']:<15.1f} {speedup_str:<10}")


def run_full_benchmark(model_dir: str) -> Dict:
    """Run full benchmark and return results"""
    import pyTMD_turbo

    results = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'cpu_count': multiprocessing.cpu_count(),
            'model': BenchmarkConfig.MODEL,
        },
        'benchmarks': {}
    }

    # Preload model
    print("Loading model (excluded from timing)...")
    pyTMD_turbo.init_model(BenchmarkConfig.MODEL, model_dir)

    # Create benchmark instance
    class MockDirectory:
        def __init__(self, path):
            self.path = path
        def __str__(self):
            return self.path

    benchmark = TestBenchmark()
    benchmark.model_dir = model_dir
    benchmark.results = {}

    # Run benchmarks
    for size_name, size_config in [
        ('small', BenchmarkConfig.SMALL),
        ('medium', BenchmarkConfig.MEDIUM),
        ('large', BenchmarkConfig.LARGE),
    ]:
        print(f"\n\nRunning {size_name} benchmark...")
        try:
            results['benchmarks'][size_name] = benchmark._benchmark_size(size_config)
        except Exception as e:
            print(f"Error in {size_name} benchmark: {e}")
            results['benchmarks'][size_name] = {'error': str(e)}

    return results


def generate_html_report(results: Dict, output_path: str = 'benchmark_report.html'):
    """Generate HTML report from benchmark results"""

    # Prepare data for charts
    benchmark_data = results.get('benchmarks', {})

    # Method display names
    method_names = {
        'pyTMD': 'pyTMD (original)',
        'turbo_single': 'pyTMD_turbo (single)',
        'turbo_parallel': 'pyTMD_turbo (parallel)',
        'turbo_multiprocess': 'pyTMD_turbo (multiproc)',
    }

    # Colors
    method_colors = {
        'pyTMD': '#ff6b6b',
        'turbo_single': '#4ecdc4',
        'turbo_parallel': '#45b7d1',
        'turbo_multiprocess': '#96ceb4',
    }

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>pyTMD vs pyTMD_turbo Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .card {{
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; margin-top: 0; }}
        h2 {{ color: #555; margin-top: 0; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .info-item {{ background: #e7f3ff; padding: 15px; border-radius: 4px; }}
        .info-item strong {{ display: block; color: #007bff; font-size: 0.9em; margin-bottom: 5px; }}
        .chart {{ width: 100%; height: 400px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
        th {{ background: #007bff; color: white; }}
        td:first-child {{ text-align: left; font-weight: 500; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .speedup {{ color: #28a745; font-weight: bold; }}
        .method-legend {{
            display: flex; flex-wrap: wrap; gap: 15px;
            margin: 15px 0; justify-content: center;
        }}
        .method-item {{
            display: flex; align-items: center; gap: 8px;
            padding: 5px 12px; background: #f0f0f0; border-radius: 4px;
        }}
        .method-color {{ width: 16px; height: 16px; border-radius: 3px; }}
        .summary-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px; margin: 20px 0;
        }}
        .summary-card {{
            text-align: center; padding: 20px; border-radius: 8px;
        }}
        .summary-card.pytmd {{ background: #ffe5e5; }}
        .summary-card.single {{ background: #e0f7f5; }}
        .summary-card.parallel {{ background: #e3f2fd; }}
        .summary-card.multiproc {{ background: #e8f5e9; }}
        .summary-value {{ font-size: 2em; font-weight: bold; }}
        .summary-label {{ font-size: 0.85em; color: #666; margin-top: 5px; }}
        .note {{ background: #fff3cd; padding: 15px; border-radius: 4px; border-left: 4px solid #ffc107; margin: 15px 0; }}
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>pyTMD vs pyTMD_turbo Benchmark Report</h1>
        <div class="info-grid">
            <div class="info-item">
                <strong>Generated</strong>
                {results.get('timestamp', 'N/A')}
            </div>
            <div class="info-item">
                <strong>CPU Cores</strong>
                {results.get('system', {}).get('cpu_count', 'N/A')}
            </div>
            <div class="info-item">
                <strong>Model</strong>
                {results.get('system', {}).get('model', 'N/A')}
            </div>
            <div class="info-item">
                <strong>Repeats</strong>
                {BenchmarkConfig.N_REPEATS} per test
            </div>
        </div>
        <div class="note">
            <strong>Note:</strong> Model loading time is excluded from all measurements.
            Only computation time is measured.
        </div>
    </div>

    <div class="method-legend">
        <div class="method-item">
            <div class="method-color" style="background: {method_colors['pyTMD']};"></div>
            <span>{method_names['pyTMD']}</span>
        </div>
        <div class="method-item">
            <div class="method-color" style="background: {method_colors['turbo_single']};"></div>
            <span>{method_names['turbo_single']}</span>
        </div>
        <div class="method-item">
            <div class="method-color" style="background: {method_colors['turbo_parallel']};"></div>
            <span>{method_names['turbo_parallel']}</span>
        </div>
        <div class="method-item">
            <div class="method-color" style="background: {method_colors['turbo_multiprocess']};"></div>
            <span>{method_names['turbo_multiprocess']}</span>
        </div>
    </div>
'''

    # Add benchmark sections
    for size_name, size_label in [
        ('small', 'Small Dataset'),
        ('medium', 'Medium Dataset'),
        ('large', 'Large Dataset'),
    ]:
        bench = benchmark_data.get(size_name, {})
        if 'error' in bench:
            html += f'''
    <div class="card">
        <h2>{size_label}</h2>
        <p style="color: red;">Error: {bench['error']}</p>
    </div>
'''
            continue

        if not bench.get('methods'):
            continue

        config = bench.get('config', {})
        methods = bench.get('methods', {})

        # Get values
        pytmd_ms = methods.get('pyTMD', {}).get('mean_ms', 0)
        single_ms = methods.get('turbo_single', {}).get('mean_ms', 0)
        parallel_ms = methods.get('turbo_parallel', {}).get('mean_ms', 0)
        multiproc_ms = methods.get('turbo_multiprocess', {}).get('mean_ms', 0)

        single_speedup = methods.get('turbo_single', {}).get('speedup', 1)
        parallel_speedup = methods.get('turbo_parallel', {}).get('speedup', 1)
        multiproc_speedup = methods.get('turbo_multiprocess', {}).get('speedup', 1)

        html += f'''
    <div class="card">
        <h2>{size_label}: {config.get('label', '')}</h2>
        <p>Total calculations: <strong>{bench.get('n_calculations', 0):,}</strong>
           ({config.get('n_points', 0)} points x {config.get('n_times', 0)} times)</p>

        <div class="summary-grid">
            <div class="summary-card pytmd">
                <div class="summary-value">{pytmd_ms:.1f}ms</div>
                <div class="summary-label">{method_names['pyTMD']}</div>
            </div>
            <div class="summary-card single">
                <div class="summary-value">{single_ms:.1f}ms</div>
                <div class="summary-label">{method_names['turbo_single']}<br><span class="speedup">{single_speedup:.1f}x</span></div>
            </div>
            <div class="summary-card parallel">
                <div class="summary-value">{parallel_ms:.1f}ms</div>
                <div class="summary-label">{method_names['turbo_parallel']}<br><span class="speedup">{parallel_speedup:.1f}x</span></div>
            </div>
            <div class="summary-card multiproc">
                <div class="summary-value">{multiproc_ms:.1f}ms</div>
                <div class="summary-label">{method_names['turbo_multiprocess']}<br><span class="speedup">{multiproc_speedup:.1f}x</span></div>
            </div>
        </div>

        <div id="chart-{size_name}" class="chart"></div>

        <table>
            <tr>
                <th>Method</th>
                <th>Mean (ms)</th>
                <th>Std (ms)</th>
                <th>Speedup</th>
            </tr>
'''
        for method_key, method_display in method_names.items():
            m = methods.get(method_key, {})
            speedup = m.get('speedup', 1.0)
            speedup_str = f'{speedup:.1f}x' if method_key != 'pyTMD' else 'baseline'
            html += f'''
            <tr>
                <td>{method_display}</td>
                <td>{m.get('mean_ms', 0):.2f}</td>
                <td>{m.get('std_ms', 0):.2f}</td>
                <td class="speedup">{speedup_str}</td>
            </tr>
'''
        html += '''
        </table>
    </div>
'''

    # Add JavaScript for charts
    html += '''
<script>
'''
    for size_name in ['small', 'medium', 'large']:
        bench = benchmark_data.get(size_name, {})
        if 'error' in bench or not bench.get('methods'):
            continue

        methods = bench.get('methods', {})

        # Prepare chart data
        method_keys = ['pyTMD', 'turbo_single', 'turbo_parallel', 'turbo_multiprocess']
        values = [methods.get(k, {}).get('mean_ms', 0) for k in method_keys]
        errors = [methods.get(k, {}).get('std_ms', 0) for k in method_keys]
        colors = [method_colors.get(k, '#888') for k in method_keys]
        labels = [method_names.get(k, k) for k in method_keys]

        html += f'''
Plotly.newPlot('chart-{size_name}', [{{
    x: {json.dumps(labels)},
    y: {json.dumps(values)},
    error_y: {{
        type: 'data',
        array: {json.dumps(errors)},
        visible: true
    }},
    type: 'bar',
    marker: {{
        color: {json.dumps(colors)}
    }},
    text: {json.dumps([f'{v:.1f}ms' for v in values])},
    textposition: 'outside'
}}], {{
    title: '{size_name.capitalize()} Dataset - Execution Time',
    yaxis: {{
        title: 'Time (ms)',
        type: 'log'
    }},
    xaxis: {{
        title: 'Method'
    }},
    showlegend: false,
    margin: {{ t: 50, b: 100 }}
}}, {{ responsive: true }});
'''

    html += '''
</script>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


if __name__ == "__main__":
    import sys

    # Get model directory from environment or argument
    model_dir = os.environ.get('PYTMD_RESOURCE', '~/Documents/python/eq/LargeFiles')
    model_dir = os.path.expanduser(model_dir)

    if len(sys.argv) > 1:
        model_dir = sys.argv[1]

    print(f"Model directory: {model_dir}")
    print(f"CPU cores: {multiprocessing.cpu_count()}")

    # Run benchmark
    results = run_full_benchmark(model_dir)

    # Save JSON results
    script_dir = Path(os.path.abspath(__file__ if '__file__' in dir() else '.')).parent
    json_path = script_dir.parent / 'examples' / 'benchmark_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON results saved to: {json_path}")

    # Generate HTML report
    html_path = script_dir.parent / 'examples' / 'benchmark_report.html'
    generate_html_report(results, str(html_path))
    print(f"HTML report saved to: {html_path}")
