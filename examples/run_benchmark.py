#!/usr/bin/env python
"""
Benchmark runner script

Compares performance of:
1. pyTMD (original)
2. pyTMD_turbo (single-thread)
3. pyTMD_turbo (multi-thread/Numba parallel)
4. pyTMD_turbo (multi-process)

Model loading time is EXCLUDED from measurements.
"""

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyTMD.compute
import pyTMD.io
import timescale.time

import pyTMD_turbo

# Global cache for pyTMD preloaded data
_pytmd_cache = {}


# Configuration
MODEL = 'GOT5.5'
N_REPEATS = 3
BASE_LAT = 35.0
BASE_LON = 140.0

# Test sizes
SIZES = {
    'small': {'n_points': 10, 'n_times': 24, 'label': 'Small (10pts x 24h)'},
    'medium': {'n_points': 100, 'n_times': 168, 'label': 'Medium (100pts x 1week)'},
    'large': {'n_points': 1000, 'n_times': 720, 'label': 'Large (1000pts x 1month)'},
}


def generate_test_data(n_points, n_times, seed=42):
    """Generate test data"""
    np.random.seed(seed)
    lats = BASE_LAT + np.random.uniform(-5, 5, n_points)
    lons = BASE_LON + np.random.uniform(-10, 10, n_points)
    mjd_start = 60676.0  # 2026-01-01
    mjd = mjd_start + np.arange(n_times) / 24.0
    return lats, lons, mjd


def _worker_turbo(args):
    """Worker for multiprocess"""
    lon, lat, mjd, model, model_dir = args
    pyTMD_turbo.init_model(model, model_dir)
    return pyTMD_turbo.tide_elevations(lon, lat, mjd, model=model)


def preload_pytmd(model_dir, model=MODEL):
    """Preload pyTMD model and dataset"""
    global _pytmd_cache
    if model not in _pytmd_cache:
        m = pyTMD.io.model(model_dir).from_database(model)
        ds = m.open_dataset(group='z', chunks=None, append_node=False)
        _pytmd_cache[model] = {'model': m, 'dataset': ds}
    return _pytmd_cache[model]


def run_pytmd_preloaded(lats, lons, mjd, model_dir):
    """Run pyTMD benchmark using preloaded dataset (computation only)"""
    global _pytmd_cache
    cache = _pytmd_cache.get(MODEL)
    if not cache:
        raise RuntimeError("Call preload_pytmd() first")

    m = cache['model']
    ds = cache['dataset']

    n_points, n_times = len(lats), len(mjd)
    delta_time = (mjd - 48622.0) * 86400.0

    lon_exp = np.repeat(lons, n_times)
    lat_exp = np.repeat(lats, n_times)
    time_exp = np.tile(delta_time, n_points)

    # Convert MJD-based delta_time to datetime64
    # delta_time is seconds since 1992-01-01
    epoch_dt64 = np.datetime64('1992-01-01T00:00:00', 'ns')
    times = epoch_dt64 + (time_exp * 1e9).astype('timedelta64[ns]')

    start = time.perf_counter()

    # Convert to timescale
    ts = timescale.time.Timescale().from_datetime(times)

    # Nodal corrections
    nodal_corrections = m.corrections
    deltat = ts.tt_ut1

    # Convert coordinates
    X, Y = ds.tmd.coords_as(lon_exp, lat_exp, type='drift', crs=4326)

    # Interpolate
    local = ds.tmd.interp(X, Y, method='linear', extrapolate=False, cutoff=10.0)

    # Predict
    tide = local.tmd.predict(ts.tide, deltat=deltat, corrections=nodal_corrections)

    # Infer minor
    tide += local.tmd.infer(ts.tide, deltat=deltat, corrections=nodal_corrections, minor=m.minor)

    elapsed = time.perf_counter() - start
    return elapsed


def run_pytmd(lats, lons, mjd, model_dir):
    """Run pyTMD benchmark (includes model loading)"""
    n_points, n_times = len(lats), len(mjd)
    delta_time = (mjd - 48622.0) * 86400.0

    lon_exp = np.repeat(lons, n_times)
    lat_exp = np.repeat(lats, n_times)
    time_exp = np.tile(delta_time, n_points)

    start = time.perf_counter()
    pyTMD.compute.tide_elevations(
        lon_exp, lat_exp, time_exp,
        directory=model_dir,
        model=MODEL,
        crs=4326,
        epoch=(1992, 1, 1, 0, 0, 0),
        type='drift',
        method='linear',
    )
    elapsed = time.perf_counter() - start
    return elapsed


def run_turbo(lats, lons, mjd, model_dir, include_model_load=False):
    """Run pyTMD_turbo benchmark (includes Numba parallel internally)"""
    n_points, n_times = len(lats), len(mjd)

    lon_exp = np.repeat(lons, n_times)
    lat_exp = np.repeat(lats, n_times)
    mjd_exp = np.tile(mjd, n_points)

    if include_model_load:
        # Reset cache to force model reload
        pyTMD_turbo.compute._cache = None
        pyTMD_turbo.compute._loaded_models = set()

    start = time.perf_counter()
    if include_model_load:
        pyTMD_turbo.init_model(MODEL, model_dir)
    pyTMD_turbo.tide_elevations(lon_exp, lat_exp, mjd_exp, model=MODEL)
    elapsed = time.perf_counter() - start
    return elapsed


def run_turbo_multiprocess(lats, lons, mjd, model_dir, n_workers=None):
    """Run pyTMD_turbo with multiprocessing"""
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), len(lats))

    n_points = len(lats)
    chunk_size = max(1, n_points // n_workers)

    args_list = []
    for i in range(0, n_points, chunk_size):
        end_idx = min(i + chunk_size, n_points)
        chunk_lons = np.repeat(lons[i:end_idx], len(mjd))
        chunk_lats = np.repeat(lats[i:end_idx], len(mjd))
        chunk_mjd = np.tile(mjd, end_idx - i)
        args_list.append((chunk_lons, chunk_lats, chunk_mjd, MODEL, model_dir))

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(_worker_turbo, args_list))
    elapsed = time.perf_counter() - start
    return elapsed, n_workers


def run_benchmark(model_dir, size_name='medium'):
    """Run benchmark for specified size"""
    size_config = SIZES[size_name]
    n_points = size_config['n_points']
    n_times = size_config['n_times']
    label = size_config['label']

    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"Total calculations: {n_points * n_times:,}")
    print(f"{'='*60}")

    lats, lons, mjd = generate_test_data(n_points, n_times)

    results = {
        'config': size_config,
        'n_calculations': n_points * n_times,
        'methods': {}
    }

    # 1. pyTMD (with model load - baseline)
    print("\n1. pyTMD (with model load)...")
    times = []
    for i in range(N_REPEATS):
        t = run_pytmd(lats, lons, mjd, model_dir)
        times.append(t)
        print(f"   Run {i+1}: {t*1000:.1f} ms")
    results['methods']['pyTMD'] = {
        'times': times,
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
    }

    # 2. pyTMD (preloaded - computation only)
    print("\n2. pyTMD (preloaded)...")
    times = []
    for i in range(N_REPEATS):
        t = run_pytmd_preloaded(lats, lons, mjd, model_dir)
        times.append(t)
        print(f"   Run {i+1}: {t*1000:.1f} ms")
    results['methods']['pyTMD_preloaded'] = {
        'times': times,
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
    }

    # 3. pyTMD_turbo (with model loading - fair comparison with pyTMD)
    print("\n3. pyTMD_turbo (with model loading)...")
    times = []
    for i in range(N_REPEATS):
        t = run_turbo(lats, lons, mjd, model_dir, include_model_load=True)
        times.append(t)
        print(f"   Run {i+1}: {t*1000:.1f} ms")
    results['methods']['turbo_with_load'] = {
        'times': times,
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
    }

    # Reload model for subsequent tests
    pyTMD_turbo.init_model(MODEL, model_dir)

    # 4. pyTMD_turbo (pre-loaded model - computation only)
    print("\n4. pyTMD_turbo (preloaded)...")
    times = []
    for i in range(N_REPEATS):
        t = run_turbo(lats, lons, mjd, model_dir, include_model_load=False)
        times.append(t)
        print(f"   Run {i+1}: {t*1000:.1f} ms")
    results['methods']['turbo_preloaded'] = {
        'times': times,
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
    }

    # Calculate speedups (relative to pyTMD with load)
    pytmd_mean = results['methods']['pyTMD']['mean_ms']
    for method_name, method_data in results['methods'].items():
        if method_name != 'pyTMD':
            method_data['speedup'] = pytmd_mean / method_data['mean_ms']

    # Also calculate speedup vs pyTMD_preloaded for fair comparison
    pytmd_preloaded_mean = results['methods']['pyTMD_preloaded']['mean_ms']
    results['methods']['turbo_preloaded']['speedup_vs_pytmd_preloaded'] = (
        pytmd_preloaded_mean / results['methods']['turbo_preloaded']['mean_ms']
    )

    return results


def generate_html_report(all_results, output_path):
    """Generate HTML report"""

    method_names = {
        'pyTMD': 'pyTMD (with load)',
        'pyTMD_preloaded': 'pyTMD (preloaded)',
        'turbo_with_load': 'turbo (with load)',
        'turbo_preloaded': 'turbo (preloaded)',
    }

    method_colors = {
        'pyTMD': '#ff6b6b',
        'pyTMD_preloaded': '#ff9999',
        'turbo_with_load': '#45b7d1',
        'turbo_preloaded': '#4ecdc4',
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
                {all_results.get('timestamp', 'N/A')}
            </div>
            <div class="info-item">
                <strong>CPU Cores</strong>
                {all_results.get('cpu_count', multiprocessing.cpu_count())}
            </div>
            <div class="info-item">
                <strong>Model</strong>
                {MODEL}
            </div>
            <div class="info-item">
                <strong>Repeats</strong>
                {N_REPEATS} per test
            </div>
        </div>
        <div class="note">
            <strong>Important Notes:</strong>
            <ul style="margin: 5px 0 0 20px; padding: 0;">
                <li><strong>With load:</strong> Includes model loading time (~2.5-3.5s overhead)</li>
                <li><strong>Preloaded:</strong> Model/dataset pre-loaded, only computation time measured</li>
                <li><strong>Fair comparison:</strong> Compare pyTMD (preloaded) vs turbo (preloaded) for pure computation speed</li>
            </ul>
        </div>
    </div>

    <div class="method-legend">
'''
    for key, name in method_names.items():
        html += f'''        <div class="method-item">
            <div class="method-color" style="background: {method_colors[key]};"></div>
            <span>{name}</span>
        </div>
'''
    html += '    </div>\n'

    # Add benchmark sections
    for size_name in ['small', 'medium', 'large']:
        bench = all_results.get('benchmarks', {}).get(size_name, {})
        if not bench.get('methods'):
            continue

        config = bench.get('config', {})
        methods = bench.get('methods', {})

        pytmd_ms = methods.get('pyTMD', {}).get('mean_ms', 0)
        pytmd_pre_ms = methods.get('pyTMD_preloaded', {}).get('mean_ms', 0)
        turbo_load_ms = methods.get('turbo_with_load', {}).get('mean_ms', 0)
        turbo_preload_ms = methods.get('turbo_preloaded', {}).get('mean_ms', 0)

        pytmd_pre_speedup = methods.get('pyTMD_preloaded', {}).get('speedup', 1)
        turbo_load_speedup = methods.get('turbo_with_load', {}).get('speedup', 1)
        methods.get('turbo_preloaded', {}).get('speedup', 1)
        turbo_vs_pytmd_pre = methods.get('turbo_preloaded', {}).get('speedup_vs_pytmd_preloaded', 1)

        html += f'''
    <div class="card">
        <h2>{config.get('label', size_name)}</h2>
        <p>Total calculations: <strong>{bench.get('n_calculations', 0):,}</strong>
           ({config.get('n_points', 0)} points x {config.get('n_times', 0)} times)</p>

        <div class="summary-grid">
            <div class="summary-card pytmd">
                <div class="summary-value">{pytmd_ms:.0f}ms</div>
                <div class="summary-label">{method_names['pyTMD']}<br>(baseline)</div>
            </div>
            <div class="summary-card" style="background:#ffe5e5;">
                <div class="summary-value">{pytmd_pre_ms:.0f}ms</div>
                <div class="summary-label">{method_names['pyTMD_preloaded']}<br><span class="speedup">{pytmd_pre_speedup:.1f}x</span></div>
            </div>
            <div class="summary-card parallel">
                <div class="summary-value">{turbo_load_ms:.0f}ms</div>
                <div class="summary-label">{method_names['turbo_with_load']}<br><span class="speedup">{turbo_load_speedup:.1f}x</span></div>
            </div>
            <div class="summary-card single">
                <div class="summary-value">{turbo_preload_ms:.0f}ms</div>
                <div class="summary-label">{method_names['turbo_preloaded']}<br><span class="speedup">{turbo_vs_pytmd_pre:.1f}x vs pyTMD(pre)</span></div>
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
        for method_key in ['pyTMD', 'pyTMD_preloaded', 'turbo_with_load', 'turbo_preloaded']:
            m = methods.get(method_key, {})
            if not m:
                continue
            speedup = m.get('speedup', 1.0)
            speedup_str = f'{speedup:.1f}x' if method_key != 'pyTMD' else 'baseline'
            # For turbo_preloaded, also show speedup vs pyTMD_preloaded
            if method_key == 'turbo_preloaded':
                vs_pre = m.get('speedup_vs_pytmd_preloaded', 1.0)
                speedup_str += f' ({vs_pre:.1f}x vs pyTMD preloaded)'
            html += f'''            <tr>
                <td>{method_names.get(method_key, method_key)}</td>
                <td>{m.get('mean_ms', 0):.2f}</td>
                <td>{m.get('std_ms', 0):.2f}</td>
                <td class="speedup">{speedup_str}</td>
            </tr>
'''
        html += '''        </table>
    </div>
'''

    # JavaScript for charts
    html += '\n<script>\n'
    for size_name in ['small', 'medium', 'large']:
        bench = all_results.get('benchmarks', {}).get(size_name, {})
        if not bench.get('methods'):
            continue

        methods = bench.get('methods', {})
        method_keys = ['pyTMD', 'pyTMD_preloaded', 'turbo_with_load', 'turbo_preloaded']
        values = [methods.get(k, {}).get('mean_ms', 0) for k in method_keys]
        errors = [methods.get(k, {}).get('std_ms', 0) for k in method_keys]
        colors = [method_colors.get(k, '#888') for k in method_keys]
        labels = [method_names.get(k, k) for k in method_keys]

        html += f'''
Plotly.newPlot('chart-{size_name}', [{{
    x: {json.dumps(labels)},
    y: {json.dumps(values)},
    error_y: {{ type: 'data', array: {json.dumps(errors)}, visible: true }},
    type: 'bar',
    marker: {{ color: {json.dumps(colors)} }},
    text: {json.dumps([f'{v:.0f}ms' for v in values])},
    textposition: 'outside'
}}], {{
    title: '{size_name.capitalize()} - Execution Time (log scale)',
    yaxis: {{ title: 'Time (ms)', type: 'log' }},
    showlegend: false,
    margin: {{ t: 50, b: 100 }}
}}, {{ responsive: true }});
'''

    html += '</script>\n</body>\n</html>'

    with open(output_path, 'w') as f:
        f.write(html)
    return output_path


def main():
    model_dir = os.environ.get('PYTMD_RESOURCE', '~/Documents/python/eq/LargeFiles')
    model_dir = os.path.expanduser(model_dir)

    print(f"Model directory: {model_dir}")
    print(f"CPU cores: {multiprocessing.cpu_count()}")

    # Load model first (excluded from timing)
    print("\nPreloading models (excluded from timing)...")
    pyTMD_turbo.init_model(MODEL, model_dir)
    preload_pytmd(model_dir, MODEL)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'cpu_count': multiprocessing.cpu_count(),
        'model': MODEL,
        'benchmarks': {}
    }

    # Run benchmarks
    for size_name in ['small', 'medium', 'large']:
        print(f"\n\n{'#'*60}")
        print(f"Running {size_name} benchmark...")
        print(f"{'#'*60}")
        try:
            all_results['benchmarks'][size_name] = run_benchmark(model_dir, size_name)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            all_results['benchmarks'][size_name] = {'error': str(e)}

    # Print summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for size_name, bench in all_results['benchmarks'].items():
        if 'error' in bench:
            continue
        config = bench.get('config', {})
        methods = bench.get('methods', {})
        print(f"\n{config.get('label', size_name)}:")
        for method_key in ['pyTMD', 'pyTMD_preloaded', 'turbo_with_load', 'turbo_preloaded']:
            m = methods.get(method_key, {})
            if not m:
                continue
            speedup = m.get('speedup')
            speedup_str = f" ({speedup:.1f}x)" if speedup else ""
            # For turbo_preloaded, also show vs pyTMD_preloaded
            if method_key == 'turbo_preloaded':
                vs_pre = m.get('speedup_vs_pytmd_preloaded')
                if vs_pre:
                    speedup_str += f" [{vs_pre:.1f}x vs pyTMD_pre]"
            print(f"  {method_key:<25}: {m.get('mean_ms', 0):>10.0f} ms{speedup_str}")

    # Save results
    output_dir = Path(__file__).parent
    json_path = output_dir / 'benchmark_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON saved to: {json_path}")

    html_path = output_dir / 'benchmark_report.html'
    generate_html_report(all_results, str(html_path))
    print(f"HTML saved to: {html_path}")

    return all_results


if __name__ == "__main__":
    main()
