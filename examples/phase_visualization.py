"""
Phase Calculation and Visualization Example (Multi-constituent)

Demonstrates:
1. Calculating tidal phase using multiple constituents (M2, S2, K1, O1, N2)
2. Computing derivative (rate of change) from all constituents
3. Visualizing with tangent lines for validation

Location: 27.0744° N, 142.2178° E (near Ogasawara Islands)
Period: 2026-01-01 to 2026-01-04
"""

import sys
import time
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyTMD_turbo.compute import tide_elevations, init_model


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

# Constituents to use for fitting
CONSTITUENTS = ['m2', 's2', 'k1', 'o1', 'n2']

# Constituent periods (hours)
CONSTITUENT_PERIODS = {
    'm2': 12.4206012,
    's2': 12.0000000,
    'n2': 12.6583482,
    'k2': 11.9672348,
    'k1': 23.9344696,
    'o1': 25.8193417,
    'p1': 24.0658902,
    'q1': 26.8683567,
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_omega(constituent: str) -> float:
    """Get angular frequency (rad/s) for constituent"""
    period_hours = CONSTITUENT_PERIODS.get(constituent.lower())
    if period_hours is None:
        raise ValueError(f"Unknown constituent: {constituent}")
    return 2 * np.pi / (period_hours * 3600)


def datetime_to_mjd(dt: datetime) -> float:
    mjd_epoch = datetime(1858, 11, 17, 0, 0, 0)
    return (dt - mjd_epoch).total_seconds() / 86400.0


def mjd_to_datetime(mjd: float) -> datetime:
    mjd_epoch = datetime(1858, 11, 17, 0, 0, 0)
    return mjd_epoch + timedelta(days=mjd)


def fit_multi_constituent(t: np.ndarray, y: np.ndarray,
                          constituents: List[str]) -> Dict[str, dict]:
    """
    Fit multiple sinusoids simultaneously:
    y = sum_i A_i * sin(w_i * t + phi_i) + C
    """
    columns = []
    for const in constituents:
        omega = get_omega(const)
        columns.append(np.sin(omega * t))
        columns.append(np.cos(omega * t))
    columns.append(np.ones_like(t))

    X = np.column_stack(columns)
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    results = {}
    for i, const in enumerate(constituents):
        a = coeffs[2*i]
        b = coeffs[2*i + 1]
        results[const] = {
            'amplitude': float(np.sqrt(a**2 + b**2)),
            'phase': float(np.arctan2(b, a)),
            'omega': get_omega(const),
        }
    results['_offset'] = float(coeffs[-1])

    return results


def compute_derivative_multi(t_sec: float, fit_result: dict,
                              constituents: List[str]) -> tuple:
    """
    Compute total derivative from all constituents
    Returns (total_derivative, constituent_details)
    """
    details = {}
    total_deriv = 0.0

    for const in constituents:
        r = fit_result[const]
        phase = r['omega'] * t_sec + r['phase']
        phase_norm = phase % (2 * np.pi)
        cos_phase = np.cos(phase)
        deriv = r['amplitude'] * r['omega'] * cos_phase

        details[const] = {
            'phase': float(phase_norm),
            'phase_deg': float(np.degrees(phase_norm)),
            'cos_phase': float(cos_phase),
            'derivative': float(deriv),
        }
        total_deriv += deriv

    return total_deriv, details


def numerical_derivative(t: np.ndarray, y: np.ndarray, t_point: float) -> float:
    """Compute numerical derivative using central difference"""
    idx = np.argmin(np.abs(t - t_point))
    if idx == 0:
        return (y[1] - y[0]) / (t[1] - t[0])
    elif idx == len(t) - 1:
        return (y[-1] - y[-2]) / (t[-1] - t[-2])
    else:
        return (y[idx+1] - y[idx-1]) / (t[idx+1] - t[idx-1])


# =============================================================================
# Main
# =============================================================================

def main():
    timings = {}

    print("=" * 70)
    print("Phase Calculation and Visualization (Multi-constituent)")
    print("=" * 70)
    print(f"Location: {LAT}°N, {LON}°E")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Constituents: {CONSTITUENTS}")
    print()

    # Time range
    mjd_start = datetime_to_mjd(START_DATE)
    mjd_end = datetime_to_mjd(END_DATE)
    mjd_grid = np.linspace(mjd_start, mjd_end, N_GRID_POINTS)
    t_grid_seconds = (mjd_grid - mjd_start) * 86400

    # Random points
    np.random.seed(42)
    random_mjd = np.sort(np.random.uniform(mjd_start, mjd_end, N_RANDOM_POINTS))
    random_t_seconds = (random_mjd - mjd_start) * 86400

    print("Random time points:")
    for i, mjd in enumerate(random_mjd):
        dt = mjd_to_datetime(mjd)
        print(f"  Point {i+1}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Model initialization
    print("Loading model...")
    t0 = time.perf_counter()
    init_model(MODEL_NAME, MODEL_DIR)
    timings['init_model'] = time.perf_counter() - t0
    print(f"  Done: {timings['init_model']*1000:.1f} ms")

    # Tide computation
    print("Computing tides...")
    t0 = time.perf_counter()
    tide_grid = tide_elevations(
        np.full(N_GRID_POINTS, LON),
        np.full(N_GRID_POINTS, LAT),
        mjd_grid,
        model=MODEL_NAME
    )
    tide_random = tide_elevations(
        np.full(N_RANDOM_POINTS, LON),
        np.full(N_RANDOM_POINTS, LAT),
        random_mjd,
        model=MODEL_NAME
    )
    timings['tide_compute'] = time.perf_counter() - t0
    print(f"  Done: {timings['tide_compute']*1000:.1f} ms")

    # Multi-constituent fitting
    print("Fitting constituents...")
    t0 = time.perf_counter()
    fit_result = fit_multi_constituent(t_grid_seconds, tide_grid, CONSTITUENTS)
    timings['fitting'] = time.perf_counter() - t0
    print(f"  Done: {timings['fitting']*1000:.3f} ms")

    print("\nFitted constituents:")
    for const in CONSTITUENTS:
        r = fit_result[const]
        print(f"  {const.upper()}: A={r['amplitude']:.4f} m, φ={np.degrees(r['phase']):.1f}°")
    print(f"  Offset: {fit_result['_offset']:.4f} m")

    # Compute results for random points
    print("\nPhase and derivative at random points:")
    print("-" * 70)

    results = []
    for i, (mjd, t_sec, tide_val) in enumerate(zip(random_mjd, random_t_seconds, tide_random)):
        dt = mjd_to_datetime(mjd)

        total_deriv, const_details = compute_derivative_multi(t_sec, fit_result, CONSTITUENTS)
        deriv_numerical = numerical_derivative(t_grid_seconds, tide_grid, t_sec)

        result = {
            'datetime': dt,
            'mjd': mjd,
            't_seconds': t_sec,
            'tide': float(tide_val),
            'derivative_phase': float(total_deriv),
            'derivative_numerical': float(deriv_numerical),
            'constituents': const_details,
        }
        results.append(result)

        print(f"\nPoint {i+1}: {dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Tide height: {tide_val:.4f} m")
        for const in CONSTITUENTS:
            d = const_details[const]
            print(f"  {const.upper()}: phase={d['phase_deg']:6.1f}°, "
                  f"cos(φ)={d['cos_phase']:7.4f}, deriv={d['derivative']*3600:8.4f} m/h")
        print(f"  TOTAL derivative (phase):     {total_deriv*3600:.4f} m/h")
        print(f"  TOTAL derivative (numerical): {deriv_numerical*3600:.4f} m/h")
        print(f"  Difference: {abs(total_deriv - deriv_numerical)*3600:.4f} m/h")

    # Generate HTML
    print("\nGenerating HTML...")
    html = generate_html(mjd_grid, tide_grid, results, fit_result, mjd_start, timings)

    output_path = Path(__file__).parent / 'phase_visualization.html'
    output_path.write_text(html)
    print(f"Output: {output_path}")

    return results


# =============================================================================
# HTML Generation
# =============================================================================

def generate_constituent_breakdown(points_data, constituents):
    """Generate HTML for constituent breakdown details"""
    html_parts = []
    for i, p in enumerate(points_data):
        rows = []
        for const in constituents:
            c = p['constituents'][const]
            rows.append(f"""
                <tr>
                    <td>{const.upper()}</td>
                    <td>{c['phase_deg']:.1f}</td>
                    <td>{c['cos_phase']:.4f}</td>
                    <td>{c['derivative']*3600:.4f}</td>
                </tr>""")

        html_parts.append(f"""
        <details>
            <summary><strong>Point {i+1}: {p['datetime']}</strong></summary>
            <table style="margin: 10px 0;">
                <tr><th>Constituent</th><th>Phase (deg)</th><th>cos(phase)</th><th>Derivative (m/h)</th></tr>
                {''.join(rows)}
            </table>
        </details>""")

    return ''.join(html_parts)


def generate_html(mjd_grid, tide_grid, results, fit_result, mjd_start, timings):
    hours_grid = (mjd_grid - mjd_start) * 24
    tide_data = list(zip(hours_grid.tolist(), tide_grid.tolist()))

    points_data = []
    for r in results:
        hours = (r['mjd'] - mjd_start) * 24
        points_data.append({
            'hours': hours,
            'tide': r['tide'],
            'derivative': r['derivative_phase'] * 3600,
            'derivative_numerical': r['derivative_numerical'] * 3600,
            'datetime': r['datetime'].strftime('%Y-%m-%d %H:%M'),
            'constituents': r['constituents'],
        })

    constituents_data = []
    for const in CONSTITUENTS:
        r = fit_result[const]
        constituents_data.append({
            'name': const.upper(),
            'amplitude': r['amplitude'],
            'phase_deg': np.degrees(r['phase']),
            'period_hours': CONSTITUENT_PERIODS[const],
        })

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tidal Phase Visualization (Multi-constituent)</title>
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
        .info-item {{ background: #e7f3ff; padding: 10px; border-radius: 4px; }}
        .info-item strong {{ display: block; color: #007bff; font-size: 0.9em; }}
        #chart {{ width: 100%; height: 500px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background: #007bff; color: white; }}
        td:first-child {{ text-align: left; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .timing {{ background: #fff3cd; padding: 15px; border-radius: 4px; border-left: 4px solid #ffc107; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 15px 0; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-color {{ width: 20px; height: 3px; }}
        .constituent-table {{ margin-top: 15px; }}
        .good {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>Tidal Phase Visualization (Multi-constituent)</h1>
        <div class="info-grid">
            <div class="info-item">
                <strong>Location</strong>
                27.0744°N, 142.2178°E
            </div>
            <div class="info-item">
                <strong>Period</strong>
                2026-01-01 ~ 2026-01-04 (72h)
            </div>
            <div class="info-item">
                <strong>Model</strong>
                GOT5.5
            </div>
            <div class="info-item">
                <strong>Constituents</strong>
                {', '.join(c.upper() for c in CONSTITUENTS)}
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Tide Curve with Tangent Lines</h2>
        <div id="chart"></div>
        <div class="legend">
            <div class="legend-item">
                <span class="legend-color" style="background: #1f77b4;"></span>
                Tide (240 points)
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #ff7f0e; width: 10px; height: 10px; border-radius: 50%;"></span>
                Random Points
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #d62728;"></span>
                Tangent (Phase-based)
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #2ca02c;"></span>
                Tangent (Numerical)
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Fitted Constituents</h2>
        <table class="constituent-table">
            <tr>
                <th>Constituent</th>
                <th>Period (hours)</th>
                <th>Amplitude (m)</th>
                <th>Phase (°)</th>
            </tr>
            {''.join(f"""
            <tr>
                <td>{c['name']}</td>
                <td>{c['period_hours']:.4f}</td>
                <td>{c['amplitude']:.4f}</td>
                <td>{c['phase_deg']:.1f}</td>
            </tr>
            """ for c in constituents_data)}
            <tr style="background: #e7f3ff;">
                <td><strong>Offset</strong></td>
                <td>-</td>
                <td>{fit_result['_offset']:.4f}</td>
                <td>-</td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h2>Random Point Analysis</h2>
        <table>
            <tr>
                <th>Point</th>
                <th>DateTime</th>
                <th>Tide (m)</th>
                <th>Deriv Phase (m/h)</th>
                <th>Deriv Numerical (m/h)</th>
                <th>Difference (m/h)</th>
                <th>Match</th>
            </tr>
            {''.join(f"""
            <tr>
                <td>Point {i+1}</td>
                <td>{p['datetime']}</td>
                <td>{p['tide']:.4f}</td>
                <td>{p['derivative']:.4f}</td>
                <td>{p['derivative_numerical']:.4f}</td>
                <td>{abs(p['derivative'] - p['derivative_numerical']):.4f}</td>
                <td class="{'good' if abs(p['derivative'] - p['derivative_numerical']) < 0.02 else 'warning'}">
                    {'Good' if abs(p['derivative'] - p['derivative_numerical']) < 0.02 else 'Check'}
                </td>
            </tr>
            """ for i, p in enumerate(points_data))}
        </table>

        <h3>Constituent Breakdown</h3>
        {generate_constituent_breakdown(points_data, CONSTITUENTS)}
    </div>

    <div class="card timing">
        <h2>Computation Timing</h2>
        <table>
            <tr><th>Step</th><th>Time (ms)</th></tr>
            <tr><td>Model initialization</td><td>{timings.get('init_model', 0)*1000:.1f}</td></tr>
            <tr><td>Tide computation</td><td>{timings.get('tide_compute', 0)*1000:.1f}</td></tr>
            <tr><td>Constituent fitting</td><td>{timings.get('fitting', 0)*1000:.3f}</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>Verification Guide</h2>
        <ul>
            <li><strong>Blue curve:</strong> Actual tide prediction (240 points)</li>
            <li><strong>Red dashed line:</strong> Tangent from phase-based derivative (sum of all constituents)</li>
            <li><strong>Green dashed line:</strong> Tangent from numerical derivative (ground truth)</li>
            <li>If the red and green lines align, the phase calculation is correct</li>
        </ul>
    </div>
</div>

<script>
const tideData = {json.dumps(tide_data)};
const pointsData = {json.dumps(points_data)};

const traces = [];

// Tide curve
traces.push({{
    x: tideData.map(d => d[0]),
    y: tideData.map(d => d[1]),
    mode: 'lines',
    name: 'Tide',
    line: {{ color: '#1f77b4', width: 2 }}
}});

// Random points
traces.push({{
    x: pointsData.map(d => d.hours),
    y: pointsData.map(d => d.tide),
    mode: 'markers',
    name: 'Points',
    marker: {{ color: '#ff7f0e', size: 12 }},
    text: pointsData.map((d, i) => `Point ${{i+1}}<br>${{d.datetime}}`),
    hoverinfo: 'text'
}});

// Tangent lines
const tangentLen = 2.5;
pointsData.forEach((p, i) => {{
    // Phase-based tangent (red)
    traces.push({{
        x: [p.hours - tangentLen, p.hours + tangentLen],
        y: [p.tide - p.derivative * tangentLen, p.tide + p.derivative * tangentLen],
        mode: 'lines',
        name: i === 0 ? 'Tangent (Phase)' : undefined,
        legendgroup: 'phase',
        showlegend: i === 0,
        line: {{ color: '#d62728', width: 2, dash: 'dash' }}
    }});

    // Numerical tangent (green)
    traces.push({{
        x: [p.hours - tangentLen, p.hours + tangentLen],
        y: [p.tide - p.derivative_numerical * tangentLen, p.tide + p.derivative_numerical * tangentLen],
        mode: 'lines',
        name: i === 0 ? 'Tangent (Numerical)' : undefined,
        legendgroup: 'numerical',
        showlegend: i === 0,
        line: {{ color: '#2ca02c', width: 2, dash: 'dot' }}
    }});
}});

Plotly.newPlot('chart', traces, {{
    title: 'Tide Prediction with Multi-constituent Phase Analysis',
    xaxis: {{ title: 'Hours from 2026-01-01 00:00', gridcolor: '#eee' }},
    yaxis: {{ title: 'Tide Height (m)', gridcolor: '#eee' }},
    hovermode: 'closest',
    legend: {{ x: 0, y: 1.15, orientation: 'h' }}
}}, {{ responsive: true }});
</script>
</body>
</html>
'''
    return html


if __name__ == '__main__':
    main()
