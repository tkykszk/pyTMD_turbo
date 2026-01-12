"""
Phase Calculation and Visualization Example (Multiple Ocean Tide Models + SET)

Demonstrates:
1. Ocean Tide: GOT5.5, GOT5.6 (short-period), RE14 (long-period)
2. Solid Earth Tide (SET): Vertical displacement from solar/lunar gravity
3. Computing derivative (rate of change) for all
4. Visualizing with tangent lines for validation

Location: 27.0744° N, 142.2178° E (near Ogasawara Islands)
Period: 2026-01-01 to 2026-01-04
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyTMD_turbo.compute import SET_displacements, init_model, tide_elevations

# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = '~/Documents/python/eq/LargeFiles'

# Short-period ocean tide models (semi-diurnal, diurnal)
SHORT_PERIOD_MODELS = [
    {'name': 'GOT5.5', 'color': '#1f77b4', 'constituents': ['m2', 's2', 'k1', 'o1', 'n2']},
    {'name': 'GOT5.6', 'color': '#2ca02c', 'constituents': ['m2', 's2', 'k1', 'o1', 'n2']},
]

# Long-period ocean tide model (fortnightly, monthly)
LONG_PERIOD_MODEL = {'name': 'RE14', 'color': '#9467bd', 'constituents': ['mf', 'mm']}

# SET configuration
SET_CONSTITUENTS = ['m2', 's2', 'k1', 'o1']

LAT = 27.0744
LON = 142.2178

START_DATE = datetime(2026, 1, 1, 0, 0, 0)
END_DATE = datetime(2026, 1, 4, 0, 0, 0)

N_GRID_POINTS = 240
N_RANDOM_POINTS = 3

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
    # Long-period constituents
    'mf': 327.8599,      # 13.66 days
    'mm': 661.3092,      # 27.55 days
    'mt': 219.2068,      # 9.13 days (Mtm)
    'ssa': 4383.0521,    # 182.6 days
    'sa': 8766.1521,     # 365.25 days
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
                          constituents: list[str]) -> dict[str, dict]:
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
                              constituents: list[str]) -> tuple:
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
    print("Phase Calculation and Visualization (Multiple Models + SET)")
    print("=" * 70)
    print(f"Location: {LAT}°N, {LON}°E")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Short-period Models: {[m['name'] for m in SHORT_PERIOD_MODELS]}")
    print(f"Long-period Model: {LONG_PERIOD_MODEL['name']}")
    print(f"SET Constituents: {SET_CONSTITUENTS}")
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

    # ==========================================================================
    # Short-period Ocean Tide Computation
    # ==========================================================================
    short_period_data = {}

    for model_cfg in SHORT_PERIOD_MODELS:
        model_name = model_cfg['name']
        constituents = model_cfg['constituents']

        print(f"Loading {model_name}...")
        t0 = time.perf_counter()
        init_model(model_name, MODEL_DIR)
        timings[f'init_{model_name}'] = time.perf_counter() - t0
        print(f"  Init: {timings[f'init_{model_name}']*1000:.1f} ms")

        print("  Computing tides...")
        t0 = time.perf_counter()
        grid = tide_elevations(
            np.full(N_GRID_POINTS, LON),
            np.full(N_GRID_POINTS, LAT),
            mjd_grid,
            model=model_name
        )
        random_vals = tide_elevations(
            np.full(N_RANDOM_POINTS, LON),
            np.full(N_RANDOM_POINTS, LAT),
            random_mjd,
            model=model_name
        )
        timings[f'compute_{model_name}'] = time.perf_counter() - t0
        print(f"  Compute: {timings[f'compute_{model_name}']*1000:.1f} ms")

        # Fit constituents
        fit_result = fit_multi_constituent(t_grid_seconds, grid, constituents)

        # Compute results for random points
        results = []
        for mjd, t_sec, val in zip(random_mjd, random_t_seconds, random_vals):
            dt = mjd_to_datetime(mjd)
            deriv, details = compute_derivative_multi(t_sec, fit_result, constituents)
            deriv_num = numerical_derivative(t_grid_seconds, grid, t_sec)
            results.append({
                'datetime': dt,
                'mjd': mjd,
                't_seconds': t_sec,
                'value': float(val),
                'derivative_phase': float(deriv),
                'derivative_numerical': float(deriv_num),
                'constituents': details,
            })

        short_period_data[model_name] = {
            'grid': grid,
            'random': random_vals,
            'fit': fit_result,
            'results': results,
            'constituents': constituents,
            'color': model_cfg['color'],
        }

        print("  Fitted constituents:")
        for const in constituents:
            r = fit_result[const]
            print(f"    {const.upper()}: A={r['amplitude']*100:.2f} cm, φ={np.degrees(r['phase']):.1f}°")
        print(f"    Offset: {fit_result['_offset']*100:.2f} cm")
        print()

    # ==========================================================================
    # Long-period Ocean Tide (RE14) Computation
    # ==========================================================================
    model_name = LONG_PERIOD_MODEL['name']
    constituents = LONG_PERIOD_MODEL['constituents']

    print(f"Loading {model_name} (long-period)...")
    t0 = time.perf_counter()
    init_model(model_name, MODEL_DIR)
    timings[f'init_{model_name}'] = time.perf_counter() - t0
    print(f"  Init: {timings[f'init_{model_name}']*1000:.1f} ms")

    print("  Computing tides...")
    t0 = time.perf_counter()
    lp_grid = tide_elevations(
        np.full(N_GRID_POINTS, LON),
        np.full(N_GRID_POINTS, LAT),
        mjd_grid,
        model=model_name
    )
    lp_random = tide_elevations(
        np.full(N_RANDOM_POINTS, LON),
        np.full(N_RANDOM_POINTS, LAT),
        random_mjd,
        model=model_name
    )
    timings[f'compute_{model_name}'] = time.perf_counter() - t0
    print(f"  Compute: {timings[f'compute_{model_name}']*1000:.1f} ms")

    # For long-period, use numerical derivative only (fitting is unreliable for 72h)
    lp_results = []
    for mjd, t_sec, val in zip(random_mjd, random_t_seconds, lp_random):
        dt = mjd_to_datetime(mjd)
        deriv_num = numerical_derivative(t_grid_seconds, lp_grid, t_sec)
        lp_results.append({
            'datetime': dt,
            'mjd': mjd,
            't_seconds': t_sec,
            'value': float(val),
            'derivative_numerical': float(deriv_num),
        })

    long_period_data = {
        'grid': lp_grid,
        'random': lp_random,
        'results': lp_results,
        'color': LONG_PERIOD_MODEL['color'],
    }

    print(f"  Long-period tide range: {lp_grid.min()*100:.2f} ~ {lp_grid.max()*100:.2f} cm")
    print()

    # ==========================================================================
    # Solid Earth Tide Computation
    # ==========================================================================
    print("Computing Solid Earth Tides...")
    t0 = time.perf_counter()

    _, _, set_grid = SET_displacements(
        np.array([LON]),
        np.array([LAT]),
        mjd_grid
    )
    set_grid = set_grid.flatten()

    _, _, set_random = SET_displacements(
        np.array([LON]),
        np.array([LAT]),
        random_mjd
    )
    set_random = set_random.flatten()

    timings['set_compute'] = time.perf_counter() - t0
    print(f"  Done: {timings['set_compute']*1000:.1f} ms")

    # Fit SET constituents
    set_fit = fit_multi_constituent(t_grid_seconds, set_grid, SET_CONSTITUENTS)

    set_results = []
    for mjd, t_sec, val in zip(random_mjd, random_t_seconds, set_random):
        dt = mjd_to_datetime(mjd)
        deriv, details = compute_derivative_multi(t_sec, set_fit, SET_CONSTITUENTS)
        deriv_num = numerical_derivative(t_grid_seconds, set_grid, t_sec)
        set_results.append({
            'datetime': dt,
            'mjd': mjd,
            't_seconds': t_sec,
            'value': float(val),
            'derivative_phase': float(deriv),
            'derivative_numerical': float(deriv_num),
            'constituents': details,
        })

    print("  SET fitted constituents:")
    for const in SET_CONSTITUENTS:
        r = set_fit[const]
        print(f"    {const.upper()}: A={r['amplitude']*1000:.3f} mm, φ={np.degrees(r['phase']):.1f}°")
    print(f"    Offset: {set_fit['_offset']*1000:.3f} mm")

    # ==========================================================================
    # Print summary for random points
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Summary at random points")
    print("=" * 70)

    for i, mjd in enumerate(random_mjd):
        dt = mjd_to_datetime(mjd)
        print(f"\nPoint {i+1}: {dt.strftime('%Y-%m-%d %H:%M')}")
        print("-" * 50)

        for model_name, data in short_period_data.items():
            r = data['results'][i]
            print(f"  {model_name}: {r['value']*100:.2f} cm, deriv={r['derivative_phase']*3600*100:.3f} cm/h")

        lpr = long_period_data['results'][i]
        print(f"  RE14 (long): {lpr['value']*100:.3f} cm, deriv={lpr['derivative_numerical']*3600*100:.4f} cm/h")

        sr = set_results[i]
        print(f"  SET (Up): {sr['value']*1000:.3f} mm, deriv={sr['derivative_phase']*3600*1000:.4f} mm/h")

    # Generate HTML
    print("\nGenerating HTML...")
    html = generate_html(
        mjd_grid, t_grid_seconds, short_period_data, long_period_data,
        set_grid, set_results, set_fit,
        mjd_start, timings
    )

    output_path = Path(__file__).parent / 'phase_visualization.html'
    output_path.write_text(html)
    print(f"Output: {output_path}")

    return short_period_data, long_period_data, set_results


# =============================================================================
# HTML Generation
# =============================================================================

def generate_constituent_breakdown(points_data, constituents, unit, scale):
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
                    <td>{c['derivative']*3600*scale:.4f}</td>
                </tr>""")

        html_parts.append(f"""
        <details>
            <summary><strong>Point {i+1}: {p['datetime']}</strong></summary>
            <table style="margin: 10px 0;">
                <tr><th>Constituent</th><th>Phase (°)</th><th>cos(φ)</th><th>Deriv ({unit}/h)</th></tr>
                {''.join(rows)}
            </table>
        </details>""")

    return ''.join(html_parts)


def find_phase_zero_crossings(fit_result: dict, constituents: list[str],
                                t_max_seconds: float) -> dict[str, list[float]]:
    """
    Find times (in hours from start) where each constituent's phase crosses 0 (mod 2π).

    Phase: φ(t) = ω*t + φ₀
    Zero crossing when: ω*t + φ₀ = 2πn
    Solving: t = (2πn - φ₀) / ω

    Returns dict mapping constituent name to list of crossing times in hours.
    """
    crossings = {}
    t_max_seconds / 3600

    for const in constituents:
        r = fit_result[const]
        omega = r['omega']  # rad/s
        phase0 = r['phase']  # initial phase offset

        # Find all n such that t = (2πn - φ₀) / ω is in [0, t_max_seconds]
        # t >= 0: 2πn - φ₀ >= 0 → n >= φ₀/(2π)
        # t <= t_max: 2πn - φ₀ <= ω*t_max → n <= (ω*t_max + φ₀)/(2π)

        n_min = int(np.ceil(phase0 / (2 * np.pi)))
        n_max = int(np.floor((omega * t_max_seconds + phase0) / (2 * np.pi)))

        const_crossings = []
        for n in range(n_min, n_max + 1):
            t_sec = (2 * np.pi * n - phase0) / omega
            if 0 <= t_sec <= t_max_seconds:
                t_hours = t_sec / 3600
                const_crossings.append(t_hours)

        crossings[const] = const_crossings

    return crossings


def generate_html(mjd_grid, t_grid_seconds, short_period_data, long_period_data,
                  set_grid, set_results, set_fit, mjd_start, timings):
    hours_grid = (mjd_grid - mjd_start) * 24
    t_max_seconds = t_grid_seconds[-1]

    # Prepare short-period data for JSON
    sp_json_data = {}
    for model_name, data in short_period_data.items():
        sp_json_data[model_name] = {
            'curve': list(zip(hours_grid.tolist(), (data['grid'] * 100).tolist())),
            'points': [],
            'color': data['color'],
        }
        for r in data['results']:
            hours = (r['mjd'] - mjd_start) * 24
            sp_json_data[model_name]['points'].append({
                'hours': hours,
                'value': r['value'] * 100,
                'derivative': r['derivative_phase'] * 3600 * 100,
                'derivative_numerical': r['derivative_numerical'] * 3600 * 100,
                'datetime': r['datetime'].strftime('%Y-%m-%d %H:%M'),
            })

    # Long-period data
    lp_json_data = {
        'curve': list(zip(hours_grid.tolist(), (long_period_data['grid'] * 100).tolist())),
        'points': [],
        'color': long_period_data['color'],
    }
    for r in long_period_data['results']:
        hours = (r['mjd'] - mjd_start) * 24
        lp_json_data['points'].append({
            'hours': hours,
            'value': r['value'] * 100,
            'derivative_numerical': r['derivative_numerical'] * 3600 * 100,
            'datetime': r['datetime'].strftime('%Y-%m-%d %H:%M'),
        })

    # SET data
    set_json_data = {
        'curve': list(zip(hours_grid.tolist(), (set_grid * 1000).tolist())),
        'points': [],
    }
    for r in set_results:
        hours = (r['mjd'] - mjd_start) * 24
        set_json_data['points'].append({
            'hours': hours,
            'value': r['value'] * 1000,
            'derivative': r['derivative_phase'] * 3600 * 1000,
            'derivative_numerical': r['derivative_numerical'] * 3600 * 1000,
            'datetime': r['datetime'].strftime('%Y-%m-%d %H:%M'),
        })

    # Calculate phase=0 crossings for vertical lines
    # Use first short-period model's fit for ocean tide crossings
    first_sp_model = next(iter(short_period_data.keys()))
    first_sp_fit = short_period_data[first_sp_model]['fit']
    first_sp_constituents = short_period_data[first_sp_model]['constituents']
    sp_phase_crossings = find_phase_zero_crossings(first_sp_fit, first_sp_constituents, t_max_seconds)

    # SET phase crossings
    set_phase_crossings = find_phase_zero_crossings(set_fit, SET_CONSTITUENTS, t_max_seconds)

    # Prepare crossing data for JSON - use dominant constituent (M2 for ocean/SET)
    # For visibility, show M2 crossings as they're the dominant constituent
    sp_m2_crossings = sp_phase_crossings.get('m2', [])
    set_m2_crossings = set_phase_crossings.get('m2', [])

    phase_crossings_json = {
        'sp_m2': sp_m2_crossings,
        'set_m2': set_m2_crossings,
        'sp_all': sp_phase_crossings,
        'set_all': set_phase_crossings,
    }

    # Generate constituent tables for short-period models
    sp_constituent_tables = []
    for model_name, data in short_period_data.items():
        rows = []
        for const in data['constituents']:
            r = data['fit'][const]
            rows.append(f"""
                <tr>
                    <td>{const.upper()}</td>
                    <td>{CONSTITUENT_PERIODS[const]:.4f}</td>
                    <td>{r['amplitude']*100:.2f}</td>
                    <td>{np.degrees(r['phase']):.1f}</td>
                </tr>""")
        rows.append(f"""
            <tr style="background: #e7f3ff;">
                <td><strong>Offset</strong></td>
                <td>-</td>
                <td>{data['fit']['_offset']*100:.2f}</td>
                <td>-</td>
            </tr>""")

        sp_constituent_tables.append(f"""
        <div class="card" style="flex: 1; min-width: 280px;">
            <h3 style="color: {data['color']}; border-bottom: 2px solid {data['color']};">{model_name}</h3>
            <table class="constituent-table">
                <tr><th>Constituent</th><th>Period (h)</th><th>Amp (cm)</th><th>Phase (°)</th></tr>
                {''.join(rows)}
            </table>
        </div>""")

    # Generate analysis tables for short-period models
    sp_analysis_tables = []
    for model_name, data in short_period_data.items():
        rows = []
        for i, p in enumerate(data['results']):
            diff = abs(p['derivative_phase'] - p['derivative_numerical']) * 3600 * 100
            match_class = 'good' if diff < 0.5 else 'warning'
            rows.append(f"""
                <tr>
                    <td>Pt {i+1}</td>
                    <td>{p['datetime'].strftime('%m-%d %H:%M')}</td>
                    <td>{p['value']*100:.2f}</td>
                    <td>{p['derivative_phase']*3600*100:.3f}</td>
                    <td>{p['derivative_numerical']*3600*100:.3f}</td>
                    <td class="{match_class}">{'✓' if diff < 0.5 else '△'}</td>
                </tr>""")

        sp_analysis_tables.append(f"""
        <div class="card" style="flex: 1; min-width: 320px;">
            <h3 style="color: {data['color']};">{model_name} Analysis</h3>
            <table>
                <tr><th>Pt</th><th>DateTime</th><th>cm</th><th>Phase</th><th>Num</th><th></th></tr>
                {''.join(rows)}
            </table>
            <h4>Constituent Breakdown</h4>
            {generate_constituent_breakdown(
                [{'datetime': r['datetime'].strftime('%Y-%m-%d %H:%M'), 'constituents': r['constituents']}
                 for r in data['results']],
                data['constituents'], 'cm', 100
            )}
        </div>""")

    # Long-period analysis table
    lp_analysis_rows = []
    for i, p in enumerate(long_period_data['results']):
        lp_analysis_rows.append(f"""
            <tr>
                <td>Pt {i+1}</td>
                <td>{p['datetime'].strftime('%m-%d %H:%M')}</td>
                <td>{p['value']*100:.3f}</td>
                <td>{p['derivative_numerical']*3600*100:.4f}</td>
            </tr>""")

    # SET constituent table
    set_const_rows = []
    for const in SET_CONSTITUENTS:
        r = set_fit[const]
        set_const_rows.append(f"""
            <tr>
                <td>{const.upper()}</td>
                <td>{CONSTITUENT_PERIODS[const]:.4f}</td>
                <td>{r['amplitude']*1000:.3f}</td>
                <td>{np.degrees(r['phase']):.1f}</td>
            </tr>""")

    # SET analysis table
    set_analysis_rows = []
    for i, p in enumerate(set_results):
        diff = abs(p['derivative_phase'] - p['derivative_numerical']) * 3600 * 1000
        match_class = 'good' if diff < 1.0 else 'warning'
        set_analysis_rows.append(f"""
            <tr>
                <td>Pt {i+1}</td>
                <td>{p['datetime'].strftime('%m-%d %H:%M')}</td>
                <td>{p['value']*1000:.2f}</td>
                <td>{p['derivative_phase']*3600*1000:.3f}</td>
                <td>{p['derivative_numerical']*3600*1000:.3f}</td>
                <td class="{match_class}">{'✓' if diff < 1.0 else '△'}</td>
            </tr>""")

    # Timing table
    timing_rows = []
    for model_name in short_period_data:
        timing_rows.append(f"<tr><td>{model_name} init</td><td>{timings.get(f'init_{model_name}', 0)*1000:.1f}</td></tr>")
        timing_rows.append(f"<tr><td>{model_name} compute</td><td>{timings.get(f'compute_{model_name}', 0)*1000:.1f}</td></tr>")
    timing_rows.append(f"<tr><td>RE14 init</td><td>{timings.get('init_RE14', 0)*1000:.1f}</td></tr>")
    timing_rows.append(f"<tr><td>RE14 compute</td><td>{timings.get('compute_RE14', 0)*1000:.1f}</td></tr>")
    timing_rows.append(f"<tr><td>SET compute</td><td>{timings.get('set_compute', 0)*1000:.1f}</td></tr>")

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tidal Phase Visualization (Multi-Model + SET)</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .card {{
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
        }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; margin-top: 0; }}
        h2 {{ color: #555; margin-top: 0; }}
        h3 {{ color: #666; margin-top: 0; }}
        h4 {{ color: #888; margin: 10px 0 5px 0; font-size: 0.9em; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }}
        .info-item {{ background: #e7f3ff; padding: 10px; border-radius: 4px; }}
        .info-item.lp {{ background: #f3e5f5; }}
        .info-item.set {{ background: #fff3e0; }}
        .info-item strong {{ display: block; color: #007bff; font-size: 0.9em; }}
        .info-item.lp strong {{ color: #7b1fa2; }}
        .info-item.set strong {{ color: #e65100; }}
        .chart {{ width: 100%; height: 400px; }}
        .chart-small {{ width: 100%; height: 300px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; }}
        th {{ background: #007bff; color: white; }}
        th.lp {{ background: #7b1fa2; }}
        th.set {{ background: #e65100; }}
        td:first-child {{ text-align: left; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .timing {{ background: #fff3cd; padding: 15px; border-radius: 4px; border-left: 4px solid #ffc107; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 15px 0; font-size: 0.9em; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-color {{ width: 20px; height: 3px; }}
        .constituent-table {{ margin-top: 10px; }}
        .good {{ color: #28a745; font-weight: bold; }}
        .warning {{ color: #ffc107; }}
        .flex-row {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
        .ocean-header {{ color: #1f77b4; border-bottom: 2px solid #1f77b4; }}
        .lp-header {{ color: #9467bd; border-bottom: 2px solid #9467bd; }}
        .set-header {{ color: #ff7f0e; border-bottom: 2px solid #ff7f0e; }}
        details {{ margin: 5px 0; }}
        .phase-toggle {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }}
        .phase-btn {{
            padding: 4px 12px; border-radius: 4px; cursor: pointer;
            font-size: 0.85em; border: 2px solid; transition: all 0.2s;
            font-weight: 500;
        }}
        .phase-btn.active {{ color: white; }}
        .phase-btn:not(.active) {{ background: white; opacity: 0.6; }}
        .phase-btn:hover {{ opacity: 1; }}
        summary {{ cursor: pointer; padding: 5px; background: #f0f0f0; border-radius: 4px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>Tidal Phase Visualization (GOT5.5 + GOT5.6 + RE14 + SET)</h1>
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
                <strong>Short-period</strong>
                GOT5.5, GOT5.6
            </div>
            <div class="info-item lp">
                <strong>Long-period</strong>
                RE14 (Mf, Mm)
            </div>
            <div class="info-item set">
                <strong>SET</strong>
                M2, S2, K1, O1
            </div>
        </div>
    </div>

    <div class="card">
        <h2 class="ocean-header">Short-period Ocean Tide (GOT5.5 / GOT5.6)</h2>
        <div id="sp-chart" class="chart"></div>
        <div class="legend">
            {''.join(f'<div class="legend-item"><span class="legend-color" style="background: {data["color"]};"></span>{name}</div>' for name, data in short_period_data.items())}
            <div class="legend-item"><span class="legend-color" style="background: #d62728;"></span>Tangent (Phase)</div>
            <div class="legend-item"><span class="legend-color" style="background: #17becf;"></span>Tangent (Numerical)</div>
        </div>
        <div class="phase-toggle" id="sp-phase-toggle">
            <strong style="margin-right: 5px; line-height: 28px;">Phase=0:</strong>
            <button class="phase-btn active" data-const="m2" style="border-color: #e41a1c; background: #e41a1c;" onclick="toggleSpPhase('m2', this)">M2</button>
            <button class="phase-btn active" data-const="s2" style="border-color: #377eb8; background: #377eb8;" onclick="toggleSpPhase('s2', this)">S2</button>
            <button class="phase-btn active" data-const="k1" style="border-color: #4daf4a; background: #4daf4a;" onclick="toggleSpPhase('k1', this)">K1</button>
            <button class="phase-btn active" data-const="o1" style="border-color: #984ea3; background: #984ea3;" onclick="toggleSpPhase('o1', this)">O1</button>
            <button class="phase-btn active" data-const="n2" style="border-color: #ff7f00; background: #ff7f00;" onclick="toggleSpPhase('n2', this)">N2</button>
        </div>
    </div>

    <div class="two-col">
        <div class="card">
            <h2 class="lp-header">Long-period Ocean Tide (RE14)</h2>
            <div id="lp-chart" class="chart-small"></div>
            <p style="font-size: 0.85em; color: #666;">
                Long-period tides (Mf: 13.66 days, Mm: 27.55 days) show slow variations.
                Only numerical derivative is shown (phase fitting unreliable for 72h window).
            </p>
        </div>

        <div class="card">
            <h2 class="set-header">Solid Earth Tide (Up Component)</h2>
            <div id="set-chart" class="chart-small"></div>
            <div class="legend">
                <div class="legend-item"><span class="legend-color" style="background: #ff7f0e;"></span>SET Up</div>
                <div class="legend-item"><span class="legend-color" style="background: #d62728;"></span>Tangent (Phase)</div>
                <div class="legend-item"><span class="legend-color" style="background: #17becf;"></span>Tangent (Numerical)</div>
            </div>
            <div class="phase-toggle" id="set-phase-toggle">
                <strong style="margin-right: 5px; line-height: 28px;">Phase=0:</strong>
                <button class="phase-btn active" data-const="m2" style="border-color: #e41a1c; background: #e41a1c;" onclick="toggleSetPhase('m2', this)">M2</button>
                <button class="phase-btn active" data-const="s2" style="border-color: #377eb8; background: #377eb8;" onclick="toggleSetPhase('s2', this)">S2</button>
                <button class="phase-btn active" data-const="k1" style="border-color: #4daf4a; background: #4daf4a;" onclick="toggleSetPhase('k1', this)">K1</button>
                <button class="phase-btn active" data-const="o1" style="border-color: #984ea3; background: #984ea3;" onclick="toggleSetPhase('o1', this)">O1</button>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Fitted Constituents</h2>
        <div class="flex-row">
            {''.join(sp_constituent_tables)}
            <div class="card" style="flex: 1; min-width: 280px; background: #fff8f0;">
                <h3 style="color: #e65100; border-bottom: 2px solid #e65100;">SET</h3>
                <table class="constituent-table">
                    <tr><th class="set">Constituent</th><th class="set">Period (h)</th><th class="set">Amp (mm)</th><th class="set">Phase (°)</th></tr>
                    {''.join(set_const_rows)}
                    <tr style="background: #fff3e0;">
                        <td><strong>Offset</strong></td>
                        <td>-</td>
                        <td>{set_fit['_offset']*1000:.3f}</td>
                        <td>-</td>
                    </tr>
                </table>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Random Point Analysis</h2>
        <div class="flex-row">
            {''.join(sp_analysis_tables)}
            <div class="card" style="flex: 1; min-width: 280px; background: #faf0ff;">
                <h3 style="color: #9467bd;">RE14 (Long-period)</h3>
                <table>
                    <tr><th class="lp">Pt</th><th class="lp">DateTime</th><th class="lp">cm</th><th class="lp">Num Deriv</th></tr>
                    {''.join(lp_analysis_rows)}
                </table>
            </div>
            <div class="card" style="flex: 1; min-width: 320px; background: #fff8f0;">
                <h3 style="color: #e65100;">SET Analysis</h3>
                <table>
                    <tr><th class="set">Pt</th><th class="set">DateTime</th><th class="set">mm</th><th class="set">Phase</th><th class="set">Num</th><th class="set"></th></tr>
                    {''.join(set_analysis_rows)}
                </table>
                <h4>Constituent Breakdown</h4>
                {generate_constituent_breakdown(
                    [{'datetime': r['datetime'].strftime('%Y-%m-%d %H:%M'), 'constituents': r['constituents']}
                     for r in set_results],
                    SET_CONSTITUENTS, 'mm', 1000
                )}
            </div>
        </div>
    </div>

    <div class="card timing">
        <h2>Computation Timing</h2>
        <table style="max-width: 400px;">
            <tr><th>Step</th><th>Time (ms)</th></tr>
            {''.join(timing_rows)}
        </table>
    </div>
</div>

<script>
const spData = {json.dumps(sp_json_data)};
const lpData = {json.dumps(lp_json_data)};
const setData = {json.dumps(set_json_data)};
const phaseCrossings = {json.dumps(phase_crossings_json)};

// Constituent colors for phase=0 lines
const constituentColors = {{
    'm2': '#e41a1c',
    's2': '#377eb8',
    'k1': '#4daf4a',
    'o1': '#984ea3',
    'n2': '#ff7f00'
}};

// Track which constituents are visible
const spVisibleConstituents = {{ m2: true, s2: true, k1: true, o1: true, n2: true }};
const setVisibleConstituents = {{ m2: true, s2: true, k1: true, o1: true }};

// Store y-axis ranges
let spYRange = {{ min: 0, max: 0 }};
let setYRange = {{ min: 0, max: 0 }};

// Create vertical line shapes for selected constituents
function createPhaseZeroShapes(crossings, yMin, yMax, visibleConstituents) {{
    const shapes = [];
    Object.entries(crossings).forEach(([const_name, times]) => {{
        if (!visibleConstituents[const_name]) return;
        const color = constituentColors[const_name] || '#888888';
        times.forEach(t => {{
            shapes.push({{
                type: 'line',
                x0: t, x1: t,
                y0: yMin, y1: yMax,
                line: {{ color: color, width: 1.5, dash: 'dot' }},
                opacity: 0.6
            }});
        }});
    }});
    return shapes;
}}

// Toggle short-period phase lines
function toggleSpPhase(constituent, btn) {{
    spVisibleConstituents[constituent] = !spVisibleConstituents[constituent];
    btn.classList.toggle('active');
    if (!spVisibleConstituents[constituent]) {{
        btn.style.background = 'white';
        btn.style.color = btn.style.borderColor;
    }} else {{
        btn.style.background = btn.style.borderColor;
        btn.style.color = 'white';
    }}
    updateSpShapes();
}}

// Toggle SET phase lines
function toggleSetPhase(constituent, btn) {{
    setVisibleConstituents[constituent] = !setVisibleConstituents[constituent];
    btn.classList.toggle('active');
    if (!setVisibleConstituents[constituent]) {{
        btn.style.background = 'white';
        btn.style.color = btn.style.borderColor;
    }} else {{
        btn.style.background = btn.style.borderColor;
        btn.style.color = 'white';
    }}
    updateSetShapes();
}}

// Update short-period chart shapes
function updateSpShapes() {{
    const shapes = createPhaseZeroShapes(phaseCrossings.sp_all, spYRange.min, spYRange.max, spVisibleConstituents);
    Plotly.relayout('sp-chart', {{ shapes: shapes }});
}}

// Update SET chart shapes
function updateSetShapes() {{
    const shapes = createPhaseZeroShapes(phaseCrossings.set_all, setYRange.min, setYRange.max, setVisibleConstituents);
    Plotly.relayout('set-chart', {{ shapes: shapes }});
}}

// Short-period Ocean Tide Chart
function createSpChart() {{
    const traces = [];

    Object.entries(spData).forEach(([name, data]) => {{
        traces.push({{
            x: data.curve.map(d => d[0]),
            y: data.curve.map(d => d[1]),
            mode: 'lines',
            name: name,
            line: {{ color: data.color, width: 2 }}
        }});

        traces.push({{
            x: data.points.map(d => d.hours),
            y: data.points.map(d => d.value),
            mode: 'markers',
            name: name + ' pts',
            showlegend: false,
            marker: {{ color: data.color, size: 10 }}
        }});
    }});

    const firstModel = Object.values(spData)[0];
    const tangentLen = 2.5;
    firstModel.points.forEach((p, i) => {{
        traces.push({{
            x: [p.hours - tangentLen, p.hours + tangentLen],
            y: [p.value - p.derivative * tangentLen, p.value + p.derivative * tangentLen],
            mode: 'lines',
            name: i === 0 ? 'Phase' : undefined,
            legendgroup: 'phase',
            showlegend: i === 0,
            line: {{ color: '#d62728', width: 2, dash: 'dash' }}
        }});
        traces.push({{
            x: [p.hours - tangentLen, p.hours + tangentLen],
            y: [p.value - p.derivative_numerical * tangentLen, p.value + p.derivative_numerical * tangentLen],
            mode: 'lines',
            name: i === 0 ? 'Numerical' : undefined,
            legendgroup: 'numerical',
            showlegend: i === 0,
            line: {{ color: '#17becf', width: 2, dash: 'dot' }}
        }});
    }});

    // Calculate y-axis range for vertical lines
    const allYValues = Object.values(spData).flatMap(d => d.curve.map(c => c[1]));
    spYRange.min = Math.min(...allYValues) * 1.1;
    spYRange.max = Math.max(...allYValues) * 1.1;

    // Add phase=0 vertical lines (showing all constituents)
    const shapes = createPhaseZeroShapes(phaseCrossings.sp_all, spYRange.min, spYRange.max, spVisibleConstituents);

    Plotly.newPlot('sp-chart', traces, {{
        xaxis: {{ title: 'Hours from 2026-01-01 00:00', gridcolor: '#eee' }},
        yaxis: {{ title: 'Ocean Tide (cm)', gridcolor: '#eee' }},
        hovermode: 'closest',
        legend: {{ x: 0, y: 1.1, orientation: 'h' }},
        margin: {{ t: 40 }},
        shapes: shapes
    }}, {{ responsive: true }});
}}

// Long-period Ocean Tide Chart
function createLpChart() {{
    const traces = [];

    traces.push({{
        x: lpData.curve.map(d => d[0]),
        y: lpData.curve.map(d => d[1]),
        mode: 'lines',
        name: 'RE14',
        line: {{ color: lpData.color, width: 2 }}
    }});

    traces.push({{
        x: lpData.points.map(d => d.hours),
        y: lpData.points.map(d => d.value),
        mode: 'markers',
        name: 'Points',
        marker: {{ color: '#1f77b4', size: 10 }}
    }});

    const tangentLen = 4;
    lpData.points.forEach((p, i) => {{
        traces.push({{
            x: [p.hours - tangentLen, p.hours + tangentLen],
            y: [p.value - p.derivative_numerical * tangentLen, p.value + p.derivative_numerical * tangentLen],
            mode: 'lines',
            name: i === 0 ? 'Numerical' : undefined,
            legendgroup: 'numerical',
            showlegend: i === 0,
            line: {{ color: '#17becf', width: 2, dash: 'dot' }}
        }});
    }});

    Plotly.newPlot('lp-chart', traces, {{
        xaxis: {{ title: 'Hours', gridcolor: '#eee' }},
        yaxis: {{ title: 'Long-period Tide (cm)', gridcolor: '#eee' }},
        hovermode: 'closest',
        legend: {{ x: 0, y: 1.15, orientation: 'h' }},
        margin: {{ t: 30 }}
    }}, {{ responsive: true }});
}}

// SET Chart
function createSetChart() {{
    const traces = [];

    traces.push({{
        x: setData.curve.map(d => d[0]),
        y: setData.curve.map(d => d[1]),
        mode: 'lines',
        name: 'SET Up',
        line: {{ color: '#ff7f0e', width: 2 }}
    }});

    traces.push({{
        x: setData.points.map(d => d.hours),
        y: setData.points.map(d => d.value),
        mode: 'markers',
        name: 'Points',
        marker: {{ color: '#1f77b4', size: 10 }}
    }});

    const tangentLen = 2.5;
    setData.points.forEach((p, i) => {{
        traces.push({{
            x: [p.hours - tangentLen, p.hours + tangentLen],
            y: [p.value - p.derivative * tangentLen, p.value + p.derivative * tangentLen],
            mode: 'lines',
            name: i === 0 ? 'Phase' : undefined,
            legendgroup: 'phase',
            showlegend: i === 0,
            line: {{ color: '#d62728', width: 2, dash: 'dash' }}
        }});
        traces.push({{
            x: [p.hours - tangentLen, p.hours + tangentLen],
            y: [p.value - p.derivative_numerical * tangentLen, p.value + p.derivative_numerical * tangentLen],
            mode: 'lines',
            name: i === 0 ? 'Numerical' : undefined,
            legendgroup: 'numerical',
            showlegend: i === 0,
            line: {{ color: '#17becf', width: 2, dash: 'dot' }}
        }});
    }});

    // Calculate y-axis range for vertical lines
    const setYValues = setData.curve.map(c => c[1]);
    setYRange.min = Math.min(...setYValues) * 1.1;
    setYRange.max = Math.max(...setYValues) * 1.1;

    // Add phase=0 vertical lines (showing all SET constituents)
    const setShapes = createPhaseZeroShapes(phaseCrossings.set_all, setYRange.min, setYRange.max, setVisibleConstituents);

    Plotly.newPlot('set-chart', traces, {{
        xaxis: {{ title: 'Hours', gridcolor: '#eee' }},
        yaxis: {{ title: 'SET Up (mm)', gridcolor: '#eee' }},
        hovermode: 'closest',
        legend: {{ x: 0, y: 1.15, orientation: 'h' }},
        margin: {{ t: 30 }},
        shapes: setShapes
    }}, {{ responsive: true }});
}}

createSpChart();
createLpChart();
createSetChart();
</script>
</body>
</html>
'''
    return html


if __name__ == '__main__':
    main()
