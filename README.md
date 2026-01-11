# pyTMD_turbo

> **Technical Preview**
>
> This is a **Technical Preview** version of pyTMD_turbo. APIs may change without notice.
> Not recommended for production use. Feedback and bug reports are welcome.

High-performance tidal prediction module - a derivative work of [pyTMD](https://github.com/tsutterley/pyTMD).

## Status

| Item | Status |
|------|--------|
| Version | 0.1.0 (Technical Preview) |
| Implementation | ~58% of pyTMD features |
| Tests | 119 passed |
| Stability | Experimental |

## Performance

pyTMD is designed for flexibility and scientific accuracy. pyTMD_turbo trades some flexibility for batch processing speed:

- **~75x faster** for batch predictions (with caching enabled)
- **>99% correlation** with pyTMD results (typically <3cm RMS difference)
- Optimized specifically for processing many locations/times simultaneously
- **Zero-config caching** for repeated model loading

For single-point predictions or exploratory analysis, pyTMD remains an excellent choice.

### Benchmark Results

Tested with GOT5.5 model, computing tides at N points × 24 time points:

| Points | pyTMD | pyTMD_turbo (no cache) | pyTMD_turbo (cached) | Speedup |
|--------|-------|------------------------|----------------------|---------|
| 10     | 93.9s | 2.2s                   | 1.3s                 | **73x** |
| 100    | 95.4s | 2.2s                   | 1.2s                 | **77x** |
| 1000   | 92.1s | 2.2s                   | 1.2s                 | **76x** |

*Note: pyTMD requires separate calls per time step (drift mode), while pyTMD_turbo processes all points × times in a single vectorized call. This reflects typical real-world usage patterns.*

## Features

- **Fast ocean tide prediction** (`pyTMD_turbo.compute`)
  - Batch interpolation using SciPy `ndimage.map_coordinates`
  - Vectorized harmonic synthesis with NumPy
  - Model caching for repeated calculations
  - Tidal currents (u, v components)

- **Solid Earth tide** (`pyTMD_turbo.predict.solid_earth`)
  - IERS 2010 compliant displacement calculation
  - Frequency-dependent Love numbers
  - Body tide (spectral method)

- **Long-period equilibrium tide** (`pyTMD_turbo.predict.equilibrium`)
  - Cartwright-Tayler-Edden method (15 constituents)
  - LPET elevations

- **Multiple format support** (`pyTMD_turbo.io`)
  - OTIS binary format
  - ATLAS compact format
  - FES/GOT NetCDF format
  - xarray Dataset integration

- **Spatial tools** (`pyTMD_turbo.spatial`, `pyTMD_turbo.interpolate`)
  - Coordinate transformations (ECEF, geodetic, spherical)
  - k-d tree extrapolation
  - Bilinear interpolation

- **Standalone operation**
  - Does not require pyTMD for core functionality
  - Built-in constituent tables and nodal corrections
  - Includes model database (same as pyTMD)

- **Zero-config caching** (`pyTMD_turbo.cache`)
  - Automatic model data caching for faster repeated loads
  - Environment variable configuration
  - Temporary cache mode for ephemeral processing
  - Per-model cache control

## Supported Models

Over 50 tidal models are supported, including:

| Format | Models |
|--------|--------|
| GOT-netcdf | GOT5.5, GOT5.6, GOT4.10 |
| OTIS | TPXO9-atlas-v5, CATS2008, AOTIM-5 |
| ATLAS-netcdf | TPXO9-atlas-v5-nc, TPXO10-atlas-v2-nc |
| FES-netcdf | FES2022, FES2014, EOT20 |

## Installation

### From GitHub

```bash
# Basic installation
pip install git+https://github.com/tkykszk/pyTMD_turbo.git

# With Numba support (optional, for additional JIT acceleration)
pip install "pyTMD_turbo[numba] @ git+https://github.com/tkykszk/pyTMD_turbo.git"
```

### From Source

```bash
# Clone the repository
git clone https://github.com/tkykszk/pyTMD_turbo.git
cd pyTMD_turbo

# Install in editable mode (for development)
pip install -e .

# With Numba support
pip install -e ".[numba]"

# With development dependencies (pytest, etc.)
pip install -e ".[dev]"
```

### Using Pixi (recommended for reproducible environments)

```bash
# Install pixi: https://pixi.sh
pixi install

# Run tests
pixi run test
```

### Verify Installation

```python
import pyTMD_turbo
print(pyTMD_turbo.__file__)

# Check Numba availability
from pyTMD_turbo.predict.harmonic_numba import HAS_NUMBA
print(f"Numba available: {HAS_NUMBA}")
```

## Dependencies

### Required

| Package | Version | Usage |
|---------|---------|-------|
| Python | >= 3.9 | |
| NumPy | any | Array operations |
| SciPy | >= 1.10.1 | Fast interpolation (`ndimage.map_coordinates`) |
| xarray | any | Model data loading |
| netCDF4 | any | NetCDF file reading |

### Optional

| Package | Version | Usage |
|---------|---------|-------|
| Numba | any | Alternative JIT-compiled harmonic synthesis (not used in default path) |
| pyTMD | >= 3.0 | For running comparison tests |
| timescale | any | For running comparison tests |

## Model Data

pyTMD_turbo requires tidal model data files. These are **not included** in the package.

### Downloading Model Data

Model data can be obtained from various sources:

| Model | Source | Size |
|-------|--------|------|
| GOT5.5, GOT5.6 | [NASA GSFC](https://earth.gsfc.nasa.gov/geo/data/ocean-tide-models) | ~500 MB |
| TPXO9-atlas | [OSU TPXO](https://www.tpxo.net/) (registration required) | ~2 GB |
| FES2014, FES2022 | [AVISO](https://www.aviso.altimetry.fr/) (registration required) | ~3 GB |

### Using pyTMD to Download

If you have pyTMD installed, you can use its download utilities:

```python
import pyTMD
pyTMD.utilities.from_http(
    ['GOT5.5'],
    directory='/path/to/models'
)
```

### Directory Structure

After downloading, your model directory should look like:

```
/path/to/models/
├── GOT5.5/
│   ├── GOT5.5_ocean_load.nc
│   ├── GOT5.5_ocean_pole.nc
│   └── ...
├── TPXO9-atlas-v5/
│   └── ...
└── FES2014/
    └── ...
```

## Quick Start

```python
import numpy as np
import pyTMD_turbo

# Create time array using numpy datetime64
times = np.arange(
    '2024-01-01', '2024-01-02',
    dtype='datetime64[h]'  # hourly
)

# Predict tide at a single location
tide = pyTMD_turbo.tide_elevations(
    x=140.0,           # longitude
    y=35.0,            # latitude
    times=times,
    model='GOT5.5',
    directory='/path/to/models'
)
print(tide)  # tide height in meters
```

## Usage

### Multiple Locations

```python
import numpy as np
import pyTMD_turbo

times = np.arange('2024-01-01', '2024-01-08', dtype='datetime64[h]')

# Multiple locations (same time series for all)
x = np.array([140.0, 150.0, 160.0])  # longitudes
y = np.array([35.0, 30.0, 25.0])     # latitudes

tide = pyTMD_turbo.tide_elevations(
    x, y, times,
    model='GOT5.5',
    directory='/path/to/models'
)
# Returns shape (n_points, n_times)
```

### High-Speed Batch API (MJD)

For maximum performance, use MJD (Modified Julian Day) directly:

```python
import numpy as np
import pyTMD_turbo

# MJD = JD - 2400000.5
# 2024-01-01 00:00 UTC = MJD 60310.0
mjd = 60310.0 + np.arange(24*7) / 24.0  # 1 week hourly

lats = np.array([35.0, 36.0, 37.0])
lons = np.array([140.0, 141.0, 142.0])

tide = pyTMD_turbo.predict_batch(
    lats, lons, mjd,
    model='GOT5.5',
    directory='/path/to/models'
)
```

### Converting datetime to MJD

```python
from datetime import datetime, timezone
import pyTMD_turbo

dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
mjd = pyTMD_turbo.datetime_to_mjd(dt)  # 60310.5
```

## Cache System

pyTMD_turbo automatically caches model data for faster repeated loads. Caching is enabled by default.

### Basic Control

```python
from pyTMD_turbo import cache

# Disable/enable caching globally
cache.disable_cache()
cache.enable_cache()

# Disable/enable for specific models
cache.disable_cache_for('GOT5.5', 'TPXO9')
cache.enable_cache_for('GOT5.5')

# Check status
cache.show_cache_status()
```

### Context Managers

```python
from pyTMD_turbo import cache
import pyTMD_turbo
import numpy as np

times = np.arange('2024-01-01', '2024-01-02', dtype='datetime64[h]')

# Temporarily disable caching
with cache.cache_disabled():
    tide = pyTMD_turbo.tide_elevations(140.0, 35.0, times, model='GOT5.5')

# Temporarily disable for specific model
with cache.cache_disabled_for('GOT5.5'):
    tide = pyTMD_turbo.tide_elevations(140.0, 35.0, times, model='GOT5.5')
```

### Cache Operations

```python
from pyTMD_turbo import cache

# Clear cache for a model
cache.clear_cache('GOT5.5')

# Clear all caches
cache.clear_all_cache()

# Force rebuild (clears existing, rebuilds on next load)
cache.rebuild_cache('GOT5.5')
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `PYTMD_TURBO_DISABLED` | Disable caching globally | `1`, `true` |
| `PYTMD_TURBO_DISABLED_MODELS` | Disable caching for specific models | `GOT5.5,TPXO9` |
| `PYTMD_TURBO_CACHE_DIR` | Custom cache directory | `/tmp/tide_cache` |
| `PYTMD_TURBO_TEMP_CACHE` | Auto-delete caches on exit | `1`, `true` |

## Module Structure

```
pyTMD_turbo/
├── compute.py          # Main prediction API (tide_elevations, tide_currents, SET_displacements, etc.)
├── cache.py            # Zero-config cache control system
├── constituents.py     # Tidal constituent tables and nodal corrections
├── spatial.py          # Coordinate transformations
├── interpolate.py      # Extrapolation and interpolation
├── io/
│   ├── model.py        # Model loading and database
│   ├── OTIS.py         # OTIS binary format reader
│   ├── ATLAS.py        # ATLAS compact format reader
│   ├── FES.py          # FES NetCDF format reader
│   └── dataset.py      # xarray TMD accessor
├── predict/
│   ├── cache_optimized.py  # Optimized prediction engine
│   ├── solid_earth.py      # Solid Earth tide calculations
│   ├── equilibrium.py      # Long-period equilibrium tide
│   ├── infer_minor.py      # Minor constituent inference
│   └── harmonic_numba.py   # Numba-accelerated functions (optional)
├── astro/              # Astronomical calculations
│   └── ephemeris.py    # Solar/lunar ephemerides
└── data/
    └── database.json   # Model database
```

## Testing

```bash
# Run unit tests (no external data required)
pytest test/

# Run E2E tests comparing with pyTMD (requires model data)
export PYTMD_RESOURCE=/path/to/model/data
pytest test/test_e2e_models.py -v
```

## License

MIT License

This software is a derivative work of pyTMD by Tyler Sutterley.
See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

## Acknowledgments

- [pyTMD](https://github.com/tsutterley/pyTMD) - Original tidal prediction library
- [Tyler Sutterley](https://github.com/tsutterley) - Original author
