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

- **~700x faster** for batch predictions (many locations/times at once)
- **>99% correlation** with pyTMD results (typically <3cm RMS difference)
- Optimized specifically for processing many locations/times simultaneously

For single-point predictions or exploratory analysis, pyTMD remains an excellent choice.

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

## Supported Models

Over 50 tidal models are supported, including:

| Format | Models |
|--------|--------|
| GOT-netcdf | GOT5.5, GOT5.6, GOT4.10 |
| OTIS | TPXO9-atlas-v5, CATS2008, AOTIM-5 |
| ATLAS-netcdf | TPXO9-atlas-v5-nc, TPXO10-atlas-v2-nc |
| FES-netcdf | FES2022, FES2014, EOT20 |

## Installation

```bash
pip install -e .
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

## Module Structure

```
pyTMD_turbo/
├── compute.py          # Main prediction API (tide_elevations, tide_currents, SET_displacements, etc.)
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
