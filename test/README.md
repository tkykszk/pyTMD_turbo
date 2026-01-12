# pyTMD_turbo Tests

This directory contains the test suite for pyTMD_turbo.

## Running Tests

```bash
# Run all tests
pytest

# Run with model data
PYTMD_RESOURCE=/path/to/models pytest

# Run specific test file
pytest test/test_benchmark.py

# Run with verbose output
pytest -v -s
```

## Test Categories

### Core Tests

- **test_cache.py** - Cache system functionality
- **test_cache_control.py** - Cache enable/disable API
- **test_cache_env.py** - Environment variable handling
- **test_cache_matrix.py** - Cache matrix operations
- **test_model.py** - Model loading and initialization
- **test_interpolate.py** - Spatial interpolation

### Astronomical Tests

- **test_astro.py** - Astronomical calculations
- **test_solar.py** - Solar position calculations
- **test_lunar.py** - Lunar position calculations
- **test_mean_longitudes.py** - Mean longitude calculations
- **test_constituents.py** - Tidal constituent parameters

### Tide Model Tests

- **test_otis_read.py** - OTIS format reading
- **test_otis_turbo.py** - OTIS optimized operations
- **test_atlas_read.py** - ATLAS format reading
- **test_atlas_turbo.py** - ATLAS optimized operations
- **test_fes_predict.py** - FES model prediction
- **test_fes_turbo.py** - FES optimized operations
- **test_perth3_read.py** - PERTH3 format reading
- **test_perth5_predict.py** - PERTH5 prediction

### Solid Earth Tests

- **test_set.py** - Solid Earth Tide calculations
- **test_solid_earth.py** - SET implementation details
- **test_love_numbers.py** - Love number calculations
- **test_equilibrium_tide.py** - Equilibrium tide
- **test_pole_tide.py** - Pole tide calculations

### Phase Calculation Tests

- **test_phase4.py** - Phase calculation v4
- **test_phase5.py** - Phase calculation v5

### Integration Tests

- **test_integration.py** - Full workflow tests
- **test_e2e_models.py** - End-to-end model tests
- **test_compare_pytmd.py** - Comparison with pyTMD

### Benchmark Tests

- **test_benchmark.py** - Performance benchmark (pytest version)

### Other Tests

- **test_arguments.py** - Argument validation
- **test_coordinates.py** - Coordinate transformations
- **test_spatial.py** - Spatial operations
- **test_math.py** - Mathematical utilities
- **test_utilities.py** - General utilities
- **test_tmd_accessor.py** - TMD accessor functionality
- **test_tide_currents.py** - Tide current calculations
- **test_infer_minor.py** - Minor constituent inference
- **test_noaa_queries.py** - NOAA API queries

## Test Requirements

- pytest
- numpy
- pyTMD (for comparison tests)
- Model data files (for integration/benchmark tests)

## Environment Variables

| Variable         | Description                       |
| ---------------- | --------------------------------- |
| `PYTMD_RESOURCE` | Path to tide model data directory |

## Benchmark Test

The `test_benchmark.py` provides pytest-compatible benchmarking:

```bash
PYTMD_RESOURCE=/path/to/models pytest test/test_benchmark.py -v -s
```

For standalone HTML report generation, use `examples/run_benchmark.py` instead.
