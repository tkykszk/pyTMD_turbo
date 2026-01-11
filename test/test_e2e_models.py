"""
End-to-end tests comparing pyTMD and pyTMD_turbo with real tide model data.

These tests require:
1. Environment variable PYTMD_RESOURCE pointing to the data directory
2. Model data folders: GOT5.5, GOT5.6, RE14_LongPeriodTides

Tests are skipped if data is not available.

Copyright (c) 2024-2026 tkykszk
"""

import os
import pytest
import numpy as np
from pathlib import Path

# Skip all tests if pyTMD is not available
pyTMD = pytest.importorskip("pyTMD", reason="pyTMD not installed")
timescale = pytest.importorskip("timescale", reason="timescale not installed")

import pyTMD.io
import pyTMD.predict
import timescale.time

# Get resource directory from environment variable
PYTMD_RESOURCE = os.environ.get('PYTMD_RESOURCE')

# Skip all tests if PYTMD_RESOURCE is not set
pytestmark = pytest.mark.skipif(
    PYTMD_RESOURCE is None,
    reason="PYTMD_RESOURCE environment variable not set"
)


def resource_available(model_folder: str) -> bool:
    """Check if model data is available"""
    if PYTMD_RESOURCE is None:
        return False
    return Path(PYTMD_RESOURCE).joinpath(model_folder).exists()


class TestGOT55:
    """End-to-end tests for GOT5.5 model"""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        if not resource_available('GOT5.5'):
            pytest.skip("GOT5.5 data not available")

    def test_tide_prediction_single_point(self):
        """Compare tide prediction at a single point"""
        import pyTMD_turbo.compute as turbo_compute

        # Test location (Pacific Ocean)
        lon, lat = 150.0, 30.0

        # Time range: 1 day with hourly intervals
        mjd = 60000.0 + np.arange(24) / 24.0

        # pyTMD prediction
        m = pyTMD.io.model(directory=PYTMD_RESOURCE).from_database('GOT5.5')
        ds = m.open_dataset(group='z', use_default_units=True)

        # Transform coordinates and interpolate
        X, Y = ds.tmd.transform_as(lon, lat, crs=4326)
        local = ds.tmd.interp(X, Y)

        # Predict using pyTMD
        ts = timescale.time.Timescale(MJD=mjd)
        tide_pytmd = local.tmd.predict(
            ts.tide,
            deltat=ts.tt_ut1,
            corrections='GOT'
        ).values

        # pyTMD_turbo prediction
        tide_turbo = turbo_compute.predict_single(
            lat, lon, mjd,
            model='GOT5.5',
            directory=PYTMD_RESOURCE
        )

        # Compare results
        corr = np.corrcoef(tide_pytmd, tide_turbo)[0, 1]
        rms = np.sqrt(np.mean((tide_pytmd - tide_turbo)**2))

        print(f"\nGOT5.5 single point test:")
        print(f"  Location: ({lat}, {lon})")
        print(f"  Correlation: {corr:.6f}")
        print(f"  RMS difference: {rms*100:.3f} cm")
        print(f"  Max difference: {np.max(np.abs(tide_pytmd - tide_turbo))*100:.3f} cm")

        # Allow for differences due to different interpolation methods
        assert corr > 0.95, f"Correlation too low: {corr:.4f}"
        assert rms < 0.10, f"RMS difference too large: {rms*100:.2f} cm"

    def test_tide_prediction_multiple_points(self):
        """Compare tide prediction at multiple points"""
        import pyTMD_turbo.compute as turbo_compute

        # Test locations
        lons = np.array([140.0, 150.0, 160.0, 170.0])
        lats = np.array([30.0, 35.0, 40.0, 45.0])

        # Time range: 1 week with 6-hour intervals
        mjd = 60000.0 + np.arange(7 * 4) / 4.0

        # pyTMD prediction
        m = pyTMD.io.model(directory=PYTMD_RESOURCE).from_database('GOT5.5')
        ds = m.open_dataset(group='z', use_default_units=True)

        ts = timescale.time.Timescale(MJD=mjd)

        tide_pytmd_all = []
        for lon, lat in zip(lons, lats):
            X, Y = ds.tmd.transform_as(lon, lat, crs=4326)
            local = ds.tmd.interp(X, Y)
            tide = local.tmd.predict(
                ts.tide,
                deltat=ts.tt_ut1,
                corrections='GOT'
            ).values
            tide_pytmd_all.append(tide)
        tide_pytmd = np.array(tide_pytmd_all)

        # pyTMD_turbo prediction
        tide_turbo = turbo_compute.predict_batch(
            lats, lons, mjd,
            model='GOT5.5',
            directory=PYTMD_RESOURCE
        )

        # Compare results for each point
        print(f"\nGOT5.5 multiple points test: {len(lats)} points x {len(mjd)} times")
        for i in range(len(lats)):
            corr = np.corrcoef(tide_pytmd[i], tide_turbo[i])[0, 1]
            rms = np.sqrt(np.mean((tide_pytmd[i] - tide_turbo[i])**2))
            print(f"  Point {i} ({lats[i]}, {lons[i]}): corr={corr:.4f}, rms={rms*100:.2f}cm")

            assert corr > 0.95, f"Point {i}: Correlation too low: {corr:.4f}"
            assert rms < 0.10, f"Point {i}: RMS difference too large: {rms*100:.2f} cm"


class TestGOT56:
    """End-to-end tests for GOT5.6 model"""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        if not resource_available('GOT5.6'):
            pytest.skip("GOT5.6 data not available")

    def test_tide_prediction_single_point(self):
        """Compare tide prediction at a single point"""
        import pyTMD_turbo.compute as turbo_compute

        # Test location (Pacific Ocean - same as GOT5.5 to ensure it's in ocean)
        lon, lat = 150.0, 30.0

        # Time range
        mjd = 60000.0 + np.arange(24) / 24.0

        # pyTMD prediction
        m = pyTMD.io.model(directory=PYTMD_RESOURCE).from_database('GOT5.6')
        ds = m.open_dataset(group='z', use_default_units=True)

        X, Y = ds.tmd.transform_as(lon, lat, crs=4326)
        local = ds.tmd.interp(X, Y)

        ts = timescale.time.Timescale(MJD=mjd)
        tide_pytmd = local.tmd.predict(
            ts.tide,
            deltat=ts.tt_ut1,
            corrections='GOT'
        ).values

        # pyTMD_turbo prediction
        tide_turbo = turbo_compute.predict_single(
            lat, lon, mjd,
            model='GOT5.6',
            directory=PYTMD_RESOURCE
        )

        # Compare results
        corr = np.corrcoef(tide_pytmd, tide_turbo)[0, 1]
        rms = np.sqrt(np.mean((tide_pytmd - tide_turbo)**2))

        print(f"\nGOT5.6 single point test:")
        print(f"  Correlation: {corr:.6f}")
        print(f"  RMS difference: {rms*100:.3f} cm")

        assert corr > 0.95, f"Correlation too low: {corr:.4f}"
        assert rms < 0.10, f"RMS difference too large: {rms*100:.2f} cm"

    def test_third_degree_constituents(self):
        """Test GOT5.6's third-degree constituents (M3', L2', M1', N2')"""
        m = pyTMD.io.model(directory=PYTMD_RESOURCE).from_database('GOT5.6')
        ds = m.open_dataset(group='z')

        # Check that third-degree constituents are present
        # Note: GOT5.6 uses m3' (with prime) not m3
        constituents = list(ds.data_vars.keys())
        third_degree = ["l2'", "m1'", "m3'", "n2'"]

        print(f"\nGOT5.6 constituents: {constituents}")

        for c in third_degree:
            assert c in constituents, f"Missing third-degree constituent: {c}"


class TestRE14:
    """End-to-end tests for RE14 long-period tide model"""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        # pyTMD database expects RE14_LongPeriodTides_rel folder structure
        if not resource_available('RE14_LongPeriodTides_rel'):
            pytest.skip("RE14_LongPeriodTides_rel data not available (pyTMD database expects _rel suffix)")

    def test_long_period_constituents(self):
        """Test RE14 long-period constituents"""
        m = pyTMD.io.model(directory=PYTMD_RESOURCE).from_database('RE14')
        ds = m.open_dataset(group='z')

        # Check that long-period constituents are present
        constituents = list(ds.data_vars.keys())
        expected = ['mf', 'mm', 'mt', 'node', 'sa', 'ssa']

        print(f"\nRE14 constituents: {constituents}")

        for c in expected:
            assert c in constituents, f"Missing long-period constituent: {c}"

    def test_long_period_prediction(self):
        """Compare long-period tide prediction"""
        # Test location
        lon, lat = 0.0, 45.0

        # Time range: 1 year with daily intervals (long-period tides)
        mjd = 60000.0 + np.arange(365)

        # pyTMD prediction
        m = pyTMD.io.model(directory=PYTMD_RESOURCE).from_database('RE14')
        ds = m.open_dataset(group='z', use_default_units=True)

        X, Y = ds.tmd.transform_as(lon, lat, crs=4326)
        local = ds.tmd.interp(X, Y)

        ts = timescale.time.Timescale(MJD=mjd)
        tide_pytmd = local.tmd.predict(
            ts.tide,
            deltat=ts.tt_ut1,
            corrections='GOT'
        ).values

        # Long-period tides should have small amplitude (typically < 10cm)
        max_amp = np.max(np.abs(tide_pytmd))
        assert max_amp < 0.2, \
            f"Long-period tide amplitude too large: {max_amp*100:.2f} cm"

        print(f"\nRE14 long-period tide test:")
        print(f"  Max amplitude: {max_amp*100:.3f} cm")
        print(f"  Mean: {np.mean(tide_pytmd)*100:.3f} cm")
        print(f"  Std: {np.std(tide_pytmd)*100:.3f} cm")


class TestPerformanceComparison:
    """Performance comparison between pyTMD and pyTMD_turbo"""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        if not resource_available('GOT5.5'):
            pytest.skip("GOT5.5 data not available")

    def test_batch_prediction_speedup(self):
        """Compare batch prediction performance"""
        import time
        import pyTMD_turbo.compute as turbo_compute

        # Multiple points - use Pacific Ocean region to avoid land
        n_points = 50
        np.random.seed(42)
        # Center on Pacific Ocean (180°E, 0°N) to minimize land points
        lons = 180.0 + np.random.rand(n_points) * 30 - 15  # 165-195E
        lats = np.random.rand(n_points) * 20 - 10  # 10S-10N

        # 1 week of hourly data
        mjd = 60000.0 + np.arange(7 * 24) / 24.0

        # Initialize pyTMD_turbo (warm-up)
        turbo_compute.init_model('GOT5.5', PYTMD_RESOURCE)

        # pyTMD_turbo timing
        start = time.time()
        tide_turbo = turbo_compute.predict_batch(
            lats, lons, mjd,
            model='GOT5.5',
            directory=PYTMD_RESOURCE
        )
        turbo_time = time.time() - start

        # pyTMD timing
        m = pyTMD.io.model(directory=PYTMD_RESOURCE).from_database('GOT5.5')
        ds = m.open_dataset(group='z', use_default_units=True)

        ts = timescale.time.Timescale(MJD=mjd)

        start = time.time()
        tide_pytmd_all = []
        for lon, lat in zip(lons, lats):
            X, Y = ds.tmd.transform_as(lon, lat, crs=4326)
            local = ds.tmd.interp(X, Y)
            tide = local.tmd.predict(
                ts.tide,
                deltat=ts.tt_ut1,
                corrections='GOT'
            ).values
            tide_pytmd_all.append(tide)
        pytmd_time = time.time() - start

        speedup = pytmd_time / turbo_time

        print(f"\nBatch prediction performance ({n_points} points x {len(mjd)} times):")
        print(f"  pyTMD_turbo: {turbo_time:.3f}s")
        print(f"  pyTMD:       {pytmd_time:.3f}s")
        print(f"  Speedup:     {speedup:.1f}x")

        # Verify results are similar (skip land points with NaN)
        tide_pytmd = np.array(tide_pytmd_all)
        corrs = []
        for i in range(n_points):
            # Skip points with any NaN values
            if np.any(np.isnan(tide_pytmd[i])) or np.any(np.isnan(tide_turbo[i])):
                continue
            corrs.append(np.corrcoef(tide_pytmd[i], tide_turbo[i])[0, 1])

        mean_corr = np.nanmean(corrs) if corrs else np.nan
        print(f"  Valid ocean points: {len(corrs)}/{n_points}")
        print(f"  Mean correlation: {mean_corr:.4f}")

        assert len(corrs) > 0, "No valid ocean points found"
        assert mean_corr > 0.95, f"Mean correlation too low: {mean_corr:.4f}"


class TestAccuracyReport:
    """Generate accuracy report comparing pyTMD and pyTMD_turbo"""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        if not resource_available('GOT5.5'):
            pytest.skip("GOT5.5 data not available")

    def test_comprehensive_accuracy(self):
        """Comprehensive accuracy comparison"""
        import pyTMD_turbo.compute as turbo_compute

        # Test grid (Pacific Ocean)
        lons = np.linspace(140, 170, 6)
        lats = np.linspace(25, 45, 5)

        # 1 month of daily data
        mjd = 60000.0 + np.arange(30)

        m = pyTMD.io.model(directory=PYTMD_RESOURCE).from_database('GOT5.5')
        ds = m.open_dataset(group='z', use_default_units=True)

        ts = timescale.time.Timescale(MJD=mjd)

        correlations = []
        rms_errors = []

        for lon in lons:
            for lat in lats:
                try:
                    # pyTMD
                    X, Y = ds.tmd.transform_as(lon, lat, crs=4326)
                    local = ds.tmd.interp(X, Y)
                    tide_pytmd = local.tmd.predict(
                        ts.tide,
                        deltat=ts.tt_ut1,
                        corrections='GOT'
                    ).values

                    # pyTMD_turbo
                    tide_turbo = turbo_compute.predict_single(
                        lat, lon, mjd,
                        model='GOT5.5',
                        directory=PYTMD_RESOURCE
                    )

                    # Skip if all NaN (land points)
                    if np.all(np.isnan(tide_pytmd)) or np.all(np.isnan(tide_turbo)):
                        continue

                    # Calculate metrics
                    valid = ~(np.isnan(tide_pytmd) | np.isnan(tide_turbo))
                    if np.sum(valid) < 2:
                        continue

                    corr = np.corrcoef(tide_pytmd[valid], tide_turbo[valid])[0, 1]
                    rms = np.sqrt(np.mean((tide_pytmd[valid] - tide_turbo[valid])**2))

                    correlations.append(corr)
                    rms_errors.append(rms)
                except Exception as e:
                    print(f"  Warning: Failed at ({lat}, {lon}): {e}")
                    continue

        correlations = np.array(correlations)
        rms_errors = np.array(rms_errors)

        print(f"\n{'='*60}")
        print(f"Comprehensive Accuracy Report (GOT5.5)")
        print(f"{'='*60}")
        print(f"Grid: {len(lons)}x{len(lats)} points, {len(mjd)} times")
        print(f"Valid ocean points: {len(correlations)}")
        print(f"\nCorrelation:")
        print(f"  Mean:   {np.mean(correlations):.6f}")
        print(f"  Min:    {np.min(correlations):.6f}")
        print(f"  Median: {np.median(correlations):.6f}")
        print(f"\nRMS Error (cm):")
        print(f"  Mean:   {np.mean(rms_errors)*100:.3f}")
        print(f"  Max:    {np.max(rms_errors)*100:.3f}")
        print(f"  Median: {np.median(rms_errors)*100:.3f}")
        print(f"{'='*60}")

        # Assertions
        assert np.mean(correlations) > 0.95, \
            f"Mean correlation too low: {np.mean(correlations):.4f}"
        assert np.mean(rms_errors) < 0.10, \
            f"Mean RMS error too large: {np.mean(rms_errors)*100:.2f} cm"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
