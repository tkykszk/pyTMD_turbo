import pytest
import pathlib
import os
import warnings

# Flag to track if we're using fallback
_using_fallback = False

# default working data directory for tide models
# Try pyTMD if available, otherwise use a fallback
try:
    from pyTMD.utilities import get_cache_path
    _default_directory = get_cache_path()
except ImportError:
    # Fallback: use environment variable or home directory
    _default_directory = pathlib.Path(
        os.environ.get('PYTMD_DATA', pathlib.Path.home() / '.pyTMD')
    )
    _using_fallback = True


def pytest_configure(config):
    """Called after command line options have been parsed and all plugins loaded."""
    if _using_fallback:
        warnings.warn(
            "\n"
            "=" * 70 + "\n"
            "WARNING: pyTMD is not installed.\n"
            f"Using fallback data directory: {_default_directory}\n"
            "Tests requiring pyTMD will be skipped.\n"
            "=" * 70,
            UserWarning
        )


def pytest_addoption(parser):
    parser.addoption("--directory", action="store", help="Directory for test data", default=_default_directory, type=pathlib.Path)


@pytest.fixture(scope="session")
def directory(request):
    """ Returns Data Directory """
    return request.config.getoption("--directory")
