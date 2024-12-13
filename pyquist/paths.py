import os
import pathlib

LIB_DIR = pathlib.Path(__file__).resolve().parent

if "PYQUIST_CACHE_DIR" in os.environ:
    CACHE_DIR = pathlib.Path(os.environ["PYQUIST_CACHE_DIR"])
else:
    CACHE_DIR = pathlib.Path(pathlib.Path.home(), ".cache", "pyquist")
CACHE_DIR = CACHE_DIR.resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEST_DATA_DIR = LIB_DIR / "test_data"
