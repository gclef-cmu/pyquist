from .audio import Audio, AudioBuffer
from .paths import LIB_DIR
from .realtime import AudioProcessor, AudioProcessorStream

__all__ = [
    "Audio",
    "AudioBuffer",
    "AudioProcessor",
    "AudioProcessorStream",
]


# NOTE: This changes the test discovery pattern from "test*.py" (default) to "*test.py".
def load_tests(loader, standard_tests, pattern):
    package_tests = loader.discover(start_dir=LIB_DIR, pattern="*test.py")
    standard_tests.addTests(package_tests)
    return standard_tests
