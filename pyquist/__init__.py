from .audio import Audio
from .device import (
    play,
    record,
    set_input_device,
    set_output_device,
)
from .paths import LIB_DIR
from .plot import plot, plot_freq, plot_spec
from .score import (
    BasicMetronome,
    Event,
    Metronome,
    Score,
)

__all__ = [
    "Audio",
    "BasicMetronome",
    "Metronome",
    "Score",
    "Event",
    "play",
    "plot",
    "plot_freq",
    "plot_spec",
    "record",
    "set_input_device",
    "set_output_device",
]


# NOTE: This changes the test discovery pattern from "test*.py" (default) to "*test.py".
def load_tests(loader, standard_tests, pattern):
    package_tests = loader.discover(start_dir=LIB_DIR, pattern="*test.py")
    standard_tests.addTests(package_tests)
    return standard_tests
