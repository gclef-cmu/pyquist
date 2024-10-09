import numpy as np


def dbfs_to_amplitude(dbfs: float | np.ndarray) -> float:
    """Converts a decibel value to an amplitude value."""
    return 10.0 ** (dbfs / 20.0)


def amplitude_to_dbfs(amplitude: float | np.ndarray) -> float:
    """Converts an amplitude value to a decibel value."""
    return 20.0 * np.log10(amplitude)
