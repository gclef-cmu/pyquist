import numpy as np


def dbfs_to_amplitude(dbfs: float | np.ndarray) -> float:
    """Converts a decibel value to an amplitude value."""
    return 10.0 ** (dbfs / 20.0)


def amplitude_to_dbfs(amplitude: float | np.ndarray) -> float:
    """Converts an amplitude value to a decibel value."""
    return 20.0 * np.log10(amplitude)


def frequency_to_pitch(frequency: float | np.ndarray) -> float:
    """Converts a frequency value to a pitch value."""
    return 69 + 12 * np.log2(frequency / 440.0)


def pitch_to_frequency(pitch: float | np.ndarray) -> float:
    """Converts a pitch value to a frequency value."""
    return 440.0 * 2 ** ((pitch - 69) / 12)
