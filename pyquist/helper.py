import numpy as np


def dbfs_to_gain(dbfs: float | np.ndarray) -> float | np.ndarray:
    """Converts a decibel value to an gain value."""
    return 10.0 ** (dbfs / 20.0)


def gain_to_dbfs(gain: float | np.ndarray) -> float | np.ndarray:
    """Converts an gain value to a decibel value."""
    return 20.0 * np.log10(gain)


def frequency_to_pitch(frequency: float | np.ndarray) -> float | np.ndarray:
    """Converts a frequency value to a pitch value."""
    return 69 + 12 * np.log2(frequency / 440.0)


def pitch_to_frequency(pitch: float | np.ndarray) -> float | np.ndarray:
    """Converts a pitch value to a frequency value."""
    return 440.0 * 2 ** ((pitch - 69) / 12)
