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


PITCHES = {"a": 9, "b": 11, "c": 0, "d": 2, "e": 4, "f": 5, "g": 7}
ACCIDENTALS = ["#", "b"]


def _scientific_pitch_name_to_pitch(pitch_name: str) -> int:
    # Reference: https://en.wikipedia.org/wiki/Scientific_pitch_notation
    if len(pitch_name) < 2:
        raise ValueError(f"Invalid pitch name: {pitch_name}")

    # Parse pitch
    pitch_class = pitch_name[0].lower()
    if pitch_class not in PITCHES:
        raise ValueError(f"Invalid pitch: {pitch_name}")
    pitch = PITCHES[pitch_class]

    # Parse accidentals
    accidentals = pitch_name[1:-1]
    while accidentals:
        if accidentals.startswith("#"):
            pitch += 1
        elif accidentals.startswith("b"):
            pitch -= 1
        else:
            raise ValueError(f"Invalid accidental: {pitch_name}")
        accidentals = accidentals[1:]

    # Parse octave
    try:
        octave = int(pitch_name[-1])
    except ValueError:
        raise ValueError(f"Invalid octave: {pitch_name}")
    pitch += 12 * (octave + 1)

    return pitch


def pitch_name_to_pitch(pitch: str | np.ndarray) -> int | np.ndarray:
    """Converts a pitch name to a pitch value."""
    if isinstance(pitch, str):
        return _scientific_pitch_name_to_pitch(pitch)
    return np.vectorize(_scientific_pitch_name_to_pitch)(pitch)
