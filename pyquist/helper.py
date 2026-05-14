import numpy as np


def db_to_amplitude(
    db: float | np.ndarray, *, reference: float = 1.0
) -> float | np.ndarray:
    """Converts a decibel level to a linear amplitude.

    Decibels are a logarithmic ratio: ``amplitude = reference * 10**(db / 20)``.
    With the default ``reference=1.0`` and float audio (where ``±1.0`` is
    digital full scale), this is dBFS:

    * ``0  dB`` → ``1.0`` (unity / full scale)
    * ``-6 dB`` → ``≈ 0.501`` (roughly half amplitude)
    * ``-20 dB`` → ``0.1`` (exactly 1/10th amplitude)
    * ``+6 dB`` → ``≈ 1.995`` (roughly double amplitude)

    Pass a different ``reference`` to compare against another baseline (e.g.
    a measured RMS level, or a non-unity peak).

    Args:
        db: Decibel value. Scalar or numpy array.
        reference: The linear amplitude that corresponds to 0 dB. Defaults
            to ``1.0`` (full scale).
    """
    return reference * 10.0 ** (db / 20.0)


def amplitude_to_db(
    amplitude: float | np.ndarray, *, reference: float = 1.0
) -> float | np.ndarray:
    """Converts a linear amplitude to a decibel level.

    Inverse of :func:`db_to_amplitude`:
    ``db = 20 * log10(amplitude / reference)``.
    With the default ``reference=1.0``, this is dBFS:

    * ``1.0``  → ``0 dB`` (full scale)
    * ``0.5``  → ``≈ -6.02 dB``
    * ``0.1``  → ``-20 dB``
    * ``2.0``  → ``≈ +6.02 dB``

    An amplitude of ``0.0`` produces ``-inf`` and emits a numpy log warning;
    callers that may pass exactly zero should clamp first.

    Args:
        amplitude: Linear amplitude. Scalar or numpy array.
        reference: The linear amplitude that corresponds to 0 dB. Defaults
            to ``1.0`` (full scale).
    """
    return 20.0 * np.log10(amplitude / reference)


def frequency_to_pitch(frequency: float | np.ndarray) -> float | np.ndarray:
    """Converts a frequency in Hz to a MIDI pitch number.

    Uses the standard 12-tone equal-temperament tuning anchored to A4 = 440 Hz
    (MIDI pitch 69): ``pitch = 69 + 12 * log2(frequency / 440)``.

    The result is a real number, not just an integer — fractional values
    represent intermediate frequencies (one semitone = 1.0, one cent = 0.01).
    For example, ``frequency_to_pitch(466.16) ≈ 70.0`` (A#4).
    """
    return 69 + 12 * np.log2(frequency / 440.0)


def pitch_to_frequency(pitch: float | np.ndarray) -> float | np.ndarray:
    """Converts a MIDI pitch number to a frequency in Hz.

    Inverse of :func:`frequency_to_pitch`: ``frequency = 440 * 2**((pitch - 69) / 12)``,
    using A4 = 440 Hz (MIDI 69) and 12-tone equal temperament.

    Fractional pitches are valid: ``pitch_to_frequency(60.5)`` is a quarter-tone
    above C4.
    """
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
    """Converts scientific pitch notation (e.g. ``"C4"``) to a MIDI pitch number.

    Recognized forms:

    * Pitch class: one letter ``A``-``G`` (case insensitive).
    * Accidentals (optional, may repeat): ``#`` for sharp, ``b`` for flat.
      ``"C##4"`` (double-sharp) and ``"Bbb3"`` (double-flat) both work.
    * Octave (required): one digit, using scientific octave numbering where
      C4 = middle C = MIDI 60 and A4 = MIDI 69.

    Examples: ``"C4"`` → 60, ``"A4"`` → 69, ``"Bb3"`` → 58, ``"C#4"`` → 61.

    Accepts a single string or a numpy array of strings (vectorized via
    :func:`numpy.vectorize`). Raises ``ValueError`` on malformed input.
    """
    if isinstance(pitch, str):
        return _scientific_pitch_name_to_pitch(pitch)
    return np.vectorize(_scientific_pitch_name_to_pitch)(pitch)
