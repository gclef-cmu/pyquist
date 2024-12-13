"""
This module provides functionality for creating and rendering musical scores in PyQuist.

A score in PyQuist is represented as a sequence of sound events, where each event has:
- A timestamp (in seconds) or beat number
- An instrument function that generates the audio
- Parameters for the instrument function

The module supports both time-based and beat-based scoring through the Metronome system.
"""

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple

from .audio import Audio

# Type alias for dictionary of keyword arguments
_KwargsDict = Dict[str, Any]

# Type alias for a timestamped sound event (timestamp, parameters)
SoundEvent = Tuple[float, _KwargsDict]

# Type alias for a score, a list of sound events
Score = List[SoundEvent]

# Type alias for an instrument which renders sound events as audio
Instrument = Callable[..., Audio]

# Type alias for a playable sound event (timestamp/beat, instrument, parameters)
PlayableSoundEvent = Tuple[float, Instrument, _KwargsDict]

# Type alias for a complete playable score
PlayableScore = List[PlayableSoundEvent]


class Metronome(abc.ABC):
    """Abstract base class for metronome implementations.
    
    A metronome provides conversion between musical beats and time in seconds,
    allowing scores to be written using musical timing rather than absolute time.
    """

    @abc.abstractmethod
    def beat_to_time(self, beat: float) -> float:
        """Convert a beat number to time in seconds.

        Args:
            beat: The beat number to convert.

        Returns:
            The time in seconds corresponding to the given beat.
        """
        pass

    @abc.abstractmethod
    def time_to_beat(self, time: float) -> float:
        """Convert time in seconds to a beat number.

        Args:
            time: The time in seconds to convert.

        Returns:
            The beat number corresponding to the given time.
        """
        pass


class BasicMetronome(Metronome):
    """A simple metronome implementation with constant tempo.

    This metronome maintains a constant tempo throughout the score, converting between
    beats and time using a fixed beats-per-minute (BPM) value.
    """

    def __init__(self, bpm: float):
        """Initialize a BasicMetronome.

        Args:
            bpm: The tempo in beats per minute.
        """
        self.bpm = bpm
        self.beat_duration = 60 / bpm

    def beat_to_time(self, beat: float) -> float:
        """Convert a beat number to time in seconds.

        Args:
            beat: The beat number to convert.

        Returns:
            The time in seconds corresponding to the given beat.
        """
        return beat * self.beat_duration

    def time_to_beat(self, time: float) -> float:
        """Convert time in seconds to a beat number.

        Args:
            time: The time in seconds to convert.

        Returns:
            The beat number corresponding to the given time.
        """
        return time / self.beat_duration


def render_score(score: PlayableScore, metronome: Optional[Metronome] = None) -> Audio:
    """Renders a score to audio.

    This function takes a playable score (a list of sound events with their associated
    instruments) and renders it to a single audio object. Each sound event is rendered
    at its specified time or beat position and mixed into the final output.

    Args:
        score: A list of sound events to render. Each event is a tuple containing:
            - A timestamp (in seconds) or beat number
            - An instrument function that generates audio
            - A dictionary of parameters for the instrument function
        metronome: Optional metronome for converting beat numbers to timestamps.
            If None, the first element of each event is treated as a timestamp in seconds.
            If provided, the first element is treated as a beat number and converted
            to seconds using the metronome.

    Returns:
        An Audio object containing the rendered score.

    Raises:
        ValueError: If the audio objects produced by the instruments have inconsistent
            sample rates or numbers of channels.
    """
    # Render all sound events
    audios = []
    for beat_or_time, instrument_fn, kwargs in score:
        if metronome is None:
            time = beat_or_time
        else:
            time = metronome.beat_to_time(beat_or_time)
        audios.append((time, instrument_fn(**kwargs)))

    # Compute sample rate
    sample_rates = set(audio.sample_rate for _, audio in audios)
    if len(sample_rates) != 1:
        raise ValueError("Inconsistent sample rates")
    sample_rate = sample_rates.pop()

    # Compute duration
    duration = max(time + audio.duration for time, audio in audios)

    # Compute max channels
    all_channels = set(audio.num_channels for _, audio in audios)
    if len(all_channels) != 1:
        raise ValueError("Inconsistent number of channels")
    num_channels = all_channels.pop()

    # Output audio
    output = Audio(
        num_channels=num_channels,
        num_samples=round(duration * sample_rate),
        sample_rate=sample_rate,
    )

    # Mix all sound events
    for time, audio in audios:
        sample = round(time * sample_rate)
        output[:, sample : sample + audio.num_samples] += audio

    return output
