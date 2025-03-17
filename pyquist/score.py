import abc
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

from .audio import Audio

# Type alias for dictionary of keyword arguments
_KwargsDict: TypeAlias = Dict[str, Any]

# Type alias for a timestamped sound event
SoundEvent: TypeAlias = Tuple[float, _KwargsDict]

# Type alias for a score, a list of sound events
Score: TypeAlias = List[SoundEvent]

# Type alias for an instrument which renders sound events as audio
Instrument: TypeAlias = Callable[..., Audio]
PlayableSoundEvent: TypeAlias = Tuple[float, Instrument, _KwargsDict]
PlayableScore: TypeAlias = List[PlayableSoundEvent]


def is_sound_event(obj: Any) -> bool:
    """Returns True if the object is a sound event."""
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], (int, float))
        and isinstance(obj[1], dict)
    )


def is_score(obj: Any) -> bool:
    """Returns True if the object is a score."""
    return isinstance(obj, list) and all(is_sound_event(se) for se in obj)


def is_playable_sound_event(obj: Any) -> bool:
    """Returns True if the object is a playable sound event."""
    return (
        isinstance(obj, tuple)
        and len(obj) == 3
        and callable(obj[1])
        and is_sound_event((obj[0], obj[2]))
    )


def is_playable_score(obj: Any) -> bool:
    """Returns True if the object is a playable score."""
    return isinstance(obj, list) and all(is_playable_sound_event(se) for se in obj)


class Metronome(abc.ABC):
    @abc.abstractmethod
    def beat_to_time(self, beat: float) -> float:
        pass

    @abc.abstractmethod
    def time_to_beat(self, time: float) -> float:
        pass


class BasicMetronome(Metronome):
    def __init__(self, bpm: float):
        self.bpm = bpm
        self.beat_duration = 60 / bpm

    def beat_to_time(self, beat: float) -> float:
        return beat * self.beat_duration

    def time_to_beat(self, time: float) -> float:
        return time / self.beat_duration


def render_score(score: PlayableScore, metronome: Optional[Metronome] = None) -> Audio:
    """Renders a score to audio."""
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
        num_samples=math.ceil(duration * sample_rate),
        sample_rate=sample_rate,
    )

    # Mix all sound events
    for time, audio in audios:
        sample = int(time * sample_rate)
        output[sample : sample + audio.num_samples, :] += audio

    return output
