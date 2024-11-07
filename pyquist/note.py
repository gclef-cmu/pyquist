from typing import Any, Callable, Dict, List, Tuple

from .audio import Audio

# Type alias for dictionary of keyword arguments
_KwargsDict = Dict[str, Any]

# Type alias for a timestamped sound event
SoundEvent = Tuple[float, _KwargsDict]

# Type alias for an instrument which renders keyword arguments to audio
Instrument = Callable[..., Audio]

# Type alias for an instrument track
Track = Tuple[Instrument, List[SoundEvent]]

# Type alias for a score, a list of tracks
Score = List[Track]


def render_score(score: Score) -> Audio:
    """Renders a score to audio."""
    # Render all sound events
    audios = []
    for instrument_fn, sound_events in score:
        for time, kwargs in sound_events:
            audios.append((time, instrument_fn(**kwargs)))

    # Compute sample rate
    sample_rates = set(audio.sample_rate for _, audio in audios)
    if len(sample_rates) != 1:
        raise ValueError("Inconsistent sample rates")
    sample_rate = sample_rates.pop()

    # Compute max duration
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
