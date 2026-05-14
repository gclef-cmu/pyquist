"""Score representation and rendering.

A :class:`Score` is a list of :class:`SoundEvent` — onset-based musical events.
Each event carries a ``beat`` (treated as seconds when no metronome is supplied
to :func:`render_score`) and an instrument-specific ``kwargs`` dict. Once you
bind an :class:`Instrument` callable to each event (via :func:`bind_instrument`
or by constructing :class:`PlayableSoundEvent` directly), you have a
:class:`PlayableScore` that :func:`render_score` can turn into an
:class:`pyquist.Audio`.
"""

import abc
import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TypeAlias

from .audio import Audio

_KwargsDict: TypeAlias = Dict[str, Any]

# A callable that renders a single sound event as an Audio.
Instrument: TypeAlias = Callable[..., Audio]


class SoundEvent(NamedTuple):
    """A timestamped, instrument-agnostic musical event.

    ``beat`` is in beats when the score is rendered with a :class:`Metronome`,
    or in seconds otherwise. ``kwargs`` is opaque to the score — it is passed
    through to the instrument at render time.

    SoundEvent is a ``NamedTuple``, so it unpacks like a regular tuple
    (``beat, kwargs = event``) and can be constructed positionally
    (``SoundEvent(0.5, {"pitch": 60})``) or by keyword.
    """

    beat: float
    kwargs: _KwargsDict


class PlayableSoundEvent(NamedTuple):
    """A :class:`SoundEvent` bound to an :class:`Instrument`."""

    beat: float
    instrument: Instrument
    kwargs: _KwargsDict


Score: TypeAlias = List[SoundEvent]
PlayableScore: TypeAlias = List[PlayableSoundEvent]


def bind_instrument(score: Score, instrument: Instrument) -> PlayableScore:
    """Wraps every event in ``score`` with the given instrument.

    Useful for the common case of writing a melody once and then rendering
    it with different instruments:

        >>> melody = [SoundEvent(0.0, {"pitch": 60}), SoundEvent(1.0, {"pitch": 64})]
        >>> piano_score = bind_instrument(melody, piano)
        >>> render_score(piano_score, metronome)
    """
    return [PlayableSoundEvent(e.beat, instrument, e.kwargs) for e in score]


class Metronome(abc.ABC):
    """Converts between beat positions and wall-clock seconds.

    Subclass and override :meth:`beat_to_time` (and :meth:`time_to_beat` if
    you need the inverse) to model arbitrary tempo curves.
    """

    @abc.abstractmethod
    def beat_to_time(self, beat: float) -> float:
        """Returns the time in seconds at which ``beat`` occurs."""

    @abc.abstractmethod
    def time_to_beat(self, time: float) -> float:
        """Returns the beat position at the given time in seconds."""


class BasicMetronome(Metronome):
    """A fixed-tempo metronome."""

    def __init__(self, bpm: float):
        self.bpm = bpm
        self.beat_duration = 60.0 / bpm

    def beat_to_time(self, beat: float) -> float:
        return beat * self.beat_duration

    def time_to_beat(self, time: float) -> float:
        return time / self.beat_duration


def render_score(score: PlayableScore, metronome: Optional[Metronome] = None) -> Audio:
    """Renders a :class:`PlayableScore` to a single mixed :class:`Audio`.

    Each event's instrument is invoked with its ``kwargs`` to produce an
    :class:`Audio`. The resulting audios are mixed at their onset times into
    a single output. Onsets are interpreted as beats (and converted via
    ``metronome.beat_to_time``) when a metronome is given, otherwise as
    seconds.

    All instrument outputs must share a ``sample_rate`` and a
    ``num_channels``; otherwise ``ValueError`` is raised. An empty score
    returns a zero-length, mono, ``sample_rate=None`` ``Audio``.
    """
    if not score:
        return Audio.zeros(0, 1)

    # Render each event and resolve its onset time.
    rendered: List[Tuple[float, Audio]] = []
    for event in score:
        time = event.beat if metronome is None else metronome.beat_to_time(event.beat)
        rendered.append((time, event.instrument(**event.kwargs)))

    # All rendered audios must agree on sample_rate and num_channels.
    sample_rates = {audio.sample_rate for _, audio in rendered}
    if len(sample_rates) != 1:
        raise ValueError(f"Inconsistent sample rates: {sample_rates}.")
    sample_rate = sample_rates.pop()
    if sample_rate is None:
        raise ValueError("Rendered audio is missing a sample_rate.")

    channel_counts = {audio.num_channels for _, audio in rendered}
    if len(channel_counts) != 1:
        raise ValueError(f"Inconsistent channel counts: {channel_counts}.")
    num_channels = channel_counts.pop()

    # Allocate the output and mix in each event.
    duration = max(time + audio.duration for time, audio in rendered)
    output = Audio.zeros(
        math.ceil(duration * sample_rate), num_channels, sample_rate=sample_rate
    )
    for time, audio in rendered:
        sample = int(time * sample_rate)
        output[sample : sample + audio.num_samples, :] += audio
    return output
