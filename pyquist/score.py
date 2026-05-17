"""Score representation, rendering, and MIDI ingestion.

A :class:`Score` is a list of :class:`SoundEvent` — onset-based musical events.
``Score`` behaves like a regular Python list (subclass of
:class:`collections.UserList`): you can iterate, index, slice, concatenate
with ``+``, repeat with ``*``, append, etc. — all preserving ``Score`` type.

Each event carries a ``time`` (in *seconds* if rendered without a metronome,
in *ticks* otherwise — a "tick" being whatever discrete time unit the
metronome maps to seconds) and an instrument-specific ``kwargs`` dict. The
metronome is kept as a separate object (a :class:`Metronome` instance) and
passed explicitly to :meth:`Score.render` when needed.

To turn a ``Score`` into audio, call :meth:`Score.render` with an
:class:`Instrument` — a callable taking a :class:`SoundEvent` and returning
:class:`Audio`. For per-event dispatch (e.g. different sounds for drums vs.
pitched notes), just branch inside the instrument::

    def my_instrument(event):
        if event.kwargs.get("is_drum"):
            return drum_kit(event)
        return piano(event)

Existing ``**kwargs``-style functions adapt with a one-line lambda::

    score.render(lambda event: my_kwargs_instrument(**event.kwargs))

Common "tick" units:

* **beats** — paired with :class:`BasicMetronome` (e.g., 120 BPM).
* **MIDI ticks** — paired with :class:`MIDIMetronome` (subdivisions of a
  quarter note as defined by the MIDI file's PPQ resolution and tempo map).

For MIDI ingestion, see :meth:`Score.from_midi`.
"""

import abc
import bisect
import math
import os
from collections import UserList
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

import mido

from .audio import Audio

_KwargsDict: TypeAlias = Dict[str, Any]

# An off-by-this-amount epsilon used as the right boundary of Score.segment.
_SEGMENT_EPS = 1e-9


class SoundEvent(NamedTuple):
    """A timestamped, instrument-agnostic musical event.

    ``time`` is in seconds when the score is rendered without a metronome,
    or in ticks (whatever unit the metronome maps to seconds — beats, MIDI
    ticks, ...) when a metronome is supplied. ``kwargs`` is opaque to the
    score; it is forwarded to the instrument at render time.

    ``SoundEvent`` is a ``NamedTuple``: it unpacks like a regular tuple
    (``time, kwargs = event``), can be constructed positionally
    (``SoundEvent(0.5, {"pitch": 60})``) or by keyword.
    """

    time: float
    kwargs: _KwargsDict


# A callable that takes a SoundEvent and returns its rendered Audio.
Instrument: TypeAlias = Callable[[SoundEvent], Audio]


# ---------------------------------------------------------------------------
# Metronome
# ---------------------------------------------------------------------------


class Metronome(abc.ABC):
    """Converts between score-time *ticks* and wall-clock *seconds*.

    A "tick" here is the abstract time unit a :class:`Score` uses; different
    ``Metronome`` subclasses define what a tick means concretely (a beat for
    :class:`BasicMetronome`, a MIDI PPQ tick for :class:`MIDIMetronome`).
    """

    @abc.abstractmethod
    def tick_to_seconds(self, tick: float) -> float:
        """Returns the wall-clock time in seconds at which ``tick`` occurs."""

    @abc.abstractmethod
    def seconds_to_tick(self, seconds: float) -> float:
        """Returns the score tick at the given wall-clock time."""


class BasicMetronome(Metronome):
    """A fixed-tempo metronome where 1 tick = 1 beat.

    Defaults to 60 BPM, which makes 1 tick = 1 second — a convenient
    identity mapping for scores whose ``time`` field is already in seconds.
    Pass an explicit ``bpm`` to map beats to seconds at a different tempo.
    """

    def __init__(self, bpm: float = 60.0):
        self.bpm = bpm
        self.beat_duration = 60.0 / bpm

    def tick_to_seconds(self, tick: float) -> float:
        return tick * self.beat_duration

    def seconds_to_tick(self, seconds: float) -> float:
        return seconds / self.beat_duration


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


class Score(UserList):
    """A list of :class:`SoundEvent` with music-specific helpers.

    ``Score`` subclasses :class:`collections.UserList`, so all the standard
    list operations work and preserve ``Score`` type — ``+``, ``*``,
    slicing, ``+=``, ``.copy()``, ``.append()``, etc.

    The ``time`` field of each event is interpreted in *seconds* unless you
    pass a :class:`Metronome` to :meth:`render`, in which case ``time`` is
    treated as *ticks* and converted via ``metronome.tick_to_seconds``. The
    metronome is not stored on the score — keep it as a separate variable
    (or use the ``(score, metronome)`` tuple returned by :meth:`from_midi`).

    Construct from any iterable of events::

        Score([SoundEvent(0.0, {"pitch": 60}), SoundEvent(1.0, {"pitch": 64})])

    Or load from a MIDI file via :meth:`from_midi`.
    """

    # --- properties --------------------------------------------------------

    @property
    def start_time(self) -> float:
        """The earliest event's ``time``. Raises ``ValueError`` if empty."""
        if not self:
            raise ValueError("Empty score has no start_time.")
        return min(e.time for e in self)

    @property
    def end_time(self) -> float:
        """The latest event's ``time``. Raises ``ValueError`` if empty.

        Note: this is the latest *onset*, not the end of any sustained note.
        Events have no intrinsic duration at the score level — only their
        instrument knows.
        """
        if not self:
            raise ValueError("Empty score has no end_time.")
        return max(e.time for e in self)

    @property
    def duration(self) -> float:
        """``end_time - start_time``. Raises ``ValueError`` if empty."""
        return self.end_time - self.start_time

    # --- factory + slicing methods -----------------------------------------

    @classmethod
    def from_midi(
        cls, midi: Union[str, os.PathLike, IO, mido.MidiFile]
    ) -> Tuple["Score", "MIDIMetronome"]:
        """Parses a MIDI file into a ``(score, metronome)`` pair.

        Each note across all tracks becomes one :class:`SoundEvent`:

        * ``time`` — the MIDI tick at which the note begins (NOTE_ON).
        * ``kwargs["off_tick"]`` — the tick at which the note ends.
        * ``kwargs["duration"]`` — the note's duration in seconds.
        * ``kwargs["pitch"]`` — MIDI pitch (0–127).
        * ``kwargs["velocity"]`` — MIDI NOTE_ON velocity (0–127).
        * ``kwargs["program"]`` — MIDI program number (0–127), reflecting
          the most recent ``program_change`` on the same channel of the
          same track.
        * ``kwargs["is_drum"]`` — ``True`` if the note is on MIDI channel
          10 (zero-indexed: 9), the conventional percussion channel.

        Pass the returned ``metronome`` to :meth:`render` so ticks are
        converted to seconds using the file's actual tempo map.

        Args:
            midi: A path, a file-like object, or an already-parsed
                :class:`mido.MidiFile`.
        """
        mid = _load_midi(midi)
        metronome = MIDIMetronome(mid)
        events = _parse_midi_notes(mid, metronome)
        events.sort(key=lambda e: e.time)
        return cls(events), metronome

    def segment(
        self,
        *,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> "Score":
        """Returns the events whose ``time`` falls in ``[offset, offset+duration)``.

        The right boundary is open with a small epsilon (``offset + duration
        - 1e-9``) to suppress floating-point off-by-ones at exact-end matches.

        Args:
            offset: Lower bound on event time (inclusive). Defaults to ``0.0``.
            duration: Length of the window. If ``None`` (default), no upper
                bound is applied.
        """
        end = float("inf") if duration is None else offset + duration - _SEGMENT_EPS
        return type(self)(e for e in self if offset <= e.time < end)

    def render(
        self,
        instrument: Instrument,
        *,
        metronome: Optional[Metronome] = None,
    ) -> Audio:
        """Renders this score to a single mixed :class:`Audio`.

        Each event is rendered by calling ``instrument(event)`` and mixed
        in at its onset time. Onsets are interpreted as ticks (and
        converted via ``metronome.tick_to_seconds``) when a metronome is
        given, otherwise as seconds.

        For per-event dispatch (e.g. different instruments for drums vs.
        pitched notes), branch inside the instrument — there is no
        separate "factory" concept.

        All instrument outputs must share a ``sample_rate`` and a
        ``num_channels``; otherwise ``ValueError`` is raised. An empty
        score returns a zero-length, mono ``Audio``.
        """
        if not self:
            return Audio.zeros(0, 1)

        rendered: List[Tuple[float, Audio]] = []
        for event in self:
            audio = instrument(event)
            if not isinstance(audio, Audio):
                raise TypeError(
                    f"instrument(event) must return an Audio; "
                    f"got {type(audio).__name__}."
                )
            seconds = (
                event.time
                if metronome is None
                else metronome.tick_to_seconds(event.time)
            )
            rendered.append((seconds, audio))

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
        duration = max(seconds + audio.duration for seconds, audio in rendered)
        output = Audio.zeros(
            math.ceil(duration * sample_rate),
            num_channels,
            sample_rate=sample_rate,
        )
        for seconds, audio in rendered:
            sample = int(seconds * sample_rate)
            output[sample : sample + audio.num_samples, :] += audio
        return output


# ---------------------------------------------------------------------------
# MIDI ingestion (mido-based)
# ---------------------------------------------------------------------------

# A MIDI tempo map: list of (absolute_tick, microseconds_per_quarter_note),
# sorted by tick, with an entry at tick 0.
_TempoMap: TypeAlias = List[Tuple[int, int]]

# Default MIDI tempo if a file has no set_tempo events: 120 BPM.
_DEFAULT_MSPB = 500_000


def _load_midi(midi: Union[str, os.PathLike, IO, mido.MidiFile]) -> mido.MidiFile:
    """Returns a :class:`mido.MidiFile` from a path, file-like, or instance."""
    if isinstance(midi, mido.MidiFile):
        return midi
    if isinstance(midi, (str, os.PathLike)):
        return mido.MidiFile(midi)
    return mido.MidiFile(file=midi)


def _extract_tempo_map(mid: mido.MidiFile) -> _TempoMap:
    """Returns a sorted list of ``(absolute_tick, microseconds_per_beat)``.

    Always begins with an entry at tick 0 (uses the default 120 BPM if no
    earlier ``set_tempo`` event exists).
    """
    events: List[Tuple[int, int]] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                events.append((abs_tick, msg.tempo))
    events.sort(key=lambda e: e[0])
    if not events or events[0][0] != 0:
        events.insert(0, (0, _DEFAULT_MSPB))
    return events


class MIDIMetronome(Metronome):
    """A metronome that follows a MIDI file's PPQ resolution and tempo map.

    1 tick = 1 MIDI tick (a PPQ subdivision of a quarter note). Tempo
    changes inside the file are honored, so ``tick_to_seconds`` is
    generally non-linear across the file.

    Args:
        midi: A path, a file-like object, or an already-parsed
            :class:`mido.MidiFile`.
    """

    def __init__(self, midi: Union[str, os.PathLike, IO, mido.MidiFile]):
        mid = _load_midi(midi)
        self.ticks_per_beat = mid.ticks_per_beat
        tempo_map = _extract_tempo_map(mid)

        # Precompute (start_tick, start_seconds, mspb) per tempo segment so
        # both directions of the conversion are O(log n).
        self._tick_starts: List[int] = []
        self._second_starts: List[float] = []
        self._mspbs: List[int] = []

        seconds = 0.0
        prev_tick, prev_mspb = tempo_map[0]
        self._tick_starts.append(prev_tick)
        self._second_starts.append(0.0)
        self._mspbs.append(prev_mspb)

        for next_tick, next_mspb in tempo_map[1:]:
            seconds += mido.tick2second(
                next_tick - prev_tick, self.ticks_per_beat, prev_mspb
            )
            self._tick_starts.append(next_tick)
            self._second_starts.append(seconds)
            self._mspbs.append(next_mspb)
            prev_tick, prev_mspb = next_tick, next_mspb

    def tick_to_seconds(self, tick: float) -> float:
        idx = max(0, bisect.bisect_right(self._tick_starts, tick) - 1)
        start_tick = self._tick_starts[idx]
        start_seconds = self._second_starts[idx]
        mspb = self._mspbs[idx]
        return start_seconds + mido.tick2second(
            tick - start_tick, self.ticks_per_beat, mspb
        )

    def seconds_to_tick(self, seconds: float) -> float:
        idx = max(0, bisect.bisect_right(self._second_starts, seconds) - 1)
        start_tick = self._tick_starts[idx]
        start_seconds = self._second_starts[idx]
        mspb = self._mspbs[idx]
        return start_tick + mido.second2tick(
            seconds - start_seconds, self.ticks_per_beat, mspb
        )


def _parse_midi_notes(
    mid: mido.MidiFile, metronome: "MIDIMetronome"
) -> List[SoundEvent]:
    """Walks all tracks, pairing note_on/note_off into SoundEvents."""
    events: List[SoundEvent] = []
    for track in mid.tracks:
        abs_tick = 0
        # Per-track-and-channel program (MIDI Program Change is per-channel,
        # but channels mean different things in different tracks).
        programs: Dict[int, int] = {}
        # Active notes keyed by (channel, pitch); two simultaneous notes on
        # the same channel/pitch in the same track aren't standard MIDI.
        active: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

        for msg in track:
            abs_tick += msg.time
            if msg.type == "program_change":
                programs[msg.channel] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (
                    abs_tick,
                    msg.velocity,
                    programs.get(msg.channel, 0),
                )
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                if key not in active:
                    continue  # Stray note_off; skip silently.
                on_tick, velocity, program = active.pop(key)
                off_tick = abs_tick
                duration = metronome.tick_to_seconds(
                    off_tick
                ) - metronome.tick_to_seconds(on_tick)
                events.append(
                    SoundEvent(
                        time=on_tick,
                        kwargs={
                            "off_tick": off_tick,
                            "duration": duration,
                            "pitch": msg.note,
                            "velocity": velocity,
                            "program": program,
                            "is_drum": msg.channel == 9,
                        },
                    )
                )
    return events
