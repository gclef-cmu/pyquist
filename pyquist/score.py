"""Score representation, rendering, and MIDI ingestion.

A :class:`Score` is a list of :class:`Event` — onset-based musical events.
``Score`` behaves like a regular Python list (subclass of
:class:`collections.UserList`): you can iterate, index, slice, concatenate
with ``+``, repeat with ``*``, append, etc. — all preserving ``Score`` type.

Each event carries a ``time`` (in *seconds* if rendered without a metronome,
in *ticks* otherwise — a "tick" being whatever discrete time unit the
metronome maps to seconds) and an instrument-specific ``kwargs`` dict. The
metronome is kept as a separate object (a :class:`Metronome` instance) and
passed explicitly to :meth:`Score.render` when needed.

To turn a ``Score`` into audio, call :meth:`Score.render` with an
:class:`Instrument` — a callable invoked as ``instrument(**event.kwargs)``
that returns :class:`Audio`. An instrument simply declares the kwargs it
cares about and absorbs the rest with ``**kwargs``::

    def sine(pitch, duration, **kwargs):
        ...
        return Audio(samples, sample_rate=sr)

For per-event dispatch (e.g. different sounds for drums vs. pitched notes),
capture the deciding key as a named parameter and forward the rest::

    def my_instrument(is_drum, **kwargs):
        if is_drum:
            return drum_kit(**kwargs)
        return sine(**kwargs)

Because instruments are called with ``**event.kwargs``, every kwargs key
should be a valid Python identifier.

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


class Event(NamedTuple):
    """A timestamped, instrument-agnostic musical event.

    ``time`` is in seconds when the score is rendered without a metronome,
    or in ticks (whatever unit the metronome maps to seconds — beats, MIDI
    ticks, ...) when a metronome is supplied. ``kwargs`` is opaque to the
    score; it is forwarded to the instrument as ``instrument(**kwargs)`` at
    render time, so every key should be a valid Python identifier (e.g.,
    avoid dashes).

    ``Event`` is a ``NamedTuple``: it unpacks like a regular tuple
    (``time, kwargs = event``), can be constructed positionally
    (``Event(0.5, {"pitch": 60})``) or by keyword.
    """

    time: float
    kwargs: _KwargsDict


# A callable invoked as ``instrument(**event.kwargs)`` that returns the
# rendered Audio for one event. Declare the kwargs you use and absorb the
# rest with ``**kwargs``.
Instrument: TypeAlias = Callable[..., Audio]


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
    """A list of :class:`Event` with music-specific helpers.

    ``Score`` subclasses :class:`collections.UserList`, so all the standard
    list operations work and preserve ``Score`` type — ``+``, ``*``,
    slicing, ``+=``, ``.copy()``, ``.append()``, etc.

    The ``time`` field of each event is interpreted in *seconds* unless you
    pass a :class:`Metronome` to :meth:`render`, in which case ``time`` is
    treated as *ticks* and converted via ``metronome.tick_to_seconds``. The
    metronome is not stored on the score — keep it as a separate variable
    (or use the ``(score, metronome)`` tuple returned by :meth:`from_midi`).

    Construct from any iterable of events::

        Score([Event(0.0, {"pitch": 60}), Event(1.0, {"pitch": 64})])

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
        cls,
        midi: Union[str, os.PathLike, IO, mido.MidiFile],
        *,
        as_notes: bool = True,
        all_events: bool = False,
    ) -> Tuple["Score", "MIDIMetronome"]:
        """Parses a MIDI file into a ``(score, metronome)`` pair.

        Every emitted :class:`Event` has ``time`` set to its absolute MIDI
        tick and ``kwargs["mtype"]`` (message type) identifying the kind of
        event. The exact kwargs schema depends on the flags below.

        When ``as_notes=True`` (default), each ``note_on``/``note_off`` pair
        across all tracks is collapsed into a single ``"note"`` event with:

        * ``kwargs["mtype"]`` — the literal string ``"note"``.
        * ``kwargs["duration"]`` — the note's duration in seconds.
        * ``kwargs["duration_ticks"]`` — the note's duration in MIDI ticks.
        * ``kwargs["pitch"]`` — MIDI pitch (0–127).
        * ``kwargs["velocity"]`` — MIDI NOTE_ON velocity (0–127).
        * ``kwargs["program"]`` — MIDI program (instrument) number (0–127).
        * ``kwargs["is_drum"]`` — ``True`` if the note is on MIDI channel
          10 (zero-indexed: 9), the conventional percussion channel.
        * ``kwargs["channel"]`` — MIDI channel (0-15).

        When ``as_notes=False``, raw ``"note_on"`` and ``"note_off"`` events
        are emitted as separate :class:`Event`\\ s instead — each with
        ``kwargs`` equal to the underlying ``mido`` message's attributes
        (with ``"type"`` renamed to ``"mtype"``) sans its delta-``time``
        field.

        When ``all_events=True``, all other MIDI messages (tempo changes,
        program changes, control changes, ...) are also emitted with the
        same barebones ``msg.dict()``-style kwargs.

        Pass the returned ``metronome`` to :meth:`render` so ticks are
        converted to seconds using the file's actual tempo map.

        Args:
            midi: A path, a file-like object, or an already-parsed
                :class:`mido.MidiFile`.
            as_notes: If ``True`` (default), collapse ``note_on``/``note_off``
                pairs into a single ``"note"`` event. If ``False``, emit them
                as separate events.
            all_events: If ``True``, also emit non-note MIDI messages
                (``set_tempo``, ``program_change``, ``control_change``, ...).
                Defaults to ``False`` (note events only).
        """
        mid = _load_midi(midi)
        metronome = MIDIMetronome(mid)
        events = _parse_midi_events(
            mid, metronome, as_notes=as_notes, all_events=all_events
        )
        events.sort(key=lambda e: e.time)
        return cls(events), metronome

    def segment(
        self,
        *,
        offset: Optional[float] = None,
        duration: Optional[float] = None,
        relativize: bool = True,
    ) -> "Score":
        """Returns the events whose ``time`` falls in ``[offset, offset+duration)``.

        The right boundary is open with a small epsilon (``offset + duration
        - 1e-9``) to suppress floating-point off-by-ones at exact-end matches.

        Args:
            offset: Lower bound on event time (inclusive). Defaults to the
                beginning of the score (``0.0``).
            duration: Length of the window. Defaults to the rest of the score
                (no upper bound).
            relativize: If ``True`` (default), shift every kept event's
                ``time`` by ``-offset`` so the returned score begins at 0.
                If ``False``, keep the original timestamps.
        """
        start = offset or 0.0
        end = float("inf") if duration is None else start + duration - _SEGMENT_EPS
        shift = start if relativize else 0.0
        return Score(
            Event(e.time - shift, e.kwargs) for e in self if start <= e.time < end
        )

    def render(
        self,
        instrument: Instrument,
        metronome: Optional[Metronome] = None,
    ) -> Audio:
        """Renders this score to a single mixed :class:`Audio`.

        Each event is rendered by calling ``instrument(**event.kwargs)`` and
        mixed in at its onset time. Onsets are interpreted as ticks (and
        converted via ``metronome.tick_to_seconds``) when a metronome is
        given, otherwise as seconds.

        The instrument is time-agnostic: it receives only the event's
        ``kwargs``, not its ``time``. To vary an instrument by onset, copy
        the time into ``kwargs`` when building the score. For per-event
        dispatch (e.g. different instruments for drums vs. pitched notes),
        branch inside the instrument.

        All instrument outputs must share a ``sample_rate`` and a
        ``num_channels``; otherwise ``ValueError`` is raised. An empty
        score returns a zero-length, mono ``Audio``.
        """
        if not self:
            return Audio.zeros(0, 1)

        rendered: List[Tuple[float, Audio]] = []
        for event in self:
            audio = instrument(**event.kwargs)
            if not isinstance(audio, Audio):
                raise TypeError(
                    f"instrument(**event.kwargs) must return an Audio; "
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


def _parse_midi_events(
    mid: mido.MidiFile,
    metronome: "MIDIMetronome",
    *,
    as_notes: bool = True,
    all_events: bool = False,
) -> List[Event]:
    """Walks all tracks and converts MIDI messages to :class:`Event`\\ s.

    See :meth:`Score.from_midi` for the semantics of ``as_notes`` and
    ``all_events``.
    """
    events: List[Event] = []
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
            is_note_on = msg.type == "note_on" and msg.velocity > 0
            is_note_off = msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            )
            is_note = is_note_on or is_note_off

            if msg.type == "program_change":
                programs[msg.channel] = msg.program

            if is_note and as_notes:
                # Pair note_on/note_off into a single "note" event.
                if is_note_on:
                    active[(msg.channel, msg.note)] = (
                        abs_tick,
                        msg.velocity,
                        programs.get(msg.channel, 0),
                    )
                else:  # is_note_off
                    key = (msg.channel, msg.note)
                    if key not in active:
                        continue  # Stray note_off; skip silently.
                    on_tick, velocity, program = active.pop(key)
                    off_tick = abs_tick
                    duration = metronome.tick_to_seconds(
                        off_tick
                    ) - metronome.tick_to_seconds(on_tick)
                    events.append(
                        Event(
                            time=on_tick,
                            kwargs={
                                "mtype": "note",
                                "duration": duration,
                                "duration_ticks": off_tick - on_tick,
                                "pitch": msg.note,
                                "velocity": velocity,
                                "program": program,
                                "is_drum": msg.channel == 9,
                                "channel": msg.channel,
                            },
                        )
                    )
            elif (is_note and not as_notes) or (not is_note and all_events):
                # Barebones path: kwargs = msg attributes (sans delta-time).
                # Rename mido's ``"type"`` to ``"mtype"`` (message type) so it
                # doesn't shadow the ``type`` builtin when unpacked as **kwargs.
                kwargs = msg.dict()
                kwargs.pop("time", None)
                kwargs["mtype"] = kwargs.pop("type")
                events.append(Event(time=abs_tick, kwargs=kwargs))
    return events
