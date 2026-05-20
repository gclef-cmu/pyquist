"""TheoryTab API client.

`TheoryTab <https://www.hooktheory.com/theorytab>`_ is a community database of
pop-song melody and chord progressions hosted by Hooktheory. This module
fetches a song by URL or 11-character ID and turns it into pyquist
:class:`pyquist.Score` objects ready for rendering.

Most users only need :func:`fetch`, which combines :func:`fetch_theorytab_json`
(network + caching) with :func:`theorytab_json_to_score` (parsing).

Downloaded JSON is cached under ``CACHE_DIR / "theorytab"`` so repeat lookups
are free.
"""

import json
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests

from ..helper import pitch_name_to_pitch
from ..paths import CACHE_DIR as _ROOT_CACHE_DIR
from ..score import BasicMetronome, Event, Metronome, Score

_CACHE_DIR = _ROOT_CACHE_DIR / "theorytab"

# TheoryTab song identifiers (``idOfSong`` in Hookpad URLs) are 11 characters.
_THEORYTAB_ID_LENGTH = 11

_HOOKTHEORY_API_URL = "https://api.hooktheory.com/v1/songs/public"

# Diatonic scales as ordered (whole-/half-) step intervals between successive
# notes. Each tuple has 6 entries (7 notes → 6 gaps).
_THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS = {
    "major": (2, 2, 1, 2, 2, 2),
    "dorian": (2, 1, 2, 2, 2, 1),
    "phrygian": (1, 2, 2, 2, 1, 2),
    "lydian": (2, 2, 2, 1, 2, 2),
    "mixolydian": (2, 2, 1, 2, 2, 1),
    "minor": (2, 1, 2, 2, 1, 2),
    "locrian": (1, 2, 2, 1, 2, 2),
    "harmonicMinor": (2, 1, 2, 2, 1, 3),
    "phrygianDominant": (1, 3, 1, 2, 1, 2),
}


# ---------------------------------------------------------------------------
# Music-theory helpers
# ---------------------------------------------------------------------------


def _cumulative_intervals(intervals: Tuple[int, int, int, int, int, int]) -> List[int]:
    """Converts inter-note steps to cumulative semitones from the tonic.

    e.g. major ``(2, 2, 1, 2, 2, 2)`` → ``[0, 2, 4, 5, 7, 9, 11]``.
    """
    assert len(intervals) == 6
    return [0] + np.cumsum(intervals).tolist()


def _theorytab_note_to_pitch(note: Dict[str, Any], key: Dict[str, Any]) -> int:
    """Converts a TheoryTab JSON melody note into an absolute MIDI pitch.

    TheoryTab notates pitches as ``<accidentals><scale-degree>`` (e.g. ``"#5"``,
    ``"b3"``, ``"5"``) plus an ``octave`` integer. The scale degree is
    interpreted in the song's current scale; accidentals shift by ±1 semitone.
    """
    sd_str = note["sd"]
    *accidentals, degree_char = sd_str
    degree = int(degree_char)

    key_pitches = _cumulative_intervals(
        _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS[key["scale"]]
    )
    pitch = key_pitches[degree - 1]
    for c in accidentals:
        if c == "#":
            pitch += 1
        elif c == "b":
            pitch -= 1
        else:
            raise ValueError(f"Invalid scale degree {sd_str!r}: unexpected {c!r}.")

    pitch += int(pitch_name_to_pitch(key["tonic"] + "0"))
    pitch += note["octave"] * 12
    return pitch


def _theorytab_chord_to_pitches(
    chord: Dict[str, Any], key: Dict[str, Any]
) -> List[int]:
    """Converts a TheoryTab JSON chord into a list of absolute MIDI pitches.

    Handles type (triad / 7 / 9 / ...), suspensions, additions, omissions,
    alterations, borrowed scales, and secondary dominants. Returns the chord
    in root position (no inversions).
    """
    # Build chord scale degrees
    chord_degrees = set(range(1, chord["type"] + 1, 2))

    # Apply suspensions
    for i, d in enumerate(chord["suspensions"]):
        if i == 0:
            assert 3 in chord_degrees
            chord_degrees.remove(3)
        assert d not in chord_degrees
        chord_degrees.add(d)

    # Apply adds
    for d in chord["adds"]:
        if d in [4, 6]:
            d += 7
        chord_degrees.add(d)

    # Apply omits
    for d in chord["omits"]:
        assert d in [3, 5]
        assert d in chord_degrees
        chord_degrees.remove(d)

    # Apply alterations
    for d in chord["alterations"]:
        d = int(d[1:])
        chord_degrees.add(d)

    chord_degrees_list = sorted(chord_degrees)

    # Find scale intervals
    key_tonic_pc = pitch_name_to_pitch(key["tonic"] + "0")
    key_scale_intervals_rel = _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS[key["scale"]]

    # Apply borrow (changes intervals)
    if isinstance(chord["borrowed"], list):
        key_scale_intervals = chord["borrowed"]
    else:
        if chord["borrowed"] in _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS:
            key_scale_intervals_rel = _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS[
                chord["borrowed"]
            ]
        key_scale_intervals = _cumulative_intervals(key_scale_intervals_rel)
    assert len(key_scale_intervals) == 7

    # Apply secondary (changes tonic and intervals)
    major_scale_intervals = _cumulative_intervals(
        _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS["major"]
    )
    if chord["applied"] > 0:
        key_tonic_pc = (key_tonic_pc + key_scale_intervals[chord["root"] - 1]) % 12
        chord["root"] = chord["applied"]
        key_scale_intervals = major_scale_intervals

    # Convert scale degrees to pitch offsets
    chord_degree_to_interval = {}
    for d in chord_degrees_list:
        d_abs = (chord["root"] - 1) + (d - 1)
        interval = key_scale_intervals[d_abs % 7]
        interval += 12 * (d_abs // 7)
        chord_degree_to_interval[d] = interval
        # NOTE: Not sure if this is a bug in Hookpad or what?
        if d == 7 and chord["applied"] == 7:
            chord_degree_to_interval[d] -= 1

    # Apply alterations
    for alt in chord["alterations"]:
        d = int(alt[1:])
        assert d in chord_degree_to_interval
        chord_degree_to_interval[d] += -1 if alt[0] == "b" else 1

    return [key_tonic_pc + v for _, v in chord_degree_to_interval.items()]


# ---------------------------------------------------------------------------
# URL / ID parsing
# ---------------------------------------------------------------------------


def _extract_theorytab_id(id_or_url: str) -> str:
    """Returns the 11-character TheoryTab song ID for ``id_or_url``.

    Accepts either a bare ID or a Hookpad/Hooktheory URL containing an
    ``idOfSong=...`` query parameter (e.g.
    ``https://hookpad.hooktheory.com/?idOfSong=RPxeJeJaob_``).
    Raises ``ValueError`` if neither form is recognized.
    """
    s = id_or_url.strip()
    if len(s) == _THEORYTAB_ID_LENGTH:
        return s
    url = urlparse(s)
    if "hooktheory.com" not in url.netloc:
        raise ValueError(f"Not a TheoryTab URL or ID: {id_or_url!r}.")
    ids = parse_qs(url.query).get("idOfSong")
    if not ids:
        raise ValueError(
            f"TheoryTab URL is missing 'idOfSong' query parameter: {id_or_url!r}."
        )
    return ids[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_theorytab_json(id_or_url: str, *, ignore_cache: bool = False) -> dict:
    """Fetches the raw TheoryTab JSON for a song, caching it on disk.

    Args:
        id_or_url: A bare 11-character TheoryTab song ID or any URL that
            contains it as the ``idOfSong`` query parameter.
        ignore_cache: If ``True``, re-fetch even if a cached copy exists.

    Raises:
        ValueError: if ``id_or_url`` can't be parsed.
        RuntimeError: on a non-200 response from the Hooktheory API or
            unexpected response shape.
        NotImplementedError: if the song's data is in a legacy format
            without parsed ``jsonData``.
    """
    theorytab_id = _extract_theorytab_id(id_or_url)
    assert len(theorytab_id) == _THEORYTAB_ID_LENGTH

    cache_file = _CACHE_DIR / f"{theorytab_id}.json"
    if cache_file.exists() and not ignore_cache:
        with open(cache_file, "r") as f:
            return json.load(f)

    response = requests.get(
        f"{_HOOKTHEORY_API_URL}/{theorytab_id}",
        params={"fields": "ID,song,jsonData"},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"TheoryTab API request failed for {theorytab_id}: "
            f"{response.status_code} - {response.text}"
        )
    data = response.json()
    if "jsonData" not in data:
        raise RuntimeError(f"Unexpected response from TheoryTab API: {data}")
    if data["jsonData"] is None:
        # TODO(chrisdonahue): parse legacy XML data?
        raise NotImplementedError(
            "This song has an unsupported format. Try a different one."
        )
    result = json.loads(data["jsonData"])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result, f)
    return result


def theorytab_json_to_score(
    song_data: dict,
    *,
    durations_in_beats: bool = False,
    melody_octave: int = 5,
    harmony_octave: int = 4,
) -> Tuple[Metronome, Score, Score]:
    """Parses a TheoryTab JSON object into a metronome and two scores.

    The ``time`` of each event is in **beats** (starting from 0). Pair it
    with the returned :class:`BasicMetronome` when rendering so beats are
    converted to seconds.

    Chord events are emitted as one :class:`Event` per chord tone in
    root position, each with a ``"pitch"`` kwarg — so a triad at beat 4
    becomes three events sharing ``time == 4.0``.

    Args:
        song_data: TheoryTab JSON (as returned by :func:`fetch_theorytab_json`).
        durations_in_beats: If ``False`` (default), each event's
            ``kwargs["duration"]`` is converted to **seconds** via the
            song's tempo. If ``True``, durations stay in beats and the
            instrument is responsible for any further conversion.
        melody_octave: Octave offset added to melody pitches.
        harmony_octave: Octave offset added to harmony pitches.

    Returns:
        ``(metronome, melody, harmony)`` where ``metronome`` is a
        :class:`BasicMetronome` carrying the song's tempo, and ``melody`` /
        ``harmony`` are :class:`Score` objects.

    Raises:
        NotImplementedError: if the song's time signature / tempo / key
            structure is more complex than this parser supports.
    """
    # We only support single-section songs in 3/4 or 4/4, single-tempo, single-key.
    meters = song_data["meters"]
    if (
        len(meters) != 1
        or meters[0]["beat"] != 1
        or meters[0]["numBeats"] not in [3, 4]
        or meters[0]["beatUnit"] != 1
    ):
        raise NotImplementedError(
            "This song has an unsupported time signature. Try a different one."
        )
    tempos = song_data["tempos"]
    if len(tempos) != 1 or tempos[0]["beat"] != 1:
        raise NotImplementedError(
            "This song has an unsupported tempo. Try a different one."
        )
    keys = song_data["keys"]
    if len(keys) != 1 or keys[0]["beat"] != 1:
        raise NotImplementedError(
            "This song has an unsupported key signature. Try a different one."
        )

    metronome = BasicMetronome(tempos[0]["bpm"])
    key = keys[0]

    # Parse melody
    melody: List[Event] = []
    for note in song_data["notes"]:
        if note["isRest"]:
            continue
        beat = note["beat"] - 1.0
        duration = note["duration"]
        if not durations_in_beats:
            duration = metronome.tick_to_seconds(duration)
        pitch = _theorytab_note_to_pitch(note, key) + melody_octave * 12
        melody.append(Event(beat, {"duration": duration, "pitch": pitch}))
    melody.sort(key=lambda e: e.time)

    # Parse harmony (chords)
    harmony: List[Event] = []
    for chord in song_data["chords"]:
        if chord["isRest"]:
            continue
        beat = chord["beat"] - 1.0
        duration = chord["duration"]
        if not durations_in_beats:
            duration = metronome.tick_to_seconds(duration)
        pitches = _theorytab_chord_to_pitches(chord, key)
        for pitch in pitches:
            harmony.append(
                Event(
                    beat,
                    {"duration": duration, "pitch": pitch + harmony_octave * 12},
                )
            )
    harmony.sort(key=lambda e: e.time)

    return metronome, Score(melody), Score(harmony)


def fetch_theorytab(id_or_url: str, **kwargs) -> Tuple[Metronome, Score, Score]:
    """Fetches and parses a TheoryTab song in one call.

    Equivalent to ``theorytab_json_to_score(fetch_theorytab_json(id_or_url),
    **kwargs)``.

    Example::

        from pyquist.web import fetch_theorytab
        metronome, melody, harmony = fetch_theorytab(
            "https://hookpad.hooktheory.com/?idOfSong=RPxeJeJaob_"
        )

    To get a URL: find a song on `TheoryTab
    <https://www.hooktheory.com/theorytab>`_, right-click "Open in Hookpad",
    and "Copy Link Address."

    Args:
        id_or_url: A bare 11-character TheoryTab ID or a Hookpad/Hooktheory URL.
        **kwargs: Forwarded to :func:`theorytab_json_to_score`.
    """
    return theorytab_json_to_score(fetch_theorytab_json(id_or_url), **kwargs)


if __name__ == "__main__":
    import sys

    from ..audio import Audio
    from ..device import play
    from ..helper import db_to_amplitude, pitch_to_frequency

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m pyquist.web.theorytab <id-or-url>")

    def _osc(
        duration: float, pitch: int, dbfs: float, sample_rate: int = 44100
    ) -> Audio:
        """A tiny sine-wave instrument. TODO: add an envelope."""
        t = np.arange(int(duration * sample_rate)) / sample_rate
        samples = np.sin(2 * np.pi * t * pitch_to_frequency(pitch)) * db_to_amplitude(
            dbfs
        )
        return Audio(samples, sample_rate=sample_rate)

    def _melody(event):
        return _osc(**event.kwargs, dbfs=-12)

    def _harmony(event):
        return _osc(**event.kwargs, dbfs=-18)

    # Grab the score and render each part with its own instrument, then mix.
    metronome, melody, harmony = fetch_theorytab(sys.argv[1])
    melody_audio = melody.render(_melody, metronome=metronome)
    harmony_audio = harmony.render(_harmony, metronome=metronome)
    mixed = Audio.zeros(
        max(melody_audio.num_samples, harmony_audio.num_samples),
        1,
        sample_rate=melody_audio.sample_rate,
    )
    mixed[: melody_audio.num_samples, :] += melody_audio
    mixed[: harmony_audio.num_samples, :] += harmony_audio
    play(mixed, normalize=True)
