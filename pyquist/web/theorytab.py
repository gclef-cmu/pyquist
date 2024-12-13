import enum
import json
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests

from ..helper import pitch_name_to_pitch
from ..paths import CACHE_DIR as _ROOT_CACHE_DIR
from ..score import BasicMetronome, Metronome, Score

_CACHE_DIR = _ROOT_CACHE_DIR / "theorytab"

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

_CHORD_DEGREES_TO_LILY_NAME = {
    (4, 3): "",  # Major
    (3, 4): "m",  # Minor
    (3, 4, 3): "m7",  # Minor 7
    (4, 3, 3): "7",  # 7
    (4, 3, 4): "maj7",  # Major 7
    (5, 2): "sus4",  # Sus 4
    (4, 3, 7): "9^7",  # Add 9
    (3, 3): "dim",  # Dim
    (3, 3, 4): "m7.5-",  # Dim minor 7
    (2, 5): "sus2",  # Sus 2
    (5, 2, 3): "7sus4",  # 7 Sus 4
    (3, 4, 3, 4): "m9",  # Minor 9
    (4, 3, 4, 3): "maj9",  # Major 9
    (2, 5, 3): "7sus2",  # 7 Sus 2
    (3, 4, 7): "m9^7",  # Minor Add 9
    (3, 3, 3): "dim7",  # Dim 7
    (2, 5, 4): "maj7sus2",  # Major 7 Sus 2
    (4, 3, 3, 4): "9",  # 9
    (2, 3, 2): "11.9^7",  # Sus 2 Sus 4
    (4, 4): "aug",  # Aug
}


class HarmonyType(enum.Enum):
    SYMBOL = 0
    ROOT_POSITION_NOTES = 1


def _cumulative_intervals(intervals: Tuple[int, int, int, int, int, int]) -> List[int]:
    assert len(intervals) == 6
    return [0] + np.cumsum(intervals).tolist()


def _theorytab_note_to_pitch(note: Dict[str, Any], key: Dict[str, Any]) -> int:
    sd = note["sd"]
    root = pitch_name_to_pitch(key["tonic"] + "0")
    key_pitches = _cumulative_intervals(
        _THEORYTAB_SCALE_NAME_TO_PITCH_INTERVALS[key["scale"]]
    )
    pitch = key_pitches[int(sd[-1:]) - 1]
    sd = sd[:-1]
    while len(sd) > 0:
        if sd[0] == "#":
            pitch += 1
        elif sd[0] == "b":
            pitch -= 1
        else:
            raise ValueError(f"Invalid scale degree: {note['sd']}")
        sd = sd[1:]
    pitch += root
    pitch += note["octave"] * 12
    return pitch


def _theorytab_chord_to_pitches(
    chord: Dict[str, Any], key: Dict[str, Any]
) -> List[int]:
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

    # Convert to list
    chord_degrees_list = sorted(list(chord_degrees))

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

    # Create final chord
    return [key_tonic_pc + v for _, v in chord_degree_to_interval.items()]


def _theorytab_chord_to_symbol(chord: Dict[str, Any], key: Dict[str, Any]) -> str:
    # Goal is to convert theorytab JSON chord / key to something like "Cmaj7"
    # See _CHORD_DEGREES_TO_LILY_NAME above
    raise NotImplementedError("Left as exercise to intrepid student.")


def fetch_theorytab_json(id_or_url: str, *, ignore_cache=False) -> dict:
    # Parse TheoryTab ID from input
    id_or_url = id_or_url.strip()
    if len(id_or_url) == 11:
        theorytab_id = id_or_url
    else:
        url = urlparse(id_or_url)
        if "hooktheory.com" not in url.netloc:
            raise ValueError("Invalid TheoryTab URL.")
        query_params = parse_qs(url.query)
        theorytab_ids = query_params.get("idOfSong")
        if theorytab_ids is None:
            raise ValueError("Invalid TheoryTab URL.")
        theorytab_id = theorytab_ids[0]
    assert len(theorytab_id) == 11

    cache_file = _CACHE_DIR / f"{theorytab_id}.json"
    if cache_file.exists() and not ignore_cache:
        # Return cached data
        with open(cache_file, "r") as f:
            return json.load(f)
    else:
        # Call API
        hooktheory_api_url = f"https://api.hooktheory.com/v1/songs/public/{theorytab_id}?fields=ID,song,jsonData"
        response = requests.get(hooktheory_api_url)
        if response.status_code != 200:
            raise Exception(
                f"Error retrieving TheoryTab data: {response.status_code} - {response.text}"
            )
        data = response.json()
        if "jsonData" not in data:
            raise Exception(f"Unexpected response from TheoryTab API: {data}")
        if data["jsonData"] is None:
            # TODO(chrisdonahue): parse legacy XML data?
            raise NotImplementedError(
                "This song has an unsupported format. Try a different one."
            )
        result = json.loads(data["jsonData"])
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f)
        return result


def theorytab_json_to_score(
    song_data: dict,
    *,
    durations_in_beats: bool = False,
    harmony_type: HarmonyType = HarmonyType.ROOT_POSITION_NOTES,
    melody_octave: int = 5,
    harmony_octave: int = 4,
) -> Tuple[Metronome, Score, Score]:
    # Create mapping between beats and times
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
    bpm = tempos[0]["bpm"]
    met = BasicMetronome(bpm)

    # Create mapping between scale degrees and pitches
    keys = song_data["keys"]
    if len(keys) != 1 or keys[0]["beat"] != 1:
        raise NotImplementedError(
            "This song has an unsupported key signature. Try a different one."
        )
    key = keys[0]

    # Parse melody
    melody = []
    for note in song_data["notes"]:
        if note["isRest"]:
            continue
        beat = note["beat"] - 1.0
        duration = note["duration"]
        if not durations_in_beats:
            duration = met.beat_to_time(duration)
        pitch = _theorytab_note_to_pitch(note, key) + melody_octave * 12
        melody.append((beat, {"duration": duration, "pitch": pitch}))
    melody = sorted(melody, key=lambda x: x[0])

    # Parse chords
    harmony = []
    for chord in song_data["chords"]:
        if chord["isRest"]:
            continue
        beat = chord["beat"] - 1.0
        duration = chord["duration"]
        if not durations_in_beats:
            duration = met.beat_to_time(duration)
        if harmony_type == HarmonyType.SYMBOL:
            symbol = _theorytab_chord_to_symbol(chord, key)
            harmony.append((beat, {"duration": duration, "symbol": symbol}))
        elif harmony_type == HarmonyType.ROOT_POSITION_NOTES:
            pitches = _theorytab_chord_to_pitches(chord, key)
            pitches = [p + harmony_octave * 12 for p in pitches]
            for pitch in pitches:
                harmony.append((beat, {"duration": duration, "pitch": pitch}))
        else:
            raise NotImplementedError()
    harmony = sorted(harmony, key=lambda x: x[0])

    return met, melody, harmony


def fetch_from_theorytab(id_or_url: str, **kwargs) -> Tuple[Metronome, Score, Score]:
    """Fetches a pop song from TheoryTab and returns a score.

    To get an `id_or_url` input:
        1. Find a song on TheoryTab, e.g., https://www.hooktheory.com/theorytab/view/the-beatles/a-hard-days-night
        2. Right click on "Open in Hookpad", and click "Copy Link Address."
        3. Paste the link address into the `id_or_url` input.

    Args:
        id_or_url: The TheoryTab ID or URL.
        **kwargs: Additional keyword arguments to `theorytab_json_to_score`.
    """
    return theorytab_json_to_score(fetch_theorytab_json(id_or_url), **kwargs)


if __name__ == "__main__":
    import sys

    from ..audio import Audio
    from ..cli import play
    from ..helper import dbfs_to_gain, pitch_to_frequency
    from ..score import render_score

    # Get ID from command line
    if len(sys.argv) != 2:
        raise ValueError("Usage: python -m pyquist_lib.pyquist.web.theorytab <id>")
    id_or_url = sys.argv[1]

    # Basic instrument functions
    # TODO(chrisdonahue): add something similar to library?
    def _osc(
        duration: float, pitch: int, dbfs: float, sample_rate: int = 44100
    ) -> Audio:
        t = np.arange(int(duration * sample_rate)) / sample_rate
        result = np.sin(2 * np.pi * t * pitch_to_frequency(pitch)) * dbfs_to_gain(dbfs)
        # TODO(chrisdonahue): Add envelope
        return Audio.from_array(result, sample_rate)

    def _melody(*args, **kwargs):
        return _osc(*args, **kwargs, dbfs=-12)

    def _harmony(*args, **kwargs):
        return _osc(*args, **kwargs, dbfs=-18)

    # Grab score
    metronome, melody, harmony = fetch_from_theorytab(sys.argv[1])

    # Convert to playable score
    playable_score = []
    for time, kwargs in melody:
        playable_score.append((time, _melody, kwargs))
    for time, kwargs in harmony:
        playable_score.append((time, _harmony, kwargs))

    # Play audio
    play(render_score(playable_score, metronome), normalize=True)
