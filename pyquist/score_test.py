import unittest

import numpy as np

from .audio import Audio
from .paths import TEST_DATA_DIR
from .score import (
    BasicMetronome,
    Event,
    Metronome,
    MIDIMetronome,
    Score,
)

_BOLERO_MIDI = TEST_DATA_DIR / "ravel_bolero.mid"


def _const_tone(duration, value, sample_rate=1000, **kwargs) -> Audio:
    """Test instrument: returns a flat-amplitude audio of ``duration`` seconds."""
    return Audio(
        np.full(int(duration * sample_rate), value, dtype=np.float32),
        sample_rate=sample_rate,
    )


def _stereo_tone(duration, value, sample_rate=1000, **kwargs) -> Audio:
    return Audio(
        np.full((int(duration * sample_rate), 2), value, dtype=np.float32),
        sample_rate=sample_rate,
    )


class TestEvent(unittest.TestCase):
    def test_construction_positional_and_keyword(self):
        a = Event(0.5, {"pitch": 60})
        b = Event(time=0.5, kwargs={"pitch": 60})
        self.assertEqual(a, b)
        self.assertEqual(a.time, 0.5)
        self.assertEqual(a.kwargs, {"pitch": 60})

    def test_unpacks_like_tuple(self):
        event = Event(0.5, {"pitch": 60})
        time, kwargs = event
        self.assertEqual(time, 0.5)
        self.assertEqual(kwargs, {"pitch": 60})


class TestScoreListBehavior(unittest.TestCase):
    def test_construction_from_iterable(self):
        events = [Event(0.0, {}), Event(1.0, {})]
        score = Score(events)
        self.assertEqual(len(score), 2)
        self.assertEqual(score[0], events[0])

    def test_empty_construction(self):
        self.assertEqual(len(Score()), 0)

    def test_append_and_iterate(self):
        score = Score()
        score.append(Event(0.0, {}))
        score.append(Event(1.0, {}))
        times = [e.time for e in score]
        self.assertEqual(times, [0.0, 1.0])

    def test_slicing_returns_score(self):
        score = Score([Event(t, {}) for t in (0.0, 1.0, 2.0)])
        sliced = score[1:]
        self.assertIsInstance(sliced, Score)
        self.assertEqual(len(sliced), 2)

    def test_addition_returns_score(self):
        a = Score([Event(0.0, {})])
        b = Score([Event(1.0, {})])
        combined = a + b
        self.assertIsInstance(combined, Score)
        self.assertEqual(len(combined), 2)
        # The whole point of UserList: this also works for 3+ scores.
        c = Score([Event(2.0, {})])
        big = a + b + c
        self.assertIsInstance(big, Score)
        self.assertEqual([e.time for e in big], [0.0, 1.0, 2.0])

    def test_iadd_returns_score(self):
        a = Score([Event(0.0, {})])
        a += Score([Event(1.0, {})])
        self.assertIsInstance(a, Score)
        self.assertEqual(len(a), 2)

    def test_multiplication_returns_score(self):
        a = Score([Event(0.0, {})])
        repeated = a * 3
        self.assertIsInstance(repeated, Score)
        self.assertEqual(len(repeated), 3)

    def test_copy_returns_score(self):
        a = Score([Event(0.0, {})])
        self.assertIsInstance(a.copy(), Score)


class TestScoreProperties(unittest.TestCase):
    def test_start_end_duration(self):
        score = Score(Event(t, {}) for t in (0.5, 1.5, 3.0))
        self.assertEqual(score.start_time, 0.5)
        self.assertEqual(score.end_time, 3.0)
        self.assertEqual(score.duration, 2.5)

    def test_single_event(self):
        score = Score([Event(2.0, {})])
        self.assertEqual(score.start_time, 2.0)
        self.assertEqual(score.end_time, 2.0)
        self.assertEqual(score.duration, 0.0)

    def test_unordered_events(self):
        # start_/end_time use min/max, not first/last.
        score = Score([Event(5.0, {}), Event(1.0, {}), Event(3.0, {})])
        self.assertEqual(score.start_time, 1.0)
        self.assertEqual(score.end_time, 5.0)

    def test_empty_score_properties_raise(self):
        empty = Score()
        with self.assertRaises(ValueError):
            empty.start_time
        with self.assertRaises(ValueError):
            empty.end_time
        with self.assertRaises(ValueError):
            empty.duration


class TestScoreSegment(unittest.TestCase):
    def test_offset_and_duration_default_relativizes(self):
        # Default: event times are shifted so the segment starts at 0.
        score = Score(Event(t, {}) for t in (0.0, 0.5, 1.0, 1.5, 2.0))
        sliced = score.segment(offset=0.5, duration=1.0)
        self.assertEqual([e.time for e in sliced], [0.0, 0.5])

    def test_relativize_false_preserves_original_times(self):
        score = Score(Event(t, {}) for t in (0.0, 0.5, 1.0, 1.5, 2.0))
        sliced = score.segment(offset=0.5, duration=1.0, relativize=False)
        self.assertEqual([e.time for e in sliced], [0.5, 1.0])

    def test_relativize_preserves_kwargs(self):
        # Only `time` is touched; kwargs (including duration_ticks etc.) stay.
        score = Score([Event(2.0, {"pitch": 60, "duration": 0.25})])
        sliced = score.segment(offset=1.0, duration=2.0)
        self.assertEqual(sliced[0].time, 1.0)
        self.assertEqual(sliced[0].kwargs, {"pitch": 60, "duration": 0.25})

    def test_returns_score(self):
        score = Score([Event(0.0, {}), Event(0.5, {})])
        self.assertIsInstance(score.segment(offset=0.0, duration=10.0), Score)

    def test_eps_excludes_exact_end(self):
        score = Score([Event(0.0, {}), Event(1.0, {})])
        # At offset=0 the relativize shift is 0, so the kept event equals the original.
        self.assertEqual(list(score.segment(offset=0.0, duration=1.0)), [score[0]])

    def test_inclusive_left_boundary(self):
        score = Score([Event(0.0, {}), Event(0.5, {})])
        sliced = score.segment(offset=0.5, duration=1.0)
        self.assertEqual([e.time for e in sliced], [0.0])

    def test_no_duration_means_unbounded(self):
        # Default still relativizes when only offset is given.
        score = Score(Event(t, {}) for t in (0.0, 1.0, 100.0))
        self.assertEqual([e.time for e in score.segment(offset=0.5)], [0.5, 99.5])


class TestBasicMetronome(unittest.TestCase):
    def test_basic(self):
        m = BasicMetronome(120)
        self.assertEqual(m.bpm, 120)
        self.assertEqual(m.beat_duration, 0.5)
        self.assertEqual(m.tick_to_seconds(1.0), 0.5)
        self.assertEqual(m.seconds_to_tick(0.5), 1.0)

    def test_60_bpm(self):
        m = BasicMetronome(60)
        self.assertEqual(m.beat_duration, 1.0)
        self.assertEqual(m.tick_to_seconds(1.0), 1.0)
        self.assertEqual(m.seconds_to_tick(1.0), 1.0)

    def test_round_trip(self):
        m = BasicMetronome(140)
        for tick in [0.0, 1.0, 2.5, 17.3]:
            self.assertAlmostEqual(m.seconds_to_tick(m.tick_to_seconds(tick)), tick)

    def test_subclassable(self):
        class Halving(Metronome):
            def tick_to_seconds(self, tick: float) -> float:
                return tick / 2

            def seconds_to_tick(self, seconds: float) -> float:
                return seconds * 2

        h = Halving()
        self.assertEqual(h.tick_to_seconds(4.0), 2.0)
        self.assertEqual(h.seconds_to_tick(2.0), 4.0)


class TestScoreRender(unittest.TestCase):
    def test_uniform_instrument(self):
        # _const_tone is an Instrument: called with **event.kwargs → Audio.
        score = Score([Event(0.0, {"value": 0.5, "duration": 0.01})])
        audio = score.render(_const_tone)
        self.assertEqual(audio.sample_rate, 1000)
        self.assertEqual(audio.num_channels, 1)
        self.assertEqual(audio.num_samples, 10)
        self.assertTrue(np.allclose(audio.samples, 0.5))

    def test_per_event_dispatch_inside_instrument(self):
        # Per-event dispatch is just a branch inside the instrument — no
        # separate factory concept needed. The deciding kwarg is captured as
        # a named parameter; the rest flow through via **kwargs.
        captured: list[tuple[str, float]] = []

        def dispatch(value, **kwargs):
            tag = "loud" if value >= 0.15 else "quiet"
            captured.append((tag, value))
            return _const_tone(value=value, **kwargs)

        score = Score(
            [
                Event(0.0, {"value": 0.1, "duration": 0.01}),
                Event(0.01, {"value": 0.2, "duration": 0.01}),
            ]
        )
        score.render(dispatch)
        self.assertEqual(captured, [("quiet", 0.1), ("loud", 0.2)])

    def test_kwargs_style_instrument_passed_directly(self):
        # A **kwargs-style function is the native instrument form — no
        # wrapping needed; render calls it as instrument(**event.kwargs).
        def kwargs_only(duration, value, **kw):
            return Audio(
                np.full(int(duration * 1000), value, dtype=np.float32),
                sample_rate=1000,
            )

        score = Score([Event(0.0, {"value": 0.5, "duration": 0.01})])
        audio = score.render(kwargs_only)
        self.assertEqual(audio.num_samples, 10)
        self.assertTrue(np.allclose(audio.samples, 0.5))

    def test_instrument_missing_kwargs_sink_raises_on_extra_keys(self):
        # An instrument without a **kwargs sink raises TypeError when the
        # event carries keys it doesn't declare (e.g. from_midi's rich kwargs).
        def strict(duration, value):
            return Audio(
                np.full(int(duration * 1000), value, dtype=np.float32),
                sample_rate=1000,
            )

        score = Score([Event(0.0, {"value": 0.5, "duration": 0.01, "extra": 1})])
        with self.assertRaises(TypeError):
            score.render(strict)

    def test_single_event_with_offset(self):
        score = Score([Event(0.005, {"value": 0.5, "duration": 0.01})])
        audio = score.render(_const_tone)
        self.assertGreaterEqual(audio.num_samples, 15)
        self.assertTrue(np.all(audio.samples[:5] == 0.0))
        self.assertTrue(np.all(audio.samples[5:15] == 0.5))

    def test_non_overlapping(self):
        score = Score(
            [
                Event(0.0, {"value": 0.3, "duration": 0.01}),
                Event(0.01, {"value": 0.7, "duration": 0.01}),
            ]
        )
        audio = score.render(_const_tone)
        self.assertEqual(audio.num_samples, 20)
        self.assertTrue(np.allclose(audio.samples[:10], 0.3))
        self.assertTrue(np.allclose(audio.samples[10:20], 0.7))

    def test_overlapping_events_mix_additively(self):
        score = Score(
            [
                Event(0.0, {"value": 0.2, "duration": 0.01}),
                Event(0.0, {"value": 0.3, "duration": 0.01}),
            ]
        )
        audio = score.render(_const_tone)
        self.assertEqual(audio.num_samples, 10)
        self.assertTrue(np.allclose(audio.samples, 0.5))

    def test_metronome_converts_ticks(self):
        # 60_000 BPM → 1 tick = 0.001s = 1 sample @ sr=1000.
        m = BasicMetronome(60_000)
        score = Score([Event(5.0, {"value": 0.5, "duration": 0.01})])
        audio = score.render(_const_tone, metronome=m)
        self.assertTrue(np.all(audio.samples[:5] == 0.0))
        self.assertTrue(np.all(audio.samples[5:15] == 0.5))

    def test_inconsistent_sample_rates_raises(self):
        score = Score(
            [
                Event(0.0, {"value": 0.5, "duration": 0.01, "sample_rate": 1000}),
                Event(0.01, {"value": 0.5, "duration": 0.01, "sample_rate": 2000}),
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            score.render(_const_tone)
        self.assertIn("sample rate", str(ctx.exception).lower())

    def test_inconsistent_channel_counts_raises(self):
        def dispatch(stereo, **kwargs):
            return _stereo_tone(**kwargs) if stereo else _const_tone(**kwargs)

        score = Score(
            [
                Event(0.0, {"value": 0.5, "duration": 0.01, "stereo": False}),
                Event(0.01, {"value": 0.5, "duration": 0.01, "stereo": True}),
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            score.render(dispatch)
        self.assertIn("channel", str(ctx.exception).lower())

    def test_missing_sample_rate_raises(self):
        def _no_sr_instrument(**kwargs):
            return Audio(np.zeros(10, dtype=np.float32))  # no sample_rate

        score = Score([Event(0.0, {})])
        with self.assertRaises(ValueError):
            score.render(_no_sr_instrument)

    def test_invalid_return_type_raises(self):
        def bad(**kwargs):
            return 42

        with self.assertRaises(TypeError) as ctx:
            Score([Event(0.0, {})]).render(bad)
        self.assertIn("Audio", str(ctx.exception))

    def test_empty_score_returns_zero_length_audio(self):
        audio = Score().render(_const_tone)
        self.assertEqual(audio.num_samples, 0)


class TestScoreFromMidi(unittest.TestCase):
    """Tests against the bundled Ravel Bolero MIDI file."""

    def test_returns_score_and_metronome(self):
        score, metronome = Score.from_midi(_BOLERO_MIDI)
        self.assertIsInstance(score, Score)
        self.assertIsInstance(metronome, MIDIMetronome)
        self.assertGreater(len(score), 0)
        for event in score:
            self.assertIsInstance(event, Event)

    def test_event_kwargs_have_expected_keys(self):
        score, _ = Score.from_midi(_BOLERO_MIDI)
        expected = {
            "mtype",
            "duration_ticks",
            "duration",
            "pitch",
            "velocity",
            "program",
            "is_drum",
            "channel",
        }
        for event in score[:50]:
            self.assertEqual(set(event.kwargs.keys()), expected)
            self.assertEqual(event.kwargs["mtype"], "note")

    def test_kwarg_value_ranges(self):
        score, _ = Score.from_midi(_BOLERO_MIDI)
        for event in score:
            kw = event.kwargs
            self.assertEqual(kw["mtype"], "note")
            self.assertGreaterEqual(kw["pitch"], 0)
            self.assertLessEqual(kw["pitch"], 127)
            self.assertGreaterEqual(kw["velocity"], 0)
            self.assertLessEqual(kw["velocity"], 127)
            self.assertGreaterEqual(kw["program"], 0)
            self.assertLessEqual(kw["program"], 127)
            self.assertIsInstance(kw["is_drum"], bool)
            self.assertGreater(kw["duration"], 0.0)
            self.assertGreater(kw["duration_ticks"], 0)

    def test_as_notes_false_emits_raw_note_on_and_note_off(self):
        score, _ = Score.from_midi(_BOLERO_MIDI, as_notes=False)
        types = {e.kwargs["mtype"] for e in score}
        # Only note_on / note_off in this mode (all_events=False).
        self.assertEqual(types, {"note_on", "note_off"})
        # Each event's kwargs should be the raw mido message attributes
        # (no "duration"/"duration_ticks"/"program"/"is_drum").
        for event in score[:20]:
            self.assertIn("note", event.kwargs)
            self.assertIn("velocity", event.kwargs)
            self.assertIn("channel", event.kwargs)
            self.assertNotIn("duration", event.kwargs)
            self.assertNotIn("duration_ticks", event.kwargs)
            self.assertNotIn("time", event.kwargs)  # delta-time stripped

    def test_as_notes_false_emits_at_least_twice_as_many_events(self):
        # Each paired "note" consumes one note_on + one note_off, so the raw
        # count is at least 2x the paired count (and possibly more if the
        # file contains stray unmatched note_on / note_off messages).
        notes, _ = Score.from_midi(_BOLERO_MIDI, as_notes=True)
        raw, _ = Score.from_midi(_BOLERO_MIDI, as_notes=False)
        self.assertGreaterEqual(len(raw), 2 * len(notes))

    def test_all_events_true_includes_non_note_events(self):
        score, _ = Score.from_midi(_BOLERO_MIDI, all_events=True)
        types = {e.kwargs["mtype"] for e in score}
        # "note" should still appear (as_notes=True is the default), and
        # standard non-note types should also be present in Bolero.
        self.assertIn("note", types)
        self.assertIn("set_tempo", types)
        self.assertIn("program_change", types)

    def test_all_events_with_as_notes_false_emits_only_raw_events(self):
        score, _ = Score.from_midi(_BOLERO_MIDI, as_notes=False, all_events=True)
        types = {e.kwargs["mtype"] for e in score}
        # No paired "note" events in this mode.
        self.assertNotIn("note", types)
        # Raw note events and tempo/program changes should still appear.
        self.assertIn("note_on", types)
        self.assertIn("note_off", types)
        self.assertIn("set_tempo", types)

    def test_events_are_sorted_by_time(self):
        score, _ = Score.from_midi(_BOLERO_MIDI)
        times = [e.time for e in score]
        self.assertEqual(times, sorted(times))

    def test_first_event_is_a_drum_at_tick_zero(self):
        score, _ = Score.from_midi(_BOLERO_MIDI)
        first = score[0]
        self.assertEqual(first.time, 0)
        self.assertTrue(first.kwargs["is_drum"])

    def test_accepts_already_parsed_mido_object(self):
        import mido

        mid = mido.MidiFile(str(_BOLERO_MIDI))
        score_a, _ = Score.from_midi(_BOLERO_MIDI)
        score_b, _ = Score.from_midi(mid)
        self.assertEqual(len(score_a), len(score_b))


class TestMIDIMetronome(unittest.TestCase):
    def test_zero_tick_is_zero_seconds(self):
        m = MIDIMetronome(_BOLERO_MIDI)
        self.assertEqual(m.tick_to_seconds(0), 0.0)

    def test_round_trip_within_tick_resolution(self):
        m = MIDIMetronome(_BOLERO_MIDI)
        for tick in [0, 384, 1000, 4608, 100_000]:
            seconds = m.tick_to_seconds(tick)
            self.assertAlmostEqual(m.seconds_to_tick(seconds), tick, delta=1)

    def test_known_tick_matches_known_seconds(self):
        # Probed against the bundled file: tempo is ~62 BPM (mspb=967741).
        # 1 quarter note = 384 ticks → 384 ticks ≈ 0.9677 s.
        m = MIDIMetronome(_BOLERO_MIDI)
        self.assertAlmostEqual(m.tick_to_seconds(384), 0.9677, delta=0.001)

    def test_tick_to_seconds_is_monotonic(self):
        m = MIDIMetronome(_BOLERO_MIDI)
        previous = -1.0
        for tick in range(0, 100_000, 1000):
            seconds = m.tick_to_seconds(tick)
            self.assertGreaterEqual(seconds, previous)
            previous = seconds

    def test_ticks_per_beat_matches_file(self):
        m = MIDIMetronome(_BOLERO_MIDI)
        self.assertEqual(m.ticks_per_beat, 384)


if __name__ == "__main__":
    unittest.main()
