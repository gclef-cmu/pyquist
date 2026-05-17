import unittest

import numpy as np

from .audio import Audio
from .paths import TEST_DATA_DIR
from .score import (
    BasicMetronome,
    Metronome,
    MIDIMetronome,
    Score,
    SoundEvent,
)

_BOLERO_MIDI = TEST_DATA_DIR / "ravel_bolero.mid"


def _const_tone(event: SoundEvent) -> Audio:
    """Test instrument: returns a flat-amplitude audio of ``event.kwargs['duration']`` seconds."""
    sample_rate = event.kwargs.get("sample_rate", 1000)
    return Audio(
        np.full(
            int(event.kwargs["duration"] * sample_rate),
            event.kwargs["value"],
            dtype=np.float32,
        ),
        sample_rate=sample_rate,
    )


def _stereo_tone(event: SoundEvent) -> Audio:
    sample_rate = event.kwargs.get("sample_rate", 1000)
    return Audio(
        np.full(
            (int(event.kwargs["duration"] * sample_rate), 2),
            event.kwargs["value"],
            dtype=np.float32,
        ),
        sample_rate=sample_rate,
    )


class TestSoundEvent(unittest.TestCase):
    def test_construction_positional_and_keyword(self):
        a = SoundEvent(0.5, {"pitch": 60})
        b = SoundEvent(time=0.5, kwargs={"pitch": 60})
        self.assertEqual(a, b)
        self.assertEqual(a.time, 0.5)
        self.assertEqual(a.kwargs, {"pitch": 60})

    def test_unpacks_like_tuple(self):
        event = SoundEvent(0.5, {"pitch": 60})
        time, kwargs = event
        self.assertEqual(time, 0.5)
        self.assertEqual(kwargs, {"pitch": 60})


class TestScoreListBehavior(unittest.TestCase):
    def test_construction_from_iterable(self):
        events = [SoundEvent(0.0, {}), SoundEvent(1.0, {})]
        score = Score(events)
        self.assertEqual(len(score), 2)
        self.assertEqual(score[0], events[0])

    def test_empty_construction(self):
        self.assertEqual(len(Score()), 0)

    def test_append_and_iterate(self):
        score = Score()
        score.append(SoundEvent(0.0, {}))
        score.append(SoundEvent(1.0, {}))
        times = [e.time for e in score]
        self.assertEqual(times, [0.0, 1.0])

    def test_slicing_returns_score(self):
        score = Score([SoundEvent(t, {}) for t in (0.0, 1.0, 2.0)])
        sliced = score[1:]
        self.assertIsInstance(sliced, Score)
        self.assertEqual(len(sliced), 2)

    def test_addition_returns_score(self):
        a = Score([SoundEvent(0.0, {})])
        b = Score([SoundEvent(1.0, {})])
        combined = a + b
        self.assertIsInstance(combined, Score)
        self.assertEqual(len(combined), 2)
        # The whole point of UserList: this also works for 3+ scores.
        c = Score([SoundEvent(2.0, {})])
        big = a + b + c
        self.assertIsInstance(big, Score)
        self.assertEqual([e.time for e in big], [0.0, 1.0, 2.0])

    def test_iadd_returns_score(self):
        a = Score([SoundEvent(0.0, {})])
        a += Score([SoundEvent(1.0, {})])
        self.assertIsInstance(a, Score)
        self.assertEqual(len(a), 2)

    def test_multiplication_returns_score(self):
        a = Score([SoundEvent(0.0, {})])
        repeated = a * 3
        self.assertIsInstance(repeated, Score)
        self.assertEqual(len(repeated), 3)

    def test_copy_returns_score(self):
        a = Score([SoundEvent(0.0, {})])
        self.assertIsInstance(a.copy(), Score)


class TestScoreProperties(unittest.TestCase):
    def test_start_end_duration(self):
        score = Score(SoundEvent(t, {}) for t in (0.5, 1.5, 3.0))
        self.assertEqual(score.start_time, 0.5)
        self.assertEqual(score.end_time, 3.0)
        self.assertEqual(score.duration, 2.5)

    def test_single_event(self):
        score = Score([SoundEvent(2.0, {})])
        self.assertEqual(score.start_time, 2.0)
        self.assertEqual(score.end_time, 2.0)
        self.assertEqual(score.duration, 0.0)

    def test_unordered_events(self):
        # start_/end_time use min/max, not first/last.
        score = Score([SoundEvent(5.0, {}), SoundEvent(1.0, {}), SoundEvent(3.0, {})])
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
    def test_offset_and_duration(self):
        score = Score(SoundEvent(t, {}) for t in (0.0, 0.5, 1.0, 1.5, 2.0))
        sliced = score.segment(offset=0.5, duration=1.0)
        self.assertEqual([e.time for e in sliced], [0.5, 1.0])

    def test_returns_score(self):
        score = Score([SoundEvent(0.0, {}), SoundEvent(0.5, {})])
        self.assertIsInstance(score.segment(offset=0.0, duration=10.0), Score)

    def test_eps_excludes_exact_end(self):
        score = Score([SoundEvent(0.0, {}), SoundEvent(1.0, {})])
        self.assertEqual(list(score.segment(offset=0.0, duration=1.0)), [score[0]])

    def test_inclusive_left_boundary(self):
        score = Score([SoundEvent(0.0, {}), SoundEvent(0.5, {})])
        self.assertEqual(list(score.segment(offset=0.5, duration=1.0)), [score[1]])

    def test_no_duration_means_unbounded(self):
        score = Score(SoundEvent(t, {}) for t in (0.0, 1.0, 100.0))
        self.assertEqual([e.time for e in score.segment(offset=0.5)], [1.0, 100.0])


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
        # _const_tone is an Instrument: takes SoundEvent → returns Audio.
        score = Score([SoundEvent(0.0, {"value": 0.5, "duration": 0.01})])
        audio = score.render(_const_tone)
        self.assertEqual(audio.sample_rate, 1000)
        self.assertEqual(audio.num_channels, 1)
        self.assertEqual(audio.num_samples, 10)
        self.assertTrue(np.allclose(audio.samples, 0.5))

    def test_per_event_dispatch_inside_instrument(self):
        # Per-event dispatch is just a branch inside the instrument — no
        # separate factory concept needed.
        captured: list[tuple[str, float]] = []

        def dispatch(event):
            tag = "loud" if event.time >= 0.005 else "quiet"
            captured.append((tag, event.kwargs["value"]))
            return _const_tone(event)

        score = Score(
            [
                SoundEvent(0.0, {"value": 0.1, "duration": 0.01}),
                SoundEvent(0.01, {"value": 0.2, "duration": 0.01}),
            ]
        )
        score.render(dispatch)
        self.assertEqual(captured, [("quiet", 0.1), ("loud", 0.2)])

    def test_kwargs_style_via_lambda_wrap(self):
        # Existing **kwargs-style functions adapt with a one-line lambda.
        def kwargs_only(**kw):
            return Audio(
                np.full(int(kw["duration"] * 1000), kw["value"], dtype=np.float32),
                sample_rate=1000,
            )

        score = Score([SoundEvent(0.0, {"value": 0.5, "duration": 0.01})])
        audio = score.render(lambda e: kwargs_only(**e.kwargs))
        self.assertEqual(audio.num_samples, 10)
        self.assertTrue(np.allclose(audio.samples, 0.5))

    def test_single_event_with_offset(self):
        score = Score([SoundEvent(0.005, {"value": 0.5, "duration": 0.01})])
        audio = score.render(_const_tone)
        self.assertGreaterEqual(audio.num_samples, 15)
        self.assertTrue(np.all(audio.samples[:5] == 0.0))
        self.assertTrue(np.all(audio.samples[5:15] == 0.5))

    def test_non_overlapping(self):
        score = Score(
            [
                SoundEvent(0.0, {"value": 0.3, "duration": 0.01}),
                SoundEvent(0.01, {"value": 0.7, "duration": 0.01}),
            ]
        )
        audio = score.render(_const_tone)
        self.assertEqual(audio.num_samples, 20)
        self.assertTrue(np.allclose(audio.samples[:10], 0.3))
        self.assertTrue(np.allclose(audio.samples[10:20], 0.7))

    def test_overlapping_events_mix_additively(self):
        score = Score(
            [
                SoundEvent(0.0, {"value": 0.2, "duration": 0.01}),
                SoundEvent(0.0, {"value": 0.3, "duration": 0.01}),
            ]
        )
        audio = score.render(_const_tone)
        self.assertEqual(audio.num_samples, 10)
        self.assertTrue(np.allclose(audio.samples, 0.5))

    def test_metronome_converts_ticks(self):
        # 60_000 BPM → 1 tick = 0.001s = 1 sample @ sr=1000.
        m = BasicMetronome(60_000)
        score = Score([SoundEvent(5.0, {"value": 0.5, "duration": 0.01})])
        audio = score.render(_const_tone, metronome=m)
        self.assertTrue(np.all(audio.samples[:5] == 0.0))
        self.assertTrue(np.all(audio.samples[5:15] == 0.5))

    def test_inconsistent_sample_rates_raises(self):
        score = Score(
            [
                SoundEvent(0.0, {"value": 0.5, "duration": 0.01, "sample_rate": 1000}),
                SoundEvent(0.01, {"value": 0.5, "duration": 0.01, "sample_rate": 2000}),
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            score.render(_const_tone)
        self.assertIn("sample rate", str(ctx.exception).lower())

    def test_inconsistent_channel_counts_raises(self):
        def dispatch(event):
            return _stereo_tone(event) if event.time >= 0.005 else _const_tone(event)

        score = Score(
            [
                SoundEvent(0.0, {"value": 0.5, "duration": 0.01}),
                SoundEvent(0.01, {"value": 0.5, "duration": 0.01}),
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            score.render(dispatch)
        self.assertIn("channel", str(ctx.exception).lower())

    def test_missing_sample_rate_raises(self):
        def _no_sr_instrument(_event):
            return Audio(np.zeros(10, dtype=np.float32))  # no sample_rate

        score = Score([SoundEvent(0.0, {})])
        with self.assertRaises(ValueError):
            score.render(_no_sr_instrument)

    def test_invalid_return_type_raises(self):
        def bad(_event):
            return 42

        with self.assertRaises(TypeError) as ctx:
            Score([SoundEvent(0.0, {})]).render(bad)
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
            self.assertIsInstance(event, SoundEvent)

    def test_event_kwargs_have_expected_keys(self):
        score, _ = Score.from_midi(_BOLERO_MIDI)
        expected = {"off_tick", "duration", "pitch", "velocity", "program", "is_drum"}
        for event in score[:50]:
            self.assertEqual(set(event.kwargs.keys()), expected)

    def test_kwarg_value_ranges(self):
        score, _ = Score.from_midi(_BOLERO_MIDI)
        for event in score:
            kw = event.kwargs
            self.assertGreaterEqual(kw["pitch"], 0)
            self.assertLessEqual(kw["pitch"], 127)
            self.assertGreaterEqual(kw["velocity"], 0)
            self.assertLessEqual(kw["velocity"], 127)
            self.assertGreaterEqual(kw["program"], 0)
            self.assertLessEqual(kw["program"], 127)
            self.assertIsInstance(kw["is_drum"], bool)
            self.assertGreater(kw["duration"], 0.0)
            self.assertGreaterEqual(kw["off_tick"], event.time)

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
