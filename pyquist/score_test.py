import unittest

import numpy as np

from .audio import Audio
from .score import (
    BasicMetronome,
    Metronome,
    PlayableSoundEvent,
    Score,
    SoundEvent,
    bind_instrument,
    render_score,
)


def _const_tone(*, value: float, duration: float, sample_rate: int = 1000) -> Audio:
    """Test instrument: returns a flat-amplitude audio of the given duration."""
    return Audio(
        np.full(int(duration * sample_rate), value, dtype=np.float32),
        sample_rate=sample_rate,
    )


def _stereo_tone(*, value: float, duration: float, sample_rate: int = 1000) -> Audio:
    return Audio(
        np.full((int(duration * sample_rate), 2), value, dtype=np.float32),
        sample_rate=sample_rate,
    )


class TestSoundEvent(unittest.TestCase):
    def test_construction_positional_and_keyword(self):
        a = SoundEvent(0.5, {"pitch": 60})
        b = SoundEvent(beat=0.5, kwargs={"pitch": 60})
        self.assertEqual(a, b)
        self.assertEqual(a.beat, 0.5)
        self.assertEqual(a.kwargs, {"pitch": 60})

    def test_unpacks_like_tuple(self):
        event = SoundEvent(0.5, {"pitch": 60})
        beat, kwargs = event
        self.assertEqual(beat, 0.5)
        self.assertEqual(kwargs, {"pitch": 60})

    def test_plain_tuple_is_not_isinstance(self):
        # NamedTuple identity is real: a plain tuple is not a SoundEvent.
        self.assertNotIsInstance((0.5, {"pitch": 60}), SoundEvent)

    def test_playable_construction(self):
        p = PlayableSoundEvent(0.5, _const_tone, {"value": 0.1, "duration": 0.01})
        self.assertEqual(p.beat, 0.5)
        self.assertIs(p.instrument, _const_tone)
        self.assertEqual(p.kwargs, {"value": 0.1, "duration": 0.01})


class TestBindInstrument(unittest.TestCase):
    def test_binds_instrument_to_every_event(self):
        score: Score = [
            SoundEvent(0.0, {"value": 0.1, "duration": 0.01}),
            SoundEvent(0.5, {"value": 0.2, "duration": 0.01}),
        ]
        playable = bind_instrument(score, _const_tone)
        self.assertEqual(len(playable), 2)
        for orig, new in zip(score, playable):
            self.assertIsInstance(new, PlayableSoundEvent)
            self.assertEqual(new.beat, orig.beat)
            self.assertEqual(new.kwargs, orig.kwargs)
            self.assertIs(new.instrument, _const_tone)

    def test_empty_score(self):
        self.assertEqual(bind_instrument([], _const_tone), [])


class TestBasicMetronome(unittest.TestCase):
    def test_basic(self):
        m = BasicMetronome(120)
        self.assertEqual(m.bpm, 120)
        self.assertEqual(m.beat_duration, 0.5)
        self.assertEqual(m.beat_to_time(1.0), 0.5)
        self.assertEqual(m.time_to_beat(0.5), 1.0)

    def test_60_bpm(self):
        m = BasicMetronome(60)
        self.assertEqual(m.beat_duration, 1.0)
        self.assertEqual(m.beat_to_time(1.0), 1.0)
        self.assertEqual(m.time_to_beat(1.0), 1.0)

    def test_round_trip(self):
        m = BasicMetronome(140)
        for beat in [0.0, 1.0, 2.5, 17.3]:
            self.assertAlmostEqual(m.time_to_beat(m.beat_to_time(beat)), beat)

    def test_subclassable(self):
        # Sanity: Metronome is a usable ABC.
        class Halving(Metronome):
            def beat_to_time(self, beat: float) -> float:
                return beat / 2

            def time_to_beat(self, time: float) -> float:
                return time * 2

        h = Halving()
        self.assertEqual(h.beat_to_time(4.0), 2.0)
        self.assertEqual(h.time_to_beat(2.0), 4.0)


class TestRenderScoreSingleEvent(unittest.TestCase):
    def test_single_event_no_metronome(self):
        # Onset at 0.0s, 10-sample tone of value 0.5.
        score = [PlayableSoundEvent(0.0, _const_tone, {"value": 0.5, "duration": 0.01})]
        audio = render_score(score)
        self.assertEqual(audio.sample_rate, 1000)
        self.assertEqual(audio.num_channels, 1)
        self.assertEqual(audio.num_samples, 10)
        self.assertTrue(np.allclose(audio.samples, 0.5))

    def test_single_event_with_offset(self):
        # Onset at 0.005s = sample 5 (sr=1000), 10-sample tone.
        score = [
            PlayableSoundEvent(0.005, _const_tone, {"value": 0.5, "duration": 0.01})
        ]
        audio = render_score(score)
        # Output must extend to at least sample 15.
        self.assertGreaterEqual(audio.num_samples, 15)
        # First 5 samples are silent, samples 5..15 are 0.5.
        self.assertTrue(np.all(audio.samples[:5] == 0.0))
        self.assertTrue(np.all(audio.samples[5:15] == 0.5))


class TestRenderScoreMultipleEvents(unittest.TestCase):
    def test_non_overlapping(self):
        # Two tones, 10 samples each, at 0.0s and 0.01s.
        score = [
            PlayableSoundEvent(0.0, _const_tone, {"value": 0.3, "duration": 0.01}),
            PlayableSoundEvent(0.01, _const_tone, {"value": 0.7, "duration": 0.01}),
        ]
        audio = render_score(score)
        self.assertEqual(audio.num_samples, 20)
        self.assertTrue(np.allclose(audio.samples[:10], 0.3))
        self.assertTrue(np.allclose(audio.samples[10:20], 0.7))

    def test_overlapping_events_mix_additively(self):
        # Two tones at the same onset add together.
        score = [
            PlayableSoundEvent(0.0, _const_tone, {"value": 0.2, "duration": 0.01}),
            PlayableSoundEvent(0.0, _const_tone, {"value": 0.3, "duration": 0.01}),
        ]
        audio = render_score(score)
        self.assertEqual(audio.num_samples, 10)
        self.assertTrue(np.allclose(audio.samples, 0.5))


class TestRenderScoreWithMetronome(unittest.TestCase):
    def test_beats_get_converted_to_time(self):
        # 60 BPM → 1 beat = 1 second. Event at beat 0.005 → time 0.005s.
        m = BasicMetronome(60_000)  # 60_000 BPM → 1 beat = 0.001s = 1 sample
        score = [PlayableSoundEvent(5.0, _const_tone, {"value": 0.5, "duration": 0.01})]
        audio = render_score(score, metronome=m)
        # Onset converted to 5 * 0.001 = 0.005s = sample 5.
        self.assertTrue(np.all(audio.samples[:5] == 0.0))
        self.assertTrue(np.all(audio.samples[5:15] == 0.5))


class TestRenderScoreErrors(unittest.TestCase):
    def test_inconsistent_sample_rates_raises(self):
        score = [
            PlayableSoundEvent(
                0.0, _const_tone, {"value": 0.5, "duration": 0.01, "sample_rate": 1000}
            ),
            PlayableSoundEvent(
                0.01,
                _const_tone,
                {"value": 0.5, "duration": 0.01, "sample_rate": 2000},
            ),
        ]
        with self.assertRaises(ValueError) as ctx:
            render_score(score)
        self.assertIn("sample rate", str(ctx.exception).lower())

    def test_inconsistent_channel_counts_raises(self):
        score = [
            PlayableSoundEvent(0.0, _const_tone, {"value": 0.5, "duration": 0.01}),
            PlayableSoundEvent(0.01, _stereo_tone, {"value": 0.5, "duration": 0.01}),
        ]
        with self.assertRaises(ValueError) as ctx:
            render_score(score)
        self.assertIn("channel", str(ctx.exception).lower())

    def test_missing_sample_rate_raises(self):
        def _no_sr_instrument(**kwargs):
            return Audio(np.zeros(10, dtype=np.float32))  # no sample_rate

        score = [PlayableSoundEvent(0.0, _no_sr_instrument, {})]
        with self.assertRaises(ValueError):
            render_score(score)


class TestRenderScoreEmpty(unittest.TestCase):
    def test_empty_score_returns_zero_length_audio(self):
        audio = render_score([])
        self.assertEqual(audio.num_samples, 0)


if __name__ == "__main__":
    unittest.main()
