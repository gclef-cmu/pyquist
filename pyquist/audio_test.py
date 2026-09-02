import pathlib
import tempfile
import unittest
from io import BytesIO

import numpy as np

from .audio import Audio
from .paths import TEST_DATA_DIR

# Bundled freesound clips named by their numeric freesound ID for traceability.
_BLUES_RIFF_WAV = TEST_DATA_DIR / "388954__fullmetaljedi__blues-riff-in-g-nylon.wav"
_DRUM_PATTERN_MP3 = TEST_DATA_DIR / "434013__mrpearch__drum-patern.mp3"


class TestAudio(unittest.TestCase):
    def test_construction_and_setter_normalization(self):
        # 2D float32 — pass-through
        arr = np.zeros((10, 2), dtype=np.float32)
        audio = Audio(arr, sample_rate=44100)
        self.assertIs(audio.samples, arr)
        self.assertEqual(audio.shape, (10, 2))
        self.assertEqual(audio.num_samples, 10)
        self.assertEqual(audio.num_channels, 2)
        self.assertEqual(audio.sample_rate, 44100)

        # 1D float32 -> (n, 1)
        audio = Audio(np.zeros(1000, dtype=np.float32), sample_rate=44100)
        self.assertEqual(audio.shape, (1000, 1))

        # 0D float32 -> (1, 1)
        audio = Audio(np.array(5.0, dtype=np.float32), sample_rate=44100)
        self.assertEqual(audio.shape, (1, 1))

        # float64 auto-converted to float32
        audio = Audio(np.zeros((10, 2), dtype=np.float64), sample_rate=44100)
        self.assertEqual(audio.samples.dtype, np.float32)

        # No sample rate is fine
        audio = Audio(np.zeros((10, 2), dtype=np.float32))
        self.assertIsNone(audio.sample_rate)

        # Reassigning samples re-runs validation
        audio.samples = np.ones(5, dtype=np.float32)
        self.assertEqual(audio.shape, (5, 1))

        # Reassigning sample_rate re-runs validation
        audio.sample_rate = 22050
        self.assertEqual(audio.sample_rate, 22050)
        audio.sample_rate = None
        self.assertIsNone(audio.sample_rate)

    def test_invalid_construction(self):
        # Non-ndarray
        with self.assertRaises(TypeError):
            Audio([0.0, 0.1, 0.2])  # type: ignore[arg-type]

        # Bad dtype
        with self.assertRaises(TypeError):
            Audio(np.zeros((10, 2), dtype=np.int16))

        # Too many dimensions
        with self.assertRaises(ValueError):
            Audio(np.zeros((1, 1, 1), dtype=np.float32))

        # Bad sample_rate
        with self.assertRaises(ValueError):
            Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=-1)
        with self.assertRaises(ValueError):
            Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=0)
        with self.assertRaises(TypeError):
            Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=44100.0)  # type: ignore[arg-type]

    def test_empty_classmethod(self):
        audio = Audio.empty(10, 2, sample_rate=44100)
        self.assertEqual(audio.shape, (10, 2))
        self.assertEqual(audio.samples.dtype, np.float32)
        self.assertEqual(audio.sample_rate, 44100)

        audio = Audio.empty(10, 2)
        self.assertIsNone(audio.sample_rate)

        with self.assertRaises(ValueError):
            Audio.empty(-1, 2)
        with self.assertRaises(ValueError):
            Audio.empty(10, -1)

    def test_zeros_classmethod(self):
        audio = Audio.zeros(10, 2, sample_rate=44100)
        self.assertEqual(audio.shape, (10, 2))
        self.assertEqual(audio.samples.dtype, np.float32)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertTrue(np.all(audio.samples == 0.0))

        audio = Audio.zeros(10, 2)
        self.assertIsNone(audio.sample_rate)

        with self.assertRaises(ValueError):
            Audio.zeros(-1, 2)
        with self.assertRaises(ValueError):
            Audio.zeros(10, -1)

    def test_concatenate_classmethod(self):
        a = Audio(np.ones(10), sample_rate=44100)
        b = Audio(np.zeros(5), sample_rate=44100)
        joined = Audio.concatenate([a, b])
        self.assertEqual(joined.shape, (15, 1))
        self.assertEqual(joined.sample_rate, 44100)
        self.assertTrue(np.all(joined.samples[:10] == 1.0))
        self.assertTrue(np.all(joined.samples[10:] == 0.0))

        # Stereo concatenation preserves channel count.
        s1 = Audio(np.ones((4, 2)), sample_rate=22050)
        s2 = Audio(np.zeros((3, 2)), sample_rate=22050)
        self.assertEqual(Audio.concatenate([s1, s2]).shape, (7, 2))

        with self.assertRaises(ValueError):
            Audio.concatenate([])
        with self.assertRaises(ValueError):
            Audio.concatenate(
                [
                    Audio(np.ones(3), sample_rate=44100),
                    Audio(np.ones(3), sample_rate=22050),
                ]
            )
        with self.assertRaises(ValueError):
            Audio.concatenate(
                [
                    Audio(np.ones((3, 1)), sample_rate=44100),
                    Audio(np.ones((3, 2)), sample_rate=44100),
                ]
            )

    def test_from_file_wav(self):
        # 388954__fullmetaljedi__blues-riff-in-g-nylon.wav: stereo, 48 kHz.
        audio = Audio.from_file(_BLUES_RIFF_WAV)
        self.assertEqual(audio.num_channels, 2)
        self.assertEqual(audio.num_samples, 216873)
        self.assertEqual(audio.sample_rate, 48000)
        self.assertEqual(audio.shape, (216873, 2))
        self.assertAlmostEqual(audio.peak_amplitude, 0.950, places=3)
        self.assertAlmostEqual(audio.duration, 4.518, places=3)

    def test_from_file_mp3(self):
        # 434013__mrpearch__drum-patern.mp3: stereo, 44.1 kHz.
        # MP3 frame counts can vary slightly between decoder versions, so the
        # length assertion is loose.
        audio = Audio.from_file(_DRUM_PATTERN_MP3)
        self.assertEqual(audio.num_channels, 2)
        self.assertEqual(audio.sample_rate, 44100)
        self.assertAlmostEqual(audio.duration, 36.0, places=1)
        self.assertAlmostEqual(audio.peak_amplitude, 0.328, places=3)

    def test_from_file_missing_raises_friendly_error(self):
        # Without our pre-check, soundfile / libsndfile raises an opaque
        # ``LibsndfileError: "System error"`` here. We promote it to a clean
        # FileNotFoundError that includes the missing path.
        missing = TEST_DATA_DIR / "definitely_not_a_real_file.wav"
        with self.assertRaises(FileNotFoundError) as ctx:
            Audio.from_file(missing)
        self.assertIn(str(missing), str(ctx.exception))

    def test_from_file_accepts_file_handle(self):
        # File-like inputs skip the path pre-check (we can't sensibly probe a
        # buffer for "existence"). Confirm the happy path still works.
        with open(_BLUES_RIFF_WAV, "rb") as f:
            audio = Audio.from_file(f)
        self.assertEqual(audio.num_channels, 2)
        self.assertEqual(audio.sample_rate, 48000)

    def test_normalize(self):
        audio = Audio.from_file(_BLUES_RIFF_WAV)
        # peak_dbfs=0 (default) → 1.0; -6 → ~0.501; +6 → ~1.995. These
        # values don't depend on the input peak (they're absolute targets).
        audio_norm = audio.normalize(in_place=False)
        self.assertAlmostEqual(audio_norm.peak_amplitude, 1.0, places=3)
        audio_norm = audio.normalize(peak_dbfs=-6.0, in_place=False)
        self.assertAlmostEqual(audio_norm.peak_amplitude, 0.501, places=3)
        audio_norm = audio.normalize(peak_dbfs=6.0, in_place=False)
        self.assertAlmostEqual(audio_norm.peak_amplitude, 1.995, places=3)
        # in_place=False above doesn't mutate the source.
        self.assertAlmostEqual(audio.peak_amplitude, 0.950, places=3)
        audio.normalize()
        self.assertAlmostEqual(audio.peak_amplitude, 1.0, places=3)

        # Silent audio normalizes to silence
        silent = Audio.zeros(10, 1, sample_rate=44100)
        silent.normalize()
        self.assertEqual(silent.peak_amplitude, 0.0)

    def test_clip(self):
        audio = Audio.from_file(_BLUES_RIFF_WAV)
        audio.normalize()
        audio_clipped = audio.clip(in_place=False)
        self.assertAlmostEqual(audio_clipped.peak_amplitude, 1.0, places=3)
        audio_clipped = audio.clip(peak_amplitude=0.5, in_place=False)
        self.assertAlmostEqual(audio_clipped.peak_amplitude, 0.5, places=3)
        self.assertAlmostEqual(audio.peak_amplitude, 1.0, places=3)
        audio.clip(peak_amplitude=0.25)
        self.assertAlmostEqual(audio.peak_amplitude, 0.25, places=3)

    def test_resample(self):
        # Downsample 48 kHz → 24 kHz (clean 2× ratio).
        audio = Audio.from_file(_BLUES_RIFF_WAV)
        resampled = audio.resample(24000)
        self.assertEqual(resampled.num_channels, 2)
        # Output length follows new_rate/old_rate; allow ±1 for the
        # resampler's rounding of a non-integer result.
        self.assertAlmostEqual(resampled.num_samples, audio.num_samples / 2, delta=1)
        self.assertEqual(resampled.sample_rate, 24000)
        self.assertAlmostEqual(resampled.duration, audio.duration, places=3)

        with self.assertRaises(ValueError):
            audio.resample(-1)
        with self.assertRaises(ValueError):
            Audio(np.zeros((10, 1), dtype=np.float32)).resample(44100)

    def test_as_mono(self):
        # Already mono: returns self
        mono = Audio(np.ones((10, 1), dtype=np.float32), sample_rate=44100)
        self.assertIs(mono.as_mono(), mono)

        # Stereo: averages channels
        stereo_samples = np.stack(
            [np.full(10, 0.2, dtype=np.float32), np.full(10, 0.6, dtype=np.float32)],
            axis=1,
        )
        stereo = Audio(stereo_samples, sample_rate=44100)
        result = stereo.as_mono()
        self.assertEqual(result.shape, (10, 1))
        self.assertEqual(result.samples.dtype, np.float32)
        self.assertEqual(result.sample_rate, 44100)
        self.assertTrue(np.allclose(result.samples[:, 0], 0.4))

        # Multi-channel: averages all channels
        multi = Audio(np.full((4, 4), 1.0, dtype=np.float32), sample_rate=44100)
        result = multi.as_mono()
        self.assertEqual(result.shape, (4, 1))
        self.assertTrue(np.allclose(result.samples, 1.0))

        # Original unchanged
        self.assertEqual(stereo.shape, (10, 2))

    def test_as_stereo(self):
        # Already stereo: returns self
        stereo = Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=44100)
        self.assertIs(stereo.as_stereo(), stereo)

        # Mono: duplicates channel
        mono = Audio(np.arange(10, dtype=np.float32).reshape(10, 1), sample_rate=44100)
        result = mono.as_stereo()
        self.assertEqual(result.shape, (10, 2))
        self.assertEqual(result.samples.dtype, np.float32)
        self.assertEqual(result.sample_rate, 44100)
        self.assertTrue(np.array_equal(result.samples[:, 0], result.samples[:, 1]))
        self.assertTrue(
            np.array_equal(result.samples[:, 0], np.arange(10, dtype=np.float32))
        )

        # 3+ channels: raises
        multi = Audio(np.zeros((10, 3), dtype=np.float32), sample_rate=44100)
        with self.assertRaises(ValueError):
            multi.as_stereo()
        multi4 = Audio(np.zeros((10, 4), dtype=np.float32), sample_rate=44100)
        with self.assertRaises(ValueError):
            multi4.as_stereo()

    def test_pan(self):
        stereo = Audio(np.ones((10, 2), dtype=np.float32), sample_rate=44100)

        # Center: equal gain of -3 dB in both channels.
        centered = stereo.pan(in_place=False)
        self.assertEqual(centered.shape, (10, 2))
        self.assertEqual(centered.samples.dtype, np.float32)
        self.assertEqual(centered.sample_rate, 44100)
        self.assertTrue(np.allclose(centered.samples, np.sqrt(0.5), atol=1e-6))

        # Hard left/right mute the opposite channel and pass the other through.
        left = stereo.pan(-1.0, in_place=False)
        self.assertTrue(np.allclose(left.samples[:, 0], 1.0, atol=1e-6))
        self.assertTrue(np.allclose(left.samples[:, 1], 0.0, atol=1e-6))
        right = stereo.pan(1.0, in_place=False)
        self.assertTrue(np.allclose(right.samples[:, 0], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(right.samples[:, 1], 1.0, atol=1e-6))

        # Equal power: the summed power is constant across positions.
        for position in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
            panned = stereo.pan(position, in_place=False)
            self.assertAlmostEqual(
                float((panned.samples**2).sum()),
                float((stereo.samples**2).sum()) / 2,
                places=5,
            )

        # in_place=False above didn't mutate the source; in_place=True does.
        self.assertTrue(np.allclose(stereo.samples, 1.0))
        self.assertIs(stereo.pan(1.0), stereo)
        self.assertTrue(np.allclose(stereo.samples[:, 0], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(stereo.samples[:, 1], 1.0, atol=1e-6))

        # Mono is widened first: as_stereo().pan() is a full-amplitude hard pan.
        mono = Audio(np.ones((10, 1), dtype=np.float32), sample_rate=44100)
        panned = mono.as_stereo().pan(1.0)
        self.assertTrue(np.allclose(panned.samples[:, 0], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(panned.samples[:, 1], 1.0, atol=1e-6))

        # Out-of-range positions raise.
        with self.assertRaises(ValueError):
            stereo.pan(-1.5)
        with self.assertRaises(ValueError):
            stereo.pan(1.5)

        # Non-stereo: raises
        with self.assertRaises(ValueError):
            mono.pan()
        multi = Audio(np.zeros((10, 3), dtype=np.float32), sample_rate=44100)
        with self.assertRaises(ValueError):
            multi.pan()

    def test_segment(self):
        sr = 1000
        # 1s = 1000 samples; values 0..999 in channel 0
        audio = Audio(np.arange(sr, dtype=np.float32), sample_rate=sr)

        # No args → returns self (identity, no copy)
        self.assertIs(audio.segment(), audio)

        # offset only
        seg = audio.segment(offset=0.1)
        self.assertEqual(seg.shape, (900, 1))
        self.assertEqual(seg.sample_rate, sr)
        self.assertEqual(seg.samples[0, 0], 100.0)
        self.assertEqual(seg.samples[-1, 0], 999.0)

        # duration only
        seg = audio.segment(duration=0.2)
        self.assertEqual(seg.shape, (200, 1))
        self.assertEqual(seg.samples[0, 0], 0.0)
        self.assertEqual(seg.samples[-1, 0], 199.0)

        # offset + duration
        seg = audio.segment(offset=0.25, duration=0.5)
        self.assertEqual(seg.shape, (500, 1))
        self.assertEqual(seg.samples[0, 0], 250.0)
        self.assertEqual(seg.samples[-1, 0], 749.0)

        # Negative offset or duration is an error, not a clamp.
        with self.assertRaises(ValueError) as ctx:
            audio.segment(offset=-1.0, duration=0.1)
        self.assertIn("offset", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            audio.segment(duration=-0.1)
        self.assertIn("duration", str(ctx.exception))

        # Int arguments are seconds, same as floats.
        self.assertEqual(audio.segment(duration=1).shape, (1000, 1))
        self.assertEqual(audio.segment(offset=1).shape, (0, 1))

        # Duration past end is truncated
        seg = audio.segment(offset=0.8, duration=10.0)
        self.assertEqual(seg.shape, (200, 1))
        self.assertEqual(seg.samples[-1, 0], 999.0)

        # Offset past end gives an empty segment, no crash
        seg = audio.segment(offset=10.0)
        self.assertEqual(seg.shape, (0, 1))

        # Requires sample_rate when offset/duration given
        no_sr = Audio(np.zeros(100, dtype=np.float32))
        self.assertIs(no_sr.segment(), no_sr)
        with self.assertRaises(ValueError):
            no_sr.segment(offset=0.1)

    def test_clear(self):
        audio = Audio(np.ones((10, 2), dtype=np.float32), sample_rate=44100)
        self.assertEqual(audio.peak_amplitude, 1.0)
        audio.clear()
        self.assertEqual(audio.peak_amplitude, 0.0)

    def test_indexing_returns_audio_for_valid_patterns(self):
        audio = Audio(np.arange(20, dtype=np.float32).reshape(10, 2), sample_rate=44100)
        self.assertEqual(len(audio), 10)

        # Sample slice: 2-D result, both axes preserved.
        sliced = audio[1:5]
        self.assertIsInstance(sliced, Audio)
        self.assertEqual(sliced.shape, (4, 2))
        self.assertEqual(sliced.sample_rate, 44100)

        # Single-channel slice via int on axis 1: Audio shaped (n, 1).
        ch0 = audio[:, 0]
        self.assertIsInstance(ch0, Audio)
        self.assertEqual(ch0.shape, (10, 1))

        # Both axes sliced.
        both = audio[2:6, 0:1]
        self.assertIsInstance(both, Audio)
        self.assertEqual(both.shape, (4, 1))

        # Length-1 slice is the way to spell "single sample as Audio".
        one = audio[3:4]
        self.assertIsInstance(one, Audio)
        self.assertEqual(one.shape, (1, 2))
        self.assertTrue(np.array_equal(one.samples, [[6.0, 7.0]]))

    def test_indexing_axis_0_with_int_raises(self):
        audio = Audio(np.arange(20, dtype=np.float32).reshape(10, 2), sample_rate=44100)
        # Bare int.
        with self.assertRaises(TypeError) as ctx:
            audio[3]
        self.assertIn("audio.samples", str(ctx.exception))
        # Negative bare int.
        with self.assertRaises(TypeError):
            audio[-1]
        # Two-int tuple (scalar read).
        with self.assertRaises(TypeError):
            audio[0, 0]
        # Int on axis 0, slice on axis 1.
        with self.assertRaises(TypeError):
            audio[3, :]
        # The recommended raw-numpy access still works.
        self.assertEqual(audio.samples[0, 0], 0.0)
        self.assertEqual(audio.samples[3, 1], 7.0)

    def test_indexing_returns_view_when_possible(self):
        # Basic slicing should return a view of the underlying samples, so
        # mutating the returned Audio's samples writes through to the parent.
        audio = Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=44100)
        sliced = audio[2:5]
        sliced.samples[:] = 1.0
        self.assertTrue(np.all(audio.samples[2:5, :] == 1.0))
        self.assertTrue(np.all(audio.samples[0:2, :] == 0.0))

    def test_indexing_by_seconds(self):
        sr = 100
        audio = Audio(np.arange(500, dtype=np.float32), sample_rate=sr)

        # Floats are seconds: audio[1.0:3.0] == samples 100 through 299.
        by_time = audio[1.0:3.0]
        self.assertIsInstance(by_time, Audio)
        self.assertEqual(by_time.shape, (200, 1))
        self.assertEqual(by_time.sample_rate, sr)
        self.assertTrue(np.array_equal(by_time.samples, audio.samples[100:300]))

        # Ints stay sample numbers.
        self.assertEqual(audio[1:3].shape, (2, 1))

        # Open-ended and negative bounds behave like numpy on the converted
        # sample indices.
        self.assertTrue(np.array_equal(audio[:2.0].samples, audio.samples[:200]))
        self.assertTrue(np.array_equal(audio[4.5:].samples, audio.samples[450:]))
        self.assertTrue(np.array_equal(audio[-1.0:].samples, audio.samples[-100:]))

        # Truncation toward zero, as int(seconds * sample_rate).
        self.assertEqual(audio[0.0:1.999].shape, (199, 1))

        # numpy floats work too.
        self.assertEqual(audio[np.float32(1.0) : np.float32(2.0)].shape, (100, 1))

        # Time slice on axis 0, channel index on axis 1.
        stereo = Audio(np.zeros((500, 2), dtype=np.float32), sample_rate=sr)
        self.assertEqual(stereo[1.0:2.0, 0].shape, (100, 1))

        # Still a view of the parent.
        stereo[1.0:2.0].samples[:] = 1.0
        self.assertTrue(np.all(stereo.samples[100:200] == 1.0))

    def test_indexing_by_seconds_with_channel_axis(self):
        sr = 100
        stereo = Audio(
            np.arange(1000, dtype=np.float32).reshape(500, 2), sample_rate=sr
        )

        # Seconds on the sample axis mix freely with ints on the channel axis.
        for key in [np.s_[1.0:2.0, :1], np.s_[1.0:2.0, 0], np.s_[1.0:2.0, 1:]]:
            seg = stereo[key]
            self.assertIsInstance(seg, Audio)
            self.assertEqual(seg.shape, (100, 1))
            self.assertEqual(seg.sample_rate, sr)
        self.assertTrue(
            np.array_equal(stereo[1.0:2.0, :1].samples, stereo.samples[100:200, :1])
        )
        self.assertTrue(
            np.array_equal(stereo[1.0:2.0, 1].samples, stereo.samples[100:200, 1:2])
        )
        # Both channels, and an open-ended time bound.
        self.assertEqual(stereo[1.0:2.0, :].shape, (100, 2))
        self.assertEqual(stereo[4.0:, :1].shape, (100, 1))
        # Writes take the same mixed key.
        stereo[1.0:2.0, :1] = 0.0
        self.assertTrue(np.all(stereo.samples[100:200, 0] == 0.0))
        self.assertTrue(np.all(stereo.samples[100:200, 1] != 0.0))

    def test_indexing_by_seconds_with_int_channel(self):
        # An int on the channel axis picks one channel and keeps it as a
        # channel, so the result is Audio shaped (n, 1) rather than 1-D.
        sr = 100
        audio = Audio(np.arange(2000, dtype=np.float32).reshape(500, 4), sample_rate=sr)

        for channel in range(audio.num_channels):
            seg = audio[1.0:2.0, channel]
            self.assertIsInstance(seg, Audio)
            self.assertEqual(seg.shape, (100, 1))
            self.assertEqual(seg.sample_rate, sr)
            self.assertTrue(
                np.array_equal(
                    seg.samples, audio.samples[100:200, channel : channel + 1]
                )
            )

        # Negative channel indices count from the last channel.
        self.assertTrue(
            np.array_equal(audio[1.0:2.0, -1].samples, audio[1.0:2.0, 3].samples)
        )

        # Open-ended and negative time bounds combine with an int channel.
        self.assertEqual(audio[:1.0, 2].shape, (100, 1))
        self.assertEqual(audio[-1.0:, 1].shape, (100, 1))
        self.assertTrue(
            np.array_equal(audio[-1.0:, 1].samples, audio.samples[-100:, 1:2])
        )

        # Ints on both axes are unchanged: samples and a channel.
        self.assertEqual(audio[100:200, 0].shape, (100, 1))

        # Same span as the equivalent segment() call.
        self.assertTrue(
            np.array_equal(
                audio[1.0:2.0, 0].samples,
                audio.segment(offset=1.0, duration=1.0)[:, 0].samples,
            )
        )

        # Still a view: writing through the result reaches the parent, and
        # only in the indexed channel and time range.
        audio[1.0:2.0, 0].samples[:] = -1.0
        self.assertTrue(np.all(audio.samples[100:200, 0] == -1.0))
        self.assertTrue(np.all(audio.samples[100:200, 1:] != -1.0))
        self.assertTrue(np.all(audio.samples[:100, 0] != -1.0))

        # Writes take an int channel directly, too.
        audio[2.0:3.0, 2] = 0.0
        self.assertTrue(np.all(audio.samples[200:300, 2] == 0.0))
        self.assertTrue(np.all(audio.samples[200:300, 3] != 0.0))
        self.assertTrue(np.all(audio.samples[300:, 2] != 0.0))

    def test_indexing_channel_axis_rejects_seconds(self):
        stereo = Audio(np.zeros((500, 2), dtype=np.float32), sample_rate=100)
        # Only the sample axis is converted; the channel axis goes to numpy,
        # which takes ints alone and rejects a float itself.
        for key in [np.s_[:, 0.0], np.s_[:, 0.0:1.0], np.s_[1.0:2.0, 0.0:1.0]]:
            with self.assertRaises((TypeError, IndexError)):
                stereo[key]
        with self.assertRaises((TypeError, IndexError)):
            stereo[1.0:2.0, 0.0] = 0.0

    def test_indexing_by_seconds_invalid(self):
        audio = Audio(np.zeros((500, 1), dtype=np.float32), sample_rate=100)

        # A bare float is a single index, which is rejected like a bare int.
        with self.assertRaises(TypeError):
            audio[1.0]
        with self.assertRaises(TypeError):
            audio[1.0, 0]

        # Units can't be mixed within one slice.
        with self.assertRaises(TypeError) as ctx:
            audio[100:2.0]
        self.assertIn("mix", str(ctx.exception))
        with self.assertRaises(TypeError):
            audio[1.0:200]

        # A step is always a stride in samples, whatever the bounds are.
        self.assertEqual(audio[1.0:2.0:2].shape, (50, 1))
        self.assertEqual(audio[100:200:2].shape, (50, 1))
        # A float step is numpy's to reject.
        with self.assertRaises(TypeError):
            audio[100:200:0.5]

    def test_indexing_by_seconds_requires_sample_rate(self):
        audio = Audio(np.zeros((500, 1), dtype=np.float32))
        with self.assertRaises(ValueError) as ctx:
            audio[1.0:2.0]
        self.assertIn("sample_rate", str(ctx.exception))
        with self.assertRaises(ValueError):
            audio[1.0:2.0] = 0.0
        # Sample indexing still works without a sample rate.
        self.assertEqual(audio[100:200].shape, (100, 1))

    def test_setitem_by_seconds(self):
        sr = 100
        audio = Audio(np.ones((500, 1), dtype=np.float32), sample_rate=sr)
        audio[1.0:2.0] = 0.0
        self.assertTrue(np.all(audio.samples[100:200] == 0.0))
        self.assertTrue(np.all(audio.samples[:100] == 1.0))
        self.assertTrue(np.all(audio.samples[200:] == 1.0))
        # In-place ops on a time slice flow through to the samples.
        audio[3.0:4.0] *= 0.5
        self.assertTrue(np.all(audio.samples[300:400] == 0.5))
        # Writes may collapse the sample axis, so a scalar time is allowed.
        audio[4.0] = 7.0
        self.assertEqual(audio.samples[400, 0], 7.0)

    def test_setitem_and_inplace_ops(self):
        # __setitem__ is unrestricted (writes don't have the dim-collapse
        # ambiguity that reads do).
        audio = Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=44100)
        audio[0:2, :] = 99.0
        self.assertTrue(np.all(audio.samples[0:2, :] == 99.0))
        # In-place ops on slices flow through to underlying samples.
        audio[2:4, :] = 2.0
        audio[2:4, :] *= 0.5
        self.assertTrue(np.all(audio.samples[2:4, :] == 1.0))

    def test_arithmetic_with_scalars(self):
        audio = Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=44100)
        result = audio + 0.1
        self.assertIsInstance(result, Audio)
        self.assertEqual(result.sample_rate, 44100)
        self.assertTrue(np.allclose(result.samples, 0.1))
        self.assertTrue(np.all(audio.samples == 0.0))  # original unchanged

        result = 1.0 + audio
        self.assertIsInstance(result, Audio)
        self.assertTrue(np.allclose(result.samples, 1.0))

        result = audio - 0.5
        self.assertTrue(np.allclose(result.samples, -0.5))
        result = 1.0 - audio
        self.assertTrue(np.allclose(result.samples, 1.0))

        result = (audio + 1.0) * 2.0
        self.assertTrue(np.allclose(result.samples, 2.0))
        result = 3.0 * (audio + 1.0)
        self.assertTrue(np.allclose(result.samples, 3.0))

        result = (audio + 4.0) / 2.0
        self.assertTrue(np.allclose(result.samples, 2.0))

        result = -(audio + 1.0)
        self.assertTrue(np.allclose(result.samples, -1.0))

    def test_arithmetic_in_place(self):
        audio = Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=44100)
        audio += 0.5
        self.assertTrue(np.allclose(audio.samples, 0.5))
        audio -= 0.25
        self.assertTrue(np.allclose(audio.samples, 0.25))
        audio *= 4.0
        self.assertTrue(np.allclose(audio.samples, 1.0))
        audio /= 2.0
        self.assertTrue(np.allclose(audio.samples, 0.5))

    def test_arithmetic_audio_audio(self):
        a = Audio(np.full((10, 2), 0.25, dtype=np.float32), sample_rate=44100)
        b = Audio(np.full((10, 2), 0.5, dtype=np.float32), sample_rate=44100)
        c = a + b
        self.assertIsInstance(c, Audio)
        self.assertEqual(c.sample_rate, 44100)
        self.assertTrue(np.allclose(c.samples, 0.75))

        c = a * b
        self.assertTrue(np.allclose(c.samples, 0.125))

        # In-place
        a += b
        self.assertTrue(np.allclose(a.samples, 0.75))

    def test_shape_mismatch(self):
        # Strictly incompatible shapes
        a = Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=44100)
        b = Audio(np.zeros((20, 2), dtype=np.float32), sample_rate=44100)
        with self.assertRaises(ValueError):
            a + b
        with self.assertRaises(ValueError):
            a * b
        with self.assertRaises(ValueError):
            a += b

        # Broadcastable but different shapes (mono + stereo) — should still fail
        mono = Audio(np.zeros((10, 1), dtype=np.float32), sample_rate=44100)
        with self.assertRaises(ValueError):
            a + mono
        with self.assertRaises(ValueError):
            a * mono
        with self.assertRaises(ValueError):
            a += mono

    def test_sample_rate_compatibility(self):
        a = Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=44100)
        b = Audio(np.zeros((10, 2), dtype=np.float32), sample_rate=48000)
        with self.assertRaises(ValueError):
            a + b
        with self.assertRaises(ValueError):
            a += b

        # If one has no sample_rate, the other's wins
        c = Audio(np.zeros((10, 2), dtype=np.float32))
        d = a + c
        self.assertEqual(d.sample_rate, 44100)
        d = c + a
        self.assertEqual(d.sample_rate, 44100)

    def test_numpy_interop_via_array_protocol(self):
        audio = Audio(np.full((10, 2), 0.5, dtype=np.float32), sample_rate=44100)
        # np.asarray returns the underlying samples
        arr = np.asarray(audio)
        self.assertIs(arr, audio.samples)
        # ufuncs work via __array__
        result = np.sin(audio)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, 2))

    def test_buffer_view_pattern(self):
        # Audio without a sample_rate (buffer-style) keeps a reference to the
        # underlying array (no copy) — useful when wrapping an externally
        # owned buffer such as one a real-time callback writes into.
        backing = np.zeros((512, 2), dtype=np.float32)
        buffer = Audio(backing)
        self.assertIs(buffer.samples, backing)
        buffer[:] = 1.0
        self.assertTrue(np.all(backing == 1.0))

    def test_write_channel_limits(self):
        surround = Audio(np.zeros((100, 6), dtype=np.float32), sample_rate=44100)
        stereo = Audio(np.zeros((100, 2), dtype=np.float32), sample_rate=44100)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)

            # mp3 tops out at 2 channels, flac at 8.
            for path in (tmp / "out.mp3", tmp / "out.MP3"):
                with self.assertRaises(ValueError) as ctx:
                    surround.write(path)
                self.assertIn("MP3 supports at most 2 channels", str(ctx.exception))
                self.assertIn("6", str(ctx.exception))
                self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                Audio(np.zeros((100, 9), dtype=np.float32), sample_rate=44100).write(
                    tmp / "out.flac"
                )

            # Formats without a low limit, and channel counts within a limit,
            # are unaffected.
            surround.write(tmp / "out.wav")
            self.assertEqual(Audio.from_file(tmp / "out.wav").num_channels, 6)
            stereo.write(tmp / "out.mp3")
            self.assertEqual(Audio.from_file(tmp / "out.mp3").num_channels, 2)

        # File-like destinations are checked via the explicit format kwarg.
        with self.assertRaises(ValueError):
            surround.write(BytesIO(), format="MP3")
        surround.write(BytesIO(), format="WAV")


if __name__ == "__main__":
    unittest.main()
