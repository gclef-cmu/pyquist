import unittest

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
        self.assertEqual(resampled.num_samples, audio.num_samples // 2)
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

        # Negative offset clamps to 0
        seg = audio.segment(offset=-1.0, duration=0.1)
        self.assertEqual(seg.shape, (100, 1))
        self.assertEqual(seg.samples[0, 0], 0.0)

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

    def test_indexing_and_len(self):
        audio = Audio(np.arange(20, dtype=np.float32).reshape(10, 2), sample_rate=44100)
        self.assertEqual(len(audio), 10)

        # Read returns ndarray view
        sliced = audio[:5]
        self.assertIsInstance(sliced, np.ndarray)
        self.assertEqual(sliced.shape, (5, 2))

        # Single-channel slice is 1D ndarray
        ch0 = audio[:, 0]
        self.assertIsInstance(ch0, np.ndarray)
        self.assertEqual(ch0.shape, (10,))

        # Setitem works
        audio[0:2, :] = 99.0
        self.assertTrue(np.all(audio.samples[0:2, :] == 99.0))

        # In-place ops on slices flow through to underlying samples
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
        # underlying array — important for the realtime callback pattern.
        backing = np.zeros((512, 2), dtype=np.float32)
        buffer = Audio(backing)
        self.assertIs(buffer.samples, backing)
        buffer[:] = 1.0
        self.assertTrue(np.all(backing == 1.0))


if __name__ == "__main__":
    unittest.main()
