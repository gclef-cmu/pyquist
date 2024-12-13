import unittest

import numpy as np

from .audio import Audio, AudioBuffer
from .paths import TEST_DATA_DIR


class TestAudioBuffer(unittest.TestCase):
    def test_audio_buffer(self):
        # Valid initialization
        AudioBuffer(num_channels=0, num_samples=10)
        AudioBuffer(num_channels=2, num_samples=0)
        AudioBuffer(num_channels=0, num_samples=0)
        buffer = AudioBuffer(num_channels=2, num_samples=10)
        self.assertIsInstance(buffer, AudioBuffer)
        self.assertIsInstance(buffer, np.ndarray)
        self.assertEqual(buffer.num_channels, 2)
        self.assertEqual(buffer.num_samples, 10)
        self.assertEqual(buffer.shape, (10, 2))

        # Slicing
        comparison = np.zeros((10, 2), dtype=np.float32)
        should_be_audio_buffer = [
            buffer[:, :0],
            buffer[:, :1],
            buffer[:, 1:2],
            buffer[:, 1:],
            buffer[:0, :],
            buffer[:1, :],
            buffer[1:2, :],
            buffer[1:, :],
            buffer[1:2, :1],
        ]
        for s in should_be_audio_buffer:
            self.assertIsInstance(s, AudioBuffer)
            self.assertIsInstance(s, np.ndarray)
        should_be_numpy_array = [
            buffer[:, 0],
            buffer[1:, 0],
            buffer[0, :],
            buffer.sum(axis=0),
            buffer.sum(axis=1),
        ]
        for s in should_be_numpy_array:
            self.assertNotIsInstance(s, AudioBuffer)
            self.assertIsInstance(s, np.ndarray)
            self.assertEqual(type(s), type(comparison[0]))
        should_be_numpy_float = [
            buffer.sum(),
            buffer.mean(),
        ]
        for s in should_be_numpy_float:
            self.assertNotIsInstance(s, AudioBuffer)
            self.assertIsInstance(s, np.float32)
            self.assertEqual(type(s), type(comparison.sum()))

        # Arithmetic
        buffer += 2.0
        self.assertAlmostEqual(buffer.sum(), 40.0)
        self.assertIsInstance(buffer, AudioBuffer)
        buffer.clear()
        self.assertAlmostEqual(buffer.sum(), 0.0)
        self.assertIsInstance(buffer, AudioBuffer)

        # Invalid initialization
        with self.assertRaises(ValueError):
            AudioBuffer(num_channels=-1, num_samples=10)
        with self.assertRaises(ValueError):
            AudioBuffer(num_channels=2, num_samples=-1)
        with self.assertRaises(ValueError):
            AudioBuffer(num_channels=2, num_samples=10, array=np.zeros((2, 5)))
        with self.assertRaises(ValueError):
            AudioBuffer(num_channels=2, num_samples=10, array=np.zeros((3, 10)))
        with self.assertRaises(TypeError):
            AudioBuffer(
                num_channels=2,
                num_samples=10,
                array=np.zeros((10, 2), dtype=np.float64),
            )

    def test_audio(self):
        # from_file
        audio = Audio.from_file(TEST_DATA_DIR / "short_stereo_mp3_22050hz.mp3")
        self.assertEqual(audio.num_channels, 2)
        self.assertEqual(audio.num_samples, 101155)
        self.assertEqual(audio.sample_rate, 22050)
        self.assertEqual(audio.shape, (101155, 2))
        self.assertAlmostEqual(audio.peak_gain, 0.291, places=3)
        self.assertAlmostEqual(audio.duration, 4.59, places=2)

        # normalize
        audio_norm = audio.normalize(in_place=False)
        self.assertAlmostEqual(audio_norm.peak_gain, 1.0, places=3)
        audio_norm = audio.normalize(peak_dbfs=-6.0, in_place=False)
        self.assertAlmostEqual(audio_norm.peak_gain, 0.501, places=3)
        audio_norm = audio.normalize(peak_dbfs=6.0, in_place=False)
        self.assertAlmostEqual(audio_norm.peak_gain, 1.995, places=3)
        self.assertAlmostEqual(audio.peak_gain, 0.291, places=3)
        audio.normalize()
        self.assertAlmostEqual(audio.peak_gain, 1.0, places=3)

        # clip
        audio_clipped = audio.clip(in_place=False)
        self.assertAlmostEqual(audio_clipped.peak_gain, 1.0, places=3)
        audio_clipped = audio.clip(peak_gain=0.5, in_place=False)
        self.assertAlmostEqual(audio_clipped.peak_gain, 0.5, places=3)
        self.assertAlmostEqual(audio.peak_gain, 1.0, places=3)
        audio.clip(peak_gain=0.25)
        self.assertAlmostEqual(audio.peak_gain, 0.25, places=3)

        # resample
        resampled = audio.resample(44100)
        self.assertEqual(resampled.num_channels, 2)
        self.assertEqual(resampled.num_samples, audio.num_samples * 2)
        self.assertEqual(resampled.sample_rate, 44100)
        self.assertEqual(resampled.shape, (101155 * 2, 2))
        self.assertAlmostEqual(resampled.duration, 4.59, places=2)

        # from_array scalar
        scalar = np.array(5.0, dtype=np.float32)
        audio = Audio.from_array(scalar, sample_rate=44100)
        self.assertEqual(audio.shape, (1, 1))
        self.assertAlmostEqual(audio.duration, 1 / 44100)

        # from_array mono
        mono_implicit = np.zeros(1000, dtype=np.float32)
        audio = Audio.from_array(mono_implicit, sample_rate=44100)
        self.assertEqual(audio.shape, (1000, 1))
        self.assertAlmostEqual(audio.duration, 1000 / 44100)
        mono_explicit = np.zeros((1000, 1), dtype=np.float32)
        audio = Audio.from_array(mono_explicit, sample_rate=44100)
        self.assertEqual(audio.shape, (1000, 1))
        self.assertAlmostEqual(audio.duration, 1000 / 44100)

        # from_array stereo
        stereo = np.zeros((1000, 2), dtype=np.float32)
        audio = Audio.from_array(stereo, sample_rate=44100)
        self.assertEqual(audio.shape, (1000, 2))
        self.assertAlmostEqual(audio.duration, 1000 / 44100)

        # Check all
        for array in (scalar, mono_implicit, mono_explicit, stereo):
            audio = Audio.from_array(array, sample_rate=44100)
            self.assertIsInstance(audio, Audio)
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(audio.sample_rate, 44100)
            self.assertEqual(audio.ndim, 2)

        # Slicing
        audio = Audio.from_array(np.zeros((10, 2), dtype=np.float32), sample_rate=44100)
        self.assertIsInstance(audio[:, :1], Audio)
        self.assertIsInstance(audio[:5, :1], Audio)
        self.assertIsInstance(audio[:5, :], Audio)
        self.assertEqual(audio[:, :1].sample_rate, audio.sample_rate)
        self.assertEqual(audio[:, :1].duration, audio.duration)
        self.assertAlmostEqual(audio[:5, :1].duration, audio.duration / 2)
        self.assertNotIsInstance(audio[:, 0], Audio)
        self.assertIsInstance(audio[:, 0], np.ndarray)
        self.assertNotIsInstance(audio.sum(), Audio)
        self.assertIsInstance(audio.sum(), np.float32)

        # Invalid initialization
        with self.assertRaises(ValueError):
            Audio.from_array(stereo, sample_rate=-1)
        with self.assertRaises(ValueError):
            Audio.from_array(np.zeros((1, 1, 1), dtype=np.float32), sample_rate=44100)
        with self.assertRaises(ValueError):
            audio.resample(-1)

        # Simple array manipulation
        audio = Audio.from_array(np.zeros((10, 2), dtype=np.float32), sample_rate=44100)
        self.assertIsInstance(audio + 0.1, Audio)
        self.assertIsInstance(audio + audio, Audio)
        self.assertEqual((audio + 0.1).sample_rate, audio.sample_rate)
        self.assertIsInstance(np.concatenate([audio, audio]), Audio)
        self.assertEqual(np.concatenate([audio, audio]).shape, (20, 2))
        self.assertEqual(np.concatenate([audio, audio]).sample_rate, audio.sample_rate)
        self.assertEqual(np.concatenate([audio, audio], axis=-1).shape, (10, 4))
        audio_48k = Audio.from_array(
            np.zeros((10, 2), dtype=np.float32), sample_rate=48000
        )
        with self.assertRaises(ValueError):
            np.concatenate([audio, audio_48k], axis=-1)
        audio_48k += 0.25
        self.assertTrue(np.all(audio_48k == 0.25))


if __name__ == "__main__":
    unittest.main()
