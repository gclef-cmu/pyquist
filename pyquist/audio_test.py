import unittest

import numpy as np

from pyquist.audio import Audio, AudioBuffer
from pyquist.paths import TEST_DATA_DIR


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
        self.assertEqual(buffer.shape, (2, 10))

        # Slicing
        comparison = np.zeros((2, 10), dtype=np.float32)
        self.assertNotIsInstance(buffer[0], AudioBuffer)
        self.assertIsInstance(buffer[0], np.ndarray)
        self.assertEqual(type(buffer[0]), type(comparison[0]))
        self.assertNotIsInstance(buffer.sum(), AudioBuffer)
        self.assertIsInstance(buffer.sum(), np.float32)
        self.assertEqual(type(buffer.sum()), type(comparison.sum()))

        # Arithmetic
        buffer[:] += 2.0
        self.assertAlmostEqual(buffer.sum(), 40.0)
        self.assertIsInstance(buffer, AudioBuffer)
        self.assertIsInstance(buffer, np.ndarray)
        buffer.clear()
        self.assertAlmostEqual(buffer.sum(), 0.0)
        self.assertIsInstance(buffer, AudioBuffer)
        self.assertIsInstance(buffer, np.ndarray)

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
                array=np.zeros((2, 10), dtype=np.float64),
            )

    def test_audio(self):
        # from_file
        audio = Audio.from_file(TEST_DATA_DIR / "short_stereo_mp3_22050hz.mp3")
        self.assertEqual(audio.num_channels, 2)
        self.assertEqual(audio.num_samples, 101155)
        self.assertEqual(audio.sample_rate, 22050)
        self.assertEqual(audio.shape, (2, 101155))
        self.assertAlmostEqual(audio.peak_amplitude, 0.291, places=3)
        self.assertAlmostEqual(audio.duration, 4.59, places=2)

        # normalize
        audio_norm = audio.normalize(in_place=False)
        self.assertAlmostEqual(audio_norm.peak_amplitude, 1.0, places=3)
        audio_norm = audio.normalize(peak_dbfs=-6.0, in_place=False)
        self.assertAlmostEqual(audio_norm.peak_amplitude, 0.501, places=3)
        audio_norm = audio.normalize(peak_dbfs=6.0, in_place=False)
        self.assertAlmostEqual(audio_norm.peak_amplitude, 1.995, places=3)
        self.assertAlmostEqual(audio.peak_amplitude, 0.291, places=3)
        audio.normalize()
        self.assertAlmostEqual(audio.peak_amplitude, 1.0, places=3)

        # resample
        resampled = audio.resample(44100)
        self.assertEqual(resampled.num_channels, 2)
        self.assertEqual(resampled.num_samples, audio.num_samples * 2)
        self.assertEqual(resampled.sample_rate, 44100)
        self.assertEqual(resampled.shape, (2, 101155 * 2))
        self.assertAlmostEqual(resampled.duration, 4.59, places=2)

        # from_array scalar
        scalar = np.array(5.0, dtype=np.float32)
        audio = Audio.from_array(scalar, sample_rate=44100)
        self.assertEqual(audio.shape, (1, 1))
        self.assertAlmostEqual(audio.duration, 1 / 44100)

        # from_array mono
        mono_implicit = np.zeros(1000, dtype=np.float32)
        audio = Audio.from_array(mono_implicit, sample_rate=44100)
        self.assertEqual(audio.shape, (1, 1000))
        self.assertAlmostEqual(audio.duration, 1000 / 44100)
        mono_explicit = np.zeros((1, 1000), dtype=np.float32)
        audio = Audio.from_array(mono_explicit, sample_rate=44100)
        self.assertEqual(audio.shape, (1, 1000))
        self.assertAlmostEqual(audio.duration, 1000 / 44100)

        # from_array stereo
        stereo = np.zeros((2, 1000), dtype=np.float32)
        audio = Audio.from_array(stereo, sample_rate=44100)
        self.assertEqual(audio.shape, (2, 1000))
        self.assertAlmostEqual(audio.duration, 1000 / 44100)

        # Check all
        for array in (scalar, mono_implicit, mono_explicit, stereo):
            audio = Audio.from_array(array, sample_rate=44100)
            self.assertIsInstance(audio, Audio)
            self.assertIsInstance(audio, np.ndarray)
            self.assertEqual(audio.sample_rate, 44100)
            self.assertEqual(audio.ndim, 2)

        # Invalid initialization
        with self.assertRaises(ValueError):
            Audio.from_array(stereo, sample_rate=-1)
        with self.assertRaises(ValueError):
            Audio.from_array(np.zeros((1, 1, 1), dtype=np.float32), sample_rate=44100)
        with self.assertRaises(ValueError):
            audio.resample(-1)


if __name__ == "__main__":
    unittest.main()
