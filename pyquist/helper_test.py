import unittest

import numpy as np

from .helper import dbfs_to_gain, frequency_to_pitch, gain_to_dbfs, pitch_to_frequency


class TestHelper(unittest.TestCase):
    def test_dbfs_gain_conversion(self):
        dbfs = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
        gain = np.array([0.501187, 0.707946, 1.0, 1.412538, 1.995262])
        for d, a in zip(dbfs, gain):
            self.assertAlmostEqual(dbfs_to_gain(d), a, places=5)
            self.assertAlmostEqual(gain_to_dbfs(a), d, places=5)
            self.assertAlmostEqual(dbfs_to_gain(gain_to_dbfs(a)), a, places=6)
            self.assertAlmostEqual(gain_to_dbfs(dbfs_to_gain(d)), d, places=6)

        self.assertTrue(np.allclose(dbfs_to_gain(dbfs), gain))
        self.assertTrue(np.allclose(gain_to_dbfs(gain), dbfs))
        self.assertTrue(np.allclose(dbfs_to_gain(gain_to_dbfs(gain)), gain))
        self.assertTrue(np.allclose(gain_to_dbfs(dbfs_to_gain(dbfs)), dbfs))

    def test_frequency_pitch_conversion(self):
        frequency = np.array([440.0, 523.251, 659.255, 783.991, 987.767])
        pitch = np.array([69.0, 72.0, 76.0, 79.0, 83.0])
        for f, p in zip(frequency, pitch):
            self.assertAlmostEqual(frequency_to_pitch(f), p, places=2)
            self.assertAlmostEqual(pitch_to_frequency(p), f, places=2)
            self.assertAlmostEqual(
                frequency_to_pitch(pitch_to_frequency(p)), p, places=6
            )
            self.assertAlmostEqual(
                pitch_to_frequency(frequency_to_pitch(f)), f, places=6
            )

        self.assertTrue(np.allclose(frequency_to_pitch(frequency), pitch))
        self.assertTrue(np.allclose(pitch_to_frequency(pitch), frequency))
        self.assertTrue(
            np.allclose(frequency_to_pitch(pitch_to_frequency(pitch)), pitch)
        )
        self.assertTrue(
            np.allclose(pitch_to_frequency(frequency_to_pitch(frequency)), frequency)
        )


if __name__ == "__main__":
    unittest.main()
