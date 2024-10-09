import unittest

import numpy as np

from .helper import amplitude_to_dbfs, dbfs_to_amplitude


class TestHelper(unittest.TestCase):
    def test_dbfs_amplitude_conversion(self):
        dbfs = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
        amplitude = np.array([0.501187, 0.707946, 1.0, 1.412538, 1.995262])
        for d, a in zip(dbfs, amplitude):
            self.assertAlmostEqual(dbfs_to_amplitude(d), a, places=5)
            self.assertAlmostEqual(amplitude_to_dbfs(a), d, places=5)
            self.assertAlmostEqual(dbfs_to_amplitude(amplitude_to_dbfs(a)), a, places=6)
            self.assertAlmostEqual(amplitude_to_dbfs(dbfs_to_amplitude(d)), d, places=6)

        self.assertTrue(np.allclose(dbfs_to_amplitude(dbfs), amplitude))
        self.assertTrue(np.allclose(amplitude_to_dbfs(amplitude), dbfs))
        self.assertTrue(
            np.allclose(dbfs_to_amplitude(amplitude_to_dbfs(amplitude)), amplitude)
        )
        self.assertTrue(np.allclose(amplitude_to_dbfs(dbfs_to_amplitude(dbfs)), dbfs))


if __name__ == "__main__":
    unittest.main()
