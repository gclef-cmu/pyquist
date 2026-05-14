import unittest

import numpy as np

from .helper import (
    amplitude_to_db,
    db_to_amplitude,
    frequency_to_pitch,
    pitch_name_to_pitch,
    pitch_to_frequency,
)


class TestHelper(unittest.TestCase):
    def test_db_amplitude_conversion(self):
        db = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])
        amp = np.array([0.501187, 0.707946, 1.0, 1.412538, 1.995262])
        for d, a in zip(db, amp):
            self.assertAlmostEqual(db_to_amplitude(d), a, places=5)
            self.assertAlmostEqual(amplitude_to_db(a), d, places=5)
            self.assertAlmostEqual(db_to_amplitude(amplitude_to_db(a)), a, places=6)
            self.assertAlmostEqual(amplitude_to_db(db_to_amplitude(d)), d, places=6)

        self.assertTrue(np.allclose(db_to_amplitude(db), amp))
        self.assertTrue(np.allclose(amplitude_to_db(amp), db))
        self.assertTrue(np.allclose(db_to_amplitude(amplitude_to_db(amp)), amp))
        self.assertTrue(np.allclose(amplitude_to_db(db_to_amplitude(db)), db))

    def test_db_amplitude_well_known_values(self):
        # Sanity-check a few canonical values with default reference=1.0.
        self.assertAlmostEqual(db_to_amplitude(0.0), 1.0, places=6)
        self.assertAlmostEqual(db_to_amplitude(-20.0), 0.1, places=6)
        self.assertAlmostEqual(amplitude_to_db(1.0), 0.0, places=6)
        self.assertAlmostEqual(amplitude_to_db(0.1), -20.0, places=6)

    def test_db_amplitude_with_reference(self):
        # 0 dB relative to a reference returns the reference itself.
        self.assertAlmostEqual(db_to_amplitude(0.0, reference=0.5), 0.5, places=6)
        self.assertAlmostEqual(db_to_amplitude(0.0, reference=2.0), 2.0, places=6)
        # Conversely, an amplitude equal to the reference is 0 dB relative.
        self.assertAlmostEqual(amplitude_to_db(0.5, reference=0.5), 0.0, places=6)
        self.assertAlmostEqual(amplitude_to_db(2.0, reference=2.0), 0.0, places=6)
        # -6 dB below a non-unity reference still halves the level (roughly).
        self.assertAlmostEqual(
            db_to_amplitude(-6.0, reference=0.5), 0.5 * 0.501187, places=5
        )
        # Round-trip with a non-unity reference.
        for ref in [0.25, 0.5, 2.0, 10.0]:
            for d in [-12.0, -3.0, 0.0, 6.0]:
                a = db_to_amplitude(d, reference=ref)
                self.assertAlmostEqual(amplitude_to_db(a, reference=ref), d, places=6)

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

    def test_pitch_name_to_pitch(self):
        self.assertEqual(pitch_name_to_pitch("B3"), 59)
        self.assertEqual(pitch_name_to_pitch("C4"), 60)
        self.assertEqual(pitch_name_to_pitch("D4"), 62)
        self.assertEqual(pitch_name_to_pitch("E4"), 64)
        self.assertEqual(pitch_name_to_pitch("F4"), 65)
        self.assertEqual(pitch_name_to_pitch("G4"), 67)
        self.assertEqual(pitch_name_to_pitch("A4"), 69)
        self.assertEqual(pitch_name_to_pitch("B4"), 71)
        self.assertEqual(pitch_name_to_pitch("C5"), 72)

        self.assertEqual(pitch_name_to_pitch("Bb3"), 58)
        self.assertEqual(pitch_name_to_pitch("Bbbbb3"), 55)
        self.assertEqual(pitch_name_to_pitch("C#4"), 61)
        self.assertEqual(pitch_name_to_pitch("C####4"), 64)
        self.assertEqual(pitch_name_to_pitch("c4"), 60)

        self.assertEqual(
            pitch_name_to_pitch(np.array(["B3", "C4", "D4"])).tolist(), [59, 60, 62]
        )

        with self.assertRaises(ValueError):
            pitch_name_to_pitch("A")
        with self.assertRaises(ValueError):
            pitch_name_to_pitch("Z3")
        with self.assertRaises(ValueError):
            pitch_name_to_pitch("Cd3")
        with self.assertRaises(ValueError):
            pitch_name_to_pitch("BK")


if __name__ == "__main__":
    unittest.main()
