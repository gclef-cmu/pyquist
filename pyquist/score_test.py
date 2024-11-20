import unittest

from .score import BasicMetronome


class TestScore(unittest.TestCase):
    @unittest.skip("Test not implemented")
    def test_metronome(self):
        raise NotImplementedError("Test not implemented")

    def test_basic_metronome(self):
        met = BasicMetronome(120)
        self.assertEqual(met.bpm, 120)
        self.assertEqual(met.beat_duration, 0.5)
        self.assertEqual(met.beat_to_time(1.0), 0.5)
        self.assertEqual(met.time_to_beat(0.5), 1.0)
        met = BasicMetronome(60)
        self.assertEqual(met.beat_duration, 1.0)
        self.assertEqual(met.beat_to_time(1.0), 1.0)
        self.assertEqual(met.time_to_beat(1.0), 1.0)

    @unittest.skip("Test not implemented")
    def test_render_score(self):
        raise NotImplementedError("Test not implemented")


if __name__ == "__main__":
    unittest.main()
