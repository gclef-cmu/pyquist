import unittest

from .realtime import AudioProcessor


class TestAudioProcessor(unittest.TestCase):
    def test_audio_processor(self):
        class TestProcessor(AudioProcessor):
            def process_block(self, buffer):
                pass

        processor = TestProcessor()
        self.assertIsNone(processor.sample_rate)
        self.assertIsNone(processor.block_size)
        processor.prepare_to_play(44100, 512)
        self.assertEqual(processor.sample_rate, 44100)
        self.assertEqual(processor.block_size, 512)
        processor.release_resources()
        processor.process_block(None)

        with self.assertRaises(TypeError):
            AudioProcessor()


if __name__ == "__main__":
    unittest.main()
