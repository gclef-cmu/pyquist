import unittest

# from math import isclose
from typing import List

import numpy as np

from .audio import Audio, AudioBuffer
from .realtime import AudioProcessor, BlockMessage, iter_process, process


class SineAudioProcessor(AudioProcessor):
    def __init__(self):
        super().__init__(num_input_channels=1, num_output_channels=1)

    def process_block(self, buffer: AudioBuffer, messages: List[BlockMessage]):
        del messages  # Unused
        np.sin(buffer, out=buffer)


class MessageAudioProcessor(AudioProcessor):
    def __init__(self):
        super().__init__(num_input_channels=0, num_output_channels=1)
        self.current_value = 0.0

    def process_block(self, buffer: AudioBuffer, messages: List[BlockMessage]):
        for i in range(buffer.shape[1]):
            while messages and messages[0].offset == i:
                new_value = messages[0].data
                if not isinstance(new_value, float):
                    raise TypeError("Expected float message data")
                self.current_value = new_value
                messages.pop(0)
            buffer[0, i] = self.current_value


class TestAudioProcessing(unittest.TestCase):

    def test_audio_processor(self):
        processor = SineAudioProcessor()
        self.assertEqual(processor.num_input_channels, 1)
        self.assertEqual(processor.num_output_channels, 1)
        self.assertFalse(processor.prepared)

    def test_prepare_to_play(self):
        processor = SineAudioProcessor()
        processor.prepare(sample_rate=44100, block_size=512)
        self.assertTrue(processor.prepared)
        self.assertEqual(processor.sample_rate, 44100)
        self.assertEqual(processor.block_size, 512)

    def test_release_resources(self):
        processor = SineAudioProcessor()
        processor.prepare(sample_rate=44100, block_size=512)
        processor.release()
        self.assertFalse(processor.prepared)
        with self.assertRaises(RuntimeError):
            processor.sample_rate
        with self.assertRaises(RuntimeError):
            processor.block_size

    def test_iter_process_sine(self):
        # Test parameters
        sample_rate = 44100
        duration = 0.01  # short duration for testing
        frequency = 440.0

        # Create input: a phasor that increases linearly in time:
        # phase(t) = 2π * f * t
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        phase = 2 * np.pi * frequency * t
        input_audio = Audio(
            num_channels=1, num_samples=num_samples, sample_rate=sample_rate
        )
        input_audio[0, :] = phase

        processor = SineAudioProcessor()
        block_size = 512

        # No scheduled messages for this test
        messages = []

        # Run iter_process
        blocks = list(
            iter_process(
                processor=processor,
                duration=duration,
                input_audio=input_audio,
                messages=messages,
                block_size=block_size,
                sample_rate=sample_rate,
            )
        )

        # Check that blocks are of expected shape and that processor outputs sine
        output_data = np.hstack(
            [blk[0] for blk in blocks]
        )  # blk is (buffer, block_messages)
        self.assertEqual(output_data.shape[0], num_samples)

        # Compare output to np.sin(phase)
        expected = np.sin(phase)
        # Allow a small numerical tolerance
        self.assertTrue(np.allclose(output_data, expected, atol=1e-6))

    def test_process_sine(self):
        # Test parameters
        sample_rate = 44100
        duration = 0.01  # short duration for testing
        frequency = 440.0

        # Create input phasor
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        phase = 2 * np.pi * frequency * t
        input_audio = Audio(
            num_channels=1, num_samples=num_samples, sample_rate=sample_rate
        )
        input_audio[0, :] = phase

        processor = SineAudioProcessor()
        output = process(
            processor=processor,
            duration=duration,
            input_audio=input_audio,
            block_size=512,
            sample_rate=sample_rate,
        )

        # Check output matches expected sine
        expected = np.sin(phase)
        self.assertTrue(np.allclose(output[0, :], expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
