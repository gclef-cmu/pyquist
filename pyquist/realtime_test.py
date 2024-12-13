import unittest

# from math import isclose
from typing import List

import numpy as np

from .audio import Audio, AudioBuffer
from .realtime import AudioProcessor, BlockMessage, Message, iter_process, process


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
        messages = messages[:]
        for i in range(buffer.shape[0]):
            while messages and messages[0].offset == i:
                new_value = messages[0].data
                if not isinstance(new_value, float):
                    raise TypeError("Expected float message data")
                self.current_value = new_value
                messages.pop(0)
            buffer[i] = self.current_value


class TestAudioProcessing(unittest.TestCase):

    def test_audio_processor(self):
        with self.assertRaises(TypeError):
            AudioProcessor()
        processor = SineAudioProcessor()
        self.assertEqual(processor.num_input_channels, 1)
        self.assertEqual(processor.num_output_channels, 1)
        self.assertFalse(processor.prepared)
        with self.assertRaises(RuntimeError):
            processor.sample_rate
        with self.assertRaises(RuntimeError):
            processor.block_size

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

    def test_process(self):
        processor = SineAudioProcessor()
        freq = 440.0

        for duration in [0.0, 0.01, 0.1]:
            for sample_rate in [22050, 44100]:
                # Create input: a phasor that increases linearly in time:
                # phase(t) = 2π * f * t
                num_samples = int(sample_rate * duration)
                input_audio = Audio(
                    num_channels=1, num_samples=num_samples, sample_rate=sample_rate
                )
                t = np.arange(num_samples) / sample_rate
                phase = (2 * np.pi * freq * t).astype(np.float32)
                expected = np.sin(phase)
                if num_samples > 0:
                    input_audio[:, 0] = phase

                for block_size in [1, 256, 512, 1024]:

                    # Check iter_process
                    for pad_end in [False, True]:
                        # Run iter_process
                        num_blocks = 0
                        for block_offset, block, block_messages in iter_process(
                            duration=duration,
                            processor=processor,
                            input_audio=input_audio,
                            pad_end=pad_end,
                            block_size=block_size,
                            sample_rate=sample_rate,
                        ):
                            self.assertEqual(block.shape, (block_size, 1))
                            expected_block = expected[
                                block_offset : block_offset + block_size
                            ]
                            self.assertTrue(
                                pad_end or expected_block.size == block_size
                            )
                            self.assertTrue(
                                np.array_equal(
                                    block[: expected_block.size, 0],
                                    expected_block,
                                )
                            )
                            self.assertEqual(len(block_messages), 0)
                            num_blocks += 1

                        # Check block count
                        expected_blocks = (np.ceil if pad_end else np.floor)(
                            num_samples / block_size
                        )
                        self.assertEqual(num_blocks, expected_blocks)

                    # Check process
                    audio = process(
                        processor=processor,
                        duration=duration,
                        input_audio=input_audio,
                        block_size=block_size,
                        sample_rate=sample_rate,
                    )
                    self.assertEqual(audio.num_channels, 1)
                    self.assertEqual(audio.num_samples, num_samples)
                    self.assertEqual(audio.sample_rate, sample_rate)
                    self.assertTrue(np.array_equal(audio[:, 0], expected))

    def test_process_messages(self):
        processor = MessageAudioProcessor()

        duration = 0.01  # 10 samples at 1000 Hz (sample_rate=1000)
        sample_rate = 1000
        num_samples = int(duration * sample_rate)  # 10 samples

        # Messages at sample times 0, 3, and 5
        messages = [
            Message(time=0.000, data=1.0),
            Message(time=0.003, data=2.0),
            Message(time=0.005, data=3.0),
        ]

        # Expected output values after each message:
        # Samples: 0..9
        # Initial value: 0.0
        # After sample 0: value=1.0
        # After sample 3: value=2.0
        # After sample 5: value=3.0
        # Final expected output: [1,1,1,2,2,3,3,3,3,3]

        expected_output = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3], dtype=np.float32)

        for block_size in [2, 3, 4]:
            blocks = []
            for _, block, block_msgs in iter_process(
                processor=processor,
                duration=duration,
                input_audio=None,
                messages=messages[:],
                pad_end=True,
                block_size=block_size,
                sample_rate=sample_rate,
            ):
                self.assertTrue(len(block_msgs) in (0, 1, 2))
                blocks.append(block[:, 0].copy())

            # Concatenate processed blocks up to num_samples
            output = np.concatenate(blocks)[:num_samples]
            self.assertTrue(np.array_equal(output, expected_output))

            # Check process
            audio = process(
                processor=processor,
                duration=duration,
                messages=messages[:],
                block_size=block_size,
                sample_rate=sample_rate,
            )
            self.assertEqual(audio.num_channels, 1)
            self.assertEqual(audio.num_samples, num_samples)
            self.assertEqual(audio.sample_rate, sample_rate)
            self.assertTrue(np.array_equal(audio[:, 0], expected_output))


if __name__ == "__main__":
    unittest.main()
