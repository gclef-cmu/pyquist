import abc
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

import sounddevice as sd

from .audio import Audio, AudioBuffer


@dataclass
class BlockMessage:
    """A real-time control input to an audio processor.

    Can come from a variety of sources, such as MIDI, OSC, or a GUI.

    Attributes:
        offset: The offset in samples from the start of the block.
        data: The message data.
    """

    data: Any
    offset: int = 0


@dataclass
class Message:
    """A message timestamped in seconds, mainly used for scheduling.

    Attributes:
        time: The time in seconds. ASAP if <=0.
        data: The message data.
    """

    data: Any
    time: float = 0.0


class AudioProcessor(abc.ABC):
    """
    Abstract base class for audio processors.

    Type signature inspired by JUCE AudioProcessor.

    https://docs.juce.com/master/classAudioProcessor.html
    """

    def __init__(self, num_input_channels: int = 0, num_output_channels: int = 1):
        """Initializes the audio processor."""
        self._sample_rate: Optional[int] = None
        self._block_size: Optional[int] = None
        self._prepared = False
        self._num_input_channels = num_input_channels
        self._num_output_channels = num_output_channels

    @property
    def num_input_channels(self) -> int:
        """Returns the number of input channels."""
        return self._num_input_channels

    @property
    def num_output_channels(self) -> int:
        """Returns the number of output channels."""
        return self._num_output_channels

    @property
    def prepared(self) -> bool:
        """Returns True if the processor is prepared for playback."""
        return self._prepared

    @property
    def sample_rate(self) -> int:
        """Once prepared, returns the sample rate of the AudioProcessor."""
        if not self.prepared:
            raise RuntimeError("Audio processor is not prepared for playback")
        assert self._sample_rate is not None
        return self._sample_rate

    @property
    def block_size(self) -> int:
        """Once prepared, returns the block size of the AudioProcessor."""
        if not self.prepared:
            raise RuntimeError("Audio processor is not prepared for playback")
        assert self._block_size is not None
        return self._block_size

    def prepare(self, sample_rate: int, block_size: int):
        """Prepares the processor for playback with the given sample rate and block size."""
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._prepared = True

    def release(self):
        """Releases any resources that are no longer needed."""
        self._sample_rate = None
        self._block_size = None
        self._prepared = False

    @abc.abstractmethod
    def process_block(self, buffer: AudioBuffer, messages: List[BlockMessage]):
        """Override with your block processing logic.

        The audio buffer will be of shape `(num_channels, block_size)`, where
        `num_channels = max(num_input_channels, num_output_channels)`. The first
        `num_input_channels` channels will contain the input audio, and the first
        `num_output_channels` channels should be written with the output audio.

        Args:
            buffer: The audio buffer to process in place.
            messages: The block messages to process. Assume they are sorted by offset.
        """

    def __call__(self, buffer: AudioBuffer, messages: List[BlockMessage]):
        """Processes a block of audio with the given messages."""
        if not self.prepared:
            raise RuntimeError("Audio processor is not prepared for playback")
        self.process_block(buffer, messages)


class AudioProcessorStream(sd.OutputStream):
    """A sounddevice output stream that processes audio with an AudioProcessor."""

    def __init__(
        self,
        processor: AudioProcessor,
        *,
        block_size: int = 512,
        sample_rate: int = 44100,
    ):
        if processor.num_input_channels > 0:
            raise NotImplementedError("Input channels are not supported yet")
        super().__init__(
            samplerate=sample_rate,
            blocksize=block_size,
            channels=processor.num_output_channels,
            callback=self.callback,
        )
        self._processor = processor
        self._block_size = block_size
        self._sample_rate = sample_rate
        self._message_queue: deque[Message] = deque()
        self._dequeud_messages: list[Message] = []
        self._blocks_elapsed = 0
        self._seconds_per_block = block_size / sample_rate

    @property
    def message_queue(self) -> deque[Message]:
        return self._message_queue

    def callback(self, outdata, *args):
        del args

        # Prepare AudioBuffer
        buffer = AudioBuffer(
            num_channels=self._processor.num_output_channels,
            num_samples=self._block_size,
            array=outdata,
        )

        # Dequeue all available messages
        while len(self._message_queue):
            self._dequeud_messages.append(self._message_queue.popleft())

        # Process dequeued messages
        block_messages = []
        remaining_messages = []
        t = self._blocks_elapsed * self._seconds_per_block
        t_max = t + self._seconds_per_block
        for message in self._dequeud_messages:
            if message.time < t_max:
                block_message = BlockMessage(
                    offset=max(int((message.time - t) * self._sample_rate), 0),
                    data=message.data,
                )
                assert block_message.offset < self._block_size
                block_messages.append(block_message)
            else:
                remaining_messages.append(message)
        self._dequeud_messages = remaining_messages

        # Process block
        self._processor(buffer, block_messages)
        self._blocks_elapsed += 1

    def start(self, *args, **kwargs):
        self._processor.prepare(self._sample_rate, self._block_size)
        self._blocks_elapsed = 0
        self._message_queue.clear()
        super().start(*args, **kwargs)

    def stop(self, *args, **kwargs):
        super().stop(*args, **kwargs)

    def abort(self, *args, **kwargs):
        super().abort(*args, **kwargs)

    def close(self, *args, **kwargs):
        super().close(*args, **kwargs)
        self._processor.release()

    def sleep(self, duration: float):
        sd.sleep(int(duration * 1000))


def iter_process(
    processor: AudioProcessor,
    duration: float,
    input_audio: Optional[Audio] = None,
    messages: Optional[List[Message]] = None,
    *,
    pad_end: bool = True,
    block_size: int = 512,
    sample_rate: int = 44100,
) -> Iterator[Tuple[int, AudioBuffer, List[BlockMessage]]]:
    """Iteratively processes audio with the given audio processor.

    Yields audio blocks as they are processed.

    Args:
        processor: The audio processor to use.
        duration: The duration of the audio in seconds.
        input_audio: The input audio to process.
        messages: The scheduled messages to send to the processor.
        pad_end: Whether to pad the end of the audio with zeros.
        block_size: The block size to use for processing.
        sample_rate: The sample rate to use for processing.

    Yields:
        AudioBuffer: The processed audio block.
    """
    if messages is None:
        messages = []
    if input_audio is None and processor.num_input_channels > 0:
        raise ValueError("Input audio is required for this processor")

    processor.prepare(sample_rate, block_size)
    num_samples = int(duration * sample_rate)
    block = AudioBuffer(
        num_channels=max(processor.num_input_channels, processor.num_output_channels),
        num_samples=block_size,
    )
    messages = sorted(messages, key=lambda m: m.time)

    for i in range(0, num_samples, block_size):
        # Handle incomplete blocks
        block_i_size = min(block_size, num_samples - i)
        if block_i_size < block_size and not pad_end:
            break

        # Handle scheduled messages
        block_messages = []
        block_i_time = i / sample_rate
        block_ip1_time = (i + block_size) / sample_rate
        while messages and messages[0].time < block_ip1_time:
            message = messages.pop(0)
            offset = int((message.time - block_i_time) * sample_rate)
            assert 0 <= offset < block_size
            block_messages.append(BlockMessage(offset=offset, data=message.data))

        # Buffer input audio
        if input_audio is not None:
            input_block = input_audio[
                i : i + block_i_size, : processor.num_input_channels
            ]
            block[: input_block.shape[0], : input_block.shape[1]] = input_block

        # Compute output audio
        processor(block, block_messages)

        # Pad end of block with zeros if needed
        if block_i_size < block_size:
            block[block_i_size:, : processor.num_output_channels] = 0.0

        yield i, block[:, : processor.num_output_channels], block_messages


def process(
    processor: AudioProcessor,
    duration: float,
    input_audio: Optional[Audio] = None,
    messages: Optional[List[Message]] = None,
    *,
    block_size: int = 512,
    sample_rate: int = 44100,
) -> Audio:
    num_samples = int(duration * sample_rate)
    output = Audio(
        num_channels=processor.num_output_channels,
        num_samples=num_samples,
        sample_rate=sample_rate,
    )
    for block_offset, block, _ in iter_process(
        processor=processor,
        duration=duration,
        input_audio=input_audio,
        messages=messages,
        block_size=block_size,
        sample_rate=sample_rate,
        pad_end=True,
    ):
        output_block = output[
            block_offset : block_offset + block.shape[0],
            : processor.num_output_channels,
        ]
        output_block[:] = block[
            : output_block.shape[0],
            : processor.num_output_channels,
        ]
    return output
