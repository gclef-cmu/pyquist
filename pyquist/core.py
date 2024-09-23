import abc
from typing import Optional

import numpy as np


class AudioBuffer:
    """Represents a buffer of raw audio samples."""

    def __init__(
        self,
        num_channels: int,
        num_samples: int,
        raw_buffer: Optional[np.ndarray] = None,
    ):
        """
        Initializes a buffer of raw 32-bit float audio samples.

        Parameters:
            num_channels: The number of channels in the buffer.
            num_samples: The number of samples in the buffer.
            raw_buffer: An optional numpy array to use as the buffer.
                        If None, a zero-filled buffer will be created.
                        Must be same shape as num_channels / num_samples.
        """
        if raw_buffer is None:
            raw_buffer = np.zeros((num_channels, num_samples), dtype=np.float32)
        if raw_buffer.shape != (num_channels, num_samples):
            raise ValueError("Provided buffer does not match specified dimensions.")
        self._buffer = raw_buffer

    @property
    def num_channels(self):
        """Returns the number of channels in the buffer."""
        return self._buffer.shape[0]

    @property
    def num_samples(self):
        """Returns the number of samples in the buffer."""
        return self._buffer.shape[1]

    def resize(self, num_channels: int, num_samples: int):
        """
        Resizes the buffer to the specified number of channels and samples.

        Parameters:
            num_channels: The new number of channels in the buffer.
            num_samples: The new number of samples in the buffer.
        """
        if num_channels != self.num_channels or num_samples != self.num_samples:
            self._buffer = np.zeros((num_channels, num_samples), dtype=np.float32)

    def __getitem__(self, *args, **kwargs):
        """Passes slice syntax to underlying buffer for reads."""
        return self._buffer.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        """Passes slice syntax to underlying buffer for writes."""
        return self._buffer.__setitem__(*args, **kwargs)


class AudioProcessor(abc.ABC):
    """
    Abstract base class for audio processors.

    Type signature inspired by JUCE AudioProcessor.
    """

    def __init__(self):
        """Initializes the audio processor."""
        self.sample_rate = None
        self.block_size = None

    def prepare_to_play(self, sample_rate: float, block_size: int):
        """Prepares the processor for playback with the given sample rate and block size."""
        self.sample_rate = sample_rate
        self.block_size = block_size

    def release_resources(self):
        """Releases any resources that are no longer needed."""
        pass

    @abc.abstractmethod
    def process_block(self, buffer: AudioBuffer):
        """Processes the audio block in the buffer."""
        pass
