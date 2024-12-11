import abc
from typing import Optional

from .audio import AudioBuffer


class AudioProcessor(abc.ABC):
    """
    Abstract base class for audio processors.

    Type signature inspired by JUCE AudioProcessor.

    https://docs.juce.com/master/classAudioProcessor.html
    """

    def __init__(self):
        """Initializes the audio processor."""
        self._sample_rate = None
        self._block_size = None
        self._prepared = False

    @property
    def prepared(self):
        """Returns True if the processor is prepared for playback."""
        return self._prepared

    @property
    def sample_rate(self):
        """Returns the sample rate of the audio processor."""
        if not self.prepared:
            raise RuntimeError("Audio processor is not prepared for playback")
        return self._sample_rate

    @property
    def block_size(self):
        """Returns the block size of the audio processor."""
        if not self.prepared:
            raise RuntimeError("Audio processor is not prepared for playback")
        return self._block_size

    def prepare_to_play(self, sample_rate: float, block_size: int):
        """Prepares the processor for playback with the given sample rate and block size."""
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._prepared = True

    def release_resources(self):
        """Releases any resources that are no longer needed."""
        self._sample_rate = None
        self._block_size = None
        self._prepared = False

    @abc.abstractmethod
    def process_block(self, buffer: Optional[AudioBuffer]):
        """Processes the audio block in the buffer."""
        pass

    def __call__(self, buffer: Optional[AudioBuffer]):
        """Processes the audio block in the buffer."""
        if not self.prepared:
            raise RuntimeError("Audio processor is not prepared for playback")
        self.process_block(buffer)
