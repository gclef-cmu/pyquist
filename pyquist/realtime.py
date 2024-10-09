import abc

from .audio import AudioBuffer


class AudioProcessor(abc.ABC):
    """
    Abstract base class for audio processors.

    Type signature inspired by JUCE AudioProcessor.

    https://docs.juce.com/master/classAudioProcessor.html
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
