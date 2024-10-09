from IPython.display import Audio as IPythonAudio
from IPython.display import display

from .audio import Audio


def play(audio: Audio, *, normalize: bool = False):
    """Plays the audio."""
    display(IPythonAudio(audio, rate=audio.sample_rate, normalize=normalize))
