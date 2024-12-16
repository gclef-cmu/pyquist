from IPython.display import Audio as IPythonAudio
from IPython.display import display

from .audio import Audio


def play(audio: Audio, *, safe: bool = True, normalize: bool = False):
    """Plays the audio."""

    if normalize:
        audio = audio.normalize(in_place=False)
    audio = audio.clip(in_place=False)
    if safe:
        audio = audio.normalize(peak_dbfs=-18.0, in_place=False)

    display(IPythonAudio(audio.swapaxes(0, 1), rate=audio.sample_rate, normalize=False))
