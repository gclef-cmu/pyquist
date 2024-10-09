from .audio import Audio


def play(audio: Audio, *, safe: bool = True, normalize: bool = False):
    """Plays the audio using sounddevice."""
    import sounddevice as sd

    if normalize:
        audio = audio.normalize(in_place=False)
    audio = audio.clip(in_place=False)
    if safe:
        audio = audio.normalize(peak_dbfs=-18.0, in_place=False)
    sd.play(audio.swapaxes(0, 1), audio.sample_rate)
    sd.wait()
