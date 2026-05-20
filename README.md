# `pyquist`

`pyquist` provides basic utilities for **low-level computer music programming in Python and NumPy**.

`pyquist` is **designed for learning** and is the teaching library for CMU's 15-322 Intro to Computer Music. Its primary purpose is to provide a _barebones foundation_ for working with audio in Python, e.g., allocating sample buffers, audio file decoding and playback. Accordingly, it intentionally lacks a lot of functionality found in full-fledged computer music programming frameworks, such as a rich collection of unit generators.

Want to try it without installing anything? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gclef-cmu/pyquist/blob/main/examples/HelloPyquist.ipynb)

## Quick example

```python
import numpy as np
import pyquist as pq

# A 1-second, 440 Hz sine wave at CD quality.
sr = 44100
t = np.arange(sr) / sr
audio = pq.Audio(0.5 * np.sin(2 * np.pi * 440 * t), sample_rate=sr)

pq.play(audio)        # play through default output (or notebook widget)
pq.plot(audio)        # waveform
pq.plot_spec(audio)   # spectrogram
```

Loading and transforming:

```python
riff = pq.Audio.from_url(
    "https://github.com/gclef-cmu/pyquist/raw/refs/heads/main/pyquist/test_data/388954__fullmetaljedi__blues-riff-in-g-nylon.wav"
)
clip = riff.segment(offset=5.0, duration=3.0).resample(8000)
pq.play(clip)
```

Rendering a `Score` with a custom instrument:

```python
from pyquist.score import Score, Event, BasicMetronome
from pyquist.helper import pitch_to_frequency

def sine_instrument(event):
    sr = 44100
    t = np.arange(int(event.kwargs["duration"] * sr)) / sr
    freq = pitch_to_frequency(event.kwargs["pitch"])
    return pq.Audio(0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-3 * t), sample_rate=sr)

score = Score([
    Event(0, {"pitch": 60, "duration": 0.5}),
    Event(1, {"pitch": 64, "duration": 0.5}),
    Event(2, {"pitch": 67, "duration": 0.5}),
])
pq.play(score.render(sine_instrument, metronome=BasicMetronome(120)))
```

For a guided walkthrough — visualization, MIDI parsing, scores, instruments — open [`examples/HelloPyquist.ipynb`](examples/HelloPyquist.ipynb) or [run it in Colab](https://colab.research.google.com/github/gclef-cmu/pyquist/blob/main/examples/HelloPyquist.ipynb).

## Installation

Requires **Python 3.10 or later** and **git** (used by `pip` to fetch the package).

### macOS

```sh
brew install git python@3.10          # or 3.11, 3.12, 3.13
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade git+https://github.com/gclef-cmu/pyquist.git
```

If `pq.play(...)` is silent, give Terminal (or your IDE) microphone/audio access in **System Settings → Privacy & Security → Microphone**.

### Linux

Install Python, git, and the PortAudio system library that `sounddevice` wraps:

```sh
# Debian / Ubuntu
sudo apt install python3 python3-venv git libportaudio2

# Fedora
sudo dnf install python3 python3-virtualenv git portaudio

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade git+https://github.com/gclef-cmu/pyquist.git
```

### Windows

Install [Python](https://www.python.org/downloads/) (3.10 or later, with "Add Python to PATH" checked) and [Git for Windows](https://git-scm.com/download/win), then open **Command Prompt**:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade git+https://github.com/gclef-cmu/pyquist.git
```

### From source

For hacking on pyquist itself:

```sh
git clone https://github.com/gclef-cmu/pyquist.git
cd pyquist
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebook]"
pre-commit install
```

### Pick default audio devices

Once installed, the `pyquist` CLI lets you choose persistent default input/output devices (handy for laptops with multiple interfaces):

```sh
pyquist devices
```

### Run notebooks

```sh
pip install jupyter ipykernel ipywidgets
python -m ipykernel install --user --name=pyquist --display-name "Pyquist"
cd examples
jupyter notebook
```

Then open [`HelloPyquist.ipynb`](examples/HelloPyquist.ipynb) and select the **Pyquist** kernel.

## Acknowledgements

Inspired in part by:

- [Nyquist](https://sourceforge.net/projects/nyquist/)
- [Tone.js](https://tonejs.github.io/)
- [JUCE](https://juce.com/)
