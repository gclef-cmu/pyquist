# `pyquist`

`pyquist` provides basic utilities for **low-level computer music programming in Python and NumPy**.

`pyquist` is **designed for learning** and is the teaching library for CMU's 15-322 Intro to Computer Music. Its primary purpose is to provide a _barebones foundation_ for working with audio in Python, e.g., allocating sample buffers, audio file decoding and playback. Accordingly, it intentionally lacks a lot of functionality found in found in full-fledged computer music programming frameworks, such as a rich collection of unit generators.

## Installation

Requires Python 3.10 or later. `venv` is recommended.

### Via pip

`pip install --upgrade git+https://github.com/gclef-cmu/pyquist.git`

### From source

```sh
git clone git@github.com:gclef-cmu/pyquist.git
cd pyquist
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run via Jupyter

```sh
pip install jupyter ipykernel ipywidgets
python -m ipykernel install --user --name=pyquist --display-name "Pyquist"
cd examples
jupyter notebook
```

## For development on MacOS

```sh
git clone git@github.com:gclef-cmu/pyquist.git
brew install python@3.10
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pre-commit
pre-commit install
```

## Acknowledgements

Some inspiration from:

- [Nyquist](https://sourceforge.net/projects/nyquist/)
- [Tone.js](https://tonejs.github.io/)
- [JUCE](https://juce.com/)
