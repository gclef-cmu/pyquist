# `pyquist`

New Python library for CMU 15-322 Intro to Computer Music. WIP.

`pyquist` is not a full-fledged computer music programming framework. You will not find a rich collection of high-level unit generators for synthesis and processing.

Instead, `pyquist` provides basic abstractions for adapting Python for lower-level computer music programming via NumPy. It is a simple foundation for building more sophisticated computer music applications.

Principles:

- Simple
- Minimal dependencies
- Accessible implementations
- Well-documented core functionality

## Installation

Requires Python 3.10 or later. `virtualenv` is recommended.

### Via pip

`pip install --upgrade git+https://github.com/gclef-cmu/pyquist.git`

### From source

```sh
git clone git@github.com:gclef-cmu/pyquist.git
cd pyquist
python3 -m virtualenv .venv
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

## For development

```sh
git clone git@github.com:gclef-cmu/pyquist.git
brew install python@3.10
python3.10 -m virtualenv .venv
source .venv/bin/activate
pip install -e .
pip install pre-commit
pre-commit install
```

## Acknowledgements

Inspired by:

- Nyquist
- Tone.js
- JUCE
