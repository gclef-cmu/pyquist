# `pyquist`

Python library for CMU 15-322 Intro to Computer Music. A work in progress.

`pyquist` provides a basic foundation for **low-level development of computer music applications in Python / NumPy**.

`pyquist` is **designed for teaching**. Accordingly, it lacks a lot of functionality found in found in full-fledged computer music programming frameworks, such as a rich collection of unit generators.

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
