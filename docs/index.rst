Pyquist documentation
=====================

``pyquist`` provides basic utilities for **low-level computer music programming
in Python and NumPy**. It is designed for learning and is the teaching library
for CMU's 15-322 *Intro to Computer Music*.

* Source code: https://github.com/gclef-cmu/pyquist
* Hands-on walkthrough:
  `HelloPyquist notebook <https://github.com/gclef-cmu/pyquist/blob/main/examples/HelloPyquist.ipynb>`_
  (or `run it in Colab
  <https://colab.research.google.com/github/gclef-cmu/pyquist/blob/main/examples/HelloPyquist.ipynb>`_
  without installing anything)

This site is the auto-generated API reference.


Installation
------------

Requires **Python 3.10 or later** and **git** (used by ``pip`` to fetch the
package).

macOS
~~~~~

.. code-block:: sh

   brew install git python@3.10          # or 3.11, 3.12, 3.13
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade git+https://github.com/gclef-cmu/pyquist.git

If ``pq.play(...)`` is silent, give Terminal (or your IDE) microphone/audio
access in **System Settings → Privacy & Security → Microphone**.

Linux
~~~~~

Install Python, git, and the PortAudio system library that ``sounddevice``
wraps:

.. code-block:: sh

   # Debian / Ubuntu
   sudo apt install python3 python3-venv git libportaudio2

   # Fedora
   sudo dnf install python3 python3-virtualenv git portaudio

   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade git+https://github.com/gclef-cmu/pyquist.git

Windows
~~~~~~~

Install `Python <https://www.python.org/downloads/>`_ (3.10 or later, with
"Add Python to PATH" checked) and `Git for Windows
<https://git-scm.com/download/win>`_, then in **Command Prompt**:

.. code-block:: bat

   python -m venv .venv
   .venv\Scripts\activate.bat
   pip install --upgrade git+https://github.com/gclef-cmu/pyquist.git


Key API
-------

Most users only need a handful of symbols. Click through for details.

* :class:`pyquist.Audio` — wrap a numpy array of samples as audio
* :func:`pyquist.play` — playback (auto-detects notebook vs. sounddevice)
* :func:`pyquist.plot`, :func:`pyquist.plot_freq`, :func:`pyquist.plot_spec` — visualization
* :class:`pyquist.score.Score`, :class:`pyquist.score.Event`,
  :class:`pyquist.score.BasicMetronome` — onset-based scores and rendering


All modules
-----------

.. toctree::
   :maxdepth: 2

   api/audio
   api/score
   api/plot
   api/device
   api/helper
   api/web


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
