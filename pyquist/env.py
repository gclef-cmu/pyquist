"""Detects the runtime environment pyquist is running in.

These checks decide how audio can reach the speakers: an inline player in a
notebook, the Web Audio API in a browser, or :mod:`sounddevice` otherwise.
"""

import sys


def in_ipython_notebook() -> bool:
    """Returns True if running inside a notebook-style IPython kernel.

    Anything that isn't plain Python or the IPython terminal REPL counts —
    this covers Jupyter (``ZMQInteractiveShell``), Google Colab
    (``google.colab._shell.Shell``), VSCode notebooks, and other kernel
    hosts where inline display is the right playback path.
    """
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    ipy = get_ipython()
    if ipy is None:
        return False
    # The only IPython context where inline display ISN'T the right answer
    # is the terminal REPL, which has its own audio output.
    return ipy.__class__.__name__ != "TerminalInteractiveShell"


def in_browser() -> bool:
    """Returns True if running inside a browser-hosted Python runtime.

    Pyodide (and JupyterLite, which runs a Pyodide kernel) compiles
    CPython to WebAssembly via Emscripten, so ``sys.platform`` reports
    ``"emscripten"``. In these environments there is no OS audio device, so
    capture and playback must go through the Web Audio API rather than
    :mod:`sounddevice`.
    """
    return sys.platform == "emscripten"


def in_colab() -> bool:
    """Returns True if running inside Google Colab.

    Colab runs on a remote machine with no OS audio devices, so
    :class:`~pyquist.Stream` and sounddevice playback are unavailable there.
    Colab's kernel imports ``google.colab`` at startup.
    """
    return "google.colab" in sys.modules
