import sys
import unittest
from unittest import mock

from . import env


class TestInColab(unittest.TestCase):
    def test_true_when_google_colab_imported(self):
        with mock.patch.dict("sys.modules", {"google.colab": mock.MagicMock()}):
            self.assertTrue(env.in_colab())

    def test_false_when_not_imported(self):
        with mock.patch.dict("sys.modules"):
            sys.modules.pop("google.colab", None)
            self.assertFalse(env.in_colab())


class TestInIPythonNotebook(unittest.TestCase):
    """``in_ipython_notebook`` must recognise every flavour of notebook
    kernel — Jupyter, Colab, etc. — so that ``play`` doesn't fall through
    to a sounddevice call that would crash in browser-hosted environments
    with no audio env."""

    def _fake_ipython_module(self, shell_class_name: str):
        """Returns a fake ``IPython`` module whose ``get_ipython()`` yields a
        shell of the given class name."""
        # A real class (not MagicMock) is needed so .__class__.__name__ works.
        shell_cls = type(shell_class_name, (), {})
        fake_ipython = mock.MagicMock()
        fake_ipython.get_ipython.return_value = shell_cls()
        return fake_ipython

    def test_jupyter_kernel_detected(self):
        with mock.patch.dict(
            "sys.modules", {"IPython": self._fake_ipython_module("ZMQInteractiveShell")}
        ):
            self.assertTrue(env.in_ipython_notebook())

    def test_colab_kernel_detected(self):
        # Google Colab's shell is `google.colab._shell.Shell`. The class name
        # alone is enough to distinguish from the terminal REPL.
        with mock.patch.dict(
            "sys.modules", {"IPython": self._fake_ipython_module("Shell")}
        ):
            self.assertTrue(env.in_ipython_notebook())

    def test_terminal_repl_not_detected(self):
        with mock.patch.dict(
            "sys.modules",
            {"IPython": self._fake_ipython_module("TerminalInteractiveShell")},
        ):
            self.assertFalse(env.in_ipython_notebook())

    def test_no_active_kernel_not_detected(self):
        fake_ipython = mock.MagicMock()
        fake_ipython.get_ipython.return_value = None
        with mock.patch.dict("sys.modules", {"IPython": fake_ipython}):
            self.assertFalse(env.in_ipython_notebook())

    def test_no_ipython_installed_not_detected(self):
        # Simulate IPython being missing entirely.
        with mock.patch.dict("sys.modules", {"IPython": None}):
            self.assertFalse(env.in_ipython_notebook())


class TestInBrowser(unittest.TestCase):
    """``in_browser`` must recognise browser-hosted runtimes (Pyodide /
    JupyterLite) so that ``record`` delegates to the Web Audio backend instead
    of calling sounddevice, which would crash where there is no audio env."""

    def test_emscripten_platform_detected(self):
        with mock.patch.object(env.sys, "platform", "emscripten"):
            self.assertTrue(env.in_browser())

    def test_native_platform_not_detected(self):
        for platform_name in ("darwin", "linux", "win32"):
            with mock.patch.object(env.sys, "platform", platform_name):
                self.assertFalse(env.in_browser())
