"""Sphinx configuration for the pyquist API documentation.

Build with ``make html`` from this directory. See the project README for
setup instructions.
"""

import os
import sys

# Make the pyquist package importable so autodoc can introspect it.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Pyquist"
copyright = "2024-present, Chris Donahue, Ben Stoler"
author = "Chris Donahue, Ben Stoler"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # pull docstrings from the source
    "sphinx.ext.autosummary",  # generate per-module summary tables
    "sphinx.ext.napoleon",  # parse Google-style docstrings
    "sphinx.ext.viewcode",  # add "[source]" links
    "sphinx.ext.intersphinx",  # cross-link to Python / numpy docs
    "sphinx_autodoc_typehints",  # render type hints in arg descriptions
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mock heavy / system-library-dependent imports so the docs can be built on
# any machine (and in CI) without needing PortAudio, etc. None of these
# modules appear in pyquist's public signatures, so mocking them doesn't
# affect the rendered API.
autodoc_mock_imports = [
    "sounddevice",
    "soundfile",
    "resampy",
    "mido",
    "tqdm",
    "requests",
    "scipy",
    "matplotlib",
    "IPython",
]

# Cross-reference targets: clicking, e.g., `np.ndarray` in a signature lands
# on the right page in the upstream docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Autodoc options ---------------------------------------------------------

# Include both the class docstring and the __init__ docstring.
autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,  # skip members without docstrings
    "show-inheritance": True,
    "member-order": "bysource",  # match source order, not alphabetical
}

# Render type hints in argument descriptions (cleaner than inline in the
# signature when callables are involved).
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# -- HTML output -------------------------------------------------------------

html_theme = "alabaster"
