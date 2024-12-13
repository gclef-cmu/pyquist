# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Pyquist"
copyright = "2024, Chris Donahue, Ben Stoler"
author = "Chris Donahue, Ben Stoler"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# Generate documentation for __init__ method of classes
autoclass_content = "both"

# If True, insert documentation from class docstring and
# the __init__ docstring into the class documentation.
# Useful if __init__ docstring describes attributes.

# Show members by default
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
}
