# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../ephysiopy/"))


project = "ephysiopy"
copyright = "2025, Robin Hayman"
author = "Robin Hayman"
release = "2.0.57"  # use ephysiopy.version


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
    "autoapi.extension",
    "sphinx_copybutton",
    "sphinx-prompt",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rhayman/ephysiopy",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ]
}
# -- Options for numpydoc ---------------------------
numpydoc_class_members_toctree = False

# -- Options for autoapi -------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../ephysiopy"]
autoapi_keep_files = True
autoapi_root = "api"
autoapi_member_order = "groupwise"
autoapi_ignore = [
    "*tests*",
    "*__about__*",
    # "*version*",
    # "*setup*",
    "*RENAME*",
    "*scripts*",
    # "*debug",
]

# -- Options for autodoc -------------------------------------------------------
autodoc_typehints = "description"  # uses type hints to render some documentation

# -- Options for myst_parser -----------------------------------------------------
myst_enable_extensions = ["colon_fence"]
