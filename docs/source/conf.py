# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Video Detection BP'
copyright = '2025, David Skalka'
author = 'David Skalka'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []
autodoc_mock_imports = ['database', 'interface', 'constants', 'detectors', 
                        'parsers', 'processors', 'profiling_utils', 'xclip',
                        'siglip', 'vector_database']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# Add extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

# Theme settings
html_theme = 'sphinx_rtd_theme'