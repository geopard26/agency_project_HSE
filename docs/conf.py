import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Agency Project HSE"
author = "Ваша команда"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # для Google/NumPy docstrings
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"
html_static_path = ["_static"]
