"""Configuration file for mkdocs-gallery plugin."""

import plotly.io as pio

pio.renderers.default = 'sphinx_gallery'

conf = {'capture_repr': ('__repr__', '_repr_html_')}
