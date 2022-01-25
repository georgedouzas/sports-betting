# -*- coding: utf-8 -*-

import sys
import os

import sphinx_rtd_theme
import sphinx
from distutils.version import LooseVersion


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]
if LooseVersion(sphinx.__version__) < LooseVersion('1.4'):
    extensions.append('sphinx.ext.pngmath')
else:
    extensions.append('sphinx.ext.imgmath')
numpydoc_show_class_members = False
autodoc_default_options = {
    'members': None,
    'inherited-members': None,
    'member-order': 'bysource',
}
templates_path = ['_templates']
autosummary_generate = True
source_suffix = '.rst'
plot_gallery = True
master_doc = 'index'

# General information about the project.
project = u'sports-betting'
copyright = u'2019, Georgios Douzas'
from sportsbet import __version__

version = __version__
release = __version__
exclude_patterns = ['_build', '_templates']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Custom style
html_style = 'css/sports-betting.css'
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
htmlhelp_basename = 'sports-bettingdoc'

# Latex
latex_elements = {}
latex_documents = [
    (
        'index',
        'sports-betting.tex',
        u'sports-betting Documentation',
        u'Georgios Douzas',
        'manual',
    ),
]


man_pages = [
    (
        'index',
        'sports-betting.tex',
        u'sports-betting Documentation',
        [u'Georgios Douzas'],
        1,
    )
]
texinfo_documents = [
    (
        'index',
        'sports-betting',
        u'sports-betting Documentation',
        u'Georgios Douzas',
        'sports-betting',
        'Sports betting toolbox.',
        'Miscellaneous',
    ),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# sphinx-gallery configuration
sphinx_gallery_conf = {
    'doc_module': 'sportsbet',
    'backreferences_dir': os.path.join('generated'),
    'reference_url': {'sportsbet': None},
    'filename_pattern': '/*',
}


def setup(app):
    app.add_js_file('js/copybutton.js')
