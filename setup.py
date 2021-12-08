#! /usr/bin/env python
"""Python sports betting toolbox."""

import codecs
import os

from setuptools import find_packages, setup

ver_file = os.path.join('sportsbet', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'sports-betting'
DESCRIPTION = 'Python sports betting toolbox.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'G. Douzas'
MAINTAINER_EMAIL = 'gdouzas@icloud.com'
URL = 'https://github.com/georgedouzas/sports-betting'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/AlgoWit/sports-betting'
VERSION = __version__
INSTALL_REQUIRES = ['pandas>=1.0.0', 'scikit-learn>=1.0.0', 'cloudpickle>=2.0.0', 'beautifulsoup4>=4.0.0', 'rich>=4.28']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx==1.8.5',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib',
        'pandas'
    ]
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE
)