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
URL = 'https://github.com/AlgoWit/sports-betting'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/AlgoWit/sports-betting'
VERSION = __version__
INSTALL_REQUIRES = ['scipy>=0.17', 'numpy>=1.1', 'pandas==0.24.2', 'scikit-learn>=0.21', 'imbalanced-learn>=0.4.3', 'joblib==0.13.2', 'tqdm==4.28.1']
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
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
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
ENTRY_POINTS = {
    'console_scripts': [
        'download=sportsbet.soccer.data:download',
        'backtest=sportsbet.soccer.optimization:backtest',
        'predict=sportsbet.soccer.optimization:predict'
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
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS
)