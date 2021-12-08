.. -*- mode: rst -*-

.. _scikit-learn: http://scikit-learn.org/stable/

|CircleCI|_ |ReadTheDocs|_ |PythonVersion|_ |Pypi|_ |Conda|_

.. |CircleCI| image:: https://circleci.com/gh/georgedouzas/sports-betting/tree/master.svg?style=svg
.. _CircleCI: https://circleci.com/gh/georgedouzas/sports-betting/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sports-betting/badge/?version=latest
.. _ReadTheDocs: https://sports-betting.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/sports-betting.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/sports-betting.svg

.. |Pypi| image:: https://badge.fury.io/py/sports-betting.svg
.. _Pypi: https://badge.fury.io/py/sports-betting

.. |Conda| image:: https://anaconda.org/algowit/sports-betting/badges/installer/conda.svg
.. _Conda: https://conda.anaconda.org/algowit

==============
sports-betting
==============

sports-betting is a collection of tools that makes it easy to create machine 
learning models for sports betting and evaluate their performance. 
It is compatible with scikit-learn_.

Documentation
-------------

Installation documentation, API documentation, and examples can be found on the
documentation_.

.. _documentation: https://sports-betting.readthedocs.io/en/latest/

Dependencies
------------

sports-betting is tested to work under Python 3.6+. The dependencies are the
following:

- pandas(>=1.1.0)
- rich(>=4.28)

Installation
------------

sports-betting is currently available on the PyPi's repository and you can
install it via `pip`::

  pip install -U sports-betting

The package is released also in Anaconda Cloud platform::

  conda install -c algowit sports-betting

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies::

  git clone https://github.com/AlgoWit/sports-betting.git
  cd sports-betting
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/AlgoWit/sports-betting.git

Testing
-------

After installation, you can use `pytest` to run the test suite::

  make test
