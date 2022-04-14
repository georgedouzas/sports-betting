.. -*- mode: rst -*-

.. _scikit-learn: http://scikit-learn.org/stable/

.. _documentation: https://sports-betting.readthedocs.io/en/latest/

|ReadTheDocs|_ |PythonVersion|_ |Pypi|_ |Black|_

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sports-betting/badge/?version=latest
.. _ReadTheDocs: https://sports-betting.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/sports-betting.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/sports-betting.svg

.. |Pypi| image:: https://badge.fury.io/py/sports-betting.svg
.. _Pypi: https://badge.fury.io/py/sports-betting

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: :target: https://github.com/psf/black

##############
sports-betting
##############

************
Introduction
************

The `sports-betting` package is a collection of tools that makes it easy to 
create machine learning models for sports betting and evaluate their performance. 
It is compatible with scikit-learn_.

*****
Usage
*****

The `sports-betting` package makes it easy to download 
training and fixtures sports betting data::

  >>> from sportsbet.datasets import SoccerDataLoader
  >>> dataloader = SoccerDataLoader(param_grid={'league': ['Italy'], 'year': [2020]})
  >>> X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='market_maximum', drop_na_thres=1.0)
  >>> X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()

The historical data can be used to backtest the performance of a bettor model::

  >>> from sportsbet.evaluation import ClassifierBettor
  >>> from sklearn.dummy import DummyClassifier
  >>> bettor = ClassifierBettor(DummyClassifier())
  >>> bettor.backtest(X_train, Y_train, O_train)

We can get the value bets using fixtures data::

  >>> bettor.bet(X_fix, O_fix)

************
Installation
************

`sports-betting` is currently available on the PyPi's repositories and you can
install it via `pip`::

  pip install -U sports-betting

*************
Documentation
*************

Installation documentation, API documentation, and examples can be found in the
documentation_.
