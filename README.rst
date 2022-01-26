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

.. |Conda| image:: https://anaconda.org/gdouzas/sports-betting/badges/installer/conda.svg
.. _Conda: https://conda.anaconda.org/gdouzas

##############
sports-betting
##############

************
Introduction
************

The sports-betting package is a collection of tools that makes it easy to 
create machine learning models for sports betting and evaluate their performance. 
It is compatible with scikit-learn_.

*****
Usage
*****

You can download sports betting data::

  from sportsbet.datasets import FTESoccerDataLoader
  dataloader = FTESoccerDataLoader()
  X_train, Y_train, O_train = dataloader.extract_train_data()

Use the historical data to backtest the performance of models::

  from sportsbet.evaluation import ClassifierBettor
  num_features = [
    col
    for col in X_train.columns
    if X_train[col].dtype in (np.dtype(int), np.dtype(float))
  ]
  X_train = X_train[num_features]
  bettor = ClassifierBettor(KNeighborsClassifier())
  bettor.backtest(X_train, Y_train, O_train)

Get the value bets using fixtures data::
  
  X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
  value_bets = bettor.bet(X_fix[num_features], O_fix)

************
Installation
************

`sports-betting` is currently available on the PyPi's repositories and you can
install it via `pip`::

  pip install -U sports-betting

The package is released also in Anaconda Cloud platform::

  conda install -c gdouzas sports-betting

*************
Documentation
*************

Installation documentation, API documentation, and examples can be found in the
documentation_.

.. _documentation: https://sports-betting.readthedocs.io/en/latest/
