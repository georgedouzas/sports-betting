.. _introduction:

============
Introduction
============

.. _api_sportsbet:

API
---

The dataloader object included in `sports-betting` provides two main
methods to download training and fixtures data. More specifically:

It implements the ``extract_train_data`` method to download training data::

      X, Y, O = dataloader.extract_train_data()

it implements the ``extract_fixtures_data`` method to download fixtures::

      X, Y, O = dataloader.extract_fixtures_data()

