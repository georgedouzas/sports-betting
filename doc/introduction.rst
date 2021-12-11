.. _Football-Data.co.uk: http://www.football-data.co.uk/data.php

.. _FiveThirtyEight: https://github.com/fivethirtyeight/data/tree/master/soccer-spi

============
Introduction
============

.. currentmodule:: sportsbet.datasets

The `sports-betting` package provides a set of classes that help to download 
sports betting data. Additionally, it includes a backtesting engine to
evaluate the performance of machine learning models.

The dataloader objects and their methods can be used to download sports 
betting data. For each data source or combination of data sources a 
corresponding dataloader class is provided. For example, the 
:class:`FTESoccerDataLoader` class can be used to download soccer historical data 
and fixtures from FiveThirtyEight_::

      from sportsbet.datasets import FTESoccerDataLoader
      dataloader = FTESoccerDataLoader()
      X_train, Y_train, Odds_train = dataloader.extract_train_data()
      X_fix, Y_fix, Odds_fix = dataloader.extract_fixtures_data()

Similarly the, :class:`FDSoccerDataLoader` class can be used to download 
soccer historical data and fixtures from Football-Data.co.uk_::

      from sportsbet.datasets import FDSoccerDataLoader
      dataloader = FDSoccerDataLoader(param_grid={'league': ['Italy', 'Spain'], 'year': [2019, 2020]})
      X_train, Y_train, Odds_train = dataloader.extract_train_data(odds_type='pinnacle')
      X_fix, Y_fix, Odds_fix = dataloader.extract_fixtures_data()

The backtesting part is still under active development and it will be available soon.
