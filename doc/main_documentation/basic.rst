.. _fivethirtyeight: https://github.com/fivethirtyeight/data/tree/master/soccer-spi

.. _football-data.co.uk: http://www.football-data.co.uk/data.php

############
 Basic Usage
############

The `sports-betting` package provides a set of classes that help to
download sports betting data. Additionally, it includes a backtesting
engine to evaluate the performance of machine learning models.

The dataloader objects and their methods can be used to download sports
betting data. For each data source or combination of data sources a
corresponding dataloader class is provided. For example, the
:class:`~sportsbet.datasets.FTESoccerDataLoader` class can be used to
download soccer historical data and fixtures from FiveThirtyEight_::

   from sportsbet.datasets import FTESoccerDataLoader
   dataloader = FTESoccerDataLoader()
   X_train, Y_train, O_train = dataloader.extract_train_data()
   X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()

Similarly the, :class:`FDSoccerDataLoader` class can be used to download
soccer historical data and fixtures from Football-Data.co.uk_::

   from sportsbet.datasets import FDSoccerDataLoader
   dataloader = FDSoccerDataLoader(param_grid={'league': ['Italy', 'Spain'], 'year': [2019, 2020]})
   X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='pinnacle')
   X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()

There is also a dataloader that combines the above two sources::

   from sportsbet.datasets import SoccerDataLoader
   dataloader = SoccerDataLoader(param_grid={'league': ['France'], 'year': [2019, 2020]})
   X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='pinnacle')
   X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()

Bettor classes like :class:`~sportsbet.evaluation.ClassifierBettor`
provide an easy way to backtest a model and get the value bets::

   from sportsbet.evaluation import ClassifierBettor
   num_features = [
      col
      for col in X_train.columns
      if X_train[col].dtype in (np.dtype(int), np.dtype(float))
   ]
   X_train = X_train[num_features]
   bettor = ClassifierBettor(KNeighborsClassifier())
   bettor.backtest(X_train, Y_train, O_train)
   value_bets = bettor.bet(X_fix[num_features], O_fix)