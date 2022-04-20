.. _football-data.co.uk: https://www.football-data.co.uk/data.php
.. _fivethirtyeight: https://github.com/fivethirtyeight/data/tree/master/soccer-spi

*******************
Basic functionality
*******************

The `sports-betting` package provides a set of classes that help to
download sports betting data. Additionally, it includes tools 
to evaluate the performance of predictive models.

The dataloader objects and their methods can be used to download sports
betting data. For each sport, a combination of data sources 
is used and a dataloader class is provided. For example, the
:class:`~sportsbet.datasets.SoccerDataLoader` class can be used to 
download soccer historical and fixtures data from Football-Data.co.uk_ 
and FiveThirtyEight_::

   >>> from sportsbet.datasets import SoccerDataLoader
   >>> dataloader = SoccerDataLoader(param_grid={'league': ['Italy', 'Spain'], 'year': [2019, 2020]})
   >>> X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='pinnacle')
   >>> X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()

Bettor objects like :class:`~sportsbet.evaluation.ClassifierBettor`
provide an easy way to backtest a model and get the value bets::

   >>> from sportsbet.evaluation import ClassifierBettor
   >>> from sklearn.dummy import DummyClassifier
   >>> bettor = ClassifierBettor(DummyClassifier())
   >>> bettor.backtest(X_train, Y_train, O_train)
   ClassifierBettor(classifier=DummyClassifier())
   >>> value_bets = bettor.bet(X_fix, O_fix)

More information about dataloaders and bettors is provided below.

**Dataloaders**

A dataloader class corresponds to a combination of data sources 
for a specific sport. Their methods return datasets suitable for machine 
learning modelling. There various dataloaders available while all of 
them inherit from the same base class, thus providing the same public API.

A dataloader is initialized with the parameter ``param_grid`` that selects the 
training data to extract. This the same parameter as the initialization parameter
of :class:`~sklearn.model_selection.ParameterGrid`. 

You can get the available parameters and their values 
from the class method :func:`~sportsbet.datasets._base._BaseDataLoader.get_all_params`. 
For example, the available parameters for the 
:class:`~sportsbet.datasets.DummySoccerDataLoader` are the following::

   >>> from sportsbet.datasets import DummySoccerDataLoader
   >>> DummySoccerDataLoader.get_all_params()
   [{'division': 1, 'year': 1998}, ...]

The default value of ``param_grid`` is ``None`` and corresponds to the selection 
of all training data. In the following example, we select only the training data of 
the Spanish end English leagues for all available divisions and years::

   >>> dataloader = DummySoccerDataLoader(param_grid={'league': ['Spain', 'England']})

We can then extract the training or fixtures data.

**Bettors**

The various bettor classes provide an easy way to evaluate the
performance of models. They inherit from the same base class for bettor that
defines a general  and flexible interface. Therefore any type of a bettor can 
be implemented like classifier-based or rule-based bettors.

For example a classifier-based can be initialized with any Scikit-Learn's classifier::

   >>> from sklearn.neighbors import KNeighborsClassifier
   >>> bettor = ClassifierBettor(KNeighborsClassifier())

A minimum requirement is that a bettor should estimate the probabilities of sports
betting events. Thus all bettors, even rule-based, are also classifiers and they are
compatible with the Scikit-Learn's `classifier 
<https://scikit-learn.org/stable/glossary.html#class-apis-and-estimator-types>`_ interface.
Specifically, they provide the :func:`~sportsbet.evaluation._BaseBettor.fit`,
:func:`~sportsbet.evaluation._BaseBettor.predict` and :func:`~sportsbet.evaluation._BaseBettor.score`
methods. Additionally, bettors provide the :func:`~sportsbet.evaluation._BaseBettor.backtest` and 
:func:`~sportsbet.evaluation._BaseBettor.bet` methods. The first calculates various
backtesting statistics, while the second estimates the value bets.
