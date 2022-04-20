.. _pandas: https://pandas.pydata.org

.. _scikit-learn: https://scikit-learn.org

.. _vectorbt: https://vectorbt.pro

.. _`scikit learn classifiers`: https://scikit-learn.org/stable/glossary.html#class-apis-and-estimator-types

.. _`dummy classifier`: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

.. _introduction: 

############
Introduction
############

The goal of the project is to provide various tools to extract sports
betting data and create predictive models. It integrates with 
other well-known Python libraries like pandas_, scikit-learn_
and vectorbt_.

.. _datasets:

********
Datasets
********

Sports betting datasets usually come in a format not suitable for modelling.
The dataloader objects deal with this issue by providing methods to extract 
the data in a cosistent format that makes it easy to create predictive models. 
Currently, there are various dataloaders available and `sports-betting` 
aims to include more in the future, covering multiple sports betting markets. 
For every dataloader, the extracted data are either training data or fixtures 
data, returned as tuple ``(X, Y, O)`` where ``X`` is the input data, 
``Y`` is the output data (equal to ``None`` for fixtures) and ``O`` is 
the odds data. Therefore, they are extracted in a suitable format for 
modelling and they are compatible to each other i.e. the training and fixtures
data have the same features. Specifically, the methods of the dataloaders 
that extract the training and fixtures data are the
:func:`~sportsbet.datasets._base._BaseDataLoader.extract_train_data` and 
:func:`~sportsbet.datasets._base._BaseDataLoader.extract_fixtures_data`,
respectively.

As an example, we initialize a dataloader with soccer dummy data::
    
    >>> from sportsbet.datasets import DummySoccerDataLoader
    >>> dataloader = DummySoccerDataLoader()

Then we extract the training data using the 
:func:`~sportsbet.datasets.DummySoccerDataLoader.extract_train_data` method, selecting
odds data for the Interwetten bookmaker::

    >>> X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='interwetten')

The fixtures data are extracted using the 
:func:`~sportsbet.datasets.DummySoccerDataLoader.extract_fixtures_data` method::

    >>> X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()

A detailed description of the above data is provided below.

**X_train**

The input training data ``X_train`` is the first component of the data 
tuple ``(X_train, Y_train, O_train)``::

    >>> print(X_train)  # doctest: +NORMALIZE_WHITESPACE
                division   league  year      home_team  ...  odds__williamhill__away_win__full_time_goals
    date
    1997-05-04         1    Spain  1997    Real Madrid  ...                                           NaN
    1998-03-04         3  England  1998      Liverpool  ...                                           NaN
    ...

``X_train`` is a :class:`~pandas.DataFrame` that contains information known before
the start of the betting event like the date, the names of the opponents, indices 
related to the strength of the opponents etc. It may also include odds data as 
shown above. The index of ``X_train`` is a :class:`~pandas.DatetimeIndex` and the data 
are always sorted by date. For various reasons, ``X_train`` may contain missing values. 
The ``drop_na_thres`` parameter of the 
:func:`~sportsbet.datasets._base._BaseDataLoader.extract_train_data`  method, adjusts 
the tolerance level i.e. higher values drop more columns of ``X_train`` that 
contain missing values. 

**Y_train**

The output training data ``Y_train`` is the second component of the data 
tuple ``(X_train, Y_train, O_train)``::

    >>> print(Y_train)
       output__home_win__full_time_goals  output__draw__full_time_goals  output__away_win__full_time_goals
    0                               True                          False                              False
    1                              False                          False                               True
    ...

``Y_train`` is a :class:`~pandas.DataFrame` that contains information
known after the end of the betting event like goals or points
scored, fouls commited etc. Column names follow a naming convention 
of the form ``f'output__{betting_market}__{target}'``. The ``betting_market`` 
parameter is any supported betting market like home win, over 2.5, draw and home 
points while the ``target`` parameter is the outcome that was used to extract the 
targets like ``'full_time_goals'``, ``'half_time_goals'`` and ``'full_time_points'``.
The entries of ``Y_train`` show whether or not an outcome of a betting event is 
``True`` or ``False``. In order to make the data suitable for modelling, ``Y_train``
does not contain any missing values i.e. rows of raw data that contain any missing 
values are removed. This last step also includes ``X_train`` and ``O_train``. Their
corresponding rows are removed to match ``Y_train``.

**O_train**

The odds training data ``O_train`` is the last component of the data 
tuple ``(X_train, Y_train, O_train)``::

    >>> print(O_train)
       odds__interwetten__home_win__full_time_goals  odds__interwetten__draw__full_time_goals  odds__interwetten__away_win__full_time_goals
    0                                           1.5                                       3.5                                           2.5
    1                                           2.0                                       4.5                                           3.5
    ...

``O_train`` is a :class:`~pandas.DataFrame` that contains information related 
to the odds for various betting markets. Column names follow a naming convention 
of the form ``f'odds__{bookmaker}__{betting_market}__{target}'``. The ``bookmaker`` 
parameter is any supported bookmaker or aggregation of bookmakers like ``'pinnacle'``', 
``'bet365'`` and ``'market_maximum'`` as returned by the class method 
:func:`~sportsbet.datasets._base._BaseDataLoader.get_odds_types`. 
The ``betting_market`` and ``target`` parameters are similar to the ones appearing to 
the columns of ``Y_train``. The entries of ``O_train`` are the odd values of 
betting events and, depending on the data source, it may contain missing values. 
``Y_train`` and ``O_train`` columns match, i.e. ``Y_train`` and ``O_train`` have the 
same shape and ``f'output__{betting_market}__{target}'`` column of ``Y_train`` is at the 
same position as the ``f'odds__{bookmaker}__{betting_market}__{target}'`` column of ``O_train``. 
The correspondence is clear in the examples above.

**X_fix**

The input fixtures data ``X_fix`` is the first component of the data 
tuple ``(X_fix, Y_fix, O_fix)``::

    >>> print(X_fix) # doctest: +NORMALIZE_WHITESPACE
                                division  league  year  home_team  ...  odds__williamhill__away_win__full_time_goals
    date
    2022...                            4     NaN  2022  Barcelona  ...                                           2.0
    2022...                            3  France  2022     Monaco  ...                                           2.5

``X_fix`` is a :class:`~pandas.DataFrame` that contains information known before
the start of the betting event. The features of ``X_fix`` are identical to the features
of ``X_train``. ``X_fix`` is not affected by the initialization parameter ``param_grid``
of the dataloader i.e. it contains the latest fixtures for every league, division or
any other parameter, even if they are not included in the training data.

**Y_fix**

``Y_fix`` is always equal to ``None`` since the output of betting events for fixtures
data is not known::

    >>> Y_fix is None
    True

**O_fix**

The odds fixtures data ``O_fix`` is the last component of the data 
tuple ``(X_fix, Y_fix, O_fix)``::

    >>> print(O_fix)
       odds__interwetten__home_win__full_time_goals  odds__interwetten__draw__full_time_goals  odds__interwetten__away_win__full_time_goals
    0                                           3.0                                       2.5                                           2.0
    1                                           1.5                                       3.5                                           2.5

``O_fix`` is a :class:`~pandas.DataFrame` that contains information related 
to the odds for various betting markets. The features of ``O_fix`` are identical 
to the features of ``O_train``.

**********
Evaluation
**********

The evaluation of models is made via the bettor objects. All bettors 
are `scikit learn classifiers`_, therefore they provide various methods, 
that can be used to fit the training data as well as evaluate their performance 
on test data. Specifically, bettors implement the 
:func:`~sportsbet.evaluation._base._BaseBettor.fit` method that fits the model 
to any input data ``X`` and multi-ouput targets ``Y``. The model can be based on a 
machine learning classifier but also rule-based models are supported. The 
bettors provide the :func:`~sportsbet.evaluation._base._BaseBettor.predict` and 
:func:`~sportsbet.evaluation._base._BaseBettor.predict_proba` methods that 
predict class labels and positive class probabilities, respectively. Additionally,
the betors provide the method :func:`~sportsbet.evaluation._base._BaseBettor.backtest`
that calculates various backtesting statistics, as well as the method 
:func:`~sportsbet.evaluation._base._BaseBettor.bet` that returns the value bets.

As an example, we initialize a classfier-based bettor that uses Scikit-Learn's
`dummy classifier`_::
    
    >>> from sklearn.dummy import DummyClassifier
    >>> from sportsbet.evaluation import ClassifierBettor
    >>> bettor = ClassifierBettor(classifier=DummyClassifier())

**Model fit**

The bettor is fitted to the training data ``(X_train, Y_train)`` via the
:func:`~sportsbet.evaluation._base._BaseBettor.fit` method. This fitting
procedure does not necessarily requires machine learning models but more
generally means that the bettor extracts information from ``(X_train, Y_train)``
that will be used when predictions are made. Fitting the model is very
simple::

    >>> bettor.fit(X_train, Y_train)
    ClassifierBettor(classifier=DummyClassifier())

**Model prediction**

Once the model is fitted, predicting class labels, i.e. ``True`` or ``False`` 
values of ``Y``, is straightforward::

    >>> bettor.predict(X_fix)
    array([[False, False, False],
           [False, False, False]])

Similarly, predicting positive class probabilities, i.e. the value ``True`` of ``Y`` 
is simple::

    >>> bettor.predict_proba(X_fix)
    array([[0.375, 0.25 , 0.375],
           [0.375, 0.25 , 0.375]])

**Backtest**

Backtesting the bettor requires the full data tuple ``(X_train, Y_train, O_train)``
to be used::

    >>> bettor.backtest(X_train, Y_train, O_train)
    ClassifierBettor(classifier=DummyClassifier())

The backtesting results include information of the various training/testing 
periods and metrics::
    
    >>> print(bettor.backtest_results_)
      Training Start Training End Training Period Testing Start Testing End Testing Period  Start Value  End Value ...
    0     1997-05-04   1998-03-04        304 days    1999-03-04  1999-03-04         1 days       1000.0     1002.5 ...
    1     1997-05-04   1999-03-04        669 days    2000-03-04  2000-03-04         1 days       1000.0      999.0 ...
    2     1997-05-04   2000-03-04       1035 days    2001-06-04  2001-06-04         1 days       1000.0      999.0 ...
    3     1997-05-04   2001-06-04       1492 days    2017-03-17  2017-03-17         1 days       1000.0     1000.0 ...
    4     1997-05-04   2017-03-17       7257 days    2019-03-17  2019-03-17         1 days       1000.0      999.0 ...
