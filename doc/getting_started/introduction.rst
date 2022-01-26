.. _introduction:

#############
 Introduction
#############

The goal of the project is to provide simple tools to extract sports
betting data and create backtesting strategies. It integrates with 
other well-known Python libraries like `pandas
<https://pandas.pydata.org/>`_, `scikit-learn
<https://scikit-learn.org/stable/>`_ and `vectorbt
<https://vectorbt.dev/>`_.

****************
 Data extraction
****************

The data extraction is based on the dataloader objects.
Currently, there are various dataloaders available and `sports-betting`
aims to include more in the future. The data are extracted in a suitable
format for modelling. Particularly, the data are always returned in a
data tuple ``(X, Y, O)``.

Input data
==========

The input data ``X`` are the first commponent of the data tuple. ``X``
is a :class:`~pandas.DataFrame` that contains information known before
the start of the betting event like the date, the names of the
opponents, indices related to the strength of the opponents etc. It may
also include odds data. The index of ``X`` is a :class:`~pandas.DatetimeIndex`
and the data are always sorted by date.

Multi-output targets
====================

The multi-output targets ``Y`` are the second component of the data
tuple. ``Y`` is a :class:`~pandas.DataFrame` that contains information
known after the end of the betting event like goals or points
scored, fouls commited etc. Column names follow a naming convention 
of the form ``'betting_market__key'``. Some examples are 
``'home_win__full_time_goals'``, ``'over_2.5__full_time_goals'`` and
``'draw__half_time_goals'``. More generally, ``'betting_market'`` prefix
is any supported betting market like home win, over 2.5, draw and home points
while ``'key'`` postfix is the outcome that was used to extract the targets like
``'full_time_goals'``, ``'half_time_goals'`` and ``'full_time_points'``.

Odds data
=========

The odds data ``O`` are the third component of the data tuple. ``O`` is a 
:class:`~pandas.DataFrame` that contains information related to the odds for 
various betting markets. Column names follow a naming convention of the form 
``'bookmaker__betting_market__odds'``. Some examples are 
``'pinnacle__home_win__odds'``,  ``'market_average__over_2.5_goals__odds'`` and
``'bet365__over_2.5__half_time_goals'``. More generally, ``'bookmaker'`` prefix
is any supported bookmaker or aggregation of bookmakers like Pinnacle, Bet365 and 
market maximum, ``'betting_market'`` infix is similar to the one appearing to the 
columns of ``Y`` while ``'odds'`` postfix is always present to denote an odd column.


Data matching
=============
An effort is made to extract data suitable for modelling. Odds data 
are not always available but when they are extracted, then ``Y`` and
``O`` columns always match, i.e. ``Y`` and ``O`` have the same shape and 
``'betting_market__key'`` column of ``Y`` is at the same position as the
``'bookmaker__betting_market__odds'`` column of ``O``. For example if ``Y`` has
the columns ``['home_win__full_time_goals', 'pinnacle____full_time_goals']`` then 
``O`` may have the columns ``['pinnacle__home_win__odds', 'pinnacle__draw__odds']``. 

**********
Evaluation
**********

The evaluation of models is based on the bettor objects. All bettors 
are `classifiers <https://scikit-learn.org/stable/glossary.html#class-apis-and-estimator-types>`_, 
therefore they provide various methods that can be used to fit the training data and 
evaluate their performance on test data. Specifically, bettors implement the 
:func:`~sportsbet.evaluation._base._BaseBettor.fit` method that fits the model 
to the input data ``X`` and the multi-ouput targets ``Y``. The model can be a 
machine learning classifier but any other model is also supported. Also the 
bettors provide the :func:`~sportsbet.evaluation._base._BaseBettor.predict` and 
:func:`~sportsbet.evaluation._base._BaseBettor.predict_proba` methods that 
predict class labels and positive class probabilities, respectively. Additionally,
the betors provide the method :func:`~sportsbet.evaluation._base._BaseBettor.backtest`
that calculates various backtesting statistics, as well as the method 
:func:`~sportsbet.evaluation._base._BaseBettor.bet` that returns the value bets.
