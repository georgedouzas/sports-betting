*****************************
Creation of predictive models
*****************************

**Backtesting**

The extracted training data ``(X_train, Y_train, O_train)`` can be used
to backtest the performance of predictive models that are represented 
by bettor objects. Calculating various backtesting statistics on 
``(X_train, Y_train, O_train)`` is straightforward::

    >>> from sportsbet.datasets import DummySoccerDataLoader
    >>> from sportsbet.evaluation import ClassifierBettor
    >>> from sklearn.dummy import DummyClassifier
    >>> dataloader = DummySoccerDataLoader()
    >>> X_train, Y_train, O_train = dataloader.extract_train_data(odds_type='interwetten')
    >>> bettor = ClassifierBettor(DummyClassifier())
    >>> bettor.backtest(X_train, Y_train, O_train)
    ClassifierBettor(classifier=DummyClassifier())
    >>> print(bettor.backtest_results_) # doctest: +NORMALIZE_WHITESPACE
      Training Start Training End Training Period Testing Start Testing End Testing Period  Start Value  End Value  Total Return [%]  ...
    0     1997-05-04   1998-03-04        304 days    1999-03-04  1999-03-04         1 days       1000.0     1002.5              0.25  ...
    ...

**Value bets**

The extracted fixtures data can be used to predict the value bets::

    >>> X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()
    >>> bettor.bet(X_fix, O_fix)
       odds__interwetten__home_win__full_time_goals  odds__interwetten__draw__full_time_goals  odds__interwetten__away_win__full_time_goals
    0                                          True                                     False                                         False
    1                                         False                                     False                                         False