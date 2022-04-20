*********************************
Extraction of sports betting data
*********************************

**Training data**

You can extract the training data using the method
:func:`~sportsbet.datasets._base.extract_train_data` that accepts the parameters
``drop_na_thres`` and ``odds_type``. The training data is a tuple of the input 
matrix ``X_train``, the multi-output targets ``Y_train`` and the odds matrix ``O_train``.

Tha parameter ``drop_na_thres`` adjusts the threshold of a column with 
missing values to be removed from the input matrix ``X_train``. It takes values in 
the range :math:`$[0.0, 1.0]$`. This parameter is included for convenience since historical 
data often come with columns that have a large number of missing value, therefore their 
presence does not enhance the predictive power of models. The value ``0.0`` keeps all 
columns, while ``1.0`` removes any column with missing values. 

The parameter ``odds_type`` selects the type of odds that will be used for the odds matrix ``O_train``. 
It also affects the columns of the multi-output targets ``Y_train`` since there is a match between 
``Y_train`` and ``Odds_train`` columns as explained in the :ref:`datatasets <datasets>` 
section of the  :ref:`introduction <introduction>`. You can get the available odds types from the
method :func:`~sportsbet.datasets._BaseDataLoader.get_odds_types`:

   >>> from sportsbet.datasets import DummySoccerDataLoader
   >>> dataloader = DummySoccerDataLoader()
   >>> dataloader.get_odds_types()
   ['interwetten', 'williamhill']

We can extract the training data using the default values of ``drop_na_thres`` and ``odds_type``
which are ``None`` for both of them::
   
   >>> X_train, Y_train, O_train = dataloader.extract_train_data()

No columns are dropped from the input matrix ``X_train``::

   >>> print(X_train) # doctest: +NORMALIZE_WHITESPACE
               division   league  year    home_team    away_team  ...  odds__williamhill__away_win__full_time_goals
   date
   1997-05-04         1    Spain  1997  Real Madrid    Barcelona  ...                                           NaN
   1998-03-04         3  England  1998    Liverpool      Arsenal  ...                                           NaN
   1999-03-04         2    Spain  1999    Barcelona  Real Madrid  ...                                           NaN
   ...

The multi-output targets matrix ``Y_train`` is the following::

   >>> print(Y_train)
      output__home_win__full_time_goals  output__away_win__full_time_goals  output__draw__full_time_goals  output__over_2.5__full_time_goals  output__under_2.5__full_time_goals
   0                               True                              False                          False                               True                               False
   1                              False                               True                          False                               True                               False
   2                              False                               True                          False                               True                               False
   3                              False                              False                           True                               True                               False
   ...

No odds matrix is returned:

   >>> O_train is None
   True

Instead, if we may extract the training data using specific values of ``drop_na_thres`` and ``odds_type``::
   
   >>> X_train, Y_train, O_train = dataloader.extract_train_data(drop_na_thres=1.0, odds_type='williamhill')

Columns that contain missing values are dropped from the input matrix ``X_train``::

   >>> print(X_train) # doctest: +NORMALIZE_WHITESPACE
                division  year      home_team      away_team  odds__interwetten__draw__full_time_goals  odds__interwetten__away_win__full_time_goals  odds__williamhill__home_win__full_time_goals
   date                                                                                                                                       
   1997-05-04          1  1997    Real Madrid      Barcelona                                       3.5                                           2.5                                           2.5
   1998-03-04          3  1998      Liverpool        Arsenal                                       4.5                                           3.5                                           2.0
   1998-03-04          1  1998      Liverpool        Arsenal                                       2.5                                           3.5                                           4.0
   ...

The multi-output targets ``Y_train`` is the following matrix::

   >>> print(Y_train)
      output__home_win__full_time_goals  output__draw__full_time_goals  output__away_win__full_time_goals
   0                               True                          False                              False
   1                              False                          False                               True
   2                              False                          False                               True
   3                              False                           True                              False
   ...

The odds data are the following:

   >>> print(O_train)
      odds__williamhill__home_win__full_time_goals  odds__williamhill__draw__full_time_goals  odds__williamhill__away_win__full_time_goals
   0                                           2.5                                       2.5                                           NaN
   1                                           2.0                                       NaN                                           NaN
   2                                           4.0                                       NaN                                           NaN
   3                                           2.0                                       NaN                                           NaN
   4                                           2.5                                       2.5                                           3.0
   ...
   
**Fixtures data**

Once the training data are extracted, it is straightforward to extract 
the corresponding fixtures data using the method
:func:`~sportsbet.datasets._BaseDataLoader.extract_fixtures_data`:

   >>> X_fix, Y_fix, O_fix = dataloader.extract_fixtures_data()

The method accepts no parameters and the extracted fixtures input matrix has 
the same columns as the latest extracted input matrix for the training data::

   >>> print(X_fix) # doctest: +NORMALIZE_WHITESPACE
                               division  year  ...  odds__williamhill__home_win__full_time_goals
   date                                                                                                                                                                                      
   ...                                4  2022  ...                                           3.5
   ...                                3  2022  ...                                           2.5

The odds matrix is the following::

   >>> print(O_fix)
      odds__williamhill__home_win__full_time_goals  odds__williamhill__draw__full_time_goals  odds__williamhill__away_win__full_time_goals
   0                                           3.5                                       2.5                                           2.0
   1                                           2.5                                       1.5                                           2.5

Since we are extracting the fixtures data, there is no target matrix::

   >>> Y_fix is None
   True
