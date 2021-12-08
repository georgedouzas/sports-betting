.. _scikit-learn: http://scikit-learn.org/stable/

.. _sportsbet:

==============
Sports betting
==============

.. currentmodule:: sportsbet.datasets

A practical guide
-----------------

sports-betting is a collection of tools that can be used to download sports 
betting data and evaluate the performance of machine learning models. 
It is compatible with scikit-learn_. It is straightforward to download training 
data suitable for machine learning modelling::

   >>> from sportsbet.datasets import DummyDataLoader
   >>> dataloader = DummyDataLoader(param_grid={'league': ['Spain', 'England']})
   >>> X_train, Y_train, _ = dataloader.extract_train_data()

The corresponding fixtures data is also easy to download:

   >>> X_fix, *_ = dataloader.extract_fixtures_data()

The data can be used to train machine learning models and make predictions on fixtures:

   >>> from sklearn.tree import DecisionTreeClassifier
   >>> dt = DecisionTreeClassifier(random_state=0)
   >>> train_cols = ['interwetten__home_win__odds', 'interwetten__draw__odds']
   >>> dt.fit(X_train[train_cols], Y_train)
   DecisionTreeClassifier(random_state=0)
   >>> dt.predict(X_fix[train_cols])
   array([[False, False,  True,  True, False],
          [False, False,  True,  True, False]])