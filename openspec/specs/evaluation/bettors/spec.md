# evaluation/bettors Specification

## Purpose

A bettor turns predicted probabilities into value bets on the markets of the moment-aware data a dataloader extracts.
The library ships a base contract, a classifier bettor, an odds comparison bettor, a builder and save and load helpers.

## Requirements

### Requirement: Bettor contract
`BaseBettor(betting_markets=None, init_cash=None, stake=None)` SHALL be an abstract scikit-learn multi-output
classifier. A subclass MUST implement `_fit(X, Y, O)` and `_predict_proba(X)`. It SHALL then provide `fit`,
`predict_proba`, `predict`, `bet` and `score`, and work with `backtest` and `BettorGridSearchCV`. A subclass missing
either method SHALL raise `TypeError` when instantiated.

#### Scenario: An incomplete bettor
- **GIVEN** a subclass of `BaseBettor` that implements neither method
- **WHEN** the user instantiates it
- **THEN** a `TypeError` is raised whose message contains `Can't instantiate abstract class`

#### Scenario: A bettor of your own
- **GIVEN** a subclass whose `_fit` stores the mean of `Y` and whose `_predict_proba` returns those rates normalized
- **WHEN** the user fits it on the England sample with `betting_markets=['home_win', 'draw', 'away_win']` and calls
  `bet(X, O)`
- **THEN** the result has shape `(380, 3)`

### Requirement: Fit input validation
`fit(X, Y, O=None)` SHALL require `X` to be a DataFrame with a `DatetimeIndex` and `Y` to be a DataFrame, and SHALL
raise `TypeError` otherwise. Inputs of different lengths SHALL raise `ValueError`. Each `Y` column MUST follow
`{market}__{event_status}__{event_time}`, or `ValueError` SHALL be raised. When `O` is given, it SHALL pass the odds
validation, and its markets MUST equal the `Y` markets, or `ValueError` SHALL be raised.

#### Scenario: Input data without a date index
- **WHEN** the user calls `fit(pd.DataFrame(X.to_numpy()), Y)`
- **THEN** a `TypeError` is raised whose message contains `` Input data `X` ``

#### Scenario: Output data that is not a DataFrame
- **WHEN** the user calls `fit(X, Y.to_numpy())`
- **THEN** a `TypeError` is raised whose message contains `` Output data `Y` ``

#### Scenario: Inconsistent lengths
- **WHEN** the user calls `fit(X.iloc[:-1], Y)`
- **THEN** a `ValueError` is raised whose message contains `inconsistent numbers of samples`

#### Scenario: A target column without the grammar
- **GIVEN** `Y` with the column `home_win__postplay__0min` renamed to `home_win`
- **WHEN** the user calls `fit(X, Y)`
- **THEN** a `ValueError` is raised whose message contains `Output data column names`

### Requirement: Odds input validation
Wherever a bettor reads `O`, `O` MUST be a DataFrame, or `TypeError` SHALL be raised. Its length MUST match `X`, or
`ValueError` SHALL be raised. Each column MUST follow `{provider}__{market}__{event_status}__{event_time}`, or
`ValueError` SHALL be raised. All columns MUST share one provider, or `ValueError` SHALL be raised.

#### Scenario: Odds that are not a DataFrame
- **WHEN** the user calls `bet(X, O.to_numpy())` on a fitted bettor
- **THEN** a `TypeError` is raised whose message contains `` Odds data `O` ``

#### Scenario: An odds column without the grammar
- **WHEN** the user calls `bet` with an odds column renamed to `home_win__postplay__0min`
- **THEN** a `ValueError` is raised whose message contains `Odds data column names`

#### Scenario: Mixed providers
- **WHEN** the user calls `bet` with one odds column renamed to the provider `other`
- **THEN** a `ValueError` is raised whose message contains `Providers of odds data column names`

### Requirement: Betting markets
`betting_markets` SHALL select markets by base name, the part of a column before the first `__`. `None` SHALL select
every market of `Y` in column order. A value that is not a list, or an empty list, SHALL raise `TypeError` at fit. A
name absent from `Y` SHALL raise `ValueError` at fit. After fit, `betting_markets_` SHALL hold the selected names and
`feature_names_out_` the matching `Y` columns.

#### Scenario: Default markets
- **WHEN** the user fits a bettor with `betting_markets=None`
- **THEN** `betting_markets_` equals the base names of the `Y` columns

#### Scenario: A subset of markets
- **WHEN** the user fits a bettor with `betting_markets=['draw', 'away_win']`
- **THEN** `betting_markets_` equals `['draw', 'away_win']`
- **AND** `feature_names_out_` has two entries

#### Scenario: Markets that are not a list
- **WHEN** the user fits a bettor with `betting_markets='home_win'`
- **THEN** a `TypeError` is raised whose message contains `` Parameter `betting_markets` ``

#### Scenario: An unknown market
- **WHEN** the user fits a bettor with `betting_markets=['not_a_market']`
- **THEN** a `ValueError` is raised whose message contains `does not contain valid names`

### Requirement: Cash and stake
`init_cash` SHALL default to `10000.0` and `stake` to `50.0`. Each MUST be an `int` or a `float`, or `TypeError` SHALL
be raised at fit. Each MUST be greater than zero, or `ValueError` SHALL be raised at fit. After fit, `init_cash_` and
`stake_` SHALL hold the checked values as floats.

#### Scenario: Defaults
- **WHEN** the user fits a bettor with no `init_cash` and no `stake`
- **THEN** `init_cash_` is `10000.0` and `stake_` is `50.0`

#### Scenario: A non-positive stake
- **WHEN** the user fits a bettor with `stake=0.0`
- **THEN** a `ValueError` is raised whose message contains `stake == 0.0, must be > 0.0.`

#### Scenario: A stake that is not a number
- **WHEN** the user fits a bettor with `stake='stake'`
- **THEN** a `TypeError` is raised whose message contains `stake must be an instance of {float, int}`

### Requirement: Complementary events are derived from the markets
`derive_complementary_events(markets)` SHALL return the groups of mutually exclusive and exhaustive markets. The present
members of `home_win`, `draw` and `away_win` SHALL form one group when there are at least two. Each `over_{line}` and
`under_{line}` pair at the same line SHALL form a group. A lone market SHALL form no group. A bettor SHALL store the
groups of its `Y` markets as `complementary_events_`, unless its class sets `COMPLEMENTARY_EVENTS`.

#### Scenario: Soccer markets
- **WHEN** the user calls `derive_complementary_events(['home_win', 'draw', 'away_win', 'over_2.5', 'under_2.5'])`
- **THEN** it returns `[['home_win', 'draw', 'away_win'], ['over_2.5', 'under_2.5']]`

#### Scenario: A sport without a draw
- **WHEN** the user calls `derive_complementary_events(['home_win', 'away_win', 'over_220.5', 'under_220.5'])`
- **THEN** it returns `[['home_win', 'away_win'], ['over_220.5', 'under_220.5']]`

#### Scenario: A lone market
- **WHEN** the user calls `derive_complementary_events(['draw'])`
- **THEN** it returns `[]`

### Requirement: Market name helpers
`derive_market_base(market)` SHALL return the part of a column name before the first `__`.
`find_latest_odds_column(columns, base, provider=None)` SHALL return the odds column of market `base`, and of
`provider` when given, at the latest moment by status rank and then event time, or `None` when no column matches.

#### Scenario: The base of a target column
- **WHEN** the user calls `derive_market_base('over_2.5__inplay__60min')`
- **THEN** it returns `'over_2.5'`

#### Scenario: The latest odds column of a provider
- **GIVEN** the columns `bet365__home_win__preplay__0min`, `bet365__home_win__inplay__90min`,
  `bet365__home_win__inplay__30min` and `market_average__home_win__inplay__60min`
- **WHEN** the user calls `find_latest_odds_column(columns, 'home_win', provider='bet365')`
- **THEN** it returns `'bet365__home_win__inplay__90min'`
- **AND** `find_latest_odds_column(columns, 'draw')` returns `None`

### Requirement: Predicted probabilities
`predict_proba(X)` SHALL raise `NotFittedError` before fit. It SHALL check that `X` has the fitted feature names. It
SHALL return an array of shape `(len(X), number of selected markets)`, and an empty `X` SHALL give zero rows. A
`_predict_proba` result of another width SHALL raise `TypeError`. For each complementary group whose members are all
selected, the probabilities SHALL be scaled to sum to one.

#### Scenario: Normalized outcome probabilities
- **GIVEN** a `ClassifierBettor` fitted with `betting_markets=['home_win', 'draw', 'away_win']`
- **WHEN** the user calls `predict_proba(X)` on the 380 England matches
- **THEN** the result has shape `(380, 3)` and each row sums to one

#### Scenario: A line outside the soccer markets is normalized
- **GIVEN** a `ClassifierBettor` fitted on the markets `over_1.5` and `under_1.5`
- **WHEN** the user calls `predict_proba(X)`
- **THEN** each row sums to one

### Requirement: Predicted labels
`predict(X)` SHALL return a boolean array that is `True` where the predicted probability exceeds `0.5`.

#### Scenario: Label shape
- **GIVEN** a bettor fitted with three markets on 380 matches
- **WHEN** the user calls `predict(X)`
- **THEN** the result has shape `(380, 3)`

### Requirement: Value bets
`bet(X, O)` SHALL raise `NotFittedError` before fit. It SHALL use, per market, the latest odds column. A market SHALL
be a value bet where its probability times its odds exceeds one. Within a complementary group, a row SHALL keep at most
the bet with the highest expected return. It SHALL return a boolean array with one row per match and one column per
selected market. Odds lacking a selected market SHALL raise `ValueError`.

#### Scenario: Shape and type
- **GIVEN** a bettor fitted on `X` and `Y`
- **WHEN** the user calls `bet(X, O)`
- **THEN** the result has shape `(len(X), number of selected markets)` and dtype `bool`

#### Scenario: Bet before fit
- **WHEN** the user calls `bet(X, O)` on an unfitted bettor
- **THEN** a `NotFittedError` is raised

### Requirement: Classes
After fit, `classes_` SHALL be a list with one array `[0, 1]` per selected market. Before fit, reading `classes_`
SHALL raise `AttributeError`.

#### Scenario: Binary classes per market
- **GIVEN** a fitted `ClassifierBettor`
- **WHEN** the user reads `classes_`
- **THEN** it has one entry per selected market, each equal to `[0, 1]`

### Requirement: ClassifierBettor
`ClassifierBettor(classifier, betting_markets=None, init_cash=None, stake=None)` SHALL fit a clone of `classifier` on
`X` and the selected `Y` columns, store it as `classifier_`, and use its positive-class probabilities. A `classifier`
that is not a scikit-learn classifier SHALL raise `TypeError` at fit. The odds `O` SHALL be optional at fit.

#### Scenario: A regressor is refused
- **WHEN** the user calls `ClassifierBettor(DummyRegressor()).fit(X, Y)`
- **THEN** a `TypeError` is raised whose message contains `` `ClassifierBettor` requires a classifier ``

#### Scenario: The clone is fitted
- **WHEN** the user calls `ClassifierBettor(classifier=DummyClassifier()).fit(X, Y)`
- **THEN** `classifier_` is a fitted `DummyClassifier`
- **AND** the bettor has no `classifier_` before fit

#### Scenario: Fit, predict and bet on extracted data
- **GIVEN** a `ClassifierBettor(DummyClassifier(strategy='prior'))` fitted on training `X`, `Y` and `O`
- **WHEN** the user calls `bet(X_fix, O_fix)` on the fixtures
- **THEN** the result has one row per fixture, one column per market and dtype `bool`

### Requirement: OddsComparisonBettor
`OddsComparisonBettor(odds_types=None, alpha=0.05, betting_markets=None, init_cash=None, stake=None)` SHALL append `O`
to `X` and estimate each market's probability as one over the mean of the latest odds of `odds_types`, minus `alpha`,
clipped at zero, with missing values as zero. It MUST receive `O` at fit, or `ValueError` SHALL be raised with
`Input data do not include any odds columns.`

#### Scenario: Fit without odds
- **WHEN** the user calls `OddsComparisonBettor().fit(X, Y)`
- **THEN** a `ValueError` is raised whose message contains `Input data do not include any odds columns.`

#### Scenario: Any number of markets can be backtested
- **WHEN** the user backtests `OddsComparisonBettor(alpha=0.03, betting_markets=['home_win'])` on the England sample
- **THEN** the results are not empty

### Requirement: Odds types and alpha of OddsComparisonBettor
`odds_types` SHALL be `None` or a list of provider names present in the odds. `None` SHALL resolve to
`['market_average']` when that provider is present, and to every provider sorted otherwise. A value that is not a list
of strings SHALL raise `TypeError`, and an absent provider SHALL raise `ValueError`. `alpha` MUST be a `float`, or
`TypeError` SHALL be raised, and MUST lie in `[0.0, 1.0]`, or `ValueError` SHALL be raised.

#### Scenario: Default odds types
- **WHEN** the user fits `OddsComparisonBettor()` with `O` from the `market_average` provider
- **THEN** `odds_types_` is `['market_average']`

#### Scenario: Odds types that are not a list
- **WHEN** the user fits `OddsComparisonBettor(odds_types='market_average')` with `O`
- **THEN** a `TypeError` is raised whose message contains `` Parameter `odds_types` ``

#### Scenario: An absent provider
- **WHEN** the user fits `OddsComparisonBettor(odds_types=['williamhill'])` with `O`
- **THEN** a `ValueError` is raised whose message contains `` Parameter `odds_types` ``

#### Scenario: An integer alpha
- **WHEN** the user fits `OddsComparisonBettor(alpha=3)` with `O`
- **THEN** a `TypeError` is raised whose message contains `alpha must be an instance of float`

#### Scenario: An alpha outside the range
- **WHEN** the user fits `OddsComparisonBettor(alpha=1.4)` with `O`
- **THEN** a `ValueError` is raised whose message contains `alpha == 1.4, must be`

### Requirement: Build a bettor from text
`build_bettor(model)` SHALL return a bettor from a Python expression, evaluated with every scikit-learn estimator,
`make_pipeline`, `make_column_transformer`, `ClassifierBettor`, `OddsComparisonBettor` and `BettorGridSearchCV` in
scope. A `model` with a colon and no parenthesis SHALL be read as `path.py:name`, the object `name` in that file. It
SHALL raise `BuildError` when the text is not a model, the reference is not loadable, or the result is not a bettor.

#### Scenario: A bettor from a file
- **GIVEN** a file `bettors.py` that defines `bettor = OddsComparisonBettor()`
- **WHEN** the user calls `build_bettor(f'{path}:bettor')`
- **THEN** it returns `OddsComparisonBettor()`

#### Scenario: An expression that is not a model
- **WHEN** the user calls `build_bettor('NotAModel(')`
- **THEN** a `BuildError` is raised whose message contains `is not a model`

#### Scenario: An estimator that is not a bettor
- **WHEN** the user calls `build_bettor('LogisticRegression()')`
- **THEN** a `BuildError` is raised whose message contains `is not a bettor`

### Requirement: Save and load a bettor
`save_bettor(bettor, path)` SHALL write the bettor to `path` with cloudpickle and return `None`. `load_bettor(path)`
SHALL read it back. A fitted bettor SHALL come back fitted, ready to call `bet` without fitting again.

#### Scenario: A fitted bettor comes back fitted
- **GIVEN** `OddsComparisonBettor(betting_markets=['home_win', 'draw', 'away_win'])` fitted on the England sample and
  saved with `save_bettor`
- **WHEN** the user calls `load_bettor(path)`
- **THEN** its `betting_markets_` is `['home_win', 'draw', 'away_win']`
- **AND** its `bet(X, O)` returns an array of shape `(380, 3)` without fitting again
