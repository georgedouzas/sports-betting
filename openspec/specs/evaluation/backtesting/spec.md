# evaluation/backtesting Specification

## Purpose

Backtesting measures how a bettor would have performed on historical data with forward-in-time splits, and the grid
search tunes a bettor's parameters on the same kind of splits, scored by the bettor's Sharpe ratio by default.

## Requirements

### Requirement: Backtest input validation
`backtest(bettor, X, Y, O, cv=None, n_jobs=-1, verbose=0)` SHALL require `X` to be a DataFrame with a `DatetimeIndex`
and `Y` and `O` to be DataFrames, and SHALL raise `TypeError` otherwise. Inputs of different lengths SHALL raise
`ValueError`. `cv` SHALL be `None` or a scikit-learn `TimeSeriesSplit`, and another value SHALL raise `TypeError`.

#### Scenario: A cross-validator that is not a TimeSeriesSplit
- **WHEN** the user calls `backtest(bettor, X, Y, O, cv=3)`
- **THEN** a `TypeError` is raised whose message contains `` Parameter `cv` ``

#### Scenario: Input data that is not a DataFrame
- **WHEN** the user calls `backtest(bettor, X.to_numpy(), Y, O)`
- **THEN** a `TypeError` is raised whose message contains `` Input data `X` ``

#### Scenario: Input data without a date index
- **WHEN** the user calls `backtest(bettor, pd.DataFrame(X.values), Y, O)`
- **THEN** a `TypeError` is raised whose message contains `` Input data `X` ``

### Requirement: Backtest splits forward in time
`backtest` SHALL sort the rows by date and split them with `cv`, where `None` means `TimeSeriesSplit()` with its
default five splits. For each split it SHALL fit the bettor on the training rows with their odds and bet on the test
rows. It SHALL run the splits in parallel on `n_jobs` workers, where `-1` means every processor, logging at `verbose`.

#### Scenario: One result per split
- **WHEN** the user calls `backtest(bettor, X, Y, O, cv=TimeSeriesSplit(3))` with a `ClassifierBettor` or an
  `OddsComparisonBettor`
- **THEN** it returns a DataFrame with three rows

### Requirement: Backtest results
The results SHALL be indexed by `Training start`, `Training end`, `Testing start` and `Testing end`, as dates. The
columns SHALL be `Number of betting days`, `Number of bets`, `Yield percentage per bet`, `ROI percentage` and
`Final cash`, then `Number of bets ({market})` and `Yield percentage per bet ({market})` for each selected market.

#### Scenario: Result columns for three markets
- **GIVEN** a bettor with `betting_markets=['home_win', 'draw', 'away_win']`
- **WHEN** the user calls `backtest(bettor, X, Y, O, cv=TimeSeriesSplit(2), n_jobs=1)`
- **THEN** the index names are `['Training start', 'Training end', 'Testing start', 'Testing end']`
- **AND** the columns end with `Number of bets (home_win)`, `Number of bets (draw)`, `Number of bets (away_win)`,
  `Yield percentage per bet (home_win)`, `Yield percentage per bet (draw)` and `Yield percentage per bet (away_win)`

### Requirement: Backtest metrics
A bet's return SHALL be its outcome times its odds minus one. `Number of betting days` SHALL count distinct test dates.
`Yield percentage per bet` SHALL be 100 times the mean bet return, rounded to one decimal, and zero with no bets.
`ROI percentage` SHALL be 100 times `stake_` times the summed returns over `init_cash_`, rounded to one decimal.
`Final cash` SHALL be `init_cash_` plus `stake_` times the summed returns.

#### Scenario: Metrics are present
- **WHEN** the user backtests an `OddsComparisonBettor(alpha=0.03)` on the England sample
- **THEN** the results contain `Number of bets` and `Yield percentage per bet`

### Requirement: Sharpe ratio score
`score(X, Y, O)` SHALL raise `NotFittedError` before fit and validate `X`, `Y` and `O` as `fit` does. It SHALL sum the
returns of the value bets per date, fill every calendar day without bets with zero, and return the square root of 365
times the mean daily return over its standard deviation. A zero or undefined deviation SHALL give `100.0` when the mean
is positive and `-100.0` otherwise.

#### Scenario: A finite score for an odds-appending bettor
- **GIVEN** `OddsComparisonBettor(alpha=0.03)` fitted on `X`, `Y` and `O`
- **WHEN** the user calls `score(X, Y, O)`
- **THEN** it returns a finite number

### Requirement: BettorGridSearchCV wraps a bettor
`BettorGridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=TimeSeriesSplit(n_splits=3),
verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=False)` SHALL be a bettor that searches
`param_grid` as scikit-learn's `GridSearchCV` does. `estimator` MUST be a `BaseBettor`, or `fit` SHALL raise
`TypeError`. `cv` MUST be a `TimeSeriesSplit`, or `fit` SHALL raise `TypeError`.

#### Scenario: An estimator that is not a bettor
- **WHEN** the user calls `BettorGridSearchCV(DummyClassifier(), {}).fit(X, Y, O)`
- **THEN** a `TypeError` is raised whose message contains `` `BettorGridSearchCV` requires a bettor as estimator ``

#### Scenario: A cross-validator that is not a TimeSeriesSplit
- **WHEN** the user calls `BettorGridSearchCV(OddsComparisonBettor(), {}, cv=KFold()).fit(X, Y, O)`
- **THEN** a `TypeError` is raised whose message contains `` Parameter `cv` ``

### Requirement: Grid search scoring
With `scoring=None`, the search SHALL score each candidate with the bettor's `score`, passing `O` to `fit` and `score`,
and `fit` MUST receive `O`, or `TypeError` SHALL be raised. With a `scoring` given, the search SHALL score each
candidate on the `Y` columns of its selected markets, and SHALL NOT require `O`.

#### Scenario: Default scoring without odds
- **WHEN** the user calls `BettorGridSearchCV(OddsComparisonBettor(), {}).fit(X, Y)`
- **THEN** a `TypeError` is raised whose message contains `The default scoring method requires the odds data`

#### Scenario: A finite best score
- **WHEN** the user fits `BettorGridSearchCV(OddsComparisonBettor(), {'alpha': [0.02, 0.05]}, cv=TimeSeriesSplit(2))`
  on `X`, `Y` and `O`
- **THEN** `best_score_` is finite

### Requirement: Grid search exposes the best bettor
After fit with refit, the search SHALL copy `init_cash_`, `stake_`, `betting_markets_` and `feature_names_out_` from
`best_estimator_`, and SHALL set `feature_names_odds_` when `O` is given. `predict_proba`, `predict` and `bet` SHALL
delegate to `best_estimator_` and SHALL raise `NotFittedError` before fit. `classes_` SHALL raise `AttributeError`
before fit.

#### Scenario: Markets of the best bettor
- **GIVEN** a search over `betting_markets` in `[None, ['draw'], ['home_win', 'away_win']]` with `TimeSeriesSplit(2)`
- **WHEN** the user fits it on `X`, `Y` and `O`
- **THEN** `n_splits_` is 2
- **AND** `betting_markets_` and `feature_names_out_` equal those of `best_estimator_`
- **AND** `cv_results_['params']` has three entries

### Requirement: A grid search can be backtested
A `BettorGridSearchCV` SHALL be accepted by `backtest`, which then runs the search inside every training split.

#### Scenario: Backtesting a search
- **GIVEN** `BettorGridSearchCV(estimator=OddsComparisonBettor(), param_grid={'alpha': [0.02, 0.05, 0.1]},
  cv=TimeSeriesSplit(2))`
- **WHEN** the user calls `backtest(search, X, Y, O, cv=TimeSeriesSplit(2))`
- **THEN** the results contain `Number of bets`
