# Contract: `sportsbet.evaluation` public API

Bettors remain scikit-learn estimators; only their input format changes to the new
moment-aware `X`/`Y`/`O` columns (grammar in [research.md R4](../research.md)).

## Public exports (unchanged surface)

`BaseBettor`, `ClassifierBettor`, `OddsComparisonBettor`, `BettorGridSearchCV`, `backtest`,
`save_bettor`, `load_bettor`.

## Estimator contract (preserved — Constitution Principle I)

```text
bettor.fit(X, Y, O=None) -> self
bettor.predict_proba(X) -> ndarray          # class probabilities
bettor.predict(X) -> ndarray[bool]          # value-bet class labels
bettor.bet(X, O) -> ndarray[bool]           # value-bet selections
bettor.score(X, Y, O) -> float
bettor.classes_ -> list                     # fitted state
```

## Behavioral requirements for migration

- **CR-1**: `fit`/`predict`/`bet` MUST accept the `X`/`Y`/`O` tables produced by
  `extract_train_data`/`extract_fixtures_data` for any supported target moment (pre-match or
  in-play) without manual reshaping (FR-016).
- **CR-2**: Market discovery MUST parse market/outcome names from the new column grammar
  (`{provider}__{market}__{status}__{time}` for odds; target names for `Y`) — replacing the
  legacy `odds__market__outcome__market_type` parsing in
  `BaseBettor._get_feature_names_odds` and `OddsComparisonBettor._check_odds_types`.
- **CR-3**: `O`'s columns and `Y`'s columns MUST be reconcilable so a bettor can align a
  market's odds with its target outcome for value-bet identification.
- **CR-4**: A bettor fitted on a training extraction MUST apply unchanged to the fixtures
  extraction from the same loader (guaranteed by identical column structure).

## `backtest(bettor, X, Y, O, ...)`

- Returns per-period performance results across the backtest windows.
- MUST operate on moment-aware data unchanged from the caller's perspective.

## Doctest contract (offline, via sample loader)

```python
>>> from sportsbet.datasets import DummySoccerDataLoader
>>> from sportsbet.evaluation import ClassifierBettor, backtest
>>> from sklearn.dummy import DummyClassifier
>>> loader = DummySoccerDataLoader(param_grid={'league': ['England']})
>>> X, Y, O = loader.extract_train_data(odds_type='market_average')
>>> bettor = ClassifierBettor(DummyClassifier())
>>> results = backtest(bettor, X, Y, O)          # returns performance results
>>> X_fix, _, O_fix = loader.extract_fixtures_data()
>>> _ = bettor.fit(X, Y)
>>> selections = bettor.bet(X_fix, O_fix)        # value bets for fixtures
```
