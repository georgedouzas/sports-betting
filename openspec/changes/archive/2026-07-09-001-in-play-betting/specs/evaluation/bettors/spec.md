## Purpose

Bettors are scikit-learn estimators that learn outcome probabilities from extracted features and targets, compare them
with odds, and select value bets for any target moment.

## ADDED Requirements

### Requirement: Bettor estimator contract
A bettor SHALL be a scikit-learn estimator with `fit(X, Y, O=None)`, `predict_proba(X)`, `predict(X)`, `bet(X, O)` and
`score(X, Y, O)`, and the fitted attribute `classes_`. `predict_proba` MUST return class probabilities and `predict` and
`bet` MUST return boolean selections.

#### Scenario: Fit and predict
- **WHEN** the user fits a `ClassifierBettor` on extracted `X` and `Y`
- **THEN** `classes_` is set and `predict_proba(X)` returns one probability column per class

### Requirement: Bettors accept moment-aware tables
`fit`, `predict` and `bet` SHALL accept the `X`, `Y` and `O` tables from `extract_train_data` and
`extract_fixtures_data` for any supported target moment, pre-match or in-play, with no manual reshaping. A bettor fitted
on a training extraction MUST apply unchanged to the fixtures extraction from the same loader.

#### Scenario: In-play training data
- **WHEN** the user fits a bettor on tables extracted for an in-play target
- **THEN** the bettor fits with no reshaping of the tables

#### Scenario: Train on training data, bet on fixtures
- **WHEN** the user fits a bettor on a training extraction and calls `bet` on the fixtures extraction of the same loader
- **THEN** the call succeeds with no column reconciliation by the user

### Requirement: Market discovery from the column grammar
Bettors SHALL discover betting markets by parsing the extraction column grammar, odds as
`{provider}__{market}__{status}__{time}` and targets by their target names, instead of the legacy
`odds__market__outcome__market_type` convention. The `O` and `Y` columns MUST reconcile so a bettor aligns each market's
odds with its target outcome.

#### Scenario: Odds comparison parses the new grammar
- **WHEN** the user fits an `OddsComparisonBettor` on tables extracted with an odds type
- **THEN** the bettor identifies its markets and odds types from the new column names

### Requirement: Value bets for fixtures
A fitted bettor SHALL return value-bet selections for fixtures data through `bet(X, O)`, aligned row for row to the
fixtures.

#### Scenario: Value bets on fixtures
- **GIVEN** a bettor fitted on training data for a target moment
- **WHEN** the user calls `bet` with the fixtures `X` and `O` for the same target moment
- **THEN** it returns one row of selections per fixture

### Requirement: Bettor persistence
The library SHALL save a bettor with `save_bettor` and reload it with `load_bettor`.

#### Scenario: Save and reload a bettor
- **WHEN** the user saves a fitted bettor and reloads it
- **THEN** the reloaded bettor returns the same selections on the same data

### Requirement: Public evaluation exports
`sportsbet.evaluation` SHALL export `BaseBettor`, `ClassifierBettor`, `OddsComparisonBettor`, `BettorGridSearchCV`,
`backtest`, `save_bettor` and `load_bettor`.

#### Scenario: Import the public names
- **WHEN** the user imports each of those names from `sportsbet.evaluation`
- **THEN** every import succeeds
