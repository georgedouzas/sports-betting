## Purpose

Measure how a betting strategy would have performed on moment-aware historical data, period by period, and search its
parameters over the same data.

## ADDED Requirements

### Requirement: Backtest on moment-aware data
`backtest(bettor, X, Y, O)` SHALL run the bettor over moment-aware training data and return performance results for
each backtest period. It MUST accept the tables for any supported target moment unchanged from the caller's view.

#### Scenario: Backtest returns per-period results
- **GIVEN** training data extracted for a target moment
- **WHEN** the user calls `backtest(bettor, X, Y, O)`
- **THEN** it returns performance results that cover the backtest periods

### Requirement: Grid search on moment-aware data
`BettorGridSearchCV` SHALL fit over moment-aware `X`, `Y` and `O` tables and select bettor parameters with time-series
cross-validation.

#### Scenario: Grid search fits
- **WHEN** the user fits a `BettorGridSearchCV` on tables extracted for a target moment
- **THEN** the search completes and exposes the best bettor
