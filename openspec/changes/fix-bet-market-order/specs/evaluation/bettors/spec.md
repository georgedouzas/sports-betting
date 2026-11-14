## MODIFIED Requirements

### Requirement: Value bets
`bet(X, O)` SHALL raise `NotFittedError` before fit. It SHALL use, per market, the latest odds column. A market SHALL
be a value bet where its probability times its odds exceeds one. Within a complementary group, a row SHALL keep at most
the bet with the highest expected return. It SHALL return a boolean array with one row per match and one column per
selected market, with the columns in the order of `betting_markets_`. Odds lacking a selected market SHALL raise
`ValueError`.

#### Scenario: Shape and type
- **GIVEN** a bettor fitted on `X` and `Y`
- **WHEN** the user calls `bet(X, O)`
- **THEN** the result has shape `(len(X), number of selected markets)` and dtype `bool`

#### Scenario: Bet before fit
- **WHEN** the user calls `bet(X, O)` on an unfitted bettor
- **THEN** a `NotFittedError` is raised

#### Scenario: Groups ordered against the markets
- **GIVEN** a bettor whose complementary groups list `over_2.5` and `under_2.5` before `home_win`, `draw` and
  `away_win`, while `betting_markets_` lists `home_win` first
- **WHEN** the user calls `bet(X, O)`
- **THEN** the first column is true exactly for the rows where `home_win` is the value bet

## ADDED Requirements

### Requirement: Every market falls in a group
`fit` SHALL raise `ValueError` when a selected market belongs to no complementary group, naming the markets left out.

#### Scenario: A market left out of the groups
- **GIVEN** a bettor whose `COMPLEMENTARY_EVENTS` is `[['home_win', 'draw', 'away_win']]`
- **WHEN** the user fits it with `betting_markets=['home_win', 'over_2.5']`
- **THEN** a `ValueError` is raised whose message names `over_2.5`
