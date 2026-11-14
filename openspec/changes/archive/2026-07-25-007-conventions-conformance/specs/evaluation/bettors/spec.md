## ADDED Requirements

### Requirement: Derive complementary events
`sportsbet.evaluation.derive_complementary_events(markets)` SHALL return the groups of markets that are mutually
exclusive and exhaustive among the given markets. The outcome group holds whichever of `home_win`, `draw` and `away_win`
are present when there are at least two. Each `over_X` market pairs with the `under_X` market of the same line.

#### Scenario: Outcomes and a total line
- **WHEN** a user calls `derive_complementary_events(['home_win', 'draw', 'away_win', 'over_2.5', 'under_2.5'])`
- **THEN** it returns `[['home_win', 'draw', 'away_win'], ['over_2.5', 'under_2.5']]`

#### Scenario: Two outcomes
- **WHEN** a user calls `derive_complementary_events(['home_win', 'away_win'])`
- **THEN** it returns `[['home_win', 'away_win']]`

#### Scenario: A lone market
- **WHEN** a user calls `derive_complementary_events(['draw'])`
- **THEN** it returns `[]`

### Requirement: Derive the market base
`sportsbet.evaluation.derive_market_base(market)` SHALL return the base market name of a market label by dropping the
`__status__time` suffix.

#### Scenario: A market with a moment suffix
- **WHEN** a user calls `derive_market_base('home_win__inplay__60min')`
- **THEN** it returns `'home_win'`

### Requirement: Find the latest odds column
`sportsbet.evaluation.find_latest_odds_column(columns, base, provider=None)` SHALL return the odds column of the form
`{provider}__{base}__{status}__{time}` for the given market base at the latest event status and time. When `provider` is
given, it SHALL match only that provider. It SHALL return `None` when no column matches.

#### Scenario: Two snapshots of one market
- **GIVEN** the columns `['pinnacle__home_win__preplay__0min', 'pinnacle__home_win__inplay__60min']`
- **WHEN** a user calls `find_latest_odds_column(columns, 'home_win')`
- **THEN** it returns `'pinnacle__home_win__inplay__60min'`

#### Scenario: No matching column
- **WHEN** a user calls `find_latest_odds_column(['pinnacle__draw__preplay__0min'], 'home_win')`
- **THEN** it returns `None`

## REMOVED Requirements

### Requirement: Complementary markets derived from the data
**Reason**: The public function `complementary_events` began with a noun, which Principle VI forbids. It is renamed to
`derive_complementary_events` with the same behaviour. The fitted attribute `complementary_events_` and the class
constant `COMPLEMENTARY_EVENTS` keep their names because they belong to the scikit-learn contract surface.
**Migration**: Replace `from sportsbet.evaluation import complementary_events` and its calls with
`derive_complementary_events`.
