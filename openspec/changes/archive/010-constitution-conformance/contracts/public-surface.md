# Contract: the public surface this feature must not change

**Feature**: [spec.md](../spec.md) | **Plan**: [plan.md](../plan.md) | **Date**: 2026-09-21

This feature renames 110 names and rewrites hundreds of docstrings. This document is the list it must leave alone.
Every name below is re-exported by a package `__init__`, so a user can reach it, and a user who can reach it may
depend on it. Anything not on this list is internal, whatever it looks like today.

The contract is negative: no name here is renamed, removed, or given a different signature, return, or exception. The
only changes permitted to these names are to their docstrings.

## The 88 re-exported names, of which 5 moved package

### `sportsbet.core`, 26 names

`ALIASES`, `BoolData`, `BuildError`, `CancellationUnsupportedError`, `CredentialError`, `Data`, `EVENT_COLS`,
`ExecutionError`, `FixturesData`, `GROUPS_COLS`, `IDENTITY_COLS`, `IDENTITY_FIELDS`, `Indices`, `MATCH_COLS`,
`NON_PREPLAY_EVENT_STATUSES`, `NotExtractedError`, `PREPLAY_EVENT_STATUSES`, `ParamGrid`, `STATUSES`, `STATUS_RANK`,
`TEAMS_COLS`, `TrainData`, `VenueBlockedError`, `format_event_time`, `load_object`, `parse_event_time`

Every named exception moved here at constitution version 12.0.0, so a caller finds them all in one place.

`DATE_COLS` became private, and `Param`, `Schema` and `OutputsMapping` were removed as dead code, under the surface
rules added at constitution version 10.3.0.

### `sportsbet.sources`, 26 names

`BaseOddsSchema`, `BaseOddsSource`, `BaseSource`, `BaseStatsSchema`, `BaseStatsSource`, `EuroLeagueStats`,
`FootballDataOdds`, `FootballDataStats`, `NBAStats`, `OddsApi`, `RawItem`, `RawPayload`, `SampleSoccerOdds`,
`SampleSoccerStats`, `build_roster`, `count_common_prefix`, `derive_market_outcomes`, `fetch_payloads`,
`measure_names_similarity`, `normalize_identity`, `normalize_team_name`, `optional_col`, `pair_rosters`,
`read_csv_content`, `required_col`, `resolve_odds`

### `sportsbet.execution`, 15 names

`BaseVenue`, `BetIdentity`, `BrowserSession`, `CredentialRef`, `FixedSession`, `PageSnapshot`, `PlacementIntent`,
`PlacementReceipt`, `PlacementStatus`, `Placer`, `build_receipts_frame`, `build_venue`, `execute_event`,
`find_betting_moment`, `resolve`

`ExecutionError`, `CancellationUnsupportedError`, `VenueBlockedError` and `CredentialError` moved to
`sportsbet.core`.

`PlacementReceiptSchema` became private, and `PlacementQuote` and `ExposureLimits` were removed as dead code.
`Placer` stays, since it names the type of `execute_event`'s `placer` parameter.

### `sportsbet.evaluation`, 11 names

`BaseBettor`, `BettorGridSearchCV`, `ClassifierBettor`, `OddsComparisonBettor`, `backtest`, `build_bettor`,
`derive_complementary_events`, `derive_market_base`, `find_latest_odds_column`, `load_bettor`, `save_bettor`

### `sportsbet.dataloaders`, 8 names

`BaseDataLoader`, `DEFAULT_KEY_ENV`, `DataLoader`, `ODDS_SOURCES`, `STATS_SOURCES`, `build_dataloader`,
`build_extraction_settings`, `load_dataloader`

### `sportsbet.mcp`, 2 names

`run`, `server`

### `sportsbet.cli`, 1 name

`main`

## The other two surfaces

The Library section requires the surfaces to expose the same capabilities, and a parity test already asserts it. This
feature changes neither, and the parity test is the contract.

- **The CLI**, `sportsbet`, 21 commands. No command name, option name, argument order, or exit code changes.
- **The MCP server**, `sportsbet-mcp`, 21 tools. No tool name, parameter name, or returned shape changes.

## What is not in the contract

- The 110 names defined in private modules that no package re-exports. They are renamed to carry an underscore. A user
  reaching one of them today is reaching past a surface, which the Surface section already forbids.
- Every name that already starts with an underscore.
- Docstrings, everywhere. They are what the feature changes.

## How the contract is checked

- The existing test suite passes unchanged. A test that needs an edit is the signal that a name on this list moved.
- The parity test asserts the three surfaces still match.
- The import of every module on this list succeeds from its documented surface, which the doctest run exercises as a
  user would.
- `git diff` over the feature touches no line that defines or re-exports a name on this list, except inside a
  docstring.
