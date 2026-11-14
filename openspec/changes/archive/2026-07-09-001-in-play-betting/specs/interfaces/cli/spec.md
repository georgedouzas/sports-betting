## Purpose

The `sportsbet` command line and the `sportsbet-gui` app let a user extract data, backtest and find value bets without
writing Python, with the same results as the API.

## ADDED Requirements

### Requirement: Surface parity for moment-aware modelling
The CLI and the GUI SHALL expose moment-aware training extraction, fixtures extraction, backtesting and value-bet
identification. Every capability MUST be reachable from the API, the CLI and the GUI alike. For the same configuration
the CLI MUST produce the same extraction, backtest and value-bet results as the API.

#### Scenario: CLI matches the API
- **GIVEN** a configuration that uses `DummySoccerDataLoader` and a target moment
- **WHEN** the user runs extraction and a backtest through the CLI
- **THEN** the results equal those of the equivalent API workflow

#### Scenario: Value bets from the CLI
- **WHEN** the user asks the CLI for value bets with a configuration that sets a target moment
- **THEN** it outputs value-bet selections for the fixtures at that target moment

### Requirement: Configuration declares the target moment
The CLI SHALL read a Python configuration module that declares `DATALOADER_CLASS`, `PARAM_GRID`, `DROP_NA_THRES`,
`ODDS_TYPE`, `BETTOR` and `CV`, plus `TARGET_EVENT_STATUS` of `preplay`, `inplay` or `postplay`, defaulting to
`postplay`, and `TARGET_EVENT_TIME`, such as `'60min'`, which applies only when the status is `inplay`.

#### Scenario: In-play target from the configuration
- **WHEN** the configuration sets `TARGET_EVENT_STATUS = 'inplay'` and `TARGET_EVENT_TIME = '60min'`
- **THEN** the CLI extracts training data for the in-play moment at 60 minutes

#### Scenario: Fixtures keep the training layout
- **WHEN** the user extracts fixtures through the CLI
- **THEN** the fixtures reproduce the training column layout with no targets

### Requirement: CLI error reporting
The CLI SHALL report an invalid configuration, such as an unknown symbol or a bad target moment, as a clear message
without a traceback. It SHALL report an empty selection or the absence of resolvable outcomes with an informative
message and a non-zero exit status where the command fails.

#### Scenario: Bad target moment
- **WHEN** the configuration sets an unknown `TARGET_EVENT_STATUS`
- **THEN** the CLI prints a clear error and no traceback

### Requirement: GUI exposes the moment-aware workflow
The `sportsbet-gui` app SHALL let the user choose a target moment, an event status and an in-play time, when creating a
dataloader and a model, and SHALL present the same outcomes and value bets as the API for that configuration.

#### Scenario: Backtest in the GUI
- **WHEN** the user configures a target moment in the GUI and runs a backtest
- **THEN** the app presents the same outcomes and value bets as the API
