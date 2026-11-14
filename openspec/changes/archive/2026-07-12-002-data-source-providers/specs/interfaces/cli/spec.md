## ADDED Requirements

### Requirement: Prepare command
The CLI SHALL provide a `prepare` subcommand that prepares the configured dataloader's store, and a `--dry-run` option
that reports the items to fetch, the items held and the cost estimate without fetching or spending anything.

#### Scenario: Dry run from the CLI
- **WHEN** the user runs the `prepare` subcommand with `--dry-run`
- **THEN** the CLI prints the plan and the cost estimate and fetches nothing

#### Scenario: Prepare from the CLI
- **WHEN** the user runs the `prepare` subcommand without `--dry-run`
- **THEN** the store is populated and a later extraction through the CLI succeeds

### Requirement: Parameter discovery asks the statistics source
The `dataloader params` command SHALL list the available parameters by asking the configured statistics source, not the
dataloader.

#### Scenario: List parameters
- **WHEN** the user runs `dataloader params` with a configuration that sets `STATS`
- **THEN** the CLI prints the combinations that statistics source publishes

## MODIFIED Requirements

### Requirement: Configuration declares the target moment
The CLI SHALL read a Python configuration module that declares `DATALOADER_CLASS`, `PARAM_GRID`, `DROP_NA_THRES`,
`ODDS_TYPE`, `BETTOR` and `CV`, the sources and store as `STATS`, `ODDS` and `STORE` mirroring the constructor, plus
`TARGET_EVENT_STATUS` of `preplay`, `inplay` or `postplay`, defaulting to `postplay`, and `TARGET_EVENT_TIME`, such as
`'60min'`, which applies only when the status is `inplay`.

#### Scenario: In-play target from the configuration
- **WHEN** the configuration sets `TARGET_EVENT_STATUS = 'inplay'` and `TARGET_EVENT_TIME = '60min'`
- **THEN** the CLI extracts training data for the in-play moment at 60 minutes

#### Scenario: Fixtures keep the training layout
- **WHEN** the user extracts fixtures through the CLI
- **THEN** the fixtures reproduce the training column layout with no targets

#### Scenario: Sources from the configuration
- **WHEN** the configuration sets `STATS`, `ODDS` and `STORE`
- **THEN** the CLI builds the dataloader with those sources and that store

### Requirement: GUI exposes the moment-aware workflow
The `sportsbet-gui` app SHALL let the user choose a target moment, an event status and an in-play time, when creating a
dataloader and a model, and SHALL present the same outcomes and value bets as the API. Its dataloader creation flow MUST
include a preparation step that shows progress and the cost estimate before spending anything, and its parameter pickers
MUST be filled from the source.

#### Scenario: Backtest in the GUI
- **WHEN** the user configures a target moment in the GUI and runs a backtest
- **THEN** the app presents the same outcomes and value bets as the API

#### Scenario: Prepare in the GUI
- **WHEN** the user creates a dataloader in the GUI
- **THEN** the app shows the preparation estimate and fetches only after the user proceeds
