## ADDED Requirements

### Requirement: Reach every sport and source from the command line
The `sportsbet` command line SHALL reach every sport, league and data source the Python API reaches, including a source
that needs a paid credential. Every existing command line capability MUST keep working with the new configuration.

#### Scenario: Basketball with a credentialled odds source
- **GIVEN** a configuration whose `DATALOADER` is a `BasketballDataLoader` with `NBAStats()` and an `OddsApi` key
- **WHEN** the user runs `sportsbet dataloader prepare` and then extracts training data
- **THEN** the command line returns the same data the Python API returns for that dataloader

#### Scenario: Soccer with the free sources
- **GIVEN** a configuration with `DATALOADER = SoccerDataLoader(param_grid={'league': ['England'], 'year': [2025]})`
- **WHEN** the user runs the same commands a soccer user ran before
- **THEN** they work with no extra configuration

### Requirement: Reject the old configuration contract
The command line SHALL reject a configuration that defines `DATALOADER_CLASS` with an error message stating that
`DATALOADER_CLASS` and `PARAM_GRID` are replaced by a configured `DATALOADER`. It MUST NOT fail with an `AttributeError`
and MUST NOT silently ignore the old names.

#### Scenario: A configuration written for the old contract
- **GIVEN** a configuration with `DATALOADER_CLASS = SoccerDataLoader` and a `PARAM_GRID`
- **WHEN** the user runs any `sportsbet` command with it
- **THEN** the command fails with a message that names the replacement `DATALOADER`

### Requirement: Discover parameters before a selection exists
`sportsbet dataloader params` SHALL work with a `DATALOADER` that has no `param_grid`. It SHALL report the parameters
the configured sources publish, and it SHALL download no data to do so.

#### Scenario: Nothing selected yet
- **GIVEN** a configuration with `DATALOADER = SoccerDataLoader()`
- **WHEN** the user runs `sportsbet dataloader params`
- **THEN** the command lists the leagues and seasons the sources publish

### Requirement: Read credentials from the environment
A configuration SHALL be able to read a source credential from the environment, as in
`OddsApi(key=os.environ['ODDS_API_KEY'])`, so the configuration file never holds the credential. When the variable is
absent, the error MUST name the missing variable and MUST NOT print a credential value.

#### Scenario: Credential set in the environment
- **GIVEN** `ODDS_API_KEY` is set and the configuration reads it with `os.environ`
- **WHEN** the user runs a command that fetches odds
- **THEN** the source authenticates with that key and the configuration file contains no key

#### Scenario: Credential absent
- **GIVEN** `ODDS_API_KEY` is not set
- **WHEN** the user runs a command with that configuration
- **THEN** the error names `ODDS_API_KEY` and shows no credential value

## MODIFIED Requirements

### Requirement: Configuration declares the target moment
The `sportsbet` command line SHALL read a Python configuration module that defines `DATALOADER`, a configured
`BaseDataLoader` instance whose constructor carries the selection, the sources and any credential. `BETTOR`, a
configured `BaseBettor`, SHALL be required for `backtest` and `bet`. `ODDS_TYPE`, `DROP_NA_THRES`,
`TARGET_EVENT_STATUS`, `TARGET_EVENT_TIME`, `CV`, `N_JOBS` and `VERBOSE` SHALL stay optional. `PARAM_GRID` SHALL NOT
be read.

#### Scenario: In-play target from the configuration
- **WHEN** the configuration sets `TARGET_EVENT_STATUS = 'inplay'` and `TARGET_EVENT_TIME = '60min'`
- **THEN** the command line extracts training data for the in-play moment at 60 minutes

#### Scenario: Fixtures keep the training layout
- **WHEN** the user extracts fixtures through the command line
- **THEN** the fixtures reproduce the training column layout with no targets

#### Scenario: Sources from the configuration
- **WHEN** the configuration's `DATALOADER` is built with its statistics and odds sources
- **THEN** the command line uses those sources, read from the configured instance rather than from `STATS` and `ODDS`

#### Scenario: A configured dataloader
- **GIVEN** a configuration module whose `DATALOADER` is a `SoccerDataLoader` built with a `param_grid`
  and whose `BETTOR = OddsComparisonBettor(alpha=0.03)`
- **WHEN** the user runs `sportsbet bettor backtest --config-path` with that module
- **THEN** the command backtests the bettor on the data the dataloader selects

#### Scenario: A class in place of an instance
- **GIVEN** a configuration with `DATALOADER = SoccerDataLoader`, the class itself
- **WHEN** the user runs any `sportsbet` command with it
- **THEN** the command rejects the configuration because `DATALOADER` is not a configured `BaseDataLoader`

## REMOVED Requirements

### Requirement: GUI exposes the moment-aware workflow
**Reason**: The `sportsbet-gui` app reached only soccer, had no tests, was excluded by name from the test run, pinned
`reflex==0.7.0`, and needed a Node toolchain. It is removed with its `gui` extra, its `sportsbet-gui` entry point, its
`--ignore` in the test configuration, and the `bandit` skips its launcher needed. The changelog announces the removal as
a breaking change and names the replacement.
**Migration**: Use the `sportsbet` command line, or install `sportsbet[mcp]` and drive the library from an assistant
through the `sportsbet-mcp` server.
