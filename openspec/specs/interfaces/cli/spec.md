# interfaces/cli Specification

## Purpose

The `sportsbet` command line drives the library from a shell. Its command groups mirror the Python API: `dataloader`
selects, downloads and extracts data, `evaluation` backtests, fits and bets with a model, and `execution` places bets.

## Requirements

### Requirement: Entry points
The library SHALL install a `sportsbet` console command, and `python -m sportsbet` SHALL run the same command. The top
group SHALL hold the command groups `dataloader`, `evaluation` and `execution`. Called with no command, it SHALL print
its help and exit with status 2. Called with `--help`, it SHALL print its help and exit with status 0.

#### Scenario: No command prints the help
- **WHEN** the user runs `sportsbet` with no arguments
- **THEN** the output contains `Create, test and use sports betting models.`
- **AND** the exit status is 2

#### Scenario: Help lists the groups
- **WHEN** the user runs `sportsbet --help`
- **THEN** the exit status is 0
- **AND** the output names the `dataloader` and `evaluation` groups

### Requirement: Commands are configured by their own options
Every command SHALL take everything it needs as command line options. No command SHALL read a configuration file. A
model, a dataloader or a venue that the user built in Python SHALL be passed as a file path or as a reference of the
form `file.py:NAME`.

#### Scenario: Discovery needs no file
- **WHEN** the user runs `sportsbet dataloader params --stats football-data --odds football-data --league England`
- **THEN** the exit status is 0
- **AND** the output lists `England`

### Requirement: Data selection options
The commands that select data SHALL accept `--stats` (required, one of `euroleague`, `football-data`, `nba`), `--odds`
(one of `football-data`, `odds-api`), the repeatable `--league`, `--division` (integer) and `--year` (integer, the year
a season ends), `--odds-key-env` (default `ODDS_API_KEY`), and the repeatable `--odds-market`, `--odds-region`,
`--odds-moment` and `--alias`.

#### Scenario: The statistics source is required
- **WHEN** the user runs `sportsbet dataloader params` with no `--stats`
- **THEN** the exit status is not 0
- **AND** the output names `--stats`

#### Scenario: An unknown source lists the known ones
- **WHEN** the user runs `sportsbet dataloader params --stats quidditch`
- **THEN** the exit status is not 0
- **AND** the output lists `football-data`

### Requirement: Secrets come from the environment
A command SHALL read a source key from the environment variable named by `--odds-key-env`, and no command SHALL accept a
key, password, secret, token, app key, username or credential as an option. When the odds source needs a key and the
variable is not set, the command SHALL print a message that names the variable and `--odds-key-env`.

#### Scenario: A missing key names its variable
- **GIVEN** `ODDS_API_KEY` is not set
- **WHEN** the user runs `sportsbet dataloader train extract --stats nba --odds odds-api --league NBA -o loader.pkl`
- **THEN** the output contains `ODDS_API_KEY`

#### Scenario: A set key is read
- **GIVEN** `ODDS_API_KEY` is set
- **WHEN** the user runs `sportsbet dataloader params --stats nba --odds odds-api --league NBA --odds-market h2h`
- **THEN** the output does not contain `needs a key`

### Requirement: Malformed moments and aliases
An `--odds-moment` value MUST be a status and a minute separated by a colon, and an `--alias` value MUST be two names
separated by `=`, the odds source's name first and the statistics source's name second. The help SHALL state that
order. A malformed value SHALL be answered with a message that shows the correct form, `inplay:45` for a moment and
`Olimpia Milano=EA7 Emporio Armani Milan` for an alias.

#### Scenario: A malformed moment
- **WHEN** the user passes `--odds-moment halftime` to `sportsbet dataloader train extract`
- **THEN** the output contains `inplay:45`

#### Scenario: A malformed alias
- **WHEN** the user passes `--alias nonsense` to `sportsbet dataloader train extract`
- **THEN** the output contains `Olimpia Milano`

#### Scenario: An alias in the stated order pairs the club
- **GIVEN** a club the odds source spells `Utd of Manchester` and the statistics source spells `Man United`
- **WHEN** the user passes `--alias 'Utd of Manchester=Man United'`
- **THEN** the matches of `Man United` keep their odds

### Requirement: Extraction options
`dataloader train extract` SHALL accept `--odds-type`, `--drop-na-thres` (default 0.0), `--target-event-status` and
`--input-event-status` (each one of `preplay`, `inplay`, `postplay`), and `--target-event-time` and `--input-event-time`
(a duration such as `45min`). `dataloader exploration extract` SHALL accept the same options except `--odds-type`.

#### Scenario: The moment options reach the extraction
- **WHEN** the user runs `sportsbet dataloader train extract` with `--odds-type market_average` and
  `--target-event-status postplay`
- **THEN** the saved dataloader holds the training data that `extract_train_data(odds_type='market_average',
  target_event_status='postplay')` returns for the same selection

### Requirement: dataloader params
`sportsbet dataloader params` SHALL take the data selection options and print a table titled `Available parameters`
with one row for each league, division and season combination the statistics source publishes. A combination that
lacks a field SHALL show `-` in that column. The table SHALL have no index column.

#### Scenario: The parameters are listed
- **WHEN** the user runs `sportsbet dataloader params --stats football-data --odds football-data --league England`
- **THEN** the exit status is 0
- **AND** the output contains `Available parameters`

### Requirement: dataloader odds-types
`sportsbet dataloader odds-types` SHALL take the data selection options, download the selected data, and print a table
titled `Available odds types` with one column `Type` that lists the odds types the selection carries.

#### Scenario: The odds types are listed
- **WHEN** the user runs `sportsbet dataloader odds-types --stats football-data --odds football-data --league England`
- **THEN** the exit status is 0
- **AND** the output contains `Available odds types`

### Requirement: dataloader train extract
`sportsbet dataloader train extract` SHALL take the selection and extraction options and a required `--output`/`-o`
file path. It SHALL download the training data, save the dataloader to that path, print the tables `Training input
data`, `Training output data` when there are targets, and `Training odds data` when there are odds, and then print
`Saved the dataloader to <path>.`

#### Scenario: The dataloader is saved
- **WHEN** the user runs `sportsbet dataloader train extract --stats football-data --odds football-data --league
  England --odds-type market_average --target-event-status postplay -o loader.pkl`
- **THEN** the exit status is 0
- **AND** the output contains `Training input data`
- **AND** `loader.pkl` exists and `load_dataloader` reads it

### Requirement: dataloader exploration extract
`sportsbet dataloader exploration extract` SHALL take the selection options, the extraction options without
`--odds-type`, and an optional `--output`/`-o` directory. It SHALL print the features under the title `Exploration
features`. With `-o`, it SHALL write them to `<directory>/sports-betting-data/X.csv`, creating the folders it needs.

#### Scenario: The features are written
- **WHEN** the user runs `sportsbet dataloader exploration extract --stats football-data --odds football-data --league
  England -o out`
- **THEN** the exit status is 0
- **AND** the output contains `Exploration features`
- **AND** `out/sports-betting-data/X.csv` exists

### Requirement: dataloader fixtures extract
`sportsbet dataloader fixtures extract` SHALL take a required `--dataloader`/`-d` path to an existing saved dataloader
and an optional `--output`/`-o` directory. It SHALL print `Fixtures input data`, and `Fixtures odds data` when there are
odds. With `-o` it SHALL write `X_fix.csv`, and `O_fix.csv` when there are odds, under
`<directory>/sports-betting-data`.

#### Scenario: The fixtures are written
- **GIVEN** a saved dataloader with a match still to be played
- **WHEN** the user runs `sportsbet dataloader fixtures extract -d loader.pkl -o out`
- **THEN** the exit status is 0
- **AND** the output contains `Fixtures input data`
- **AND** `out/sports-betting-data/X_fix.csv` exists

#### Scenario: No upcoming matches
- **GIVEN** a saved dataloader whose seasons are finished
- **WHEN** the user runs `sportsbet dataloader fixtures extract -d loader.pkl`
- **THEN** the command prints `There are no upcoming matches.` in a panel and writes nothing

### Requirement: Model option
`--model` SHALL accept a scikit-learn estimator written as a Python expression, such as
`OddsComparisonBettor(alpha=0.05)` or `ClassifierBettor(LogisticRegression(C=1.0))`, or a bettor built in a file and
named as `models.py:BETTOR`. `evaluation backtest` and `evaluation fit` MUST be given `--model`, and without it they
SHALL fail with a usage error.

#### Scenario: The model is required
- **WHEN** the user runs `sportsbet evaluation backtest -d loader.pkl` with no `--model`
- **THEN** the exit status is not 0
- **AND** the output names `--model`

#### Scenario: A model from a file
- **GIVEN** `models.py` defines `BETTOR = OddsComparisonBettor(alpha=0.02)`
- **WHEN** the user runs `sportsbet evaluation backtest -d loader.pkl --model models.py:BETTOR --cv 2`
- **THEN** the exit status is 0
- **AND** the output contains `Backtesting results`

#### Scenario: A classifier pipeline as an expression
- **WHEN** the user passes a `ClassifierBettor(make_pipeline(...))` expression to `evaluation backtest --model`
- **THEN** the backtest runs with no file written and prints `Backtesting results`

#### Scenario: Not a model
- **WHEN** the user runs `sportsbet evaluation backtest -d loader.pkl --model wishful-thinking`
- **THEN** the output contains `is not a model` and shows how to write one

### Requirement: evaluation backtest
`sportsbet evaluation backtest` SHALL take `-d`, `--model`, `--cv` (default 3, time-ordered folds), `--n-jobs` (default
-1), `--verbose` (default 0) and an optional `-o` directory. It SHALL backtest the model on the saved training data and
print `Backtesting results`. With `-o` it SHALL write `<directory>/sports-betting-data/backtesting_results.csv` with no
index column. A dataloader with no odds SHALL get a panel that says there is nothing to backtest against.

#### Scenario: The backtest matches the Python API
- **WHEN** the user runs `sportsbet evaluation backtest -d loader.pkl --model "OddsComparisonBettor(alpha=0.03)" --cv 2
  -o out`
- **THEN** the exit status is 0
- **AND** `out/sports-betting-data/backtesting_results.csv` has the rows and the `Number of bets` values that
  `backtest(OddsComparisonBettor(alpha=0.03), X, Y, O, cv=TimeSeriesSplit(2))` returns on the same data

#### Scenario: One betting market
- **WHEN** the user backtests `OddsComparisonBettor(betting_markets=["home_win"])`
- **THEN** the exit status is 0
- **AND** the output contains `Backtesting results`

### Requirement: evaluation fit
`sportsbet evaluation fit` SHALL take `-d`, `--model` and a required `--output`/`-o` file path. It SHALL fit the
model on the saved training data, save the fitted bettor to the path, and print `Saved the fitted model to <path>.`

#### Scenario: The fitted model is saved
- **WHEN** the user runs `sportsbet evaluation fit -d loader.pkl --model "OddsComparisonBettor(alpha=0.03)"
  -o model.pkl`
- **THEN** the exit status is 0
- **AND** `model.pkl` exists

### Requirement: evaluation bet
`sportsbet evaluation bet` SHALL take `-d`, a required `--bettor`/`-b` path to an existing model saved by `fit`, and an
optional `-o` directory. It SHALL print `Value bets`, a table with one row per upcoming match and one column per betting
market of the bettor. With `-o` it SHALL write `value_bets.csv` with its index under `<directory>/sports-betting-data`.

#### Scenario: Fit once, then bet
- **GIVEN** `model.pkl` was written by `evaluation fit`
- **WHEN** the user runs `sportsbet evaluation bet -d loader.pkl -b model.pkl -o out`
- **THEN** the exit status is 0
- **AND** the output contains `Value bets`
- **AND** `out/sports-betting-data/value_bets.csv` exists

#### Scenario: The fitted model is required
- **WHEN** the user runs `sportsbet evaluation bet -d loader.pkl` with no `--bettor`
- **THEN** the exit status is not 0
- **AND** the output names `--bettor`

#### Scenario: Nothing to bet on
- **WHEN** the saved dataloader has no upcoming matches or no odds for them
- **THEN** the command prints `There are no upcoming matches to bet on.` in a panel

### Requirement: Download once and reuse
Only `dataloader params`, `dataloader odds-types`, `dataloader train extract` and `dataloader exploration extract` SHALL
take the data selection. `dataloader fixtures extract` and the `evaluation` and `execution` commands SHALL read a
dataloader saved by `dataloader train extract` through `--dataloader`, and `evaluation bet` and `execution run` SHALL
read a model saved by `evaluation fit` through `--bettor`.

#### Scenario: The evaluation commands read the saved file
- **WHEN** the user runs `evaluation backtest`, `evaluation fit` and `evaluation bet` after `dataloader train extract`
- **THEN** each command reads the saved dataloader and none of them takes `--stats`

### Requirement: Console tables
A command SHALL print each result frame under its bold title as a table. Missing values SHALL show as `-`, booleans as
`yes` or `no`, floats with two decimals and thousands separators, and timestamps as `YYYY-MM-DD`. A table wider than the
terminal SHALL be printed one record at a time, and output taller than the terminal SHALL be shown in a pager.

#### Scenario: A wide frame
- **WHEN** a result frame is wider than the terminal
- **THEN** the command prints each row as a list of column name and value pairs

### Requirement: Error reporting
A usage error, such as a missing required option, a value outside a choice, or a `--dataloader` or `--bettor` path that
does not exist, SHALL be reported by the option parser with exit status 2. An error the library raises as a
`BuildError` or `ValueError` SHALL be printed as its message in a red panel with no traceback, and the command SHALL
exit with status 1, whether the error comes from building the selection, the model or the extraction settings.

#### Scenario: An unknown odds type
- **WHEN** the user runs `sportsbet dataloader train extract` with `--odds-type nonexistent`
- **THEN** the exit status is 1
- **AND** the output lists `market_average`
- **AND** the output contains no `Traceback`

#### Scenario: A selection that does not build
- **WHEN** the user runs `sportsbet dataloader params --stats football-data --alias nonsense`
- **THEN** the exit status is 1
- **AND** the output contains no `Traceback`

#### Scenario: Extraction settings that do not build
- **WHEN** the user runs `sportsbet dataloader train extract` with `--target-event-status inplay` and
  `--target-event-time soon`
- **THEN** the exit status is 1
- **AND** the output contains no `Traceback`

### Requirement: execution commands
The `execution` group SHALL hold `venue`, `run`, `markets`, `balance`, `status` and `cancel`, each taking `--venue` as a
`venue.py:VENUE` reference. `venue` prints the key, `cancels` and notes. `run` takes `-d`, `-b`, `--event`, `--stake`,
repeatable `--url`, `--live`, `--poll` (default `30s`) and `-o`. `status` and `cancel` take `--match`, `--market` and
`--selection`. `markets` takes `-d`. Their behavior is specified in execution/venue-contract and execution/event-run.

#### Scenario: A dry run prints the receipt
- **WHEN** the user runs `sportsbet execution run --venue venue.py:VENUE -d loader.pkl -b model.pkl --event "Arsenal
  vs Chelsea" --stake 10 --url https://book.invalid/ac` with no `--live`
- **THEN** the run is unarmed, the exit status is 0, and the receipts are printed under `Receipts`

#### Scenario: Status of one bet
- **WHEN** the user runs `sportsbet execution status --venue venue.py:VENUE --match "Arsenal vs Chelsea" --market
  home_win --selection Arsenal`
- **THEN** the exit status is 0
- **AND** the output contains `What the venue holds`

### Requirement: execution outputs
`execution run` SHALL print the receipts under `Receipts` when there are any, and with `-o` SHALL write them to
`<directory>/sports-betting-data/receipts.csv` with no index column. `markets` SHALL print `Markets`, `balance` SHALL
print the balance and the exposure in a panel, `status` SHALL print `What the venue holds`, and `cancel` SHALL print the
receipt detail in a panel. `run` SHALL log the run to the terminal while it runs.

#### Scenario: Receipts are written
- **WHEN** `execution run` returns receipts and `-o out` was given
- **THEN** `out/sports-betting-data/receipts.csv` holds them

### Requirement: execution page commands
The `execution page` group SHALL hold `read`, `act` and `fix`, each taking `--venue`. `read` takes optional `--url`,
`--selector` and `--depth` and prints the page snapshot. `act` takes a required `--url` and one of `--click`, `--type`
with `--text`, or `--select` with `--value`, and prints the resulting snapshot. `fix` takes `--url`, `--match` and
repeatable `--locator` as `name=locator` and prints the pinned match, URL and locators.

#### Scenario: act needs an action
- **WHEN** the user runs `sportsbet execution page act --venue venue.py:VENUE --url https://book.invalid` with no
  `--click`, `--type` or `--select`
- **THEN** the command fails with a usage error that names `--click`, `--type` with `--text`, and `--select` with
  `--value`

#### Scenario: A malformed locator
- **WHEN** the user passes `--locator stake` to `execution page fix`
- **THEN** the command fails with a usage error that shows the form `stake=textbox[name="Stake"]`

### Requirement: Venue kind matches the command
`execution run` and the `execution page` commands MUST be given a browser session, and `markets`, `balance`, `status`
and `cancel` MUST be given a venue with an API. A reference of the other kind SHALL fail with a usage error that says
which kind it is. A `--venue` reference without `:` or naming no venue SHALL be reported as a red panel with exit 1.

#### Scenario: A malformed venue reference
- **WHEN** the user runs `sportsbet execution venue --venue nofile`
- **THEN** the output says the reference must name a venue in a Python file, as in `venue.py:VENUE`
- **AND** the exit status is 1

### Requirement: Execution extra
The commands that drive a browser session, `execution run` and the `execution page` commands, SHALL require the
`execution` extra, installed with `pip install 'sports-betting[execution]'`. The `dataloader` and `evaluation` commands
SHALL need no extra.

#### Scenario: Data commands without extras
- **GIVEN** the library is installed without extras
- **WHEN** the user runs a `dataloader` or `evaluation` command
- **THEN** the command runs

### Requirement: Surface parity with the Python API
Every parameter of `BaseDataLoader.extract_train_data`, `BaseDataLoader.extract_exploration_data`, `DataLoader`,
`backtest` and `OddsApi` SHALL be reachable as an option of the matching command, except `self`, `param_grid`, `key`,
`classifier`, `X`, `Y`, `O` and `bettor`. The `OddsApi` parameters `key_env`, `markets`, `regions` and `moments` SHALL
be reached as `odds_key_env`, `odds_markets`, `odds_regions` and `odds_moments`.

#### Scenario: A new API parameter
- **WHEN** a parameter is added to `BaseDataLoader.extract_train_data` and not to `dataloader train extract`
- **THEN** the parity test fails and names the missing parameter

#### Scenario: Sources are configurable
- **WHEN** the options of `dataloader train extract` are read
- **THEN** they include `stats`, `odds`, `odds_key_env`, `odds_markets`, `odds_regions`, `odds_moments` and `aliases`

### Requirement: Execution parity
Every execution capability SHALL be reachable from the command line. `execution venue`, `markets`, `balance`, `status`
and `cancel` SHALL take a venue reference, the `execution page` commands SHALL mirror the browser tools of the MCP
server, and `cancel` SHALL name a bet by its match, market and selection.

#### Scenario: The browser primitives are commands
- **WHEN** the options of `execution page act` and `execution page fix` are read
- **THEN** `act` takes `--click`, `--type` and `--select`
- **AND** `fix` takes `--match` and `--locator`
