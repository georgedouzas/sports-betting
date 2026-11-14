# interfaces/mcp-server Specification

## Purpose

The `sportsbet-mcp` server lets an agent drive the library through MCP tools. Each tool takes in its arguments what the
matching `sportsbet` command takes as options and returns JSON records instead of console tables.

## Requirements

### Requirement: Server entry point
The library SHALL install a `sportsbet-mcp` console command that runs an MCP server named `sportsbet` over standard
input and output. The server SHALL require the `mcp` extra, installed with `pip install 'sports-betting[mcp]'`. The
Python module `sportsbet.mcp` SHALL expose the server as `server` and the command as `run`.

#### Scenario: The server lists its tools
- **GIVEN** the `mcp` extra is installed
- **WHEN** a client lists the tools of `sportsbet.mcp.server`
- **THEN** it receives the tools of this capability

### Requirement: Tool set
The server SHALL offer, in this order, `available_params`, `odds_types`, `extract_train_data`,
`extract_exploration_data`, `extract_fixtures_data`, `backtest`, `fit`, `bet`, `execution_venue_info`,
`execution_authenticate`, `execution_read_balance`, `execution_list_markets`, `execution_read_status`,
`execution_cancel`, `execution_run`, `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`,
`browser_select` and `browser_fix`.

#### Scenario: Every tool is listed
- **WHEN** a client lists the tools
- **THEN** the names are exactly these twenty-one, in this order

### Requirement: Tools are configured by their arguments
Every tool SHALL take everything it needs as arguments. No tool SHALL take a `config_path` or read a configuration file.
A tool that selects data SHALL take `stats`. A tool that reads saved data SHALL take `dataloader`. Every other tool
SHALL take `venue`.

#### Scenario: The argument that locates the work
- **WHEN** a client reads the input schema of each tool
- **THEN** `available_params`, `odds_types`, `extract_train_data` and `extract_exploration_data` take `stats`
- **AND** `extract_fixtures_data`, `backtest`, `fit` and `bet` take `dataloader`
- **AND** every `execution_*` and `browser_*` tool takes `venue`

### Requirement: Data selection arguments
`available_params`, `odds_types`, `extract_train_data` and `extract_exploration_data` SHALL take `stats` (required),
`odds`, `leagues`, `divisions`, `years`, `odds_key_env` (default `ODDS_API_KEY`), `odds_markets`, `odds_regions`,
`odds_moments` and `aliases`, with the meanings and the forms that `build_dataloader` gives them.

#### Scenario: An unknown statistics source
- **WHEN** a client calls `available_params` with `stats='quidditch'`
- **THEN** the call fails with a message that lists `euroleague`, `football-data` and `nba`

### Requirement: Secrets are never arguments
No tool SHALL take a key, password, secret, token, app key, username or credential as an argument. A data tool SHALL
name the environment variable that holds the odds key through `odds_key_env`. An execution tool SHALL name a venue,
and the venue SHALL name the variables that hold its own secrets.

#### Scenario: No tool takes a secret
- **WHEN** a client reads the input schema of every tool
- **THEN** no schema has a property named `key`, `password`, `secret`, `token`, `app_key`, `username` or `credential`
- **AND** every data selection tool has `odds_key_env`

### Requirement: Records
A tool that returns tabular data SHALL return a list of records, one per row in row order, each mapping column names to
values. A missing value SHALL be `null`. An absent or empty frame SHALL be an empty list. The records SHALL hold the
frame columns and not the frame index.

#### Scenario: Missing values
- **WHEN** a returned frame holds a missing value
- **THEN** the matching record value is `null`

### Requirement: available_params
`available_params` SHALL take the data selection arguments and return a list of objects, each one a league, division
and season combination the statistics source publishes.

#### Scenario: Parameters are returned
- **WHEN** a client calls `available_params` with `stats='football-data'`, `odds='football-data'` and
  `leagues=['England']`
- **THEN** the result is a non-empty list

### Requirement: odds_types
`odds_types` SHALL take the data selection arguments, download the selected data, and return the sorted list of odds
type names the selection carries.

#### Scenario: Odds types are returned
- **WHEN** a client calls `odds_types` with a football-data selection
- **THEN** the result is a non-empty list of names

### Requirement: extract_train_data
`extract_train_data` SHALL take the data selection arguments, `odds_type`, `drop_na_thres`, `target_event_status`,
`target_event_time`, `input_event_status`, `input_event_time` and an optional `output` file path. It SHALL return an
object with the records `X`, `Y` and `O` and the `output` it was given. With `output` it SHALL save the dataloader
there.

#### Scenario: The training data is extracted and saved
- **WHEN** a client calls `extract_train_data` with a football-data selection, `odds_type='market_average'` and
  `output='loader.pkl'`
- **THEN** `X`, `Y` and `O` are non-empty
- **AND** `output` is `loader.pkl` and the file holds the saved dataloader

### Requirement: extract_exploration_data
`extract_exploration_data` SHALL take the same arguments as `extract_train_data` except `output`, and return an object
with the records `X` of the features alone, with no targets and no odds. It SHALL ignore `odds_type`.

#### Scenario: The moment arguments are reachable
- **WHEN** a client reads the input schema of `extract_exploration_data`
- **THEN** it has `target_event_status`, `target_event_time`, `input_event_status`, `input_event_time` and
  `drop_na_thres`

### Requirement: extract_fixtures_data
`extract_fixtures_data` SHALL take `dataloader`, the path of a dataloader saved by `extract_train_data`, and return an
object with the records `X` and `O` of the games not yet played, in the shape the training data took.

#### Scenario: Fixtures from the saved dataloader
- **WHEN** a client calls `extract_fixtures_data` with the path `extract_train_data` saved
- **THEN** the result has `X`

### Requirement: backtest
`backtest` SHALL take `dataloader`, `model`, `cv` (default 3, time-ordered folds), `n_jobs` (default -1) and `verbose`
(default 0). It SHALL backtest the model on the saved training data and return the backtesting results as records.
`model` SHALL be a scikit-learn estimator written as a Python expression or a bettor named as `models.py:BETTOR`.

#### Scenario: The model expression reaches the model
- **WHEN** a client backtests `OddsComparisonBettor(betting_markets=["home_win"])` and `OddsComparisonBettor()` on the
  same saved dataloader
- **THEN** the first record of the single-market backtest has fewer `Number of bets` than that of the full backtest

### Requirement: fit
`fit` SHALL take `dataloader`, `output` and `model`. It SHALL fit the model on the saved training data, save the fitted
bettor to `output`, and return an object with `output` and `betting_markets`, the markets the fitted bettor bets on.

#### Scenario: The fitted model is saved
- **WHEN** a client calls `fit` with `model='OddsComparisonBettor()'` and `output='model.pkl'`
- **THEN** the result `output` is `model.pkl`
- **AND** `betting_markets` is non-empty

### Requirement: bet
`bet` SHALL take `dataloader` and `bettor`, the path of a model saved by `fit`. It SHALL return one record per game not
yet played, holding `date`, `home_team`, `away_team` and one boolean per betting market. When there are no games or no
odds for them, it SHALL return an empty list.

#### Scenario: Value bets from the fitted model
- **WHEN** a client calls `bet` with a saved dataloader and a saved model
- **THEN** the result is a list

### Requirement: Download once and reuse
`extract_fixtures_data`, `backtest`, `fit` and `bet` SHALL read the dataloader that `extract_train_data` saved, and
SHALL NOT build a dataloader from a selection. `backtest` and `bet` SHALL NOT take `stats`. `bet` SHALL read the model
`fit` saved through `bettor` and SHALL NOT take `model`.

#### Scenario: Nothing after the extract builds a dataloader
- **GIVEN** `extract_train_data` saved a dataloader
- **WHEN** a client calls `backtest`, `fit`, `bet` and `extract_fixtures_data` on it
- **THEN** no dataloader is built from a selection

### Requirement: Fetching inside a tool
A data or model tool SHALL run its work off the server event loop, so that a source can open its own event loop to
fetch while the tool is answered.

#### Scenario: A source that opens an event loop
- **WHEN** `extract_train_data` runs a dataloader whose extraction calls `asyncio.run`
- **THEN** the extraction completes and the tool returns

### Requirement: Error reporting
A tool that raises SHALL fail with a tool error whose message is `Error executing tool <name>: <message>`, where the
message is the one the library raised. Arguments that do not match the input schema, such as a missing `stats`, SHALL
fail with a tool error that names the invalid argument.

#### Scenario: A model that does not exist
- **WHEN** a client calls `backtest` with `model='wishful'` on a saved dataloader
- **THEN** the call fails with a message that says `wishful` is not a model and shows how to write one

### Requirement: Execution tools
`execution_venue_info(venue)` SHALL return `key`, `can_cancel`, `url` and `notes`. `execution_authenticate(venue)`
returns `venue` and `authenticated`. `execution_read_balance(venue)` returns `balance` and `exposure`.
`execution_list_markets(venue, matches)` and `execution_read_status(venue, match, market, selection)` return records.
`execution_cancel` takes the same arguments as `execution_read_status` and returns `status` and `detail`.

#### Scenario: The notes of a browser session
- **GIVEN** `venue.py` defines a `BrowserSession` with a key, a url and notes
- **WHEN** a client calls `execution_venue_info` with `venue='venue.py:VENUE'`
- **THEN** the result holds that key, url and notes

#### Scenario: Status of one bet
- **WHEN** a client calls `execution_read_status` for a bet the venue holds
- **THEN** the result is one record with the bet stake

### Requirement: execution_run
`execution_run` SHALL take `venue`, `dataloader`, `bettor`, `event`, `stake`, `urls` (default none), `live` (default
false), `poll` (default `30s`) and an optional `output` directory. It SHALL return the receipts as records and, with
`output`, write them to `<output>/sports-betting-data/receipts.csv`. The run itself is specified in execution/event-run.

#### Scenario: A dry run returns the bet it would make
- **WHEN** a client calls `execution_run` with `live=False`
- **THEN** the result holds the receipt with `status` `dry_run`, staking nothing

### Requirement: Browser tools
`browser_navigate(venue, url)`, `browser_snapshot(venue, selector, depth)`, `browser_click(venue, ref)`,
`browser_type(venue, ref, text)` and `browser_select(venue, ref, value)` SHALL return an object with the page snapshot
`yaml` and the page `url`. `browser_fix(venue, match, locators)` SHALL return `match`, `url` and `locators`. The browser
session is specified in execution/venue-contract.

#### Scenario: A ref from a snapshot
- **WHEN** a client calls `browser_navigate` and then `browser_click` with a ref from the returned snapshot
- **THEN** `browser_click` returns the snapshot of the page the click produced

### Requirement: Browser session lifetime
The browser tools SHALL open the browser session a venue reference names on the first call and keep it open for the
life of the server, so that later browser calls with the same reference act on the same page.

#### Scenario: Two calls share a page
- **WHEN** a client calls `browser_navigate` and then `browser_snapshot` with the same `venue`
- **THEN** the snapshot is of the page `browser_navigate` opened

### Requirement: Venue kind matches the tool
`execution_authenticate`, `execution_read_balance`, `execution_list_markets`, `execution_read_status` and
`execution_cancel` MUST be given a venue with an API, and authenticate it before they act. `execution_run` and the
browser tools MUST be given a browser session. A reference of the other kind SHALL fail with an `ExecutionError`
message.

#### Scenario: A browser session given to a venue tool
- **WHEN** a client calls `execution_read_balance` with a reference to a `BrowserSession`
- **THEN** the call fails with a message that says it is a browser session and points to the browser tools

### Requirement: Execution extra
`execution_run` and the `browser_*` tools SHALL require the `execution` extra, installed with
`pip install 'sports-betting[execution]'`. The data and model tools SHALL need only the `mcp` extra.

#### Scenario: Data tools without the execution extra
- **GIVEN** only the `mcp` extra is installed
- **WHEN** a client calls `extract_train_data`
- **THEN** the tool runs

### Requirement: Surface parity with the command line
Each command SHALL have a tool that accepts every option of the command, with `dataloader_path` as `dataloader`,
`bettor_path` as `bettor`, `venue_ref` as `venue`, and `output` exempt. The pairs are `params`, `odds-types`,
the three `extract` commands, `backtest`, `fit`, `bet` and `execution run`, `balance`, `status`, `cancel` and `venue`.

#### Scenario: A command gains an option
- **WHEN** an option is added to `sportsbet evaluation backtest` and not to the `backtest` tool
- **THEN** the parity test fails and names the missing argument

#### Scenario: The command pairs
- **WHEN** the tools are listed
- **THEN** `available_params`, `odds_types`, `extract_train_data`, `extract_exploration_data`, `extract_fixtures_data`,
  `backtest`, `fit`, `bet`, `execution_run`, `execution_read_balance`, `execution_read_status`, `execution_cancel`
  and `execution_venue_info` are present

### Requirement: An agent chooses the moment
`extract_train_data` and `extract_exploration_data` SHALL take `target_event_status`, `target_event_time`,
`input_event_status`, `input_event_time` and `drop_na_thres`, so an agent chooses the moment a model bets at.
`extract_train_data` and `fit` SHALL take `output`, so that their results are saved once and read back.

#### Scenario: The moment and output arguments
- **WHEN** a client reads the input schemas
- **THEN** `extract_train_data` has the five moment arguments and `output`
- **AND** `fit` has `output`
