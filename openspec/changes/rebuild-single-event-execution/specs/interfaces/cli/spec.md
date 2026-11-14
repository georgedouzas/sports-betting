# Spec Delta

## MODIFIED Requirements

### Requirement: execution commands
The `execution` group SHALL hold `venue`, `run`, `markets`, `balance`, `status` and `cancel`, each taking `--venue` as a
`venue.py:VENUE` reference. `run` takes `-d`, `-b`, `--event`, `--stake`, `--lead` (default `5min`), `--poll` (default
`5min`), `--ledger`, `--live` and `-o`. `status` and `cancel` take `--match`, `--market` and `--selection`. `markets`
takes `-d`. Their behavior is specified in execution/venue-contract and execution/event-run.

#### Scenario: A dry run prints the receipt
- **WHEN** the user runs `sportsbet execution run --venue venue.py:VENUE -d loader.pkl -b model.pkl --event "Arsenal
  vs Chelsea" --stake 10 --ledger bets.jsonl` with no `--live`
- **THEN** the run is unarmed, the exit status is 0, and the steps and the receipt are printed

#### Scenario: Status of one bet
- **WHEN** the user runs `sportsbet execution status --venue venue.py:VENUE --match "Arsenal vs Chelsea" --market
  home_win --selection Arsenal`
- **THEN** the exit status is 0
- **AND** the output contains `What the venue holds`

### Requirement: execution outputs
`execution run` SHALL print the run's steps and outcome, and its receipt when there is one. With `-o` it SHALL write the
steps to `<directory>/sports-betting-data/run_steps.csv` and the receipts to `receipts.csv` beside it, with no index
column. `markets`, `balance`, `status` and `cancel` print as before. `run` SHALL log the run to the terminal as it runs.

#### Scenario: Receipts are written
- **WHEN** `execution run` finishes and `-o out` was given
- **THEN** `out/sports-betting-data/run_steps.csv` holds the steps and `out/sports-betting-data/receipts.csv` holds the
  receipts

### Requirement: Venue kind matches the command
`execution run`, `markets`, `balance`, `status` and `cancel` SHALL accept any venue, a browser venue included. The
`execution page` commands SHALL accept a browser session or a browser venue, using its session. A reference of another
kind SHALL fail with a usage error. A malformed `--venue` reference SHALL be reported as a red panel with exit 1.

#### Scenario: A malformed venue reference
- **WHEN** the user runs `sportsbet execution venue --venue nofile`
- **THEN** the output says the reference must name a venue in a Python file, as in `venue.py:VENUE`
- **AND** the exit status is 1

#### Scenario: Operation a venue does not support
- **WHEN** the user runs `sportsbet execution balance` for a venue that does not read balances
- **THEN** the output says the venue does not read its balance and the exit status is 1

### Requirement: Execution parity
Every execution capability SHALL be reachable from the command line. `execution venue`, `markets`, `balance`, `status`
and `cancel` SHALL take a venue reference, the `execution page` commands SHALL mirror the browser read and act tools of
the MCP server, and `cancel` SHALL name a bet by its match, market and selection.

#### Scenario: The browser primitives are commands
- **WHEN** the commands of the `execution page` group are read
- **THEN** `act` takes `--click`, `--type` and `--select`
- **AND** the group has no `fix` command

## ADDED Requirements

### Requirement: Exploring a page
The `execution page` group SHALL hold `read` and `act`, each taking `--venue`. `read` takes optional `--url`,
`--selector` and `--depth` and prints the page snapshot. `act` takes a required `--url` and one of `--click`, `--type`
with `--text`, or `--select` with `--value`, checks them before opening the browser, and prints the resulting snapshot.

#### Scenario: act needs an action
- **WHEN** the user runs `sportsbet execution page act --venue venue.py:VENUE --url https://book.invalid` with no
  `--click`, `--type` or `--select`
- **THEN** the command fails with a usage error that names `--click`, `--type` with `--text`, and `--select` with
  `--value`, and no browser is opened

## REMOVED Requirements

### Requirement: execution page commands
**Reason**: It held `execution page fix`, whose pins lived only for the one command and never reached a run.

**Migration**: Use `execution page read` and `execution page act`, now specified under "Exploring a page", to find the
controls, and write them into a `BaseBrowserVenue` subclass.
