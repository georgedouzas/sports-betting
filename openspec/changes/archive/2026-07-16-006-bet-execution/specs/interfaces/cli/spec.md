## ADDED Requirements

### Requirement: Execution venue reference
The `sportsbet execution` group SHALL take every parameter with `--` and no positional argument. Each command SHALL
take `--venue` naming a user venue as `venue.py:VENUE` and resolve it with `build_venue`. No option SHALL carry a
credential. `sportsbet execution venue --venue venue.py:VENUE` SHALL print the venue's key, URL and notes.

#### Scenario: Showing a venue
- **WHEN** the user runs `sportsbet execution venue --venue venue.py:VENUE`
- **THEN** the command prints the venue's key, URL and notes as configured

### Requirement: Execution quote and place commands
`sportsbet execution quote` SHALL write the itemised batch and its totals to `--output`. `sportsbet execution place`
SHALL read it with `--quote` and take `--confirm-stake`, `--confirm-exposure`, `--max-stake` and `--max-exposure`. It
MUST have no `--dry-run` flag. Without both confirmations it SHALL print the full quote, stake nothing and exit
non-zero.

#### Scenario: Place without confirmation
- **WHEN** the user runs `sportsbet execution place --venue venue.py:VENUE --quote quote.json` with no confirmations
- **THEN** the command prints the quote, reports `DRY_RUN` on every receipt, stakes nothing, and exits non-zero

#### Scenario: Place with wrong figures
- **WHEN** the user passes `--confirm-stake` and `--confirm-exposure` that differ from the quote
- **THEN** every receipt is `REFUSED_UNCONFIRMED` and the output names the real figures

### Requirement: Execution reading commands
The group SHALL provide `sportsbet execution markets`, `sportsbet execution balance`, `sportsbet execution status` and
`sportsbet execution cancel`, which read the markets, the balance and open exposure, and the status of placed bets,
and cancel a bet named by `--identity`.

#### Scenario: Reading the balance
- **WHEN** the user runs `sportsbet execution balance --venue venue.py:VENUE`
- **THEN** the command prints the balance and the open exposure

### Requirement: Execution page commands
The group SHALL provide `sportsbet execution page read`, `sportsbet execution page act` and
`sportsbet execution page fix`, which read a page snapshot, act on a ref, and pin a session's locators for a match.

#### Scenario: Pinning a session from the command line
- **WHEN** the user runs `sportsbet execution page fix` with a match and role and name locators
- **THEN** the command pins the locators and prints the pinned session

### Requirement: Execution commands without the extra
When the `execution` extra is not installed, every `sportsbet execution` command SHALL report the extra to install
and MUST NOT raise an `ImportError` traceback.

#### Scenario: Running an execution command without the extra
- **WHEN** the user runs `sportsbet execution venue` in an environment without the `execution` extra
- **THEN** the command names the `execution` extra to install

### Requirement: Execution surface parity
Every execution capability SHALL be reachable from the Python API, the CLI and the MCP server, and the parity test MUST
cover the execution group, so no surface holds an execution capability the others cannot reach.

#### Scenario: Parity test covers execution
- **WHEN** the parity test runs
- **THEN** every `sportsbet execution` command has a matching MCP tool and Python API entry
