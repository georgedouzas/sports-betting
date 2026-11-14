## ADDED Requirements

### Requirement: Import execution errors from sportsbet.core
`ExecutionError`, `CancellationUnsupportedError`, `VenueBlockedError` and `CredentialError` SHALL be importable from
`sportsbet.core`, which holds every named exception of the library. `sportsbet.execution` SHALL NOT re-export them.
Execution code SHALL raise the same exceptions under the same conditions as before.

#### Scenario: Catch a credential error
- **GIVEN** the environment variable `VENUE_KEY` is not set
- **WHEN** a user calls `resolve(CredentialRef('VENUE_KEY'))` inside `try` with `except CredentialError` imported from
  `sportsbet.core`
- **THEN** the handler catches the error, and its message names `VENUE_KEY`

#### Scenario: The old import place
- **WHEN** a user runs `from sportsbet.execution import ExecutionError`
- **THEN** Python raises `ImportError`

### Requirement: Execution public names
`sportsbet.execution` SHALL export exactly `BaseVenue`, `BetIdentity`, `BrowserSession`, `CredentialRef`,
`FixedSession`, `PageSnapshot`, `PlacementIntent`, `PlacementReceipt`, `PlacementStatus`, `Placer`,
`build_receipts_frame`, `build_venue`, `execute_event`, `find_betting_moment` and `resolve`. `PlacementReceiptSchema` is
private and SHALL NOT be exported.

#### Scenario: Inspect the execution surface
- **WHEN** a user reads `sportsbet.execution.__all__`
- **THEN** it lists the 15 names and does not list `PlacementReceiptSchema`
