## ADDED Requirements

### Requirement: Shared data model names in sportsbet.core
`sportsbet.core` SHALL export exactly `ALIASES`, `BoolData`, `BuildError`, `CancellationUnsupportedError`,
`CredentialError`, `Data`, `EVENT_COLS`, `ExecutionError`, `FixturesData`, `GROUPS_COLS`, `IDENTITY_COLS`,
`IDENTITY_FIELDS`, `Indices`, `MATCH_COLS`, `NON_PREPLAY_EVENT_STATUSES`, `NotExtractedError`,
`PREPLAY_EVENT_STATUSES`, `ParamGrid`, `STATUSES`, `STATUS_RANK`, `TEAMS_COLS`, `TrainData`, `VenueBlockedError`,
`format_event_time`, `load_object` and `parse_event_time`.

#### Scenario: Inspect the core surface
- **WHEN** a user reads `sportsbet.core.__all__`
- **THEN** it lists the 26 names

#### Scenario: Removed and private names
- **WHEN** a user runs `from sportsbet.core import Param`, `Schema`, `OutputsMapping` or `DATE_COLS`
- **THEN** Python raises `ImportError`
