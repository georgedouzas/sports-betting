## ADDED Requirements

### Requirement: Single-event execution run tool
The MCP server SHALL expose `execution_run` for the single-event unit, mirroring `sportsbet execution run` with the
parameters `venue`, `dataloader`, `bettor`, `event`, `stake`, `urls`, `live`, `poll` and `output`. Without `live` it
SHALL stake nothing. It SHALL place through the default placer over the pinned controls.

#### Scenario: An agent runs one event unarmed
- **WHEN** an agent calls `execution_run` with an event, a stake and candidate URLs and without `live`
- **THEN** the tool monitors, decides, and returns no placed bet

#### Scenario: The tool matches its command
- **WHEN** the parity test compares `sportsbet execution run` with `execution_run`
- **THEN** the tool can be told everything the command can be told

## REMOVED Requirements

### Requirement: Execution quote and place tools
**Reason**: The batch quote-and-confirm model is retired, together with `execution_quote` and `execution_place`.
**Migration**: Call `execution_run` with the event, the stake and the candidate URLs, and set `live` to place for real.
