## ADDED Requirements

### Requirement: Basketball competitions
`OddsApi` SHALL cover the basketball competitions that the user's subscription covers. It SHALL map the
`basketball_euroleague` key to league `Euroleague`, division `1`, and SHALL map the other `basketball_*` keys to library
leagues.

#### Scenario: EuroLeague is in the catalogue
- **WHEN** a user asks `OddsApi` for the parameters it publishes
- **THEN** the result includes league `Euroleague` with division `1`

### Requirement: Priced moments follow the statistics
`OddsApi` SHALL price only the moments that the statistics carry, read from the schedule the dataloader hands it. It
SHALL NOT price a moment for which the statistics have no snapshot.

#### Scenario: Basketball buys no half-time price
- **WHEN** the statistics are EuroLeague snapshots with pre-play and post-play moments only
- **THEN** `OddsApi` requests no in-play price at minute 45

#### Scenario: Soccer still buys its in-play price
- **WHEN** the statistics are soccer snapshots that include half-time snapshots
- **THEN** `OddsApi` still requests the in-play price for those moments
