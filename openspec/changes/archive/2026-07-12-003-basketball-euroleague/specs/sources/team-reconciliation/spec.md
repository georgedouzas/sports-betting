## ADDED Requirements

### Requirement: Basketball rosters reconcile per season
When basketball statistics and odds come from two sources, the library SHALL reconcile the two rosters within each
season and never across the whole history, because clubs change between seasons. It SHALL report the outcome as matched
and unmatched counts and rates, and SHALL fail loudly when a game is left without odds above the configured tolerance.

#### Scenario: Reconciliation is reported
- **WHEN** a user prepares a basketball dataloader with a statistics source and an odds source
- **THEN** `reconciliation_` reports the matched count, the unmatched rate and the unmatched names on each side

#### Scenario: Unplaceable club fails loudly
- **WHEN** neither the pairing nor an alias can place a club
- **THEN** extraction raises `UnmatchedError` naming the club, and does not emit its games without odds

#### Scenario: User alias places a club
- **WHEN** the pairing cannot place a club and the user supplies an alias for it
- **THEN** the club reconciles and its games carry odds

#### Scenario: Unpriced game is counted
- **WHEN** the statistics list a game that the odds vendor never priced
- **THEN** the game is counted as unmatched rather than silently dropped

### Requirement: Olimpia Milano alias
The library SHALL ship an alias that pairs the odds vendor's `Olimpia Milano` with the EuroLeague feed's
`EA7 Emporio Armani Milan`, because the two names share only the city and no pairing can place them.

#### Scenario: Milan pairs through the alias
- **WHEN** EuroLeague statistics and EuroLeague odds are reconciled for a season that includes Milan
- **THEN** `Olimpia Milano` pairs with `EA7 Emporio Armani Milan`
