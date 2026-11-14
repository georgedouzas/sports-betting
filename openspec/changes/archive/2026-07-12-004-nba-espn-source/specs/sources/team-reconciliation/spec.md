## ADDED Requirements

### Requirement: NBA team names reconcile with the odds vendor
The library SHALL reconcile the NBA team names from `NBAStats` against the odds vendor's names without silently dropping
a club. Each of the 30 clubs SHALL pair with the vendor's name for it, and an alias SHALL be added only where a club
cannot be placed, with a stated reason.

#### Scenario: Every club is paired
- **WHEN** the two sources' names for the NBA are reconciled
- **THEN** every club is paired

#### Scenario: Unplaceable club fails loudly
- **WHEN** a club cannot be placed by the pairing or by an alias
- **THEN** reconciliation fails and names the club, rather than dropping its games
