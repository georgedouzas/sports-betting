## Purpose

Join statistics and odds that come from different providers into one match identity, report how well the join worked,
and refuse a dataset whose odds are missing beyond a tolerance.

## ADDED Requirements

### Requirement: Reconcile only when the sources differ
The dataloader SHALL reconcile statistics and odds only when they come from different sources. When both come from the
same upstream, such as `FootballDataStats` with `FootballDataOdds`, the identities are equal by construction and no
reconciliation MUST run.

#### Scenario: Free path skips reconciliation
- **WHEN** a loader uses `FootballDataStats` and `FootballDataOdds`
- **THEN** no reconciliation runs and `reconciliation_` is not set

### Requirement: Match identity across sources
When the sources differ, the system SHALL map each source's competition key to the canonical `league` and `division`,
reconcile team names, and match the kick-off within a tolerance window, producing one identity of `league`, `division`,
`year`, `date`, `home_team` and `away_team`.

#### Scenario: Different timezone and naming
- **WHEN** the odds source names `Man United` as `Manchester United` and states kick-off in another timezone
- **THEN** the statistics row and the odds row join as the same match

### Requirement: Roster pairing within a league-season
Team names SHALL be paired within each `(league, division, year)` group. Exact matches pair first. Remaining names pair
by token-prefix similarity only when the best candidate exceeds a similarity floor and beats the runner-up by a clear
margin. The last unpaired name MUST NOT be paired for being last, and unrelated names MUST NOT be paired.

#### Scenario: Abbreviated names pair
- **WHEN** the statistics hold `Wolves` and `Nott'm Forest` and the odds hold `Wolverhampton Wanderers` and
  `Nottingham Forest`
- **THEN** each abbreviated name pairs with its full name

#### Scenario: A club the statistics lack
- **WHEN** the odds hold `Sheffield Wednesday` and the statistics roster holds only `Sheffield United`
- **THEN** the two are not paired

#### Scenario: Unrelated leftovers
- **WHEN** the only unpaired names are `Liverpool` on one side and `Everton` on the other
- **THEN** they are not paired

### Requirement: Aliases as the escape hatch
`SoccerDataLoader` SHALL accept `aliases`, a mapping from the odds source's team names to the statistics source's names,
added to the names the library already knows, for clubs the pairing cannot place.

#### Scenario: Alias resolves a mismatch
- **WHEN** the user passes an alias for a club the pairing left unmatched
- **THEN** that club's odds join its matches

### Requirement: Reconciliation report
When the sources differ, the dataloader SHALL set `reconciliation_` to a `ReconciliationReport` with `matched`,
`unmatched_stats`, `unmatched_odds` and `unmatched_rate`, and with example unmatched rows that show the naming mismatch.
Unmatched matches MUST be reported even when the rate is within tolerance and MUST never be dropped without a count.

#### Scenario: Within tolerance
- **WHEN** the unmatched rate is above zero but within `max_unmatched_rate`
- **THEN** extraction succeeds and the report states the unmatched count and rate

### Requirement: Unmatched tolerance gate
`SoccerDataLoader` SHALL accept `max_unmatched_rate`, defaulting to `0.0`. When the proportion of statistics matches
without odds exceeds it, extraction MUST raise `UnmatchedError` naming example unmatched matches, and MUST NOT return a
dataset with silently missing odds.

#### Scenario: Above tolerance
- **WHEN** deliberately mismatched naming leaves matches without odds beyond the tolerance
- **THEN** extraction raises `UnmatchedError` with an accurate unmatched rate

### Requirement: Odds outside the competition
Odds rows for a club with no counterpart in the statistics roster, such as a cup opponent listed under a league key,
SHALL be dropped without firing the gate. Such a club MUST NOT be suggested as an alias unless a statistics club is left
for it to be.

#### Scenario: Cup opponent under a league key
- **WHEN** the odds source lists `Arsenal` against `Coventry City` under the Premier League key
- **THEN** that odds row is dropped, the gate does not fire and no alias is suggested for `Coventry City`
