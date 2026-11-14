## MODIFIED Requirements

### Requirement: Odds resolution

`resolve_odds(stats, odds, aliases=None)` SHALL return the odds rows whose match, identified by league, division,
year, the two teams and the kick-off, exists in the statistics. An odds row SHALL resolve to the statistics match of the
same league, division, year and teams whose kick-off is nearest to its own, within 36 hours, and an odds row with no
such match MUST be dropped. Pairing MUST run separately for each league, division and year. Each returned row MUST
carry the statistics spelling of the teams and the statistics `date`, with the columns of `odds` in their original
order.

#### Scenario: Statistics spelling carried

- **GIVEN** statistics naming `Man United` and odds naming `Manchester United` for the same match
- **WHEN** a caller calls `resolve_odds(stats, odds)`
- **THEN** the returned row has `home_team` `Man United` and keeps the price of the odds

#### Scenario: Every sample odds row resolved

- **WHEN** a caller resolves the test odds sample against the test statistics sample
- **THEN** the result has as many rows as the odds, and its home teams are all spelled as in the statistics

#### Scenario: One pairing meets twice in a season

- **GIVEN** statistics with `Olympiacos` at home to `Real Madrid` on `2025-10-16` and on `2026-03-05`, and odds with
  one row for each of those days
- **WHEN** a caller calls `resolve_odds(stats, odds)`
- **THEN** each odds row carries the date of its own game and keeps its own price

#### Scenario: No kick-off near the odds

- **GIVEN** an odds row for a pairing whose nearest statistics kick-off is three days away
- **WHEN** a caller calls `resolve_odds(stats, odds)`
- **THEN** that odds row is not returned
