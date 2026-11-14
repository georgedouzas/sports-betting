# Proposal

## Why

`resolve_odds` joins odds to statistics on league, division, year and the two teams, and ignores the kick-off. The
statistics side is reduced to one row per pairing with `drop_duplicates(subset=MATCH_COLS)`, which keeps the first
meeting. A pairing that meets twice in a season with the same home team, as basketball teams do, gets every odds row of
both games stamped with the first game's date. The second game's odds then sit on the first game, and the second game
has none. Soccer is not affected in a league season, where a home pairing is played once.

Change 002 already stated that the kick-off is matched within a tolerance window. The code never did it.

## What Changes

- An odds row resolves to the statistics match of the same league, division, year and teams whose kick-off is nearest
  to the odds row's kick-off, within a tolerance of 36 hours.
- An odds row with no statistics kick-off within the tolerance is dropped, as an odds row of an unknown match is today.

## Non-goals

- Changing the team pairing or the aliases.
- Matching on kick-off alone, without the teams.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `sources/team-reconciliation`: odds resolution matches the kick-off as well as the teams.

## Impact

- Code: `src/sportsbet/sources/_resolver.py`.
- Tests: `tests/sources/` gains a case with one pairing that meets twice in a season.
- Results: basketball datasets built from `EuroLeagueStats` or `NBAStats` with `OddsApi` change, and are now right.
