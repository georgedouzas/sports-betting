# Proposal

## Why

The library supports basketball but knows only one competition. A user who wants the NBA cannot have it, because no
source carries its games. The NBA is the most priced and most modelled basketball league.

Everything else already exists. The odds vendor already prices the NBA, and `basketball_nba` has been mapped since
feature 003. The dataloader is sport-agnostic, the market is two-way, the reconciliation pairs rosters and the store
caches. The only missing part is a source that reports the games.

This feature adds that one source and changes nothing else. It is the first independent test of the engine move in
feature 003: if adding a league requires changing the engine, the engine is wrong.

Research against the live feeds decided the provider. The NBA's official feed back-fills results once a year, months
after a season ends, so it cannot serve a season in progress. ESPN's public feed is unofficial but live, needs no key
and no headers, and is long-stable.

## What Changes

- Add `NBAStats`, a free statistics source for the NBA backed by ESPN's public JSON endpoints. It takes no arguments and
  needs no credential.
- Export `NBAStats` from `sportsbet.datasets`.
- Use the NBA through the existing `BasketballDataLoader` by passing `stats=NBAStats()`. No new dataloader and no new
  sport.
- Change no existing source, dataloader, fetch layer or preparation engine (SC-007).

## Capabilities

### New Capabilities

- `sources/nba`: the NBA statistics source, its seasons, its monthly requests, its exhibition and pre-season filter,
  its tip-off times, its form features and its two-way outcome.

### Modified Capabilities

- `sources/team-reconciliation`: adds the reconciliation of NBA team names against the odds vendor.
- `interfaces/cli`: adds the NBA statistics source to the command line, as the EuroLeague source is reached.

## Impact

- Code: new `src/sportsbet/datasets/_sources/_nba.py`. `src/sportsbet/datasets/__init__.py` gains one export.
- Files that show a zero diff by design: `_fetch.py`, `_store.py`, `_sources/_base.py`, `_base/_sourced.py`,
  `_basketball/_dataloader.py`, `_sources/_odds_api.py`, `_sources/_euroleague.py`, `_sources/_football_data.py`, and
  `_resolver.py` unless a club cannot be placed.
- API: new public name `NBAStats`.
- Tests: new `tests/datasets/test_nba.py`. Every existing test passes unmodified. No test touches the network.
- Docs: `docs/overview/user_guide/sources.md` gains the NBA row, `docs/overview/user_guide/dataloader.md` gains
  basketball's second league, and `CHANGELOG.md` gains an entry. The public-API docs audit still finds an example for
  every public name.
- Dependencies: none added. JSON parsing uses the standard library.
- Data: the project ships no NBA data.
- Specification quality checklist: every item passed after a second pass. The first draft named the feed, the endpoints
  and the exhibition field, and these moved to the research. SC-007 keeps its names for the engine, the fetch layer and
  the dataloaders on purpose, because "no engine change" is the outcome being claimed. FR-014 looks like an
  implementation constraint, but it is the requirement that rejects the official archive.
