# Proposal

## Why

The library supports only soccer, but almost nothing in it depends on soccer. Feature 002 made the data sources
injectable. The extraction engine derives the providers, the markets, the features and the mutually exclusive markets
from the data it receives. A sport without a draw therefore gets a two-way outcome without any setting. The match
identity (`league`, `division`, `year`, `home_team`, `away_team`) already accepts `league='Euroleague'`.

The missing part is a source of basketball statistics. The EuroLeague publishes an official, public API that needs no
key. It carries final scores, schedules and tip-off times, which is everything the soccer feed provides. A user's own
commercial odds subscription already covers the competition. A statistics source plus that odds source gives the
library a second sport.

Research against the live API found two facts that changed the scope. The API publishes its `date` field in Central
European Time for every venue, and documents no time zone. Basketball has no fixed totals line, so the first cut covers
the head-to-head market only.

## What Changes

- Add `EuroLeagueStats`, a free statistics source over the EuroLeague's official public API that needs no key.
- Add `BasketballDataLoader`, which uses the same engine as `SoccerDataLoader` with `EuroLeagueStats` as its default
  statistics source and no default odds source.
- Move the sport-agnostic preparation, store, cost estimate and reconciliation logic from `SoccerDataLoader` onto a
  shared base. Soccer behavior stays unchanged and every existing test passes unmodified.
- Keep `sportsbet.datasets.market_outcomes` public while it moves to a sport-neutral module.
- Map the `basketball_euroleague` key and the other `basketball_*` keys of `OddsApi` to library leagues.
- Make `OddsApi` price only the moments the statistics carry, instead of soccer's default moments.
- Treat every odds value column as time-varying, so a source with a single provider keeps the provider in its column
  names.
- Raise a clear error when a dataloader has no markets to predict, instead of an internal schema error.
- Add one team alias, `Olimpia Milano` to `EA7 Emporio Armani Milan`.
- Export `BasketballDataLoader` and `EuroLeagueStats` from `sportsbet.datasets`.

## Capabilities

### New Capabilities

- `sources/euroleague`: the EuroLeague statistics source, its seasons, its games, its tip-off times, its form features
  and its two-way outcome.

### Modified Capabilities

- `dataloaders/data-extraction`: adds the basketball dataloader, the two-way outcome derived from the data, the
  no-markets error and the rule that odds columns are never fixed.
- `sources/odds-api`: adds the basketball league mapping and moments derived from the statistics.
- `sources/team-reconciliation`: adds per-season basketball reconciliation and the `Olimpia Milano` alias.
- `interfaces/cli`: adds the basketball dataloader to the command line.

## Impact

- Code: new `src/sportsbet/datasets/_sources/_euroleague.py`, new `src/sportsbet/datasets/_basketball/`, new
  `src/sportsbet/datasets/_base/_sourced.py`, new `src/sportsbet/datasets/_utils.py`. `_soccer/_dataloader.py` shrinks
  to its default sources. `_sources/_odds_api.py` gains the `basketball_*` mapping.
- API: new public names `BasketballDataLoader` and `EuroLeagueStats`. `market_outcomes` keeps its public path.
- Tests: new `tests/datasets/test_euroleague.py` and `tests/datasets/test_basketball.py`. No test touches the network.
- Docs: `docs/overview/user_guide/dataloader.md`, `docs/overview/user_guide/index.md` and the README gain the sport, the
  source, the two-way market and the statement that basketball needs a paid odds key.
- Dependencies: none added. JSON parsing uses the standard library.
- Data: the project publishes no basketball data.
- Specification quality checklist: every item passed. Named products appear only in the context, and FR-014 is written
  as a prohibition on assuming a time zone rather than as a value.
