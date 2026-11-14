# Proposal

## Why

The library shipped data, not just code. A Prefect flow on the `data` branch downloaded football-data.co.uk, transformed
it, and republished the result, bookmaker odds included, as CSVs that `SoccerDataLoader` fetched from GitHub. Nobody
granted the right to republish those odds. football-data.co.uk is free to download but reserves all rights, and every
commercial odds vendor forbids redistribution. Comparable open-source projects (kloppy, statsbombpy, soccerdata,
ScraperFC, socceraction) ship the fetching code and let the user pull the data. This library must do the same.

The same change unlocks a capability the library only claimed. `001-in-play-betting` gave in-play features and in-play
targets, but the only odds available were pre-match closing prices. An in-play bet could not be backtested, because the
price available at minute 45 was unknown. Time-stamped odds snapshots exist only behind a commercial API and a user's
own key, and this feature makes room for exactly that shape.

## What Changes

- `SoccerDataLoader` gains `stats`, `odds` and `store` constructor parameters, plus `aliases` and `max_unmatched_rate`.
  Each source owns its configuration. The dataloader stays a single public class.
- New free sources `FootballDataStats` and `FootballDataOdds` download football-data.co.uk on the user's machine and run
  the relocated transform locally, reproducing the mirrored dataset exactly.
- New keyed odds source `OddsApi(key, markets, regions)` adds time-stamped historical, upcoming and in-play prices.
- New source contract: `BaseSource`, `BaseStatsSource`, `BaseOddsSource`, `RawItem` and `RawPayload`. Sources declare
  the items they need and transform payloads. They do not fetch.
- New `LocalStore` keeps raw payloads forever, derived snapshot tables as Parquet, and a JSONL manifest.
- New `prepare(dry_run=False, refresh=False)` on the dataloader fetches what the store lacks, reports progress, and in a
  dry run reports what it would fetch and cost while spending nothing.
- **BREAKING**: extraction never fetches. `extract_train_data` and `extract_fixtures_data` raise `NotPreparedError` on
  an unprepared store. Every example that builds a `SoccerDataLoader` gains a `prepare()` call.
- **BREAKING**: `get_all_params` is removed from the dataloaders. Discovery moves to the source, for example
  `FootballDataStats().available_params()`. The dataloader selects only what both its sources publish.
- `date` becomes the kick-off instant in UTC for every source.
- New reconciliation of statistics and odds from different sources, with a `ReconciliationReport`, an `UnmatchedError`
  and a strict default `max_unmatched_rate=0.0`.
- Bettors derive their complementary market groups from the data instead of a hardcoded soccer list.
- The CLI gains a `prepare` subcommand with `--dry-run`. Its configuration gains `STATS`, `ODDS` and `STORE`, and
  `dataloader params` asks the statistics source. The GUI gains a preparation step and source-backed parameter pickers.
- The redistributing `data` branch and its Prefect flow are retired.

## Capabilities

### New Capabilities

- `sources/source-contract`: the source plugin contract, raw items and payloads, discovery on the source, the
  preparation step, the local store, the UTC kick-off convention and credential handling.
- `sources/football-data`: the free football-data.co.uk statistics and odds sources and the relocated transform.
- `sources/odds-api`: the keyed The Odds API odds source, its schedule-driven plan, cost model and coverage.
- `sources/team-reconciliation`: pairing statistics and odds from different sources, the report and the quality gate.

### Modified Capabilities

- `dataloaders/data-extraction`: the dataloader composes injected sources, extraction never fetches, the offline entry
  points need no preparation, parameter grid selection takes the intersection of the sources, and public parameter
  discovery is removed.
- `interfaces/cli`: the `prepare` subcommand, source-backed parameter discovery, the `STATS`, `ODDS` and `STORE`
  configuration keys, and the GUI preparation step.
- `evaluation/bettors`: complementary market groups are derived from the markets in the data.

## Impact

- Code: new `src/sportsbet/datasets/_sources/` (`_base.py`, `_football_data.py`, `_odds_api.py`),
  `src/sportsbet/datasets/_store.py`, `src/sportsbet/datasets/_resolver.py` and `src/sportsbet/datasets/_fetch.py`.
  Changed `src/sportsbet/datasets/_base/_dataloader.py`, `src/sportsbet/datasets/_soccer/_dataloader.py`,
  `src/sportsbet/datasets/__init__.py`, the bettor base in `src/sportsbet/evaluation/`, `src/sportsbet/cli/_data.py`
  and `src/sportsbet/gui/app/` including `states.py`. The `_snapshots()` contract of `BaseDataLoader` is unchanged.
- API: new exports `FootballDataStats`, `FootballDataOdds`, `OddsApi`, `LocalStore` and the source, store and
  reconciliation types. `get_all_params` is removed. Extraction requires `prepare()`.
- Tests: a session socket guard in `tests/conftest.py`, and new `tests/datasets/test_equivalence.py`,
  `test_sources.py`, `test_store.py`, `test_resolver.py` and `test_football_data.py`, with recorded payloads under
  `tests/samples/odds_api/` and a committed fingerprint `tests/samples/reference_fingerprint.json`.
- Docs: `docs/overview/user_guide/dataloader.md`, `docs/overview/user_guide/in_practice.md`,
  `docs/overview/user_guide/bettor.md`, `docs/examples/plot_soccer_data.py`,
  `docs/examples/modelling/plot_classifier_bettor.py` and `README.md`. `CHANGELOG.md` is generated by `git-changelog`
  from commit messages, so the breaking changes travel in the release commit message.
- Dependencies: exactly one new runtime dependency, `pyarrow>=17.0.0`. `beautifulsoup4` was already declared and is now
  used for index discovery.
- Repository: the `data` branch and the Prefect flow are deleted, so 919 odds CSVs are no longer published.
- Quality checks: the specification quality checklist passed every item. Named products were kept out of the spec body
  and referred to by role. The `BaseDataLoader` contract and the dummy and factory entry points were named there as
  scope boundaries, not implementation choices.
