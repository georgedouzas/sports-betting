# Tasks

## 1. Move the engine onto a shared base

- [x] 1.1 Create `src/sportsbet/datasets/_base/_sourced.py` with `SourcedDataLoader(BaseDataLoader)`, holding the
  sport-agnostic code moved out of `SoccerDataLoader`: `__init__`, `_resolved`, `sources`, `_authorize`, `_unique`,
  `_catalogue`, `_params`, `_all_params`, `_schedule`, `_items`, `_report`, `_unavailable`, `prepare`, `_snapshots`,
  `_derive` (T001)
- [x] 1.2 Give `SourcedDataLoader` the class attributes `DEFAULT_STATS` and `DEFAULT_ODDS`, so a sport is its defaults
  and nothing else (T002)
- [x] 1.3 Reduce `src/sportsbet/datasets/_soccer/_dataloader.py` to a class name and two defaults (`FootballDataStats`,
  `FootballDataOdds`) (T003)
- [x] 1.4 Move `market_outcomes` from `_soccer/_utils.py` to a sport-neutral `src/sportsbet/datasets/_utils.py`, and
  keep `sportsbet.datasets.market_outcomes` working because it now serves two sports (T004)
- [x] 1.5 Run the full suite and the soccer equivalence gate with zero test edits allowed (T005)

## 2. EuroLeagueStats

- [x] 2.1 Create `src/sportsbet/datasets/_sources/_euroleague.py`. `index_items()` requests the free
  `/v2/competitions/E/seasons`. `catalogue(payloads)` returns the seasons the API publishes, with `year` as the year the
  season ends (`E2024` becomes 2025), never a fabricated range (T006)
- [x] 2.2 `required_items(params)` returns one item per season (`/v2/competitions/E/seasons/E{year}/games`), because the
  endpoint returns the whole season in one response, verified at 330 games (T007)
- [x] 2.3 Use JSON instead of XML. `bandit` flags `xml.etree.ElementTree` as open to entity expansion, and the fix
  (`defusedxml`) is a new dependency. The v2 endpoint returns JSON, which needs nothing and gives better team names.
  See design D4 (T008)
- [x] 2.4 Take `utcDate`. The API publishes `date` in its head-office zone for every country, so an Istanbul game reads
  `18:30` and tips off at 20:30 local. The v2 endpoint gives UTC directly. See design D1 (T009)
- [x] 2.5 Build the form features: per-team expanding and rolling(3) means over points for and against, shifted by one,
  on a frame with the fixtures already appended. Port the shape of the soccer version and invent no basketball
  statistics from a score line (T010)
- [x] 2.6 Split a game into a pre-play snapshot (form) and a post-play snapshot (score and outcome), using the played
  flag so an unplayed game becomes a fixture and never a training row (T011)
- [x] 2.7 Derive the outcome with `market_outcomes`: `home_win`, `away_win`. No draw because a tie goes to overtime, and
  no totals because the line moves per game (T012)
- [x] 2.8 Write `tests/datasets/test_euroleague.py` against a payload shaped like the real response: the UTC tip-off,
  the played and unplayed split, the two-way outcome, and form that never sees its own game (T013)

## 3. BasketballDataLoader and the odds

- [x] 3.1 Create `src/sportsbet/datasets/_basketball/_dataloader.py` with `BasketballDataLoader(SourcedDataLoader)`,
  `DEFAULT_STATS = EuroLeagueStats` and `DEFAULT_ODDS = None`. If it needs more than that, fix the base (T014)
- [x] 3.2 Extend `OddsApi.LEAGUES_MAPPING` with `basketball_euroleague` mapped to `('Euroleague', 1)` and the other
  `basketball_*` keys (T015)
- [x] 3.3 Export `BasketballDataLoader` and `EuroLeagueStats` from `sportsbet.datasets`, and add basketball to the CLI
  and GUI dataloader mapping so all three surfaces gain the sport at once (T016)
- [x] 3.4 Write `tests/datasets/test_basketball.py`: `Y` carries a home win and an away win and no draw, and a bettor's
  probabilities for the two sum to one, while soccer keeps its draw (T017)
- [x] 3.5 Verify end to end against the live API that a EuroLeague season prepares and extracts with `X`, `Y` and `O`
  aligned (T018)

## 4. Reconcile the two rosters

- [x] 4.1 Verify that 17 of the 18 clubs pair with no aliases, sponsor names included: `Maccabi Tel Aviv` to
  `Maccabi Playtika Tel Aviv`, `Partizan` to `Partizan Mozzart Bet Belgrade`, `ASVEL` to `LDLC ASVEL Villeurbanne`
  (T019)
- [x] 4.2 Add one alias, `Olimpia Milano` to `EA7 Emporio Armani Milan`. The vendor uses the historic club name and the
  feed uses the sponsor name, and they share only the city. The resolver refused the pair and suggested it (T020)
- [x] 4.3 Test that an unplaceable club raises and names itself, rather than silently dropping its game (T021)

## 5. The clear failure, the gate and the docs

- [x] 5.1 Reproduce the empty-odds failure and make it a clear error: a dataloader with no odds source has no markets
  and nothing to predict, and it says so instead of
  `expected series 'event_status' to have type string[pyarrow], got object` (T022)
- [x] 5.2 Test both shapes of it: no odds source, and an odds source carrying no markets (T023)
- [x] 5.3 Moved to change `verify-basketball-sources`, task 1. Freeze a EuroLeague fingerprint beside the soccer one.
  Deferred: the EuroLeague API is volatile and a season is re-read on every prepare, so a fingerprint would drift as the
  season progresses. It is worth doing for a completed season only (T024)
- [x] 5.4 Update `docs/overview/user_guide/dataloader.md`, which said the library is soccer-only. Add the sport, the
  source, the two-way market and the statement that basketball needs a paid odds key (T025)
- [x] 5.5 Update the README and `docs/overview/user_guide/index.md` (T026)
- [x] 5.6 Confirm no new runtime dependency (SC-010) and that no basketball data is published (SC-007) (T027)
