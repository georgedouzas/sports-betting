# Design

## Context

Feature 002 built an engine that reads injected statistics and odds sources, stores their raw payloads, prepares the
store with an exact cost estimate, and reconciles team names. All of that logic lived inside `SoccerDataLoader`, but
none of it depended on soccer apart from two default class names. This feature adds a second sport by adding a data
source. The engine, the store, the preparation step, the cost estimate and the reconciliation are reused unchanged.

Every finding in the research was obtained by calling the live API. A EuroLeague season holds about 330 games, and the
library covers a handful of seasons.

Constraints carried from feature 002:

- Sources never fetch. Only the store fetches.
- Extraction never fetches. Only `prepare()` fetches.
- `date` is the tip-off instant in UTC. Every source resolves its own time zone at its own boundary.
- No new runtime dependency (SC-010).
- No basketball data is redistributed (SC-007).
- No test touches the network (SC-008). The socket guard in `tests/conftest.py` enforces it.

## Goals / Non-Goals

**Goals:**

- A user goes from a fresh install to an extracted EuroLeague training set with one configuration and one preparation
  call (SC-001).
- The targets contain no draw, and predicted probabilities of a home win and an away win sum to one, while soccer keeps
  its draw (SC-003).
- The real rosters of a EuroLeague season reconcile with no hand-written aliases, or each exception is named and
  justified (SC-004).
- A game whose odds are missing is always reported or raised (SC-005).
- A dry run reports an exact cost with zero credits spent (SC-006).
- The soccer path is unchanged and still reproduces its frozen reference (SC-009).
- A completed season reproduces a frozen reference exactly (SC-002). This goal was deferred, see the tasks.

**Non-Goals:**

- The NBA. Its free statistics live behind an unofficial, rate-limited endpoint of a different shape.
- Player and box-score data. Only team scores are used.
- In-play basketball. The first cut covers pre-play and post-play only.
- The totals market. Its line moves per game, and supporting it changes the data model for soccer too.
- Other basketball competitions that the odds vendor covers but that have no free statistics source.

## Decisions

### D1. Take the tip-off from `utcDate`

The tip-off comes from the API's `utcDate` field. This superseded the first plan, which parsed the published
`date`/`time` as `Europe/Berlin`.

The API publishes its `date` field in head-office time for every venue, and documents no time zone. The published times
of a full season, grouped by the home venue's zone, prove it:

| Home venue | Venue zone | Published time | Implied local tip-off |
| --- | --- | --- | --- |
| Istanbul (Efes, Fenerbahce) | UTC+3 | 18:30 / 18:45 | 20:30 / 20:45 |
| Tel Aviv (Maccabi) | UTC+2 | 20:05 | 21:05 |
| Athens (Panathinaikos, Olympiacos) | UTC+2 | 20:15 | 21:15 |
| Kaunas (Zalgiris) | UTC+2 | 19:00 | 20:00 |
| Madrid, Barcelona, Milan, Munich | UTC+1 | 20:30 / 20:45 | 20:30 / 20:45 |

Read as venue time, an Istanbul game would start at 18:30. Read as UTC, a Barcelona game would start at 22:30 local.
Read as CET, every venue lands on its known slot. The v2 endpoint then confirmed it: `date: 2025-05-25T19:00` against
`utcDate: 2025-05-25T17:00Z`.

Rationale: `date + event_time` is the address that a time-stamped odds source receives. A wrong zone shifts every
purchased price by one to two hours and nothing reports it. football-data.co.uk does the same with UK time, so an
undocumented time is treated as unknown until proven.

Alternatives considered: assume UTC, because the format looks like ISO. Assume venue time, because it is the intuitive
reading. Both are wrong and both fail silently.

### D2. Head-to-head market only

The first cut ships `home_win` and `away_win`. Totals are out of scope.

Rationale: the library expresses a market as a column. Soccer's `over_2.5` works because 2.5 is the line for every game
and every bookmaker. Over a full EuroLeague season the total points of a game run from 125 to 229, with a median of 167,
and a bookmaker sets a different line for each game. A market whose line moves per row needs a pair of columns, a line
and an outcome, which changes the data model and soccer with it.

Consequence for `OddsApi`: it filters totals to a single point (`TOTALS_POINT = 2.5`). For basketball this filter
matches nothing. The research recorded that a request for `totals` returns nothing and that the source must say so
rather than return an empty odds table.

Alternatives considered: a fixed line such as a season median, which fabricates a market no bookmaker offers. A per-row
line, which is real but belongs in its own feature.

### D3. One request per season, with the catalogue read from the API

- `index_items()` requests `GET /v2/competitions/E/seasons`, which is free and lists every season with its code, year
  and dates.
- `required_items(params)` returns one item per season, `GET /v2/competitions/E/seasons/E{year}/games`.

Rationale: a season returns in one response of 330 games. The v1 endpoint behaves the same way. The season list comes
from the API, so no year range is fabricated, which is the rule that removed the `Netherlands_2` bug in the soccer feed.

Season numbering: the API's `E2024` is the 2024-25 season and carries `year: 2024`. The library's `year` is the year the
season ends, so `E2024` becomes `year = 2025`. This must agree with `OddsApi`, or the intersection of the two catalogues
is empty.

Alternatives considered: one request per round, which spends about thirty requests to fetch what one returns.

### D4. The v2 JSON endpoint instead of the v1 XML

This superseded the first plan, which parsed v1 XML with `xml.etree.ElementTree`. The contract first named
`GET /v1/results?seasoncode=E{year}`, whose games look like this:

```xml
<game>
  <round>RS</round><gameday>1</gameday>
  <date>Oct 03, 2024</date><time>18:45</time>
  <gamecode>E2024_1</gamecode>
  <hometeam>ALBA BERLIN</hometeam><homecode>BER</homecode><homescore>77</homescore>
  <awayteam>PANATHINAIKOS AKTOR ATHENS</awayteam><awaycode>PAN</awaycode><awayscore>87</awayscore>
  <played>true</played>
</game>
```

Rationale:

1. Security. `bandit` flags `xml.etree.ElementTree` (B405, B314) as open to entity expansion. The fix, `defusedxml`,
   is a new runtime dependency, which SC-010 forbids. JSON needs nothing.
2. The v2 payload carries `utcDate`, so the time zone needs no inference.
3. Better names. v1 writes `ALBA BERLIN` and v2 writes `Alba Berlin`, which is closer to the odds vendor's spelling.

The v2 payload also carries `partials`, the score of each quarter, which is the input a later in-play dataset needs.

### D5. Points for and against, and nothing invented

The form features have the shape of the soccer ones: per-team expanding mean and rolling mean over the last three games,
shifted by one, computed on a frame that already has the upcoming fixtures appended. They cover points scored and
points conceded only.

Columns: `home_points_for_avg`, `home_points_against_avg`, `home_points_for_latest_avg`,
`home_points_against_latest_avg`, and the same four for `away_`.

Rationale: the results feed carries a score line and nothing else. Soccer builds an adjusted goals figure from shots,
corners and cards, which this feed lacks. The shift by one stops a game from seeing its own result. Appending the
fixtures first gives an upcoming game the form of the games before it.

### D6. Move the engine onto a shared base

The sport-agnostic code moves out of `SoccerDataLoader` onto `SourcedDataLoader` in
`src/sportsbet/datasets/_base/_sourced.py`, with class attributes `DEFAULT_STATS` and `DEFAULT_ODDS`. Each sport keeps
only its default sources. `market_outcomes` moves to `src/sportsbet/datasets/_utils.py` and keeps its public path.

Rationale: about 200 lines of `SoccerDataLoader` have nothing to do with soccer. A copy in `BasketballDataLoader` would
duplicate the whole preparation and reconciliation contract, and the two copies would drift. If a sport needs to change
the engine, the abstraction is wrong. The existing suite and the soccer equivalence gate protect the move.

Alternatives considered: copy the loader, which gives two engines to fix. Subclass `SoccerDataLoader`, which makes a
basketball loader a soccer loader.

The resulting constructor:

```python
BasketballDataLoader(
    param_grid=None,
    stats=EuroLeagueStats(),
    odds=None,
    store=LocalStore(),
    aliases=None,
    max_unmatched_rate=0.0,
)
```

`odds` defaults to `None` because no free basketball odds source exists. The user brings their own.

### D7. Say plainly that there is nothing to predict

When the odds table carries no markets, extraction raises a clear error saying that there is nothing to predict.

Rationale: `targets_` comes from the odds table's value columns, so a dataloader with no odds source has no `Y`. That is
coherent for a betting library, and it is why basketball requires an odds source. Before this change the failure was
`expected series 'event_status' to have type string[pyarrow], got object`, which told the user nothing.

### D8. Verify reconciliation against real names before writing aliases

The roster pairing is checked against the real names of both sources before any alias is written.

Rationale: basketball names carry sponsors, such as `PANATHINAIKOS AKTOR ATHENS`, `CRVENA ZVEZDA MERIDIANBET BELGRADE`,
`MACCABI PLAYTIKA TEL AVIV` and `EA7 EMPORIO ARMANI MILAN`. Basketball always mixes two sources, so reconciliation is
the normal path. A wrong alias attaches one club's odds to another club's game silently, so an alias is added only where
the pairing cannot place a club, with a reason.

Outcome: 17 of the 18 clubs paired with no alias, for example `Maccabi Tel Aviv` to `Maccabi Playtika Tel Aviv`,
`Partizan` to `Partizan Mozzart Bet Belgrade` and `ASVEL` to `LDLC ASVEL Villeurbanne`. One alias was added,
`Olimpia Milano` to `EA7 Emporio Armani Milan`. The vendor uses the historic club name and the feed uses the sponsor
name, and the two share only the city. The resolver refused the pair and suggested the alias.

### D9. Price only the moments the statistics carry

`OddsApi` derives the moments it prices from the schedule that the dataloader hands it, and that schedule now carries
the moments the statistics have.

Rationale: the default moments were soccer's, `[('preplay', 0), ('inplay', 45)]`. Minute 45 has no basketball snapshot
to pair with, so half of a EuroLeague season's cost bought unusable data.

| Case | Snapshots | Credits |
| --- | --- | --- |
| Basketball, soccer's default moments | 547 | 5,461 |
| Basketball, moments from its statistics | 274 | 2,731 |
| Soccer, moments from its statistics | 433 | 4,321 |

Soccer still buys the in-play price because it has half-time snapshots. Neither sport is configured.

### D10. An odds price is never fixed

Odds value columns are always time-varying and always carry the provider in their name.

Rationale: `fixed` means constant within a match and is computed by grouping on the identity columns, which exclude
`provider`. With one provider every odds column looked constant, was marked fixed, lost its provider prefix, and the
bettor then rejected the odds. Soccer never hit this because football-data ships two providers. The change is a no-op
for soccer, verified against its frozen fingerprint.

### Package layout

`_basketball/` mirrors `_soccer/` so that a third sport has an obvious place. The plan named `_base/_dataloader.py` as
the target of the move. The tasks placed it in the new `_base/_sourced.py`.

## Risks / Trade-offs

- The move touches working soccer code. Mitigation: zero test edits allowed, and the soccer equivalence gate runs at
  every phase boundary.
- Basketball ships without a totals market. A fixed line would fabricate a market, so the gap is accepted.
- The user brings their own odds. No free basketball odds source exists, and the sources are constructor arguments.
- A EuroLeague fingerprint is not frozen. The API is volatile and a season is re-read on every prepare, so a
  fingerprint would drift. It is worth freezing for a completed season only.
- Task 3.3 recorded basketball in the CLI and GUI dataloader mapping. Feature 004 later found that the GUI maps only
  `{'Soccer': SoccerDataLoader}` because it cannot supply a paid odds key, so the GUI part did not hold.
- The phases ran in order. The move onto the shared base blocked everything else, and the soccer equivalence gate ran
  at every phase boundary.

## Quality checks

Validation steps recorded for this feature, each tied to the spec:

- The move: `pdm run tests` and `pdm run pytest tests/datasets/test_equivalence.py -m network --no-cov` pass with no
  test edited.
- A EuroLeague dataset: `X`, `Y` and `O` share rows and index, and a season costs one statistics request.
- No draw: `pdm tests -k complementary` passes and `Y` has no `draw` column.
- Reconciliation: `dataloader.reconciliation_` reports matched and unmatched counts, and an unplaceable club raises
  `UnmatchedError` naming the club.
- Dry run: `prepare(dry_run=True)` reports `estimated_cost` with zero credits spent.
- No odds: extraction fails with a message about missing markets.
- Tip-off: a game published at `18:30` in Istanbul is `17:30 UTC`.
- Nothing published: `git ls-tree` shows no EuroLeague data and `pyproject.toml` gains no runtime dependency.
