# Contract: `NBAStats`

`NBAStats` is a `BaseStatsSource`. It implements the four methods every source implements, and **it never fetches** —
each one is a pure function of its inputs. The store does the downloading, which is what makes `prepare(dry_run=True)`
free and makes extraction structurally unable to reach the network.

```python
from sportsbet.datasets import BasketballDataLoader, NBAStats, OddsApi

dataloader = BasketballDataLoader(
    param_grid={'league': ['NBA'], 'year': [2025]},
    stats=NBAStats(),
    odds=OddsApi(key='...'),
)
```

`NBAStats` takes no arguments. It is free and needs no credential. The **odds** do — there is no free basketball odds
feed anywhere.

## `index_items() -> list[RawItem]`

One item: the league's seasons index.

```text
GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons?limit=100
```

Free, ~8.7 KB. Volatile, because a new season appears in it every year.

## `catalogue(payloads) -> list[dict]`

Reads the seasons the feed publishes out of that index, as `{'league': 'NBA', 'division': 1, 'year': <year>}`.

**The year is the feed's year, unchanged.** ESPN names a season by the year it ends in, and so does this library
(research D3). Adding one would be a bug.

The catalogue is never a fabricated range of years. What the feed lists is what is offered — and the engine then
intersects it with the odds vendor's coverage, so only seasons that *both* publish become selectable.

## `required_items(params, schedule=None) -> list[RawItem]`

**One item per month** of each selected season, September of `year - 1` through July of `year`:

```text
GET https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD-YYYYMMDD&limit=1000
```

**This is a correctness requirement, not a performance choice.** The feed silently truncates at 1,000 events: ask for a
whole season and it returns exactly 1,000 of ~1,400 and says nothing. A month peaks at 239 — a 4× margin. Widening the
window to save requests would delete a quarter of a season without an error anywhere (research D4, FR-012).

The empty months (September, July) cost nothing and are insurance against a calendar shift.

`schedule` is ignored: this source does not price anything, so it has no need of one.

## `to_snapshots(payloads) -> pd.DataFrame`

Turns the fetched months into the long statistics snapshots.

**The filter — both halves matter, and each is a trap on its own** (research D5):

| Rule | Why |
| --- | --- |
| drop `season.type == 1` | the pre-season, which nobody bets seriously |
| drop `competition.type.abbreviation == 'ALLSTAR'` | the exhibition, whose teams (`Team Stars`, `Team Stripes`, `World`) are **not clubs** |
| keep everything else | the regular season, the play-in **and the post-season** |

Filtering on `season.type` alone **admits the exhibition**, because ESPN files it under `regular-season`. Filtering on
`competition.type == 'STD'` alone **drops all 85 playoff games**, because they are `RD16` / `QTR` / `SEMI` / `FINAL`.

The rule is written as an **exclusion**, so an unrecognised label is dropped rather than admitted (FR-011). A missing
game is a visible bug; an invented team in the roster is silent corruption that breaks the odds pairing for the whole
league.

**Per game**:

- `date` ← the event's `date`, which is ISO-8601 in UTC. Read, never inferred.
- teams ← the competitors' canonical display names, home and away.
- played ← `competitions[0].status.type.completed`, **per game**. A finished season still contains games that were never
  played (research D7), so this is never inferred from the season being over.
- a **pre-play** snapshot always; a **post-play** snapshot only if played.
- outcomes ← `market_outcomes(home_points, away_points, ['home_win', 'away_win'])`. No draw, no totals.
- features ← per-team expanding and rolling(3) means over `points_for`, `points_against` and `wins`, shifted by one, on a
  frame with the fixtures already appended.

## Cost

Zero. `estimate()` returns `0` for every item: the feed is free and unmetered. Only the odds cost credits.
