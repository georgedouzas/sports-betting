# Design

## Context

The NBA is a new league inside a sport the library already supports. It reaches the user through the existing
dataloader:

```python
from sportsbet.datasets import BasketballDataLoader, NBAStats, OddsApi

dataloader = BasketballDataLoader(
    param_grid={'league': ['NBA'], 'year': [2025]},
    stats=NBAStats(),
    odds=OddsApi(key='...'),
)
```

The odds side is complete, because `basketball_nba` maps to `('NBA', 1)` in `OddsApi.LEAGUES_MAPPING` since feature
003. The engine is sport-agnostic since the move in feature 003. The feature is one source file, one export, one test
file and documentation.

Constraints:

- Sources never fetch. `index_items`, `catalogue`, `required_items` and `to_snapshots` are pure functions of their
  inputs. Only the store touches the network (FR-015).
- Extraction never fetches. Only `prepare()` downloads.
- No test touches the network. A network-touching class has no executable doctest.
- No new runtime dependency (SC-006). Ship no data (FR-018, SC-010).
- Every request stays far below the feed's cap of 1,000 events (FR-012), because the feed truncates silently.

Scale: about 1,400 events per season across 11 monthly requests, fetched concurrently by the existing async fetcher. 30
clubs. The feed publishes 81 seasons, and only those the odds vendor also covers are selectable.

## Goals / Non-Goals

**Goals:**

- A user obtains an NBA training set and NBA fixtures with the same three steps and the same objects used for soccer and
  the EuroLeague (SC-001).
- A transformed season contains exactly 30 teams and no exhibition selection (SC-002).
- A transformed season contains every game of its regular season, play-in and post-season, and none of its pre-season
  (SC-003).
- The current season yields results for the games already played (SC-004).
- Every one of the 30 clubs pairs with the odds vendor's name, with as few aliases as the data allows and a reason for
  each (SC-005).
- No existing source, dataloader, fetch layer or engine changes, and every existing test passes unmodified (SC-007).
- The soccer and EuroLeague datasets stay unchanged (SC-009).

**Non-Goals:**

- The WNBA, college basketball and the Australian NBL. Each needs its own statistics source.
- Player and box-score features.
- In-play snapshots. The feed carries period scores, but the first cut is pre-play and post-play.
- A frozen equivalence fingerprint for the NBA. The feed is live, so a fingerprint of a season in progress would drift.
  It is worth freezing for a completed season only.

## Decisions

### D1. ESPN instead of the official NBA feed

The statistics come from ESPN.

The official `data.nba.com/data/10s/v2015/json/mobile_teams/nba/{y}/league/00_full_schedule.json` is free, needs no
key, returns a season in one request of about 4 MB and carries the tip-off in UTC. Its results are back-filled once,
months after the season ends:

| Season | File last modified | Regular-season games with a final score |
| --- | --- | --- |
| 2023-24 | Sep 2024 | 1230 / 1230 |
| 2024-25 | Oct 2025 | 1230 / 1230 |
| 2025-26 | Oct 2025 | 0 / 1200 |

The 2025-26 season ended in June 2026 and its file still had no scores. During a live season this feed gives fixtures
and no results, so every current form feature would be missing and no target would exist. FR-014 exists to forbid a
later move onto this feed.

Alternatives considered: the official archive, which cannot see a live season. `stats.nba.com` through `nba_api`, which
is a new runtime dependency and timed out even with a full browser header set. Every `cdn.nba.com` path returned 403.
Scraping Basketball-Reference, which conflicts with its terms of use and returns HTML.

### D2. ESPN needs nothing new

ESPN answers with no `User-Agent` and no headers, which is the request the existing fetcher makes. An earlier design
around the official feed needed a `headers` field on `RawItem` and a probe concept, a `HEAD` request whose failure is
not an error. Neither is needed. Nothing is added to `RawItem` or `BaseStore`, and `_fetch.py` does not change.

### D3. The catalogue comes from the feed and the year needs no conversion

`index_items()` returns one item:

```text
GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons?limit=100
```

It is free, about 8.7 KB and volatile, because a new season appears each year. It returns 81 seasons as `$ref` URLs
whose last path segment is the season year. The catalogue is parsed from those, as
`{'league': 'NBA', 'division': 1, 'year': <year>}`. No year range is fabricated.

ESPN's `season.year` is already the year the season ends: a game on 2025-10-21 carries `season.year: 2026`. The
EuroLeague needed a conversion and the NBA does not. Adding one would shift every season and empty the intersection with
the odds vendor. A test pins it.

### D4. One request per month

`required_items(params)` returns one item per month, from September of `year - 1` to July of `year`:

```text
GET https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD-YYYYMMDD&limit=1000
```

The feed truncates at 1,000 events without any signal. A six-month range returns exactly 1,000 events, and a season has
about 1,400. Events per month over a full season:

| Oct | Nov | Dec | Jan | Feb | Mar | Apr | May | Jun |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 151 | 219 | 198 | 237 | 170 | 239 | 144 | 38 | 5 |

The peak is 239 against the cap of 1,000, a fourfold margin. September and July are empty and cost nothing, and they
cover a calendar shift. A guard test pins the window. `schedule` is ignored, because this source prices nothing.

Alternatives considered: one request per season, which truncates at 1,000. The core API's
`seasons/{y}/types/{t}/events`, whose items are `$ref` links that cost 1,239 more requests. One request per day, about
170 requests with no benefit over 11.

### D5. The filter excludes pre-season and exhibitions

A game is kept unless it is pre-season or an exhibition: drop `season.type == 1`, drop
`competitions[0].type.abbreviation == 'ALLSTAR'`, keep everything else.

Each event carries two labels. `season.type` is 1 for pre-season, 2 for regular season, 3 for post-season and 5 for the
play-in. `competitions[0].type.abbreviation` is `STD`, `RD16`, `QTR`, `SEMI`, `FINAL` or `ALLSTAR`.

Keeping `season.type == 2` admits the exhibition, because ESPN files the all-star games under
`season.type: 2, slug: 'regular-season'`. Their competitors are `Team Stars`, `Team Stripes` and `World`. These names
would enter the roster and break the pairing with the odds vendor for the whole league.

Keeping `competition.type == 'STD'` drops the playoffs, which are `RD16`, `QTR`, `SEMI` and `FINAL`. This rule was tried
first. It returned 1,238 games and no playoff game.

The chosen rule, verified on two seasons:

| Season | Raw events | Kept | Regular | Play-in | Post | Distinct teams | Exhibition teams |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-24 | 1395 | 1321 | 1233 | 6 | 82 | 30 | none |
| 2025-26 | 1401 | 1326 | 1235 | 6 | 85 | 30 | none |

An unrecognised label is excluded rather than admitted (FR-011). A missing game is a visible gap. An invented team in
the roster corrupts the dataset silently.

### D6. The tip-off is read in UTC

The tip-off comes from the event's `date`, which is ISO-8601 with a `Z`, for example `2026-01-01T23:00Z`. No zone is
inferred. football-data.co.uk publishes UK time and the EuroLeague publishes CET, and neither documents it. ESPN is the
first of the three feeds that states UTC directly.

### D7. The shape is the EuroLeague's

Markets are `home_win` and `away_win`, derived with the public `market_outcomes`. There is no draw and no totals market,
as settled for the EuroLeague.

Features are per-team expanding and rolling(3) means over `points_for`, `points_against` and `wins`, shifted by one, on
a frame with the fixtures already appended. The shift stops a game from seeing its own result. The ordering gives a
fixture the form of the games before it. The feed carries a score line and nothing else, so no other statistic is
invented.

Played is `competitions[0].status.type.completed`, a boolean read per game. The completed 2023-24 season still contains
2 games that were postponed and never played, so a season being over does not mean every game has a score.

Snapshots: every kept game yields a `preplay` snapshot with form. A played game also yields a `postplay` snapshot with
`home_points`, `away_points` and the outcome. `event_time` is `0` for both. Teams are the competitors' display names,
such as `Boston Celtics`, `Los Angeles Lakers` and `Philadelphia 76ers`. The existing statistics schema validates the
result, so no new schema is needed.

### D8. Reconciliation is expected to need no alias, and was not verified

The existing roster-bijection resolver is relied on, with verification against the vendor before any alias.

ESPN writes canonical names, which is what the odds vendor writes, so all 30 clubs are expected to pair with zero
aliases. The EuroLeague needed one alias and the Premier League none. A wrong alias attaches one club's prices to
another club's game silently, so aliases are added only where a club cannot be placed.

Outcome: not verified. The NBA was out of season in July and the vendor's events endpoint returned zero NBA events. The
vendor's historical endpoint is paid-only and answered `401 HISTORICAL_UNAVAILABLE` on the free tier, with no credit
spent. The vendor writes basketball clubs as city plus nickname, checked against the WNBA, as ESPN does for the NBA.
Zero aliases is the expected result. The pairing is to be re-run when the season starts in October, or with a paid key.

### D9. The odds side is already finished

`basketball_nba` has mapped to `('NBA', 1)` since feature 003, and `OddsApi.catalogue` offers every mapped league from
`FIRST_YEAR` to the next year, so the intersection with ESPN's seasons is not empty. NBA odds need the user's own paid
key. A basketball dataloader with no odds source already raises and says there is nothing to predict.

### Cost

`NBAStats` is free and unmetered. Its cost estimate is `0` for every item. Only the odds cost credits.

### Surfaces

No registration point exists, and none was invented. `EuroLeagueStats` is not registered anywhere. The CLI loads a user
configuration module that names a `DATALOADER_CLASS`, so the NBA is reachable through a subclass of the basketball
dataloader with no change. The GUI maps only `{'Soccer': SoccerDataLoader}` and has no basketball, because it cannot
supply a paid odds key. That gap predates this feature.

## Risks / Trade-offs

- The feed truncates a request silently, which would leave a season missing games. Mitigation: monthly windows with a
  fourfold margin and a guard test that pins the window.
- An exhibition enters the dataset, which would break the odds pairing for the whole league. Mitigation: the filter
  excludes an unrecognised label, and two regression tests cover each direction of the trap.
- ESPN is unofficial, and it can change or disappear. Accepted on purpose, because the official alternative fails FR-014.
  The source is one file behind a stable contract, so a replacement feed is a rewrite of one file.
- A later maintainer moves the source onto the official feed, which makes the NBA backtest-only. Mitigation: D1 records
  the evidence, and FR-014 forbids it.
- A later maintainer adds one to the season year by analogy with the EuroLeague. Mitigation: D3 records the reason and a
  test pins the year.
- The NBA roster pairing is unverified, see D8.
- The odds half of the end-to-end check was not bought. A dry run quoted 17,722 credits for an NBA season with zero
  spent, and the free key held 498. The statistics half was verified live: 1326 games, exactly 30 teams, no exhibitions,
  4 postponed games left as fixtures, and UTC tip-offs from 2025-10-21 to 2026-06-14.

## Quality checks

- The gate: `pdm run formatting`, `pdm run checks` and `pdm run tests` pass with no existing test edited.
- `pdm run pytest tests/datasets/test_nba.py -v` covers the exhibition exclusion, the playoff inclusion and the monthly
  window guard, offline.
- `NBAStats().available_params()` returns the seasons named by their ending year, for free.
- A dry run reports `to_fetch` and `estimated_cost` with zero credits spent.
- After extraction, `Y` has `home_win` and `away_win` and no `draw`, `X`, `Y` and `O` have the same number of rows, the
  teams are the 30 clubs, the game count matches the real one, and a season in progress has results for played games.
- The public-API docs audit finds a runnable example for `NBAStats`.
