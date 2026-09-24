# Research: The NBA, as a second basketball league

Every finding below was obtained by hitting the live APIs, not by reading about them. Two of them inverted the obvious
design, and both would have failed silently.

## D1. The league's own feed is an archive, not a feed — so it loses

**Decision**: the statistics come from **ESPN**, not from the NBA.

That is an uncomfortable sentence, so here is what forced it.

`data.nba.com/data/10s/v2015/json/mobile_teams/nba/{y}/league/00_full_schedule.json` is **official**, **free**, needs no
key, returns a whole season in **one ~4 MB request**, and even carries the tip-off in UTC. On every axis this project
normally cares about, it wins. It is also a trap.

It is not updated as a season is played. Each season's file is written when the schedule is published, and its results
are back-filled **once**, months **after** the season has ended:

| Season | File last modified | Regular-season games with a final score |
| --- | --- | --- |
| 2023-24 | Sep 2024 | 1230 / 1230 |
| 2024-25 | Oct 2025 | 1230 / 1230 |
| **2025-26** | **Oct 2025** | **0 / 1200** |

The 2025-26 season **ended in June 2026**. Its file still has no scores.

So during any live season the official feed hands back fixtures and no results. Every current-season form feature would
be `NaN`, and every target would be absent. A user could backtest the NBA on it and could never bet on it — which, in a
betting library, is the whole point missed. That is **FR-014**, and it is written as a requirement precisely so that a
future maintainer looking at a 4 MB official endpoint and an unofficial one does not "simplify" back onto this and
quietly destroy the current season.

**Also probed and dead**: `stats.nba.com` times out even with a complete browser header set (User-Agent, Referer,
Origin, `x-nba-stats-*`). Every `cdn.nba.com` path returns 403. There is no third official option.

**Alternatives rejected**: the official archive (above — cannot see a live season); `stats.nba.com` via `nba_api` (a new
runtime dependency, forbidden by SC-006, and unreachable anyway); scraping Basketball-Reference (terms of use, and HTML).

## D2. ESPN needs nothing new — which is what makes this feature one file

**Decision**: build on ESPN's public JSON endpoints, using the store's **existing** fetcher.

**Verified**: ESPN answers with **no `User-Agent` and no headers at all** — the exact request `_fetch.py` makes today.

This matters more than it sounds, because the *previous* design (built around the official NBA feed, which 403s a
non-browser agent) required two new pieces of machinery:

- a `headers` field on `RawItem`, threaded through the store and the fetch layer, and
- a "probe" concept — a `HEAD` request whose absence is not an error — because the official feed has no season index and
  the only way to discover a season was to ask whether its file exists.

**Both evaporated.** Recorded here so that nobody reintroduces them: with ESPN there is nothing to add to `RawItem`,
nothing to add to `BaseStore`, and nothing to change in `_fetch.py`.

The consequence is SC-007. This feature is a **new file**, not a new seam. That is the first independent test of the
engine hoist from feature 003, and it passes.

## D3. The catalogue comes from the feed, and the year needs no conversion

**Decision**:

- `index_items()` → `GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons?limit=100` (~8.7 KB,
  free). It returns 81 seasons as `$ref` URLs whose last path segment is the season year.
- The catalogue is parsed from those. **No year range is ever fabricated** — the rule that killed the `Netherlands_2`
  bug in the soccer feed.

**Season numbering — do not "fix" this.** ESPN's `season.year` is **already** the year the season *ends*: a game played
on 2025-10-21 carries `season.year: 2026`. The library's convention is identical. So, unlike the EuroLeague — whose
`E2024` is the 2024-25 season and needed `+1` — **there is no conversion here**.

This is written down because it is exactly the kind of thing someone later notices, assumes is a bug by analogy with the
EuroLeague, and "corrects". Adding one would shift every NBA season and empty the catalogue intersection with the odds
vendor, whose coverage is a different year range. A test pins it.

## D4. One request per month, because the feed truncates silently

**Decision**: `required_items(params)` → one item per **month** of the season, September of year−1 through July of year:

```text
GET https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD-YYYYMMDD&limit=1000
```

**The 1,000-event cap is a hard, silent truncation.** A six-month range returns **exactly 1,000** events and says
nothing whatsoever about what it dropped. A whole season is ~1,400. There is no `next` link, no total count, no warning
— just a short answer that looks like a complete one.

Monthly windows are **provably** safe, measured across a full season:

| Oct | Nov | Dec | Jan | Feb | Mar | Apr | May | Jun |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 151 | 219 | 198 | 237 | 170 | 239 | 144 | 38 | 5 |

Peak **239**, against a cap of 1,000 — a **4× margin**. September and July are empty and cost nothing; they are
insurance against a calendar shift.

**Never widen the window to save requests.** Eleven cheap requests are the price of knowing the season is complete. A
guard test pins the window, so that an "optimisation" fails loudly instead of quietly deleting a quarter of a season
(FR-012, SC-003).

**Alternatives rejected**: one request per season (truncated at 1,000 — this is the bug); the core API's
`seasons/{y}/types/{t}/events` (111 KB, but the items are `$ref`s, so it costs 1,239 further requests); one request per
day (~170 requests for no benefit over 11).

## D5. The all-star trap: filtering on season type is wrong in *both* directions

**Decision**: keep a game unless it is **pre-season** or an **exhibition**. Concretely: drop `season.type == 1`, and
drop `competition.type.abbreviation == 'ALLSTAR'`. Keep everything else.

Each event carries **two** independent labels:

- `season.type` — 1 = preseason, 2 = regular-season, 3 = post-season, 5 = play-in
- `competitions[0].type.abbreviation` — `STD`, `RD16`, `QTR`, `SEMI`, `FINAL`, `ALLSTAR`

Neither one alone is sufficient, and each of the two obvious rules is wrong in an opposite and silent way:

**"Keep `season.type == 2`" admits the exhibition.** ESPN files the all-star games under
`season.type: 2, slug: 'regular-season'`. Their competitors are **`Team Stars`**, **`Team Stripes`** and **`World`** —
invented teams. They would enter the roster, and a polluted roster breaks the bijection with the odds vendor for the
*entire* league, not just for those games. And a model would train on a game nobody was trying to win.

**"Keep `competition.type == 'STD'`" drops the playoffs.** The post-season rounds are `RD16`, `QTR`, `SEMI` and `FINAL`
— *not* `STD`. This rule was tried first. It returned 1,238 games and **zero** playoff games, and looked entirely
plausible.

The correct rule, verified on two seasons:

| Season | Raw events | Kept | Regular | Play-in | Post | Distinct teams | Exhibition teams |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-24 | 1395 | **1321** | 1233 | 6 | 82 | **30** | none |
| 2025-26 | 1401 | **1326** | 1235 | 6 | 85 | **30** | none |

**The filter must exclude an unrecognised label, not admit one** (FR-011). If ESPN invents a new exhibition tomorrow,
the failure mode must be a *missing* game — visible, and a bug report — rather than an invented team in the roster,
which is silent corruption that spreads.

## D6. The kick-off is given in UTC, and that is remarkable

**Decision**: read the tip-off from the event's `date`, which is ISO-8601 with a `Z` (e.g. `2026-01-01T23:00Z`).

The library's permanent convention — `date` is the kick-off **instant in UTC**, and `date + event_time` is the
wall-clock instant of a snapshot, which is the address a time-stamped odds vendor is asked for — is satisfied by reading
the field directly. **No zone is inferred.**

Worth recording *why* this is worth a decision at all: this is the **third** feed, and the **first** that needed no
detective work. football-data.co.uk publishes every league in **UK** time, whatever country it is played in. The
EuroLeague publishes everything in **CET**, whatever country it is played in. **Neither documents this**, and both were
caught only by checking published times against known kick-off slots.

The rule stands — *treat an undocumented time as unknown until proven* — and here the feed simply proved it for us.

## D7. The shape is the EuroLeague's

**Decision**: port the *shape* of `EuroLeagueStats`, and invent nothing.

**Markets**: `home_win` and `away_win`, derived with the existing public `market_outcomes`. **No draw** — a tie goes to
overtime. **No totals** — the line moves from game to game, and a market whose line moves is not a column. Both were
settled for the EuroLeague and neither is re-litigated here.

**Features**: per-team **expanding** and **rolling(3)** means over `points_for`, `points_against` and `wins`, **shifted
by one**, computed on a frame that already has the fixtures appended. The shift-by-one and the
append-fixtures-before-averaging ordering are not stylistic: they are why a game never sees its own result, and why an
upcoming game gets the form of the games before it.

**Played** is `competitions[0].status.type.completed` — a boolean, not a string to match against. An unplayed game gets
a **pre-play snapshot only**: it is a fixture, never a training row with a result nobody knows.

**A verified edge case that contradicts the obvious assumption**: the **completed** 2023-24 season still contains **2
games that were never played** — postponed, never made up. So *"the season is over, therefore every game has a score"*
is **false**, and code that assumes it would emit training rows with no result. The played/unplayed split must be driven
by the flag, per game, and never by the season.

## D8. Reconciliation should need no aliases — verify, do not assume

**Decision**: rely on the existing roster-bijection resolver, and **verify it live** against the vendor before writing a
single alias.

ESPN writes the canonical names — `Boston Celtics`, `Los Angeles Lakers`, `Philadelphia 76ers` — which is what The Odds
API writes. All 30 should pair with **zero** aliases, which would make the NBA easier than the EuroLeague (one alias:
`Olimpia Milano` → `EA7 Emporio Armani Milan`) and as clean as the Premier League (zero).

But basketball **always** mixes two independent sources — there is no single feed carrying both games and odds — so
reconciliation is the normal path here, not an edge case, and an unplaceable club means a game with no price, which
means a backtest that is confidently wrong.

Aliases get added **only** where the pairing genuinely cannot place a club, and each one gets a reason. A wrong alias
attaches one club's prices to another club's game and says nothing about it — worse than not matching at all.

### Outcome: **not verified**, and not claimed

The verification could not be done, and the honest thing is to say so rather than to assume the expected answer.

- The NBA is **out of season** (July). The vendor's `/v4/sports/basketball_nba/events` returns **zero events**, so
  there is no roster to pair against. It cost nothing to find out.
- The vendor's **historical** endpoint, which would have the rosters of a played season, is **paid-only**: it answers
  `401 HISTORICAL_UNAVAILABLE` on the free tier, spending no credits.

So the NBA roster pairing is **unverified**, and no alias has been added — adding one blind would be exactly the wrong
move.

What *was* established, live and free: the vendor writes basketball clubs as **`City Nickname`** — `Atlanta Dream`,
`Los Angeles Sparks`, `New York Liberty` (checked against the WNBA, which is in season). ESPN writes the NBA the same
way — `Atlanta Hawks`, `Los Angeles Lakers`, `New York Knicks`. So **zero aliases is the expected result**.

**Expected is not verified.** This project has twice been wrong about exactly this kind of "obviously fine" assumption
(the two time zones), so it stays open: re-run the pairing when the season starts in October, or against a paid key,
before believing it.

## D9. The odds side is already finished

**Decision**: build nothing. `basketball_nba → ('NBA', 1)` has been in `OddsApi.LEAGUES_MAPPING` since feature 003, and
`OddsApi.catalogue` offers every mapped league for `FIRST_YEAR..now+1`, so the intersection with ESPN's NBA seasons is
non-empty.

Stated explicitly so that nobody builds it twice.

As with the EuroLeague, NBA odds require the user's **own paid key**: there is no free basketball odds feed anywhere,
from anyone. That is not a gap to be closed but a fact to be reported, and the dataloader already reports it — a
basketball dataloader with no odds source raises and says there is nothing to bet on.
