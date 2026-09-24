# Research: Basketball, starting with the EuroLeague

Every finding below was obtained by hitting the live API, not by reading about it. Two of them changed the scope.

## D1. The API publishes every game in **CET** — but it also gives UTC, so take that

**Decision**: take the tip-off from the API's `utcDate`. **Superseded the original plan** to parse the published
`date`/`time` as `Europe/Berlin`, which the v2 endpoint makes unnecessary.

**The CET finding still stands, and it is why this decision is safe.** The API publishes a `date` field in its own
head-office time whatever country the game is played in, and a naive implementation would have read it. The evidence
below is what proved that, and the v2 endpoint then confirmed it outright: `date: 2025-05-25T19:00` against
`utcDate: 2025-05-25T17:00Z` — a two-hour offset, i.e. CEST.

**Reading `date` as UTC, or as the venue's local time, would have moved every game by one to two hours and nothing
would have said so.** That is the trap; `utcDate` is the escape from it.

**This was determined from the data, not assumed.** The API documents no time zone at all. The published times of a full
season, grouped by the home venue's own zone, are decisive:

| Home venue | Venue's zone | Published time | Implied local tip-off |
| --- | --- | --- | --- |
| Istanbul (Efes, Fenerbahçe) | UTC+3 | **18:30 / 18:45** | 20:30 / 20:45 |
| Tel Aviv (Maccabi) | UTC+2 | **20:05** | 21:05 |
| Athens (Panathinaikos, Olympiacos) | UTC+2 | **20:15** | 21:15 |
| Kaunas (Žalgiris) | UTC+2 | **19:00** | 20:00 |
| Madrid, Barcelona, Milan, Munich | UTC+1 | **20:30 / 20:45** | 20:30 / 20:45 |

Read as **venue-local**, an Istanbul game would tip off at 18:30 — absurd. Read as **UTC**, a Barcelona game would tip off
at 22:30 local — also absurd. Read as **CET**, every single venue lands on its real, well-known slot: Maccabi's 21:05,
the Greek 21:15, Žalgiris' 20:00, Istanbul's 20:30. The competition publishes its own head-office time for everything.

**Why this matters more than it looks**: `date + event_time` is the address of a moment of a game, and it is what a
time-stamped odds source is asked for. A wrong zone here shifts every price the user buys by one to two hours, silently.
This is the **second** time a feed has done this: football-data.co.uk publishes every league in **UK** time, which its own
notes do not say either. Treat an undocumented time as unknown until proven, every time.

**Alternatives rejected**: assuming UTC (the format looks ISO-ish); assuming venue-local (the intuitive reading). Both are
wrong, and both fail silently.

## D2. Basketball has no totals market, because it has no line

**Decision**: the first cut ships the **head-to-head market only** — `home_win` and `away_win`. Totals are out of scope.

**Rationale**: the library expresses a market as a **column**. Soccer's `over_2.5`/`under_2.5` works because 2.5 *is* the
line — one number, every game, every bookmaker. Basketball has no such number. Measured over a full EuroLeague season, the
total points of a game run from **125 to 229**, median **167**, and a bookmaker sets a *different* line for every game.

There is no column to write. `over_167.5` would be true of one game and meaningless for the next.

Supporting a market whose line **moves per row** is a change to the data model — the market would become a pair of
columns (a line and an outcome) rather than one — and it would change soccer too. That is its own feature, not something
to smuggle in behind a new sport.

**Consequence for `OddsApi`**: it already filters totals to a single hardcoded point (`TOTALS_POINT = 2.5`). For
basketball this filter simply matches nothing, so a basketball user asking for `markets=['h2h']` gets exactly what the
library can represent. Ask for `totals` and you get nothing — which the source must say, rather than silently returning an
empty odds table.

## D3. One request per season, and the catalogue comes from the API

**Decision**:

- `index_items()` → `GET /v2/competitions/E/seasons` (free), which lists every season the competition publishes, with its
  code (`E2024`), its year and its dates.
- `required_items(params)` → one item per season: `GET /v2/competitions/E/seasons/E{year}/games`.

**Rationale**: verified live. A season comes back in a **single** response — 330 games — so it costs **one** request,
not one per round. (The v1 endpoint behaves the same way: `results?seasoncode=E2024` with no `gamenumber` returns all 330;
with `gamenumber=N` it returns a single round of 9.) The season list comes from the API, so no year range is ever
fabricated — the same rule that killed the `Netherlands_2` bug in the soccer feed.

**Season numbering**: the API's `E2024` is the **2024-25** season and carries `year: 2024`. The library's convention (from
soccer) is that `year` is the year the season **ends**, so `E2024` becomes `year = 2025`. This must agree with what
`OddsApi` publishes for the same competition, or the intersection of the two catalogues will be empty and nothing will be
selectable.

**Alternatives rejected**: one request per round. Thirty-odd requests to fetch what one returns.

## D4. The v2 JSON endpoint, not the v1 XML

**Decision**: build on `GET /v2/competitions/E/seasons/E{year}/games`, which returns JSON.

**Superseded the original plan** to parse the v1 XML with `xml.etree.ElementTree`. Three things forced it, and all three
are improvements:

1. **Security.** `bandit` flags `xml.etree.ElementTree` (B405/B314): it is vulnerable to entity-expansion attacks, and we
   would be parsing a remote response. The fix would be `defusedxml` — **a new runtime dependency**, which SC-010
   forbids. JSON has no such problem and needs nothing.
2. **The time zone stops being a guess.** The v2 payload carries `utcDate` outright (D1). The v1 XML carries only the
   head-office time, which has to be inferred.
3. **Better names.** v1 shouts `ALBA BERLIN`; v2 says `Alba Berlin`. The reconciler compares names against an odds
   vendor's, and the vendor writes them like a human.

It also carries **`partials`** — the score of each quarter — which is what an in-play basketball dataset would be built
from. Out of scope now, but it means the road is open.

## D5. Points for and against, and nothing invented

**Decision**: port the *shape* of the soccer form features — per-team expanding mean and rolling mean over the last three
games, shifted by one, computed on a frame that already has the upcoming fixtures appended — but over **points scored and
conceded**, and nothing else.

**Rationale**: the soccer feed carries shots, corners and cards, so it can build an "adjusted goals" figure. The EuroLeague
results feed carries a **score line and nothing else**. Inventing basketball advanced statistics from two numbers would be
making things up. Points for, points against, and the win/loss that follows from them, are what the data actually supports.

The shift-by-one and the append-fixtures-first ordering are **not** stylistic. They are why a game never sees its own
result, and why an upcoming game gets the form of the games before it. Both were load-bearing in the soccer port and both
carry over unchanged.

## D6. The dataloader is the soccer one with different defaults — so hoist it

**Decision**: move everything sport-agnostic out of `SoccerDataLoader` and onto the base, leaving each sport with only its
default sources.

**Rationale**: `SoccerDataLoader` is ~200 lines, and essentially all of it — the store plumbing, `_catalogue`, `_params`,
`_items`, `_report`, `prepare`, `_snapshots`, `_derive`, `_authorize`, the resolver call, the intersection rule — has
nothing to do with soccer. Copying it into `BasketballDataLoader` would duplicate the *entire* preparation and
reconciliation contract, and the two copies would drift.

The test of the abstraction is this: **if a sport needs to change anything in the engine, the abstraction is wrong.** So
the plan hoists first and adds the sport second, and a `BasketballDataLoader` that is more than a class name and two
defaults is a failed design.

## D7. A market you cannot price is not a market — but say so

**Decision**: when the odds table carries no markets, raise a clear error saying there is nothing to predict.

**Rationale**: `targets_` is derived from the odds table's value columns, so a dataloader with no odds source has no
markets and therefore no `Y`. That is coherent for a **betting** library, and it is why basketball requires an odds source
rather than degrading to a statistics-only dataset.

But today it does not say that. It dies inside schema validation with
`expected series 'event_status' to have type string[pyarrow], got object` — the empty-frame dtype problem again. The user
is told about a dtype when the truth is that they forgot the odds.

## D8. Reconciliation is the one thing that cannot be assumed to work

**Decision**: verify the roster pairing against the **real** names of both sources before writing any aliases.

**Rationale**: the roster-bijection resolver pairs 20 English football clubs with zero aliases, but basketball names are a
different shape: `PANATHINAIKOS AKTOR ATHENS`, `CRVENA ZVEZDA MERIDIANBET BELGRADE`, `MACCABI PLAYTIKA TEL AVIV`,
`EA7 EMPORIO ARMANI MILAN` — sponsor names welded onto club names. The odds vendor writes them differently.

Unlike soccer, basketball **always** mixes two sources, so this is not an edge case, it is the normal path. And an
unplaceable club means a game with no odds, which means a backtest that is confidently wrong.

Aliases get added **only** where the pairing genuinely cannot place a club, and each one gets a reason. A wrong alias
attaches one club's odds to another club's game and says nothing about it — worse than not matching at all.

## D9. Never buy a price for a moment there is nothing to pair with

**Decision**: `OddsApi` derives the moments it prices from the **statistics**, not from a default. The schedule the
dataloader hands it now carries the moments the statistics actually have.

**Rationale**: found by measuring, not by reading. `OddsApi`'s default moments were `[('preplay', 0), ('inplay', 45)]` —
**soccer's** moments. Minute 45 means nothing in basketball, and `EuroLeagueStats` produces no in-play snapshot to pair
it with. So a EuroLeague season cost **5,461 credits**, of which **half bought data that could never be used**.

A price is bought to be paired with what was known at that moment. A moment the statistics do not have is a price with
nothing to pair it with — and it costs exactly as much as one that has.

Deriving it fixes both sports at once and configures neither:

| | snapshots | credits |
| --- | --- | --- |
| Basketball, soccer's default moments | 547 | 5,461 |
| Basketball, moments derived from its statistics | **274** | **2,731** |
| Soccer, moments derived from its statistics | 433 | 4,321 |

Soccer still buys the in-play price, **because soccer has half-time snapshots to pair it with**. Basketball does not, so
it does not. Neither was told which sport it is.

## D10. A price is never "fixed"

**Decision**: odds value columns are always time-varying and always carry the provider in their name.

**Rationale**: a bug basketball found. `fixed` means "constant within a match", and it is computed by grouping on the
identity columns — which do **not** include `provider`. With a **single** provider, every odds column is trivially
constant within a match, so it was marked fixed and **lost its provider prefix**, and the bettor then rejected the odds
it was handed.

Soccer never hits this only because football-data happens to ship **two** providers (`market_average`,
`market_maximum`). Any single-bookmaker odds source breaks — which is an entirely plausible `OddsApi` configuration.

An odds price is, by definition, *a provider's price at a moment*. It cannot be a property of the match alone. So it can
never be fixed, and the derivation is told so. The change is a **no-op for soccer** — verified against its frozen
fingerprint — and a fix for every source with one provider.
