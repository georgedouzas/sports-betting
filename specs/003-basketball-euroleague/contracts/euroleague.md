# Contract: `EuroLeagueStats`

`src/sportsbet/datasets/_sources/_euroleague.py`

A free, key-less statistics source over the EuroLeague's official public API. It implements the **existing** `BaseSource`
contract from `002` with no additions. That it needs no additions is the point: if a second sport had required a new
method, the abstraction would have been wrong.

## The API (verified live, 200, no key)

| Purpose | Request | Cost |
| --- | --- | --- |
| Catalogue | `GET /v2/competitions/E/seasons` | free |
| A whole season | `GET /v1/results?seasoncode=E{year}` | free |

`v1/results` **without** `gamenumber` returns the entire season — 330 games in one response. With `gamenumber=N` it
returns a single round. So a season is **one** item, not thirty.

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

## The methods

```python
class EuroLeagueStats(BaseStatsSource):

    name: ClassVar[str] = 'euroleague'

    def index_items(self) -> list[RawItem]:
        """The season list. Free, so a preparation can be priced without spending anything."""

    def catalogue(self, payloads: list[RawPayload]) -> list[Param]:
        """The seasons the competition publishes. Read from the API, never a range of years."""

    def required_items(self, params: list[Param], schedule: pd.DataFrame | None = None) -> list[RawItem]:
        """One item per selected season. The schedule is ignored: the source carries its own."""

    def to_snapshots(self, payloads: list[RawPayload]) -> pd.DataFrame:
        """The long `stats` snapshots: pre-play form, and post-play score and outcome."""
```

## Rules

1. **It never fetches.** `index_items`, `catalogue`, `required_items` and `to_snapshots` are pure. The store fetches.
   That is what makes `prepare(dry_run=True)` free and extraction fetch-free *by construction*.

2. **The kick-off is in CET, and must be converted.** The API publishes **every** game in Central European Time,
   regardless of the country it is played in — an Istanbul game reads `18:30` and tips off at 20:30 local. This was
   determined from the data (research D1), not from documentation, which says nothing. Parse as `Europe/Berlin`, convert
   to UTC. **A wrong zone here shifts every odds snapshot the user buys by one to two hours, silently.**

3. **The season is the year it ends in.** The API's `E2024` is the 2024-25 season and carries `year: 2024`. The library's
   convention is the **ending** year, so it becomes `year = 2025`. This must agree with what `OddsApi` publishes for the
   same competition, or the intersection of the two catalogues is empty and nothing is selectable.

4. **A game is played or it is not.** `<played>` decides. An unplayed game gets a pre-play snapshot only, so it becomes a
   fixture and never a training row with an invented result.

5. **The form never sees its own game.** Per-team expanding and rolling means over points scored and conceded, **shifted
   by one**, computed on a frame that already has the upcoming fixtures appended so an upcoming game gets the form of the
   games before it. Both details are load-bearing, and both are carried over from the soccer port rather than reinvented.

6. **Points for and against, and nothing invented.** The results feed carries a score line and nothing else. Soccer's
   adjusted-goals machinery leans on shots, corners and cards, which do not exist here. Fabricating basketball advanced
   statistics from two numbers would be making things up.

7. **Head-to-head only.** The outcome is `home_win` and `away_win`. There is **no draw** — a tie goes to overtime. There
   is **no totals market** either: the line moves per game (research D2), and the library expresses a market as a column.

8. **No executable doctests.** It is network-touching, and `--doctest-modules` runs everything in `src/`.

## `BasketballDataLoader`

```python
BasketballDataLoader(
    param_grid=None,
    stats=EuroLeagueStats(),
    odds=None,                 # there is no free basketball odds source; the user brings their own
    store=LocalStore(),
    aliases=None,
    max_unmatched_rate=0.0,
)
```

It is a class name and two defaults. Everything else — the store plumbing, the preparation, the cost estimate, the
reconciliation, the intersection of the two catalogues — is inherited.

**If it needs to be more than that, the abstraction is wrong** and the fix belongs on the base, not here.
