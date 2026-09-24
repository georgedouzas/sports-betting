# Data Model: Basketball, starting with the EuroLeague

**Nothing here is new.** Basketball reuses the entities of `002` exactly. This document says what each of them *contains*
for a basketball game, and where the sport differs from soccer — because those differences are derived from the data
rather than declared, and it is worth being explicit about which ones exist.

## Game (a row of the `stats` snapshots)

| Column | Basketball | Same as soccer? |
| --- | --- | --- |
| `date` | The tip-off instant, **UTC**. The API publishes CET; the source converts. | Yes — the convention is universal |
| `event_status` | `preplay` (form) or `postplay` (result) | Yes |
| `event_time` | `0min` for both | Yes |
| `league` | `Euroleague` | Yes |
| `division` | `1` | Yes |
| `year` | The year the season **ends** — the API's `E2024` becomes `2025` | Yes |
| `home_team` / `away_team` | `ALBA BERLIN`, `PANATHINAIKOS AKTOR ATHENS` | Yes |

**The invariant that matters**: `date + event_time` is the wall-clock instant of the snapshot. It is the address a
time-stamped odds source is asked for, and it is why the time zone had to be determined rather than assumed.

## Form (the pre-play value columns)

Points scored and conceded, per team, over the season so far and over recent games:

- `home_points_for_avg`, `home_points_against_avg`, `home_points_for_latest_avg`, `home_points_against_latest_avg`
- the same four for `away_`

**Shifted by one**, so a game never sees its own result. Computed on a frame with the upcoming fixtures already appended,
so an upcoming game carries the form of the games before it.

**Not** the soccer feature set. Soccer has shots, corners and cards, so it can build an adjusted-goals figure. The
EuroLeague results feed carries a **score line and nothing else**, and inventing basketball advanced statistics from two
numbers would be making things up.

## Outcome (the post-play value columns)

| | Basketball | Soccer |
| --- | --- | --- |
| Winner | `home_win`, `away_win` | `home_win`, `draw`, `away_win` |
| Totals | **none** | `over_2.5`, `under_2.5` |

**No draw**, because the sport has none — a tie goes to overtime. Nothing is configured to make this happen: the bettor
derives the complementary markets from the columns the data carries, so a two-way outcome falls out of a dataset with no
`draw` column, and soccer keeps its three-way one.

**No totals**, because basketball has no *line*. Total points run 125–229 with a different bookmaker line every game, and
the library expresses a market as a **column**. `over_167.5` would be true of one game and meaningless for the next. A
market whose line moves per row is a change to the data model, and its own feature.

## Odds (a row of the `odds` snapshots)

Unchanged. One row per provider, carrying `home_win` and `away_win`. The identity columns are the **statistics'** identity,
not the vendor's — the reconciler rewrites them, so the two tables line up exactly rather than nearly.

## Reconciliation

Unchanged, and for basketball it is the **normal path** rather than an edge case: there is no single feed carrying both
statistics and odds, so the two are *always* different sources.

The names are harder than soccer's, because sponsors are welded onto club names:

| Statistics | The vendor writes it differently |
| --- | --- |
| `PANATHINAIKOS AKTOR ATHENS` | … |
| `CRVENA ZVEZDA MERIDIANBET BELGRADE` | … |
| `MACCABI PLAYTIKA TEL AVIV` | … |
| `EA7 EMPORIO ARMANI MILAN` | … |

The roster-bijection pairing runs **within a season**, where both sources hold the same clubs. Aliases are added only
where it genuinely cannot place one, with a reason. A wrong alias attaches one club's odds to another club's game and says
nothing about it.

## What a basketball dataloader cannot do

**Nothing to predict without odds.** `targets_` is derived from the odds table's value columns, so a dataloader with no
odds source has no markets and therefore no `Y`. That is coherent for a betting library — a market you cannot price is not
a market — and it is why basketball requires an odds source rather than degrading to a statistics-only dataset.

It must **say so**. Today it dies inside schema validation complaining about a dtype, which tells the user nothing about
the fact that they forgot the odds.
