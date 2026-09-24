# Data Model: The NBA, as a second basketball league

Nothing here is new. The NBA produces the same long snapshots the EuroLeague produces, which is why it needs no
dataloader, no schema and no engine change. This document records the mapping from what the feed says to what the
library holds.

## Entities

### Season

A competition year, **named by the year it ends in**. `E2024` would be 2025 in the EuroLeague; ESPN's `2026` is already
2026, because ESPN names a season by its end year too. **No conversion** (research D3).

| Field | Value | Source |
| --- | --- | --- |
| `league` | `'NBA'` | constant |
| `division` | `1` | constant |
| `year` | the year the season ends | the seasons index, never a fabricated range |

The catalogue offered to a user is the **intersection** of this with the odds vendor's coverage, computed by the engine.
The league publishes 81 seasons; the vendor's history is far shorter; only the overlap is selectable.

### Game

A contest between two clubs at a known instant. It is either **played** — and has a score — or it is not, and is a
**fixture**.

| Field | Meaning | Rule |
| --- | --- | --- |
| `date` | the tip-off, **as an instant in UTC** | read from the feed, never inferred (research D6) |
| `home_team`, `away_team` | the two clubs | the canonical names, which the odds vendor also uses |
| `home_points`, `away_points` | the final score | present only if played |
| played | whether it happened | a per-game flag, **never** inferred from the season being over (research D7) |

**A game is in the dataset unless it is a pre-season game or an exhibition.** The regular season, the play-in and the
post-season are all in. The rule is stated as an exclusion, not an inclusion, so that an unrecognised label is dropped
rather than admitted (FR-011).

### Snapshot

The long row the library actually holds. A game becomes **one or two** of them.

| Snapshot | When | Carries |
| --- | --- | --- |
| **pre-play** | always, for every game | the form both teams brought into the game |
| **post-play** | only if the game was played | the final score and the two-way outcome |

`event_status` is `preplay` / `postplay`; `event_time` is `0` for both, because this source has no in-play moments. A
fixture therefore has a pre-play snapshot and **no** post-play snapshot, which is exactly what makes it a fixture rather
than a training row with a result nobody knows.

## Markets

`home_win`, `away_win`. Derived with the existing public `market_outcomes`.

- **No draw.** A tie goes to overtime, so it cannot happen.
- **No totals.** A bookmaker sets a different line for every game, and the library expresses a market as a *column*. A
  market whose line moves per row is not a column; it would be a change to the data model, and it would change soccer
  too.

The bettor is never told any of this. `complementary_events` derives from the data that `home_win` and `away_win` are
mutually exclusive, exactly as it derives that soccer's three outcomes are.

## Features

The EuroLeague's shape, over the only statistics a score line supports.

| Feature | Per team |
| --- | --- |
| `points_for` | expanding mean, and rolling mean over the last 3 games |
| `points_against` | expanding mean, and rolling mean over the last 3 games |
| `wins` | expanding mean, and rolling mean over the last 3 games |

Each is **shifted by one**, computed on a frame that **already has the fixtures appended**. Neither is stylistic:

- the **shift** is why a game can never see its own result;
- the **ordering** is why an upcoming fixture carries the form of the games played before it.

The feed carries a score line and nothing else. Inventing basketball advanced statistics from two numbers would be
making things up, so `points_for`, `points_against` and the wins that follow from them are the whole of it.

## What is deliberately absent

- **Player and box-score features.** Out of scope, as for the EuroLeague.
- **In-play snapshots.** The feed carries period scores and `date + event_time` addresses any moment, so the road is
  open — but the first cut is pre-play and post-play.
- **A new schema.** The NBA is the same shape as the EuroLeague, and the statistics schema already validates it. Needing
  a new one would mean the shape had changed, and it has not.
