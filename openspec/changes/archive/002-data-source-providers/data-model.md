# Data Model: Pluggable Data-Source Providers

## RawItem

The unit of fetching, caching, resuming and cost. Immutable.

| Field | Type | Meaning |
| --- | --- | --- |
| `source` | `str` | Name of the source that declared it (`football_data`, `odds_api`). |
| `key` | `str` | Stable identity within the source (e.g. `England_1_2024`, `fixtures`, `soccer_epl@2024-08-17T14:00:00Z`). |
| `url` | `str` | Where to fetch it from. |
| `params` | `dict \| None` | Query parameters, if any. Never contains credentials. |
| `volatile` | `bool` | `True` if the upstream content can still change (current season, fixtures, upcoming odds). `False` once frozen (a finished season, a historical snapshot). |
| `cost` | `int` | Units the source will be charged for fetching it. `0` for free sources. |

**Identity**: `(source, key)`. Two sources declaring the same `(source, key)` declare the *same* item and it is fetched once. This is how `FootballDataStats` and `FootballDataOdds` share one download of a league-season CSV.

**Invariant**: an item with `volatile=False` that the store already holds is never re-fetched.

## RawPayload

What a source returned, kept verbatim.

| Field | Type | Meaning |
| --- | --- | --- |
| `item` | `RawItem` | What was asked for. |
| `content` | `bytes` | Exactly what came back. Not parsed, not normalised. |
| `fetched_at` | `Timestamp` | When. |

**Invariant (FR-016)**: never deleted, never rewritten. Metered data cannot be re-obtained for free, so the raw layer is the only thing in the system that is genuinely irreplaceable. Every derived table is a pure function of it (FR-017).

## Store manifest entry

One JSONL line per held item. The index that makes planning a set-difference.

| Field | Type | Meaning |
| --- | --- | --- |
| `source` | `str` | |
| `key` | `str` | |
| `volatile` | `bool` | Whether it must be refreshed on the next prepare. |
| `fetched_at` | `Timestamp` | |
| `digest` | `str` | Content hash of the raw payload. Detects corruption and no-op refreshes. |
| `bytes` | `int` | |

## PreparationReport

What `prepare(dry_run=True)` returns and `prepare()` produces. The user's decision-support object.

| Field | Type | Meaning |
| --- | --- | --- |
| `to_fetch` | `list[RawItem]` | Missing, or held-but-volatile. |
| `held` | `list[RawItem]` | Already in the store and immutable. |
| `estimated_cost` | `dict[str, int]` | Units per source. An *estimate*, computed from the plan and the source's published multiplier — not measured. |
| `unavailable` | `list[Param]` | Requested `param_grid` combinations the sources do not publish (e.g. seasons before the vendor's history starts). Reported at plan time, never as a fetch-time 404. |

**Invariant (FR-012)**: constructing this performs zero network calls and incurs zero cost.

## NotPreparedError

Raised by `extract_train_data` / `extract_fixtures_data` when the store does not hold everything the `param_grid` requires. Carries the `PreparationReport`, so the message can state exactly what is missing and what obtaining it would cost.

**Invariant (FR-013)**: extraction raises this rather than fetching. There is no configuration that turns this into a fetch.

## ReconciliationReport

Produced when statistics and odds come from different sources.

| Field | Type | Meaning |
| --- | --- | --- |
| `matched` | `int` | Matches present in both sources and joined. |
| `unmatched_stats` | `DataFrame` | Stats rows with no odds counterpart. |
| `unmatched_odds` | `DataFrame` | Odds rows with no stats counterpart. |
| `unmatched_rate` | `float` | `unmatched / total`. |

**Invariant (FR-023)**: `unmatched_rate > max_unmatched_rate` raises. It never degrades into silently missing odds — that is the failure mode that produces a confidently wrong backtest.

## Match identity

The key the resolver produces so a statistics row and an odds row from different sources refer to the same event.

| Field | Type | Notes |
| --- | --- | --- |
| `league` | `str` | Each source's competition key maps to the canonical league. |
| `division` | `int` | |
| `year` | `int` | |
| `date` | `Timestamp` | Compared within a tolerance window, not for exact equality — sources disagree on kickoff by minutes and on timezone by hours. |
| `home_team` | `str` | Normalised through the source's alias table. |
| `away_team` | `str` | Normalised through the source's alias table. |

For the free path this is identity by construction: both sources derive from the same upstream row, so no resolution runs.

## Unchanged entities

The snapshot tables that `_snapshots()` returns — long `stats` and `odds` with `event_status` and `event_time` — are **not modified by this feature**. Their schema, their column grammar (`{col}__{status}__{time}`, `{provider}__{col}__{status}__{time}`) and their pandera validation stay exactly as `001-in-play-betting` left them. This feature changes only where those tables come from.
