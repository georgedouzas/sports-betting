# Contract: Source

`src/sportsbet/datasets/_sources/_base.py`

A source knows *what* raw items a selection of parameters needs, and *how* to turn the returned payloads into long snapshots. It does not fetch, does not cache, and does not touch the filesystem.

## `BaseSource` (ABC)

```python
class BaseSource(ABC):

    name: ClassVar[str]

    @abstractmethod
    def available_params(self, payloads: list[RawPayload]) -> list[Param]:
        """Return the league/division/year combinations this source publishes."""

    @abstractmethod
    def index_items(self) -> list[RawItem]:
        """Return the items needed to discover what is available."""

    @abstractmethod
    def required_items(self, params: list[Param]) -> list[RawItem]:
        """Return the raw items needed to cover the selected parameters."""

    @abstractmethod
    def to_snapshots(self, payloads: list[RawPayload]) -> pd.DataFrame:
        """Transform the raw payloads into a long snapshot table."""

    def estimate(self, items: list[RawItem]) -> int:
        """Return the units this source would charge to fetch the items. Free sources return 0."""
        return sum(item.cost for item in items)
```

`BaseStatsSource` and `BaseOddsSource` are marker subclasses. They exist so the dataloader can type-check that `stats=` receives a stats source and `odds=` an odds source — a mistake that would otherwise surface as a confusing schema error deep inside extraction.

## Rules

1. **No I/O.** A source that opens a socket or a file violates the contract. This is what makes `prepare(dry_run=True)` free (FR-012) and extraction fetch-free (FR-013) by construction, rather than by discipline.
2. **`required_items` is pure and cheap.** It is called on every `prepare()`, including dry runs. It must be deterministic: the same `params` produce the same items in the same order.
3. **Item keys are shared deliberately.** `FootballDataStats` and `FootballDataOdds` return items with identical `(source, key)` pairs, because they read the same upstream CSV. The store fetches each item once and hands the same payload to both.
4. **`to_snapshots` returns the long format.** `stats` sources return the `stats` table; `odds` sources return the `odds` table. Both conform to the existing schemas from `001-in-play-betting` — this feature does not change them.
5. **Credentials never enter a `RawItem`.** They go on the request headers at fetch time, from the source's own configuration. A `RawItem` is written to the manifest; a credential must never be (FR-027).
6. **No executable doctests.** Sources are network-adjacent and `--doctest-modules` runs everything in `src/`. Document with prose and non-executed fenced blocks.

## Concrete sources

### `FootballDataStats` / `FootballDataOdds`

Free, key-less. Both carry the relocated ETL.

- `index_items()`: the league index pages for the selected leagues (only those — not all 27).
- `required_items(params)`: one item per league-season CSV, plus the single global fixtures CSV. `volatile=True` for the current season and the fixtures file; `False` for completed seasons.
- `to_snapshots(payloads)`: `_preprocess_data` → `_convert_data_types` → `_extract_features` → `_to_snapshots`, ported from the `data` branch. See research D4 for the behaviour that must be preserved exactly.
- `cost`: 0.

### `OddsApi`

```python
OddsApi(key: str, markets: list[str] | None = None, regions: list[str] | None = None)
```

- `required_items(params)`: for completed seasons, one historical-snapshot item per match per requested moment; for the current season, the upcoming/in-play endpoint. `volatile=True` only for the latter.
- `cost`: the vendor's published multiplier, per market-region combination, with historical endpoints charged at a multiple of live ones. Pinned against the live documentation during Phase 5, not guessed.
- The key is held on the instance and injected into the request at fetch time. It is never persisted and never appears in an error message.
