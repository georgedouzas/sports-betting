# Contract: Store

`src/sportsbet/datasets/_store.py`

The store owns fetching and persistence. It is the only component in the feature that performs I/O.

## `BaseStore` (ABC)

```python
class BaseStore(ABC):

    @abstractmethod
    def held(self, items: list[RawItem]) -> list[RawItem]:
        """Return the subset already held and not stale. Performs no network access."""

    @abstractmethod
    def fetch(self, items: list[RawItem], headers: dict[str, str] | None = None) -> list[RawPayload]:
        """Fetch the items and persist their raw payloads. The only method that touches the network."""

    @abstractmethod
    def read(self, items: list[RawItem]) -> list[RawPayload]:
        """Return the raw payloads for items already held. Raises if any is missing."""

    @abstractmethod
    def write_snapshots(self, source: str, kind: str, data: pd.DataFrame) -> None:
        """Persist a derived snapshot table."""

    @abstractmethod
    def read_snapshots(self, source: str, kind: str, params: list[Param]) -> pd.DataFrame:
        """Read the derived snapshot table for the selected parameters."""
```

## `LocalStore`

```python
LocalStore(path: Path | str | None = None)
```

Defaults to a platform-appropriate user data directory. Layout:

```text
<path>/
├── raw/<source>/<key>.gz          # RawPayload.content, verbatim, forever
├── snapshots/<source>/<kind>/     # Parquet, zstd, partitioned by league/division/year
└── manifest.jsonl                 # one line per held item
```

## Rules

1. **Atomic writes (FR-019).** Every write goes to a temporary file in the destination directory and is then `os.replace`d into position. A partial write is never visible under its final name, so an interrupted `prepare()` leaves the store readable and resumable, never half-corrupt.
2. **Raw is append-only (FR-016).** `fetch` writes a raw payload once. There is no method that deletes or rewrites one. Re-fetching a volatile item writes a *new* payload and supersedes the manifest entry; the old payload is retained.
3. **Derived is disposable (FR-017).** `write_snapshots` may be called any number of times and always overwrites. Rebuilding every derived table from raw must fetch nothing and cost nothing — that is what lets the feature engineering change without re-charging the user.
4. **Staleness is a property of the item, not the store.** `held()` returns an item as held if the manifest has it and it is not `volatile`. A volatile item is always re-fetched. There is no TTL to tune and no cache to invalidate by hand.
5. **Dtypes survive the round trip (FR-020).** `read_snapshots(write_snapshots(df)) == df`, dtypes included. This is the requirement that rules out CSV, and it exists because an empty CSV read back as all-object has already broken schema validation once.
6. **Concurrency.** Two processes preparing the same store must not corrupt it. Atomic rename gives this for individual items; the manifest is append-only JSONL, which tolerates interleaved appends and is truncated to its last complete line on read.
7. **Credentials are never written.** `fetch` receives them as headers and forgets them. Nothing in `raw/`, `snapshots/` or `manifest.jsonl` may contain a key (FR-027).
