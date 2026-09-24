# Contract: Dataloader

`src/sportsbet/datasets/_base/_dataloader.py`, `src/sportsbet/datasets/_soccer/_dataloader.py`

## What does not change

`BaseDataLoader` remains an ABC whose single abstract method is:

```python
@abstractmethod
def _snapshots(self) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the long stats/odds snapshots."""
```

The extraction engine, the column grammar, the input horizon, `extract_train_data`, `extract_fixtures_data`, `from_snapshots`, `from_dataframe` and `DummySoccerDataLoader` are all untouched. This feature is an implementation *behind* `_snapshots()`, not a change *to* it.

## What is added

```python
class BaseDataLoader(ABC):

    def prepare(self, dry_run: bool = False) -> PreparationReport:
        """Populate the store for the selected parameters.

        Incremental and resumable: only what the store does not already hold is
        fetched. With ``dry_run=True`` nothing is fetched and nothing is spent —
        the returned report says what *would* be.
        """
```

Loaders backed by sources implement `_snapshots()` by reading from the store, and raise `NotPreparedError` if it is not populated. Loaders that are not backed by sources — `DummySoccerDataLoader`, and everything built by `from_snapshots`/`from_dataframe` — inherit a `prepare()` that is a no-op returning an empty report, so they keep working with no store and no source (FR-028).

## `SoccerDataLoader`

```python
SoccerDataLoader(
    param_grid=None,
    stats=FootballDataStats(),
    odds=FootballDataOdds(),
    store=LocalStore(),
    max_unmatched_rate=0.0,
)
```

Constructor parameters are stored unmodified and unvalidated (Constitution I). Every source-specific setting — an API key, a market list, a region — lives on the source object, not here. That is the point: adding a source must not widen this signature.

The realistic mixed configuration:

```python
SoccerDataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2024]},
    stats=FootballDataStats(),
    odds=OddsApi(key=..., markets=['h2h', 'totals']),
)
```

## Discovery is not on the dataloader

`get_all_params()` does **not** exist. Writing a `param_grid` requires knowing what is available, so discovery cannot
live on an object you construct *with* a `param_grid` — that is a chicken-and-egg, and it is why the method was awkward
to place. Ask the source instead:

```python
FootballDataStats().available_params()
```

The dataloader keeps a **private** `_all_params()`, used for one thing only: refusing to request a combination the
sources do not publish. It returns the **intersection** of what the statistics source and the odds source publish,
because a season whose statistics exist but whose odds do not cannot be modelled (FR-034). Asking only the statistics
source — as an earlier version did — would offer seasons back to 1994 alongside an odds vendor whose history starts in
2020, and the missing odds would surface as a silently smaller dataset. That is precisely the failure this feature
exists to prevent.

## Rules

1. **Extraction never fetches (FR-013).** `extract_train_data` and `extract_fixtures_data` raise `NotPreparedError` on an unprepared store. The error carries the `PreparationReport`, so it names what is missing and what obtaining it would cost. There is no flag that turns this into a fetch.
2. **`prepare()` is required on every path**, free or metered. A safety property with an exception is not a safety property; see research D6.
3. **The input horizon applies to fixtures too.** Unchanged from `001-in-play-betting`: whatever caps the features at training time caps them at serving time. Sources do not get to reintroduce asymmetry.
4. **Reconciliation runs only when the sources differ.** With `FootballDataStats` + `FootballDataOdds` the identities are equal by construction — both derive from the same upstream row — so the resolver is skipped entirely.
5. **`max_unmatched_rate` defaults to strict.** Loosening it is a deliberate act, never an accident (FR-023).

## Surface parity (Constitution I)

`prepare()` is a capability, so it must exist on all three delivery surfaces:

- **Python**: `loader.prepare()`.
- **CLI**: a `prepare` subcommand, with `--dry-run` reporting the plan and the cost estimate. The config file mirrors the constructor (`STATS`, `ODDS`, `STORE`), and `dataloader params` asks the statistics source rather than the dataloader.
- **GUI**: a preparation step in the dataloader-creation flow, showing progress and the estimate before it spends anything. Its parameter pickers are populated from the source.

A surface that cannot prepare cannot extract, which would make it useless — so this is not optional polish.
