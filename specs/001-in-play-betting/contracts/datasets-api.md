# Contract: `sportsbet.datasets` public API

The library's user-facing data interface. Column-naming grammar per
[research.md R4](../research.md).

## Public exports (`sportsbet.datasets.__all__`)

MUST expose:
- `BaseDataLoader`
- `SoccerDataLoader`
- `DummySoccerDataLoader`
- `load_dataloader`
- `BaseStatsSchema`, `BaseOddsSchema`
- `required_col`, `optional_col`

> Regression from `main`: `SoccerDataLoader`, `DummySoccerDataLoader`, and `load_dataloader`
> are currently NOT exported and MUST be restored.

## `BaseDataLoader`

```text
BaseDataLoader(stats, odds, stats_schema, odds_schema, targets)
```

### `extract_train_data(learning_type=None, target_event_status=None, target_event_time=None)`

- `learning_type`: `'supervised'` (default) | `'unsupervised'`. No other value is valid;
  reinforcement learning is a separate future method, not a `learning_type` (research.md R1).
- `target_event_status`: `'inplay'` | `'postplay'` (default `'postplay'`).
- `target_event_time`: time delta ≥ 0 (default 0).
- **Behavior**: validate `stats`/`odds` against schemas → assert snapshot-cols match →
  ensure in-play/post-play events exist → pivot into wide tables.
- **Returns**: always a uniform three-tuple `(X, Y, O)` sharing one date index —
  supervised populates `Y`; unsupervised sets `Y=None`. Callers can always unpack
  `X, Y, O = extract_train_data(...)`.
- **Errors**: `pandera.errors.SchemaError` (invalid data); `ValueError` (no resolvable
  events, invalid `learning_type` — including `'reinforcement'` — invalid
  `target_event_status`, negative time); `AssertionError` (snapshot-col mismatch).

### `extract_fixtures_data()`  *(to be implemented)*

- **Precondition**: `extract_train_data` called first (defines column layout).
- **Returns**: `(X, None, O)` with column structure identical to the training extraction.

### `save(path)` / `load_dataloader(path)`

- Round-trip a configured loader; reloaded loader reproduces the same fixtures columns.

## `SoccerDataLoader(BaseDataLoader)` / `DummySoccerDataLoader`

Convenience layer preserving the established scikit-learn-style interface.

```text
SoccerDataLoader(param_grid=None)
SoccerDataLoader.get_all_params() -> list[dict]         # class method, no download
loader.get_odds_types() -> list[str]
loader.extract_train_data(drop_na_thres=0.0, odds_type=None,
                          target_event_status=None, target_event_time=None) -> TrainData
loader.extract_fixtures_data() -> FixturesData
```

- `param_grid` mirrors scikit-learn `ParameterGrid` selection (league/division/year …);
  `None` = all.
- `drop_na_thres` ∈ `[0.0, 1.0]`: drop feature columns above this missingness; applied
  identically to training and fixtures.
- `odds_type`: one of `get_odds_types()`; `None` ⇒ no odds returned.
- `DummySoccerDataLoader` behaves identically using bundled in-play sample data (no network).

### Doctest contract (must run offline)

```python
>>> from sportsbet.datasets import DummySoccerDataLoader
>>> loader = DummySoccerDataLoader(param_grid={'league': ['England']})
>>> X_train, Y_train, O_train = loader.extract_train_data(
...     odds_type='market_average', target_event_status='postplay')
>>> X_fix, Y_fix, O_fix = loader.extract_fixtures_data()
>>> Y_fix is None
True
>>> import pandas as pd
>>> pd.testing.assert_index_equal(X_train.columns, X_fix.columns)
>>> pd.testing.assert_index_equal(O_train.columns, O_fix.columns)
```

An in-play target on the sample loader must succeed and exclude post-target information:

```python
>>> X60, Y60, O60 = loader.extract_train_data(
...     target_event_status='inplay', target_event_time=pd.Timedelta('60min'))
>>> all('90min' not in c for c in X60.columns)
True
```

Reinforcement learning is **not** a valid `learning_type` (it is a separate future method,
`make_env()`, not implemented this feature — research.md R1):

```python
>>> loader.extract_train_data(learning_type='reinforcement')   # doctest: +SKIP
ValueError: Invalid learning type. It should be one of (supervised, unsupervised)...
```
