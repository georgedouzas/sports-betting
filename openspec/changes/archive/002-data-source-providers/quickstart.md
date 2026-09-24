# Quickstart: validating the feature

How to prove each phase actually works. Every scenario below is runnable and maps to acceptance criteria in [spec.md](spec.md).

## Prerequisites

```bash
pdm install -dG tests
```

The suite must never touch the network. A socket guard in `tests/conftest.py` enforces this: if any test attempts a connection, the suite fails. That guard is itself the proof of SC-004.

## Scenario 1 — The free path reproduces today's data (US1, SC-002)

The equivalence gate. Frozen in Phase 0, and the thing every later phase is checked against.

```bash
pdm tests -k equivalence
```

**Expects**: the `stats`/`odds` long tables and the extracted `X`/`Y`/`O` for the reference `param_grid` (England, division 1, one completed season) match the committed fingerprint — per-column hashes, shapes, dtypes and the ordered column list.

**Fails loudly when**: any column of the client-side output drifts from the mirrored output. The failure names the column, so you know where to look.

To regenerate the reference frames locally for a row-level diff (they are deliberately not committed — see research D7):

```bash
python -m tests.samples.capture_reference   # writes to a gitignored path
```

## Scenario 2 — Nothing fetches without `prepare()` (US3, SC-004)

```python
loader = SoccerDataLoader(param_grid={'league': ['England'], 'division': [1], 'year': [2024]})
loader.extract_train_data()   # raises NotPreparedError
```

**Expects**: `NotPreparedError`, naming the missing items and what obtaining them would cost. **No network call is attempted** — verified by the socket guard, not by inspection.

## Scenario 3 — Know the cost before paying it (US3, SC-005)

```python
loader = SoccerDataLoader(
    param_grid={'league': ['England'], 'division': [1], 'year': [2024]},
    odds=OddsApi(key=..., markets=['h2h']),
)
report = loader.prepare(dry_run=True)
report.to_fetch        # what would be fetched
report.held            # what is already there
report.estimated_cost  # {'odds_api': 1240}
report.unavailable     # seasons outside the vendor's history
```

**Expects**: counts and an estimate. **Zero credits consumed, zero requests made.**

## Scenario 4 — Prepare, then extract (US1)

```python
loader = SoccerDataLoader(param_grid={'league': ['England'], 'division': [1], 'year': [2024]})
loader.prepare()
X, Y, O = loader.extract_train_data()
```

**Expects**: only England's index page and the selected season CSVs are fetched — not the full ~900-file catalogue (SC-010).

## Scenario 5 — Incremental and free to re-run (SC-006, SC-007)

```python
loader.prepare()                       # again
```

**Expects**: fetches nothing. Completed seasons are immutable and already held; only the fixtures file and the in-progress season are volatile.

Then change the feature engineering and rebuild:

```python
loader.prepare()                       # after a transform change
```

**Expects**: still fetches nothing, and still costs nothing. Derived tables are rebuilt from the retained raw payloads (FR-017). This is the property that means a user who paid for historical odds never pays again because we improved a feature.

## Scenario 6 — A bad join fails loudly (US4, SC-008)

```bash
pdm tests -k resolver
```

**Expects**: with deliberately mismatched team naming between two sources, the reported unmatched rate is accurate and extraction **raises** rather than returning a frame with missing odds.

This is the scenario that matters most. A silently failed join does not look like a bug; it looks like a slightly smaller dataset and a suspiciously clean backtest.

## Scenario 7 — Offline entry points are untouched (US5, SC-011)

```bash
pdm tests -k "dummy or factory"
```

**Expects**: `DummySoccerDataLoader`, `from_snapshots` and `from_dataframe` work with no store, no source and no `prepare()`. Passing unmodified is the point — if these needed editing, the migration broke the escape hatches the rest of the verification depends on.

## Scenario 8 — No odds are redistributed (SC-001)

```bash
git ls-tree -r --name-only origin/data | grep -c odds   # must be 0 after Phase 4
```

**Expects**: after Phase 4, no branch and no published artifact contains third-party odds data. The only in-repo remnant is the fingerprint, which is hashes rather than data.

## Full gate

```bash
pdm run formatting
pdm run checks
pdm run tests
```

All three must pass before any phase is considered done (Constitution IV).
