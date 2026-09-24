# Phase 0 Research: In-Play (Live) Betting Support

This document resolves the open clarification and the key technical decisions required
before design. Each entry follows Decision / Rationale / Alternatives.

## R1 — Reinforcement-learning API placement and scope (resolves FR-011 `[NEEDS CLARIFICATION]`)

**Decision**: Remove reinforcement learning from `extract_train_data` entirely. The
`learning_type` argument accepts only `'supervised'` (default) and `'unsupervised'`, so
`extract_train_data` **always returns a uniform three-tuple** `(X, Y, O)` — `Y` is `None` when
unsupervised — never an environment. The uniform shape keeps every call site unpackable as
`X, Y, O = extract_train_data(...)`. Reinforcement learning becomes a *separate future method* (working name
`make_env()`) whose full design is captured below as a forward contract. **No RL code ships
in this feature** — not even a `NotImplementedError` stub method. `'reinforcement'` is simply
not a valid `learning_type` value and is rejected like any other invalid value.

**Rationale**: RL does not fit the extraction paradigm. `extract_train_data` is designed to
return a static tabular tuple consumed by a scikit-learn-style bettor; RL needs a *stateful
environment* (`reset()`/`step()`) consumed by an *agent*. Overloading one method to return
either a tuple or an environment based on a string argument is a polymorphic return type:
un-annotatable (violates Principle II), un-unpackable by callers doing
`X, Y, O = extract_train_data(...)`, and at odds with the scikit-learn contract (Principle I).
Isolating RL behind its own method keeps the tabular API type-consistent. The environment is
also entirely undesigned in the source branch (a bare "gym.Env" mention), `gymnasium` is not
a dependency, and the real feed has no in-play data to train against — so shipping only the
design (not code) is the honest scope.

**Alternatives considered**:
- *Keep `learning_type='reinforcement'` returning an env* — rejected: polymorphic return,
  type-unsafe, breaks tuple-unpacking call sites and the estimator contract.
- *`NotImplementedError` stub method now* — rejected by product decision: "design doc, no
  code" was chosen over a placeholder stub. A stub adds an untested code path and a de-facto
  API commitment before the design is validated.
- *Full gym environment now* — rejected: large, speculative, unvalidatable with current data.

**Future RL contract (design only — not implemented this feature)**:
- **Method**: `make_env(...)` on the dataloader (name provisional), returning a Gymnasium-
  compatible environment; `gymnasium` would be added as an optional extra when implemented.
- **Observation**: the snapshot feature state (`X`-style row) available at the current
  in-match moment as the episode advances through `preplay → inplay → postplay`.
- **Action**: per betting market, bet / no-bet (optionally a stake fraction) at the current
  snapshot.
- **Reward**: realized betting return settled at `postplay` (net profit/loss on placed bets),
  optionally shaped per snapshot.
- **Episode**: one match, stepping through its ordered snapshots from `preplay` to `postplay`.
- **Consumption**: an external RL agent, not the existing bettor classes — this is why it
  lives outside `extract_train_data` and outside `sportsbet.evaluation`'s bettor contract.
- **Validation**: gated on an in-play data source existing (see R2); designed to reuse the
  same schemas and snapshot model so no data-layer rework is needed when implemented.

## R2 — In-play data availability ("engine now, data later")

**Decision**: Build and validate the in-play extraction engine against bundled synthetic
sample data (`tests/samples/stats.csv`, `odds.csv`, and the sample dataloader). The concrete
`SoccerDataLoader` maps the real football-data-style feed into snapshots that populate
`preplay` and `postplay` only; in-play targets against the real feed legitimately return no
rows until an in-play source exists.

**Rationale**: Confirmed product decision. The sample data already contains in-play snapshots
at multiple `event_time`s, which is sufficient to exercise and doctest the full engine
offline (Principle III), while the real loader stays honest about what the upstream source
provides.

**Alternatives considered**:
- *Block on a real in-play feed* — rejected: no such source exists; would stall the feature.
- *Fabricate in-play rows in the real loader* — rejected: would produce misleading data.

## R3 — Concrete `SoccerDataLoader` UX (design decision for the "you decide" answer)

**Decision**: Preserve the existing scikit-learn-style selection interface on the concrete
loader — `param_grid` construction, `get_all_params()` and `get_odds_types()` discovery
(class/instance methods), plus `drop_na_thres` and `odds_type` extraction options — layered
on top of the new snapshot model. Internally the loader downloads/loads long-format rows,
constructs the validated `stats`/`odds` frames and concrete schemas, and delegates the pivot
to the base engine, additionally accepting the new `target_event_status` / `target_event_time`
arguments.

**Rationale**: Satisfies Constitution Principle I (scikit-learn compatibility, `ParameterGrid`
parity) and minimizes disruption for existing users and existing docs/examples. The base
class keeps its low-level `(stats, odds, schema, targets)` constructor for direct/advanced
use; the soccer loader is the convenience layer.

**Alternatives considered**:
- *Clean-break constructor-only API* (drop `param_grid`) — rejected: breaks existing users,
  contradicts Principle I, orphans the user guide.
- *Two parallel APIs* — rejected: duplicate surfaces, higher maintenance, violates parity.

## R4 — `X` / `Y` / `O` column-naming contract (the bettor-migration linchpin)

**Decision**: Adopt a single, stable, parseable flattened-column contract emitted by
extraction and consumed by bettors:
- Fixed (time-invariant) features/odds: bare column name (e.g. `league`, `home_team`).
- Time-varying features: `{col}__{event_status}__{event_time}` (e.g. `home_goals__inplay__60min`).
- Odds: `{provider}__{col}__{event_status}__{event_time}` (e.g. `bet365__home_win__preplay__0min`).
- Targets (`Y`): `{col}__{target_event_status}__{target_event_time}`.
The delimiter is a fixed double underscore `__`; `event_time` is rendered as whole minutes
(`{n}min`). Bettors' odds-column parsing (`BaseBettor._get_feature_names_odds`,
`OddsComparisonBettor._check_odds_types`) is updated to parse market names from this contract
instead of the legacy `odds__market__outcome__market_type` convention.

**Rationale**: The bettors currently infer betting markets and odds by string-parsing column
names; a single documented naming grammar is the minimal, testable bridge between the new
extraction output and the existing estimators, avoiding a redesign of the bettor internals.

**Alternatives considered**:
- *MultiIndex columns* — rejected: doctest/readability and downstream scikit-learn
  compatibility are worse; the current base already flattens.
- *Separate metadata sidecar mapping columns→markets* — rejected: more moving parts than a
  self-describing name; harder to keep aligned across save/reload.

## R5 — Schema-driven extraction mechanics

**Decision**: Keep the `pandera` `DataFrameModel` approach already present in
`_base/_schema.py`: `required_col()`/`optional_col(include, fixed)` fields carrying metadata
(`snapshot`, `include`, `fixed`), with helpers `snapshot_cols()`, `col_metadata()`, and
dataframe checks (event-status/time consistency, snapshot uniqueness, post-match odds
nullability). Concrete soccer schemas subclass the base stats/odds schemas.

**Rationale**: Already implemented and tested for the base; directly implements Principle II;
metadata cleanly parameterizes the pivot (which statuses to include, which columns are fixed).

**Alternatives considered**:
- *Hand-rolled validation* — rejected: violates Principle II's "schemas, not ad-hoc checks".

## R6 — Reintroducing the sample (dummy) dataloader

**Decision**: Restore a `DummySoccerDataLoader` that produces in-play sample snapshots (the
same shape as `tests/samples/*.csv`) with no network access, exposing the same selection
interface as `SoccerDataLoader`. It is the primary vehicle for doctests, the CLI test
`CONFIG`, and documentation examples.

**Rationale**: FR-020 and Principle III require an offline, runnable end-to-end path; the old
dummy loader was deleted but is still referenced by `tests/conftest.py`'s CLI `CONFIG`.

**Alternatives considered**:
- *Network-dependent examples* — rejected: breaks offline/doctest runs and CI determinism.

## R7 — Persistence

**Decision**: Keep `cloudpickle`-based `save`/`load_dataloader` (already on the base) and the
existing `save_bettor`/`load_bettor`. Re-export `load_dataloader` from `datasets` (currently
dropped).

**Rationale**: Existing, proven mechanism; FR-015 requires save/reload with consistent
columns. Only the missing re-export needs restoring.

**Alternatives considered**: none warranted.

## Summary of resolved unknowns

| Ref | Question | Resolution |
|-----|----------|-----------|
| R1/FR-011 | RL API placement + scope | Removed from `extract_train_data`; separate future `make_env()`; design-only, no code |
| R2 | In-play data reality | Engine validated on sample data; real feed pre/post only |
| R3 | Soccer loader UX | Keep `param_grid` selection layered on snapshot model |
| R4 | Column contract for bettors | Fixed `__`-delimited naming grammar; bettors parse it |
| R5 | Extraction mechanics | Reuse pandera metadata-driven pivot |
| R6 | Offline demo path | Restore `DummySoccerDataLoader` with in-play sample data |
| R7 | Persistence | Keep cloudpickle; restore `load_dataloader` export |

All `[NEEDS CLARIFICATION]` markers are resolved. Ready for Phase 1.
