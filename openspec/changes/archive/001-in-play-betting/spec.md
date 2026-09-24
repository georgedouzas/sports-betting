# Feature Specification: In-Play (Live) Betting Support

**Feature Branch**: `001-in-play-betting`

**Created**: 2026-07-08

**Status**: Draft

**Input**: User description: "Complete the in-progress redesign on the `development`
branch so users can model and bet on outcomes at a chosen point in a match (pre-match
*or* in-play), built on an event-snapshot data model with schema validation. Spans data
loading, schema validation, model backtesting/prediction, and the CLI/GUI surfaces. The
branch is a partial signal, not the complete design."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Extract training data for a chosen match moment (Priority: P1)

A quantitative bettor wants to build a model that predicts a match outcome as it stands
at a specific moment — for example, the result at full time, or the leading side 60
minutes into play. They point a dataloader at historical soccer data, choose the target
moment (event status such as pre-match / in-play / post-match, plus an optional in-play
time), and receive three aligned tables: input features known *before* that moment (`X`),
the target outcomes *at* that moment (`Y`), and the corresponding bookmaker odds (`O`).

**Why this priority**: This is the foundational capability the entire feature rests on.
Without correct, moment-aware extraction there is nothing to model, backtest, or bet on.
It is independently valuable: a user can inspect and analyze the extracted data even
before any model exists.

**Independent Test**: Provide historical snapshot data and request a target of "outcome at
60 minutes in-play"; verify that `X` contains only information timestamped at or before 60
minutes, `Y` contains the outcomes as they stood at 60 minutes, and `O` contains the odds
offered up to that moment — with `X`, `Y`, and `O` sharing the same rows.

**Acceptance Scenarios**:

1. **Given** historical data containing pre-match, in-play, and post-match snapshots,
   **When** the user extracts training data targeting the post-match (full-time) outcome,
   **Then** they receive `X`, `Y`, `O` where `Y` reflects final outcomes and `X` includes
   all pre-match and in-play information.
2. **Given** the same data, **When** the user targets the in-play outcome at 60 minutes,
   **Then** `X` excludes all information from after 60 minutes and `Y` reflects the match
   state at 60 minutes.
3. **Given** data with no in-play or post-match snapshots (pre-match only), **When** the
   user attempts to extract training data, **Then** the system reports a clear error
   explaining that no resolvable outcomes are available.
4. **Given** input data that violates the expected structure (missing required fields,
   wrong types, or duplicate snapshots for one moment), **When** extraction is attempted,
   **Then** the system rejects the data with a validation error naming the offending
   field(s) before any modelling occurs.

---

### User Story 2 - Backtest and predict value bets at the chosen moment (Priority: P2)

Having extracted moment-aware training data, the user backtests a betting strategy against
it and then applies the fitted strategy to upcoming fixtures to identify value bets for the
same target moment. The bettor consumes the `X`/`Y`/`O` tables regardless of whether the
target is pre-match or in-play, and reports value bets and backtest performance.

**Why this priority**: Modelling is the user's actual goal; extraction (P1) only has value
once it feeds a bettor. It depends on P1 but delivers the end-to-end outcome (a testable
betting strategy). It is second because it cannot be demonstrated without P1.

**Independent Test**: With training data extracted for a target moment, fit a bettor,
backtest it, and confirm it returns per-period performance figures; then run it on fixtures
data and confirm it returns a set of value-bet selections aligned to the fixtures.

**Acceptance Scenarios**:

1. **Given** training data extracted for a target moment, **When** the user backtests a
   bettor, **Then** they receive performance results covering the backtest periods.
2. **Given** a fitted bettor and fixtures data for the same target moment, **When** the user
   requests bets, **Then** they receive value-bet selections aligned to the fixtures.
3. **Given** training and fixtures data extracted by the same dataloader, **When** both are
   used, **Then** their feature and odds columns correspond so a model trained on one
   applies to the other without manual reconciliation.

---

### User Story 3 - Select and reuse data through the existing loader interface (Priority: P2)

A returning user relies on the familiar scikit-learn-style selection interface — choosing
data by parameters (league, division, year), discovering available parameters and odds
types, dropping sparse columns — and expects it to keep working, now extended so the same
loader can target a match moment. They can also save a configured dataloader and reload it
later so training and fixtures extraction stay consistent over time.

**Why this priority**: Preserving the established interface protects existing users and
upholds the project's scikit-learn-compatibility principle. It is P2 because the feature is
usable via direct construction even if the convenience selection layer lags.

**Independent Test**: Query a loader for its available parameters and odds types, construct
it with a parameter selection, extract data, save it to a file, reload it, and confirm the
reloaded loader extracts fixtures whose columns match the original training extraction.

**Acceptance Scenarios**:

1. **Given** a soccer dataloader class, **When** the user requests available parameters and
   odds types, **Then** they receive the selectable values without downloading full data.
2. **Given** a configured dataloader, **When** the user saves it and reloads it later,
   **Then** the reloaded loader reproduces the same column structure for fixtures data.
3. **Given** a selection that matches no data, **When** extraction is attempted, **Then** the
   system reports a clear, empty-result condition rather than failing obscurely.

---

### User Story 4 - Access in-play modelling from the CLI and GUI (Priority: P3)

A user who prefers the command line or the graphical app can perform the same
moment-aware extraction, backtesting, and value-bet identification without writing Python,
using the same underlying capabilities exposed through the API.

**Why this priority**: The constitution requires API/CLI/GUI parity, but the surfaces can
follow once the core is proven. P3 because it is a delivery-surface concern layered on the
capabilities established in P1–P3.

**Independent Test**: Through the CLI (and separately the GUI), run a moment-targeted
extraction and a backtest against sample data and confirm the results match the equivalent
API call.

**Acceptance Scenarios**:

1. **Given** the CLI, **When** the user runs a moment-targeted extraction and backtest,
   **Then** the results match the equivalent API workflow.
2. **Given** the GUI, **When** the user configures a target moment and runs a backtest,
   **Then** the app presents the same outcomes and value bets as the API.

---

### Edge Cases

- **No resolvable outcome**: target data contains only pre-match snapshots → clear error
  (see US1 scenario 3).
- **Target moment beyond available data**: user requests an in-play time later than any
  recorded snapshot → system returns no target rows for those matches and reports it,
  rather than silently producing empty results.
- **Mismatched stats/odds coverage**: the moments recorded for statistics and for odds do
  not line up → system detects the mismatch and refuses rather than producing misaligned
  `X`/`O`.
- **Sparse features**: many feature columns are mostly missing → the user can drop columns
  above a chosen missingness threshold, and the same columns are dropped consistently for
  training and fixtures.
- **Negative or zero in-play time**: user supplies an invalid target time → rejected with a
  clear message.
- **Post-match odds absent**: bookmaker odds are naturally missing for settled outcomes →
  accepted as valid (not a data error).
- **Real feed lacks in-play data**: for the current soccer feed (pre/post-match only),
  in-play targets yield no rows; the in-play engine is nonetheless exercised by synthetic
  sample data (see Assumptions).

## Requirements *(mandatory)*

### Functional Requirements

**Data model & validation**

- **FR-001**: The system MUST represent match data as time-ordered snapshots, each labelled
  with an event status (pre-match, in-play, post-match) and an event time within the match.
- **FR-002**: The system MUST keep statistics and betting odds as separate, independently
  validated datasets that share a common snapshot identity (the columns that uniquely
  identify a match moment).
- **FR-003**: The system MUST validate every dataset that crosses a public boundary against
  an explicit schema, and MUST reject invalid data with an error that names the offending
  field(s) before any extraction or modelling occurs.
- **FR-004**: The schema MUST allow each feature to declare which event statuses it applies
  to and whether it is time-invariant (fixed) or varies across snapshots.
- **FR-005**: The system MUST reject datasets containing more than one snapshot for the same
  match moment.

**Moment-aware extraction**

- **FR-006**: Users MUST be able to extract training data for a chosen target moment defined
  by an event status and an optional in-play time.
- **FR-007**: The system MUST return training data as three aligned tables — input features
  (`X`), target outcomes (`Y`), and corresponding odds (`O`) — sharing the same rows. The
  return shape MUST be uniform across extraction modes: unsupervised extraction returns the
  same three-position result with `Y` empty (`None`), so callers can always destructure it as
  `X, Y, O`.
- **FR-008**: The system MUST include in `X` only information available strictly before the
  target moment, and MUST derive `Y` from the match state at the target moment.
- **FR-009**: Users MUST be able to extract fixtures (upcoming-match) data for the same
  target moment, returning input features and odds with no outcomes, whose columns
  correspond to those of the training extraction from the same loader.
- **FR-010**: The system MUST support supervised extraction (features, targets, odds) and
  unsupervised extraction (features and odds only, with `Y` returned as `None`). Data
  extraction MUST always return static tabular results in the uniform three-position shape of
  FR-007.
- **FR-011**: Reinforcement learning MUST NOT be part of the data-extraction method. The
  system MUST document a forward design for an eventual, separate reinforcement-learning
  capability (a stateful environment consumed by an RL agent — observation, action, reward,
  and episode boundaries defined), but MUST NOT implement it in this feature. A request for a
  reinforcement mode through the extraction method MUST be rejected as an invalid option.

**Selection interface & persistence**

- **FR-012**: Users MUST be able to discover the selectable parameters and available odds
  types for a dataloader without downloading full data.
- **FR-013**: Users MUST be able to select which data to load via a parameter grid
  (e.g., league, division, year) consistent with the project's existing selection style.
- **FR-014**: Users MUST be able to drop feature columns whose missingness exceeds a chosen
  threshold, applied consistently to training and fixtures data.
- **FR-015**: Users MUST be able to save a configured dataloader and reload it later such
  that fixtures extraction reproduces the training extraction's column structure.

**Modelling**

- **FR-016**: Bettors MUST accept the extracted `X`/`Y`/`O` tables for any supported target
  moment (pre-match or in-play) without requiring the user to manually reshape them.
- **FR-017**: Users MUST be able to backtest a bettor against moment-aware training data and
  receive performance results across the backtest periods.
- **FR-018**: Users MUST be able to apply a fitted bettor to fixtures data and obtain value-
  bet selections aligned to those fixtures.

**Delivery surfaces & compatibility**

- **FR-019**: The CLI and GUI MUST expose the moment-aware extraction, backtesting, and
  value-bet identification capabilities available through the API, with no capability
  reachable from only one surface.
- **FR-020**: The system MUST provide a lightweight built-in dataloader with representative
  in-play sample data so the full workflow can be demonstrated and tested without network
  access.
- **FR-021**: Public capabilities (dataloaders, schemas, loader persistence, bettors) MUST
  be reachable through the documented public interface.

### Key Entities *(include if feature involves data)*

- **Match snapshot**: a single observation of a match at one moment, carrying its event
  status (pre-match / in-play / post-match), event time, identifying attributes (competition,
  season, teams), and the statistics or odds recorded at that moment.
- **Statistics dataset**: the collection of match snapshots describing on-field information
  and outcomes; the source of features and targets.
- **Odds dataset**: the collection of match snapshots describing bookmaker odds by provider;
  the source of the odds table and the reference for value-bet comparison.
- **Schema**: the declared contract for a dataset — required and optional fields, their
  types, which event statuses each feature applies to, whether it is fixed, and the snapshot
  identity — used to validate data and to drive extraction.
- **Dataloader**: the object that selects, downloads/loads, validates, and reshapes data into
  the `X`/`Y`/`O` tables for a chosen target moment, and that can be persisted and reloaded.
- **Target moment**: the combination of event status and event time that defines what the
  model predicts and the boundary between features and targets.
- **Bettor**: the object that learns from `X`/`Y`, estimates outcome probabilities, compares
  them to odds `O` to identify value bets, and can be backtested.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can extract training data for a specified target moment and confirm that
  no information dated after that moment appears in the features, verified across every
  supported event status.
- **SC-002**: Training and fixtures data produced by the same dataloader have identical
  feature and odds column structures in 100% of cases, so a model trained on one applies to
  the other with no manual reconciliation.
- **SC-003**: Every dataset that fails its schema is rejected with an error identifying the
  offending field, and no invalid dataset reaches the extraction or modelling stage.
- **SC-004**: A user can complete the end-to-end workflow — load sample data, extract a
  target moment, backtest a bettor, and obtain value bets for fixtures — using only the
  built-in sample dataloader and no network access.
- **SC-005**: The same end-to-end workflow is achievable through the API, the CLI, and the
  GUI, producing equivalent results across all three.
- **SC-006**: Every documented public example (including in-play targeting) executes
  successfully as part of the automated test suite, and automated coverage of the data and
  modelling components does not regress relative to the pre-feature baseline.
- **SC-007**: A returning user's existing selection workflow (discover parameters, select by
  parameter grid, drop sparse columns, save and reload a loader) continues to work.

## Assumptions

- **Engine now, data later**: the in-play extraction engine is built and validated against
  representative synthetic sample data. The real soccer feed currently provides only
  pre-match and post-match information, so in-play targets against the live feed may return
  no rows until an in-play data source exists. This is accepted scope, per the product
  decision.
- **Selection interface preserved (design decision)**: the concrete soccer dataloader retains
  the existing scikit-learn-style selection interface (parameter grid, parameter discovery,
  odds-type selection, missingness threshold) layered on top of the new snapshot model. This
  choice favors scikit-learn compatibility and minimal disruption for existing users over a
  clean-break constructor-only API. Chosen because the feature description left the loader UX
  undecided and this best satisfies the project's compatibility principle.
- **Supervised and unsupervised extraction are in scope; reinforcement learning is design-
  only.** RL does not fit the tabular-extraction paradigm (it needs a stateful environment
  and an agent, not a bettor consuming `X`/`Y`/`O`), so it is removed from the extraction
  method and specified as a separate future method (`make_env()`) with a full forward design
  but no implementation in this feature. This resolves the earlier open question on FR-011.
- **Soccer is the only sport in scope** for this feature; additional sports (basketball, NFL,
  hockey) remain future work.
- **The built-in sample dataloader replaces the previously removed dummy loader** and is the
  primary vehicle for offline testing and documentation examples.
- **The upstream data source and its access mechanism from the current codebase are reused**;
  this feature does not introduce a new data provider.
- **Existing bettor strategies are adapted, not redesigned**; their public interface (fit,
  probability estimation, value-bet identification, backtesting) is preserved while their
  input format is updated to the moment-aware `X`/`Y`/`O` tables.
