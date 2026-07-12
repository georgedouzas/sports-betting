# Feature Specification: Pluggable Data-Source Providers

**Feature Branch**: `002-data-source-providers`

**Created**: 2026-07-12

**Status**: Draft

**Input**: User description: "Pluggable data-source providers with a local store and an explicit preparation step."

## Context

Today the library ships **data**, not just code. A Prefect flow on the `data` branch downloads football-data.co.uk, transforms it, and republishes the result — including the bookmaker odds — as CSVs that `SoccerDataLoader` fetches from GitHub. Nobody granted us the right to republish those odds: football-data.co.uk is "free to download" but reserves all rights, and every commercial odds vendor forbids redistribution outright. Every comparable open-source project in this space (kloppy, statsbombpy, soccerdata, ScraperFC, socceraction) ships the fetching code and lets the user pull the data themselves. We must do the same.

The same change unlocks the capability the library currently only pretends to have. `001-in-play-betting` gave us in-play *features* and in-play *targets*, but the only odds we can obtain from the current source are pre-match closing prices. So an in-play bet cannot be backtested: we do not know what price was actually available at minute 45. Time-stamped odds snapshots only exist behind a commercial API and a user's own key — which is exactly the shape this feature enables.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Get the same data as today, without the mirror (Priority: P1)

A user installs the library, constructs a soccer dataloader for a few leagues and seasons, runs a preparation step, and gets training data and fixtures. Under the hood the data is now downloaded from the original public source onto their own machine and transformed locally; no pre-built dataset is served by the project. To the user, the extracted `X`, `Y` and `O` are the same as before.

**Why this priority**: This is the legal fix and the migration's foundation. Until the free path works client-side, nothing else can be built and the redistribution exposure remains open. It is also the only story that must be *behaviour-preserving*, which makes it the safety net every later story is verified against.

**Independent Test**: For a fixed `param_grid` captured from today's mirrored data, the locally-produced training data, targets and odds match the frozen reference to the row and column. Shipping only this story already removes the redistribution problem while leaving users no worse off.

**Acceptance Scenarios**:

1. **Given** a dataloader configured with the free stats and odds sources and an empty local store, **When** the user runs the preparation step and extracts training data, **Then** they receive the same `X`, `Y`, `O` as the reference capture from the current mirrored dataset.
2. **Given** a `param_grid` selecting two leagues, **When** preparation runs, **Then** only the source files those leagues and seasons need are downloaded — not the full catalogue.
3. **Given** the local store is already populated, **When** the user re-runs preparation, **Then** already-held immutable data is not re-downloaded.
4. **Given** the project's published artifacts, **When** anyone inspects them, **Then** they contain no third-party odds data.

---

### User Story 2 - Bring your own odds source (Priority: P1)

A user with a commercial odds subscription supplies their own key and the markets they care about. The dataloader combines free statistics with that user's odds. Nothing about the user's key or the data it buys ever leaves their machine.

**Why this priority**: This is the whole point of the architecture — sources are complementary layers, not alternatives. "Free stats + paid odds" is the realistic configuration, and a single mode flag cannot express it. It is also what makes in-play backtesting real rather than notional, because the commercial source is the only one with time-stamped prices.

**Independent Test**: With a stats source and an odds source from two different providers, extraction yields odds columns attributed to the bookmakers of the user's chosen source while the statistics columns still come from the free source. Testable against a recorded interaction without spending credits.

**Acceptance Scenarios**:

1. **Given** a dataloader constructed with a free stats source and a keyed odds source, **When** the user extracts training data, **Then** the odds columns come from the keyed source and the feature columns from the free source.
2. **Given** the user selects specific markets, **When** they extract training data, **Then** only those markets appear as targets and odds.
3. **Given** an odds source with time-stamped in-play prices, **When** the user extracts training data with an in-play target moment, **Then** the odds correspond to the price available at that moment, not the pre-match closing price.
4. **Given** an invalid or exhausted key, **When** preparation runs, **Then** it fails with a message naming the source and the reason, and the local store is left unchanged.

---

### User Story 3 - Know the cost before paying it (Priority: P1)

Before spending anything, a user asks the dataloader what preparing their `param_grid` would involve: how much has to be fetched, how much of it is already held, and — for a metered source — roughly how many credits it would consume. Only when they choose to proceed does any fetching happen.

**Why this priority**: A metered source turns an accidental call into money. Any design where a data request can silently trigger a backfill is unacceptable, so the explicit, estimable preparation step is a first-class requirement, not a convenience.

**Independent Test**: A dry-run preparation reports counts and an estimated cost and provably performs no fetch. Extraction against an unprepared store refuses to run and explains what is missing.

**Acceptance Scenarios**:

1. **Given** an unprepared store, **When** the user runs preparation in dry-run mode, **Then** they get the number of items to fetch, the number already held, and an estimated cost for metered sources — and nothing is fetched or spent.
2. **Given** an unprepared store, **When** the user calls training-data extraction directly, **Then** it fails immediately, names what is missing, and points at the preparation step. It does not fetch anything.
3. **Given** a preparation run interrupted midway, **When** the user runs it again, **Then** it resumes and does not re-fetch or re-pay for what already completed.
4. **Given** a fully prepared store, **When** the user changes only the feature engineering and re-extracts, **Then** no fetching or spending occurs.

---

### User Story 4 - Trustworthy joins across sources (Priority: P2)

When statistics and odds come from different providers, the same match is named differently in each ("Man United" vs "Manchester United", different league keys, kickoff times in different timezones). The user is told, in numbers, how well the two sources were reconciled — and if too much failed to match, the extraction stops rather than handing back a dataset full of holes.

**Why this priority**: This is the most dangerous failure mode in the feature. A failed join does not look like an error; it looks like missing odds, which looks like a slightly smaller dataset, which produces a backtest that is confidently wrong. Silence here is worse than a crash. It is P2 only because it cannot occur until two sources are actually mixed (Story 2).

**Independent Test**: With deliberately mismatched naming between two sources, the reported unmatched rate is accurate and extraction fails when it exceeds the configured tolerance.

**Acceptance Scenarios**:

1. **Given** stats and odds from different providers, **When** the user prepares data, **Then** the proportion of matches reconciled and the proportion unmatched is reported.
2. **Given** an unmatched rate above the configured tolerance, **When** extraction is attempted, **Then** it fails and names example unmatched matches, rather than emitting missing odds.
3. **Given** an unmatched rate within tolerance, **When** extraction succeeds, **Then** the unmatched matches are still reported, never silently dropped without a count.

---

### User Story 5 - Existing offline and documented workflows keep working (Priority: P2)

Users of the dummy dataloader, the snapshot factory functions, and every example in the documentation see no change. Their code does not need to be edited and does not touch the network or the store.

**Why this priority**: The migration is only safe if the escape hatches survive it. These entry points are what the test suite and the docs are built on; breaking them means we cannot verify the rest.

**Independent Test**: The existing offline test suite and the documented examples pass unmodified against the new architecture.

**Acceptance Scenarios**:

1. **Given** a user constructing a dataloader from their own in-memory snapshots or dataframe, **When** they extract training data, **Then** it works with no store, no source and no preparation step.
2. **Given** the dummy dataloader, **When** it is used, **Then** it performs no network access and requires no preparation.

---

### Edge Cases

- A source file exists upstream but is empty or truncated (a season that has not started): preparation must record it as fetched-and-empty rather than re-fetching it forever or poisoning the extracted column types.
- A season is in progress: its data changes week to week and must be refreshed, while finished seasons must never be re-fetched.
- The upstream source changes its column layout or adds a bookmaker: extraction must not silently drop the new column or crash on the missing old one.
- A user's `param_grid` selects a league/season combination the source does not publish: this must be reported as an unavailable selection, not a network error.
- A metered source's historical coverage starts later than the requested seasons: the user must be told which part of their request is unobtainable *before* they pay for the part that is.
- Preparation is interrupted (network drop, process kill) mid-fetch: the store must never be left in a state where a partially written item is read back as complete.
- Two processes prepare the same store concurrently: the store must not be corrupted.
- A user changes the requested markets after having already prepared: previously paid-for raw data must be reused where it covers the new request.

## Requirements *(mandatory)*

### Functional Requirements

#### Sources

- **FR-001**: The soccer dataloader MUST remain a single public class, configured by *injecting* a statistics source and an odds source, rather than by selecting a mode.
- **FR-002**: Each source MUST own its own configuration (credentials, markets, regions). Source-specific parameters MUST NOT appear on the dataloader.
- **FR-003**: A statistics source and an odds source MUST be independently selectable, so that a free statistics source can be combined with a paid odds source.
- **FR-004**: The system MUST provide free, key-less statistics and odds sources that fetch the public upstream data on the user's machine and transform it locally, reproducing the dataset the project currently mirrors.
- **FR-005**: The free sources MUST fetch only the source files needed by the configured `param_grid`.
- **FR-006**: The system MUST provide a keyed commercial odds source supporting time-stamped historical snapshots, upcoming fixtures and in-play prices, with user-selected markets.
- **FR-007**: Adding a new source MUST NOT require changing the dataloader, the extraction engine, or any other source.
- **FR-038**: An odds source whose prices are addressed by instant rather than by season MUST be given the schedule of the selected matches, since a season alone does not say when its matches are played. The dataloader MUST read the statistics first and derive the schedule from them, so the odds plan is built from the real kick-offs rather than from a guess.
- **FR-039**: A credential MUST be added to a request at the moment the request is made. It MUST NOT be part of a stored item, and MUST NOT be written to the store.
- **FR-037**: `date` MUST be the kick-off instant of the match, in UTC. Every source MUST convert its own local representation into that at its own boundary; no source may emit a naive or a local instant. This makes `date + event_time` the wall-clock instant of a snapshot, which is the address a time-stamped odds source needs in order to be asked for the price at a given minute. Where a source has no kick-off time, `date` is midnight UTC of the match day, and such rows cannot be joined to a time-stamped odds source.
- **FR-008**: The data-source contract of the existing abstract dataloader (returning long-format statistics and odds snapshots) MUST NOT change; this feature is an implementation behind that existing seam.

#### Discovery

- **FR-031**: A user MUST be able to ask a source directly what parameters it publishes, **without constructing a dataloader**. Writing a `param_grid` requires knowing what exists, so discovery cannot depend on an object that is configured with a `param_grid`.
- **FR-032**: What a source publishes MAY depend on that source's own configuration (for example, a credential's subscription tier), so discovery MUST be answered per configured source and never as a property of a source's class.
- **FR-033**: The dataloader MUST NOT expose a public discovery method. It needs the available parameters only to filter `param_grid` internally, which is not a user-facing concern.
- **FR-034**: When the statistics and the odds come from different sources, the dataloader MUST select only the parameters **both** sources publish. A season whose statistics exist but whose odds do not cannot be modelled, and MUST NOT be silently selected.

#### Preparation

- **FR-009**: The system MUST expose an explicit preparation step that populates the local store for the configured `param_grid`.
- **FR-010**: Preparation MUST be incremental: it MUST fetch only what the store does not already hold.
- **FR-011**: Preparation MUST be resumable: an interrupted run MUST be able to continue without re-fetching or re-paying for completed work.
- **FR-012**: Preparation MUST support a dry run that reports what would be fetched, what is already held, and — for metered sources — an estimated cost, **without spending anything**. It MAY read free data in order to build the plan (a source's catalogue, and the statistics that say when the matches are played), because that is what makes the estimate exact rather than a guess. It MUST NOT fetch any priced item.
- **FR-013**: Data extraction MUST NEVER trigger fetching. If the store is not prepared for the requested `param_grid`, extraction MUST fail immediately, state precisely what is missing, and direct the user to the preparation step.
- **FR-014**: Preparation MUST report progress for long-running fetches.

#### Store

- **FR-015**: The system MUST persist fetched data locally, at a user-configurable location, with a sensible default.
- **FR-016**: Raw fetched payloads MUST be retained indefinitely and never discarded, because metered data cannot be re-obtained without paying again.
- **FR-017**: Derived tables MUST be rebuildable from the retained raw payloads with no fetching and no cost, so that changing the transformation or feature engineering never requires re-paying.
- **FR-018**: The store MUST distinguish immutable data (finished matches, completed seasons, historical snapshots) from volatile data (fixtures, the in-progress season) and MUST refresh only the latter.
- **FR-019**: A partially written item MUST never be read back as complete.
- **FR-020**: The store MUST preserve column types across a write/read round-trip; an empty result MUST NOT degrade the types of the data it is combined with.
- **FR-035**: Derived data MUST be invalidated when the code that derives it changes, not only when the raw data changes. An upgrade that changes the transformation MUST NOT serve the output of the previous one.
- **FR-036**: The system MUST provide a way to re-read data the store considers finished. Nothing upstream is truly immutable — a source can correct a completed season — so "finished" means "not re-read unless asked", never "cannot have changed". Because a metered source charges again for a re-read, it MUST be requested rather than done on a schedule.

#### Reconciliation

- **FR-021**: When statistics and odds come from different sources, the system MUST reconcile them into a single match identity using team names, competition and kickoff time.
- **FR-022**: The system MUST report the reconciliation outcome as counts and rates (matched, unmatched on each side).
- **FR-023**: The system MUST fail loudly when the unmatched rate exceeds a configurable tolerance, and MUST NOT degrade into silently missing odds.
- **FR-024**: Failed reconciliations MUST be inspectable by the user, with enough detail to diagnose the naming mismatch.

#### Legal posture

- **FR-025**: The project MUST NOT publish, mirror or distribute third-party odds data in any form, including as repository content, release artifacts, or documentation fixtures beyond the minimum needed for tests.
- **FR-026**: The redistributing `data` branch MUST be retired and the redistributed odds removed from it.
- **FR-027**: User credentials MUST NOT be transmitted anywhere except to the source they authenticate, and MUST NOT be written to the store or to logs.

#### Compatibility

- **FR-028**: The dummy dataloader and the in-memory snapshot factory functions MUST continue to work unchanged, requiring no store, no source and no preparation.
- **FR-029**: All existing documentation examples MUST continue to work, with updates limited to those that construct a downloading dataloader.
- **FR-030**: Before behaviour changes, a reference capture of the current statistics, odds and extracted `X`/`Y`/`O` for a fixed `param_grid` MUST be frozen and used as an equivalence gate for the client-side free path.

### Key Entities

- **Source**: A named origin of data with its own configuration and cost model. Knows how to enumerate what it can supply for a `param_grid`, how to fetch an item, and how to turn a raw payload into long-format snapshots. Is either a statistics source or an odds source.
- **Raw payload**: An exact, unmodified copy of what a source returned, plus what was asked for and when. Immutable, retained forever, and the sole input to every derived table.
- **Store**: The local persistence layer. Holds raw payloads, derived snapshot tables, and an index recording what is held, from which source, and whether it is immutable or volatile.
- **Preparation plan**: The set-difference between what the configured `param_grid` requires and what the store holds — the unit of both cost estimation and incremental fetching.
- **Match identity**: The reconciled key (competition, kickoff, home team, away team) that lets a statistics row and an odds row from different sources refer to the same event.
- **Reconciliation report**: The counts, rates and examples describing how well two sources were joined; the input to the data-quality gate.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The project distributes zero rows of third-party odds data. Verified by inspecting every published artifact and branch.
- **SC-002**: For the reference `param_grid`, the client-side free path reproduces the currently-mirrored training data, targets and odds exactly — same rows, same columns, same values.
- **SC-003**: A user can go from a fresh install to extracted training data for a single league-season with one configuration step and one preparation call.
- **SC-004**: No data request can cause a fetch or a charge: 100% of extraction calls against an unprepared store fail without network access. Verified by a test that fails if any network call is attempted.
- **SC-005**: A dry run reports a cost estimate for a metered source with zero credits consumed.
- **SC-006**: Re-running preparation on an unchanged, fully prepared store fetches nothing.
- **SC-007**: Rebuilding derived tables after a transformation change fetches nothing and costs nothing.
- **SC-008**: When two sources are mixed, the reconciliation rate is always reported, and an unmatched rate above tolerance always stops extraction — never yields a dataset with silently missing odds.
- **SC-009**: In-play targets can be backtested against the odds actually available at that moment, which is impossible today.
- **SC-010**: Preparation fetches only the leagues and seasons the user selected, not the full upstream catalogue.
- **SC-011**: The existing offline test suite and documentation examples pass with no changes beyond dataloader construction.
- **SC-012**: Exactly one new runtime dependency is added.

## Assumptions

- **User-supplied credentials.** Commercial odds require the user's own subscription and key. The project ships the client, never the data or a shared key.
- **Laptop-scale data.** The realistic working set is gigabytes, and the access pattern is "read a `param_grid`-scoped slice into a dataframe" plus "set-difference over tens of thousands of keys". A columnar file format on local disk is therefore sufficient; an embedded analytical engine, an embedded SQL database or a remote database is explicitly out of scope. A shared remote store of vendor odds would also recreate the redistribution problem this feature exists to remove.
- **One new dependency.** Storage adds a single runtime dependency (a columnar file format library), justified because it preserves column types across round-trips — eliminating a class of bug already encountered with delimited text — and is several times smaller on disk.
- **The transformation already exists.** The statistics/odds transformation logic lives in the `data` branch's flow and is being relocated into the library, not rewritten. Equivalence to its current output is the gate.
- **Historical coverage is bounded by the vendor.** The commercial source's snapshot history does not extend indefinitely into the past, and historical access is a paid tier. Requests outside coverage must be reported, not silently truncated.
- **Reconciliation is only needed when mixing sources.** A single-source configuration (free stats + free odds) needs no entity resolution, since both come from the same upstream row.
- **Tolerances are user-configurable with a strict default.** The unmatched-rate gate defaults to strict; the user may loosen it deliberately, never accidentally.
- **Recording is not laundering.** Odds we fetch ourselves are still the vendor's data. No fetching strategy makes redistribution permissible.

## Out of Scope

- **Basketball and other sports.** The design must not preclude them, but generalizing the two remaining soccer-specific spots (the complementary-events market definition in the bettor, and the identity fields on the dataloader) is a separate feature.
- **Additional statistics sources** beyond the free one being relocated.
- **A hosted or shared store.** See the legal posture above.
- **Live in-play streaming** (continuously polling and betting in real time). This feature makes in-play prices *available and backtestable*; automating live execution is separate.
