# Feature Specification: Basketball, starting with the EuroLeague

**Feature Branch**: `003-basketball-euroleague`

**Created**: 2026-07-12

**Status**: Draft

**Input**: User description: "Basketball support, starting with the EuroLeague."

## Context

The library is soccer-only, but almost nothing in it is. `002-data-source-providers` made the data sources injectable, and the extraction engine already derives everything it needs from the data it is given: the providers, the markets, the features, and — as of the last change — which markets are mutually exclusive, so a sport without a draw gets its two-way outcome without being told. The identity of a match (`league`, `division`, `year`, `home_team`, `away_team`) already works unchanged: a dataloader given `league='Euroleague'` extracts cleanly today.

What is missing is not code. It is a **source of basketball statistics**.

That gap is the whole feature. The EuroLeague publishes an official, public, key-less API carrying final scores, schedules and kick-off times — everything the soccer feed gives us. A user's own commercial odds subscription already covers the competition. Put the two together and the library has a second sport.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Get a EuroLeague dataset (Priority: P1)

A user configures a basketball dataloader for the EuroLeague, prepares it, and gets training data and fixtures: features describing each team's form, targets for the markets they can bet on, and the odds those markets were priced at.

**Why this priority**: It is the feature. Everything else here is a consequence of it.

**Independent Test**: For a completed EuroLeague season, the extracted `X`, `Y` and `O` are aligned, non-empty, and reproduce a frozen reference exactly. Shipping only this story gives the library a second sport.

**Acceptance Scenarios**:

1. **Given** a basketball dataloader configured for a completed EuroLeague season, **When** the user prepares and extracts training data, **Then** they receive features, targets and odds sharing the same rows and index.
2. **Given** the same dataloader, **When** the user asks what parameters are available, **Then** they are told which seasons the source publishes — read from the source, never fabricated.
3. **Given** an upcoming EuroLeague game, **When** the user extracts fixtures data, **Then** it carries the same columns as the training data, with the unplayed result absent.
4. **Given** a season the statistics source does not publish, **When** the user selects it, **Then** it is reported as unavailable rather than silently returning nothing.

---

### User Story 2 - The outcome of a game that cannot be drawn (Priority: P1)

The markets a basketball user bets on are not the ones a soccer user bets on. There is no draw — a tie goes to overtime — so the result is two-way.

**Why this priority**: If the markets are wrong, every probability, every value bet and every backtest is wrong. It is the correctness core of a new sport.

**Independent Test**: The derived targets contain a home win and an away win and no draw, and predicted probabilities of the two sum to one.

**Acceptance Scenarios**:

1. **Given** basketball data, **When** targets are derived, **Then** they contain no draw, because the sport has none.
2. **Given** a bettor fitted on basketball data, **When** it predicts probabilities, **Then** a home win and an away win sum to one — which they do not in soccer, where a draw is possible.
3. **Given** a soccer dataset, **When** targets are derived, **Then** the draw is still there: the sport is read from the data, not from a setting.

---

### User Story 3 - Two sources that name the same club differently (Priority: P1)

The statistics call a club `ALBA BERLIN`; the odds vendor calls it something else. The user is told how well the two were reconciled, and if a game ends up without odds, the extraction stops rather than handing back a dataset with holes in it.

**Why this priority**: This is the failure that does not look like one. A game whose odds are missing looks like a slightly smaller dataset and produces a backtest that is clean, plausible and wrong. It is P1 because, unlike soccer, basketball **always** mixes two sources — there is no single feed carrying both.

**Independent Test**: The real rosters of a EuroLeague season, as each source writes them, reconcile without hand-written aliases; and a deliberately unplaceable club raises rather than silently dropping its game.

**Acceptance Scenarios**:

1. **Given** the two sources' rosters for a season, **When** they are reconciled, **Then** the proportion matched and unmatched is reported.
2. **Given** a club neither the pairing nor an alias can place, **When** the user extracts data, **Then** it fails loudly and names the club, rather than emitting a game with no odds.
3. **Given** a club the pairing cannot place but the user supplies an alias for, **When** the user extracts data, **Then** it reconciles.

---

### User Story 4 - Know before you pay (Priority: P2)

Basketball has no free odds. Before spending a credit, the user asks what a season would cost.

**Why this priority**: It is the existing preparation contract, and it must hold for a new sport too. P2 only because it is inherited rather than built.

**Independent Test**: A dry run reports an exact cost for a EuroLeague season and provably spends nothing.

**Acceptance Scenarios**:

1. **Given** a basketball dataloader with a metered odds source, **When** the user runs a dry-run preparation, **Then** they get an exact cost and no credit is spent.
2. **Given** a dataloader with no odds source at all, **When** the user extracts training data, **Then** it fails with a message saying there are no markets to predict — not with an internal schema error.

---

### Edge Cases

- A game is postponed, or scheduled but not yet played: it must appear as a fixture, never as a training row with an invented result.
- A game goes to overtime: the final score is the score after overtime, and the result is still two-way.
- A season is in progress: its played games are training data and its unplayed ones are fixtures, in the same extraction.
- The statistics source lists a game the odds vendor never priced (or vice versa): it is counted as unmatched, not silently dropped.
- A team is promoted, relegated or replaced between seasons: the rosters of the two sources differ per season, so reconciliation is per season and never global.
- The statistics source publishes a season the odds vendor's history does not reach: only the seasons both publish are selectable.

## Requirements *(mandatory)*

### Functional Requirements

#### The statistics source

- **FR-001**: The system MUST provide a free, key-less basketball statistics source over the EuroLeague's official public API, fetched on the user's machine. No basketball data is redistributed by the project.
- **FR-002**: The source MUST report which seasons it publishes, read from the API rather than fabricated from a range of years.
- **FR-003**: The source MUST produce final scores for played games, from which the betting outcomes are derived.
- **FR-004**: The source MUST produce form features for each team — points scored and conceded, averaged over the season so far and over recent games — computed so that a game never sees its own result.
- **FR-005**: The source MUST distinguish a played game from a scheduled one, so an unplayed game becomes a fixture and never a training row.

#### The sport

- **FR-006**: The betting outcomes of a basketball game MUST NOT include a draw. The sport does not have one.
- **FR-007**: The first cut MUST cover the head-to-head market only. A totals market is **out of scope**, because basketball has no standard line: the total points of a game range from roughly 125 to 229, and a bookmaker sets a different line for every game. The library expresses a market as a **column**, and a line that moves per game is not a column. Soccer's `over_2.5` works only because 2.5 *is* the line. Supporting a moving line is a change to the data model and belongs in its own feature.
- **FR-008**: A basketball dataloader MUST use the same extraction contract, the same preparation step and the same store as the soccer one. No new engine, and no sport-specific branch in the engine.

#### Odds

- **FR-009**: The existing commercial odds source MUST cover the basketball competitions the user's subscription covers.
- **FR-010**: The system MUST offer only the seasons **both** the statistics source and the odds source publish, since a season only one of them covers cannot be modelled.

#### Reconciliation

- **FR-011**: Basketball always mixes two sources, so the two rosters MUST be reconciled per season, and the outcome reported as counts and rates.
- **FR-012**: A game that ends up without odds MUST fail loudly above the configured tolerance, and MUST NOT become a hole in the dataset.

#### Failing honestly

- **FR-013**: A dataloader with no odds source has no markets and therefore nothing to predict. It MUST say so plainly, rather than failing with an internal validation error.

#### Time

- **FR-014**: The kick-off instant MUST be in UTC, like every other source. The time zone the EuroLeague API publishes in is **undocumented**, so it MUST be determined by checking a game whose real tip-off is known — never assumed. A wrong time zone silently shifts every price a user asks for by hours.

### Key Entities

- **Game**: A basketball match: when it tipped off, which clubs played, whether it has been played, and its final score.
- **Season**: The unit a competition publishes its games in, and the unit a roster belongs to — clubs change between seasons, so names are reconciled within one.
- **Form**: What a team had done before a game — points scored and conceded, over the season and over recent games. It is what the model learns from, and it must never contain the game's own result.
- **Outcome**: The result of a game expressed as the markets that can be bet on it. For the first cut that is which club won. There is no draw.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can go from a fresh install to an extracted EuroLeague training set with one configuration and one preparation call.
- **SC-002**: The extracted data for a completed season reproduces a frozen reference exactly — same rows, same columns, same values — so a later change that alters it is caught.
- **SC-003**: The targets of a basketball dataset contain no draw, and predicted probabilities of a home win and an away win sum to one — while a soccer dataset still keeps its draw.
- **SC-004**: The two sources' real rosters for a EuroLeague season reconcile with **no hand-written aliases**, or the exceptions are named and justified.
- **SC-005**: A game whose odds are missing never reaches the dataset silently: 100% of such cases are either reported or raised.
- **SC-006**: A dry run reports an exact cost for a season with zero credits spent.
- **SC-007**: The project publishes no basketball data.
- **SC-008**: No test touches the network.
- **SC-009**: The soccer path is unchanged: its extracted data still reproduces its own frozen reference.
- **SC-010**: No new runtime dependency is added.

## Assumptions

- **The user brings their own odds.** There is no free basketball odds source anywhere. A basketball user needs their own commercial subscription. This is accepted rather than worked around: the sources are constructor arguments, and what to use is the user's decision.
- **A market you cannot price is not a market.** The targets are the markets the odds carry, so a dataloader with no odds source has nothing to predict. That is coherent for a betting library, so basketball requires an odds source rather than degrading to a statistics-only dataset.
- **Team scores are enough.** Points scored and conceded give the same shape of form features the soccer feed's goals do. Player-level and box-score detail is not needed for a first dataset.
- **The engine is already general.** The extraction, the store, the preparation, the cost estimate and the reconciliation all came from the previous feature and are not changed here. If any of them needs a sport-specific branch, that is a sign the abstraction is wrong.
- **Rosters change.** Clubs enter and leave a competition between seasons, so reconciliation is done within a season and never across the whole history.

## Out of Scope

- **The NBA.** Its free statistics live behind an unofficial, rate-limited endpoint of a different shape. A separate source, later.
- **Player and box-score data.** Only team scores.
- **In-play basketball.** The model can address any moment of a game and the odds vendor prices them, but the first cut is pre-play and post-play only.
- **The totals market.** Its line moves per game, and the library expresses a market as a column. Supporting a market whose line varies per row is a change to the data model, and it would change soccer too. It deserves its own feature rather than being smuggled in here.
- **Other basketball competitions** the odds vendor covers but for which no free statistics source has been found.
