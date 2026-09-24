# Feature Specification: The NBA, as a second basketball league

**Feature Branch**: `004-nba-espn-source`

**Created**: 2026-07-12

**Status**: Draft

**Input**: User description: "implement also an NBA data source"

## Summary

The library plays basketball, but it only knows one competition. A user who wants the NBA — by far the most priced,
most modelled basketball league in the world — cannot have it, because there is no source that carries its games.

Everything else is already there. The odds vendor already prices the NBA. The dataloader is already sport-agnostic. The
market is already two-way, the reconciliation already pairs rosters, the store already caches. What is missing is a
single thing: **a source that says what the games were**.

So this feature adds one source and changes nothing else. That is the test of the architecture the previous feature
built: if adding the world's biggest basketball league requires touching the engine, the engine is wrong.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Get an NBA dataset (Priority: P1)

A user who wants to model the NBA names it, points the dataloader at the NBA statistics, brings their own odds key, and
gets training data and fixtures back — exactly as they would for the EuroLeague, or for the English Premier League.

**Why this priority**: This is the feature. Everything else here is a way of making sure this one is not quietly wrong.

**Independent Test**: Construct a basketball dataloader whose statistics are the NBA's, prepare it, and extract. `X`,
`Y` and `O` come back aligned, with a row per game and a column per market.

**Acceptance Scenarios**:

1. **Given** a basketball dataloader with the NBA statistics and an odds source, **When** the user prepares and extracts
   training data, **Then** they receive the games of the selected seasons with their outcomes and their prices.
2. **Given** the same dataloader, **When** the user extracts fixtures, **Then** they receive the games that have not
   been played, with the form of the teams going into them and no outcome.
3. **Given** a user choosing a season, **When** they ask what is available, **Then** they are offered only the seasons
   that both the statistics and the odds actually publish.

---

### User Story 2 - The dataset contains basketball games, and only basketball games (Priority: P1)

The competition's calendar is not only its competition. It contains a pre-season nobody bets seriously, and it contains
an exhibition weekend in which the teams are not teams: they are all-star selections with invented names.

A user must never find those in their data. Not as training rows, not as fixtures, and above all not as teams.

**Why this priority**: This is the difference between a dataset and a corrupted dataset, and it fails **silently**. An
exhibition team pollutes the roster, and a polluted roster breaks the pairing with the odds vendor for the whole
competition. A model trained on an exhibition learns from a game nobody was trying to win.

**Independent Test**: Transform a season and assert the teams are exactly the league's real clubs, and that no
exhibition game survives.

**Acceptance Scenarios**:

1. **Given** a season containing an all-star exhibition, **When** the statistics are transformed, **Then** no exhibition
   game and no invented team appears in the result.
2. **Given** a season containing a pre-season, **When** the statistics are transformed, **Then** no pre-season game
   appears in the result.
3. **Given** a season containing a post-season, **When** the statistics are transformed, **Then** the post-season games
   **do** appear — they are real games between real clubs and the vendor prices them.
4. **Given** any complete season, **When** the statistics are transformed, **Then** the number of distinct teams is
   exactly the number of clubs in the league.

---

### User Story 3 - The dataset is complete (Priority: P1)

Every game of the selected season is present. Not most of them.

**Why this priority**: The feed hands back a truncated answer to a question that is too big, and it does not say that it
did. A dataset quietly missing a quarter of its games produces a backtest that is quietly wrong, and nothing anywhere
reports an error. Silent incompleteness is the worst failure this library can have, because it looks exactly like
success.

**Independent Test**: Ask the feed for a season and count. The count must be the league's real number of games, and the
way the request is split must be provably incapable of reaching the feed's limit.

**Acceptance Scenarios**:

1. **Given** a season, **When** its games are requested, **Then** every game of that season is returned.
2. **Given** the feed's limit on how many games it returns at once, **When** the requests are planned, **Then** each
   individual request is provably far below that limit.

---

### User Story 4 - The NBA is bettable today, not eventually (Priority: P1)

A user modelling the NBA in January can see the games played in December, and bet on the games in February.

**Why this priority**: A betting library that only knows seasons which finished long ago is a backtesting toy. The form
of a team is what it has done *this* season; if this season's results are absent, every current feature is empty and
there is nothing to predict from. This requirement is what disqualifies the otherwise-perfect official archive, which is
why it is written down as a requirement rather than left as an assumption.

**Independent Test**: Read the current, in-progress season and confirm the games already played carry their final score.

**Acceptance Scenarios**:

1. **Given** a season in progress, **When** the statistics are read, **Then** the games already played carry their
   results and the games still to come do not.

---

### User Story 5 - The NBA's teams find their prices (Priority: P2)

The statistics call a club one thing and the odds vendor calls it another. They must still be recognised as the same
club, or a game gets no price and disappears from the dataset.

**Why this priority**: Basketball always mixes two independent sources — there is no single feed carrying both games and
odds — so reconciliation is the normal path, not an edge case. It is P2 only because the existing resolver is expected
to handle it unaided; the requirement is to *prove* that, not to build it.

**Independent Test**: Pair the league's real roster against the vendor's real roster and confirm every club is placed.

**Acceptance Scenarios**:

1. **Given** the two sources' names for the same league, **When** they are reconciled, **Then** every club is paired.
2. **Given** a club that genuinely cannot be placed, **When** reconciliation runs, **Then** it fails loudly and names
   the club, rather than dropping its games.

---

### Edge Cases

- **A game postponed and never replayed.** A finished season still contains games that were never played. They must be
  treated as unplayed — a fixture, never a training row with a result nobody knows.
- **A season the feed lists but the odds vendor does not cover.** The vendor's history is far shorter than the league's.
  Only seasons both publish may be selected; the rest are never offered.
- **A user with no odds source.** The NBA has no free odds anywhere. The dataloader must say so plainly rather than
  produce an empty dataset or a schema error.
- **A season with no games played yet.** A schedule published before the season starts has fixtures and no results. It
  must yield fixtures and no training rows, not an error.
- **The feed changes how it labels an exhibition.** The filter must be written so that an unrecognised label is excluded
  rather than admitted: admitting an exhibition is silent corruption, while excluding a real game is a visible gap.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The library MUST provide a statistics source for the NBA that is free and requires no credential.
- **FR-002**: The NBA MUST be usable through the existing basketball dataloader, by naming it as that dataloader's
  statistics source. No new dataloader, and no new sport.
- **FR-003**: The source MUST declare the seasons it publishes by reading them from the feed. It MUST NOT offer a season
  taken from a fabricated range of years.
- **FR-004**: A season MUST be identified by the year it ends in, consistently with every other source in the library.
- **FR-005**: The source MUST produce a pre-play snapshot for every game, carrying the form each team brought into it.
- **FR-006**: The source MUST produce a post-play snapshot for every game that has been played, carrying the final score
  and the outcome. A game that has not been played MUST NOT get one.
- **FR-007**: The outcome MUST be two-way — a home win and an away win. There is no draw, because a tie goes to
  overtime. There is no totals market, because the line moves from game to game, and a market whose line moves is not a
  column.
- **FR-008**: The form features MUST be computed only from games that came before, so a game can never see its own
  result, and a fixture carries the form of the games played before it.
- **FR-009**: The source MUST exclude the pre-season, and MUST exclude all-star exhibitions, whose teams are not clubs.
- **FR-010**: The source MUST include the post-season and the play-in, which are real games between real clubs.
- **FR-011**: The exclusion in FR-009 MUST NOT rely on the feed's own season labelling alone, which files the all-star
  exhibition under the regular season.
- **FR-012**: The source MUST return every game of a selected season. The feed silently truncates an over-large request,
  so each request MUST be provably smaller than the feed's limit.
- **FR-013**: The kick-off MUST be the instant the game starts, in UTC, as the library's convention requires.
- **FR-014**: The source MUST carry results for a season in progress, so a user can model the current season.
- **FR-015**: The source MUST NOT fetch. Declaring what it needs, describing what is available, and transforming what
  was fetched MUST all be free of network access; only preparation downloads.
- **FR-016**: The library MUST reconcile the NBA's team names against the odds vendor's without silently dropping a
  club.
- **FR-017**: The NBA MUST be reachable from every delivery surface — the Python API, the command line and the GUI — as
  the EuroLeague is.
- **FR-018**: The library MUST ship no NBA data. It is fetched on the user's machine.

### Key Entities

- **NBA statistics source**: A free, key-less source of the games of the NBA. It declares which seasons exist, what must
  be fetched for a chosen season, and how the fetched games become snapshots.
- **Game**: A contest between two clubs at a known instant, which either has been played — and has a score — or has not,
  and is a fixture.
- **Season**: A competition year, named by the year it ends in, containing a regular season, a play-in and a
  post-season — and also a pre-season and an exhibition, which are not part of the dataset.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user obtains an NBA training set and NBA fixtures with the same three steps, and the same objects, they
  already use for soccer and the EuroLeague.
- **SC-002**: A transformed season contains **exactly 30 teams** — the league's clubs and nothing else. No exhibition
  selection appears under any circumstances.
- **SC-003**: A transformed season contains **every** game of that season's regular season, play-in and post-season, and
  none of its pre-season.
- **SC-004**: The current, in-progress season yields results for the games already played.
- **SC-005**: Every one of the league's 30 clubs is paired with the odds vendor's name for it, with as few hand-written
  aliases as the data allows, and each alias that exists has a stated reason.
- **SC-006**: The feature adds **no new runtime dependency**.
- **SC-007**: The feature changes **no** existing source, **no** dataloader, **no** part of the fetch layer and **no**
  part of the preparation engine. Every existing test passes unmodified. *(This is the real acceptance criterion of the
  architecture: a new league must be a new file.)*
- **SC-008**: No test touches the network.
- **SC-009**: The soccer and EuroLeague datasets are unchanged.
- **SC-010**: No NBA data is published in the repository.

## Assumptions

- **The user brings their own odds.** There is no free basketball odds feed anywhere, so an NBA user needs a paid key
  from the odds vendor, exactly as a EuroLeague user does. This is accepted rather than solved: sources are constructor
  arguments, and the user decides what to use.
- **The odds vendor already covers the NBA.** It is already mapped, so this feature adds nothing on the odds side.
- **The statistics feed is not the league's own.** The league's official archive was evaluated and rejected: it
  back-fills results only once a year, months after a season ends, which cannot satisfy FR-014. The chosen feed is
  unofficial but live, long-stable and widely used. This is a deliberate trade of provenance for currency, and the
  evidence is recorded in the research.
- **The existing reconciliation is sufficient.** The roster-bijection resolver is expected to pair all 30 clubs unaided.
  If it cannot, aliases are added only where a club genuinely cannot be placed, because a wrong alias attaches one
  club's prices to another club's game and says nothing about it — worse than not matching at all.
- **Team-level scores are enough.** Player and box-score features are out of scope, as they were for the EuroLeague.

## Out of Scope

- The WNBA, college basketball and the Australian NBL. The odds vendor already prices them; each needs its own
  statistics source, later.
- Player-level and box-score features.
- In-play snapshots. The feed carries period scores and the data model addresses any moment, but the first cut is
  pre-play and post-play, as it is for the EuroLeague.
- A frozen equivalence fingerprint for the NBA. The feed is live by design, so a fingerprint of a season in progress
  would drift. It is worth freezing for a completed season only — the same reason it was deferred for the EuroLeague.
