# Feature Specification: Single-Event Execution Unit

**Feature Branch**: `008-single-event-execution`

**Created**: 2026-07-27

**Status**: Draft

**Input**: User description: "Redesign the execution layer around a single betting event. A single execution unit
should be able to monitor a single betting event preplay, inplay, postplay, log the betting event info to the terminal,
and use a fitted bettor to decide automatically to bet at the right time on the event. A single betting event. It is
the user's responsibility to extend it to multiple events and pay the price if the bookmaker blocks this behaviour."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Set up one event from URLs, then place the model's bet at the right moment (Priority: P1)

A user has a fitted betting model, one upcoming match, and a fixed stake they are willing to put on it. They give the
unit the model, the event, one or more candidate bookmaker URLs, and the stake. The unit then runs a fixed, ordered
sequence: it explores the URLs to find the page and market for that event and selection, prompts the user to log in to
the bookmaker, and then begins logging the event to the terminal. When the event reaches the moment the model was
fitted for, the unit applies the model to the event's data at that moment, and if the model finds a value bet it places
the configured stake on the model's selection, once, and nothing else.

**Why this priority**: This is the whole feature. Without it there is no execution layer worth having.

**Independent Test**: With a fitted preplay model, one upcoming event, a candidate URL that matches the event, and a
stand-in browser, run the unit and confirm the fixed sequence runs in order and it places one bet of the configured
stake at the preplay moment on the model's selection, and places nothing else.

**Acceptance Scenarios**:

1. **Given** the model, event, candidate URLs, and stake, **When** the unit starts, **Then** it explores the URLs,
   matches the event's page and market, and prompts the user to log in before any monitoring or placing.
2. **Given** the user has logged in, **When** setup completes, **Then** the unit begins logging the event to the
   terminal.
3. **Given** a fitted preplay model, **When** the preplay moment arrives and the model finds value, **Then** the unit
   places the configured stake on the model's selection, once, and places no other bet.
4. **Given** a fitted in-play model fitted for a set number of minutes into the match, **When** the unit runs, **Then**
   it waits until kickoff plus that offset and places then, not before.
5. **Given** the model finds no value on the event at its moment, **When** the moment arrives, **Then** the unit places
   nothing and logs that no bet was found.
6. **Given** the unit has placed or declined its one bet, **When** the event continues, **Then** the unit places no
   further bet regardless of later price movement.

---

### User Story 2 - See the event and the decision as it unfolds (Priority: P2)

The user watches the terminal as the unit runs. The browser runs headless, with no live-preview window to watch, so
everything the user needs to see about the event is in the terminal. Monitoring is expressed as this terminal log, fed
by the configured source rather than by scraping the bookmaker's page. The unit logs the event through its lifecycle,
preplay to inplay to postplay: the event's identity, its current status, the current price where it is available, and
the unit's decision at each step, waiting, moment reached, bet placed or declined, and the outcome. What is about to
happen is legible before any money moves.

**Why this priority**: Monitoring and logging are how a user trusts and audits a unit that moves real money on its own.
The log is valuable on its own, even before real stakes are enabled.

**Independent Test**: Run the unit in no-stakes mode against a scripted event and assert the terminal log shows the
status transitions, the values seen, and the decision, and that the unit announces the bet before it would place it.

**Acceptance Scenarios**:

1. **Given** the unit is monitoring, **When** the event's status changes from preplay to inplay to postplay, **Then**
   each transition is logged with the time and the current price.
2. **Given** the betting moment arrives, **When** the unit is about to place, **Then** it logs the selection, stake, and
   price it is about to stake before it places.
3. **Given** the event reaches postplay, **When** monitoring ends, **Then** the unit logs the final status and the
   outcome of its bet if the outcome is known.

---

### User Story 3 - Stay within one event, by design (Priority: P3)

The unit handles exactly one event. It is not a fleet manager. Driving many units across many events at once is the
user's job, and the user accepts that a bookmaker may block or close the account for that behaviour.

**Why this priority**: A guardrail that keeps the library's promise narrow and honest. It adds no new capability, it
bounds the ones above.

**Independent Test**: Confirm the unit's inputs describe exactly one event and one fitted bettor, and that it cannot
place a second bet or span a second event.

**Acceptance Scenarios**:

1. **Given** a unit, **When** it is given its inputs, **Then** it accepts exactly one event and one fitted bettor.
2. **Given** a unit has acted on its event, **When** it is asked to act again within the same run, **Then** it does not
   place a second bet.

---

### Edge Cases

- The betting moment has already passed when the unit starts, because the event is already in-play past the fitted
  moment or already finished. The unit places nothing and logs that the moment is unreachable.
- No candidate URL matches the event during setup. The unit stops before monitoring, logs it, and stakes nothing.
- The user does not complete the login within a bound. The unit stops before monitoring, logs it, and stakes nothing.
- The venue offers no market or price for the model's selection at the moment. The unit places nothing and logs it,
  without failing.
- The price at the moment is worse than the model's minimum acceptable price. The unit declines and logs it.
- The event is postponed or abandoned, so kickoff never arrives or the status never advances. The unit waits up to a
  bound and then stops, logging why.
- Authentication to the venue fails. The unit stops before monitoring, logs the failure, and stakes nothing.
- The live data supply drops mid-monitor. The unit logs the gap and does not place a bet on stale data.
- The unit is run in no-stakes mode. It monitors, decides, and logs the bet it would have placed, but stakes nothing.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST operate on exactly one betting event per execution unit, identified by a single match, and
  MUST NOT place a bet on any other event.
- **FR-002**: The unit MUST take a fitted bettor and use it as the sole decider of whether to bet on the event and which
  selection to back. The stake is not the bettor's to decide (see FR-020).
- **FR-003**: The unit MUST derive the betting moment from the fitted bettor's fitted moment: the preplay moment for a
  preplay model, or kickoff plus the in-play offset for a live model.
- **FR-004**: The unit MUST monitor the event across its lifecycle states preplay, inplay, and postplay, and MUST know
  the current state from the event's data and timing.
- **FR-005**: The unit MUST wait until the betting moment and only then apply the fitted bettor to the event's data as
  of that moment.
- **FR-006**: When the fitted bettor returns a value bet at the moment, the unit MUST place exactly one bet, for that
  selection at the configured stake (FR-020), at the venue.
- **FR-007**: When the fitted bettor returns no value bet at the moment, the unit MUST place nothing and MUST log that
  no bet was found.
- **FR-008**: The unit MUST place at most one bet on the event over its entire run, regardless of later price movement.
- **FR-009**: The unit MUST log to the terminal, as it monitors, the event identity, the current status, the current
  price where it is available, and its decision at each step. Monitoring is realized as this terminal log.
- **FR-010**: Before placing, the unit MUST log the selection, stake, and price it is about to stake, so the action is
  legible before money moves.
- **FR-011**: The unit MUST support a no-stakes mode that performs monitoring, decision, and logging but places nothing.
- **FR-012**: The unit MUST stop and place nothing if the betting moment has already passed when it starts.
- **FR-013**: The unit MUST stop and place nothing if authentication to the venue fails, and MUST log the failure.
- **FR-014**: The unit MUST reuse the existing execution building blocks, the fitted bettor, the event-data supply, the
  venue or browser session, and the receipt shape, rather than a parallel stack, and MUST NOT re-introduce the
  multi-match batch quote-and-confirm model.
- **FR-015**: The unit MUST return a record of its run: the event, the states seen, the decision, and the placed bet's
  receipt if one was placed.
- **FR-016**: The single-event capability MUST be reachable from the Python API, and from the CLI and MCP surfaces where
  the surface parity contract requires an execution capability.
- **FR-017**: The unit MUST NOT orchestrate more than one event. Extending to multiple events is left to the user.
- **FR-018**: A run MUST be a no-stakes dry run unless the user enables stakes for the run up front. Once stakes are
  enabled, the unit MUST place automatically at the betting moment without any further confirmation, and its pre-place
  log line (FR-010) is what makes the action legible before it happens.
- **FR-019**: The unit MUST obtain the event's evolving data, for the decision and for knowing the moment, by polling a
  configured source or dataloader on a schedule, from the start of the run until postplay, so that a single run performs
  the whole watch and the caller does not write a monitoring loop. The bookmaker's page is used to log in and to place,
  not as the source of the decision data.
- **FR-020**: The stake MUST be a configured execution parameter of the unit, and the unit MUST place exactly that
  amount when it bets.
- **FR-021**: The unit MUST take one or more candidate bookmaker URLs and, during setup before the moment, explore them
  to locate the page and market that match the event and the model's selection. If no URL matches, the unit MUST stop
  and log it, and place nothing.
- **FR-022**: The unit MUST ensure the browser session is logged in to the bookmaker before monitoring and placing, and
  MUST prompt the user to log in during setup when the session is not already authenticated.
- **FR-023**: The operating flow MUST be deterministic and fixed in order: explore the URLs to match the event, prompt
  the login, begin the terminal log, then place at the betting moment.
- **FR-024**: The browser MUST run headless, with no live-preview window. It is used to navigate, explore, and place,
  not to present a monitoring view. All monitoring information MUST be surfaced in the terminal log of FR-009, fed by
  the configured source of FR-019, not by scraping the bookmaker's page.

### Key Entities *(include if feature involves data)*

- **Betting event**: The single match under watch. Identified by its teams and kickoff, carrying a lifecycle status of
  preplay, inplay, or postplay, and current prices for markets.
- **Fitted bettor**: The trained model that decides whether to bet and the selection, and that carries the moment it
  was fitted for. The stake is a separate execution parameter, not the bettor's.
- **Betting moment**: The point in the event's timeline when the model is applied, derived from the fitted bettor.
- **Execution unit**: The thing that binds one event, one fitted bettor, and one venue, monitors the event, decides, and
  places at most one bet.
- **Venue or session**: Where the account is held and the bet is placed. For this feature it is a headless browser
  session driving the bookmaker's site at one of the candidate URLs, after the user has logged in.
- **Execution parameters**: The settings the user configures for the run, the candidate bookmaker URLs to explore and
  the fixed stake to place, distinct from what the fitted bettor decides.
- **Run record**: What happened over the run, the states seen, the decision, and the placed bet's receipt if any.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user with a fitted model and one upcoming event can start the unit and get one correctly-timed bet with
  no per-step interaction beyond the one-time setup, logging in and enabling stakes.
- **SC-002**: In 100% of runs, the unit places at most one bet on its event.
- **SC-003**: The bet is placed within a small tolerance of the fitted moment, within seconds for a preplay bet and
  within the fitted in-play offset for a live bet.
- **SC-004**: A user watching the terminal can, before any money moves, see the event, its status, the current price,
  and the exact bet about to be placed.
- **SC-005**: The unit completes a single-event run as one account action for one event, with no batch behaviour a
  bookmaker would recognise as automation across many matches.
- **SC-006**: A user can run the unit with no stakes, observe the full decision, and then re-run with stakes enabled to
  place for real.

## Clarifications

### Session 2026-07-27

- Q: Once stakes are enabled, does the unit place autonomously at the moment, or confirm again at the moment? → A: Arm
  once, then autonomous. A run is a no-stakes dry run by default. Enabling stakes is a single up-front opt-in, after
  which the unit places automatically at the moment with no second confirmation.
- Q: How does the unit get the event's evolving data across the lifecycle? → A: Self-polling. The unit polls a
  configured source or dataloader on a schedule until postplay, so one run performs the whole watch.
- Q: Who decides the stake? → A: The stake is a configured execution parameter, not the bettor's decision. The bettor
  decides whether to bet and the selection.
- Q: How mature is browser-based monitoring, and what is the operating flow? → A: Browser monitoring is not mature, so
  monitoring is just the terminal log. The flow is deterministic: the user provides candidate URLs, the unit explores
  them to match the event, prompts the user to log in, begins logging to the terminal, and places the configured stake
  at the fitted moment.
- Q: Does removing the browser mean no browser at all? → A: No. Remove only the browser live preview, the visible
  window for watching. The browser stays but runs headless, to navigate, explore, and place. All information is logged
  to the terminal.

## Assumptions

- The fitted bettor comes from the evaluation package and carries its fitted moment, preplay or an in-play offset,
  consistent with the existing model.
- The polling cadence, and any backoff when the data supply is slow or drops, is a planning concern, not fixed here.
- The event's data, features, prices, and status, is supplied by the same source or dataloader components the library
  already uses, configured for that one event.
- Status follows the event's timing and status field: before kickoff is preplay, between kickoff and the end is inplay,
  after the end is postplay.
- The venue for this feature is a headless browser session driving the bookmaker's site at one of the candidate URLs,
  with no visible window. An API-venue path is not required here, and placing produces the existing receipt shape.
- Login is interactive at setup and may reuse a persisted browser profile, so the headless session runs already
  authenticated. The unit waits for login to be in place before it proceeds.
- Waiting is real by default but can be driven by an injected clock in tests, as elsewhere in the execution package.
- The model decides the selection, so only the market family relevant to that selection is priced and placed.
- The existing multi-match batch execution path may be removed or reduced once this unit exists. That migration is a
  planning concern, not part of this specification's behaviour.
