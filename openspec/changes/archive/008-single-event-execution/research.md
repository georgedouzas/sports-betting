# Phase 0 Research: Single-Event Execution Unit

All decisions below were settled by reading the existing `execution`, `dataloaders`, and `evaluation` packages. The
spec's Clarifications section already fixed the arm-once gate, the self-polling source, the stake-as-parameter split,
and the headless-browser-no-live-preview stance. This document records the design decisions those imply.

## D1: How the single event and its evolving data are supplied

**Decision**: The unit takes a fitted bettor's dataloader and an event identifier (the match, `"Home vs Away"`). It
self-polls the dataloader's `extract_fixtures_data` on a cadence and picks the single row whose match equals the
identifier. Each poll yields that event's current features and odds.

**Rationale**: `extract_fixtures_data` already returns the upcoming matches with their features and odds, keyed by
kickoff, and the sources already carry per-status snapshots (the in-play feature, 001). Re-extracting is how the unit
sees the event evolve. Reusing it avoids a parallel data path (FR-014, FR-019).

**Alternatives considered**: a bespoke live-stream source (heavier, no existing support); reading prices from the
bookmaker page (rejected, browser monitoring is not mature, FR-024).

## D2: How the current status is known

**Decision**: Status is derived from the event's timing against the clock, cross-checked with the source's
`event_status`. Before kickoff is preplay, from kickoff to the end is inplay, after is postplay. Kickoff is the
fixture's index timestamp.

**Rationale**: The dataloader already exposes kickoff as the fixtures index, and snapshots carry `event_status`. The
`core` status vocabulary (`PREPLAY_EVENT_STATUSES`, `NON_PREPLAY_EVENT_STATUSES`, `STATUS_RANK`) names the states.

## D3: The betting moment

**Decision**: Reuse `find_betting_moment(dataloader, kickoff)`. A preplay model bets at kickoff, a live model at
kickoff plus the fitted in-play offset (`dataloader.target_event_time_`). This is the single point the unit waits for.

**Rationale**: The logic already exists and is tested. Keep it, move it to the module that owns the runner if
`_schedule.py` is retired.

## D4: The decision and the stake

**Decision**: At the moment, the unit applies the fitted bettor to the event's one-row feature and odds frames via
`bettor.bet(X_event, O_event)`. A non-empty value bet names the selection to back. The stake is the configured
execution parameter, placed as-is (FR-002, FR-020). The bettor decides whether and what, not how much.

**Rationale**: `bet` is the estimator surface for value bets. Splitting the stake out keeps the estimator contract
untouched and matches the user's decision.

## D5: How the bet is placed on a headless browser

**Decision**: Keep the existing shape where the library never guesses how to place on an arbitrary site. The unit
takes a headless `BrowserSession`, explores the candidate URLs by navigating each and matching the one whose page
mentions the event and selection, pins the site's controls with `FixedSession` via user-named locators, and places by
calling a user-supplied `placer(intent, session)` that drives the pinned controls to enter the stake and confirm. The
placer returns a `PlacementReceipt`.

**Rationale**: `BrowserSession` has no `place` by design (its doctest asserts `hasattr(session, 'place') is False`),
and the batch `execute` already delegates browser placement to a `placer`. The site-specific knowledge stays the
user's, the library owns navigation, matching, timing, and logging. `FixedSession.fix` already refuses to pin a price
or a volatile ref, which fits placing at the moment.

**Alternatives considered**: generic auto-discovery of the stake box and confirm button (rejected, not reliable across
bookmakers and out of scope); an API venue (`BaseVenue`) path (kept possible but not required, the feature targets the
browser venue).

## D6: The arm gate and legibility

**Decision**: A run is a no-stakes dry run unless `live=True` is passed for the run. Armed, the unit places
automatically at the moment with no second confirmation. Just before placing, it logs the selection, stake, and price
it is about to stake (FR-010). Dry run does everything except call the placer, logging the bet it would have made.

**Rationale**: Resolves the automatic-versus-legible tension the spec called out. The up-front arm plus the pre-place
log line make the action legible without a synchronous prompt that would defeat "at the right time".

## D7: Terminal logging

**Decision**: Log through the standard `logging` module on the `sportsbet.execution` logger. The CLI attaches a rich
handler for the run, reusing the existing `_logging_to_terminal` context manager. The log carries, per step, the
event, its status, the current price where available, and the decision.

**Rationale**: The batch `execute` already logs this way and the CLI already renders it. No new logging stack.

## D8: Polling cadence and waiting

**Decision**: The runner takes an injected `clock` and `wait`, defaulting to real time and `asyncio.sleep`, exactly as
`execute` does. It polls the source every `poll` interval (a `Timedelta`, coarse by default) and sleeps until the next
poll or the moment, whichever is first. A bound stops a run whose event never advances (postponed or abandoned).

**Rationale**: Injected time is how the package keeps timed behaviour testable without waiting in tests.

## D9: What is retired and what is kept

**Decision**: Remove the multi-match batch functions `execute`, `select_feasible`, `quote`, `place`, and
`build_value_bet_intents`, and the batch CLI `run` and MCP `execution_run`/`execution_quote`/`execution_place`. Keep
and reuse `BrowserSession`, `FixedSession`, `PageSnapshot`, `BetIdentity`, `PlacementIntent`, `PlacementReceipt`,
`PlacementStatus`, `ExposureLimits`, `build_receipts_frame`, `find_betting_moment`, `build_venue`, the credential
helpers, and the `page` explore and fix commands and `browser_*` tools.

**Rationale**: The spec deems the batch path not useful and forbids re-introducing it (FR-014). The primitives it was
built from are sound and the single-event runner stands on them.

## D10: Surface parity

**Decision**: One Python entry point `execute_event`, one CLI command under the `execution` group, and one MCP tool,
named consistently and asserted by the parity test. The event-preview `venue`, `markets`, `balance`, `status`,
`cancel`, and the `page` explore and fix commands stay where they still apply.

**Rationale**: Principle VI requires the three surfaces to stay in step, and a parity test already watches them.

## D11: Testing approach

**Decision**: Unit-test the runner with an injected clock and wait, a fake source scripted to return preplay, inplay,
and postplay snapshots for the one event, a stub session, and a placer that records the intent it was handed. Assert
one bet at the moment, none when the model finds no value, dry run stakes nothing, and the moment-already-passed and
no-URL-match edges stop cleanly. No network, no real browser.

**Rationale**: Mirrors how the batch path was tested and keeps the gate offline.
