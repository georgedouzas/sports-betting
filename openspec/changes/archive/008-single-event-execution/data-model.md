# Phase 1 Data Model: Single-Event Execution Unit

The feature adds no new persisted entity. It composes existing types and adds one run configuration and one run
record. Fields are described by meaning, not by implementation type.

## Entities

### Event (input, identified, not owned)

The single match under watch.

- **match**: the identifier, `"Home vs Away"`, the sole event the unit acts on.
- **kickoff**: when the match starts, the fixtures index timestamp, the anchor for the moment and the status.
- **features**: the event's current feature row, from the polled dataloader, fed to the bettor.
- **odds**: the event's current odds row, from the polled dataloader, fed to the bettor and read for the price.
- **status**: one of preplay, inplay, postplay, derived per D2.

### Fitted bettor (input, from evaluation)

- Decides whether to bet and the selection, through `bet(X_event, O_event)`.
- Carries the fitted moment through its dataloader's `target_event_status_` and `target_event_time_`.
- Does not decide the stake.

### Execution parameters (input, user-configured)

- **stake**: the fixed amount to place, the only stake the unit uses.
- **urls**: the candidate bookmaker URLs to explore and match against the event.
- **live**: whether the run is armed. False is a no-stakes dry run.
- **poll**: the interval between source polls while monitoring.
- **bound**: how long to wait for an event that never advances before giving up.

### Session (input, headless browser)

- A headless `BrowserSession` for the bookmaker, logged in, with no visible window.
- Pinned to the event's page and controls as a `FixedSession` after matching a URL.

### Placer (input, user-supplied)

- `placer(intent, session)` drives the pinned controls to enter the stake and confirm, and returns a receipt.
- The library never guesses site controls, so the placer is the user's site knowledge.

### Betting moment (derived)

- The point in the event's timeline when the bettor is applied, from `find_betting_moment`.

### Run record (output)

- The receipts frame from `build_receipts_frame`, holding the one placement if a bet was placed, or empty.
- Accompanied by the terminal log, which is the human-facing account of the run.

## State transitions

The event's status advances one way through the run:

```text
preplay ── kickoff ──▶ inplay ── final whistle ──▶ postplay
```

The unit's own progress over a run:

```text
setup ──▶ monitoring ──▶ moment ──▶ decided ──▶ done
  │           │             │           │
  │           │             │           └─ placed one bet, or declined, logged either way
  │           │             └─ bettor applied to the event's data as of the moment
  │           └─ poll, log status and price, wait, repeat until the moment or the bound
  └─ explore URLs, match the event, pin controls, ensure login; stop here on no match or login failure
```

## Validation rules

- Exactly one event and one fitted bettor per unit (FR-001, US3).
- At most one placement over the run, regardless of later price moves (FR-008).
- No placement when the moment has already passed at start (FR-012), when authentication or login fails (FR-013,
  FR-022), when no URL matches (FR-021), or when the run is not armed (FR-018).
- The placed stake equals the configured stake (FR-020).
- Placement receipts validate against the existing `PlacementReceiptSchema`.
