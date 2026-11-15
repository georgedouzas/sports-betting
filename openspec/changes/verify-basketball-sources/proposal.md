# Proposal

## Why

Three verification tasks of the basketball changes were left open when those changes were archived, because the data
to run them did not exist at the time.

- Change 003, task 5.3: freeze a EuroLeague sample beside the soccer one. It was deferred because a season in progress
  drifts on every prepare. The 2025-26 EuroLeague season is now complete.
- Change 004, task 2.3: verify end to end that `X`, `Y` and `O` line up for the NBA. The odds for a season cost 17,722
  credits and the free key held 498.
- Change 004, task 3.1: pair the `OddsApi` NBA roster with the ESPN roster. The NBA was out of season, the vendor listed
  zero events, and the historical endpoint is paid only. The 2026-27 NBA season is now under way.

The soccer fingerprint that 003 referred to was later removed from the tests. The offline soccer sample under
`tests/sources/samples/` took its place, so the EuroLeague check follows that pattern instead.

This change carries those tasks, so the archived changes can be closed honestly.

## What Changes

- A recorded sample of games from the completed 2025-26 EuroLeague season is added to `tests/sources/samples/`, with an
  offline test that transforms it.
- The NBA roster pairing is run against the live `OddsApi` event list, and any alias it needs is added or zero aliases
  is confirmed.
- The NBA end to end alignment is run, with a paid key or on a window of the current season small enough for the free
  key.

## Non-goals

- Changing the behaviour of any source, beyond what a check finds broken. The NBA checks found two ESPN defects that
  stopped them, a refused date range and placeholder teams, and those two are fixed here so the checks can run. Any
  other defect gets its own change.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. These are checks of behaviour the specs already state.

## Impact

- Tests: a new EuroLeague sample and test under `tests/sources/`.
- Code: the NBA alias `Los Angeles Clippers=LA Clippers` in `sportsbet.core`, and the two ESPN fixes in
  `sportsbet.sources`.
- Cost: the NBA checks spend `OddsApi` credits. Each run is quoted with `prepare(dry_run=True)` first.
