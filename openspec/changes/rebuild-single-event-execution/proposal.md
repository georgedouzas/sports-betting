# Proposal

## Why

The single-event execution delivered by change 008 does not work. Its tests pass only because they replace the session,
the placer and the clock with stubs, and the defects sit exactly in those parts:

- It reads the event data once at startup and decides on that stale data at the moment.
- It always backs the home team, whatever market the model chose.
- The default placer never puts the selection on the bet slip.
- It reports `MATCHED_FULL` without reading any confirmation from the site, and never checks the price.
- The pinned controls live only in memory, so `execution page fix` followed by `execution run --live` always fails
  with "Nothing is pinned".
- `authenticate()` navigates away from the event page it had just found.
- Nothing is recorded across processes, so a crash and restart can bet twice.
- It places a preplay bet at kickoff, the moment most bookmakers suspend preplay markets.

Phase 1 of a two-phase plan rebuilds it so that one preplay bet on one event works end to end, by either of two
methods: a bookmaker's website driven with Playwright, or a venue with an API. Phase 2 extends it to many events, and
it starts only once phase 1 has been proven with a real dry run and a real minimum-stake bet.

## What Changes

- **BREAKING** One venue contract for both methods. `BaseVenue` gains `read_price` and keeps `authenticate` and `place`
  as the abstract methods a run needs. `list_markets`, `read_balance`, `read_status` and `cancel` become optional and
  raise a named error by default.
- A new `BaseBrowserVenue` implements `BaseVenue` over a `BrowserSession`. A user subclasses it per bookmaker and fills
  in a small set of hooks. The base drives the hooks in a fixed order, checks the price, and builds the receipt only
  from the confirmation the site shows.
- **BREAKING** `execute_event` takes a `BaseVenue` rather than a `BrowserSession`, and drops the `urls` and `placer`
  parameters. The subclass knows how to find the event and how to place.
- **BREAKING** The in-memory pinning is removed: `BrowserSession.fix`, `BrowserSession.resolve`, `FixedSession`, the
  `execution page fix` command and the `browser_fix` MCP tool. The exploring tools `page read`, `page act` and the
  `browser_*` tools that read and act stay, since they are how a user writes the hooks.
- The run re-reads the event's data from the dataloader at the moment, and on each poll while it waits. It aborts when
  the event has left the fixtures or the read fails.
- The run decides with a pure, public `decide_bet`. It maps each market to its selection: home team, away team,
  `Draw`, or the over/under line. When several markets are backed it takes the first in the bettor's market order.
- The minimum acceptable price is the price the model decided on. The run never falls back to a made-up price, and it
  refuses with `REFUSED_PRICE` when the venue offers less.
- A preplay bet goes on at kickoff minus a lead time, a run parameter with a default of five minutes.
- A dry run reads the venue's real price for the selection without staking, so it proves the whole site path except
  the stake and the confirm.
- A local ledger file records each bet before and after it is placed. A rerun finds the record and does not place the
  bet again.
- A run returns a run record: the steps it went through with their times, the decision, and the receipt.
- **BREAKING** In-play models are refused up front with a clear error. In-play execution returns as its own change.
- The CLI `execution run` and the MCP `execution_run` tool change to match, and the parity tests follow.

## Non-goals

- More than one event per run. This is phase 2.
- In-play betting.
- Shipping a concrete API venue for a real exchange. The library ships the contract and proves it against a reference
  venue in the tests. The user writes the venue for their exchange or bookmaker.
- Evading a website's automation controls.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `execution/event-run`: Running one preplay event from start to finish: when the bet goes on, how fresh data reaches
  the decision, which selection is backed at what minimum price, dry run and live, placing at most once across
  processes, the terminal log, the run record, and how the CLI and MCP reach it. Finding the event by page text is
  removed.
- `execution/venue-contract`: What a venue must do for a run to use it, the API method and the browser method behind
  that one contract, and the honesty rules for a receipt. Pinning controls in memory is removed.
- `interfaces/cli`: `execution run` takes the new parameters and `execution page fix` is removed.
- `interfaces/mcp-server`: `execution_run` takes the new parameters and returns the run record, and `browser_fix` is
  removed.

## Impact

- Code: `src/sportsbet/execution/` (`_base.py`, `_browser.py`, `_event.py`, `_schedule.py`, `_factory.py`,
  `__init__.py`, and a new module for the ledger), `src/sportsbet/cli/_execution.py`, and
  `src/sportsbet/mcp/_server.py`.
- Public API: the removed and changed names above. `decide_bet`, `BaseBrowserVenue` and the run record type are added.
- Tests: `tests/execution/` is rewritten around a reference API venue and a fake bookmaker page served locally. The
  parity tests in `tests/cli/test_parity.py` and `tests/mcp/test_parity_surfaces.py` are updated.
- Docs: `docs/overview/user_guide/execution.md` and the CLI and agents interface pages. The changelog records the
  breaking changes.
- Dependencies: none new. Playwright stays behind the `execution` extra.
