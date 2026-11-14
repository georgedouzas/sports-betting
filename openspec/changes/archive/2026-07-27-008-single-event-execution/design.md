# Design

## Context

Feature 006 added a batch execution path: value bets became intents, a quote itemised the batch, and placement went
ahead only against the exact quoted stake and exposure. Commits after 006 added a batch `execute` that placed value
bets one match at a time, with `select_feasible`, a batch `execution run` command and a batch `execution_run` tool.
This change replaces the whole multi-match path with one execution unit that watches exactly one betting event, logs
everything to the terminal, and lets a fitted bettor decide and place one bet at the moment the model was fitted for.

The unit self-polls a dataloader for the event's evolving data across preplay, inplay and postplay. It drives a
headless browser to explore the candidate URLs, match the event and place. It stakes a configured amount only after the
run is armed. It reuses `BrowserSession`, `FixedSession`, `BetIdentity`, `PlacementReceipt`, `build_receipts_frame` and
`find_betting_moment`.

Technical context from the plan: Python `>=3.11, <3.14`. No new dependency. `playwright` stays in the optional
`execution` extra, with headless Chromium. `rich` and the standard `logging` module carry the terminal log. Storage is
none, apart from a browser user-data directory that persists the bookmaker login between runs. Tests inject a clock and
a wait, feed a fake source scripted through preplay, inplay and postplay, and drive a stub session and placer, with no
network and no real browser. The target is placing within a few seconds of the fitted moment, with a coarse default
poll so a long preplay wait is cheap.

## Goals / Non-Goals

**Goals:**

- One unit binds one event, one fitted bettor and one venue, and places at most one bet.
- The run is legible in the terminal before any money moves.
- The run is a dry run unless the user arms it up front.
- One run performs the whole watch, so the caller writes no monitoring loop.
- Reach the unit from the Python API, the CLI and the MCP server.

**Non-Goals:**

- Orchestrating more than one event. The user runs and parallelises units and accepts the bookmaker's response.
- A browser live-preview window or monitoring by scraping the bookmaker's page.
- Generic discovery of a site's stake box and confirm button.
- An API venue (`BaseVenue`) path for the unit. It stays possible but is not required.
- Any return of the multi-match batch quote-and-confirm model.

## Decisions

### Clarifications from the spec session of 2026-07-27

- Clarified: once stakes are enabled, does the unit place autonomously at the moment, or confirm again? -> Arm once,
  then autonomous. A run is a no-stakes dry run by default. Enabling stakes is a single up-front opt-in, after which the
  unit places automatically at the moment with no second confirmation.
- Clarified: how does the unit get the event's evolving data across the lifecycle? -> Self-polling. The unit polls a
  configured source or dataloader on a schedule until postplay, so one run performs the whole watch.
- Clarified: who decides the stake? -> The stake is a configured execution parameter, not the bettor's decision. The
  bettor decides whether to bet and the selection.
- Clarified: how mature is browser-based monitoring, and what is the operating flow? -> Browser monitoring is not
  mature, so monitoring is just the terminal log. The flow is deterministic: the user provides candidate URLs, the unit
  explores them to match the event, prompts the user to log in, begins logging to the terminal, and places the
  configured stake at the fitted moment.
- Clarified: does removing the browser mean no browser at all? -> No. Remove only the browser live preview. The browser
  stays but runs headless, to navigate, explore and place. All information is logged to the terminal.

### D1: How the event and its evolving data are supplied

The unit takes a fitted bettor's dataloader and an event identifier, the match as `"Home vs Away"`. It self-polls the
dataloader's `extract_fixtures_data` on a cadence and picks the single row whose match equals the identifier. Each poll
yields the event's current features and odds. `extract_fixtures_data` already returns upcoming matches keyed by kickoff,
and the sources carry per-status snapshots since feature 001, so re-extracting is how the unit sees the event evolve.

Alternatives considered: a bespoke live-stream source, rejected as heavier with no existing support. Reading prices
from the bookmaker page, rejected because browser monitoring is not mature.

### D2: How the current status is known

Status derives from the event's timing against the clock, cross-checked with the source's `event_status`. Before
kickoff is preplay, from kickoff to the end is inplay, and after is postplay. Kickoff is the fixture's index timestamp.
The core status vocabulary (`PREPLAY_EVENT_STATUSES`, `NON_PREPLAY_EVENT_STATUSES`, `STATUS_RANK`) names the states.

### D3: The betting moment

The unit reuses `find_betting_moment(dataloader, kickoff)`. A preplay model bets at kickoff, and a live model at
kickoff plus the fitted in-play offset `dataloader.target_event_time_`. The logic already exists and is tested, so it
stays and moves to live with the runner if `_schedule.py` is retired.

### D4: The decision and the stake

At the moment, the unit applies `bettor.bet(X_event, O_event)` to the event's one-row features and odds. A non-empty
value bet names the selection. The stake is the configured parameter, placed as is. Splitting the stake out keeps the
estimator contract untouched.

### D5: Placing on a headless browser

The library never guesses how to place on an arbitrary site. The unit takes a headless `BrowserSession`, explores the
candidate URLs by navigating each and matching the one whose page names the event and selection, pins the site's
controls with `FixedSession` through user-named locators, and places by calling `placer(intent, session)`, which
returns a `PlacementReceipt`. `BrowserSession` has no `place` by design, and `FixedSession.fix` already refuses to pin a
price or a volatile ref. The library owns navigation, matching, timing and logging. The site knowledge stays the user's.
A default placer drives the pinned `stake` and `confirm` controls, so the common case needs no user code.

Alternatives considered: generic discovery of the stake box and confirm button, rejected as unreliable across
bookmakers. An API venue path, kept possible but not required.

### D6: The arm gate and legibility

A run is a no-stakes dry run unless `live=True` is passed. Armed, the unit places at the moment with no second
confirmation. Just before placing, it logs the selection, stake and price. A dry run does everything except call the
placer and logs the bet it would have made. The up-front arm plus the pre-place log line make the action legible
without a synchronous prompt that would defeat placing at the right time.

### D7: Terminal logging

The unit logs through the standard `logging` module on the `sportsbet.execution` logger. The CLI attaches a rich
handler for the run through the existing `_logging_to_terminal` context manager. Each step carries the event, its
status, the current price where available, and the decision. The batch `execute` already logged this way.

### D8: Polling cadence and waiting

The runner takes an injected `clock` and `wait`, defaulting to real time and `asyncio.sleep`. It polls the source every
`poll` interval, a `Timedelta` that is coarse by default, and sleeps until the next poll or the moment, whichever is
first. A `bound` stops a run whose event never advances, such as a postponed or abandoned match.

### D9: What is retired and what is kept

Removed: `execute`, `select_feasible`, `quote`, `place`, `build_value_bet_intents`, the batch CLI `run`, `quote` and
`place`, and the MCP `execution_run`, `execution_quote` and `execution_place` in their batch form. Kept and reused:
`BrowserSession`, `FixedSession`, `PageSnapshot`, `BetIdentity`, `PlacementIntent`, `PlacementReceipt`,
`PlacementStatus`, `ExposureLimits`, `build_receipts_frame`, `find_betting_moment`, `build_venue`, the credential
helpers, the `page` explore and fix commands and the `browser_*` tools. The batch path is deemed not useful and FR-014
forbids re-introducing it. The primitives it was built from are sound.

### D10: Surface parity

One Python entry point `execute_event`, one CLI command `execution run`, and one MCP tool `execution_run`, asserted by
the parity test. The `venue`, `markets`, `balance`, `status`, `cancel` and `page` commands stay. A custom `placer` is
Python code, so a bespoke-site placement is a Python API capability. The CLI and the MCP tool place through the default
placer over the pinned controls, and otherwise monitor, decide and dry run.

### D11: Testing approach

The runner is unit tested with an injected clock and wait, a fake source scripted with preplay, inplay and postplay
snapshots for the one event, a stub session, and a placer that records the intent it was handed. Tests assert one bet
at the moment, none without value, a dry run that stakes nothing, and clean stops when the moment has passed or no URL
matches. The runner touches the network and the browser, so it carries no executable doctest, matching
`BrowserSession`.

### D12: The Python API contract

`execute_event(event, bettor, dataloader, session, *, stake, urls, live=False, placer=None,
poll=pd.Timedelta("30s"), bound=None, clock=None, wait=None)` returns a receipts frame validated by
`PlacementReceiptSchema`, with one row if a bet was placed or recorded as a dry run and no rows otherwise.

1. Explore `urls`, navigating each headless and matching the one whose page names the event and selection, then pin
   the controls as a `FixedSession`. If none match, log and return an empty frame.
2. Ensure the session is logged in, prompting the user during setup when it is not. If login is not in place within
   the bound, log and return an empty frame.
3. Poll the dataloader on `poll`, logging the event, its status, the current price and the decision at each step,
   until the moment from `find_betting_moment`.
4. At the moment, apply `bettor.bet` to the one-row features and odds. Without a value bet, log and return an empty
   frame.
5. With a value bet, log the selection, stake and price about to be staked. When `live`, call `placer` once and record
   the receipt. When not `live`, record a dry-run receipt.
6. Place at most one bet over the whole run.

It raises `ExecutionError` when `event` is not among the fixtures or the session cannot be driven, and
`VenueBlockedError` when the site blocks automated access. `Placer` is
`Callable[[PlacementIntent, BrowserSession], Awaitable[PlacementReceipt]]`, unchanged in shape from the retired
`_schedule` alias and moved to live with the runner.

### D13: The data model

No new persisted entity. The unit composes existing types with one run configuration and one run record.

- Event: `match`, `kickoff` (the fixtures index timestamp), the current `features` and `odds` rows from the polled
  dataloader, and a derived `status`.
- Fitted bettor: decides through `bet(X_event, O_event)` and carries the fitted moment through its dataloader's
  `target_event_status_` and `target_event_time_`. It does not decide the stake.
- Execution parameters: `stake`, `urls`, `live`, `poll` and `bound`.
- Session: a headless, logged-in `BrowserSession`, pinned to the event's page and controls as a `FixedSession`.
- Placer: `placer(intent, session)`, the user's site knowledge when the default does not fit.
- Betting moment: derived by `find_betting_moment`.
- Run record: the receipts frame from `build_receipts_frame`, with the terminal log as the human account of the run.

The event's status advances one way, preplay to inplay at kickoff, then to postplay at the final whistle. The unit
moves through setup, monitoring, moment, decided and done. Setup explores the URLs, matches the event, pins the
controls and ensures login, and stops on no match or a login failure. Monitoring polls, logs status and price, waits,
and repeats until the moment or the bound. At the moment the bettor is applied. Decided places one bet or declines,
and logs either way.

Validation rules: one event and one fitted bettor per unit. At most one placement per run. No placement when the moment
has passed at start, when authentication or login fails, when no URL matches, or when the run is not armed. The placed
stake equals the configured stake. Receipts validate against `PlacementReceiptSchema`.

### D14: Retirement ledger

| Symbol | Where | Replaced by |
| --- | --- | --- |
| `execute` | `execution._schedule` | `execute_event` |
| `select_feasible` | `execution._schedule` | folded into the single-event monitor |
| `quote` | `execution._place` | none, the quote-and-confirm model is gone |
| `place` | `execution._place` | `execute_event` places directly at the moment |
| `build_value_bet_intents` | `execution._place` | the runner builds the one intent it needs |
| `execution run` (batch) | `cli._execution` | `execution run` (single event) |
| `execution quote`, `execution place` | `cli._execution` | none |
| `execution_run` (batch) | `mcp._server` | `execution_run` (single event) |
| `execution_quote`, `execution_place` | `mcp._server` | none |

`ExposureLimits` stays importable, but the single configured stake is the primary control. The parity test loses the
`(["execution", "quote"], "execution_quote")` and `(["execution", "place"], "execution_place")` pairs and keeps
`(["execution", "run"], "execution_run")` with the new parameter set.

### Constitution check

Checked against constitution v2.2.0 before and after design. Principle I passes because the bettor is used only
through `bet(X, O)`. Principle II passes through the pandera-validated receipt frame and the existing source and
dataloader schemas. Principle III passes with unit tests on an injected clock and no executable doctest on the runner.
Principle IV passes with the gate green on 3.11, 3.12 and 3.13 at every commit. Principle V passes with the example
and the user guide rewritten and `docs/generated` regenerated. Principle VI passes because the unit handles one event,
no agent loop enters the package, the three surfaces stay in step, and errors raise the `ExecutionError` family. The
removal of the batch path is a behaviour change to a recent surface, recorded in the contracts and the changelog.

## Risks / Trade-offs

- Removing the batch path is a breaking change to the public API, the CLI and the MCP server. The changelog lists the
  removed names under a `BREAKING CHANGE` footer.
- An armed run places with no second confirmation. The pre-place log line is the only warning before money moves.
- The stake and exposure ceilings and the kill switch no longer guard a run. The configured stake bounds a run to one
  bet of a known amount.
- Driving many units across many events is the user's choice, and a bookmaker can block or close the account for it.
- A dropped data supply leaves the unit without fresh data. The unit logs the gap and does not bet on stale data.

## Migration Plan

- Replace calls to `quote` and `place`, or to the batch `execute`, with one `execute_event` call per event.
- Replace `sportsbet execution quote` and `sportsbet execution place` with `sportsbet execution run --event ... --stake
  ... --url ...`, adding `--live` to place for real.
- Replace the MCP tools `execution_quote` and `execution_place` with `execution_run`.
- Pin the site's `stake` and `confirm` controls once with `sportsbet execution page fix` so the default placer can use
  them.
- Rewrite `docs/examples/execution/plot_place_value_bets.py` and `docs/overview/user_guide/execution.md`, regenerate
  `docs/generated`, and delete the tests of the removed functions.

## Validation scenarios

From the quickstart. Prerequisites: the `execution` extra with `python -m playwright install chromium`, a saved
dataloader whose source carries in-play snapshots, a model saved by `sportsbet evaluation fit`, and a bookmaker profile
logged in once. The gate is `pdm run formatting`, `pdm run checks` and `pdm run tests` on 3.11, 3.12 and 3.13, read
from the nox session summary line.

1. Dry run with a preplay model. `sportsbet execution page fix --venue venue.py:VENUE --url "$URL" --match "Arsenal vs
   Chelsea" --locator stake='textbox[name="Stake"]' --locator confirm='button[name="Place bet"]'`, then
   `sportsbet execution run --venue venue.py:VENUE -d loader.pkl -b model.pkl --event "Arsenal vs Chelsea" --stake 10
   --url "$URL"`. The terminal logs the match, the status, the price and the bet it would place. Nothing is staked.
2. An armed run with `--live` places one bet at the moment through the pinned controls and no second bet afterward.
3. No value found. Nothing is placed and the log says no bet was found.
4. Moment already passed. The unit logs that the moment is unreachable and places nothing.
5. No URL matches, or no login. The unit stops during setup, logs the reason and stakes nothing.
6. The Python API with a custom placer invokes the placer once at the moment when armed.

`tests/execution/test_event.py` reproduces scenarios 1 to 5 offline. The parity test asserts that `execution run` and
`execution_run` stay in step and that the removed quote and place surfaces are gone.
