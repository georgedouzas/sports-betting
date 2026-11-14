# Design

## Context

See proposal.md, Why, for the defects. The requirements are in `specs/execution/event-run/spec.md` and
`specs/execution/venue-contract/spec.md`. The state that shapes the approach:

- `execute_event` in `src/sportsbet/execution/_event.py` is one coroutine that explores URLs, authenticates, reads the
  data once, waits, decides and places. It takes a `BrowserSession`, so the API venue (`BaseVenue`) cannot be run at
  all, and `build_venue` returns either type.
- `BaseVenue` in `_base.py` has six abstract methods. A run needs three of them.
- `BrowserSession` in `_browser.py` is a sound Playwright transport: navigate, snapshot, act by aria ref, blocked
  status detection. Its `fix` and `resolve` keep the pinned locators in memory only.
- `DataLoader.extract_fixtures_data` downloads the upcoming matches each time it is called and keeps a match only
  while its kickoff is in the future. For a preplay moment before kickoff, calling it again at the moment returns fresh
  data, so no dataloader change is needed.
- The constitution's Library rule keeps "an interactive loop, a long-lived session" with the caller. Errors must be one
  of the six named exceptions in `core/_errors.py` unless the profile is amended. Tests must not reach the network but
  may use a locally served page.

## Goals / Non-Goals

**Goals:**

- One driver works with an API venue and a browser venue without telling them apart.
- Each piece a run is made of can be tested without the others: the decision, the venue, the ledger and the driver.
- Every receipt status traces to something the venue said, or to a check the library made before staking.

**Non-Goals:**

- A recipe or configuration format for websites. A user writes Python per bookmaker.
- Retrying a failed placement. A failure is recorded and the run ends.
- Any change to the dataloaders, sources or bettors.

## Decisions

### D1. Narrow `BaseVenue` to what a run needs

`BaseVenue` keeps `authenticate` and `place` as abstract methods and gains an abstract `read_price(identity) -> float |
None`. `list_markets`, `read_balance` and `read_status` become concrete methods that raise `ExecutionError` saying the
venue does not support them. `cancel` raises `CancellationUnsupportedError`, as it already does for `can_cancel =
False`. A new concrete `close()` releases resources and does nothing by default.

- Why not keep six abstract methods: a browser venue for a retail bookmaker often cannot list markets or read status
  reliably, and forcing stubs teaches users to fake answers, which is what broke the receipts.
- Why `ExecutionError` and not a new `VenueUnsupportedError`: the profile names exactly six exceptions. A new one
  would need a constitution amendment for a case the message already explains. It can be added later without a
  behavior change for anyone catching `ExecutionError`.

### D2. `BaseBrowserVenue` owns the placing order, the subclass owns the site

`BaseBrowserVenue(BaseVenue)` takes `key` and `session: BrowserSession` and stores them unmodified. Its concrete
`authenticate`, `read_price`, `place` and `close` call abstract async hooks the subclass writes:

| Hook | Returns |
| --- | --- |
| `check_login()` | whether the open site shows a logged-in session |
| `find_event(match)` | the event page URL, or `None` |
| `add_selection(identity)` | nothing, it clicks the selection onto the slip |
| `read_slip()` | the slip's selection label and price, or `None` |
| `enter_stake(stake)` | nothing |
| `confirm_bet()` | nothing |
| `read_confirmation()` | a `Confirmation` (status, venue bet id, price, detail), or `None` on timeout |

`place` runs: find the event, navigate, add the selection, read the slip, compare the label with the intended
selection, check the price against the minimum, enter the stake, confirm, read the confirmation. Each early exit
returns an honest receipt (`rejected`, `refused_price`, `refused_unconfirmed`) before any stake is entered.
`read_price` runs the same steps up to reading the slip. The label comparison is a concrete `match_selection(identity,
label)` method, which by default compares case-insensitively with whitespace collapsed. A subclass overrides it for a
site that translates team names.

Hooks reach the page through a new derived property `BrowserSession.page_`, which returns the Playwright page, so a
subclass writes ordinary Playwright locators. A hook that is not written makes the subclass abstract, so instantiating
it raises Python's `TypeError` naming the missing method. That covers the spec's "names the missing step".

- Alternative considered: a saved recipe of locators. The user chose hooks, since retail sites need logic (two-step
  confirms, overlays, translated names) that a locator file cannot hold.
- Alternative considered: keep the `placer` callable. A subclass is the same extension point with a contract the
  type checker can see, so the callable goes.

### D3. Thin driver over pure steps

The run is built from public pieces, each usable alone:

- `find_betting_moment(dataloader, kickoff, lead)` returns kickoff minus lead, and raises `ExecutionError` for an
  in-play dataloader.
- `decide_bet(event, bettor, X, O, *, venue_key, stake)` is pure. It returns a `PlacementIntent` or `None` and maps
  the market to its selection with a private helper. An unknown market name, one that is neither an outcome nor an
  over or under line, gives `None` and a reason.
- `venue.read_price` and `venue.place` are one-shot.

`execute_event(event, bettor, dataloader, venue, *, stake, ledger_path, lead, poll, live, clock, wait)` strings them
together: refuse in-play, read the event, compute the moment, check the ledger, authenticate and close, then loop
`wait(min(poll, remaining))` with a fresh read and a log line per poll. At the moment it reads again, decides, opens
the venue, reads the price, places or records the dry run, and closes in a `finally`. The clock and the wait stay
injectable, as today.

This keeps the loop as thin as possible, and the browser is open for seconds rather than hours, so it fits the
Library rule without an amendment. A caller who wants their own loop calls the pieces.

Defaults: `lead = 5min` and `poll = 5min`. Each poll downloads the fixtures again, so a short poll would hit the
source too hard for no gain in preplay.

### D4. Reading the event and failing safely

A private `_read_event(dataloader, event)` wraps `extract_fixtures_data` and the existing `Home vs Away` filter. It
returns the one-row `X`, `O` and the kickoff, or raises `ExecutionError` when the event is not listed. The driver
catches `ExecutionError`, `aiohttp.ClientError` and `OSError` around each read and turns them into an
`event_unlisted` or `data_failed` outcome. Anything else propagates, so a bug is not hidden as a "no bet".

### D5. Ledger as an append-only JSON Lines file

A private `_Ledger` in a new `_ledger.py` appends one JSON object per line: `ref`, `status`, `written_at`, and the
receipt fields when there is a receipt. It flushes and fsyncs each write. Before placing it writes `status:
attempted`, and after placing it writes the receipt. On reading, the lines are validated against a pandera schema,
since the file crosses a process boundary.

The latest line per `ref` decides a rerun:

- `accepted`, `matched_full`, `matched_partial`: the run records `already_placed` and places nothing.
- `attempted` or `refused_unconfirmed`: the run records `refused_unconfirmed` and tells the user to check the venue.
- Anything else, including `dry_run`, `refused_price` and `rejected`: the run proceeds.

The ref is `BetIdentity.ref_`. The identity includes the market and selection, so the driver checks the ledger after
the decision rather than at the start. A rerun that backs a different market on the same event still places once per
identity. Phase 1 accepts that, since one run binds one event and the user starts runs by hand.

- Alternative considered: CSV. JSON Lines appends safely without rewriting a header, and a torn last line is
  detectable and skipped with a log line.

### D6. Run record

`EventRunRecord` is a frozen dataclass holding `event`, `moment`, `outcome` (a `RunOutcome` StrEnum), `steps` (a
validated frame of `time`, `step`, `detail`) and `receipts` (the existing validated receipts frame, zero or one row).
The `RunOutcome` values are `placed`, `dry_run`, `no_value`, `refused`, `already_placed`, `moment_passed`,
`venue_failed`, `event_unlisted` and `data_failed`. Each step is also logged to `sportsbet.execution` as it happens,
so the terminal view and the record agree.

The command prints the steps and the receipt and writes `run_steps.csv` and `receipts.csv` under
`<output>/sports-betting-data/`. The tool returns `{outcome, moment, steps, receipts}` as records.

### D7. Surfaces

- `sportsbet execution run` takes `--venue --dataloader --bettor --event --stake --lead --poll --ledger --live
  --output`. `--url` goes.
- `execution page read` and `page act` accept a `BrowserSession` or a `BaseBrowserVenue`, using its `session`. `page
  fix` goes.
- MCP `execution_run` mirrors the command. `browser_fix` goes. `_open_session` accepts either type, as above.
- `venue`, `markets`, `balance`, `status` and `cancel` accept any `BaseVenue`, browser venues included, and surface
  the unsupported error from D1 through the existing error reporting.
- The parity tests assert the new parameter sets and that the removed command and tool are gone.

### D8. Testing

- A reference API venue in `tests/execution/conftest.py` holds an in-memory book of prices and records placements. It
  can be told to drift a price, reject over a limit, or never confirm. It is the "contract + test reference" the user
  chose, and the docs show the same shape.
- A test browser venue subclass drives a fake bookmaker served locally from `tests/execution/` through a pytest
  fixture running `http.server` on a free port. Its pages cover a logged-in marker, an event page, selection buttons,
  a slip, a stake box, a confirm button, and a confirmation that can be configured to appear, refuse or never appear.
- One contract test module runs the same scenarios against both venues, which proves the spec's "same run, two
  venues" requirement.
- The driver tests use the reference venue, the fake clock and the existing `SnapshotsDataLoader`, with a dataloader
  stand-in whose odds change between reads for the fresh-data scenarios.
- Chromium comes from the existing `playwright install --only-shell chromium` step in the nox `tests` session.

### D9. Documentation

The execution guide is rewritten. Following the rule that doc code executes, its main example defines a small API
venue inline, builds a dataloader from an inline custom source carrying one match dated tomorrow, and runs a dry run
with an injected clock, so the block runs at build time with no network and no secret. The browser venue section shows
a subclass as reference code, since driving a real site cannot run offline, which the constitution allows.

## Risks / Trade-offs

- [A bookmaker changes its page and a hook breaks] → The base turns a failed step into `rejected` before any stake is
  entered, and a dry run before each live run shows the breakage without money.
- [Confirmation detection is wrong and a real bet is reported as unconfirmed] → `refused_unconfirmed` blocks a rerun
  through the ledger and tells the user to check the venue, so the error leads to no bet rather than to two.
- [Re-reading the fixtures each poll is slow or rate limited] → The default poll is five minutes, a read failure only
  stops the run when it happens at the moment, and a failed poll is logged and the wait continues.
- [The model's price came from a market average while the bookmaker's price differs] → The minimum price is the
  model's price, so the run refuses rather than placing at a price the model did not see value at. This means some
  value bets are not placed. That is the intended trade.
- [The lead time is wrong for a bookmaker that closes markets earlier] → The lead time is a run parameter.
- [Breaking removals hurt users of 008] → 008 never worked end to end, the changelog lists each removal with its
  replacement, and the commits carry `BREAKING CHANGE` footers so the release tool bumps the version.

## Migration Plan

1. Land the change on a feature branch from `development`, behind the existing `execution` extra.
2. Release notes map each removal: `fix` and `page fix` become hooks in a `BaseBrowserVenue` subclass, `placer`
   becomes `place` on the subclass, and `urls` becomes `find_event`.
3. Phase 1 counts as proven only after the manual acceptance run in tasks.md: a dry run and then a minimum-stake live
   run on a real bookmaker with the user's own subclass. Phase 2, multi-event, is proposed as its own change only
   after that.
4. To roll back, revert the merge. No persisted format from 008 needs migrating, since 008 persisted nothing.

## Open Questions

- Whether `ExecutionError` for unsupported operations should later become its own named exception. This needs a
  profile amendment and changes nothing in the specs.
