# Proposal

## Why

The library finds value bets and stops. A user who has fitted a bettor and extracted the upcoming fixtures reads
`bettor.bet(X_fix, O_fix)` and then opens a browser and types each bet in by hand. The user asked for real betting:
authenticate to a bookmaker, read its markets, and place bets, driven by an outside agent. Placement must not live on
the bettor. The user asked for dry run by default, stake and exposure limits, a kill switch, idempotent placement,
credentials from the environment, no agent loop, and no anti-detection machinery.

Planning research shaped the contract. Betfair was the only one of four surveyed exchanges whose placement call carries
a caller reference that the venue stores and filters on, so it fixed the shape of the venue contract. No exchange offers
a placement sandbox. Betfair's delayed application key places real bets on the live exchange. All four surveyed
exchanges are closed to Greece, where the maintainer is, so the site-driven path is the one the maintainer can use.

## What Changes

- Add a `sportsbet.execution` module behind a new optional `execution` extra that holds `playwright` alone.
- Add the venue contract `BaseVenue` with `authenticate`, `list_markets`, `read_balance`, `place`, `read_status` and
  `cancel`, plus `key` and `can_cancel`.
- Add `BetIdentity`, `PlacementIntent`, `PlacementQuote`, `PlacementReceipt` with a `pandera` schema,
  `PlacementStatus`, `ExposureLimits`, `CredentialRef` and `CancellationUnsupported`.
- Add the batch entry points `quote(venue, intents, limits)` and
  `place(venue, quote, limits, confirm_stake=None, confirm_exposure=None)`. Nothing stakes unless the caller passes
  back the exact quoted stake and exposure.
- Add `BrowserSession` with `navigate`, `snapshot`, `click`, `type`, `select` and `fix`, plus `FixedSession` and
  `PageSnapshot`. `BrowserSession` is not a venue and offers no `place`, `read_status` or `cancel`.
- Add `build_venue` beside `build_bettor`, resolving a user venue named as `venue.py:VENUE`.
- Add the `sportsbet execution` CLI group: `venue`, `markets`, `balance`, `quote`, `place`, `status`, `cancel`, and
  `page read`, `page act`, `page fix`.
- Add MCP tools `execution_venue_info`, `execution_authenticate`, `execution_list_markets`, `execution_read_balance`,
  `execution_quote`, `execution_place`, `execution_read_status`, `execution_cancel`, and `browser_navigate`,
  `browser_snapshot`, `browser_click`, `browser_type`, `browser_select`, `browser_fix`.
- Add a user guide page that leads with the money and terms-of-service risks.
- A Betfair adapter `BetfairVenue` was built and then removed after implementation. The maintainer rejected shipping
  any bookmaker in the library, so a venue with an API is user provided, like a model.

## Capabilities

### New Capabilities

- `execution/venue-contract`: the venue contract, bet identity, receipts, credentials named by variable, the browser
  session with its page primitives and pinned sessions, and the no-evasion boundary.
- `execution/event-run`: the batch flow that turns value bets into intents, quotes them, and places them only against
  the exact quoted figures, under stake and exposure ceilings, a kill switch and a minimum price.

### Modified Capabilities

- `interfaces/cli`: adds the `sportsbet execution` command group and extends surface parity to it.
- `interfaces/mcp-server`: adds the `execution_*` and `browser_*` tools and extends surface parity to them.

## Impact

- Code: new `src/sportsbet/execution/` (`__init__.py`, `_base.py`, `_browser.py`, `_credentials.py`, and a
  `_betfair.py` that was later removed), new `src/sportsbet/cli/_execution.py`, an extended
  `src/sportsbet/mcp/_server.py`, and `build_venue` in `src/sportsbet/_selection.py`. `src/sportsbet/dataloaders` and
  `src/sportsbet/evaluation` show a zero diff.
- API: new public names in `sportsbet.execution`. Bettors are unchanged.
- Dependencies: no new required runtime dependency. The `execution` extra holds `playwright`, locked with
  `pdm lock -G execution`. The browser binary is a separate `playwright install` step.
- Tests: `tests/test_execution.py`, `tests/test_betfair.py`, `tests/test_browser.py`, a `FakeVenue` and a loopback page
  fixture in `tests/conftest.py`, and the execution group in `tests/cli/test_parity.py`.
- Docs: `docs/overview/user_guide/execution.md`, registered in the `properdocs.yml` nav, and `sportsbet.execution` in
  `docs/generate_api.py`.
- Requirements checklist: every item passed. The review notes are kept in the Quality checks section of design.md.
