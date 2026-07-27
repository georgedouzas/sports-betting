# Contract: Python API

The single public entry point of the feature, re-exported from `sportsbet.execution`.

## `execute_event`

```python
async def execute_event(
    event: str,
    bettor: BaseBettor,
    dataloader: BaseDataLoader,
    session: BrowserSession,
    *,
    stake: float,
    urls: list[str],
    live: bool = False,
    placer: Placer | None = None,
    poll: pd.Timedelta = pd.Timedelta("30s"),
    bound: pd.Timedelta | None = None,
    clock: Clock | None = None,
    wait: Wait | None = None,
) -> pd.DataFrame: ...
```

**Parameters**

- `event`: the one match to act on, `"Home vs Away"`. Must be among the dataloader's fixtures.
- `bettor`: a fitted bettor. Decides whether to bet and the selection, never the stake.
- `dataloader`: supplies the event's evolving data. Polled on `poll` for the event's current features and odds, and
  carries the fitted moment through `target_event_status_` and `target_event_time_`.
- `session`: a headless `BrowserSession` for the bookmaker, logged in.
- `stake`: the fixed amount to place. The only stake the unit uses.
- `urls`: candidate bookmaker URLs. Explored and matched to the event during setup.
- `live`: arms the run. `False` is a no-stakes dry run that does everything but call the placer.
- `placer`: how to place on the site. Defaults to a built-in that enters `stake` into the pinned `stake` control and
  clicks the pinned `confirm` control of the matched `FixedSession`. A custom placer overrides it for odd sites.
- `poll`: the interval between source polls while monitoring.
- `bound`: how long to wait for an event that never advances before giving up. `None` waits without a bound.
- `clock`, `wait`: injection points for time, defaulting to real time and `asyncio.sleep`, for tests.

**Returns**: a receipts frame, validated by `PlacementReceiptSchema`, with one row if a bet was placed and no rows
otherwise.

**Behaviour**

1. Explore `urls`, navigating each headless and matching the one whose page names the event and selection, then pin
   the site's controls as a `FixedSession`. If none match, log and return an empty frame.
2. Ensure the session is logged in, prompting the user during setup when it is not. If login is not in place within
   the bound, log and return an empty frame.
3. Poll the dataloader on `poll`, logging the event, its status, the current price, and the decision at each step,
   until the betting moment from `find_betting_moment`.
4. At the moment, apply `bettor.bet` to the event's one-row features and odds. If it returns no value bet, log and
   return an empty frame.
5. If it returns a value bet, log the selection, stake, and price about to be staked. When `live`, call `placer` once
   to place `stake` on the selection and record the receipt. When not `live`, record a dry-run receipt.
6. Place at most one bet over the whole run.

**Raises**: `ExecutionError` when `event` is not among the fixtures, or when the session cannot be driven.
`VenueBlockedError` when the site blocks automated access.

## `Placer`

```python
Placer = Callable[[PlacementIntent, BrowserSession], Awaitable[PlacementReceipt]]
```

Unchanged in shape from the retired `_schedule` alias, moved to live with the runner. The library ships a default
implementation driving the pinned `stake` and `confirm` controls, so the common case needs no user code.

## Re-exports

`sportsbet.execution` adds `execute_event` and keeps `Placer`, `BrowserSession`, `FixedSession`, `PageSnapshot`,
`BetIdentity`, `PlacementIntent`, `PlacementReceipt`, `PlacementReceiptSchema`, `PlacementStatus`, `ExposureLimits`,
`build_receipts_frame`, `find_betting_moment`, `build_venue`, `CredentialRef`, `resolve`, and the error family.
