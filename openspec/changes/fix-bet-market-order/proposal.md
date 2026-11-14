# Proposal

## Why

`BaseBettor.bet` builds its answer group by group, over `complementary_events_`, and joins the groups side by side.
The columns come back in group order, not in the order of `betting_markets_`. The caller reads them in the order of
`betting_markets_`, so whenever the two orders differ, `score` and `backtest` credit a bet and its return to the wrong
market. This was confirmed by running it. A bettor whose groups list `over_2.5` before `home_win`, while
`betting_markets_` lists `home_win` first, reports the over bets as home wins.

A selected market that sits in no complementary group, which a user can cause by setting `COMPLEMENTARY_EVENTS` by
hand, is dropped from the join. The result then has fewer columns than markets, or, when no group matches, pandas
raises `No objects to concatenate`, which tells the user nothing.

## What Changes

- `bet` returns its columns in the order of `betting_markets_`, whatever the order of the groups.
- A selected market that belongs to no complementary group raises `ValueError` naming the market and the groups, at
  fit, before any prediction is made.

## Non-goals

- Changing how a value bet is chosen within a group.
- Changing `score` or `backtest`, which are right once `bet` returns its columns in the stated order.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `evaluation/bettors`: the column order of `bet`, and the error for a market outside every group.

## Impact

- Code: `src/sportsbet/evaluation/_base.py`.
- Tests: `tests/evaluation/` gains a test with groups ordered against the markets, and one with a market in no group.
- Results: backtests of bettors whose two orders differed change, and are now right. The changelog records it as a fix.
