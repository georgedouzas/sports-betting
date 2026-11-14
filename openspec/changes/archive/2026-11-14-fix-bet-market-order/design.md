# Design

## Context

`bet` loops over `complementary_events_`, finds the indices of each group's markets in `betting_markets_`, keeps the
best value bet in each group, and joins the groups with `pd.concat(..., axis=1)`. The join keeps group order.

## Decisions

### D1. Reorder after the join

After the join, select the columns in the order of `betting_markets_` before turning the frame into an array. The
columns of each group frame are already the odds column names, so the reorder is one indexing step and keeps the loop as
it is. Building a full boolean frame first and filling each group into it was considered. It is the same result with
more code.

### D2. Check the groups at fit

`fit` already derives `complementary_events_`. It then checks that every market in `betting_markets_` falls in some
group, and raises `ValueError` naming the markets left out and the groups it has. `ValueError` is what the
scikit-learn contract expects for invalid input, so no named exception is needed. Checking at fit rather than in `bet`
fails before the user has spent time on a backtest.

## Risks

- Saved bettors fitted before the change carry groups that were never checked. They still load and still bet, with the
  corrected order. One whose markets fall outside the groups failed before as well, with a worse message.
