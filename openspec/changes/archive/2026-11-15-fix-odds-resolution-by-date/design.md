# Design

## Decisions

### D1. Nearest kick-off within 36 hours

After the team names are normalized, join each odds row to the statistics matches of the same league, division, year
and teams with `pd.merge_asof` on the kick-off, `direction='nearest'` and `tolerance=pd.Timedelta('36h')`, grouped by
the match columns. Both sides are UTC already, so the tolerance has to absorb a source that dates a game by its local
day or moves it by a few hours. It does not have to absorb a postponement, which gets a new date in both sources.
Thirty six hours is shorter than the shortest gap between two meetings of the same pairing in any covered league, which
is days, not hours.

### D2. Keep the statistics identity

As today, a resolved row carries the statistics `date` and team spelling, so the dataloader joins it to its match.

## Risks

- A source that dates a game more than 36 hours away from the other loses those odds. The reconciliation counts in the
  dataloader report it, so it is visible rather than silent.
