# Proposal

## Why

The football-data form features carry three defects in `sources/_common/_football_data.py`.

- `adj_goals_for` and `adj_goals_against` add the shots on target twice, once divided by 10 and once divided by 50. The
  second term was meant to be the shots, which the code already reads into `shots_for` and `shots_against` and never
  uses. Every `adj_goals_*` and `adj_points` feature, and the averages built from them, is off.
- The filter that picks the statistics for the features looks for `fouls_commited`, while the column is named
  `fouls_committed`, so fouls are silently left out of the working frame.
- The woodwork columns are named `hit_woodork`.

The fouls and woodwork columns never reach a snapshot, so the last two are internal. The first changes what users get.

## What Changes

- The adjusted goals of a team in a match are its goals, plus its red cards, plus a twentieth of its yellow cards, plus
  a tenth of its shots on target, plus a fiftieth of its shots, plus a hundredth of its corners. The adjusted goals
  against are the same sum over the opponent's numbers.
- The fouls filter and the woodwork column names are spelled correctly.

## Non-goals

- Revisiting the weights of the adjusted goals. The fix keeps every weight and replaces only the duplicated term.
- Adding fouls or woodwork to the features.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `sources/football-data`: the definition of the adjusted goals behind the form features.

## Impact

- Code: `src/sportsbet/sources/_common/_football_data.py`.
- Tests: `tests/sources/` gains a test that computes `adj_goals_for` for a known match.
- Results: the `adj_*` features of every football-data dataset change. Saved bettors trained on them keep loading but
  were trained on the old values. The changelog records it as a fix.
