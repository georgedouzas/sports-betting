## ADDED Requirements

### Requirement: Complementary markets derived from the data
Bettors SHALL derive the groups of mutually exclusive and exhaustive markets from the markets the data carries. `over_X`
and `under_X` MUST form a group at any `X`. The outcome group MUST hold whichever of home win, draw and away win the
data holds. `BaseBettor.COMPLEMENTARY_EVENTS` MUST default to `None` and act only as an override.

#### Scenario: A market outside the old list
- **WHEN** a bettor is fitted and asked to bet on data carrying `over_1.5` and `under_1.5`
- **THEN** their probabilities are normalized to sum to one and `bet` returns selections without error

#### Scenario: A sport without a draw
- **WHEN** the data carries `home_win` and `away_win` and no `draw`
- **THEN** the bettor treats home win and away win as one two-way group
