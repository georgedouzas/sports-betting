## MODIFIED Requirements

### Requirement: Football-data form features

The pre-play row SHALL carry, for the home and the away team, the mean over all earlier matches of the season
(`_avg`) and over the last three earlier matches (`_latest_avg`) of `points`, `adj_points`, `goals_for`,
`goals_against`, `adj_goals_for` and `adj_goals_against`, named such as `home_points_avg`. A feature MUST never include
the result of its own match. The adjusted goals of a team in a match SHALL be its goals, plus its red cards, plus a
twentieth of its yellow cards, plus a tenth of its shots on target, plus a fiftieth of its shots, plus a hundredth of
its corners, and the adjusted goals against SHALL be the same sum over the opponent's numbers.

#### Scenario: First match has no form

- **WHEN** the source transforms a season whose first match is Arsenal at home
- **THEN** that match's `home_points_avg` is missing

#### Scenario: Form from the earlier match

- **WHEN** Chelsea lost its first match and then plays at home
- **THEN** that pre-play `home_points_avg` is 0

#### Scenario: Adjusted goals count each statistic once

- **GIVEN** a team that in its first match scored 2 goals with 1 red card, 2 yellow cards, 5 shots on target, 10 shots
  and 4 corners
- **WHEN** the source transforms the season
- **THEN** the `adj_goals_for` average that team carries into its second match is 3.84
