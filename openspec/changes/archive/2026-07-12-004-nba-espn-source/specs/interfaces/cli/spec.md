## ADDED Requirements

### Requirement: NBA on every surface that reaches the EuroLeague
`NBAStats` SHALL be reachable from every delivery surface that reaches `EuroLeagueStats`, in the same way. The command
line SHALL reach it through a user configuration whose `DATALOADER_CLASS` is a basketball dataloader using `NBAStats`,
with no registration step.

#### Scenario: NBA configuration on the command line
- **WHEN** a user's configuration module names a `DATALOADER_CLASS` that subclasses `BasketballDataLoader` with
  `stats=NBAStats()` and an odds source
- **THEN** the `sportsbet` dataloader commands prepare and extract it as they do a EuroLeague configuration
