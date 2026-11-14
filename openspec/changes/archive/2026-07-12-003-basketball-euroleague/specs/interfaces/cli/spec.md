## ADDED Requirements

### Requirement: Basketball on the command line
The `sportsbet` command line SHALL reach `BasketballDataLoader` through the same dataloader configuration that reaches
`SoccerDataLoader`, so the Python API and the command line gain basketball at the same time.

#### Scenario: Basketball configuration
- **WHEN** a user's configuration names a basketball dataloader with an odds source
- **THEN** the `sportsbet dataloader` commands prepare and extract it as they do a soccer dataloader
