## MODIFIED Requirements

### Requirement: Data selection arguments
`available_params`, `odds_types`, `extract_train_data` and `extract_exploration_data` SHALL take `stats` (required),
`odds`, `leagues`, `divisions`, `years`, `odds_key_env` (default `ODDS_API_KEY`), `odds_markets`, `odds_regions`,
`odds_moments` and `aliases`, with the meanings and the forms that `build_dataloader` gives them. Each alias SHALL be
two names separated by `=`, the odds source's name first and the statistics source's name second.

#### Scenario: An unknown statistics source
- **WHEN** a client calls `available_params` with `stats='quidditch'`
- **THEN** the call fails with a message that lists `euroleague`, `football-data` and `nba`
