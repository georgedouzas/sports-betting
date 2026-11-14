## MODIFIED Requirements

### Requirement: Malformed moments and aliases
An `--odds-moment` value MUST be a status and a minute separated by a colon, and an `--alias` value MUST be two names
separated by `=`, the odds source's name first and the statistics source's name second. The help SHALL state that
order. A malformed value SHALL be answered with a message that shows the correct form, `inplay:45` for a moment and
`Olimpia Milano=EA7 Emporio Armani Milan` for an alias.

#### Scenario: A malformed moment
- **WHEN** the user passes `--odds-moment halftime` to `sportsbet dataloader train extract`
- **THEN** the output contains `inplay:45`

#### Scenario: A malformed alias
- **WHEN** the user passes `--alias nonsense` to `sportsbet dataloader train extract`
- **THEN** the output contains `Olimpia Milano`

#### Scenario: An alias in the stated order pairs the club
- **GIVEN** a club the odds source spells `Utd of Manchester` and the statistics source spells `Man United`
- **WHEN** the user passes `--alias 'Utd of Manchester=Man United'`
- **THEN** the matches of `Man United` keep their odds

### Requirement: Error reporting
A usage error, such as a missing required option, a value outside a choice, or a `--dataloader` or `--bettor` path that
does not exist, SHALL be reported by the option parser with exit status 2. An error the library raises as a
`BuildError` or `ValueError` SHALL be printed as its message in a red panel with no traceback, and the command SHALL
exit with status 1, whether the error comes from building the selection, the model or the extraction settings.

#### Scenario: An unknown odds type
- **WHEN** the user runs `sportsbet dataloader train extract` with `--odds-type nonexistent`
- **THEN** the exit status is 1
- **AND** the output lists `market_average`
- **AND** the output contains no `Traceback`

#### Scenario: A selection that does not build
- **WHEN** the user runs `sportsbet dataloader params --stats football-data --alias nonsense`
- **THEN** the exit status is 1
- **AND** the output contains no `Traceback`

#### Scenario: Extraction settings that do not build
- **WHEN** the user runs `sportsbet dataloader train extract` with `--target-event-status inplay` and
  `--target-event-time soon`
- **THEN** the exit status is 1
- **AND** the output contains no `Traceback`
