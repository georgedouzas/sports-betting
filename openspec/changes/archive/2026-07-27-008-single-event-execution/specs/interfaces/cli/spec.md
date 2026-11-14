## ADDED Requirements

### Requirement: Single-event execution run command
`sportsbet execution run` SHALL run the single-event unit and take `--venue` naming a browser session reference,
`--dataloader` (`-d`), `--bettor` (`-b`), `--event`, a required `--stake`, a repeatable `--url`, `--live`, `--poll` and
`--output` (`-o`) for a directory to write the receipt CSV to. It SHALL log the run to the terminal and place through
the default placer over the controls pinned by `sportsbet execution page fix`.

#### Scenario: A dry run from the command line
- **WHEN** the user runs `sportsbet execution run --venue venue.py:VENUE -d loader.pkl -b model.pkl --event "Arsenal vs
  Chelsea" --stake 10 --url "$URL"` without `--live`
- **THEN** the terminal logs the setup match, the status, the price and the bet it would place, and nothing is staked

#### Scenario: An armed run from the command line
- **WHEN** the user adds `--live` to the same command
- **THEN** at the moment the command places one bet through the pinned `stake` and `confirm` controls

## REMOVED Requirements

### Requirement: Execution quote and place commands
**Reason**: The batch quote-and-confirm model is retired. A single-event run is armed once with `--live` and places at
the moment.
**Migration**: Use `sportsbet execution run --event ... --stake ... --url ...`, adding `--live` to place for real.
