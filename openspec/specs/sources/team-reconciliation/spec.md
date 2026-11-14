# sources/team-reconciliation Specification

## Purpose

Defines how the library pairs the odds of one feed with the matches of another when the two feeds spell their teams
differently, through name normalization, roster pairing by similarity, and aliases the user supplies.

## Requirements

### Requirement: Team name normalization

`normalize_team_name(name)` SHALL return the name in lower case, without accents, without characters other than ASCII
letters, digits and spaces, and without the noise words `fc`, `afc`, `cf`, `sc`, `ac`, `as`, `ss`, `us`, `if`, `bk`,
`club` and `the`, its remaining words joined by single spaces.

#### Scenario: Accents and suffix removed

- **WHEN** a caller calls `normalize_team_name('Atlético Madrid CF')`
- **THEN** it returns `atletico madrid`

#### Scenario: Punctuated suffix removed

- **WHEN** a caller normalizes `Manchester United`, `Manchester United FC` and `Manchester United F.C.`
- **THEN** each returns `manchester united`

### Requirement: Common prefix length

`count_common_prefix(one, other)` SHALL return the number of leading characters the two strings share.

#### Scenario: Shared prefix

- **WHEN** a caller calls `count_common_prefix('manchester united', 'manchester city')`
- **THEN** it returns 11

#### Scenario: No shared prefix

- **WHEN** a caller calls `count_common_prefix('Arsenal', 'Man Utd')`
- **THEN** it returns 0

### Requirement: Name similarity

`measure_names_similarity(one, other)` SHALL return a score in `[0.0, 1.0]`. For each word of the name with fewer
words, it takes the best ratio of the common prefix to the shorter word length over the words of the other name that
share at least three leading characters, or 0.0 when none does, and averages these. An empty name MUST score 0.0.

#### Scenario: Abbreviated name

- **WHEN** a caller calls `measure_names_similarity('manchester united', 'man united')`
- **THEN** it returns 1.0

#### Scenario: Unrelated names

- **WHEN** a caller calls `measure_names_similarity('everton', 'liverpool')`
- **THEN** it returns 0.0

#### Scenario: Closer name scores higher

- **WHEN** a caller compares `man united` with `manchester united` and `man city` with `manchester united`
- **THEN** the first score is higher than the second

### Requirement: Roster pairing

`pair_rosters(normalized_stats_names, normalized_odds_names)` SHALL pair identical names first, then pair each
remaining odds name with its most similar remaining statistics name, best scores first. A pairing MUST be accepted only
when its score is at least 0.6 and exceeds the runner-up by at least 0.15, or when exactly one name is left on each
side. It SHALL return the odds-to-statistics pairing, the unpaired odds names and the unpaired statistics names.

#### Scenario: Abbreviation paired, stranger left out

- **WHEN** a caller calls `pair_rosters({'manchester united', 'arsenal'}, {'man united', 'chelsea'})`
- **THEN** it returns `({'man united': 'manchester united'}, {'chelsea'}, {'arsenal'})`

#### Scenario: Full rosters paired

- **WHEN** a caller pairs the normalized rosters of the test statistics and odds samples
- **THEN** every name is paired and both unpaired sets are empty

### Requirement: Roster building

`build_roster(data)` SHALL return the teams of the `home_team` and `away_team` columns, keyed by normalized name and
valued by the original spelling.

#### Scenario: Original spelling kept

- **WHEN** a caller calls `build_roster` on a frame with `home_team` `Manchester United` and `away_team` `Arsenal FC`
- **THEN** it returns `{'manchester united': 'Manchester United', 'arsenal': 'Arsenal FC'}`

### Requirement: Identity normalization

`normalize_identity(data, mapping=None)` SHALL return the columns `league`, `division`, `year`, `home_team` and
`away_team` with normalized team names. When `mapping` is given, keyed by `(league, division, year)` and holding
normalized odds names mapped to normalized statistics names, each team name found in its group's mapping MUST be
replaced.

#### Scenario: Names normalized

- **WHEN** a caller calls `normalize_identity` on a frame with `home_team` `Manchester United` and `away_team`
  `Arsenal FC`
- **THEN** the result has the five match columns and the teams `manchester united` and `arsenal`

### Requirement: Odds resolution

`resolve_odds(stats, odds, aliases=None)` SHALL return the odds rows whose match, identified by league, division,
year and the two teams, exists in the statistics. Pairing MUST run separately for each league, division and year. Each
returned row MUST carry the statistics spelling of the teams and the statistics `date`, with the columns of `odds` in
their original order.

#### Scenario: Statistics spelling carried

- **GIVEN** statistics naming `Man United` and odds naming `Manchester United` for the same match
- **WHEN** a caller calls `resolve_odds(stats, odds)`
- **THEN** the returned row has `home_team` `Man United` and keeps the price of the odds

#### Scenario: Every sample odds row resolved

- **WHEN** a caller resolves the test odds sample against the test statistics sample
- **THEN** the result has as many rows as the odds, and its home teams are all spelled as in the statistics

### Requirement: Unpaired teams dropped

`resolve_odds` SHALL drop an odds row whose teams it cannot pair with a match of the statistics, rather than attach the
row to another match.

#### Scenario: Unrecognizable name

- **GIVEN** odds that call `Manchester United` by the name `Utd of Manchester`
- **WHEN** a caller resolves them against statistics naming `Man United`
- **THEN** no returned row has the home team `Man United`

#### Scenario: Nickname without alias

- **GIVEN** statistics naming `Tottenham` and odds naming `Spurs` for the same match
- **WHEN** a caller calls `resolve_odds(stats, odds)`
- **THEN** the match of `Tottenham` is absent from the result

### Requirement: Aliases

`resolve_odds` SHALL accept `aliases` mapping odds team names to statistics team names, compared after normalization.
They MUST be added to the aliases the library ships, which map `Olimpia Milano` to `EA7 Emporio Armani Milan`, and an
alias the user gives MUST win over a shipped alias for the same name.

#### Scenario: Alias bridges a name

- **GIVEN** odds that call `Manchester United` by the name `Utd of Manchester`
- **WHEN** a caller calls `resolve_odds(stats, odds, {'Utd of Manchester': 'Man United'})`
- **THEN** a returned row has the home team `Man United`

### Requirement: Reconciliation in the dataloader

A `DataLoader` SHALL pass its odds through `resolve_odds` with its `aliases` parameter when the statistics and odds
sources have different names, and MUST leave the odds as read when the two sources share a name, since one feed then
carries both.

#### Scenario: Same-named pair

- **WHEN** a dataloader reads `FootballDataStats` and `FootballDataOdds`
- **THEN** the odds are not passed through `resolve_odds`

#### Scenario: Different feeds

- **WHEN** a dataloader reads `EuroLeagueStats` and `OddsApi` with `aliases` given
- **THEN** the odds are resolved against the statistics with those aliases
