# Design

## Decisions

### D1. The duplicated term is the shots

The term divided by 50 is the one replaced, with `shots_for` and `shots_against`. A shot is worth less than a shot on
target, so the smaller weight belongs to it. `shots` is already in the filtered statistics and is otherwise unused,
which supports reading the duplicate as a slip. When the feed has no shots column for a season, the term is missing, as
the shots on target term is today.

### D2. Spelling fixes without a deprecation

`hit_woodork` and `fouls_commited` never leave the module, so they are renamed in place with no alias for the old
spelling.
