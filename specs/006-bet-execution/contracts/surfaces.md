# Contract: The three surfaces

**Feature**: [spec.md](../spec.md) | **Venue contract**: [venue.md](./venue.md)

FR-020 and SC-006 require every execution capability on the Python API, the CLI and the MCP
server, with no surface holding something the others cannot reach. The CLI group mirrors the
Python API, as `dataloader` and `evaluation` already do.

## Parity

| Capability | Python API | CLI | MCP tool |
| --- | --- | --- | --- |
| Read the venue config | `venue.key`, `venue.notes` | `execution venue` | `execution_venue_info` |
| Authenticate | `venue.authenticate(cred)` | implicit in every command | `execution_authenticate` |
| Read markets | `venue.list_markets(matches)` | `execution markets` | `execution_list_markets` |
| Read balance | `venue.read_balance()` | `execution balance` | `execution_read_balance` |
| Quote a batch | `quote(venue, intents, limits)` | `execution quote` | `execution_quote` |
| Place a batch | `place(venue, quote, limits, ...)` | `execution place` | `execution_place` |
| Read status | `venue.read_status(ids)` | `execution status` | `execution_read_status` |
| Cancel | `venue.cancel(id)` | `execution cancel` | `execution_cancel` |
| Read a page | `session.navigate/snapshot` | `execution page read` | `browser_navigate`, `browser_snapshot` |
| Act on a page | `session.click/type/select` | `execution page act` | `browser_click`, `browser_type`, `browser_select` |
| Pin a session | `session.fix(match, locators)` | `execution page fix` | `browser_fix` |

`tests/cli/test_parity.py` already exists and enforces this shape for the current groups. It gains
the execution group.

## The venue is configured, not selected by name

The venue arrives as a `VENUE` object in the same Python config module the CLI and the MCP server
already read for `DATALOADER`. A venue is not a string the library looks up in a table, because
that table is the per-site knowledge FR-007 keeps out.

```python
# config.py
from sportsbet.execution import BrowserSession

DATALOADER = SoccerDataLoader(param_grid={'league': ['Greece']})
VENUE = BrowserSession(
    key='novibet',
    url='https://www.novibet.gr/stoixima',
    notes="""
    Bet history is under My Account > Bet History.
    The slip opens on the right after clicking a price; confirm is two steps.
    Check the history for this match before placing, so a retry does not stake twice.
    """,
    credential_env=('NOVIBET_USERNAME', 'NOVIBET_PASSWORD'),
)
```

Swapping Novibet for Stoiximan, or for Betfair, is an edit to this file and nothing else. The
user owns which venue, which URLs, and what the agent is told about the site. Both surfaces read
the same module, so they cannot drift and no credential enters an argument (the pattern feature
005 established).

## CLI

Every parameter is passed with `--`, matching the existing groups. No positional arguments.

```text
sportsbet execution venue    --config config.py
sportsbet execution markets  --config config.py --dataloader loader.pkl
sportsbet execution balance  --config config.py
sportsbet execution quote    --config config.py --bettor model.pkl --dataloader loader.pkl \
                             --output quote.json
sportsbet execution place    --config config.py --quote quote.json \
                             --confirm-stake 50.0 --confirm-exposure 50.0 \
                             --max-stake 10.0 --max-exposure 100.0
sportsbet execution status   --config config.py --quote quote.json
sportsbet execution cancel   --config config.py --identity <ref>
```

`quote` writes the itemised batch and its totals. `place` reads it back and refuses unless both
confirmations match. Running `place` without them prints the full quote, stakes nothing, and exits
non-zero, so a forgotten flag costs a run rather than a balance.

There is no `--dry-run` flag, deliberately. Dry run is the absence of confirmation, not the
presence of a flag, so no default can be misconfigured into spending.

The credential is named in the config, never passed on the command line, because a CLI option is a
shell history entry (FR-017).

## MCP

Tools mirror the CLI, and the confirmation rule is enforced in code rather than described in a
tool description (the pattern feature 005 established for `prepare` and `confirm_cost`).

```text
execution_venue_info(config_path) -> {key, url, notes, can_cancel}
execution_quote(config_path, bettor_path, dataloader_path, limits) -> quote
execution_place(config_path, quote, confirm_stake, confirm_exposure, limits) -> receipts
```

`execution_venue_info` is how the user's site knowledge reaches the agent: it returns the `notes`
and URLs from the config verbatim. The package stores and returns that text and never reads it,
so the site knowledge lives in the user's config and the agent's head, and nowhere in the library.

`execution_place` without both confirmations returns refusals stating the real figures. With
figures that do not match the quote it refuses and states the real ones. Only exact figures place
anything. An agent therefore cannot stake without having read the quote, which is FR-009 applied
to the caller most likely to skim.

DataFrames cross as records, as the existing MCP tools already do.

The browser tools carry the site knowledge nowhere: they navigate, snapshot and act on refs. The
agent supplies which site, which button, which market. No model, model key, or loop enters the
package (FR-022).

## Degrading without the extras

`execution` and `mcp` are separate extras and the tools need both. Missing either produces a
message naming the extra to install, rather than an `ImportError` traceback. Installing neither
leaves no way to place a bet and pulls in no execution dependency (SC-010, FR-004).
