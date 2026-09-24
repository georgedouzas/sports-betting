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

## A venue is named the way a model is named

A venue follows `build_bettor`, not `build_dataloader`, and the reason is the one
[_selection.py](../../../src/sportsbet/_selection.py) already gives for models: a dataloader is a
sport and a short closed list of names, so it fits in arguments, while an estimator can be any
pipeline anybody can build, so it is named by where it lives and built in Python.

A venue is the second kind. Its URL, its notes, and its certificate paths are not a closed list,
and a site-driven one carries prose. So `build_venue` sits beside `build_bettor` in
`_selection.py` and takes the same two forms:

```text
--venue betfair              # ready-made, as `odds-comparison` is
--venue venue.py:VENUE       # one of your own, as `models.py:BETTOR` is
```

```python
# venue.py
from sportsbet.execution import BrowserSession

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

Swapping Novibet for Stoiximan, or for Betfair, is an edit to that file and nothing else. Both
surfaces resolve the reference the same way, through the existing `_load_object`, so neither owns
a format the other has to learn and no credential enters an argument.

`build_venue` imports `sportsbet.execution` lazily, since the extra is optional and `_selection.py`
is imported on every run. Without the extra it names the extra to install, which is SC-010.

## CLI

Every parameter is passed with `--`, matching the existing groups. No positional arguments.

```text
sportsbet execution venue    --venue venue.py:VENUE
sportsbet execution markets  --venue betfair --dataloader loader.pkl
sportsbet execution balance  --venue betfair
sportsbet execution quote    --venue betfair --bettor model.pkl --dataloader loader.pkl \
                             --output quote.json
sportsbet execution place    --venue betfair --quote quote.json \
                             --confirm-stake 50.0 --confirm-exposure 50.0 \
                             --max-stake 10.0 --max-exposure 100.0
sportsbet execution status   --venue betfair --quote quote.json
sportsbet execution cancel   --venue betfair --identity <ref>
```

`quote` writes the itemised batch and its totals. `place` reads it back and refuses unless both
confirmations match. Running `place` without them prints the full quote, stakes nothing, and exits
non-zero, so a forgotten flag costs a run rather than a balance.

There is no `--dry-run` flag, deliberately. Dry run is the absence of confirmation, not the
presence of a flag, so no default can be misconfigured into spending.

The credential is named by the venue, never passed on the command line, because a CLI option is a
shell history entry (FR-017). A ready-made venue defaults to a variable NAME, so nothing sensitive
has a default.

## MCP

Tools mirror the CLI, and the confirmation rule is enforced in code rather than described in a
tool description (the pattern feature 005 established for `prepare` and `confirm_cost`).

```text
execution_venue_info(venue) -> {key, url, notes, can_cancel}
execution_quote(venue, bettor_path, dataloader_path, limits) -> quote
execution_place(venue, quote, confirm_stake, confirm_exposure, limits) -> receipts
```

`execution_venue_info` is how the user's site knowledge reaches the agent: it returns the `notes`
and the URL from the venue verbatim. The package stores and returns that text and never reads it,
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
