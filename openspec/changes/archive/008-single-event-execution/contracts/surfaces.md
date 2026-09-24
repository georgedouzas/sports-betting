# Contract: CLI and MCP Surfaces

The single-event unit is reached from all three surfaces. The parity test asserts the command and the tool stay in
step.

## CLI

The `execution run` command is repurposed from the batch runner to the single-event unit.

```text
sportsbet execution run
  --venue venue.py:VENUE        # a browser session reference
  --dataloader, -d PATH         # a saved dataloader configured for the event
  --bettor, -b PATH             # a model saved by `fit`
  --event "Home vs Away"        # the one event to act on
  --stake FLOAT                 # the fixed stake, required
  --url URL                     # a candidate bookmaker URL, repeatable
  --live                        # arm the run, off by default (dry run)
  --poll 30s                    # the source poll interval
  --output, -o PATH             # a directory to write the receipt CSV to
```

The command logs the run to the terminal as it goes, reusing the run log handler. Without `--live` it stakes nothing
and logs the bet it would have made. It places on the browser through the default placer driving the controls pinned
by `execution page fix`, so no Python is needed for the common site.

**Kept execution commands**: `venue`, `markets`, `balance`, `status`, `cancel`, and the `page` group with `read`,
`act`, and `fix` for exploring and pinning a site.

**Removed commands**: `quote` and `place`, the batch quote-and-confirm pair.

## MCP

The `execution_run` tool mirrors the command. Its parameters, by the parity contract, cover what the command can be
told: `venue`, `dataloader`, `bettor`, `event`, `stake`, `urls`, `live`, `poll`, `output`.

**Kept tools**: `execution_venue_info`, `execution_authenticate`, `execution_read_balance`, `execution_list_markets`,
`execution_read_status`, `execution_cancel`, and the `browser_*` tools for exploring and pinning.

**Removed tools**: `execution_quote` and `execution_place`.

## Parity

The parity test's pairs lose `(["execution", "quote"], "execution_quote")` and `(["execution", "place"],
"execution_place")`, and keep `(["execution", "run"], "execution_run")` with the new single-event parameter set. The
test still asserts every kept command has a tool and every tool can be told what its command can be told.

## Surface note

A custom `placer` is Python code, so a bespoke-site placement is a Python-API capability. The CLI and the MCP tool
place through the default placer over the pinned controls, and otherwise monitor, decide, and dry-run. This matches the
existing stance that everything site-specific comes from the user.
