# Proposal

## Why

The user guide states three things the code does not do.

- `docs/overview/user_guide/dataloader.md` says each extraction call downloads afresh. A later training or exploration
  extraction with the same `odds_type` reuses the snapshots the object holds and fetches nothing, as the
  `dataloaders/data-extraction` spec states.
- `FootballDataOdds` is described as carrying the pre-match closing prices, in its own docstring, in
  `docs/overview/user_guide/sources.md`, in `docs/overview/user_guide/dataloader.md`, and in the gallery examples
  `plot_football_data.py` and `plot_odds_api.py`. It reads the `Avg` and `Max` columns, which the feed records before
  the closing line. It falls back to the closing columns only where those are missing.
- The custom source example in `docs/overview/user_guide/sources.md` defines `list_index_items(self)`. The contract is
  `list_index_items(self, selection=None)`, and the dataloader passes the selection, so a source copied from the guide
  fails with `TypeError`.

## What Changes

- The extraction paragraph says the dataloader downloads on the first extraction and reuses what it holds after that,
  and that a different `odds_type` loads again.
- Every description of `FootballDataOdds` says the prices are pre-match prices recorded before the closing line, with
  the closing prices used only to fill a gap.
- The custom source example takes `selection=None` in both classes, and the page shows it running.
- `docs/interfaces/agents.md` is checked against the current MCP tool set and brought up to date.

## Non-goals

- Changing any behaviour. Docs and docstrings only.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. The specs already state the true behaviour.

## Impact

- Docs: the pages and examples named above, and the docstring of `FootballDataOdds`.
