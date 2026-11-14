## REMOVED Requirements

### Requirement: Unmatched tolerance gate
**Reason**: Reconciliation now pairs stats and odds with `resolve_odds(stats, odds, aliases)`, which pairs by fuzzy name
and drops an unmatched match silently. `resolve_odds` no longer read `max_unmatched_rate`, so the parameter was inert.
It is removed from the `DataLoader` constructor, `build_dataloader`, the `--max-unmatched-rate` command line option and
the MCP tool signatures, together with the `ReconciliationReport`, `UnmatchedError` and `resolve` documentation and the
`reconciliation_` attribute the `DataLoader` docstring described but never set.
**Migration**: Remove any `max_unmatched_rate` argument and any `--max-unmatched-rate` option. Supply team aliases to
`resolve_odds` where a pairing fails.

### Requirement: Reconciliation report
**Reason**: The `ReconciliationReport` type and the `reconciliation_` attribute were removed with the tolerance gate.
The dataloader never set `reconciliation_` once `resolve_odds` took over the pairing.
**Migration**: Compare the matches in the training data with the statistics to count the unmatched ones, and supply
team aliases to `resolve_odds` where a pairing fails.
