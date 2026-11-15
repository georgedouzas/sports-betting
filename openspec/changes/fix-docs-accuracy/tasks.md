# Tasks

The group ends green on `pdm run formatting`, `pdm run checks`, `pdm run docs build` and `pdm run tests`, read from the
nox session summary line, with the docs build run with no secret set.

## 1. Docs

- [x] 1.1 Rewrite the extraction paragraph of `docs/overview/user_guide/dataloader.md` to say what is downloaded once
  and reused. Verify against the `Extraction reuses the snapshots it holds` requirement of
  `dataloaders/data-extraction`.
- [x] 1.2 Replace "closing" in the descriptions of `FootballDataOdds`: its docstring in
  `src/sportsbet/sources/_odds/_football_data.py`, `docs/overview/user_guide/sources.md`,
  `docs/overview/user_guide/dataloader.md`, `docs/examples/sources/plot_football_data.py` and
  `docs/examples/sources/plot_odds_api.py`. Verify that grep for `closing` in those files finds only the fallback.
- [x] 1.3 Give both classes of the custom source example in `docs/overview/user_guide/sources.md` the signature
  `list_index_items(self, selection=None)`, and run the block with markdown-exec so the build proves it works.
- [x] 1.4 Compare `docs/interfaces/agents.md` with the tools `sportsbet-mcp` registers and correct what differs.
- [x] 1.5 Commit as `docs: ...`.
