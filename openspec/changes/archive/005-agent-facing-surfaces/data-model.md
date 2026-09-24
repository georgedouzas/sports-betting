# Data Model: Surfaces that reach the whole library

No data changes. The store, the schemas, the snapshots and the estimators are untouched — `datasets/**` and
`evaluation/**` must show a zero diff. What changes is a **contract**, and it is worth writing down because one word in
it was the whole bug.

## The configuration

| | Before | After |
| --- | --- | --- |
| What is handed over | `DATALOADER_CLASS` — a **class** | `DATALOADER` — a **built dataloader** |
| The selection | `PARAM_GRID`, a separate variable | `param_grid=...`, a constructor argument |
| Can it carry a source? | **No** | Yes |
| Can it carry a credential? | **No** (a class holds nothing) | Yes, from `os.environ` |
| Sports reachable | **1** of 2 | **all** |
| Sources reachable | **1** of 4 | **all** |

A class is a *kind* of dataloader. A dataloader is a *configured* one. Everything the library gained when sources became
injectable lives in that difference, and the CLI was asking for the wrong one.

## The surfaces

| Surface | Reaches | Tested |
| --- | --- | --- |
| Python API | everything | yes |
| CLI | ~~soccer + football-data~~ → **everything** | yes |
| ~~GUI~~ | ~~soccer only~~ — **deleted** | ~~**no**~~ |
| MCP server | everything | yes |

The GUI's row is why it is going: it is the only one where "reaches least" and "tested never" met.

## The cost estimate

Not new — but this feature makes it a **precondition** rather than a courtesy.

| Property | Value |
| --- | --- |
| Exact? | Yes. It is the real item list, not a heuristic. |
| Free? | Yes. The free statistics are fetched, the schedule is derived, the odds items are priced. Nothing is bought. |
| Who must see it | **Any surface that can spend.** An assistant may not fetch metered odds without confirming the quoted figure. |

A single NBA season quotes at **17,722 credits**. That number is the reason this is a safety property and not a nicety.
