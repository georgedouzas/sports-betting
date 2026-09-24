# Research: Surfaces that reach the whole library

## D1. The GUI is a quarter of the codebase and nothing tests it

**Decision**: delete `src/sportsbet/gui/` entirely.

Measured, not estimated:

| | GUI | CLI | Core (`datasets` + `evaluation`) |
| --- | --- | --- | --- |
| Lines | **1,951** | 504 | 5,892 |
| Tests | **none** | yes | yes |
| Sports reached | **1** (`DATALOADERS = {'Soccer': SoccerDataLoader}`) | 1 | 2 |

And the damning line, in `pyproject.toml`:

```toml
addopts = ["--cov", "--doctest-modules", ..., "--ignore=src/sportsbet/gui", ...]
```

A quarter of the shipped package has been **excluded by name** from pytest, from coverage, and from `--doctest-modules`
since the day it was written. The constitution calls the automated gate *non-negotiable*; one surface has never been
through it.

It also pins `reflex==0.7.0` — an **exact** pin it cannot safely move off, because nothing would catch a break — drags in
`reflex-ag-grid` and `nest-asyncio`, and needs a **Node toolchain**. `pyproject.toml` even carries `bandit` suppressions
(`B404`, `B603`, `B607`) that exist *only* because the GUI's launcher shells out to `node`. Deleting it makes the
security gate **stricter**.

**The footprint outside its own directory is tiny**: `pyproject.toml`, `README.md`, `docs/generate_api.py`, and
historical `CHANGELOG.md` entries (which stay — they are history). Nothing in `datasets` or `evaluation` imports it.

> **Trap for the implementer**: a naive `grep -r gui` matches the word "user **gui**de" in dozens of docstrings and doc
> pages. Those are false positives. Match on word boundaries, or you will "clean up" the user guide.

**Alternatives rejected**: *repair it* (bring it to parity with sources, keys and two sports — the largest untested
surface, and paid credentials in a local web UI is its own design problem); *spin it out to its own package* (defensible,
and still open to anyone who wants it — but it is not this library's job to carry it).

## D2. The root cause is the config contract, not the GUI

**Decision**: the CLI's configuration hands over a **configured dataloader instance**, not a class.

This is the real defect, and the GUI merely shares it.

`src/sportsbet/cli/_utils.py` asks the config for `DATALOADER_CLASS` — a **class** — and `PARAM_GRID`. Every command
then constructs `dataloader_cls(param_grid)`, **with no sources**. A class cannot carry a source, and a source is what
carries a credential.

The consequences, verified rather than reasoned:

- `BasketballDataLoader` has `DEFAULT_ODDS = None` and **raises** when constructed bare — *"no free default for odds"*.
  So **the CLI cannot reach basketball at all**: not the EuroLeague, not the NBA.
- `OddsApi` needs a key, and a class cannot hold one, so **the CLI cannot buy odds for any sport**, including soccer.

The CLI is stranded on soccer + football-data — one of the four sources, one of the three leagues. Every capability of
the last three releases is invisible to it.

**The fix is one sentence in the contract:**

```python
# before — a class, so no source, so no key, so no basketball
DATALOADER_CLASS = SoccerDataLoader
PARAM_GRID = {'league': ['England'], 'year': [2025]}

# after — a dataloader, so any source, so any key, so any sport
DATALOADER = BasketballDataLoader(
    param_grid={'league': ['NBA'], 'year': [2026]},
    stats=NBAStats(),
    odds=OddsApi(key=os.environ['ODDS_API_KEY']),
)
```

`get_dataloader_cls` + `get_param_grid` collapse into one `get_dataloader`, validated with `isinstance(...,
BaseDataLoader)`. Consumers: `_data.py` (`params`, `prepare`, `odds_types`, `training`, `fixtures`) and `_betting.py`
(`backtest`, `bet`). **`get_bettor` also checks for `DATALOADER_CLASS`** and is easy to miss.

**The config is Python, so the key comes from the environment** — `os.environ[...]` — and never has to be written into a
file that could be committed (FR-004).

**Two subtleties:**

1. **`params` must work before a `param_grid` exists.** It is the command that tells you *what to put in* a `param_grid`,
   so `param_grid=None` must remain a valid config. It reads `DATALOADER.sources` and asks the source.
2. **`PARAM_GRID` is removed, not deprecated.** A pre-1.0 library gets one clean break; keeping both alive would mean two
   ways to say the same thing, one of which cannot express half the library. An old config must fail with a message that
   **says what to change**.

**A note on tests.** `tests/conftest.py` holds a `CONFIG` string using the old contract, and it **must** change. Unlike
feature 004 — where an edited test would have meant the abstraction leaked — this feature's *purpose* is to change the
CLI contract, so the tests *of that contract* change with it. What must not change is any test of the **core**.

## D3. How an assistant names a dataloader: it writes a config, and the CLI's contract is reused

**Decision**: the MCP tools take a **path to a config file** — the *same* config module the CLI reads — not JSON
describing a dataloader.

The problem: MCP tools take JSON arguments, but a dataloader is a live object graph — a sport, its sources, each source's
settings, and a credential. There were two ways to bridge that.

**(a) The tools take a config path.** ✅ **Chosen.**

- **One contract, one code path.** The CLI and the server read the *same* config through the *same* `get_dataloader`. They
  cannot drift, which is precisely the disease this feature exists to cure. Building a second, JSON-shaped way to
  describe a dataloader would recreate the parity bug inside the fix.
- **The credential never enters a tool argument.** It stays in `os.environ`, read by the config. An API key passed as a
  JSON tool parameter would be logged, echoed in transcripts, and shown in the assistant's context — a genuinely bad
  place for a paid credential to live.
- **The assistant can *write* the config.** Writing a Python file is something an assistant is good at. It composes: the
  agent writes the config, then drives it, and the user can read, edit, keep and re-run that file. It is not a throwaway.
- **Every source is reachable for free**, including one added tomorrow, because the config is Python. A JSON schema would
  have to enumerate the sources and would need extending for every new one.

**(b) The tools take JSON describing sport / sources / param_grid.** Rejected. It needs a schema that must be extended
for every new source; it puts the credential in the tool call; and it duplicates the CLI's contract in a second dialect
that will drift from it.

## D4. An assistant must not be able to spend a user's money by accident

**Decision**: the tool that fetches metered data **cannot spend without being told the price**. `prepare` takes an
explicit confirmation of the cost, and refuses if it is absent or does not match what the estimate says.

This is a **safety property, not a convenience**, and it is the one thing about handing a betting library to an
autonomous agent that genuinely worries me. An odds vendor charges per request. A single NBA season quotes at **17,722
credits** — measured, not hypothetical. An agent that "just prepares the data" to be helpful can spend real money in a
single tool call, and the user finds out afterwards.

The library already makes this preventable, and for free: `prepare(dry_run=True)` performs the **two-stage plan** —
fetch the free statistics, derive the schedule, price the odds items **exactly** — and spends **nothing**. The estimate
is not a guess; it is the actual item list.

So the design is: the estimate is free and always available, and **the spending tool requires the caller to have seen
it**. An agent that skips the check gets an error, not a bill. What the confirmation looks like concretely — an explicit
`confirm_cost` argument that must equal the quoted figure — is settled in [contracts/mcp.md](contracts/mcp.md).

**Alternative rejected**: trusting the tool description to tell the model to check first. Prompt-level guardrails are not
guardrails. If the only thing between a user and 17,722 credits is a sentence in a docstring, the money will eventually
be spent.

## D5. The assistant goes outside the box, never inside it

**Decision**: the library exposes a surface an assistant can drive. It contains **no model, no model credential, and no
model call** (FR-014, SC-006).

The question that started this feature was whether to replace the GUI *with an agent*. The answer is yes to the agent —
and no to putting it in the package.

**Why not embed one:**

- It would drag in a **model credential**, a **model choice**, a **per-call cost** and **nondeterminism** — into a library
  whose entire value proposition is estimators that behave predictably inside a scikit-learn `Pipeline`. A nondeterministic
  dependency in a package that exists to be cross-validated is a category error.
- It would rot **faster** than the GUI did, and for the same reason: it would be a second product living inside the first.
- "A betting library that generates advice with a language model" is a liability nobody needs in the box.

**What is actually needed** is for the library to be *legible* to an assistant, and it very nearly is already: 37 public
names, complete type annotations, real docstrings, a working CLI. The MCP server is a **thin** adapter over that. The
user brings their own assistant; every model concern stays on their side of the line.

## D6. Two breaking changes, stated plainly

**Decision**: ship both breaks at once, in one release, each with a before/after in the changelog.

1. **`sportsbet-gui` and the `gui` extra are gone.** Also gone: `reflex`, `reflex-ag-grid`, `nest-asyncio`, and the Node
   toolchain.
2. **`DATALOADER_CLASS` + `PARAM_GRID` become `DATALOADER`.** An old config fails with a message naming the change.

A pre-1.0 library is allowed one clean break. Two half-breaks — a deprecation cycle for the config, a soft-deprecation
for the GUI — would mean carrying, for months, a config dialect that cannot express half the library. That is worse for
users than one honest release note.
