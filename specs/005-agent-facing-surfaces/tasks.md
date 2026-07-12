---

description: "Task list for surfaces that reach the whole library"
---

# Tasks: Surfaces that reach the whole library

**Input**: Design documents from `/specs/005-agent-facing-surfaces/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[contracts/cli.md](contracts/cli.md), [contracts/mcp.md](contracts/mcp.md)

**Tests**: Required. The surface being deleted had none — that is why it is being deleted. The surface replacing it is
tested, and the safety property that stops an assistant spending a user's money is tested hardest of all.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelisable (different files, no dependency on an incomplete task)
- **[Story]**: US1–US3 from [spec.md](spec.md)

## The rule that governs this feature

**THE CORE MUST NOT CHANGE.** These must show a **zero diff**:

```text
src/sportsbet/datasets/**
src/sportsbet/evaluation/**
```

This is a change to how the library is *reached*, not to what it *does*. If the core needs to change — **stop and
report, and say why**.

**But `tests/conftest.py` and the CLI's tests DO change, and that is correct.** This feature's *purpose* is to change
the CLI's configuration contract, so the tests *of that contract* change with it. (Contrast feature 004, where an edited
test would have meant the abstraction leaked. Know which kind of feature you are in.) What must not change is any test of
the **core**.

## Conventions

- No new **required** runtime dependency. `mcp` is an **optional** extra, exactly as `gui` was — the number of extras
  does not even change.
- No test touches the network (socket guard). No executable doctest on a network-touching class.
- No explanatory inline comments. Short Google-style docstrings.
- Flat test files. Never import a private name from a private module in a test.
- Line length 120, `skip-string-normalization`.

---

## Phase 1: Delete the GUI (US3 — P1)

**Goal**: after this phase, **the test run excludes nothing**, and the security gate is *stricter* than before.

> ⚠️ **A naive `grep -r gui` matches the word "user **gui**de"** in dozens of docstrings and doc pages. Match on word
> boundaries, or you will delete the user guide.

- [ ] T001 [US3] `git rm -r src/sportsbet/gui/` — 1,951 lines, zero tests, one sport.
- [ ] T002 [US3] `pyproject.toml`: remove the `gui` optional-dependency group (`nest-asyncio`, `reflex==0.7.0`, `reflex-ag-grid`) and the `sportsbet-gui = "sportsbet.gui.main:run"` entry point. The Node toolchain goes with them.
- [ ] T003 [US3] `pyproject.toml`: remove `--ignore=src/sportsbet/gui` from `addopts`. **This is the headline of the phase** — a quarter of the package has been exempt from pytest, coverage and `--doctest-modules` since it was written (SC-004).
- [ ] T004 [US3] `pyproject.toml`: remove the `bandit` skips `B404`, `B603`, `B607` **and the comment above them** ("the GUI launcher runs `node`/`reflex`..."). Keep `B101`. **If `bandit` then flags something real, STOP AND REPORT — do not put the skip back.** The gate is supposed to get stricter here.
- [ ] T005 [P] [US3] `docs/generate_api.py`: remove the GUI.
- [ ] T006 [P] [US3] `README.md`: remove the GUI screenshots section. (The agent story arrives in Phase 4.)

**Checkpoint**: `addopts` has no `--ignore`, the full suite is green, `pdm run checks` passes with the stricter `bandit`
config, and the package installs with no Node.

---

## Phase 2: The CLI takes a configured dataloader (US1 — P1)

**Purpose**: the **actual defect**. The config hands over `DATALOADER_CLASS` — a *class* — so no source can be attached,
so no credential can be carried, so basketball is unreachable and paid odds are unreachable for every sport. See
[contracts/cli.md](contracts/cli.md).

- [ ] T007 [US1] `src/sportsbet/cli/_utils.py`: replace `get_dataloader_cls` and `get_param_grid` with a single `get_dataloader(mod)` that requires `DATALOADER`, validates it with `isinstance(..., BaseDataLoader)`, and returns **the instance**.
- [ ] T008 [US1] `src/sportsbet/cli/_utils.py`: a config still carrying `DATALOADER_CLASS` must fail with a message that **names the change** — `DATALOADER_CLASS`/`PARAM_GRID` are replaced by a configured `DATALOADER` — **not** an `AttributeError`, and **not** silence. This is a breaking change and it must behave like one (FR-002).
- [ ] T009 [US1] `src/sportsbet/cli/_utils.py`: `get_bettor` **also** checks for `DATALOADER_CLASS`. Update it. It is easy to miss and nothing else will catch it.
- [ ] T010 [US1] `src/sportsbet/cli/_data.py`: update `params`, `prepare`, `odds_types`, `training`, `fixtures` to take the configured dataloader.
- [ ] T011 [US1] `src/sportsbet/cli/_betting.py`: update `backtest` and `bet`.
- [ ] T012 [US1] Keep `params` working when **nothing has been selected yet**: `param_grid=None` must remain a valid config, because `params` is *how a user learns what to put in a `param_grid`*. It reads `DATALOADER.sources` and asks the source's `available_params`, which is free.
- [ ] T013 [US1] `tests/conftest.py`: rewrite the `CONFIG` string from `DATALOADER_CLASS`/`PARAM_GRID` to a configured `DATALOADER`. **This edit is expected and correct** — the contract it tests is the one this feature changes.
- [ ] T014 [P] [US1] Test: a config using the **old** contract fails with the message that names the change.
- [ ] T015 [P] [US1] Test: a config whose dataloader carries a **credentialled source** works — the case the CLI has never been able to express.
- [ ] T016 [P] [US1] Test: `params` works with `param_grid=None`.

**Checkpoint**: the CLI prepares an **NBA** dataset with a paid odds key. It has never been able to do this. Verify with
`--dry-run`, which is **free**.

---

## Phase 3: The MCP server (US2 — P1)

**Purpose**: the surface that replaces the deleted one — and, unlike it, **is tested** (FR-009). See
[contracts/mcp.md](contracts/mcp.md).

**No model, no model credential and no model call enters the library** (FR-014). The assistant is a *consumer*.

- [ ] T017 [US2] `pyproject.toml`: add an `mcp` optional-dependency group (the `mcp` package) and a `sportsbet-mcp` entry point. `mcp` is **not currently installed** — add it to the lockfile with `pdm`, and make sure the test environment has it.
- [ ] T018 [US2] Create `src/sportsbet/mcp/`, a **peer** of `src/sportsbet/cli/`. It imports only the **public** API — the same discipline the CLI keeps, and the reason a surface can be added or removed without the core noticing.
- [ ] T019 [US2] Every tool takes a **config path** — the *same* Python config module the CLI reads (research D3). Not JSON describing a dataloader: one contract and one `get_dataloader` means the two surfaces **cannot drift**, and the **credential never enters a tool argument**, where it would be logged and shown in a transcript.
- [ ] T020 [US2] Implement the free tools: `available_params` (what leagues and seasons exist; downloads nothing) and `estimate_preparation` (what a preparation would fetch, and **exactly** what it would cost).
- [ ] T021 [US2] **The safety property, enforced in code.** `prepare` **refuses** unless the caller passes `confirm_cost` equal to the total the estimate quotes. A tool description asking the model to check first **is not a guardrail** — if the only thing between a user and 17,722 credits is a sentence in a docstring, the money gets spent (FR-007, SC-003).
- [ ] T022 [US2] A preparation whose total cost is **zero** — every source free, or everything already held — needs **no** confirmation. The rule exists to prevent surprise spending, not to add ceremony to a free download.
- [ ] T023 [US2] Implement the remaining tools: `extract_train_data`, `extract_fixtures_data`, `backtest`, `bet`.
- [ ] T024 [US2] Decide and record how a dataframe crosses the wire — an assistant cannot consume a pandas object. Records/JSON, and it must round-trip.
- [ ] T025 [P] [US2] `tests/test_mcp.py`: the tool set exists; a **free** preparation needs no confirmation; the tools reach every sport, because they reach the CLI's config.
- [ ] T026 [P] [US2] `tests/test_mcp.py`: **the refusals.** A metered preparation is refused with **no** `confirm_cost`; refused with the **wrong** `confirm_cost`, and the message **names the real cost**; **allowed** with the right one. No test touches the network.

**Checkpoint**: an assistant goes from *"what exists?"* to *"here are the value bets"* with no Python written by the
user, and **cannot** buy metered odds without the cost having been surfaced.

---

## Phase 4: Docs, and the breaking changes stated plainly

- [ ] T027 [P] `README.md`: add the agent story (`pip install 'sportsbet[mcp]'`) where the screenshots were.
- [ ] T028 [P] `docs/overview/user_guide/`: update the CLI page for the new config — show a **credentialled** source and read the key from `os.environ`. **The key must never be written into a file that could be committed** (FR-004).
- [ ] T029 [P] `docs/overview/user_guide/`: add a page for driving the library with an assistant, and register it in `properdocs.yml`.
- [ ] T030 `CHANGELOG.md`: **two** breaking changes, each with a before/after snippet — (1) `sportsbet-gui` and the `gui` extra are gone; (2) `DATALOADER_CLASS`/`PARAM_GRID` are replaced by a configured `DATALOADER`.
- [ ] T031 Re-run the public-API docs audit. Every public name must still have a runnable example (37/37 before this feature).
- [ ] T032 **Verify SC-007**: `git diff --stat -- src/sportsbet/datasets src/sportsbet/evaluation` is **empty**. Verify SC-004: `addopts` excludes nothing. Verify SC-006: no model dependency anywhere.

---

## Dependencies

- **Phase 1 is independent** and shippable on its own.
- **Phase 2 blocks Phase 3**: the MCP server reuses the CLI's `get_dataloader`, so the contract must exist first. That
  reuse is the whole point — a second way to describe a dataloader would recreate the drift this feature exists to cure.
- Phase 4 last.
- **The core must not move.** Its tests are checked at every phase boundary (SC-007).

## Gate at every phase boundary

```bash
pdm run formatting
pdm run checks
pdm run tests
```
