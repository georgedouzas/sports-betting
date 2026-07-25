# Python code conventions

How code in this repository is written. The rules are generic Python and port to any project. Examples are drawn from
this codebase, but the rules stand on their own. Copy this file into another project and it holds.

Read the gate first. It is machine enforced, so it is not negotiable. The rest is the taste the gate cannot check.

## The gate

Every change passes the same automated bar before it merges. None of it is advisory.

- `black`, line length 120, string normalization off.
- `docformatter`, wrapping docstrings to 120.
- `ruff`, the configured rule set (see `pyproject.toml`). Notable families on: `ANN` full annotations, `D` pydocstyle
  google, `EM`/`TRY` exception style, `PL` pylint, `SIM`, `PTH`, `PD`, `RUF`, `S` bandit-style security.
- `mypy`, clean, with `warn_unused_ignores` on. A `# type: ignore` that stopped being needed is an error.
- `interrogate`, docstring coverage. Every public and private module, class and function has a docstring.
- `bandit` and `pip-audit` for security.
- `pytest` with branch coverage, doctest modules, randomized order.

A rule is disabled at the source only with a one-line comment saying why, and that is the exception. Fix the finding,
do not silence it.

## Files and layout

A module reads top to bottom in one order.

```python
"""One line saying what this module does."""

# Author: ...
# License: MIT

from __future__ import annotations

import io                       # standard library
from pathlib import Path

import pandas as pd             # third party

from .._base import BaseSource  # first party

DATA = Path(__file__).parent / 'data'   # module constants, UPPER_CASE
Scheduled = list[tuple[Intent, pd.Timestamp]]   # type aliases

def _leaf_helper(...): ...      # a name is defined before it is used, so helpers come first
def entry_point(...): ...       # the function the module exists for comes last
```

- `from __future__ import annotations` is always first after the docstring and header.
- Imports are grouped standard library, third party, first party, each alphabetized. `ruff` sorts them, so do not sort
  by hand.
- Constants and type aliases live at the module top, not inside functions.
- Functions read top to bottom in dependency order: a name is defined before it is used, so the small helpers come
  first and the function the module exists for comes last.
- One module is one concern. When a file grows two concerns, split it. Sources split into `_stats/` and `_odds/`
  because a statistics feed and an odds feed are two concerns, not one.
- A definition lives in the module that owns it. A type the fetch layer produces lives with the fetch layer, not in a
  module that imports it. When two modules end up needing each other, that is a cycle and a design smell: move the
  shared definition down to the layer both import, or merge the two.
- A base module is self-contained: it imports no sibling module. Its purpose is to be imported, not to import. If a
  base needs a sibling's code, that code belongs in the base, so merge rather than import. `_fetch` merged into
  `sources/_base` for this reason, so the base reads its own content instead of importing a sibling. A type-only alias
  from the package root, under `TYPE_CHECKING`, is not a sibling and is allowed.
- No import inside a function body to break a cycle: fix the cycle instead, by the rule above. The one lazy import that
  is allowed defers an optional dependency so it does not load unless used, like `playwright` behind the `execution`
  extra. That import carries a `# noqa: PLC0415` and a reason.

## Naming

- Modules that are implementation are private, prefixed `_`: `_resolver.py`, `_schedule.py`. The package `__init__`
  re-exports the public names.
- Private classes and helpers are prefixed `_`: `_SampleSource`, `_scheduled`.
- A function name begins with a verb, always. `count_common_prefix`, not `common_prefix_length`. `measure_names_similarity`,
  not `names_similarity`. `build_roster`, not `roster`. A name that begins with a noun describes a value, and a function
  is not a value.
- The verb and its object say exactly what the function does or returns. A function that returns the length of a common
  prefix is `count_common_prefix`, not `extract_prefix`, which promises a string it does not return. Avoid empty verbs
  that say nothing: `process`, `handle`, `manage`, and `transform` with no object. `normalize_identity` says what the
  transform is; `transform_identity` does not.
- State learned at runtime carries a trailing underscore, the scikit-learn convention: `odds_type_`, `context_`,
  `target_event_status_`. A reader can tell a fitted attribute from a constructor argument at a glance.
- Names come from the domain. A `venue`, an `intent`, a `receipt`, a `snapshot`. The vocabulary of the problem, used
  consistently, is most of what makes code readable.
- Class attributes fixed at class level are `ClassVar`: `name: ClassVar[str] = 'sample_soccer'`.

## Docstrings

This is where style drifts most, so it is the most specific.

The summary line is one line, imperative mood, and says what the thing does.

```python
✓ """Return the odds with the identity of the matches they belong to."""
✓ """Read a secret from the variable named for it, never from an argument."""
✓ """Place the value bets of the upcoming matches, one match at a time."""

✗ """Implements the resolver that reconciles the matches of one source with those of another."""
✗ """This function is responsible for returning the odds ..."""
✗ """Resolver."""
```

`Implements the ...`, `This function ...`, `A class that ...` are meta narration. Say what it does, in the imperative,
as if completing the sentence "This function will ...".

For most functions the one line is the whole docstring. A private helper and an ordinary function get a single line and
nothing else.

```python
✓ def _read_urls_content(urls):
      """Return the content behind each URL, from disk for a `file://` URL and over the network for the rest."""

✗ def _read_urls_content(urls):
      """Return the content behind each URL, from disk for a `file://` URL and over the network for the rest.

      A source whose feed ships with the library reads its files exactly as another source reads a remote feed.
      """
```

A body paragraph, or an Args/Returns block, appears only on a public entry point or a public class whose parameters or
fields are not obvious from the signature: `execute`, `resolve_odds`, `RawItem`. Never on a private helper. When a body
is warranted, keep it to a few sentences.

```python
def betting_moment(dataloader, kickoff):
    """Return when the bet goes on for a match.

    A live model bets at the kickoff plus the time into the match it was fitted for. Any other model bets at the
    kickoff.
    """
```

Rules for the body:

- Never describe what the thing does not do. "This does not validate the input" is noise. If a defensive note is truly
  load bearing, state the positive: "The caller validates the input."
- Never restate the code. If the body is the function in English, delete it.
- No essays. A module docstring is a line and maybe a short paragraph, not three. A function body is two or three
  sentences.
- No editorializing. The docstring documents, it does not sell or reflect.

Arguments, returns and raises use google sections, each entry terse.

```python
    Args:
        dataloader:
            The dataloader the model was fitted on.
        window:
            How far ahead to reach. `None` reaches every upcoming match.

    Returns:
        moment:
            When the bet goes on.

    Raises:
        UnmatchedError:
            When more matches go without odds than the tolerance allows.
```

Public API carries a runnable example, checked by the doctest run. Network touching classes do not, since a doctest
must not reach the network.

```python
    Examples:
        >>> from sportsbet.execution import CredentialRef
        >>> CredentialRef('VENUE_API_KEY').var
        'VENUE_API_KEY'
```

## Comments

There are almost none. The code says what it does through its names, and the docstring says why. An inline comment that
explains the next line is a sign the line or its names are unclear, so fix those.

The only comments in source are the license header and, rarely, a `# noqa`/`# type: ignore` with a reason.

```python
✗ # loop over the intents and place each one
   for intent in intents:

✓ from playwright.async_api import async_playwright  # noqa: PLC0415  (lazy: optional extra)
```

## Typing

- Everything is annotated: arguments, returns, attributes. `ruff ANN` and `mypy` enforce it.
- Methods annotate `self`: `def place(self: BaseVenue, intent: PlacementIntent) -> PlacementReceipt:`.
- Prefer precise types. A mapping keyed by a tuple is `dict[tuple[str, str, str], float]`, not `dict`.
- Name a repeated or complex type with an alias at the module top: `Placer = Callable[[PlacementIntent, BrowserSession], Awaitable[PlacementReceipt]]`.
- Use `TYPE_CHECKING` for imports needed only for annotations, to keep import time and cycles down.
- A `# type: ignore` names its code and is removed the moment it stops being needed.

## Errors

The message is a variable, then it is raised. `ruff EM`/`TRY` require this.

```python
✓ msg = f'`{venue}` is not a venue and is not a browser session.'
   raise SelectionError(msg)

✗ raise SelectionError(f'`{venue}` is not a venue ...')
```

- Raise a specific exception type, defined for the module or package: `SelectionError`, `ExecutionError`,
  `UnmatchedError`. Not a bare `Exception` or `ValueError` where a named one carries meaning.
- The message tells the reader what to do, not only what went wrong: name the variable that was missing, the value that
  did not match, the alias to add.
- Do not catch and swallow. Catch narrowly, or let it propagate.

## Control flow

- Functions are small and do one thing. When a function needs a paragraph of docstring body to explain its branches,
  it is two functions.
- Return early. A guard clause at the top beats a nested `if` around the whole body.
- No deep nesting. Extract a helper before the third level of indentation.
- No cleverness that a plain loop would say more clearly. The reader is the priority.

## Tests

- The test tree mirrors the source tree. `sources/_stats/_nba.py` is tested by `tests/sources/stats/test_nba.py`.
- A test name is `test_<function under test>_<behavior>`. It begins with the name of the function it exercises, then a
  short behavior phrase. Keep the phrase terse and drop articles: the docstring carries the full sentence.

```python
✓ def test_resolve_odds_bridges_unpairable_club_with_alias(stats, odds):
      # names the function, short behavior

✗ def test_an_alias_bridges_a_club_the_pairing_cannot_place(stats, odds):
      # names neither the function nor points at it

✗ def test_resolve_odds_bridges_an_unpairable_club_with_an_alias_the_user_gives(stats, odds):
      # names the function, but the behavior phrase is a whole sentence
```

- The docstring is a single line starting with "Test". The name already carries the behavior, so the docstring restates
  it in one sentence and stops. A test never has a multi-line docstring body.

```python
✓ """Test resolving the odds drops a club the pairing cannot place rather than attaching a wrong one."""

✗ """Test resolving the odds drops a club the pairing cannot place.

  A vendor that carries a club the statistics do not have would otherwise attach its odds to the wrong match.
  """
```

- Fixtures are typed and small, and live in the nearest `conftest.py`.
- A test never imports a private name from a private module. It uses the public API, the way a user does. If a test
  needs an internal, the internal wants to be public, or the test wants to be at a different level.
- No test reaches the network. Use recorded payloads, a fake, or a locally served page. A live-feed test is marked and
  deselected by default.
- Assert on behavior and shape, not on incidental formatting.

## Public API

- A package exposes its surface through its `__init__`, with an explicit `__all__`. Everything else is private.
- The three surfaces (Python API, CLI, MCP server) expose the same capabilities. A capability reachable from one is
  reachable from all, and a test asserts the parity so it cannot drift.
- A credential is named, never passed. A function, a command flag or a tool argument takes the name of the variable
  holding the secret, and reads it where it is used. A secret never becomes an argument, a log line or a pickle.

## Project-specific conventions

Generic Python ends above. These hold in this codebase and in ones like it.

- Estimators follow the scikit-learn contract. The constructor stores its arguments unmodified and validates nothing.
  State learned in `fit` uses trailing-underscore attributes. Behavior is configured through explicit parameters, not
  hidden global state.
- Every DataFrame that crosses a public boundary is validated against an explicit `pandera` schema. Schema fields use
  the `required_col()` and `optional_col()` helpers. Data-shape assumptions are schemas, not scattered runtime checks.
- Time is always UTC. `date` columns are `datetime64[ns, UTC]`, durations are `timedelta64[ns]`.
- Anything that needs a credential or performs a real-world side effect lives behind an optional extra, never in the
  default install.
- Runnable examples under `docs/examples/` are part of the contract and stay working. They run against the sample data,
  offline.

## The short version

- Pass the gate. It is not optional.
- Docstring summary: one line, imperative, what it does. No `Implements`, no meta, no essay, no describing what it does
  not do.
- Names from the domain. Functions begin with a verb that says exactly what they do. Fitted state ends in `_`.
  Implementation is private.
- Full annotations. Named exceptions, message in a variable.
- Small functions, early returns, no explanatory comments.
- Tests mirror the source, name the function under test and the behavior (`test_<function>_<behavior>`), keep a
  one-line docstring, use the public API, never touch the network.
