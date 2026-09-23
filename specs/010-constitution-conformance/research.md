# Research: Constitution conformance

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Date**: 2026-09-21

The specification left no question open, so this phase answers the questions the plan raised instead: how each rule is
checked once the sweep is done, how big each pass really is, and in what order the passes run.

## R1: How the rules are enforced after the sweep

**Withdrawn.** This section proposed a conformance check under `tools/`, run from the `checks` session. The
maintainer did not ask for a new tool and removed it. The rules are enforced by review and by what the existing gate
already covers, `ruff` with the `D` rules for docstring shape, `interrogate` for docstring coverage, `mypy`, and the
doctest run for the examples.

The sweep is therefore a one-off. What holds it in place afterwards is review, as it was before.

## R2: Which public names can carry a runnable example

**Decision**: a name can carry a runnable example when it can run offline with no credential. In this tree that is the
39 public names in `core`, `dataloaders`, and `evaluation`, plus the part of `sources` that works against the sample
sources. Everything that reads a live feed or needs a key is shown as reference code in the guide instead.

**Rationale**: the rule is already stated twice, that a network-touching class carries no runnable example, and that a
capability which cannot run offline is reference code in the guide. The question is only which names fall on which
side, and the answer follows the packages.

| Package | Public names | With an example today | Can run offline |
| --- | ---: | ---: | --- |
| `core` | 4 | 0 | yes |
| `dataloaders` | 13 | 4 | yes |
| `evaluation` | 22 | 8 | yes |
| `sources` | 64 | 18 | partly, through `SampleSoccerStats` and `SampleSoccerOdds` |
| `execution` | 39 | 3 | no, every path needs a venue and a credential |
| `cli` | 28 | 0 | as commands, not as calls |
| `mcp` | 22 | 0 | as tools, not as calls |

The sample sources ship a frozen season for exactly this purpose, and 18 examples in `sources` already use them, so
the boundary inside `sources` is per name, not per package: a name that reads the live feed is exempt, a name that
shapes data a sample source produced is not.

`cli` and `mcp` are the awkward pair. Their public names are command callbacks and tool functions, and a doctest that
drives a command through a runner is a test wearing an example's clothes. They are documented by the guide, which
already shows the commands as a user types them.

**Alternatives considered**:

- An example on all 192 names, including the venue and the live-feed paths, with a skip directive. Rejected, since the
  Documentation section forbids exactly that, an example that does not run.
- Recorded payloads to make live-feed examples runnable. Rejected for this feature. It would make examples depend on
  fixtures that drift from the feed, and it is a change to how the library is tested, not a conformance sweep.

## R3: The order of the passes

**Decision**: run the passes in this order, which is not the priority order in the specification.

1. Remove the four type-checking guards and add `__all__` to the six packages that lack it. Small, mechanical, and it
   gives the conformance check something true to assert.
2. Land the conformance check itself, failing only on the rules already satisfied, then widen it as each pass lands.
3. Rename the 110 internal names to carry an underscore.
4. Write the docstring blocks on the public names, and the summary lines on the 31 that carry none.
5. Remove the 75 comment lines.
6. Write the examples.
7. Fix the Project Profile and the front page.

**Rationale**: priority orders value, and this orders work. The two differ in three places.

- The guards and `__all__` come first because the check cannot assert the surface rule until each package declares its
  surface.
- The rename comes before the docstrings, since a renamed name appears in the docstrings and the references of other
  modules, and writing the blocks first means editing them again.
- The comments come after the docstrings because a comment usually disappears when the docstring above it is written,
  so doing them together avoids two passes over one line, which is what the specification already says.

**Alternatives considered**: strict priority order, P1 through P5. Rejected because it writes 111 docstring blocks
against names that the rename then changes.

## R4: Renaming the 110 internal names

**Decision**: rename with a script that rewrites definition and references together, one package at a time, with the
gate run between packages.

**Rationale**: the names are internal by definition, since no package re-exports them, so no user can be reading them.
The risk is not breakage in the field, it is a missed reference inside the tree, which the test suite and the import of
every module catch immediately. One package at a time keeps each failure small.

Three cases need a human eye rather than a script:

- A name that is also a name in a dependency, where the rename changes a word the reader associates with the
  dependency's concept.
- A name that appears in a string, a configuration key, or a documentation page, where a blind rewrite would change a
  value rather than an identifier.
- A class attribute or a dataclass field, where the leading underscore changes what the class exposes, which is a
  behaviour change rather than a rename. Those stay public and their package re-exports them, or they move.

**Alternatives considered**: renaming by hand. Rejected at 110 names. Renaming all at once. Rejected, since one failing
import in a tree-wide commit tells you nothing about which of 110 caused it.

## R5: Whether the type-checking guards are load-bearing

**Decision**: remove all four. They are not.

**Rationale**: measured rather than assumed. The package import graph is acyclic.

```text
cli          -> core, dataloaders, evaluation, execution
dataloaders  -> core, sources
evaluation   -> core
execution    -> core, dataloaders, evaluation
mcp          -> dataloaders, evaluation, execution
sources      -> core
```

`dataloaders` and `evaluation` never import `execution`, so the two guards in `execution/_event.py` and the one in
`execution/_schedule.py` guard nothing. `execution/_browser.py` does not import `execution/_factory.py`, so the fourth
guards nothing either. Each becomes a plain import.

**Alternatives considered**: keeping them for import cost. Rejected. The modules they name are imported at runtime
elsewhere in the same package, so nothing is saved.

## R6: The surface declarations, corrected

**This section was wrong and is kept to record the correction.** It claimed that only `core` declares `__all__`. In
fact all seven packages declare it, as `__all__: list[str] = [...]`, and the measurement behind the claim searched for
`__all__ = [...]` alone. The conformance check reports zero failures for that rule.

Two further estimates in this feature were wrong the same way, by searching for a spelling the repository does not
use.

- The 75 comment lines User Story 3 was written to remove are the `# Author:` and `# License:` header lines that 37
  modules carry, two lines each. The source carries no explanatory comment at all, so the story has nothing in it.
- The claim that no module carries a licence header searched for `Copyright`, `SPDX`, and `Licensed under`, and missed
  `# License: MIT`. Thirty-seven modules carry the header, eleven of the rest are package surfaces the rule exempts,
  and the five implementation modules that lacked it now carry it.

The lesson is the one the check exists for: a count produced by a search for a guessed spelling is not a measurement.
