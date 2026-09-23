# Rule model: Constitution conformance

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Date**: 2026-09-21

This feature has no runtime data. What it has is a set of rules, each of which has to become something a program can
decide. This is that model: the entities the conformance check reasons about, the predicate each rule becomes, and the
number of places the tree breaks it today.

## Entities

### Name

A function, class, or module-level assignment defined in the source tree.

| Field | Values | Notes |
| --- | --- | --- |
| identifier | text | The name as written |
| privacy | `public`, `private` | Private when the identifier starts with one underscore and is not a dunder |
| kind | `module`, `class`, `function`, `method`, `constant` | A method is a function defined inside a class |
| owning module | path | Where it is defined |
| owning package | path | The directory whose `__init__` could re-export it |
| exported | true, false | True when the owning package's `__init__` imports it |
| docstring | text or none | |
| blocks | set of `Args`, `Returns`, `Raises`, `Attributes`, `Examples` | Sections present in the docstring |
| offline | true, false | Whether the name can run without a credential or the network |

### Module

A file in the source tree.

| Field | Values | Notes |
| --- | --- | --- |
| path | path | |
| privacy | `public`, `private` | Private when the file name starts with `_`, `__init__` and `__main__` aside |
| is surface | true, false | True for `__init__.py` |
| imports | list of (source module, name, guarded) | `guarded` is true inside an `if TYPE_CHECKING:` block |
| comments | list of (line, text) | Excluding the licence header |

### Package

A directory with an `__init__.py`.

| Field | Values | Notes |
| --- | --- | --- |
| path | path | |
| re-exports | set of names | Imported by its `__init__` |
| declares `__all__` | true, false | |
| parent | package or none | A package and its subpackages are one package for the reaching rule |

### Suppression

An inline exemption from a lint or type rule.

| Field | Values | Notes |
| --- | --- | --- |
| module, line | path, integer | |
| rule code | text or none | |
| reason | text or none | |
| occurrences of the same code | integer | Two or more means it belongs in the configuration |

## Rules as predicates

Each row is one rule, the predicate the check evaluates, and the count of places the tree fails it today. The counts
are the baseline the specification's success criteria measure against.

| # | Rule | Predicate over the model | Fails |
| --- | --- | --- | ---: |
| 1 | Everything carries a docstring | `docstring is not none` | 24 |
| 2 | A public name carries its blocks | public implies the blocks its signature calls for | 120 |
| 3 | A private name carries the line alone | private implies `blocks == {}` | 0 |
| 4 | An empty block is omitted | no `Args` without parameters, no `Returns` without a return | 8 |
| 5 | A name a package keeps is private | `exported == false` implies private | 76 |
| 6 | No reaching past a surface | an import's target is in the own package tree, or neither side is private | 0 |
| 7 | A public name leaves through the surface | only an `__init__` imports a public name from a private module | 0 |
| 8 | A package declares `__all__` | `declares __all__` | 0 |
| 9 | An `__init__` holds no logic | its body is imports and `__all__` only | 0 |
| 10 | No type-checking import guard | no import has `guarded == true` | 4 |
| 11 | Only licence and suppression comments | `comments` is empty once suppressions are removed | 0 |
| 12 | A suppression says which rule and why | both fields present | 0 |
| 13 | A recurring suppression is configured | no rule code appears inline twice | 6 |
| 14 | An offline public name carries an example | public and offline implies `Examples` present | see below |
| 15 | A name that needs the feed carries none | public and not offline implies `Examples` absent | to be measured |
| 16 | An implementation module carries the header | not a surface implies the `# Author:` and `# License:` lines | 0 |

Every count above was produced by running the check, not estimated. Four of them corrected an estimate that was
wrong: rule 8 was reported as 6 because the estimate searched for `__all__ = [...]` and every package writes
`__all__: list[str] = [...]`, rule 11 was reported as 75 because the estimate counted the `# Author:` and
`# License:` header lines as explanatory comments, rule 5 was reported as 110 before the surface packages were
carved out and the annotated constants counted, and rule 1 was reported as 31 before constructors were exempted.

Rule 14's count depends on where the offline boundary falls inside `sources`, which [research.md](./research.md)
settles per name rather than per package. The floor is 27, the names in `core`, `dataloaders`, and `evaluation` that
carry no example today.

## What the check cannot decide

Three of the rules need a human, and the check must not pretend otherwise. It reports them rather than failing on
them.

- **Whether a summary line says what the thing does.** A check can see that a line exists, is one line, and does not
  start with `Implements the` or `This function`. It cannot see whether the sentence is true.
- **Whether a name is honest.** Rule 5 is decidable because it is about underscores. The Naming section's rules, that
  a verb is honest and a constant is named for what it holds, are not.
- **Whether a name can run offline.** The check reads a list the repository maintains, since the answer depends on
  what a source does, not on how it is written.

## State transitions

A name moves through this feature in one direction, and each pass is a transition that leaves the tree green.

```text
undocumented ──▶ has a summary line ──▶ has the blocks its privacy calls for ──▶ has an example
     (31)                 (111)                      (rules 2 and 3)              (rules 14 and 15)

apparent public ──▶ declared surface ──▶ private where the package keeps it
      (192)              (__all__, 6)              (110 renames)
```
