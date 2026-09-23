<!--
SYNC IMPACT REPORT
==================
Version change: 11.0.0 -> 11.0.1
Rationale: Say who counts as a client. The tests and the documentation are client code, so a name either of them
reaches is used and stays re-exported. Without that, the rule reads as though only another package counts, which would
have stripped eight names from the sources surface that the tests reach, among them the reconciliation helpers and the
source base. PATCH: the rule always meant this, and no code changes.

Modified sections:
  - Surface: the unused-export rule names the tests and the documentation as clients, beside the public-signature
    clause it already carried.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 10.3.0 -> 11.0.0
Rationale: A module constant is never private. It is `UPPER_CASE` whatever its reach, and it is kept out of the
surface by not being re-exported rather than by an underscore. The rules that turn an unexported name private no
longer reach a constant, since `_UPPER_CASE` reads as two conventions fighting, and the underscore says nothing the
missing re-export does not already say. MAJOR: the privacy rules are redefined for constants, and 63 names that
version 10.3.0 required to be private are required not to be.

Modified sections:
  - Structure: a module constant carries no leading underscore, whatever its reach.
  - Surface: the rule that an unexported name is private, and the rule that an unused export becomes private, both
    exempt constants. An unused export stops being re-exported either way.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 10.2.0 -> 10.3.0
Rationale: Say what a surface is for. A package re-exports only the names something outside it uses, since an export
nobody reaches for is a promise the library gained nothing by making, and a name in a public signature counts as used,
since a caller reaches it through that signature. A name nothing uses at all is removed rather than made private,
since code no caller reaches is code a reader still has to read. The `core` and `execution` packages prompted all of
it: eight of their 47 exports are reached from nowhere outside them, five of those are referenced nowhere at all, and
one names the type of a public parameter. MINOR: two new rules.

Modified sections:
  - Surface: a package re-exports only what something outside it uses, and a name nothing uses is removed.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 10.1.1 -> 10.2.0
Rationale: Order a module and a class by privacy. The private functions come before the public ones, and a class puts
its private methods before its public ones, so a reader meets the parts before the whole. It sits beside the
dependency-order rule, which already puts the small helpers first and the function the module exists for last, and
the two agree in almost every module. MINOR: one new rule.

Modified sections:
  - Structure: the private functions and methods come first.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 10.1.0 -> 10.1.1
Rationale: Correct two Project Profile bullets that did not hold. `SelectionError` is named as one of the four
exceptions the package raises, and no such class exists, so it is dropped. The 31 bare `ValueError` and `TypeError`
raises all sit in the dataloaders and the evaluation package, where the scikit-learn contract expects them for invalid
input, so the bullet says that rather than leaving them looking like violations. PATCH: the profile is corrected to
describe the repository.

Modified sections:
  - Project Profile: the named exceptions are the three that exist, and the estimator packages raise what the
    ecosystem contract expects.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 10.0.1 -> 10.1.0
Rationale: A command or a tool in a surface package carries its summary line alone. Its docstring is not developer
documentation, it is the text the runner shows, `sportsbet evaluation fit --help` for a command and the tool
description an agent reads for a tool. Putting an `Args` block there would print Google sections into help output and
into what every agent sees, which is a user-visible change. The runner documents the parameters itself, from the
options and the schema it holds. MINOR: one exception stated, and nothing that conformed stops conforming.

Modified sections:
  - Docstrings: a command or a tool in a surface package carries the one-line summary alone, beside the existing rule
    for a private name.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 10.0.0 -> 10.0.1
Rationale: Restore the licence header rule. Version 10.0.0 removed it on a measurement that searched for `Copyright`
and missed the `# Author:` and `# License:` lines that 37 of the 53 modules carry. The rule described the practice
correctly all along, and the 10 modules without a header are the package surfaces the rule already exempts. PATCH:
10.0.0 removed a rule on a false premise, so this says what the constitution always meant.

Modified sections:
  - Structure: the module order carries the licence header again, and the example carries its two header lines.
  - Comments: the licence header is a comment the rule allows again.
  - Surface: the `__init__` carries no licence header, which is why the surfaces are not the gap.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 9.1.0 -> 10.0.0
Rationale: Drop the licence header rule. The project licenses at the root, in a LICENSE file, and not one of its 53
modules carries a header, so the rule described a practice the repository never had and the Comments section then
carved an exception for a comment that does not exist. MAJOR: a rule is removed.

Modified sections:
  - Structure: the module order runs docstring, imports, constants, functions, with no header between them, and the
    example drops its copyright line.
  - Comments: the only comment left in source is a suppression.
  - Surface: the `__init__` rule no longer says it carries no licence header, since no module carries one.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 9.0.0 -> 9.1.0
Rationale: Carve the surface packages out of the re-export rule. A package that serves a runner rather than an
importer, the CLI and the MCP server here, exposes its entry point alone, and the names inside it are reached by the
runner that discovers them. A command function is named `fit` because the command is `fit`, so making it private to
satisfy a rule about importers would rename the command for no reader. MINOR: the rule's reach is stated, and nothing
that conformed stops conforming.

Modified sections:
  - Surface: a surface package re-exports its entry point alone, and the rule that a name a package keeps must be
    private does not reach the names inside it.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 8.0.3 -> 9.0.0
Rationale: Say what the import rule was reaching for. Privacy is about reaching past a surface, not about consuming
one, so a module may import a public name from another package's surface whatever its own privacy, and what it MUST
NOT do is reach past that surface for a private name. A name its own package never re-exports MUST be private, which
is what the old matching rule actually caught. The two call rules go with it, since a rule about what a private
function may call says the same thing one level down and contradicted the new import rule. A `TYPE_CHECKING` guard is
forbidden, since an import only a type checker sees papers over a cycle the way an import in a function body does.
MAJOR: the import rule is redefined and the call rules are removed.

Modified sections:
  - Surface: the privacy-matching rule becomes two rules, no reaching past a surface for a private name, and a name a
    package never re-exports is private. A package and its subpackages count as one package, so a subpackage reaching
    its parent's private module is not reaching past a surface. The two call rules are gone, since the import rules
    already decide it.
  - Surface: the example shows a private module consuming another package through its surface, and the
    counter-example shows a name reached past a surface beside an internal name left public.
  - Structure: `from __future__ import annotations` is for a forward reference, a `TYPE_CHECKING` guard is forbidden,
    the base-module rule drops its type-only-alias clause, and the example imports plainly.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 8.0.2 -> 8.0.3
Rationale: Strip the Project Profile of what the body already says. A profile instantiates a rule, it does not repeat
it, so the clauses that restated Contract, Naming, Library, Schema, Gates, Documentation, and Structure are gone and
only the concrete names, versions, and commands are left. PATCH: every removed clause is still stated in its own
section, so no rule changes.

Modified sections:
  - Project Profile: dropped the opening bullet that restated the preamble, the stored-parameters and
    trailing-underscore clauses that restate Contract and Naming, the agent-is-a-client clause and the parity-test
    clause that restate Library, the crosses-a-public-boundary clause that restates Schema, the broken-example clause
    that restates Documentation and Gates, the builder-lives-with-what-it-builds clause that restates Structure, the
    second statement of line length 120, and a rationale clause about composing with pipelines.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 8.0.1 -> 8.0.2
Rationale: Keep the docstring rules in the Docstrings section. The coverage rule moves there, and the Documentation
section stops illustrating itself with a docstring and its blocks, which the Docstrings section already shows. Its
example is a documentation page whose block the build executes, and its counter-example is the same page with a block
that never runs and an output pasted by hand. PATCH: one rule relocated and two examples replaced, no rule changed.

Modified sections:
  - Docstrings: gains the rule that every module, class, and function carries a docstring.
  - Documentation: loses that rule, and both its examples are documentation pages rather than docstrings.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 8.0.0 -> 8.0.1
Rationale: Drop the docstring style clause from the Documentation rule. The Docstrings section states the shape, the
Toolchain section requires the convention to be set once in the project configuration, and the Project Profile names
it as Google, so a third, vaguer statement was the duplication the Duplication section rules out. PATCH: a redundant
clause removed, no rule changed.

Modified sections:
  - Documentation: the coverage rule reads that every module, class, and function has a docstring, with no clause
    about which style, since three other places already answer that.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 7.0.0 -> 8.0.0
Rationale: Split the docstring rule by privacy. A public function and a public class carry the summary line and the
`Args`, `Returns`, and `Raises` blocks. A private function and a private class carry the summary line alone, and carry
none of the blocks. The 7.0.0 rule asked every function for the blocks, which loaded a private helper with a paragraph
where one line says it. MAJOR: the private half of the rule is redefined.

Modified sections:
  - Docstrings: the blocks are a public rule, and a private function or class carries the one-line summary alone.
  - Docstrings: the example shows a public function with its blocks beside a private helper with its one line, and the
    counter-example adds a private helper carrying blocks it MUST NOT.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 6.1.0 -> 7.0.0
Rationale: Every function and every class carries the full docstring, a one-line summary and the `Args`, `Returns`, and
`Raises` blocks, private helpers included. The rule that let most functions stop at the summary line, and that asked
for the blocks only from a public entry point or a public class, is gone. The Documentation example carried a docstring
with no blocks at all, which the Docstrings section already ruled out, so it is corrected. MAJOR: a rule is redefined,
and code that was compliant under 6.1.0 is not compliant now.

Modified sections:
  - Docstrings: the one-line-is-enough rule and the public-only rule are replaced by one rule binding every function
    and every class. A block that would stand empty is omitted, so a function with no parameter carries no `Args`.
  - Docstrings: the top-level `__init__` bullet no longer cites a one-line rule that no longer exists.
  - Documentation: its first rule covers every module, class, and function, public or private, and its example carries
    the `Args`, `Returns`, and `Raises` blocks. Its counter-example is now a docstring with none of them.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 6.0.0 -> 6.1.0
Rationale: Show every rule from both sides. Each section gains a `Counter-example:` block after its example, in the
shape the Types section was given, so the good form and the form it rules out sit apart rather than mixed in one block.
The code blocks carry no comment of their own any more, since a comment in an example is the thing the Comments section
forbids. A one-line explanation sits under each example and each counter-example instead, which is where the labels the
comments were doing belong. The body is rewrapped to the 120 characters the Style section requires. MINOR: 24
counter-examples and 48 explanations are new guidance, and no rule changes.

Modified sections:
  - Every section: the layout is now bullets, `Example:` with its explanation, `Counter-example:` with its explanation,
    then `Rationale:`.
  - Library, Style, Naming, Comments, Duplication, Credentials, Surface: the single example that held the good form and
    the bad form together is split into the two blocks.
  - Every code block: the `# good` and `# bad` labels and the explanatory comments are gone. The only comments left are
    the ones a rule is about, the license header, a `# noqa`, a `# type: ignore`, a doctest directive, and the comment
    that stands in for a name in the Comments counter-example.
  - Surface: the example labels its three cases with module docstrings rather than comments.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 5.2.1 -> 6.0.0
Rationale: Give every section the layout the Contract section already had, the rules as bullets, then the example,
then `Rationale:` on its own line with the paragraph under it. State every one of the 155 rule bullets with MUST or
MUST NOT, so a reader never has to judge whether a plain sentence binds. Thirteen sections that had no rationale gain
one. MAJOR: the preamble rule that reserved MUST for what the gate enforces and left the rest as plain statements is
redefined, and every bullet in the document is restated.

Modified sections:
  - Constitution: the preamble says every rule is stated with MUST or MUST NOT and binds equally.
  - Every section: the bullets are restated with MUST or MUST NOT, and the layout is bullets, example, rationale.
  - Style, Naming, Structure, Surface, Docstrings, Comments, Suppressions, Errors, Control Flow, Duplication,
    Credentials, Toolchain, Amendments, Versioning, Compliance, Guidance, Project Profile: gained a rationale.
  - Style: the opening paragraph becomes its first rule, since a section carries no prose of its own now.
  - Toolchain: the closing line about the Project Profile becomes its last rule.
  - Project Profile: gained an opening rule, a code example, and a rationale, and its facts are stated as rules.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 5.2.0 -> 5.2.1
Rationale: Correct the public half of the import rule. A public module imports public names from public modules, and
never out of a private module. The package `__init__` is the one exception, and the only place a public name leaves a
private module, which is what a re-export is. Everywhere else the name comes from the surface that re-exports it.
PATCH: 5.2.0 stated the intended rule wrongly, so this says what it always meant.

Modified sections:
  - Surface: the privacy rule reads public names from public modules, with the `__init__` exception on its own bullet.
    The example marks `from _a import b` and `from _a._b import c` as bad in a public module.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 5.1.0 -> 5.2.0
Rationale: State the import rule as one rule. An import matches the privacy of the module it sits in, so a private
module imports private names from private modules, and a public module imports public names from a public or a private
module, which is what a re-export is. The re-export rule is scoped to the public modules and the tests, since the
private rule already decides what a private module may do. Every example that contrasts a good form with a bad one now
marks which is which. MINOR: the import rules gain precision, and no rule is removed.

Modified sections:
  - Surface: the four import and call bullets become one privacy rule plus the two call rules, with a worked example
    of every good and bad import form. The re-export bullet names the public modules and the tests as its scope.
  - Types, Library, Style, Naming, Comments, Duplication, Credentials: the examples label the good and the bad form.
  - Versioning: its example shows this amendment's own bump.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 5.0.0 -> 5.1.0
Rationale: Give every section a code example, so a rule is shown as well as stated. Fix the language the maintainer's
pass left rough, fill the two empty rationales, and merge the sections that were saying one thing in two places. Module
Structure and Package Layering become Structure, and the engineering workflow folds into Gates, which drops the CI
sentence it repeated. Seven headings take a shorter name. MINOR: examples and two rationales are new guidance, and no
rule is removed.

Modified sections:
  - Every section: carries a code example after its rules.
  - Types: the silencing rule is stated plainly, and the empty rationale is filled.
  - Schema: the empty rationale is filled.
  - Contract: fixed the initialization typo, and learned state is described as learned rather than stored.
  - Structure: merges Module Structure and Package Layering.
  - Gates: absorbs Development Workflow & Quality Gates, dropping the duplicated red-CI-run rule.
  - Renamed for brevity: Public Surface to Surface, Don't Repeat Yourself to Duplication, Toolchain & Standards to
    Toolchain, Versioning Policy to Versioning, Compliance Review to Compliance, Runtime Guidance to Guidance, and
    Project Profile: sports-betting to Project Profile.
  - Toolchain: cites Library by its current name.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 4.0.3 -> 5.0.0
Rationale: Flatten the document to three heading levels. The title absorbs the Constitution section, so the preamble
bullets sit under it. The Core Principles and Code Conventions groupings are dropped, and every principle and
convention becomes a subsection of Engineering in its own right. Nothing nests four deep. No rule is added or removed.
MAJOR: a structural rewrite, and the heading path of every principle and convention changes.

Modified sections:
  - Constitution: merged into the title, which the maintainer renamed from Engineering Constitution.
  - Engineering: the Core Principles and Code Conventions headings are gone, and their members are subsections now.
    The Code Conventions lead-in is folded into the Engineering intro, so its rules are not lost.
  - Tests, the convention, is renamed Test Conventions, since it would otherwise sit beside the principle Tests &
    Doctest Discipline at the same level.
  - Development Workflow & Quality Gates and Compliance Review cite the conventions and the constitution rather than
    two headings that no longer exist.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 4.0.2 -> 4.0.3
Rationale: Rename the first section from Principles to Constitution, since the principles that govern the code now live
under Engineering and the section holds only what concerns this document. Its one Purpose subsection is dropped, so the
bullets sit directly under the heading rather than nesting a single child. PATCH: a renaming, no rule changed.

Modified sections:
  - Constitution: renamed from Principles, absorbed its Purpose subsection, and dropped the intro sentence that
    restated the first bullet.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 4.0.1 -> 4.0.2
Rationale: Audit what each section holds. The PR compliance rules sat in the engineering workflow while Governance held
the same rule in its own words, so they move into Compliance Review and are stated once. PATCH: one rule relocated and
one duplication collapsed, no rule changed.

Modified sections:
  - Development Workflow & Quality Gates: dropped the PR compliance bullet, which is governance, not workflow.
  - Compliance Review: holds the PR compliance rules now, with the reviewer duty folded into its first bullet.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 4.0.0 -> 4.0.1
Rationale: Move Writing Style into Engineering. It governs the prose documents, the docstrings, and the examples, which
are project artifacts, so it does not belong in a section that holds only what concerns the constitution itself.
Principles now holds Purpose alone, and its opening sentence no longer claims a scope it does not have. PATCH: one
subsection relocated, no rule changed.

Modified sections:
  - Principles: Writing Style moves out, leaving Purpose. The opening sentence drops the style clause.
  - Engineering: gains Writing Style between Core Principles and Code Conventions, and its opening sentence names it.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 3.0.1 -> 4.0.0
Rationale: Draw the three top-level sections along a cleaner line. Principles now holds only what concerns the
constitution itself, its purpose and the style it is written in, so the six principles that govern the code move into
the second section, which is renamed Engineering because it now holds the principles, the conventions, the toolchain,
and the gate. Comments & Suppressions splits into Comments and Suppressions, two different concerns that shared a
heading. The single Governance rules subsection splits into Amendments, Versioning Policy, Compliance Review, and
Runtime Guidance. No rule is added or removed. MAJOR: another structural reorganization, and the heading path of every
principle and of the governance rules changes.

Modified sections:
  - Principles: keeps Purpose and Writing Style only. Core Principles moves out.
  - Engineering: renamed from Practice, and now opens with Core Principles before Code Conventions, Toolchain &
    Standards, and Development Workflow & Quality Gates.
  - Code Conventions: Comments & Suppressions splits into Comments and Suppressions. Automated Quality Gates cites
    Suppressions by its new name.
  - Governance: Amendments, Versioning & Compliance splits into Amendments, Versioning Policy, Compliance Review, and
    Runtime Guidance, each a subsection of its own.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 3.0.0 -> 3.0.1
Rationale: Drop the roman numerals from the Core Principles headings, so every subsection in the document is named and
none is numbered. A principle is now cited by its name, which survives a reordering where a numeral does not. The four
Project Profile bullets that cited a numeral already carried the principle's name as their label, so the citation is
removed rather than reworded. PATCH: a non-semantic renaming, no rule changed.

Modified sections:
  - Core Principles: removed the `I.` to `VI.` prefixes from the six principle headings.
  - Toolchain & Standards: the optional-extra rule cites A Library, Not an Application by name.
  - Project Profile: the ecosystem contract, library, delivery surfaces, and schema validation bullets drop their
    trailing numeral citation, since each bullet label already names its principle.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 2.5.0 -> 3.0.0
Rationale: Consolidate the eight top-level sections into three, Principles, Practice, and Governance, so the document
has one place to look for why, one for how, and one for who decides. Every former top-level section survives as a
subsection, and the principles and the code conventions move one level down with it. No rule is added, removed, or
redefined. MAJOR: the versioning policy names a structural rewrite as MAJOR, and every section's heading path changes.

Modified sections:
  - Principles: new top-level section holding Purpose & Scope, Writing Style, and Core Principles I to VI.
  - Practice: new top-level section holding Code Conventions, Toolchain & Standards, and Development Workflow &
    Quality Gates.
  - Governance: keeps its intro, gains an Amendments, Versioning & Compliance subsection for its rules, and now holds
    the Project Profile as its closing subsection.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 2.4.2 -> 2.5.0
Rationale: Recast every rule-bearing section as a bullet list, one rule per bullet, so a reader can scan the rules
instead of digging them out of prose. No rule is added, removed, or redefined. The Writing Style gains a rule that
states this, so the document keeps following the style it prescribes. MINOR: one new style rule, nothing removed.

Modified sections:
  - Writing Style: added the rule that a set of rules is written as a bullet list, one rule per bullet.
  - Purpose & Scope, Core Principles I to VI, and every Code Conventions subsection: reformatted from paragraphs into
    bullet lists. Rule text is unchanged apart from splitting joined sentences into separate bullets.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 2.4.1 -> 2.4.2
Rationale: Refine the Writing Style. Sentences are a natural length, not chopped into short choppy ones. PATCH.

---- history ----
Version change: 2.4.0 -> 2.4.1
Rationale: Clarify the example rule. A capability that cannot run offline, such as placing a real bet, is shown as
reference code in the guide, not as a runnable gallery example. PATCH: a clarification, nothing removed.

---- history ----
Version change: 2.3.0 -> 2.4.0
Rationale: Rewrite the Writing Style rules for plain, human, structured prose, and forbid the clever, inverted,
passive-for-effect, idiomatic, and defensive style the old rules allowed. Require that every documentation and docstring
example runs, proven by the build, with no un-runnable demo and no example that needs a secret or the network. Record
the 0.15.0 release lessons: a locally set secret can hide a broken example, so the build runs the examples with no
secret, and the release keeps `development` a superset of the released `main` with no version-tag collision. MINOR:
strengthened guidance, nothing removed.

Modified sections:
  - Writing Style: rewrote the rules for plain, human, structured prose, with concrete do and do-not points.
  - Principle V: every documentation and docstring example runs, proven by the build, with no secret and no network.
  - Project Profile: a local secret can hide an example, and release keeps development a superset of main with no tag
    collision.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 2.2.0 -> 2.3.0
Rationale: The gate is the full sequence in order, formatting, then checks, then the documentation build, then tests.
The documentation build executes the examples, so a broken example fails the gate and the build is never skipped. A
reused build environment can carry a stale toolchain, so a clean run can find a lint rule a cached one missed. Also
restructure the Code Conventions for clarity, splitting `Files & Module Structure` into `Module Structure` and `Package
Layering`, and lifting the credential rule out of `Tests` into its own `Credentials` subsection. No rule text changed.
MINOR: expanded guidance, nothing removed.

Modified sections:
  - Development Workflow: run the full gate in order, including the documentation build.
  - Project Profile: the gate order and commands, the docs build executes the examples, cached environments can hide a
    finding.
  - Code Conventions: split `Files & Module Structure` into `Module Structure` and `Package Layering`, and gave the
    credential rule its own `Credentials` subsection. Content unchanged.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 2.1.0 -> 2.2.0
Rationale: Fold in the lessons of the whole-src conformance sweep. The `from __future__ import annotations` import is
conditional, kept only where a forward reference or a `TYPE_CHECKING`-only name needs it, not carried by every module.
A general `_utils` module is allowed for assorted small helpers, and a helper earns its own module only for a distinct
role. An `__init__` carries no license header. DRY extends from constants to behavior, so two surface serializations
that are distinct contracts stay apart. The gate's verdict is the nox session summary, not a piped exit code. The
`__future__` change redefines a stated rule, the rest is expanded guidance, so MINOR.

Modified sections:
  - Files & Module Structure: made `from __future__ import annotations` conditional, and allowed a general `_utils`.
  - Public Surface: an `__init__` carries no license header.
  - Don't Repeat Yourself: extended the one-fact rule to behavior and distinct surface contracts.
  - Project Profile: added how to read the gate's verdict from the nox session summary.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.

---- history ----
Version change: 2.0.0 -> 2.1.0
Rationale: Sharpen the Code Conventions. An `__init__` holds only its docstring and the re-exports, with no logic, no
function, and no `__getattr__`. A module constant is `UPPER_CASE` and lives in the constants block near the top of the
module, never mid-file. A constant is named for what it holds, not a role it plays. A constant two modules define the
same way is one fact and is lifted to the shared-leaves subpackage, while two that share a value but not a meaning stay
apart. MINOR: expanded guidance, nothing removed or redefined.

Modified sections:
  - Naming: added the constant-named-for-what-it-holds rule.
  - Files & Module Structure: added the UPPER_CASE-constants-at-the-top rule.
  - Public Surface: added the no-logic-in-__init__ rule.
  - Don't Repeat Yourself: added the lift-a-shared-constant rule.

---- history ----
Version change: 1.10.0 -> 2.0.0
Rationale: Structural rewrite. The document is reorganized into a repo-agnostic body (Core Principles, Code
Conventions, Toolchain, Workflow, Governance) plus a single Project Profile that instantiates it for this repository,
so the same constitution can be reused and extended elsewhere. Every rule from 1.x is kept. Scattered guidance is
merged: lint suppressions in one place, the trailing-underscore rule in one place, surface parity and credential
handling in one place. Project-specific facts move into the Project Profile. A Writing Style section is added, and the
whole document is reformatted to obey it: every line is at most 120 characters, no sentence uses a semicolon or a dash
as punctuation, and the prose is plain English. MAJOR: principles are reorganized and regeneralized, none removed.

Modified sections:
  - I. scikit-learn-Compatible API becomes I. Honor the Ecosystem Contract, with the concrete contract in the profile.
  - Surface parity and the agent-is-a-client rule move into a new VI. A Library, Not an Application.
  - Principle VI (naming, docstrings, structure) becomes the Code Conventions section, one subsection per concern.
  - Added Writing Style, which the document itself now follows.

Templates requiring updates:
  - .specify/templates/plan-template.md: Constitution Check gate is generic. OK.
  - .specify/templates/spec-template.md: generic, no conflict. OK.
  - .specify/templates/tasks-template.md: generic, no conflict. OK.
-->

# Constitution

- A typed Python library MUST follow the engineering principles and code conventions stated here.
- The body MUST stay repo-agnostic, so it can be reused across projects.
- A repository that adopts the constitution MUST instantiate it in a single Project Profile at the end, naming the
  concrete framework contract, toolchain, dependencies, and delivery surfaces.
- A project MUST extend the constitution by growing its Project Profile, and MUST amend the body only where a new rule
  is genuinely general.
- Every rule MUST be stated with MUST or MUST NOT, and every rule binds equally.

## Engineering

This section states the principles the code obeys, the style it and its documentation are written in, the conventions it
follows, the tools it is built with, and the gate it passes before merge. Every rule here is binding, not advisory. The
conventions are generic Python and hold in any project. The automated gate checks the mechanical parts, and the
conventions are the taste it cannot check.

### Contract

- Public objects MUST conform to the contract of the framework they plug into, and MUST NOT invent a parallel convention
  that silently breaks downstream code.
- Class initialization parameters MUST be stored unmodified under their own names.
- State the object learns at runtime MUST be exposed only through the framework's convention for derived state.
- Behavior MUST be configured through explicit parameters, and MUST NOT depend on hidden global or ambient state, so an
  object is deterministic and testable in isolation.
- The concrete contract a project conforms to MUST be named in its Project Profile.

Example:

A bettor that stores its parameter unchanged and learns into a trailing-underscore attribute.

```python
from typing import Self

class OddsBettor(BaseEstimator):
    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self.odds_ = compute_odds(X)
        return self
```

Counter-example:

The same bettor, renaming its parameter and hiding what it learns in a module-level global.

```python
class OddsBettor(BaseEstimator):
    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold_ = max(threshold, MIN_THRESHOLD)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        global ODDS
        ODDS = compute_odds(X)
        return self
```

Rationale:

Interoperability with an established ecosystem is a library's core value. Drift from its contract breaks users' code in
ways the tests here cannot see.

### Types

- All code, public and internal, MUST carry complete type annotations, and MUST pass the static type checker with no new
  ignored errors.
- A name MUST be typed correctly rather than silenced, and a `# type: ignore` MUST be a last resort governed by
  Suppressions.

Example:

A function whose parameter and return are both annotated, so the checker reads the contract the reader does.

```python
def count_fixtures(items: list[Fixture]) -> int:
    return len(items)
```

Counter-example:

The same function with the parameter untyped and the checker silenced instead.

```python
def count_fixtures(items) -> int:  # type: ignore[no-untyped-def]
    return len(items)
```

Rationale:

The annotation is the contract a reader and the checker both read. A silenced error hides a real mismatch, and it stays
hidden until a user hits it.

### Schema

- Data that crosses a public boundary MUST be validated against an explicit, declared schema.
- Data-shape assumptions MUST be declared as schemas, and MUST NOT be enforced by ad-hoc runtime checks scattered
  through the code.

Example:

A fixtures frame validated against a declared schema on its way out of the boundary.

```python
FIXTURES_SCHEMA = DataFrameSchema(
    {'date': Column('datetime64[ns]'), 'home_team': Column(str), 'odds': Column(float, Check.gt(1.0))}
)


def extract_fixtures_data(self) -> pd.DataFrame:
    return FIXTURES_SCHEMA.validate(self._read_fixtures())
```

Counter-example:

The same boundary guarded by an assertion and a hand-rolled check.

```python
def extract_fixtures_data(self) -> pd.DataFrame:
    fixtures = self._read_fixtures()
    assert 'date' in fixtures.columns
    if fixtures['odds'].min() <= 1.0:
        raise ValueError('bad odds')
    return fixtures
```

Rationale:

A DataFrame carries no shape in its type. A declared schema says which columns and values a boundary accepts, so a bad
frame fails at the boundary rather than deep inside the code.

### Tests

- Every behavioral change MUST ship with tests.
- The suite MUST run with branch coverage, randomized ordering, and executable docstrings, so every code example in a
  docstring MUST be correct and runnable.
- New logic MUST NOT reduce the coverage of the module it touches.
- A bug fix MUST include a regression test that fails before the fix.
- The test tree MUST mirror the source tree.
- A test MUST be named `test_<function>_<behavior>`, beginning with the function it exercises, then a terse behavior
  phrase with articles dropped, and its docstring MUST be a single line.
- A test MUST NOT import a private name from a private module. It MUST use the public API the way a user does, and where
  it needs an internal, the internal wants to be public.
- A test MUST NOT reach the network, and MUST use a recorded payload, a fake, or a locally served page instead.
- Fixtures MUST be typed, small, and live in the nearest `conftest.py`.

Example:

A test named for the function and the behavior it exercises, with a single-line docstring.

```python
def test_count_fixtures_empty_source() -> None:
    """Return zero for source with no fixtures."""
    assert count_fixtures([]) == 0
```

Counter-example:

A vague name, a private import, and a call that reaches the network.

```python
def test_fixtures():
    """Test the fixtures."""
    from sportsbet.sources._factory import _build_source

    source = _build_source('soccer')
    assert source.read_catalogue()
```

Rationale:

Randomized, doctest-inclusive testing keeps examples honest and guards against order-dependent flakiness where subtle
regressions are otherwise invisible.

### Gates

- Code MUST pass the full automated gate before merge, covering formatting, linting, static type checking, docstring
  coverage, a security scan, and a dependency audit.
- The gate MUST run locally through pre-commit and again in CI, and a red run MUST block merge.
- Failures MUST be fixed at the source.
- A disabled rule MUST be the exception governed by Suppressions, and MUST NOT stand in as a workaround.
- Work MUST happen on feature branches, and the release branch MUST stay green.
- Before opening a PR, a contributor MUST run the full gate in order, formatting, then checks, then the documentation
  build, then tests, MUST resolve all findings, and MUST conform to the conventions in Engineering.
- The documentation build MUST NOT be skipped, since it is what runs the examples.
- A release MUST follow semantic versioning, and MUST update the changelog before tagging.

Example:

The gate run in order, with the verdict read from the session summary line.

```console
$ pdm run formatting
$ pdm run checks
$ pdm run docs build
$ pdm run tests
Session tests-3.13 was successful.
```

Counter-example:

The documentation build skipped, and the exit code read from the last stage of a pipe.

```console
$ pdm run tests | tail -1
Session tests-3.13 was successful.
$ git push
```

Rationale:

One machine-enforced bar removes style debate, keeps diffs reviewable, and stops security-sensitive dependencies from
silently degrading.

### Documentation

- A user-facing behavioral change MUST update the affected documentation, and MUST add or amend a changelog entry where
  it changes public behavior.
- Every code example in the documentation and the docstrings MUST run, and the build MUST prove it, the documentation
  build running the gallery examples and the doctest run running the docstring examples.
- An example MUST NOT be a fragment, pseudo-code, or a demo that cannot run.

Example:

A documentation page whose block the build executes, printing the output the reader sees.

````markdown
Load a dataloader and extract the training data.

```python exec="true" source="material-block"
from sportsbet.datasets import SoccerDataLoader

dataloader = SoccerDataLoader({'league': ['England']})
X_train, Y_train, O_train = dataloader.extract_train_data()
print(X_train.head().to_markdown())
```
````

Counter-example:

The same page with a block the build never runs, and an output pasted by hand under it.

````markdown
Load a dataloader and extract the training data.

```python
dataloader = SoccerDataLoader(...)
X_train, Y_train, O_train = dataloader.extract_train_data()
```

The output looks like this:

| date       | home_team |
| ---------- | --------- |
| 2026-08-15 | Arsenal   |
````

Rationale:

A library is adopted through its documentation. An undocumented capability does not exist for users, and it rots without
executable coverage.

### Library

- The package MUST stay a library, and an application concern such as an interactive loop, a long-lived session, a
  choice of model or policy, or a credential MUST stay with the caller.
- Those concerns MUST stay out, since keeping them out is what makes the library deterministic and testable.
- A capability that needs a credential or performs a real-world side effect MUST live behind an optional extra, and MUST
  NOT ship in the default install.
- Where a project exposes several delivery surfaces, for example a Python API, a command line, and a server, the
  surfaces MUST expose the same underlying capabilities, and none MUST hold logic the others cannot reach.
- A parity test MUST assert this, so the surfaces cannot drift.

Example:

The caller chooses the model and passes it in.

```python
bettor = build_bettor(classifier=LogisticRegression())
```

Counter-example:

The library picks a model and runs a loop of its own.

```python
bettor = build_bettor(model_key='claude-opus-5', auto_bet=True)
```

Rationale:

A library free of application state stays composable and testable, and surface parity keeps a capability from existing
on one surface only.

### Style

- Every document, docstring, and example MUST follow one style, written for a reader who wants to understand, not to be
  impressed.
- Sentences MUST be of a natural length. Prose MUST NOT be chopped into short choppy sentences, and MUST NOT be padded
  into long winding ones. The length MUST vary the way it does when a person writes well, and sentences MUST join with
  commas and conjunctions where that reads better.
- A sentence MUST say who does what in the normal order, subject then verb then object. It MUST read "The library places
  the bet at the venue", not "A venue is placed at by the library".
- Word order MUST NOT be inverted for effect, and the passive voice MUST NOT be used unless the doer is unknown or does
  not matter.
- Words MUST be plain, and an idiom, a metaphor, or a literary flourish MUST NOT appear. It MUST read as a person wrote
  it for another person, not as a performance.
- A real risk MUST be stated once, in the right place, plainly. It MUST NOT be hedged, and a warning MUST NOT be
  repeated.
- A document MUST carry headings and subsections so a reader can scan it, and MUST NOT run as one undivided block.
- A section that is a set of rules MUST be written as a bullet list with one rule per bullet, and prose MUST be kept for
  a short introduction or a rationale.
- A line MUST be at most 120 characters. A sentence MUST NOT use a semicolon or a dash as punctuation, and a hyphenated
  word such as trailing-underscore MUST stay allowed.

Example:

A summary line in the normal order, subject then verb then object.

```python
def read_fixtures(source: Source) -> list[Fixture]:
    """Read the fixtures from the source and validate them against the schema."""
```

Counter-example:

The same sentence inverted, passive, and dressed up.

```python
def read_fixtures(source: Source) -> list[Fixture]:
    """From the source the fixtures are read, whereupon validation against the schema occurs."""
```

Rationale:

A reader who can scan a document acts on it. Prose that performs makes a reader work for what the rule already says.

### Naming

- A function name MUST begin with a verb, and MUST name what the function actually does or returns.
- A name MUST read `count_common_prefix`, not `common_prefix_length`, `normalize_identity`, not `transform_identity`,
  and `build_roster`, not `roster`.
- A name that begins with a noun describes a value, so a function MUST NOT carry one.
- The verb MUST be honest, so a function that loads or resolves an object from a reference MUST be `load_` or
  `resolve_`, not `build_`.
- An empty verb that says nothing, such as `process`, `handle`, `manage`, or `transform` with no object, MUST NOT be
  used.
- A method MUST be an action and MUST begin with a verb. A property MUST name a value as a noun phrase, and MUST NOT be
  verb-first.
- State an instance derives at runtime MUST carry the framework's derived-state marker, which in this ecosystem is a
  trailing underscore, whether the state is a stored attribute or a computed property.
- A public instance name MUST be one of two things, a constructor parameter stored unmodified under its own name with no
  marker, or a derivation carrying the marker. Anything else an instance exposes MUST be private.
- A class-level constant that declares what a class is, such as its kind or its name, MUST be a `ClassVar`, and stands
  apart from this.
- A module MUST be named for the concern it owns, with a descriptive noun for what it does or holds, such as
  `_resolver`, `_schedule`, or `_factory`.
- A module MUST NOT be named for the data it consumes, and MUST NOT be named for the surface that happens to call it.
- A module name MUST NOT reuse a name that collides with a dependency's concept, and names MUST come from the domain,
  used consistently.
- A constant MUST be named for what it holds, not for a role it happens to play, so where the values are the preplay
  statuses the name MUST be `PREPLAY_EVENT_STATUSES`, not `INPUT_EVENT_STATUSES`.
- Every constant name MUST be read again and checked that it still describes its value.

Example:

A constant named for what it holds and a function named for what it does.

```python
PREPLAY_EVENT_STATUSES = ('scheduled', 'delayed')


def count_common_prefix(left: str, right: str) -> int:
    ...
```

Counter-example:

A constant named for a role it plays and a function named as if it were a value.

```python
INPUT_EVENT_STATUSES = ('scheduled', 'delayed')


def common_prefix_length(left: str, right: str) -> int:
    ...
```

Rationale:

A name that tells the truth saves a comment, a docstring line, and a reading of the body.

### Structure

- A module MUST read top to bottom in one order, a one-line imperative module docstring, the licence header, the
  imports grouped standard library, third party, first party, the module constants and type aliases, and then the
  functions.
- The linter sorts the imports, so they MUST NOT be sorted by hand.
- A module MUST add `from __future__ import annotations` above the imports only where a forward reference needs it.
  The supported language floor decides, and a version that resolves the annotations without it MUST NOT carry it.
- A module MUST NOT guard an import with `if TYPE_CHECKING:`. An import only a type checker sees papers over a cycle
  the way an import inside a function body does, and the cycle MUST be fixed instead.
- Functions MUST come in dependency order, so a name is defined before it is used, the small helpers first and the
  function the module exists for last.
- The private functions MUST come before the public ones, and the private methods of a class before its public
  methods, so a reader meets the parts before the whole.
- A module constant MUST be `UPPER_CASE`, and MUST live in the constants block near the top of the module, never
  mid-file among the functions.
- A module constant MUST NOT carry a leading underscore, whatever its reach. A constant is kept private by not
  re-exporting it, and the rules below that turn an unexported name private MUST NOT reach it.
- One module MUST be one concern, and a file that grows two MUST be split.
- Small general helpers MUST share a `_utils` module, whose one concern is the assorted helpers a package needs, and a
  helper MUST earn its own module only where it takes on a distinct role worth a name, as `_base` or `_types` do.
- A definition MUST live in the module that owns it.
- A base module MUST be self-contained and MUST import no sibling. Its purpose is to be imported, not to import, so a
  base that needs a sibling's code MUST absorb it by merging rather than importing.
- The top level of a package MUST hold subpackages and its `__init__`, and MUST NOT hold loose implementation modules.
- The shared leaves the whole tree imports, the type vocabulary, the shared constants, and the shared building
  primitives, MUST live in a `core` subpackage, and the rest of the tree MUST import them from there.
- A builder MUST live in the package that owns what it builds, and MUST be re-exported from there.
- Every import MUST run downward, from `core` to domain packages to surfaces, so no cycle can form.
- An import inside a function body MUST NOT paper over a cycle, and the cycle MUST be fixed instead.
- The only lazy import MUST defer an optional dependency, and MUST carry a suppression with a reason.
- This layering MUST be preferred over a suppression and a comment that paper over an out-of-order import.

Example:

A module in the one order, its constants at the top and its imports running downward.

```python
"""Resolve a venue from its name."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import json

import pandas as pd

from sportsbet.core import VENUE_SEPARATOR

DEFAULT_TIMEOUT = 30


def _read_registry(path: Path) -> dict[str, str]:
    ...


def resolve_venue(name: str) -> Venue:
    ...
```

Counter-example:

Imports sorted by hand, an upward import, a constant mid-file, and an import inside a function body.

```python
"""Utilities."""

import pandas as pd
import json

from sportsbet.cli import run

TIMEOUT_DEFAULT = 30


def resolve_venue(name: str) -> Venue:
    from sportsbet.sources import build_source

    RETRIES = 3
    ...
```

Rationale:

A module a reader can predict is a module a reader can change, and one import direction is what keeps a package free of
cycles.

### Surface

- Implementation modules, classes, and helpers MUST be private, named `_name`.
- The package `__init__` MUST re-export the public surface with an explicit `__all__`, and MUST carry only its
  docstring and those re-exports, with no licence header, since it holds no implementation of its own.
- A public name used outside the module that defines it MUST be re-exported through its owning package's `__init__` and
  imported from that surface, and MUST NOT be imported from the private module that defines it.
- A name MUST be re-exported once, where it lives, and a parent package MUST NOT re-export a subpackage's surface a
  second time.
- This MUST hold for the public modules and the tests alike. Reaching into another package's private module for a public
  name is the smell the re-export removes.
- An `__init__` MUST hold only its docstring and the re-exports. It MUST carry no logic, no function, and no
  `__getattr__`, and a name that has to be computed to be exposed MUST live in a module instead.
- A module MUST NOT import another package's private name, whatever its own privacy. A name reached past its package
  surface is a name that package never promised. A package and its subpackages are one package for this rule, so a
  subpackage reaching its parent's private module is not reaching past a surface.
- A name its own package never re-exports MUST be private, so what a package exports and what its surface offers are
  the same list. A module constant is the exception, and stays `UPPER_CASE`.
- A package MUST re-export only the names something outside it uses. An export nothing outside the package reaches
  for is a promise the library gained nothing by making, and MUST stop being re-exported. A name that is not a
  constant MUST also become private.
- The tests and the documentation are client code, so a name either of them reaches is used, and MUST stay
  re-exported. A name that appears in a public signature is used too, since a caller reaches it through that
  signature.
- A name nothing uses at all MUST be removed rather than kept private. Code that no caller reaches is code a reader
  still has to read.
- A surface package, one that exists to serve a runner rather than an importer, MUST re-export its entry point alone.
  The names inside it are reached by the runner that discovers them, a command name or a tool name, so the rule above
  MUST NOT reach them. The Project Profile names which packages these are.
- The package `__init__` MUST be the only place a public name leaves a private module. That is the re-export, and
  everywhere else the name MUST be imported from the surface that re-exports it.

Example:

A module consuming another package through its surface, and the package `__init__` re-exporting its own names.

```python
"""Private module."""

from a import b
from ._sibling import _helper

"""Package __init__, the one place a public name leaves a private module."""

from ._base import Source
from ._factory import build_source

__all__ = ['Source', 'build_source']
```

Counter-example:

A name reached past its surface, and a name a package keeps to itself but never made private.

```python
"""Private module."""

from a._b import c
from a import _c
from ._sibling import HELPER
```

Rationale:

A surface a user can see is a surface a library can keep. Every name reachable by another route is a name that cannot be
changed without breaking someone.

### Docstrings

- Every module, class, and function, public or private, MUST carry a docstring.
- The summary line MUST be one line, imperative, and MUST say what the thing does.
- It MUST NOT read `Implements the ...`, `This function ...`, or `A class that ...`, which are meta narration.
- It MUST be written in plain English, with simple, direct words over clever or roundabout phrasing, and a line that
  reads awkwardly out loud MUST be rewritten.
- A public function and a public class MUST carry an `Args` block documenting every parameter, a `Returns` block
  saying what is returned, and a `Raises` block naming every exception it raises.
- A private function and a private class MUST carry the one-line summary alone, and MUST NOT carry those blocks.
- A command or a tool in a surface package MUST carry the one-line summary alone. The runner shows that line to a
  user as help text or to an agent as a tool description, and it documents the parameters itself, from the options
  and the schema it already holds.
- A block that would stand empty MUST be omitted, so a function that takes no parameter carries no `Args`, one that
  returns nothing carries no `Returns`, and one that raises nothing carries no `Raises`.
- A constructor parameter or a dataclass field MUST be documented under `Args`, and an `Attributes` block MUST hold
  learned state only.
- A docstring MUST describe what the thing is and what it holds, plainly, and MUST NOT state its virtues, such as `free`
  or `needs no key`.
- It MUST NOT give the rationale for its shape, the `since ...` or `so ...` clause, and MUST NOT say what downstream
  code builds from it. It MUST state the content, not the sales pitch, the justification, or the uses.
- It MUST NOT describe what the code does not do, and MUST NOT restate a self-evident name, so a `url` field needs no
  "where to read it from".
- It MUST carry no essay and no editorializing. It MUST NOT join two clauses with a semicolon or a dash, and each point
  MUST be its own sentence.
- Public API MUST carry a runnable example checked by the doctest run, and a network-touching class MUST NOT.
- The top-level package `__init__` MUST be the one exception, since as the library's front page it carries a fuller
  docstring, a tagline and a short overview of the submodules.

Example:

The blocks a public function carries, and the summary line a private helper carries alone.

```python
def resolve_venue(name: str) -> Venue:
    """Resolve a venue from its name.

    Args:
        name: Name of the venue.

    Returns:
        The venue registered under the name.

    Raises:
        SelectionError: If no venue matches the name.
    """


def _read_registry(path: Path) -> dict[str, str]:
    """Read the venue registry from the path."""
```

Counter-example:

Meta narration, a virtue, a semicolon, and a private helper carrying blocks it MUST NOT.

```python
def resolve_venue(name: str) -> Venue:
    """This function implements venue resolution; it is free and needs no key.

    Since the registry moved to JSON, the CLI builds its venue table from this.
    """


def _read_registry(path: Path) -> dict[str, str]:
    """Read the venue registry from the path.

    Args:
        path: Path to the registry.

    Returns:
        The registry.
    """
```

Rationale:

The docstring is what a user reads in an editor, without the source beside it. What it claims is what they believe.

### Comments

- Source MUST carry almost no comments, since the names say what and the docstring says why.
- An inline comment that explains the next line MUST be removed by fixing the line or its names, which are what is
  unclear.
- The only comments in source MUST be the licence header and, rarely, a suppression.

Example:

A name that needs no comment.

```python
timeout_seconds = 30
```

Counter-example:

A short name propped up by a comment.

```python
t = 30  # timeout in seconds
```

Rationale:

A comment drifts away from the code it explains. A name and a docstring travel with it.

### Suppressions

- A lint or type suppression, a `# noqa` or a `# type: ignore`, MUST be a last resort for a genuine one-off, and MUST
  carry the rule code and the reason.
- A suppression that recurs across the repository MUST NOT be repeated inline. The rule MUST be configured once in the
  project configuration, as a scoped ignore, so the decision lives in one place rather than scattered through the
  source.

Example:

A one-off suppression carrying its rule code and its reason.

```python
from mcp.server import FastMCP  # noqa: PLC0415 the mcp extra is optional, so the import is deferred
```

Counter-example:

A bare suppression that says neither which rule nor why.

```python
from mcp.server import FastMCP  # noqa
```

Rationale:

A suppression is a decision. One place to read it is one place to revisit it.

### Errors

- The message MUST be built in a variable, and then raised.
- A specific named exception defined for the module or package MUST be raised, and a bare `Exception` or `ValueError`
  MUST NOT stand where a named one carries meaning.
- The message MUST tell the reader what to do, the variable that was missing, or the value that did not match.
- An exception MUST NOT be caught and swallowed, and MUST be caught narrowly or left to propagate.

Example:

A named exception raised with the message built in a variable.

```python
if venue is None:
    message = f'No venue named {name!r}. Run `sportsbet venues` to list the registered ones.'
    raise SelectionError(message)
```

Counter-example:

A bare catch, swallowed, and re-raised as a generic error.

```python
try:
    venue = REGISTRY[name]
except Exception:
    raise ValueError(f'bad name {name}')
```

Rationale:

An error is the one message a user reads at their worst moment. A named exception and a plain message tell them what to
change.

### Control Flow

- Functions MUST be small, and MUST do one thing.
- A function that needs a paragraph of docstring body to explain its branches MUST be two functions.
- A function MUST return early with guard clauses.
- Deep nesting MUST be avoided, and a helper MUST be extracted before the third level of indentation.

Example:

A guard clause and a single return.

```python
def read_best_odds(fixture: Fixture) -> float:
    if not fixture.odds:
        return DEFAULT_ODDS
    return max(fixture.odds)
```

Counter-example:

The same logic nested three deep.

```python
def read_best_odds(fixture: Fixture) -> float:
    if fixture is not None:
        if fixture.odds:
            if len(fixture.odds) > 0:
                return max(fixture.odds)
    return DEFAULT_ODDS
```

Rationale:

A shallow function is one a reader holds in their head all at once.

### Duplication

- A fact, a definition, or a derivation MUST live in exactly one place, and the same thing expressed twice MUST be
  collapsed to one.
- What can be derived from what is already kept MUST NOT be stored. The source MUST be persisted and the projection
  derived on demand, not both.
- A capability the codebase already has MUST NOT be reimplemented, and MUST be reused rather than copied into another
  module.
- A constant that two modules define the same way MUST be treated as one fact, whatever each names it, and MUST be
  lifted to the subpackage that holds the shared leaves and imported from there.
- Two constants that share a value but not a meaning, such as a column-name separator and an item-key separator that are
  both `'__'`, MUST be treated as two facts, and MUST stay apart.
- The same MUST hold for behavior, so two surfaces that serialize an object into different shapes, each a contract its
  callers depend on, MUST be treated as two facts, not one duplication.
- Only the fragment that is identical at every call site MUST be collapsed, and where call sites differ in what they
  check or emit, a helper that unifies them changes behavior, so they MUST stay apart.

Example:

The separator imported from the one place that defines it.

```python
from sportsbet.core import COLUMN_SEPARATOR
```

Counter-example:

The same separator restated in a second module.

```python
COLUMN_SEP = '__'
```

Rationale:

One fact in one place means one edit. Two facts forced into one helper means a silent change in behavior.

### Credentials

- A credential MUST be named, and MUST NOT be passed.
- A function, command flag, or tool argument MUST take the name of the variable holding the secret, and MUST read it
  where it is used.
- A secret MUST NOT become an argument value, a log line, or a pickle.

Example:

The name of the variable holding the token travels, and the code reads it where it is used.

```python
place_bet(venue='betfair', token_var='BETFAIR_TOKEN')
```

Counter-example:

The token itself travels as an argument value.

```python
place_bet(venue='betfair', token='hunter2')
```

Rationale:

A secret that never becomes a value never leaks into a trace, a log, or a pickle.

### Toolchain

- The project MUST declare its supported language versions, and MUST remain compatible across all of them.
- A new runtime dependency MUST be justified and declared in the project manifest, and MUST NOT be vendored ad hoc.
- A credentialled or side-effecting capability MUST live behind an optional extra, as Library requires.
- The build MUST use a `src`-based layout with SCM-derived versioning, and generated version metadata MUST NOT be
  hand-edited.
- Canonical task-runner sessions for tests, checks, formatting, docs, and release MUST be the entry points the gate runs
  through.
- The style constants, the line length, the docstring convention, and the formatter options, MUST be set once in the
  project configuration, and MUST NOT be overridden by hand.
- The concrete versions, tools, and dependency list MUST live in the Project Profile.

Example:

The line length and the optional extra set once in the project manifest.

```toml
[tool.ruff]
line-length = 120

[project.optional-dependencies]
mcp = ['mcp>=1.2']
```

Counter-example:

Two tools given two different answers to the same style question.

```toml
[tool.ruff]
line-length = 88

[tool.black]
line-length = 120
```

Rationale:

One configured toolchain is one answer to every question about how the code is built and checked.

## Governance

This constitution supersedes ad-hoc conventions and prior undocumented practice. It applies to all code, documentation,
and tooling changes in the repository that adopts it.

### Amendments

- An amendment MUST be proposed through a PR that edits this file, states the rationale, and updates the version and the
  Sync Impact Report.
- An amendment that adds or removes a principle or governance rule MUST carry the maintainer's approval.

Example:

An amendment committed with its version in the message.

```console
$ git commit -m "docs: amend constitution to v6.0.0 (state every rule with MUST)"
```

Counter-example:

An edit with no version bump and no Sync Impact Report.

```console
$ git commit -m "tweak the rules"
```

Rationale:

An amendment that leaves a trail can be read back later, in the words of the person who made it.

### Versioning

- The constitution MUST carry its own semantic version.
- MAJOR MUST mark a backward-incompatible removal or redefinition of a principle or governance rule, or a structural
  rewrite.
- MINOR MUST mark a new principle or section, or materially expanded guidance.
- PATCH MUST mark a clarification or a non-semantic wording fix.

Example:

A rule restated across the document, carried by a MAJOR bump.

```diff
-**Version**: 5.2.1 | **Ratified**: 2026-07-08 | **Last Amended**: 2026-09-20
+**Version**: 6.0.0 | **Ratified**: 2026-07-08 | **Last Amended**: 2026-09-20
```

Counter-example:

A rule loosened under a PATCH bump.

```diff
-**Version**: 6.0.0 | **Ratified**: 2026-07-08 | **Last Amended**: 2026-09-20
+**Version**: 6.0.1 | **Ratified**: 2026-07-08 | **Last Amended**: 2026-09-20
-- A test MUST NOT reach the network.
+- A test MUST be allowed to reach the network where no fake exists.
```

Rationale:

A version tells a reader whether the rules they learned still hold.

### Compliance

- Every PR and code review MUST verify adherence to this constitution, not just correctness.
- Every PR MUST state which principles it touches, and MUST confirm the gates pass.
- A deviation MUST be justified in the PR, and MUST be recorded in the plan where it adds complexity.
- An unjustified violation MUST block merge.

Example:

A PR stating the principles it touches, the gate it ran, and its deviations.

```markdown
## Compliance

- Principles touched: Tests, Gates.
- Gate: `pdm run tests` green on 3.11, 3.12, and 3.13.
- Deviations: none.
```

Counter-example:

A review that says nothing about either.

```markdown
## Compliance

Looks good to me.
```

Rationale:

A rule nobody checks is a rule nobody follows.

### Guidance

- Contributor-facing operational guidance MUST live alongside the code, in a `CONTRIBUTING.md` and developer docs.
- It MUST stay consistent with this constitution.

Example:

Contributor guidance that repeats the gate the constitution requires.

```markdown
## Before you push

Run `pdm run formatting`, `pdm run checks`, `pdm run docs build`, then `pdm run tests`.
```

Counter-example:

Guidance that tells a contributor to skip part of it.

```markdown
## Before you push

Run the tests. Skip the documentation build, it is slow.
```

Rationale:

A contributor reads the repository before they read this file, so what they find there has to agree with it.

### Project Profile

- Every rule below MUST bind the sports-betting repository alone.
- Ecosystem contract: the estimators, the dataloaders and the bettors, MUST conform to the scikit-learn estimator
  contract, keeping the surface of `fit`, `predict`, `predict_proba`, `bet`, `score`, `get_params`, and `set_params`.
  Sources MUST implement the source-plugin contract of `list_index_items`, `read_catalogue`, `list_required_items`,
  `list_fixtures_items`, `to_snapshots`, and `request_url`.
- Application concerns: a model, a model key, a model choice, or an agent loop MUST NOT enter the package.
- Delivery surfaces: the surfaces MUST be the Python API, the CLI (`sportsbet`), and the MCP server (`sportsbet-mcp`).
- Schema validation: the boundary frames, the training data, the fixtures, and the odds, MUST be validated against a
  `pandera` schema.
- Language: the package MUST support Python `>=3.11, <3.14`, targeting `py311`.
- Core dependencies: the package MUST depend on `scikit-learn`, `pandas`, `pandera`, `click`, `rich`, and `aiohttp`, and
  the optional `mcp` extra MUST ship the `sportsbet-mcp` server.
- Build and tooling: the build MUST use PDM with SCM-derived versioning and a `src` layout. The `nox` sessions MUST be
  `tests`, `checks`, `formatting`, `docs`, `changelog`, and `release`, with `pdm run` shortcuts. The gate MUST run in
  order, `pdm run formatting`, `pdm run checks`, `pdm run docs build`, then `pdm run tests`, covering `black`,
  `docformatter`, `ruff`, `interrogate`, `bandit`, `pip-audit`, `pytest` with `--doctest-modules`, and `mypy`.
- Reading the gate: the verdict MUST be read from the `nox` session summary line, such as `Session tests-3.13 was
  successful`. An exit code read from a piped command reports the last stage of the pipe, not the run, so it MUST NOT be
  trusted. A reused `nox` environment can carry a stale toolchain, so a clean run MUST settle a doubt about a lint rule.
  A locally set secret can hide an example that needs one, so the build MUST run the examples with no secret.
- Releasing: the release tool computes the next version from the commits since the last tag, so `development` MUST stay
  a superset of the released `main`, the computed version MUST NOT collide with a tag that already exists, and the
  branches MUST be reconciled before a release.
- Style: the line length MUST be 120, the docstrings MUST be Google style, and `black` MUST run with
  skip-string-normalization.
- Package layering: the layers MUST be `core`, then the domain packages `sources`, `dataloaders`, `evaluation`, and
  `execution`, then the surfaces `cli` and `mcp`, and the builders MUST be `build_dataloader`, `build_bettor`, and
  `build_venue`.
- Named exceptions: the package MUST raise `BuildError`, `ExecutionError`, and `CredentialError`. The dataloaders
  and the evaluation package MUST raise the `ValueError` and `TypeError` the scikit-learn contract expects for
  invalid input, since the ecosystem contract wins where the two disagree.

Example:

A user importing each surface from the package that re-exports it.

```python
from sportsbet.datasets import SoccerDataLoader
from sportsbet.evaluation import ClassifierBettor
```

Counter-example:

The same loader reached for inside a private module.

```python
from sportsbet.datasets._soccer import SoccerDataLoader
from sportsbet.evaluation import ClassifierBettor
```

Rationale:

One repo-specific section keeps the body portable. A reader of another repository reads the same rules and a different
profile.

**Version**: 11.0.1 | **Ratified**: 2026-07-08 | **Last Amended**: 2026-09-22
