# Quickstart: Plain Documentation

How to validate the feature.

## The gate, in order

A page is done only when all four are green on 3.11, 3.12, and 3.13:

```sh
pdm run formatting
pdm run checks
pdm run docs build
pdm run tests
```

The docs build executes every gallery example. The tests run executes every docstring example. Read the verdict from
the nox session summary, not a piped exit code.

## Scenario 1: the docs read plainly

Open the README and each user guide page. Every sentence is short and direct. Every page has headings. The install
section has four subsections: basic, MCP extra, execution extra, and development. No page is one undivided block.

## Scenario 2: every example runs

Run `pdm run docs build` and `pdm run tests`. Every gallery example and every docstring example runs and passes. No
example asks for a key or reaches the network.

## Scenario 3: the rules hold

Read the constitution's Writing Style rules. They require the plain style and forbid the clever one, in checkable
terms. A rule requires examples to run. The 0.15.0 release lessons are recorded.
