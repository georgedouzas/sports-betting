# Contributing

Contributions are welcome, and they are greatly appreciated.

## Tasks

This project uses [nox](https://nox.thea.codes/en/stable/) to run development tasks. Please check the `noxfile.py` at the root of
the project for more details. You can run any of the following commands and subcommands that corresponds to a particular task:

### Documentation

- `pdm docs serve` or `pdm docs`: Serve the documentation.
- `pdm docs build`: Build locally the documentation.

### Formatting

- `pdm formatting all` or `pdm formatting`: Format both the code and docstrings.
    - `pdm formatting code`: Format only the code.
    - `pdm formatting docstrings`: Format only the docstrings.

### Checks

- `pdm checks all` or `pdm checks`: Run all checks.
    - `pdm checks quality`: Check only code quality.
    - `pdm checks types`: Check only type annotations.
    - `pdm checks dependencies`: Check only for vulnerabilities in dependencies.

### Tests

- `pdm tests`: Run the tests.

### Changelog

- `pdm changelog`: Build the changelog.

### Release

- `pdm release`: Release a new Python package with an updated version.

## Development

The next steps should be followed during development:

- `git checkout -b new-branch-name` to create a new branch and then modify the code.
- `pdm formatting` to auto-format the code and docstrings.
- `pdm checks` to apply all checks.
- `pdm tests` to run the tests.
- `pdm docs serve` if you updated the documentation or the project dependencies to check that everything looks as expected.

## Commit message convention

Commit messages follow conventions based on the [Angular
style](https://gist.github.com/stephenparish/9941e89d80e2bc58a153#format-of-the-commit-message).

### Structure

```bash
<type>(<scope>): <subject>

<body>

<footer>
```

### Example

```
feat(directive): A new feature of code

A description of the new feature.
It contains **important** information.

Issue #10: https://github.com/namespace/project/issues/10
Related to PR namespace/other-project#15: https://github.com/namespace/other-project/pull/15
```

#### Guidelines

- Scope and body are optional.
- Subject and body must be valid Markdown.
- Body must add trailers at the end, for example issues and PR references or co-authors.
- Subject must have proper casing, i.e. uppercase for first letter if it makes sense.
- Subject must have no dot at the end and no punctuation.
- Type can be:
    - `feat`: New feature implementation.
    - `fix`: Bug fix.
    - `docs`: Documentation changes.
    - `style`: Code style or format changes.
    - `refactor`: Changes that are not features or bug fixes.
    - `tests`: Test additions or corrections.
    - `chore`: Maintenance code changes.

## Pull Request guidelines

Link to any related issue in the Pull Request message. We also recommend using fixups:

```bash
git commit --fixup=SHA
```

Once all the changes are approved, you can squash your commits:

```bash
git rebase -i --autosquash master
```

And force-push:

```bash
git push -f
```
