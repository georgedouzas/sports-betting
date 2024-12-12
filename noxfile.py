"""Development tasks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import nox

os.environ.update({'PDM_IGNORE_SAVED_PYTHON': '1'})

PYTHON_VERSIONS: list[str] = ['3.10', '3.11', '3.12']
FILES: list[str] = ['src', 'tests', 'docs', 'noxfile.py']
CHANGELOG_ARGS: dict[str, Any] = {
    'repository': '.',
    'convention': 'angular',
    'template': 'keepachangelog',
    'parse_trailers': True,
    'parse_refs': False,
    'sections': ['feat', 'fix', 'docs', 'style', 'refactor', 'tests', 'chore'],
    'bump_latest': True,
    'output': 'CHANGELOG.md',
}


def check_cli(session: nox.Session, args: list[str]) -> None:
    """Check the CLI arguments.

    Arguments:
        session: The nox session.
        args: The available CLI arguments.
    """
    available_args = ', '.join([f'`{arg}`' for arg in args])
    msg = f'Available subcommands are one of {available_args}.'
    if not session.posargs:
        session.skip(f'{msg} No subbcommand was provided')
    elif len(session.posargs) > 1 or session.posargs[0] not in args:
        session.skip(f'{msg} Instead `{" ".join(session.posargs)}` was given')


@nox.session
def docs(session: nox.Session) -> None:
    """Build or serve the documentation.

    Arguments:
        session: The nox session.
    """
    check_cli(session, ['serve', 'build'])
    session.run('pdm', 'install', '-dG', 'docs', external=True)
    session.run('mkdocs', session.posargs[0])


@nox.session
@nox.parametrize('file', FILES)
def formatting(session: nox.Session, file: str) -> None:
    """Format the code.

    Arguments:
        session: The nox session.
        file: The file to be formatted.
    """
    check_cli(session, ['all', 'code', 'docstrings'])
    session.run('pdm', 'install', '-dG', 'formatting', '--no-default', external=True)
    if session.posargs[0] in ['code', 'all']:
        session.run('black', file)
    if session.posargs[0] in ['docstrings', 'all']:
        session.run('docformatter', file)


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize('file', FILES)
def checks(session: nox.Session, file: str) -> None:
    """Check code quality, dependencies or type annotations.

    Arguments:
        session: The nox session.
        file: The file to be checked.
    """
    check_cli(session, ['all', 'quality', 'dependencies', 'types'])
    session.run('pdm', 'install', '-dG', 'checks', '--no-default', external=True)
    if session.posargs[0] in ['quality', 'all']:
        session.run('ruff', 'check', file)
    if session.posargs[0] in ['types', 'all']:
        session.run('mypy', file)
    if session.posargs[0] in ['dependencies', 'all']:
        requirements_path = (Path(session.create_tmp()) / 'requirements.txt').as_posix()
        args_groups = [['--prod']] + [['-dG', group] for group in ['tests', 'docs', 'maintenance']]
        requirements_types = zip(FILES, args_groups, strict=True)
        args = [
            'pdm',
            'export',
            '-f',
            'requirements',
            '--without-hashes',
            '--no-default',
            '--pyproject',
            '-o',
            requirements_path,
        ]
        session.run(*(args + dict(requirements_types)[file]), external=True)
        session.run('safety', 'check', '-r', requirements_path)


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run tests and coverage.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'tests', external=True)
    env = {'COVERAGE_FILE': f'.coverage.{session.python}'}
    if session.posargs:
        session.run('pytest', '-k', *session.posargs, env=env)
    else:
        session.run('pytest', env=env)
    session.run('coverage', 'combine')
    session.run('coverage', 'report')
    session.run('coverage', 'html')


@nox.session
def changelog(session: nox.Session) -> None:
    """Build the changelog.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'changelog', '--no-default', external=True)
    from git_changelog.cli import build_and_render

    build_and_render(**CHANGELOG_ARGS)


@nox.session
def release(session: nox.Session) -> None:
    """Kick off a release process.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'changelog', '-dG', 'release', '--no-default', external=True)
    from git_changelog.cli import build_and_render

    changelog, _ = build_and_render(**CHANGELOG_ARGS)
    if changelog.versions_list[0].tag:
        session.skip('Commit has already a tag. Release is aborted.')
    version = changelog.versions_list[0].planned_tag
    if version is None:
        session.skip('Next version was not possible to be specified. Release is aborted.')

    # Create release branch and commit changelog
    session.run('git', 'checkout', '-b', f'release_{version}', external=True)
    session.run('git', 'add', 'CHANGELOG.md', external=True)
    session.run('git', 'commit', '-m', f'chore: Release {version}', '--allow-empty', external=True)
    session.run('git', 'push', '-u', 'origin', f'release_{version}', external=True)

    # Create and merge PR from release branch to main
    session.run('gh', 'pr', 'create', '--base', 'main', external=True)
    session.run('gh', 'pr', 'merge', '--rebase', '--delete-branch', external=True)

    # Create tag
    session.run('git', 'checkout', 'main', external=True)
    session.run('git', 'pull', '--rebase', external=True)
    session.run('git', 'tag', version, external=True)
    session.run('git', 'push', '--tags', external=True)

    # Build and upload artifacts
    session.run('pdm', 'build', external=True)
    session.run('twine', 'upload', '--skip-existing', 'dist/*')
