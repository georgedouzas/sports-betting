"""Development tasks."""

import os
import shutil
from pathlib import Path
from typing import Any

import nox

os.environ.update({'PDM_IGNORE_SAVED_PYTHON': '1'})


PYTHON_VERSIONS: list[str] = ['3.11', '3.12', '3.13']
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


def check_cli(session: nox.Session, args: list[str]) -> str:
    """Check the CLI arguments.

    Arguments:
        session: The nox session.
        args: The available CLI arguments.
    """
    available_args = ', '.join([f'`{arg}`' for arg in args])
    msg = f'Available subcommands are one of {available_args}.'
    session_args = list(session.posargs)
    if not session_args:
        session_args = ['all']
    elif len(session_args) > 1 or session_args[0] not in args:
        session.skip(f'{msg} Instead `{" ".join(session_args)}` was given')
    return session_args[0]


@nox.session
def clean(session: nox.Session) -> None:
    """Clean build artifacts and cache files.

    Arguments:
        session: The nox session.
    """
    paths = [
        '.mypy_cache',
        '.pytest_cache',
        'tests/.pytest_cache',
        'build',
        'dist',
        'htmlcov',
        'pip-wheel-metadata',
        'site',
        '__pycache__',
        'docs/generated',
        '.nox',
        '.ruff_cache',
        'coverage.xml',
        'pdm.lock',
    ]
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
    for cache_path in Path().rglob('__pycache__'):
        shutil.rmtree(cache_path, ignore_errors=True)
    for file in Path().rglob('*.rej'):
        if file.exists():
            file.unlink()
    for file in Path().rglob('.coverage*'):
        if file.exists():
            file.unlink()


@nox.session
def docs(session: nox.Session) -> None:
    """Build or serve the documentation.

    Arguments:
        session: The nox session.
    """
    arg = check_cli(session, ['serve', 'build'])
    session.run('pdm', 'install', '-dG', 'docs', external=True)
    session.run('properdocs', arg, env={'DISABLE_MKDOCS_2_WARNING': 'true'})


@nox.session
@nox.parametrize('file', FILES)
def formatting(session: nox.Session, file: str) -> None:
    """Format the code.

    Arguments:
        session: The nox session.
        file: The file to be formatted.
    """
    arg = check_cli(session, ['all', 'code', 'docstrings'])
    session.run('pdm', 'install', '-dG', 'formatting', '--no-default', external=True)
    if arg in ['code', 'all']:
        session.run('black', file)
    if arg in ['docstrings', 'all']:
        session.run('docformatter', file)


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize('file', FILES)
def checks(session: nox.Session, file: str) -> None:
    """Check code quality, dependencies or type annotations.

    Arguments:
        session: The nox session.
        file: The file to be checked.
    """
    arg = check_cli(session, ['all', 'quality', 'dependencies', 'types', 'security', 'docs'])
    session.run('pdm', 'install', '-dG', 'checks', '--no-default', external=True)
    if arg in ['quality', 'all']:
        session.run('ruff', 'check', file)
    if arg in ['types', 'all']:
        session.run('mypy', file)
    if arg in ['security', 'all']:
        session.run('bandit', '-c', 'pyproject.toml', '-r', file)
    if arg in ['docs', 'all'] and file == 'src':
        session.run('interrogate', file)
    if arg in ['dependencies', 'all']:
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
        session.run('pip-audit', '-r', requirements_path)


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
    session.run('coverage', 'xml')


@nox.session
def changelog(session: nox.Session) -> None:
    """Build the changelog.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'changelog', '--no-default', external=True)
    from git_changelog.cli import build_and_render  # noqa: PLC0415

    build_and_render(**CHANGELOG_ARGS)


@nox.session
def release(session: nox.Session) -> None:
    """Kick off a release process.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'changelog', '--no-default', external=True)
    from git_changelog.cli import build_and_render  # noqa: PLC0415

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
    session.run('gh', 'pr', 'create', '--base', 'main', '--fill', external=True)
    session.run('gh', 'pr', 'merge', '--rebase', '--delete-branch', external=True)

    # Create and push the tag, which triggers the release workflow to build and
    # publish the distribution to GitHub and PyPI.
    session.run('git', 'checkout', 'main', external=True)
    session.run('git', 'pull', '--rebase', external=True)
    session.run('git', 'tag', version, external=True)
    session.run('git', 'push', 'origin', version, external=True)
