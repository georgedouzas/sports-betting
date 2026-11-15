"""Development tasks."""

import os
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

import nox

os.environ.update({'PDM_IGNORE_SAVED_PYTHON': '1'})


PYTHON_VERSIONS: list[str] = ['3.12', '3.13', '3.14']
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


def check_cli(session: nox.Session, args: list[str], default: str = 'all') -> str:
    """Check the CLI arguments.

    Arguments:
        session: The nox session.
        args: The available CLI arguments.
        default: The subcommand used when none is given.
    """
    available_args = ', '.join([f'`{arg}`' for arg in args])
    msg = f'Available subcommands are one of {available_args}.'
    session_args = list(session.posargs) or [default]
    if len(session_args) > 1 or session_args[0] not in args:
        session.skip(f'{msg} Instead `{" ".join(session_args)}` was given')
    return session_args[0]


@nox.session
def clean(session: nox.Session) -> None:
    """Clean build artifacts, cache files, and the project's virtual environment.

    Arguments:
        session: The nox session.
    """
    dir_paths = [
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
        '.venv',
        '.ruff_cache',
    ]
    file_patterns = ['*.rej', '.coverage*', 'coverage.xml']
    if sys.platform == 'win32' and Path(sys.prefix).resolve().is_relative_to(Path('.venv').resolve()):
        dir_paths.remove('.venv')
        session.warn(
            'Kept .venv, since Windows cannot delete the environment this task runs from. Remove it after the '
            'task exits, with `Remove-Item -Recurse -Force .venv` in PowerShell or `rmdir /s /q .venv` in cmd.',
        )
    for dir_path in dir_paths:
        shutil.rmtree(dir_path, ignore_errors=True)
    for cache_path in Path().rglob('__pycache__'):
        shutil.rmtree(cache_path, ignore_errors=True)
    for file_pattern in file_patterns:
        for file_path in Path().rglob(file_pattern):
            if file_path.exists():
                file_path.unlink()


@nox.session
def docs(session: nox.Session) -> None:
    """Build or serve the documentation.

    Arguments:
        session: The nox session.
    """
    arg = check_cli(session, ['serve', 'build'], default='serve')
    session.run('pdm', 'install', '-dG', 'docs', external=True)
    strict = ['--strict'] if arg == 'build' else []
    session.run('properdocs', arg, *strict, env={'DISABLE_MKDOCS_2_WARNING': 'true'})


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
        session.run('pydoclint', file)
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
        config = tomllib.loads(Path('pyproject.toml').read_text())
        ignored = config.get('tool', {}).get('pip-audit', {}).get('ignore-vulns', [])
        ignore_args = [flag for vuln in ignored for flag in ('--ignore-vuln', vuln)]
        session.run('pip-audit', '-r', requirements_path, *ignore_args)
        session.run('deptry', 'src', 'tests')


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run tests and coverage.

    Arguments:
        session: The nox session.
    """
    session.run('pdm', 'install', '-dG', 'tests', '-G', 'mcp', '-G', 'execution', external=True)
    session.run('python', '-m', 'playwright', 'install', '--only-shell', 'chromium')
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

    # Create and push the release tag, which is what the pipeline reacts to
    session.run('git', 'checkout', 'main', external=True)
    session.run('git', 'pull', '--rebase', external=True)
    session.run('git', 'tag', version, external=True)
    session.run('git', 'push', 'origin', version, external=True)

    # Sync the development branch to the released main so they do not diverge.
    session.run('git', 'checkout', 'development', external=True)
    session.run('git', 'reset', '--hard', 'main', external=True)
    session.run('git', 'push', '--force-with-lease', 'origin', 'development', external=True)
