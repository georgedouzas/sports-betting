"""Development tasks."""

import os
from pathlib import Path

import nox

os.environ.update({'PDM_IGNORE_SAVED_PYTHON': '1'})

PYTHON_VERSIONS = ['3.9', '3.10']
FILES = ['src', 'tests', 'docs', 'noxfile.py']


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
        session.run('docformatter', '--in-place', '--recursive', '--close-quotes-on-newline', file)


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
        session.run('ruff', file)
    if session.posargs[0] in ['types', 'all']:
        session.run('mypy', file)
    if session.posargs[0] in ['dependencies', 'all']:
        requirements_path = (Path(session.create_tmp()) / 'requirements.txt').as_posix()
        args_groups = [['--prod']] + [['-dG', group] for group in ['tests', 'docs', 'maintenance']]
        requirements_types = zip(FILES, args_groups)
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
        session.run('pytest', '-n', 'auto', '-k', *session.posargs, 'tests', env=env)
    else:
        session.run('pytest', '-n', 'auto', 'tests', env=env)
    session.run('coverage', 'combine')
    session.run('coverage', 'report')
    session.run('coverage', 'html')


@nox.session
def changelog(session: nox.Session) -> None:
    """Build the changelog.

    Arguments:
        session: The nox session.
    """
    from git_changelog.cli import build_and_render

    session.run('pdm', 'install', '-dG', 'changelog', external=True)
    build_and_render(
        repository='.',
        output='CHANGELOG.md',
        convention='angular',
        template='keepachangelog',
        parse_trailers=True,
        parse_refs=False,
        sections=['feat', 'fix', 'docs', 'style', 'refactor', 'tests', 'chore'],
        bump_latest=True,
    )


@nox.session
def release(session: nox.Session) -> None:
    """Kick off a release process.

    Arguments:
        session: The nox session.
    """
    if not session.posargs:
        session.skip('No version was provided')
    session.run('pdm', 'install', '-dG', 'release', external=True)

    session.run('git', 'add', 'pyproject.toml', 'CHANGELOG.md', external=True)
    session.run('git', 'commit', '-m', f'chore: Release {session.posargs[0]}', external=True)
    try:
        session.run('git', 'tag', session.posargs[0], external=True)
        session.run('git', 'push', external=True)
        session.run('git', 'push', '--tags', external=True)
    finally:
        session.run('pdm', 'build', external=True)
        session.run('twine', 'upload', '--skip-existing', 'dist/*')
