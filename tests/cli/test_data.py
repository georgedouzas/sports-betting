"""Test the data commands."""

import pytest

from sportsbet.cli import main

DUMMY = ['--sport', 'dummy', '--league', 'England']
MARKET = ['--odds-type', 'market_average', '--target-event-status', 'postplay']


def test_data(cli_runner):
    """Test the data command."""
    result = cli_runner.invoke(main, ['data'])
    exit_code = 2
    assert result.exit_code == exit_code, result.output
    assert 'Select, download and extract data.' in result.output


def test_data_help(cli_runner):
    """Test the data help command."""
    result = cli_runner.invoke(main, ['data', '--help'])
    assert result.exit_code == 0, result.output
    assert 'Select, download and extract data.' in result.output


def test_a_command_needs_to_be_told_the_sport(cli_runner):
    """Test the one thing a command cannot guess is asked for."""
    result = cli_runner.invoke(main, ['data', 'params'])
    assert result.exit_code != 0, result.output
    assert '--sport' in result.output


def test_a_sport_nobody_plays_says_which_ones_there_are(cli_runner):
    """Test a sport that does not exist is answered with the ones that do."""
    result = cli_runner.invoke(main, ['data', 'params', '--sport', 'quidditch'])
    assert result.exit_code != 0, result.output
    assert 'soccer' in result.output


def test_data_params(cli_runner):
    """Test what can be selected is shown."""
    result = cli_runner.invoke(main, ['data', 'params', *DUMMY])
    assert result.exit_code == 0, result.output
    assert 'Available parameters' in result.output


@pytest.mark.xdist_group(name='serial')
def test_data_odds_types(cli_runner):
    """Test the odds that can be extracted are shown."""
    result = cli_runner.invoke(main, ['data', 'odds-types', *DUMMY])
    assert result.exit_code == 0, result.output
    assert 'Available odds types' in result.output


@pytest.mark.xdist_group(name='serial')
def test_data_prepare_dry_run(cli_runner):
    """Test a preparation can be priced without being done."""
    result = cli_runner.invoke(main, ['data', 'prepare', *DUMMY, '--dry-run'])
    assert result.exit_code == 0, result.output
    assert 'Preparation (dry run)' in result.output


@pytest.mark.xdist_group(name='serial')
def test_data_training(cli_runner, tmp_path):
    """Test the training data is extracted and written."""
    result = cli_runner.invoke(main, ['data', 'training', *DUMMY, *MARKET, '-o', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert 'Training input data' in result.output
    assert (tmp_path / 'sports-betting-data' / 'X_train.csv').exists()


@pytest.mark.xdist_group(name='serial')
def test_data_fixtures(cli_runner, tmp_path):
    """Test the fixtures are extracted and written."""
    result = cli_runner.invoke(main, ['data', 'fixtures', *DUMMY, *MARKET, '-o', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert 'Fixtures input data' in result.output
    assert (tmp_path / 'sports-betting-data' / 'X_fix.csv').exists()


def test_a_moment_that_is_not_one_says_how_to_spell_one(cli_runner, monkeypatch):
    """Test a moment that cannot be read is answered with how a moment is written."""
    monkeypatch.setenv('ODDS_API_KEY', 'not-used-offline')
    result = cli_runner.invoke(
        main,
        [
            'data',
            'prepare',
            '--sport',
            'basketball',
            '--stats',
            'nba',
            '--odds',
            'odds-api',
            '--odds-moment',
            'halftime',
        ],
    )
    assert 'inplay:45' in result.output


def test_an_alias_that_is_not_one_says_how_to_spell_one(cli_runner, monkeypatch):
    """Test an alias that cannot be read is answered with how an alias is written."""
    monkeypatch.setenv('ODDS_API_KEY', 'not-used-offline')
    result = cli_runner.invoke(
        main,
        ['data', 'prepare', '--sport', 'basketball', '--stats', 'nba', '--odds', 'odds-api', '--alias', 'nonsense'],
    )
    assert 'Olimpia Milano' in result.output


@pytest.mark.xdist_group(name='serial')
def test_an_extraction_never_downloads(cli_runner, tmp_path):
    """Test an extraction says the data is missing rather than quietly going and buying it.

    Downloading is a separate step because it is the only one that costs anything. A command that extracted and
    downloaded in the same breath could spend thousands of credits on an odds vendor without anyone asking for it.
    """
    result = cli_runner.invoke(
        main,
        ['data', 'training', '--sport', 'soccer', '--league', 'England', '--year', '2025', '--store', str(tmp_path)],
    )
    exit_code = 1
    assert result.exit_code == exit_code
    assert 'has not been downloaded' in result.output
    assert 'sportsbet data prepare' in result.output


@pytest.mark.xdist_group(name='serial')
def test_an_odds_type_that_does_not_exist_says_what_there_is(cli_runner):
    """Test what the library says is what the user reads, rather than a stack trace."""
    result = cli_runner.invoke(main, ['data', 'training', *DUMMY, '--odds-type', 'nonexistent'])
    exit_code = 1
    assert result.exit_code == exit_code
    assert 'market_average' in result.output
    assert 'Traceback' not in result.output
