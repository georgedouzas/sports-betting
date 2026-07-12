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
