"""Test the dataloader commands."""

import pytest

from sportsbet.cli import main

SELECTION = ['--stats', 'football-data', '--odds', 'football-data', '--league', 'England']
MARKET = ['--odds-type', 'market_average', '--target-event-status', 'postplay']


def test_a_command_needs_to_be_told_where_the_data_comes_from(cli_runner):
    """Test the one thing a command cannot guess is asked for.

    The feed decides what is in the data, what it costs and who may redistribute it, and it decides the sport, so the
    sport is never asked for either.
    """
    result = cli_runner.invoke(main, ['dataloader', 'params'])
    assert result.exit_code != 0, result.output
    assert '--stats' in result.output


def test_a_source_nobody_publishes_says_which_ones_there_are(cli_runner):
    """Test a source that does not exist is answered with the ones that do."""
    result = cli_runner.invoke(main, ['dataloader', 'params', '--stats', 'quidditch'])
    assert result.exit_code != 0, result.output
    assert 'football-data' in result.output


def test_params(cli_runner, offline_dataloader):
    """Test what can be selected is shown."""
    result = cli_runner.invoke(main, ['dataloader', 'params', *SELECTION])
    assert result.exit_code == 0, result.output
    assert 'Available parameters' in result.output


@pytest.mark.xdist_group(name='serial')
def test_odds_types(cli_runner, offline_dataloader):
    """Test the odds that can be extracted are shown."""
    result = cli_runner.invoke(main, ['dataloader', 'odds-types', *SELECTION])
    assert result.exit_code == 0, result.output
    assert 'Available odds types' in result.output


@pytest.mark.xdist_group(name='serial')
def test_train_extract(cli_runner, offline_dataloader, tmp_path):
    """Test the training data is extracted and the dataloader saved."""
    path = tmp_path / 'dataloader.pkl'
    result = cli_runner.invoke(main, ['dataloader', 'train', 'extract', *SELECTION, *MARKET, '-o', str(path)])
    assert result.exit_code == 0, result.output
    assert 'Training input data' in result.output
    assert path.exists()


@pytest.mark.xdist_group(name='serial')
def test_fixtures_extract(cli_runner, saved_fixtures_dataloader, tmp_path):
    """Test the fixtures are extracted from a saved dataloader and written."""
    result = cli_runner.invoke(
        main,
        ['dataloader', 'fixtures', 'extract', '--dataloader', str(saved_fixtures_dataloader), '-o', str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert 'Fixtures input data' in result.output
    assert (tmp_path / 'sports-betting-data' / 'X_fix.csv').exists()


def test_a_moment_that_is_not_one_says_how_to_spell_one(cli_runner, monkeypatch, tmp_path):
    """Test a moment that cannot be read is answered with how a moment is written."""
    monkeypatch.setenv('ODDS_API_KEY', 'not-used-offline')
    result = cli_runner.invoke(
        main,
        [
            'dataloader',
            'train',
            'extract',
            '--stats',
            'nba',
            '--odds',
            'odds-api',
            '--odds-moment',
            'halftime',
            '-o',
            str(tmp_path / 'dataloader.pkl'),
        ],
    )
    assert 'inplay:45' in result.output


def test_an_alias_that_is_not_one_says_how_to_spell_one(cli_runner, monkeypatch, tmp_path):
    """Test an alias that cannot be read is answered with how an alias is written."""
    monkeypatch.setenv('ODDS_API_KEY', 'not-used-offline')
    result = cli_runner.invoke(
        main,
        [
            'dataloader',
            'train',
            'extract',
            '--stats',
            'nba',
            '--odds',
            'odds-api',
            '--alias',
            'nonsense',
            '-o',
            str(tmp_path / 'dataloader.pkl'),
        ],
    )
    assert 'Olimpia Milano' in result.output


@pytest.mark.xdist_group(name='serial')
def test_an_odds_type_that_does_not_exist_says_what_there_is(cli_runner, offline_dataloader, tmp_path):
    """Test what the library says is what the user reads, rather than a stack trace."""
    result = cli_runner.invoke(
        main,
        [
            'dataloader',
            'train',
            'extract',
            *SELECTION,
            '--odds-type',
            'nonexistent',
            '-o',
            str(tmp_path / 'dataloader.pkl'),
        ],
    )
    exit_code = 1
    assert result.exit_code == exit_code
    assert 'market_average' in result.output
    assert 'Traceback' not in result.output
