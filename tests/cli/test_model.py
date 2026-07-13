"""Test the model commands."""

import pytest

from sportsbet.cli import main

DUMMY = ['--sport', 'dummy', '--league', 'England']
MARKET = ['--odds-type', 'market_average', '--target-event-status', 'postplay']
ODDS_COMPARISON = ['--model', 'odds-comparison', '--alpha', '0.03', '--cv', '2']


def test_model(cli_runner):
    """Test the model command."""
    result = cli_runner.invoke(main, ['model'])
    exit_code = 2
    assert result.exit_code == exit_code, result.output
    assert 'Backtest a betting model' in result.output


def test_model_help(cli_runner):
    """Test the model help command."""
    result = cli_runner.invoke(main, ['model', '--help'])
    assert result.exit_code == 0, result.output
    assert 'Backtest a betting model' in result.output


def test_a_command_needs_to_be_told_the_model(cli_runner):
    """Test the model cannot be guessed, so it is asked for."""
    result = cli_runner.invoke(main, ['model', 'backtest', *DUMMY])
    assert result.exit_code != 0, result.output
    assert '--model' in result.output


@pytest.mark.xdist_group(name='serial')
def test_model_backtest(cli_runner, tmp_path):
    """Test a model is backtested and the results written."""
    result = cli_runner.invoke(
        main,
        ['model', 'backtest', *DUMMY, *MARKET, *ODDS_COMPARISON, '-o', str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert 'Backtesting results' in result.output
    assert (tmp_path / 'sports-betting-data' / 'backtesting_results.csv').exists()


@pytest.mark.xdist_group(name='serial')
def test_model_bet(cli_runner, tmp_path):
    """Test the value bets are predicted and written."""
    result = cli_runner.invoke(main, ['model', 'bet', *DUMMY, *MARKET, *ODDS_COMPARISON, '-o', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert 'Value bets' in result.output
    assert (tmp_path / 'sports-betting-data' / 'value_bets.csv').exists()


@pytest.mark.xdist_group(name='serial')
def test_a_ready_made_classifier_needs_no_python(cli_runner):
    """Test the ready-made models cover the ordinary case without a line of code being written."""
    result = cli_runner.invoke(main, ['model', 'backtest', *DUMMY, *MARKET, '--model', 'logistic', '--cv', '2'])
    assert result.exit_code == 0, result.output
    assert 'Backtesting results' in result.output


@pytest.mark.xdist_group(name='serial')
def test_a_model_can_bet_on_a_single_market(cli_runner):
    """Test betting on one market works, as betting on all of them does.

    The probabilities were clipped in an array that is read-only when a single market produces it, so this crashed and
    the default never did.
    """
    result = cli_runner.invoke(
        main,
        [
            'model',
            'backtest',
            *DUMMY,
            *MARKET,
            '--model',
            'odds-comparison',
            '--betting-market',
            'home_win',
            '--cv',
            '2',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Backtesting results' in result.output


@pytest.mark.xdist_group(name='serial')
def test_the_odds_a_model_compares_can_be_chosen(cli_runner):
    """Test the odds `odds-comparison` compares are chosen from the command line, as they are from Python."""
    result = cli_runner.invoke(
        main,
        [
            'model',
            'backtest',
            *DUMMY,
            *MARKET,
            '--model',
            'odds-comparison',
            '--model-odds-type',
            'market_average',
            '--cv',
            '2',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Backtesting results' in result.output
