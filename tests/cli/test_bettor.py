"""Test the bettor commands."""

import pytest

from sportsbet.cli import main


def test_bettor(cli_runner):
    """Test bettor command."""
    result = cli_runner.invoke(main, ['bettor'])
    exit_code = 2
    assert result.exit_code == exit_code, result.output
    assert 'Backtest a bettor and predict the value bets.' in result.output


def test_bettor_help(cli_runner):
    """Test bettor help command."""
    result = cli_runner.invoke(main, ['bettor', '--help'])
    assert result.exit_code == 0, result.output
    assert 'Backtest a bettor and predict the value bets.' in result.output


@pytest.mark.xdist_group(name='serial')
def test_bettor_backtest(cli_runner, cli_config_path):
    """Test bettor backtest command."""
    result = cli_runner.invoke(
        main,
        [
            'bettor',
            'backtest',
            '-c',
            cli_config_path,
            '-d',
            str(cli_config_path.parent),
        ],
    )
    data_path = cli_config_path.parent / 'sports-betting-data'
    assert result.exit_code == 0, result.output
    assert 'Backtesting results' in result.output
    assert (data_path / 'backtesting_results.csv').exists()


@pytest.mark.xdist_group(name='serial')
def test_bettor_bet(cli_runner, cli_config_path):
    """Test bettor bet command."""
    result = cli_runner.invoke(
        main,
        [
            'bettor',
            'bet',
            '-c',
            cli_config_path,
            '-d',
            str(cli_config_path.parent),
        ],
    )
    data_path = cli_config_path.parent / 'sports-betting-data'
    assert result.exit_code == 0, result.output
    assert 'Value bets' in result.output
    assert (data_path / 'value_bets.csv').exists()
