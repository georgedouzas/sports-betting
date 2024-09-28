"""Test the bettor commands."""

import pytest

from sportsbet.cli import main


def test_bettor(capsys):
    """Test bettor command."""
    with pytest.raises(SystemExit):
        main(['bettor'])
    captured = capsys.readouterr()
    assert 'Backtest a bettor and predict the value bets.' in captured.out


def test_bettor_help(capsys):
    """Test bettor help command."""
    with pytest.raises(SystemExit):
        main(['bettor', '--help'])
    captured = capsys.readouterr()
    assert 'Backtest a bettor and predict the value bets.' in captured.out


@pytest.mark.xdist_group(name='serial')
def test_bettor_backtest(capsys, cli_config_path):
    """Test bettor backtest command."""
    with pytest.raises(SystemExit):
        main(['bettor', 'backtest', '-c', cli_config_path, '-d', cli_config_path.parent])
    captured = capsys.readouterr()
    data_path = cli_config_path.parent / 'sports-betting-data'
    assert 'Backtesting results' in captured.out
    assert (data_path / 'backtesting_results.csv').exists()


@pytest.mark.xdist_group(name='serial')
def test_bettor_bet(capsys, cli_config_path):
    """Test bettor bet command."""
    with pytest.raises(SystemExit):
        main(['bettor', 'bet', '-c', cli_config_path, '-d', cli_config_path.parent])
    captured = capsys.readouterr()
    data_path = cli_config_path.parent / 'sports-betting-data'
    assert 'Value bets' in captured.out
    assert (data_path / 'value_bets.csv').exists()
