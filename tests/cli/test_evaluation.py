"""Test the bettor commands."""

from pathlib import Path

import pytest
from sportsbet.cli import main

DATALOADER_CONFIG_PATH = Path(__file__).parent / 'configs' / 'dataloader.py'
BETTOR_CONFIG_PATH = Path(__file__).parent / 'configs' / 'bettor.py'


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
def test_bettor_backtest(capsys):
    """Test bettor backtest command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'training', '-d', DATALOADER_CONFIG_PATH])
    with pytest.raises(SystemExit):
        main(['bettor', 'backtest', '-b', BETTOR_CONFIG_PATH, '-d', DATALOADER_CONFIG_PATH])
    captured = capsys.readouterr()
    assert 'Backtesting results' in captured.out
    assert (BETTOR_CONFIG_PATH.parent / 'bettor.pkl').exists()
    (BETTOR_CONFIG_PATH.parent / 'bettor.pkl').unlink()
    (DATALOADER_CONFIG_PATH.parent / 'dataloader.pkl').unlink()


@pytest.mark.xdist_group(name='serial')
def test_bettor_bet(capsys):
    """Test bettor bet command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'training', '-d', DATALOADER_CONFIG_PATH])
    with pytest.raises(SystemExit):
        main(['dataloader', 'fixtures', '-d', DATALOADER_CONFIG_PATH])
    with pytest.raises(SystemExit):
        main(['bettor', 'bet', '-b', BETTOR_CONFIG_PATH, '-d', DATALOADER_CONFIG_PATH])
    captured = capsys.readouterr()
    assert 'Value bets' in captured.out
    assert (BETTOR_CONFIG_PATH.parent / 'bettor.pkl').exists()
    (BETTOR_CONFIG_PATH.parent / 'bettor.pkl').unlink()
    (DATALOADER_CONFIG_PATH.parent / 'dataloader.pkl').unlink()
