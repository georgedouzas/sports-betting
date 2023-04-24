"""Test the dataloader commands."""

from pathlib import Path

import pytest
from sportsbet.cli import main

DATALOADER_CONFIG_PATH = Path(__file__).parent / 'configs' / 'dataloader.py'


def test_dataloader(capsys):
    """Test dataloader command."""
    with pytest.raises(SystemExit):
        main(['dataloader'])
    captured = capsys.readouterr()
    assert 'Use or create a dataloader.' in captured.out


def test_dataloader_help(capsys):
    """Test dataloader help command."""
    with pytest.raises(SystemExit):
        main(['dataloader', '--help'])
    captured = capsys.readouterr()
    assert 'Use or create a dataloader.' in captured.out


def test_dataloader_params_error(capsys):
    """Test dataloader param command, missing name."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'params'])
    captured = capsys.readouterr()
    assert 'Error: Missing option \'--dataloader-config-path\' / \'-d\'.' in captured.err


def test_dataloader_params(capsys):
    """Test dataloader param command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'params', '-d', DATALOADER_CONFIG_PATH])
    captured = capsys.readouterr()
    assert 'Available parameters' in captured.out


def test_dataloader_odds_types_error(capsys):
    """Test dataloader odds-types command, missing name."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'odds-types'])
    captured = capsys.readouterr()
    assert 'Error: Missing option \'--dataloader-config-path\' / \'-d\'.' in captured.err


def test_dataloader_odds_types(capsys):
    """Test dataloader odds-types command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'odds-types', '-d', DATALOADER_CONFIG_PATH])
    captured = capsys.readouterr()
    assert 'Available odds types' in captured.out


@pytest.mark.xdist_group(name='serial')
def test_dataloader_training(capsys):
    """Test dataloader training command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'training', '-d', DATALOADER_CONFIG_PATH])
    captured = capsys.readouterr()
    assert 'Training input data' in captured.out
    assert 'Training output data' in captured.out
    assert 'Training odds data' in captured.out
    assert (DATALOADER_CONFIG_PATH.parent / 'dataloader.pkl').exists()
    (DATALOADER_CONFIG_PATH.parent / 'dataloader.pkl').unlink()


@pytest.mark.xdist_group(name='serial')
def test_dataloader_fixtures(capsys):
    """Test dataloader fixtures command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'training', '-d', DATALOADER_CONFIG_PATH])
    with pytest.raises(SystemExit):
        main(['dataloader', 'fixtures', '-d', DATALOADER_CONFIG_PATH])
    captured = capsys.readouterr()
    assert 'Fixtures input data' in captured.out
    assert (DATALOADER_CONFIG_PATH.parent / 'dataloader.pkl').exists()
    (DATALOADER_CONFIG_PATH.parent / 'dataloader.pkl').unlink()
