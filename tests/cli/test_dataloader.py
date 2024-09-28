"""Test the dataloader commands."""

import pytest

from sportsbet.cli import main


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
    """Test dataloader param command, missing configuration path."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'params'])
    captured = capsys.readouterr()
    assert 'Error: Missing option \'--config-path\' / \'-c\'.' in captured.err


def test_dataloader_params(capsys, cli_config_path):
    """Test dataloader param command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'params', '-c', cli_config_path])
    captured = capsys.readouterr()
    assert 'Available parameters' in captured.out


def test_dataloader_odds_types_error(capsys):
    """Test dataloader odds-types command, missing name."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'odds-types'])
    captured = capsys.readouterr()
    assert 'Error: Missing option \'--config-path\' / \'-c\'.' in captured.err


def test_dataloader_odds_types(capsys, cli_config_path):
    """Test dataloader odds-types command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'odds-types', '-c', cli_config_path])
    captured = capsys.readouterr()
    assert 'Available odds types' in captured.out


@pytest.mark.xdist_group(name='serial')
def test_dataloader_training(capsys, cli_config_path):
    """Test dataloader training command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'training', '-c', cli_config_path, '-d', cli_config_path.parent])
    captured = capsys.readouterr()
    data_path = cli_config_path.parent / 'sports-betting-data'
    assert 'Training input data' in captured.out
    assert 'Training output data' in captured.out
    assert 'Training odds data' in captured.out
    assert (data_path / 'X_train.csv').exists()
    assert (data_path / 'Y_train.csv').exists()
    assert (data_path / 'O_train.csv').exists()


@pytest.mark.xdist_group(name='serial')
def test_dataloader_fixtures(capsys, cli_config_path):
    """Test dataloader fixtures command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'fixtures', '-c', cli_config_path, '-d', cli_config_path.parent])
    captured = capsys.readouterr()
    data_path = cli_config_path.parent / 'sports-betting-data'
    assert 'Fixtures input data' in captured.out
    assert 'Fixtures odds data' in captured.out
    assert (data_path / 'X_fix.csv').exists()
    assert (data_path / 'O_fix.csv').exists()
