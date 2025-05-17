"""Test the dataloader commands."""

import pytest

from sportsbet.cli import main


def test_dataloader(cli_runner):
    """Test dataloader command."""
    result = cli_runner.invoke(main, ['dataloader'])
    exit_code = 2
    assert result.exit_code == exit_code, result.output
    assert 'Use or create a dataloader.' in result.output


def test_dataloader_help(cli_runner):
    """Test dataloader help command."""
    result = cli_runner.invoke(main, ['dataloader', '--help'])
    assert result.exit_code == 0, result.output
    assert 'Use or create a dataloader.' in result.output


def test_dataloader_params_error(cli_runner):
    """Test dataloader param command, missing configuration path."""
    result = cli_runner.invoke(main, ['dataloader', 'params'])
    assert result.exit_code != 0, result.output
    assert 'Error: Missing option \'--config-path\' / \'-c\'.' in result.output


def test_dataloader_params(cli_runner, cli_config_path):
    """Test dataloader param command."""
    result = cli_runner.invoke(main, ['dataloader', 'params', '-c', cli_config_path])
    assert result.exit_code == 0, result.output
    assert 'Available parameters' in result.output


def test_dataloader_odds_types_error(cli_runner):
    """Test dataloader odds-types command, missing name."""
    result = cli_runner.invoke(main, ['dataloader', 'odds-types'])
    assert result.exit_code != 0, result.output
    assert 'Error: Missing option \'--config-path\' / \'-c\'.' in result.output


def test_dataloader_odds_types(cli_runner, cli_config_path):
    """Test dataloader odds-types command."""
    result = cli_runner.invoke(main, ['dataloader', 'odds-types', '-c', cli_config_path])
    assert result.exit_code == 0, result.output
    assert 'Available odds types' in result.output


@pytest.mark.xdist_group(name='serial')
def test_dataloader_training(cli_runner, cli_config_path):
    """Test dataloader training command."""
    result = cli_runner.invoke(
        main,
        [
            'dataloader',
            'training',
            '-c',
            cli_config_path,
            '-d',
            str(cli_config_path.parent),
        ],
    )
    data_path = cli_config_path.parent / 'sports-betting-data'
    assert result.exit_code == 0, result.output
    assert 'Training input data' in result.output
    assert 'Training output data' in result.output
    assert 'Training odds data' in result.output
    assert (data_path / 'X_train.csv').exists()
    assert (data_path / 'Y_train.csv').exists()
    assert (data_path / 'O_train.csv').exists()


@pytest.mark.xdist_group(name='serial')
def test_dataloader_fixtures(cli_runner, cli_config_path):
    """Test dataloader fixtures command."""
    result = cli_runner.invoke(
        main,
        [
            'dataloader',
            'fixtures',
            '-c',
            cli_config_path,
            '-d',
            str(cli_config_path.parent),
        ],
    )
    data_path = cli_config_path.parent / 'sports-betting-data'
    assert result.exit_code == 0, result.output
    assert 'Fixtures input data' in result.output
    assert 'Fixtures odds data' in result.output
    assert (data_path / 'X_fix.csv').exists()
    assert (data_path / 'O_fix.csv').exists()
