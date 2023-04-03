"""Test the `cli` module."""

from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp

import pandas as pd
import pytest
from sportsbet.cli import main
from sportsbet.datasets import SoccerDataLoader, load


def test_main(capsys):
    """Test main command."""
    with pytest.raises(SystemExit):
        main([])
    captured = capsys.readouterr()
    assert 'CLI for sports-betting' in captured.out


def test_main_help(capsys):
    """Test main help command."""
    with pytest.raises(SystemExit):
        main(['--help'])
    captured = capsys.readouterr()
    assert 'CLI for sports-betting' in captured.out


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


def test_dataloader_names(capsys):
    """Test dataloader names command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'names'])
    captured = capsys.readouterr()
    assert 'Available names' in captured.out


def test_dataloader_params_error(capsys):
    """Test dataloader param command, missing name."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'params'])
    captured = capsys.readouterr()
    assert 'Missing option \'--name\'.' in captured.err


def test_dataloader_params(capsys):
    """Test dataloader param command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'params', '--name', 'soccer'])
    captured = capsys.readouterr()
    assert 'Available parameters' in captured.out


def test_dataloader_create_error_name(capsys):
    """Test dataloader create command, missing name."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'create', '--dataloader-path', 'dataloader.pkl'])
    captured = capsys.readouterr()
    assert 'Missing option \'--name\'.' in captured.err


def test_dataloader_create_error_dataloader_path(capsys):
    """Test dataloader create command, missing dataloder path."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'create', '--name', 'soccer'])
    captured = capsys.readouterr()
    assert 'Missing option \'--dataloader-path\'.' in captured.err


def test_dataloader_create_help(capsys):
    """Test dataloader create help command."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'create', '--help'])
    captured = capsys.readouterr()
    assert 'Create and save a dataloader.' in captured.out


def test_dataloader_create(capsys):
    """Test dataloader create command."""
    _, path = mkstemp()
    with pytest.raises(SystemExit):
        main(['dataloader', 'create', '--name', 'soccer', '--param-grid', 'league: Italy', '--dataloader-path', path])
    dataloader = load(path)
    captured = capsys.readouterr()
    assert not captured.out
    assert isinstance(dataloader, SoccerDataLoader)


def test_dataloader_odds_types_error(capsys):
    """Test dataloader odds-types command, missing name."""
    with pytest.raises(SystemExit):
        main(['dataloader', 'odds-types'])
    captured = capsys.readouterr()
    assert 'Missing option \'--dataloader-path\'.' in captured.err


def test_dataloader_odds_types(capsys):
    """Test dataloader odds-types command."""
    _, dataloader_path = mkstemp()
    with pytest.raises(SystemExit):
        main(
            [
                'dataloader',
                'create',
                '--name',
                'soccer',
                '--param-grid',
                'league: Greece',
                '--dataloader-path',
                dataloader_path,
            ],
        )
    with pytest.raises(SystemExit):
        main(['dataloader', 'odds-types', '--dataloader-path', dataloader_path])
    captured = capsys.readouterr()
    assert 'Available odds types' in captured.out


def test_dataloader_training(capsys):
    """Test dataloader training command."""
    _, dataloader_path = mkstemp()
    data_path = TemporaryDirectory()
    with pytest.raises(SystemExit):
        main(
            [
                'dataloader',
                'create',
                '--name',
                'soccer',
                '--param-grid',
                'league: Greece',
                '--dataloader-path',
                dataloader_path,
            ],
        )
    with pytest.raises(SystemExit):
        main(
            [
                'dataloader',
                'training',
                '--odds-type',
                'betwin',
                '--drop-na-thres',
                '1.0',
                '--dataloader-path',
                dataloader_path,
                '--data-path',
                data_path.name,
            ],
        )
    captured = capsys.readouterr()
    assert 'Training input data' in captured.out
    assert 'Training output data' in captured.out
    assert 'Training odds data' in captured.out
    assert isinstance(pd.read_csv(Path(data_path.name) / 'X_train.csv'), pd.DataFrame)
    assert isinstance(pd.read_csv(Path(data_path.name) / 'Y_train.csv'), pd.DataFrame)
    assert isinstance(pd.read_csv(Path(data_path.name) / 'O_train.csv'), pd.DataFrame)


def test_dataloader_fixtures(capsys):
    """Test dataloader training command."""
    _, dataloader_path = mkstemp()
    data_path = TemporaryDirectory()
    with pytest.raises(SystemExit):
        main(
            [
                'dataloader',
                'create',
                '--name',
                'soccer',
                '--param-grid',
                'league: Greece',
                '--dataloader-path',
                dataloader_path,
            ],
        )
    with pytest.raises(SystemExit):
        main(
            [
                'dataloader',
                'training',
                '--odds-type',
                'betwin',
                '--drop-na-thres',
                '1.0',
                '--dataloader-path',
                dataloader_path,
                '--data-path',
                data_path.name,
            ],
        )
    with pytest.raises(SystemExit):
        main(['dataloader', 'fixtures', '--dataloader-path', dataloader_path, '--data-path', data_path.name])
    captured = capsys.readouterr()
    if 'Fixtures data were empty' not in captured.out:
        assert 'Fixtures input data' in captured.out
        assert 'Fixtures odds data' in captured.out
        assert isinstance(pd.read_csv(Path(data_path.name) / 'X_fix.csv'), pd.DataFrame)
        assert isinstance(pd.read_csv(Path(data_path.name) / 'O_fix.csv'), pd.DataFrame)
