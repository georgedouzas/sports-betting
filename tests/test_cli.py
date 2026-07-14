"""Test the CLI, which is told what to do in its arguments and reads no file."""

import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from sportsbet.cli import main
from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import OddsComparisonBettor, backtest
from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

SELECTION = [
    '--stats',
    'football-data',
    '--odds',
    'football-data',
    '--league',
    'England',
    '--league',
    'Spain',
]
MARKET = ['--odds-type', 'market_average', '--target-event-status', 'postplay']


@pytest.mark.xdist_group(name='serial')
def test_the_training_data_matches_the_api(cli_runner, offline_dataloader, tmp_path):
    """Test the data the command line extracts is the data the API extracts."""
    result = cli_runner.invoke(main, ['data', 'training', *SELECTION, *MARKET, '-o', str(tmp_path)])
    assert result.exit_code == 0, result.output
    X_cli = pd.read_csv(tmp_path / 'sports-betting-data' / 'X_train.csv', index_col=0)

    loader = DataLoader(param_grid={'league': ['England', 'Spain']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds())
    X_api, _, _ = loader.extract_train_data(odds_type='market_average', target_event_status='postplay', download=True)

    assert list(X_cli.columns) == list(X_api.columns)
    assert len(X_cli) == len(X_api)


@pytest.mark.xdist_group(name='serial')
def test_a_backtest_matches_the_api(cli_runner, offline_dataloader, tmp_path):
    """Test the backtest the command line runs is the backtest the API runs."""
    result = cli_runner.invoke(
        main,
        [
            'model',
            'backtest',
            *SELECTION,
            *MARKET,
            '--model',
            'odds-comparison',
            '--alpha',
            '0.03',
            '--cv',
            '2',
            '-o',
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    cli_results = pd.read_csv(tmp_path / 'sports-betting-data' / 'backtesting_results.csv')

    loader = DataLoader(param_grid={'league': ['England', 'Spain']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds())
    X, Y, O = loader.extract_train_data(odds_type='market_average', target_event_status='postplay', download=True)
    api_results = backtest(OddsComparisonBettor(alpha=0.03), X, Y, O, cv=TimeSeriesSplit(2))

    assert len(cli_results) == len(api_results)
    assert cli_results['Number of bets'].tolist() == api_results['Number of bets'].tolist()


@pytest.mark.xdist_group(name='serial')
def test_nothing_has_to_be_written_down_first(cli_runner, offline_dataloader):
    """Test a command needs no file of any kind, only what it is told."""
    result = cli_runner.invoke(main, ['data', 'params', *SELECTION])
    assert result.exit_code == 0, result.output
    assert 'England' in result.output


@pytest.mark.xdist_group(name='serial')
def test_a_source_that_needs_a_key_reads_it_from_the_environment(cli_runner, monkeypatch):
    """Test a key is never written into a command, so it stays out of a shell's history."""
    monkeypatch.setenv('ODDS_API_KEY', 'a-key-that-is-never-used-offline')
    result = cli_runner.invoke(
        main,
        [
            'data',
            'params',
            '--league',
            'NBA',
            '--stats',
            'nba',
            '--odds',
            'odds-api',
            '--odds-market',
            'h2h',
        ],
    )
    assert 'needs a key' not in result.output


@pytest.mark.xdist_group(name='serial')
def test_a_missing_key_says_which_one_is_missing(cli_runner, monkeypatch):
    """Test a key that is not there names itself, rather than failing somewhere deeper."""
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    result = cli_runner.invoke(
        main,
        ['data', 'training', '--league', 'NBA', '--stats', 'nba', '--odds', 'odds-api'],
    )
    assert 'ODDS_API_KEY' in result.output


@pytest.mark.xdist_group(name='serial')
def test_a_model_of_your_own_is_named_by_where_it_lives(cli_runner, offline_dataloader, tmp_path):
    """Test a scikit-learn model built in Python reaches the command line.

    No arrangement of arguments can describe an estimator, and inventing a syntax that tried would be a worse Python. So
    a model of your own is named as an object, and it is built where objects are built.
    """
    models = tmp_path / 'models.py'
    models.write_text(
        'from sportsbet.evaluation import OddsComparisonBettor\nBETTOR = OddsComparisonBettor(alpha=0.02)\n',
    )
    result = cli_runner.invoke(
        main,
        ['model', 'backtest', *SELECTION, *MARKET, '--model', f'{models}:BETTOR', '--cv', '2'],
    )
    assert result.exit_code == 0, result.output
    assert 'Backtesting results' in result.output


@pytest.mark.xdist_group(name='serial')
def test_a_model_that_does_not_exist_says_what_there_is(cli_runner, offline_dataloader):
    """Test a model nobody has heard of is answered with the ones there are."""
    result = cli_runner.invoke(main, ['model', 'backtest', *SELECTION, *MARKET, '--model', 'wishful-thinking'])
    assert 'odds-comparison' in result.output
