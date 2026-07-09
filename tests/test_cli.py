"""Test that the CLI produces results equivalent to the API (SC-005 parity)."""

import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from sportsbet.cli import main
from sportsbet.datasets import DummySoccerDataLoader
from sportsbet.evaluation import OddsComparisonBettor, backtest


@pytest.mark.xdist_group(name='serial')
def test_cli_training_matches_api(cli_runner, cli_config_path):
    """Test CLI-extracted training data matches the equivalent API extraction."""
    result = cli_runner.invoke(
        main,
        ['dataloader', 'training', '-c', str(cli_config_path), '-d', str(cli_config_path.parent)],
    )
    assert result.exit_code == 0, result.output
    data_path = cli_config_path.parent / 'sports-betting-data'
    X_cli = pd.read_csv(data_path / 'X_train.csv', index_col=0)

    loader = DummySoccerDataLoader(param_grid={'league': ['England', 'Spain']})
    X_api, _, _ = loader.extract_train_data(odds_type='bet365', target_event_status='postplay')

    assert list(X_cli.columns) == list(X_api.columns)
    assert len(X_cli) == len(X_api)


@pytest.mark.xdist_group(name='serial')
def test_cli_backtest_matches_api(cli_runner, cli_config_path):
    """Test CLI backtest matches the equivalent API backtest per period."""
    result = cli_runner.invoke(
        main,
        ['bettor', 'backtest', '-c', str(cli_config_path), '-d', str(cli_config_path.parent)],
    )
    assert result.exit_code == 0, result.output
    data_path = cli_config_path.parent / 'sports-betting-data'
    cli_results = pd.read_csv(data_path / 'backtesting_results.csv')

    loader = DummySoccerDataLoader(param_grid={'league': ['England', 'Spain']})
    X, Y, O = loader.extract_train_data(odds_type='bet365', target_event_status='postplay')
    api_results = backtest(OddsComparisonBettor(alpha=0.03), X, Y, O, cv=TimeSeriesSplit(2))

    assert len(cli_results) == len(api_results)
    assert cli_results['Number of bets'].tolist() == api_results['Number of bets'].tolist()
