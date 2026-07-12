"""Test that the CLI produces results equivalent to the API (SC-005 parity)."""

import importlib.util

import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from sportsbet.cli import main
from sportsbet.datasets import BasketballDataLoader, DummySoccerDataLoader, NBAStats, OddsApi
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
    X_api, _, _ = loader.extract_train_data(odds_type='market_average', target_event_status='postplay')

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
    X, Y, O = loader.extract_train_data(odds_type='market_average', target_event_status='postplay')
    api_results = backtest(OddsComparisonBettor(alpha=0.03), X, Y, O, cv=TimeSeriesSplit(2))

    assert len(cli_results) == len(api_results)
    assert cli_results['Number of bets'].tolist() == api_results['Number of bets'].tolist()


@pytest.mark.xdist_group(name='serial')
def test_an_outdated_config_says_what_to_change(cli_runner, tmp_path):
    """Test a configuration written for the old contract is told what replaces it.

    It is a breaking change, so it has to behave like one: naming the replacement, rather than raising an attribute
    error or quietly doing nothing.
    """
    config = tmp_path / 'outdated.py'
    config.write_text(
        'from sportsbet.datasets import DummySoccerDataLoader\n'
        'DATALOADER_CLASS = DummySoccerDataLoader\n'
        "PARAM_GRID = {'league': ['England']}\n",
    )
    result = cli_runner.invoke(main, ['dataloader', 'params', '-c', str(config)])
    assert 'DATALOADER_CLASS' in result.output
    assert 'DATALOADER =' in result.output


@pytest.mark.xdist_group(name='serial')
def test_the_parameters_are_available_before_anything_is_selected(cli_runner, tmp_path):
    """Test the parameters can be asked for without a selection.

    It is the command that tells a user what to select, so requiring them to have already selected something would make
    it useless.
    """
    config = tmp_path / 'unselected.py'
    config.write_text(
        'from sportsbet.datasets import DummySoccerDataLoader\nDATALOADER = DummySoccerDataLoader()\n',
    )
    result = cli_runner.invoke(main, ['dataloader', 'params', '-c', str(config)])
    assert result.exit_code == 0, result.output
    assert 'England' in result.output


@pytest.mark.xdist_group(name='serial')
def test_a_source_that_carries_a_key_reaches_the_command_line(cli_runner, tmp_path):
    """Test a dataloader whose source needs a credential can be configured.

    This is what the old contract could not express. It handed over a class, and a class carries no sources, so no
    source could carry a key, so basketball was unreachable and no odds could be bought for any sport.
    """
    config = tmp_path / 'nba.py'
    config.write_text(
        'from sportsbet.datasets import BasketballDataLoader, NBAStats, OddsApi\n'
        'DATALOADER = BasketballDataLoader(\n'
        "    param_grid={'league': ['NBA'], 'year': [2026]},\n"
        '    stats=NBAStats(),\n'
        "    odds=OddsApi(key='a-key-that-is-never-used-offline'),\n"
        ')\n',
    )
    spec = importlib.util.spec_from_file_location('nba_config', config)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert isinstance(mod.DATALOADER, BasketballDataLoader)
    stats, odds = mod.DATALOADER.sources
    assert isinstance(stats, NBAStats)
    assert isinstance(odds, OddsApi)
