"""Test the evaluation commands, which read a saved dataloader."""

import pytest

from sportsbet.cli import main

ODDS_COMPARISON = ['--model', 'OddsComparisonBettor(alpha=0.03)']
CLASSIFIER = (
    'ClassifierBettor(make_pipeline('
    'make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["league", "home_team", "away_team"]), '
    'remainder="passthrough"), SimpleImputer(), MultiOutputClassifier(LogisticRegression(solver="liblinear"))))'
)


def test_a_command_needs_to_be_told_the_model(cli_runner, saved_dataloader):
    """Test the model cannot be guessed, so it is asked for."""
    result = cli_runner.invoke(main, ['evaluation', 'backtest', '--dataloader', str(saved_dataloader)])
    assert result.exit_code != 0, result.output
    assert '--model' in result.output


@pytest.mark.xdist_group(name='serial')
def test_backtest(cli_runner, saved_dataloader, tmp_path):
    """Test a model is backtested on the saved data and the results written."""
    result = cli_runner.invoke(
        main,
        [
            'evaluation',
            'backtest',
            '--dataloader',
            str(saved_dataloader),
            *ODDS_COMPARISON,
            '--cv',
            '2',
            '-o',
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Backtesting results' in result.output
    assert (tmp_path / 'sports-betting-data' / 'backtesting_results.csv').exists()


@pytest.mark.xdist_group(name='serial')
def test_fit_then_bet(cli_runner, saved_fixtures_dataloader, tmp_path):
    """Test a model is fitted once and then reused to bet on the upcoming matches."""
    model = tmp_path / 'model.pkl'
    fitted = cli_runner.invoke(
        main,
        ['evaluation', 'fit', '--dataloader', str(saved_fixtures_dataloader), *ODDS_COMPARISON, '-o', str(model)],
    )
    assert fitted.exit_code == 0, fitted.output
    assert model.exists()

    result = cli_runner.invoke(
        main,
        [
            'evaluation',
            'bet',
            '--dataloader',
            str(saved_fixtures_dataloader),
            '--bettor',
            str(model),
            '-o',
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Value bets' in result.output
    assert (tmp_path / 'sports-betting-data' / 'value_bets.csv').exists()


@pytest.mark.xdist_group(name='serial')
def test_bet_needs_a_fitted_model(cli_runner, saved_fixtures_dataloader):
    """Test betting cannot happen without a model saved by `fit`."""
    result = cli_runner.invoke(main, ['evaluation', 'bet', '--dataloader', str(saved_fixtures_dataloader)])
    assert result.exit_code != 0, result.output
    assert '--bettor' in result.output


@pytest.mark.xdist_group(name='serial')
def test_a_classifier_is_set_through_an_expression(cli_runner, saved_dataloader):
    """Test a scikit-learn classifier is backtested from a one-line expression, with no file written."""
    result = cli_runner.invoke(
        main,
        ['evaluation', 'backtest', '--dataloader', str(saved_dataloader), '--model', CLASSIFIER, '--cv', '2'],
    )
    assert result.exit_code == 0, result.output
    assert 'Backtesting results' in result.output


@pytest.mark.xdist_group(name='serial')
def test_a_model_can_bet_on_a_single_market(cli_runner, saved_dataloader):
    """Test backtesting one market works, as backtesting all of them does.

    The probabilities were clipped in an array that is read-only when a single market produces it, so this crashed and
    the default never did.
    """
    result = cli_runner.invoke(
        main,
        [
            'evaluation',
            'backtest',
            '--dataloader',
            str(saved_dataloader),
            '--model',
            'OddsComparisonBettor(betting_markets=["home_win"])',
            '--cv',
            '2',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Backtesting results' in result.output


@pytest.mark.xdist_group(name='serial')
def test_the_odds_a_model_compares_can_be_chosen(cli_runner, saved_dataloader):
    """Test the odds an odds-comparison model compares are chosen in the expression, as they are from Python."""
    result = cli_runner.invoke(
        main,
        [
            'evaluation',
            'backtest',
            '--dataloader',
            str(saved_dataloader),
            '--model',
            'OddsComparisonBettor(odds_types=["market_average"])',
            '--cv',
            '2',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Backtesting results' in result.output
