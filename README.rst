############
Data sources
############

`Football-Data.co.uk <http://www.football-data.co.uk/data.php>`_

`FiveThirtyEight <https://github.com/fivethirtyeight/data/tree/master/soccer-spi>`_

`BetBrain <https://www.betbrain.com/>`_

#####
Usage
#####

Create betting agent::

    betting_agent = BettingAgent()

Fetch training data::

    betting_agent.fetch_training(leagues)

Load modeling data::

    betting_agent._training(predicted_result)

Backtest on historical data::

    backtest(estimator, fit_params, predicted_result, test_year, max_day_range)

Calculate backtest_statistics::
    
    calculate_backtest_stats(factor=1.5, credit_limit=5.0)
