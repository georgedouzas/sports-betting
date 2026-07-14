"""Tests that a dataloader is its sources, and nothing else.

There is one dataloader, not one per sport. The sport is a property of the feed the data came from, and the odds are
optional, because features without a price are still worth having.
"""

import pytest

from sportsbet.dataloaders import DataLoader
from sportsbet.sources import EuroLeagueStats, FootballDataOdds, FootballDataStats, NBAStats, OddsApi


def test_a_dataloader_will_not_choose_where_the_data_comes_from():
    """Test a dataloader with no statistics says so, rather than reading a feed nobody asked for."""
    with pytest.raises(ValueError, match='does not choose where its data comes from'):
        DataLoader().extract_train_data(download=True)


def test_the_sport_belongs_to_the_source():
    """Test a feed of soccer is a feed of soccer, whatever it is handed to."""
    assert DataLoader(stats=FootballDataStats(), odds=FootballDataOdds()).sport == 'soccer'
    assert DataLoader(stats=EuroLeagueStats(), odds=OddsApi(key='k')).sport == 'basketball'
    assert DataLoader(stats=NBAStats(), odds=OddsApi(key='k')).sport == 'basketball'


def test_a_vendor_of_many_sports_takes_the_sport_it_is_paired_with():
    """Test an odds vendor covering several sports is not a sport of its own."""
    assert OddsApi.sport is None
    assert DataLoader(stats=NBAStats(), odds=OddsApi(key='k')).sport == 'basketball'


def test_statistics_and_odds_of_different_sports_are_refused():
    """Test the statistics of one sport and the odds of another are not about the same matches.

    Nothing could pair them, and the failure would otherwise arrive much later, as a roster in which no team could be
    found in the other.
    """
    with pytest.raises(ValueError, match='not about the same matches'):
        _ = DataLoader(stats=NBAStats(), odds=FootballDataOdds()).sources
