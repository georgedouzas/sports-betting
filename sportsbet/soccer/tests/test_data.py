"""
Test the data module.
"""

import requests
import numpy as np

from sportsbet.soccer.data import (
    SPIDataSource, 
    FDDataSource, 
    FDTrainingDataSource, 
    FDFixturesDataSource,
    SoccerDataLoader
)

SPI_DATA_SOURCE = SPIDataSource(['E0', 'G1']).download()
FD_TRAINING_DATA_SOURCE = FDTrainingDataSource('SP1', '1819').download()
FD_FIXTURES_DATA_SOURCE = FDFixturesDataSource(['SP1', 'I1']).download()


def test_spi_connection():
    """Test SPI data source connection."""
    status_code = requests.head(SPIDataSource.url).status_code
    assert status_code == 200


def test_spi_initialization():
    """Test SPI initialization."""
    assert SPI_DATA_SOURCE.url == 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
    assert SPI_DATA_SOURCE.match_cols == ['date', 'league', 'team1', 'team2']
    assert SPI_DATA_SOURCE.full_goals_cols == ['score1', 'score2']
    assert SPI_DATA_SOURCE.spi_cols == ['spi1', 'spi2', 'prob1', 'probtie', 'prob2', 'proj_score1', 'proj_score2']
    assert SPI_DATA_SOURCE.leagues_ids == ['E0', 'G1']


def test_spi_download():
    """Test SPI download."""
    assert hasattr(SPI_DATA_SOURCE, 'content_')
    assert set(SPI_DATA_SOURCE.content_.columns) == set(SPI_DATA_SOURCE.match_cols + SPI_DATA_SOURCE.full_goals_cols + SPI_DATA_SOURCE.spi_cols)


def test_spi_transform():
    """Test SPI transform."""
    content = SPI_DATA_SOURCE.transform()
    assert np.issubclass_(content[0].date.dtype.type, np.datetime64)
    assert np.issubclass_(content[1].date.dtype.type, np.datetime64)
    assert set(content[0].league.unique()) == set(SPI_DATA_SOURCE.leagues_ids)
    assert set(content[1].league.unique()) == set(SPI_DATA_SOURCE.leagues_ids)
    assert content[0].score1.isna().sum() == 0
    assert content[0].score2.isna().sum() == 0
    assert content[1].score1.isna().sum() == content[1].score1.size
    assert content[1].score2.isna().sum() == content[1].score2.size


def test_fd_connection():
    """Test FD data sources connection."""
    status_code = requests.head(FDDataSource.base_url).status_code
    assert status_code == 200


def test_fd_initialization():
    """Test FD initialization."""
    fd_data_source = FDDataSource()
    assert fd_data_source.base_url == 'http://www.football-data.co.uk'
    assert fd_data_source.match_cols == ['Date', 'Div', 'HomeTeam', 'AwayTeam']
    assert fd_data_source.full_goals_cols == ['FTHG', 'FTAG']
    assert fd_data_source.half_goals_cols == ['HTHG', 'HTAG']
    assert fd_data_source.avg_max_odds_cols == ['BbAvH', 'BbAvD', 'BbAvA', 'BbMxH', 'BbMxD', 'BbMxA']
    assert fd_data_source.odds_cols == ['PSH', 'PSD', 'PSA', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA']


def test_fd_training_download():
    """Test FD training download."""
    assert hasattr(FD_TRAINING_DATA_SOURCE, 'content_')
    assert set(FD_TRAINING_DATA_SOURCE.content_.columns) == set(FD_TRAINING_DATA_SOURCE.match_cols + FD_TRAINING_DATA_SOURCE.full_goals_cols + FD_TRAINING_DATA_SOURCE.half_goals_cols + FD_TRAINING_DATA_SOURCE.avg_max_odds_cols + FD_TRAINING_DATA_SOURCE.odds_cols)
    assert set(FD_TRAINING_DATA_SOURCE.content_.Div.unique()) == set([FD_TRAINING_DATA_SOURCE.league_id])
    

def test_fd_training_transform():
    """Test Fd training transform."""
    content = FD_TRAINING_DATA_SOURCE.transform()
    assert np.issubclass_(content.Date.dtype.type, np.datetime64)
    assert set(content.Season.unique()) == set([FD_TRAINING_DATA_SOURCE.season])


def test_fd_fixtures_download():
    """Test FD fixtures download."""
    assert hasattr(FD_FIXTURES_DATA_SOURCE, 'content_')
    assert set(FD_FIXTURES_DATA_SOURCE.content_.columns) == set(FD_FIXTURES_DATA_SOURCE.match_cols + FD_FIXTURES_DATA_SOURCE.avg_max_odds_cols + FD_FIXTURES_DATA_SOURCE.odds_cols)


def test_fd_fixtures_transform():
    """Test Fd training transform."""
    content = FD_FIXTURES_DATA_SOURCE.transform()
    assert np.issubclass_(content.Date.dtype.type, np.datetime64)
    assert set(content.Div.unique()).issubset(FD_FIXTURES_DATA_SOURCE.leagues_ids)